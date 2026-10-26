"""Configurable TTS transports with session-owned, aligned speech history."""
import asyncio
import base64
import contextlib
import logging
from pathlib import Path
import re

import httpx
import numpy as np

from .translation import auth
from .tts_history import AudioHistory, wav_data_url
from .tts_source import SourceVoice
from .tts_audio import TempoStream, TTSStreamError

logger = logging.getLogger(__name__)


def reference_audio(path, text):
    """Read only an operator-configured local reference; never fetch arbitrary URLs."""
    import soundfile as sf
    source = Path(path)
    if not source.is_file() or source.stat().st_size > 10 * 1024 * 1024:
        raise ValueError('TTS参考音频不存在或超过10MB')
    with sf.SoundFile(source) as f:
        if not 0 < len(f) / f.samplerate <= 30:
            raise ValueError('TTS参考音频须为0～30秒')
        if not 8000 <= f.samplerate <= 96000:
            raise ValueError('TTS参考音频采样率无效')
        samples = f.read(dtype='float32', always_2d=True).mean(axis=1)
        rate = f.samplerate
    if not np.isfinite(samples).all() or not np.any(samples):
        raise ValueError('TTS参考音频为空、全零或含NaN/Inf')
    pcm = np.clip(np.rint(np.clip(samples, -1, 1) * 32768), -32768, 32767).astype('<i2').tobytes()
    return {'ref_audio': wav_data_url(pcm, rate), 'ref_text': text.strip()}


class Speech:
    def __init__(self, settings, emit, direction, voice=''):
        self.s, self.emit, self.direction = settings, emit, direction
        self.voice = voice or settings.tts_voice
        self.language = direction.split('-')[1]
        self.speed = settings.tts_speed_zh_en if direction == 'zh-en' else settings.tts_speed
        self.queue = asyncio.Queue(maxsize=16)
        self.pending = ''
        self.seq = 0
        self.sid = None
        self.bytes = 0
        self.rate = settings.tts_sample_rate
        self.client = None
        self.task = None
        self.history = AudioHistory(settings.tts_history_window_s, settings.tts_history_max_chars)
        self.anchor = None
        self.paired = (settings.tts_backend == 'vllm_omni' and settings.tts_task_type == 'Base'
                       and settings.tts_history_mode != 'none')
        self.continuity = 'paired_reference_audio' if self.paired else 'none'
        self.closed = False
        self.source_voice = SourceVoice(settings.tts_clone_first_seconds,settings.tts_clone_interval_seconds) if settings.clone_source_enabled else None
        self.source_snapshot = None
        self.source_revision = 0
        self.failed_segments = 0
        self._carry = None
        self._input_done = False

    def info(self):
        return {'backend': self.s.tts_backend, 'sample_rate': self.rate,
                'format': 's16le', 'channels': 1, 'continuity': self.continuity,
                'history_window_s': self.s.tts_history_window_s if self.continuity != 'none' else 0,
                'history_max_chars': self.s.tts_history_max_chars,
                'history_policy': 'complete_text_audio_pairs',
                'speed': self.speed,
                'clone_source': self.source_voice is not None,
                'clone_first_seconds': self.s.tts_clone_first_seconds,
                'clone_interval_seconds': self.s.tts_clone_interval_seconds}

    async def _open_session(self):
        response = await self.client.post(self.s.tts_url.rstrip('/') + '/session_open', json={
            'voice': self.voice, 'window_s': self.s.tts_history_window_s, 'speed': self.speed,
            'history_max_chars': self.s.tts_history_max_chars})
        response.raise_for_status()
        data = response.json()
        if not data.get('sid') or data.get('error'):
            raise RuntimeError('TTS session_open失败')
        self.sid = data['sid']
        rate = int(data.get('sample_rate', self.rate))
        if not 8000 <= rate <= 96000:
            raise RuntimeError('TTS会话采样率无效')
        if self.seq and rate != self.rate:
            raise RuntimeError('TTS会话采样率发生变化')
        self.rate = rate
        supports_history = bool(data.get('continuity_version'))
        if self.s.tts_history_mode == 'paired' and not supports_history:
            raise RuntimeError('TTS会话服务没有声明历史语音支持')
        self.continuity = ('backend_paired_history' if supports_history else 'backend_session_unverified')
        if self.s.tts_history_mode == 'none':
            self.continuity = 'none'

    async def start(self):
        if self.s.tts_backend == 'none':
            raise ValueError('TTS未配置；关闭tts或配置tts_backend/tts_url')
        self.client = httpx.AsyncClient(timeout=self.s.tts_timeout_seconds, trust_env=False,
                                       headers=auth(self.s.tts_api_key_env))
        try:
            if self.s.tts_backend == 'session':
                await self._open_session()
            elif self.s.tts_backend == 'vllm_omni' and self.s.tts_task_type == 'Base':
                if self.source_voice is not None:
                    await check_source_voice_service(self.s,self.client)
                if self.s.tts_reference_audio:
                    self.anchor = await asyncio.to_thread(reference_audio,self.s.tts_reference_audio,self.s.tts_reference_text)
            self.task = asyncio.create_task(self.run())
        except BaseException:
            await self.close()
            raise

    async def put(self, item):
        if self.closed or self.task is None:
            raise RuntimeError('TTS会话未启动或已关闭')
        put = asyncio.create_task(self.queue.put(item))
        try:
            done, _ = await asyncio.wait([put, self.task], return_when=asyncio.FIRST_COMPLETED)
            if self.task in done:
                await self.task
                if item is None and put.done():
                    return
                raise RuntimeError('TTS任务提前结束')
            await put
        finally:
            if not put.done():
                put.cancel()
            await asyncio.gather(put, return_exceptions=True)

    def feed_source(self, samples):
        if self.source_voice is not None:
            self.source_voice.feed(samples)

    def finish_source(self):
        if self.source_voice is not None:self.source_voice.finish()

    async def flush_pending(self):
        if not self.pending:
            return
        if self.source_voice is not None and not self.source_voice.snapshots:
            return
        # Take ownership before yielding: the service backend may feed source
        # audio and translated text concurrently.
        text, self.pending = self.pending, ''
        await self.put((text,self.source_voice.latest()) if self.source_voice is not None else text)

    async def text(self, text):
        # Preserve model text exactly, including word fragments and punctuation.
        self.pending += text
        await self.flush_pending()

    async def finish(self):
        self.finish_source()
        if self.pending and self.source_voice is not None and not self.source_voice.snapshots:
            await self.emit({'event':'tts_unavailable','detail':'音频中没有足够的有效声音用于克隆；译文已保留。'})
            self.pending=''
        if self.pending:
            await self.put((self.pending,self.source_voice.latest()) if self.source_voice is not None else self.pending)
            self.pending = ''
        await self.put(None)
        await self.task

    def _speech_body(self, text):
        body = {'model': self.s.tts_model, 'input': text, 'voice': self.voice, 'response_format': 'pcm'}
        if self.speed != 1 and self.s.tts_backend != 'vllm_omni':
            body['speed'] = self.speed
        if self.s.tts_backend == 'vllm_omni':
            if self.s.tts_max_new_tokens:
                body['max_new_tokens'] = self.s.tts_max_new_tokens
            body.update(stream=True, stream_format='audio', task_type=self.s.tts_task_type,
                        language='English' if self.language == 'en' else 'Chinese', sample_rate=self.rate)
            if self.s.tts_task_type == 'Base':
                # The model's Base adapter uses the inline reference, not a
                # CustomVoice preset. Do not imply that voice overrides ref_audio.
                body.pop('voice', None)
                reference = self.history.reference(self.language, self.voice) if self.paired else None
                # Qwen rejects references shorter than one second. Keep short
                # complete pairs to accumulate, but use the source/anchor until
                # enough history exists; do not pad or invent a transcript.
                if reference is not None and self.history.seconds < 1:
                    reference = None
                if self.source_snapshot is not None:
                    body['extra_params']={'simulinfer_source_audio':self.source_snapshot.audio}
                    body.update(reference or {'ref_audio':self.source_snapshot.audio})
                    body['x_vector_only_mode']=reference is None
                    # With speaker-only conditioning, prefill all text rather
                    # than padding a short text across codec decoding steps.
                    # Keep aligned streaming-text conditioning once paired
                    # acoustic history exists. Both stream audio over HTTP.
                    body['non_streaming_mode']=reference is None
                else:
                    body.update(reference or self.anchor)
                    body['x_vector_only_mode'] = False
        return body

    async def _next_text(self):
        if self._input_done:
            return None
        if self._carry is not None:
            item, self._carry = self._carry, None
        else:
            item = await self.queue.get()
            self.queue.task_done()
        if item is None:
            self._input_done = True
            return None
        text, snapshot = item if self.source_voice is not None else (item, None)
        # Batch only text that arrived close together. Never wait indefinitely
        # for a length threshold or split a model-provided word in the middle.
        delay = 0 if self.s.tts_min_chars == 1 else self.s.tts_batch_ms / 1000
        deadline = asyncio.get_running_loop().time() + delay
        limit = 240 if self.language == 'en' else 100
        while delay and len(text) < limit:
            remaining = deadline - asyncio.get_running_loop().time()
            if self.queue.empty() and (remaining <= 0 or re.search(r'[。！？!?\n.]\s*$', text)):
                break
            try:
                extra = self.queue.get_nowait() if not self.queue.empty() else await asyncio.wait_for(self.queue.get(), remaining)
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                break
            self.queue.task_done()
            if extra is None:
                self._input_done = True
                break
            next_text, next_snapshot = extra if self.source_voice is not None else (extra, None)
            if len(text) + len(next_text) > limit:
                self._carry = extra
                break
            text += next_text
            snapshot = next_snapshot
        return text, snapshot

    async def run(self):
        while (item := await self._next_text()) is not None:
            text, snapshot = item
            if not any(c.isalnum() for c in text):
                continue
            if self.source_voice is not None:
                if snapshot is None:
                    await self.emit({'event':'tts_unavailable','detail':'音频中没有足够的有效声音用于克隆；译文已保留。'})
                    continue
                self.source_snapshot = snapshot
                if snapshot.revision != self.source_revision:
                    self.source_revision = snapshot.revision
                    await self.emit({'event':'voice_clone','revision':snapshot.revision,
                                     'reference_seconds':snapshot.seconds,'source_end_ms':snapshot.end_sample/16})
            self.seq += 1
            before = self.bytes
            try:
                if self.s.tts_backend == 'session' and self.s.tts_history_mode == 'none' and self.seq > 1:
                    await self._close_session()
                    await self._open_session()
                await asyncio.wait_for(self._synthesize(text),self.s.tts_timeout_seconds)
            except (httpx.HTTPError, TTSStreamError, asyncio.TimeoutError) as exc:
                self.failed_segments += 1
                # Partial audio was already delivered: replaying the same text
                # would repeat words. Discard failed history and resume with
                # the input speaker/anchor on the next segment.
                self.history.clear()
                logger.warning('TTS segment %s failed (%s); translation continues: %s',
                               self.seq, type(exc).__name__, str(exc)[:500])
                await self.emit({'event':'tts_error','seq':self.seq,
                                 'partial':self.bytes > before,'audio_bytes':self.bytes-before,
                                 'detail':'This speech segment could not finish. Translation continues; the segment is not replayed.'})
        await self.emit({'event':'tts_done','audio_bytes':self.bytes,
                         'complete':self.failed_segments == 0,'failed_segments':self.failed_segments,**self.info()})

    async def _synthesize(self, text):
        await self.emit({'event':'tts_start','seq':self.seq,'text':text,**self.info(),
                         'reference_history_seconds':self.history.seconds if self.paired else None})
        total = 0
        before = self.bytes
        generated = bytearray()
        tempo = None
        tail_filter = None

        async def output_pcm(raw):
            if not raw:
                return
            self.bytes += len(raw)
            await self.emit({'event':'audio','seq':self.seq,'sample_rate':self.rate,
                             'format':'s16le','channels':1,'data':base64.b64encode(raw).decode('ascii')})

        async def input_pcm(raw):
            nonlocal total
            total += len(raw)
            if total > self.s.tts_max_audio_seconds * self.rate * 2:
                raise TTSStreamError('TTS单次音频超过配置上限')
            if tail_filter is not None:
                await tail_filter.feed(raw)
            else:
                await filtered_pcm(raw)

        async def filtered_pcm(raw):
            if self.paired:
                generated.extend(raw)
            if tempo is not None:
                await tempo.write(raw)
            else:
                await output_pcm(raw)

        try:
            if self.s.tts_tail_filter:
                from .tts_tail import SpeechTail
                tail_filter = SpeechTail(self.rate,filtered_pcm)
                await tail_filter.start()
            # Speed up the delivered PCM, preserving pitch and original-speed
            # reference history. Keep the backend's raw streaming mode.
            if self.s.tts_backend == 'vllm_omni' and self.speed != 1:
                tempo = TempoStream(self.rate,self.speed,output_pcm)
                await tempo.start()
            if self.s.tts_backend == 'mock':
                samples = (np.sin(np.arange(self.rate//8)*2*np.pi*440/self.rate)*.05*32767).astype('<i2')
                await input_pcm(samples.tobytes())
            else:
                url = self.s.tts_url.rstrip('/')
                if self.s.tts_backend == 'session':
                    request = self.client.stream('GET',url+'/session_say',params={
                        'sid':self.sid,'text':text,'lang':self.language,'voice':self.voice})
                else:
                    request = self.client.stream('POST',url+'/audio/speech',json=self._speech_body(text))
                tail = b''
                async with request as response:
                    response.raise_for_status()
                    if response.headers.get('x-error'):
                        raise TTSStreamError('TTS: '+response.headers['x-error'])
                    mime = response.headers.get('content-type','').lower()
                    if any(value in mime for value in ('json','text/','wav','mpeg','mp3','flac','ogg')):
                        raise TTSStreamError('TTS必须返回原始PCM，不能将JSON/SSE或音频容器作为PCM播放')
                    try:
                        actual_rate = int(response.headers.get('x-audio-sample-rate',response.headers.get('x-sample-rate',self.rate)))
                        channels = int(response.headers.get('x-audio-channels','1'))
                    except (TypeError, ValueError) as exc:
                        raise TTSStreamError('TTS returned invalid audio metadata') from exc
                    if actual_rate != self.rate or channels != 1:
                        raise TTSStreamError('TTS返回采样率/声道与配置不一致（要求单声道）')
                    async for raw in response.aiter_bytes():
                        tail += raw
                        n = len(tail)//2*2
                        await input_pcm(tail[:n])
                        tail = tail[n:]
                if tail or not total:
                    raise TTSStreamError('TTS返回空音频或不完整PCM')
            if tail_filter is not None:
                await tail_filter.finish()
            if tempo is not None:
                await tempo.finish()
            if self.paired:
                self.history.commit(text,generated,self.rate,self.language,self.voice)
            await self.emit({'event':'tts_end','seq':self.seq,'audio_bytes':self.bytes-before,
                             'history_seconds':self.history.seconds if self.paired else None})
        finally:
            if tempo is not None:
                await tempo.close()

    async def _close_session(self):
        if self.client and self.sid:
            sid, self.sid = self.sid, None
            with contextlib.suppress(Exception):
                await self.client.post(self.s.tts_url.rstrip('/') + '/session_close', json={'sid': sid}, timeout=5)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.task and not self.task.done():
            self.task.cancel()
        if self.task:
            await asyncio.gather(self.task, return_exceptions=True)
        await self._close_session()
        if self.client:
            await self.client.aclose()
        self.history.clear()
        self.anchor = None
        if self.source_voice is not None:self.source_voice.clear()
        self.source_snapshot=None


async def check_source_voice_service(settings,client=None):
    if not settings.clone_source_enabled:return
    own=client is None
    client=client or httpx.AsyncClient(timeout=10,trust_env=False,headers=auth(settings.tts_api_key_env))
    try:
        response=await client.get(settings.tts_url.rstrip('/')+'/simulinfer/tts_capabilities',timeout=10)
        if response.is_error or response.json().get('source_voice_history')!='source-voice-history-v1':
            raise RuntimeError('TTS服务未启用输入音色克隆；请用 simulinfer-tts-omni 启动 Qwen Base 服务')
    finally:
        if own:await client.aclose()
