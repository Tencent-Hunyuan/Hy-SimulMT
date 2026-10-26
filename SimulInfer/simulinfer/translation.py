"""Audio and conversation state live here; model calls are always HTTP/WS."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
import uuid

import httpx
import numpy as np
import websockets

from .audio import part
from .config import Model, Settings
from .vad import create_activity, FireRedNonstreamActivity
from .budget import ContextBudgetError, PromptBudget, render, speech_tokens
from .noise import speech_audio, mask_spans, padded_st_audio, speech_stats, detect_st_blocks, short_speech_policy

SRC = "<｜hy_place▁holder▁no▁790｜>"
TGT = "<｜hy_place▁holder▁no▁791｜>"
EOT = "<｜hy_EOT｜>"
EOS = "<｜hy_end▁of▁sentence｜>"
INSTRUCTIONS = {
    "zh-en": "请完成下面的同声传译任务，将陆续收到的语音翻译成英文。请结合上下文边听边译，输出能够确定的译文，不必等待整句结束。",
    "en-zh": "请完成下面的同声传译任务，将陆续收到的语音翻译成中文。请结合上下文边听边译，输出能够确定的译文，不必等待整句结束。",
}
ST_INSTRUCTIONS = {"zh-en": "把这段语音翻译成英文，要全部输出，不要分块。",
                   "en-zh": "把这段语音翻译成中文，要全部输出，不要分块。"}


def validate_mode(mode, direction, settings):
    if mode not in settings.models:
        raise ValueError(f"模型模式未配置: {mode}")
    if direction not in INSTRUCTIONS:
        raise ValueError("direction 必须为 zh-en 或 en-zh")


def split_reply(text, mode):
    text = text.removesuffix(EOT).removesuffix(EOS)
    if mode == "target_only":
        if SRC in text or TGT in text:
            raise ValueError("模型返回了源文分隔符，请将 mode 改为 source_target")
        return "", text
    text = text.removeprefix(SRC)
    if not text:
        return "", ""
    if TGT not in text:
        raise ValueError("source_target 回答缺少TGT标记，检查模型模式、chat template及输出token上限")
    source, target = text.split(TGT, 1)
    if SRC in target or TGT in target:
        raise ValueError("模型重复输出字段分隔符")
    return source, target


def auth(env):
    key = os.getenv(env, "")
    return {"Authorization": "Bearer " + key} if key else {}


def chat_request(messages, mode, prefix=None, protocol='standard'):
    """Continue an assistant prefix through vLLM's public chat API.

    Text content parts work with the released simulst template and the older
    export template, including exports whose assistant branch expects a list.
    Neither model-side SimulInfer imports nor request-supplied Jinja are needed.
    """
    if protocol == 'harness':
        kwargs={'output_mode':mode}
        if prefix is not None:kwargs['harness_prefix']=prefix
        return {'messages':messages,'chat_template_kwargs':kwargs}
    wire=[]
    for message in messages:
        copy=dict(message)
        if isinstance(copy['content'],str):
            copy['content']=[{'type':'text','text':copy['content']}]
        wire.append(copy)
    continuation=prefix if prefix is not None else (SRC if mode=='source_target' else None)
    fields={'messages':wire,'add_generation_prompt':continuation is None,
            'continue_final_message':continuation is not None,'add_special_tokens':False}
    if continuation is not None:
        wire.append({'role':'assistant','content':[{'type':'text','text':continuation}]})
    return fields


@dataclass
class Reply:
    source: str
    text: str
    raw: str
    ids: list[int]
    stop: str | int | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"


class ChatAPI:
    def __init__(self, model: Model, settings: Settings, mode, direction, http=None):
        self.model, self.settings, self.mode, self.direction = model, settings, mode, direction
        self.http = http or httpx.AsyncClient(timeout=settings.timeout_seconds, trust_env=False,
                                             headers=auth(model.api_key_env))
        self.owned = http is None
        self.max_model_len = settings.max_model_len
        self.last_budget = None

    async def check(self):
        """Check text and audio preprocessing through /tokenize, without decoding."""
        specials={}
        for text in (SRC,TGT,EOT,EOS):
            ids=(await self.tokenize(text))['tokens']
            if len(ids)!=1:raise RuntimeError(f'模型tokenizer没有单token标记: {text}')
            specials[text]=ids[0]
        messages=[{'role':'user','content':[{'type':'text','text':'probe'}]},
                  {'role':'assistant','content':'previous '},
                  {'role':'user','content':[{'type':'text','text':'next'}]}]
        base=self.model.base_url.rstrip('/')
        root=base[:-3] if base.endswith('/v1') else base
        for prefix in (None,'prefix '):
            rendered,_,_=render(messages,self.mode,prefix)
            expected=(await self.tokenize(rendered))['tokens']
            fields=chat_request(messages,self.mode,prefix,self.model.chat_protocol)
            response=await self.http.post(root+'/tokenize',json={
                **fields,'model':self.model.model,'add_special_tokens':False})
            if response.is_error:
                raise RuntimeError(f'模型/tokenize不支持chat/前缀协议: HTTP {response.status_code}: '
                                   +response.text[:400])
            if response.json().get('tokens')!=expected:
                raise RuntimeError('模型chat模板与simulst协议不一致；确认统一模型的音频模板已准备好，'
                                   '并用--chat-template-content-format openai启动。'
                                   '已有专用模板可配置chat_protocol="harness"。')
        # The real audio preprocessing path checks the checkpoint assets;
        # text-only tokenization can succeed even with an unsupported template.
        opening=part(np.zeros(25600,dtype=np.float32),self.settings.model_audio_format)
        following=part(np.full(12800,.01,dtype=np.float32),self.settings.model_audio_format)
        audio_id=(await self.tokenize('<｜hy_place▁holder▁no▁5｜>'))['tokens']
        if len(audio_id)!=1:raise RuntimeError('模型缺少音频标记')
        probes=[([{'role':'user','content':[{'type':'text','text':INSTRUCTIONS[self.direction]},opening]}],None),
                ([{'role':'user','content':[{'type':'text','text':INSTRUCTIONS[self.direction]},opening]},
                  {'role':'assistant','content':SRC+'source'+TGT+'translation' if self.mode=='source_target' else 'translation'},
                  {'role':'user','content':[following]}],'prefix ')]
        for messages,prefix in probes:
            text,_,_=render(messages,self.mode,prefix)
            compact=(await self.tokenize(text))['tokens']
            lengths=iter(speech_tokens(p.num_samples) for m in messages if m['role']=='user'
                         for p in m['content'] if p['type']=='input_audio')
            expected=[]
            for token in compact:
                if token==audio_id[0]:
                    count=next(lengths,None)
                    if count is None:raise RuntimeError('音频模板包含多余的输入片段')
                    expected.extend([token]*count)
                else:expected.append(token)
            if next(lengths,None) is not None:raise RuntimeError('音频模板未包含全部输入片段')
            fields=chat_request(messages,self.mode,prefix,self.model.chat_protocol)
            response=await self.http.post(root+'/tokenize',json={
                **fields,'model':self.model.model,'add_special_tokens':False})
            if response.is_error:
                raise RuntimeError('模型音频接入检查失败：请用 simulinfer-prepare-model 准备当前推理模板，'
                                   '让 vLLM 加载该目录后重启。模型服务返回：'+response.text[:600])
            if response.json().get('tokens')!=expected:
                raise RuntimeError('模型音频模板与当前协议不一致；请使用 simulinfer-prepare-model '
                                   '准备的模型目录，并以 --chat-template-content-format openai 启动 vLLM')
        return {'mode':self.mode,'protocol':self.model.chat_protocol,
                'max_model_len':self.max_model_len,'special_tokens':specials,'audio_checked':True}

    async def check_repetition_processors(self):
        """Verify enabled no-repeat filters with a small forced-token probe.

        Unknown vllm_xargs may otherwise be accepted but ignored by a backend
        missing the model plugin. Check behavior before advertising readiness.
        """
        sizes = {'hy_simulst_no_repeat_ngram_size': self.settings.no_repeat_ngram_size,
                 'hy_st_no_repeat_ngram_size': self.settings.st_no_repeat_ngram_size}
        if not self.settings.model_logits_processors or not any(sizes.values()):
            return {}
        first = (await self.tokenize('a'))['tokens']
        second = (await self.tokenize('b'))['tokens']
        if len(first)!=1 or len(second)!=1 or first==second:
            raise RuntimeError('Cannot validate no-repeat processors with this tokenizer')
        prompt,_,_ = render([{'role':'user','content':[{'type':'text','text':'test'}]}],self.mode)
        checked = {}
        for key,size in sizes.items():
            if not size:
                continue
            response = await self.http.post(self.model.base_url.rstrip('/')+'/completions',json={
                'model':self.model.model,'prompt':prompt,'max_tokens':size+1,
                'temperature':0,'ignore_eos':True,'return_token_ids':True,
                'logit_bias':{str(first[0]):100,str(second[0]):90},
                'vllm_xargs':{key:size}})
            if response.is_error:
                raise RuntimeError('No-repeat processor startup check failed: '+response.text[:400])
            choices=response.json().get('choices',[])
            ids=choices[0].get('token_ids',[]) if choices else []
            if len(ids)!=size+1 or ids[:size]!=first*size or ids[-1]!=second[0]:
                raise RuntimeError('No-repeat processor is missing or inactive. Install the matching '
                                   'SimulInfer package in the vLLM environment, then restart vLLM: '+key)
            checked[key]=size
        return checked

    async def tokenize(self, text):
        # /tokenize is a sibling of /v1, including when behind a URL prefix.
        base = self.model.base_url.rstrip('/')
        root = base[:-3] if base.endswith('/v1') else base
        response = await self.http.post(root + '/tokenize', json={
            'model': self.model.model, 'prompt': text, 'add_special_tokens': False})
        if response.is_error:
            raise RuntimeError(f'vLLM /tokenize HTTP {response.status_code}: {response.text[:400]}')
        data = response.json()
        count, limit, ids = data.get('count'), data.get('max_model_len'), data.get('tokens')
        if (type(count) is not int or count < 0 or type(limit) is not int or limit < 512
                or not isinstance(ids, list) or len(ids) != count):
            raise RuntimeError('vLLM /tokenize响应缺少有效count/max_model_len/tokens')
        self.max_model_len = min(self.settings.max_model_len, limit)
        return data

    async def budget(self, messages, *, prefix=None):
        text, expansion, items = render(messages, self.mode, prefix)
        data = await self.tokenize(text)
        result = PromptBudget(data['count'] + expansion, self.max_model_len, items)
        self.last_budget = result
        return result

    async def generate(self, messages, *, prefix=None, max_tokens=None, target_only=False,
                       repetition_penalty=None, source_phase=False, st=False, budget=None):
        max_tokens = max_tokens or self.settings.max_tokens
        if self.settings.token_budget_enabled:
            budget = budget or await self.budget(messages, prefix=prefix)
        if budget is not None and not budget.fits(max_tokens):
            raise ContextBudgetError(f'模型上下文预算不足：prompt={budget.prompt_tokens}, '
                f'output={max_tokens}, limit={budget.max_model_len}')
        body = {"model": self.model.model, "stream": False,
                "temperature": 0.0, "max_tokens": max_tokens,
                "repetition_penalty": (self.settings.repetition_penalty if repetition_penalty is None else repetition_penalty), "skip_special_tokens": False,
                "return_token_ids": True,
                "stop": [EOT, EOS], "request_id": "SimulInfer-" + uuid.uuid4().hex}
        body.update(chat_request(messages,self.mode,prefix,self.model.chat_protocol))
        if source_phase:
            body['stop'] = [TGT, EOT, EOS]
        xargs = {}
        ngram = self.settings.st_no_repeat_ngram_size if st else self.settings.no_repeat_ngram_size
        if ngram:
            xargs['hy_st_no_repeat_ngram_size' if st else 'hy_simulst_no_repeat_ngram_size'] = ngram
            if st and prefix:
                xargs['hy_st_prefix_token_ids'] = (await self.tokenize(prefix))['tokens']
        if self.settings.no_chinese_target and self.direction == "zh-en" and not source_phase:
            in_target = target_only or self.mode == 'target_only' or (prefix is not None and TGT in prefix)
            xargs['hy_target_language'] = 'target' if in_target else 'after_target'
        if xargs and self.settings.model_logits_processors:
            body['vllm_xargs'] = xargs
        response = await self.http.post(self.model.base_url.rstrip("/") + "/chat/completions", json=body)
        if response.is_error:
            detail = response.text[:600]
            raise RuntimeError(f"vLLM HTTP {response.status_code}: {detail}")
        data = response.json()
        if not data.get("choices"):
            raise RuntimeError("模型接口没有返回choices")
        choice = data["choices"][0]
        usage = data.get('usage') or {}
        actual = usage.get('prompt_tokens')
        if type(actual) is not int or actual < 0:
            raise RuntimeError('vLLM响应缺少prompt_tokens')
        if budget is not None and actual != budget.prompt_tokens:
            raise RuntimeError(f'模型音频token计数与SimulInfer模板不一致：'
                f'counted={budget.prompt_tokens}, actual={actual}；请核对simulst_mm和启动模板')
        completion = usage.get('completion_tokens')
        if type(completion) is not int or completion < 0:
            raise RuntimeError('vLLM响应缺少completion_tokens')
        finish = choice.get('finish_reason')
        if finish not in ('stop', 'length'):
            raise RuntimeError(f'vLLM未正常结束生成：{finish}')
        if finish=='length' and not self.settings.reset_on_max_tokens:
            raise RuntimeError('模型输出达到max_tokens上限，当前结果不完整；请调整输出预算，'
                               '或显式开启reset_on_max_tokens以保留部分输出并重置')
        text = choice.get("message", {}).get("content") or ""
        if source_phase:
            if choice.get('stop_reason') != TGT:
                raise RuntimeError("ST源文续写未到达TGT边界，不继续拼接不完整源文")
            return Reply(text.removeprefix(SRC), '', text, choice.get('token_ids') or [], TGT,
                         actual, completion, finish)
        # vLLM can return the stopping token in token_ids, and some builds also
        # retain its literal text with skip_special_tokens=false. Store only
        # the body: the chat template supplies exactly one EOT per old answer.
        text = text.removesuffix(EOT).removesuffix(EOS)
        source, target = ("", text) if target_only else split_reply(text, self.mode)
        return Reply(source, target, text, choice.get("token_ids") or [], choice.get("stop_reason"),
                     actual, completion, finish)

    async def close(self):
        if self.owned:
            await self.http.aclose()


class VLLMSession:
    """One source timeline and one outstanding model call per session.

    The model remains behind HTTP. Silero state is independent of model-run
    resets; only real source samples advance VAD and the progress cursor.
    """
    def __init__(self, model, settings, mode, direction, http=None):
        self.settings, self.mode, self.direction = settings, mode, direction
        self.api = ChatAPI(model, settings, mode, direction, http)
        self.messages = []
        self.buffer = np.empty(0, dtype=np.float32)
        self.cursor = self.index = self.run = self.run_samples = self.voiced_waits = 0
        self.voiced_wait_samples=0
        self.pending = []
        self.last_emit = None
        self.continuous_silence = 0
        self.full_text = self.full_source = ""
        self.source_segments = []
        self.closed = self.eof = False
        self.pending_attempt_key = None
        self.skipped_ms = self.empty_st_resets = self.st_count = 0
        self.activity = None
        self.preroll = np.empty(0, dtype=np.float32)
        self.session_has_speech = False
        self.last_completion_tokens = 0
        self.st_activity = None
        self.voice_revision=0
        self.rejected_voice_revision=None

    async def prepare(self):
        if self.activity is None:
            self.activity = await asyncio.to_thread(create_activity, self.settings)

    def ready(self):
        fields = ('chunk_ms', 'first_chunk_ms', 'model_audio_format', 'audio_mode', 'st_context', 'st_wait_chunks', 'window_seconds',
                  'vad_preroll_enabled', 'simulst_crop_speech',
                  'repetition_penalty', 'st_repetition_penalty',
                  'reset_after_audible_st',
                  'reset_after_silence_st', 'reset_after_empty_st', 'st_fallback_max_seconds',
                  'st_trim_trailing_silence', 'vad_mode', 'vad_threshold', 'vad_release_threshold',
                  'vad_pad_ms', 'vad_leading_pad_ms', 'vad_full_chunk_speech_ms',
                  'vad_cpu_threads', 'silence_rms', 'trailing_silence_seconds',
                  'skip_initial_silence', 'post_speech_silence_mode', 'st_vad_mode',
                  'st_vad_threshold','wait_fallback_seconds')
        return {'event': 'ready', 'backend': 'vllm', 'mode': self.mode, 'direction': self.direction,
                'sample_rate': 16000, 'input_format': 'f32le', 'greedy': True,
                'time_rollover_enabled': self.settings.window_seconds > 0,
                'vad_device': 'cpu' if self.settings.vad_mode in {'silero','firered'} else None,
                **{key: getattr(self.settings, key) for key in fields}}

    def reset(self, event=None, reason='manual'):
        self.messages = []
        self.run_samples = 0
        self.last_emit = None
        self.preroll = self.preroll[:0]
        self.last_completion_tokens = 0
        self.continuous_silence = 0
        self.run += 1
        if not self.pending:
            self.voiced_waits = 0
            self.voiced_wait_samples=0
            self.pending_attempt_key = None
            self.rejected_voice_revision=None
        if event is not None:
            event.update(window_reset=True, restart_reason=reason,
                         recovery_context_retained=False, next_run=self.run)

    def discard(self, blocks, event, reason):
        if not blocks:
            return
        count = sum(b[2] for b in blocks) // 16
        self.skipped_ms += count
        event.update(skipped_untranslated_audio_ms=event.get('skipped_untranslated_audio_ms', 0)+count,
                     skipped_untranslated_start_ms=blocks[0][0]//16,
                     skipped_untranslated_end_ms=(blocks[-1][0]+blocks[-1][2])//16,
                     recovery_warning=reason)

    async def push(self, pcm):
        if self.closed or self.eof:
            raise ValueError('会话已结束')
        self.buffer = np.concatenate((self.buffer, pcm))
        while len(self.buffer):
            size = (self.settings.first_chunk_ms if self.index == 0 else self.settings.chunk_ms) * 16
            if len(self.buffer) < size:
                break
            chunk, self.buffer = self.buffer[:size], self.buffer[size:]
            yield await self.step(chunk, size)

    def user_turn(self, samples):
        parts = ([] if self.messages else [{'type': 'text', 'text': INSTRUCTIONS[self.direction]}])
        return {'role': 'user', 'content': parts + [part(samples, self.settings.model_audio_format)]}

    async def step(self, samples, valid, *, final=False):
        begin = self.cursor
        self.cursor += valid
        await self.prepare()
        if self.settings.vad_mode in {'silero','firered'}:
            activity = await asyncio.to_thread(self.activity.analyze, samples[:valid], begin)
        else:
            activity = self.activity.analyze(samples[:valid], begin)
        speech_mask = activity.pop('_vad_speech_mask', None)
        raw_speech_mask = activity.pop('_vad_raw_speech_mask', None)
        st_metadata=None
        if self.settings.st_crop_enabled or self.settings.st_vad_mode=='firered_nonstream':
            raw_mask=raw_speech_mask if raw_speech_mask is not None else speech_mask
            if raw_mask is None:raise RuntimeError('ST人声裁剪需要VAD返回人声范围')
            if len(raw_mask)!=valid:raise RuntimeError('ST人声范围与真实音频长度不一致')
            st_metadata={'original':samples[:valid].copy(),'speech_spans':mask_spans(raw_mask,begin)}
        silent = activity['is_silence']
        mask_enabled = bool(
            self.settings.post_speech_silence_mode == 'zero_fill'
            and (self.session_has_speech or not silent)
        )
        if not silent:
            self.session_has_speech = True
            self.voice_revision+=1
        self.continuous_silence = (activity.get('vad_trailing_silence_samples',
            self.continuous_silence + valid) if silent else 0)
        event = {'event': 'chunk', 'index': self.index, 'start_ms': begin//16,
                 'end_ms': self.cursor//16, 'model_audio_ms': len(samples)//16,
                 'final_padding_ms': (len(samples)-valid)//16,
                 'action': 'WAIT', 'text': '', 'run': self.run, **activity}
        model_samples = samples
        zeroed_samples = 0
        if mask_enabled:
            if speech_mask is None:
                if silent:
                    model_samples = np.zeros_like(samples)
                    zeroed_samples = len(samples)
            else:
                mask = np.zeros(len(samples), dtype=np.bool_)
                count = min(len(mask), len(speech_mask))
                mask[:count] = speech_mask[:count]
                if self.settings.vad_leading_pad_ms and mask.any():
                    starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
                    for start in starts:
                        mask[max(0,int(start)-round(self.settings.vad_leading_pad_ms*16)):int(start)]=True
                raw_longest=0
                if raw_speech_mask is not None and len(raw_speech_mask):
                    raw=np.asarray(raw_speech_mask[:len(samples)],dtype=np.bool_)
                    edges=np.flatnonzero(np.diff(np.r_[False,raw,False].astype(np.int8)))
                    if len(edges)>=2: raw_longest=int((edges[1::2]-edges[::2]).max())
                full_chunk_passthrough=(len(samples)==self.settings.chunk_ms*16
                    and self.settings.vad_full_chunk_speech_ms>0
                    and raw_longest>=round(self.settings.vad_full_chunk_speech_ms*16))
                activity['vad_raw_longest_speech_ms']=raw_longest//16
                activity['vad_full_chunk_passthrough']=bool(full_chunk_passthrough)
                if not full_chunk_passthrough:
                    zeroed_samples = int((~mask).sum())
                    if zeroed_samples:
                        model_samples = samples.copy(); model_samples[~mask] = 0
                event.update(vad_raw_longest_speech_ms=raw_longest//16,
                             vad_full_chunk_passthrough=bool(full_chunk_passthrough))
        zero_filled = zeroed_samples == len(samples) and bool(len(samples))
        event.update(post_speech_silence_mode=self.settings.post_speech_silence_mode,
                     silence_zero_filled=zero_filled,
                     silence_partially_zero_filled=0 < zeroed_samples < len(samples),
                     silence_zero_filled_ms=zeroed_samples//16,
                     model_audio_rms=float(np.sqrt(np.mean(model_samples**2))) if len(model_samples) else 0.0)
        self.index += 1
        retained_spans=None
        if self.settings.audio_mode=='noise_resistant':
            if speech_mask is None:raise RuntimeError('抗噪模式要求VAD返回逐采样人声范围')
            if self.settings.simulst_crop_enabled:
                model_samples,retained_spans=speech_audio(samples[:valid],speech_mask,
                    start_sample=begin,leading_samples=round(self.settings.vad_leading_pad_ms*16))
                event.update(audio_mode='noise_resistant',model_audio_ms=len(model_samples)/16,
                             final_padding_ms=0,vad_removed_ms=(valid-len(model_samples))/16,
                             retained_source_spans_ms=[[a/16,b/16] for a,b in retained_spans],
                             model_audio_rms=float(np.sqrt(np.mean(model_samples**2))) if len(model_samples) else 0.0)
            else:
                event.update(audio_mode='noise_resistant',vad_removed_ms=0)
            # Silence detection is independent of cropping voiced chunks. Keep
            # original audio metadata for ST even when this chunk is skipped.
            skip_chunk = not len(model_samples) if self.settings.simulst_crop_enabled else silent
            if skip_chunk:
                event.update(silence_skipped=True,model_invoked=False,audible_wait_chunks=self.voiced_waits)
                if self.pending:
                    self.pending.append((begin,model_samples,valid,True,retained_spans,st_metadata))
                    self.run_samples += valid
                    if final or (self.settings.trailing_silence_seconds and
                            self.continuous_silence>=self.settings.trailing_silence_seconds*16000):
                        await self.flush(event,force=final,trigger='eof' if final else 'trailing_silence')
                    if not final and self.settings.window_seconds and self.run_samples>=self.settings.window_seconds*16000:
                        await self.cut(event,'time_window')
                return event
        if silent and not self.messages and self.settings.skip_initial_silence:
            if self.settings.vad_mode == 'silero' and self.settings.vad_preroll_enabled:
                self.preroll = samples[:valid][-self.settings.chunk_ms*16:].copy()
            event.update(silence_skipped=True, model_invoked=False, audible_wait_chunks=self.voiced_waits)
            return event

        # Match the service's single-block Silero onset protection. It is
        # context only: no second source event, no extra progress or WAIT count.
        if not self.messages and len(self.preroll):
            model_samples = np.concatenate((self.preroll, model_samples))
            event['vad_preroll_ms'] = len(self.preroll)//16
        self.preroll = self.preroll[:0]
        turn = self.user_turn(model_samples)
        budget = (await self.api.budget(self.messages + [turn]) if self.settings.token_budget_enabled else None)
        if budget is not None and not budget.fits(self.settings.max_tokens):
            # Defensive preflight for mismatched remote limits or unexpectedly
            # large text re-tokenization. The new source chunk is not consumed
            # by a failed generation or included in previous pending ST.
            await self.cut(event, 'token_budget_preflight')
            if retained_spans is None:model_samples = samples
            turn = self.user_turn(model_samples)
            budget = await self.api.budget([turn])
        event['run'] = self.run
        reply = await self.api.generate(self.messages + [turn], budget=budget)
        self.messages.append(turn)
        raw = SRC + reply.raw.removeprefix(SRC) if self.mode == 'source_target' else reply.raw
        self.messages.append({'role': 'assistant', 'content': raw})
        # Keep original waveform/VAD metadata separate from the model input so
        # ST can crop even when ordinary streaming chunks are unmodified.
        block=(begin,model_samples.copy(),valid,silent)
        self.pending.append(block if st_metadata is None else (*block,retained_spans,st_metadata))
        self.run_samples += valid
        self.last_completion_tokens = reply.completion_tokens
        event.update(model_invoked=True, model_audio_ms=len(model_samples)/16,
                     stop_reason=reply.stop, generated_token_ids=reply.ids,
                     prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens,
                     effective_max_model_len=budget.max_model_len if budget else None)
        self.publish_source(reply.source, event, [(begin, None, valid)])
        if reply.text.strip():
            self.commit(reply, event)
        elif not silent:
            self.voiced_waits += 1
            self.voiced_wait_samples += valid
        event['audible_wait_chunks'] = self.voiced_waits
        event['audible_wait_ms'] = self.voiced_wait_samples//16
        periodic = self.settings.st_wait_chunks and self.voiced_waits >= self.settings.st_wait_chunks
        if self.settings.wait_fallback_seconds is not None:
            periodic=(self.settings.wait_fallback_seconds>0 and
                      self.voiced_wait_samples>=self.settings.wait_fallback_seconds*16000)
        silence_end = (self.voiced_waits and silent and self.settings.trailing_silence_seconds > 0
                       and self.continuous_silence >= self.settings.trailing_silence_seconds*16000)
        prior_run = self.run
        if not reply.text.strip() and (periodic or silence_end or final):
            trigger = 'audible_wait' if periodic else 'trailing_silence' if silence_end else 'eof'
            await self.flush(event, force=final, trigger=trigger)
        if self.run == prior_run and not final:
            if reply.finish_reason == 'length':
                event['output_truncated'] = True
                await self.cut(event, 'max_tokens')
            elif self.settings.window_seconds and self.run_samples >= self.settings.window_seconds*16000:
                await self.cut(event, 'time_window')
            elif budget is not None and reply.prompt_tokens + reply.completion_tokens + self.settings.context_reserve_tokens >= budget.max_model_len:
                # Current generation is complete. Flush unresolved audio before
                # resetting; the next chunk never sees a partially written run.
                await self.cut(event, 'token_budget')
        elif reply.finish_reason == 'length':
            event['output_truncated'] = True
        return event

    async def cut(self, event, reason):
        run = self.run
        if any(not b[3] for b in self.pending):
            await self.flush(event, force=True, trigger=reason)
        if reason=='time_window' and self.pending:
            # Original time-window policy: report unresolved speech instead
            # of carrying an unbounded ST buffer across window resets.
            self.discard([b for b in self.pending if not b[3]],event,'window_limit_unresolved')
            self.pending=[]
        if self.run == run:
            self.reset(event, reason)
        event['window_cut_trigger'] = reason
        # If the optional empty-ST reset is disabled, retain pending raw audio
        # for the next recovery even though model prompt history was cleared.
        if self.pending and all(b[3] for b in self.pending):
            self.pending = []
            self.voiced_waits = 0
            self.voiced_wait_samples=0

    def source_for_blocks(self, blocks):
        return ''.join(text for a,b,text in self.source_segments
                       if any(a < block[0]+block[2] and b > block[0] for block in blocks))

    def publish_source(self, text, event, blocks):
        """Source output is independent of target WAIT/EMIT decisions.

        ST may re-transcribe pending audio that already emitted source. Replace
        that interval explicitly instead of appending a duplicate transcript.
        """
        if self.mode != 'source_target' or not text or not blocks:
            return
        previous = self.full_source
        start = blocks[0][0]
        end = blocks[-1][0]+blocks[-1][2]
        self.source_segments = [(a,b,t) for a,b,t in self.source_segments if b <= start or a >= end]
        self.source_segments.append((start,end,text))
        self.source_segments.sort(key=lambda part: part[0])
        self.full_source = ''.join(t for _,_,t in self.source_segments)
        if self.full_source == previous:
            return
        if self.full_source.startswith(previous) and 'source_text_full' not in event:
            event['source_text'] = event.get('source_text','') + self.full_source[len(previous):]
        else:
            event['source_text'] = ''
            event['source_text_full'] = self.full_source

    def commit(self, reply, event, *, context=None, recovered_audio=None):
        self.full_text += reply.text
        event['text'] = event.get('text', '') + reply.text
        event['action'] = 'EMIT'
        blocks = recovered_audio if recovered_audio is not None else self.pending
        previous = context or {'audio': [], 'source': '', 'text': ''}
        self.last_emit = {'audio': previous['audio'] + list(blocks),
                          'source': previous['source'] + self.source_for_blocks(blocks),
                          'text': previous['text'] + reply.text}
        self.pending = []
        self.voiced_waits = 0
        self.voiced_wait_samples=0
        self.pending_attempt_key = None
        self.rejected_voice_revision=None

    async def flush(self, event, *, force=False, trigger='eof'):
        original = list(self.pending)
        blocks = list(original)
        if self.settings.st_trim_trailing_silence:
            while blocks and blocks[-1][3]:
                blocks.pop()
        if not blocks or not any(not b[3] for b in blocks):
            return
        if self.rejected_voice_revision==self.voice_revision and not self.eof:
            event.update(st_fallback_skipped='short_speech_rejection_waiting_for_new_voice',
                         st_output_rejected_audio_unchanged=True)
            return
        trimmed_ms = sum(b[2] for b in original[len(blocks):])//16
        cap = round(self.settings.st_fallback_max_seconds*16000)
        dropped = []
        while cap and len(blocks) > 1 and sum(b[2] for b in blocks) > cap:
            dropped.append(blocks.pop(0))
        if dropped:
            self.discard(dropped, event, 'st_audio_limit')
            self.pending = self.pending[len(dropped):]
        key = (blocks[0][0], blocks[-1][0]+blocks[-1][2])
        if key == self.pending_attempt_key and not force:
            event['st_fallback_skipped'] = 'unchanged_audio'
            return
        self.pending_attempt_key = key
        context = self.last_emit if self.settings.st_context else None
        all_blocks = (context['audio'] if context else []) + blocks
        samples = np.concatenate([b[1][:b[2]] for b in all_blocks])
        guard_stats=None
        st_vad=None
        crop_blocks=all_blocks
        has_long_chunk=any(b[2]==self.settings.chunk_ms*16 and len(b)>5
                          and any(end-start>=self.settings.vad_full_chunk_speech_ms*16
                                  for start,end in b[5]['speech_spans'])
                          for b in blocks) if self.settings.vad_full_chunk_speech_ms>0 else False
        if self.settings.st_vad_mode=='firered_nonstream':
            if self.st_activity is None:
                self.st_activity=await asyncio.to_thread(FireRedNonstreamActivity,
                    self.settings.st_vad_model_path,self.settings.st_vad_threshold,
                    cpu_threads=self.settings.vad_cpu_threads)
            crop_blocks,st_vad=await detect_st_blocks(all_blocks,self.st_activity)
            event.update(st_vad)
            # Context contributes to conditioning, not the new-text allowance.
            guard_stats=speech_stats(crop_blocks[-len(blocks):])
            if not guard_stats['st_vad_segments']:
                event.update(st_vad_rejected=True,st_fallback_skipped='nonstream_vad_no_speech',
                             st_fallback_audio_ms=sum(b[2] for b in blocks)/16)
                return
        if self.settings.st_crop_enabled and (st_vad is not None or not self.settings.simulst_crop_enabled):
            if self.settings.audio_mode=='standard':
                samples,kept=padded_st_audio(crop_blocks,round(self.settings.st_segment_pad_ms*16))
            else:
                samples,kept=padded_st_audio(crop_blocks,round(self.settings.vad_leading_pad_ms*16),
                                             round(self.settings.vad_pad_ms*16))
        elif self.settings.simulst_crop_enabled:
            kept=[span for block in all_blocks for span in block[4]]
        elif self.settings.st_crop_enabled:
            samples,kept=padded_st_audio(all_blocks,round(self.settings.st_segment_pad_ms*16))
        if self.settings.st_crop_enabled:
            if guard_stats is None:guard_stats=speech_stats(blocks)
            event.update(st_model_audio_ms=len(samples)/16,
                st_vad_removed_ms=(sum(b[2] for b in all_blocks)-len(samples))/16,
                st_retained_source_spans_ms=[[a/16,b/16] for a,b in kept],
                st_segment_pad_ms=(self.settings.st_segment_pad_ms if self.settings.audio_mode=='standard' else None))
            if not len(samples):
                event.update(st_fallback_skipped='no_speech_after_crop',st_vad_rejected=True)
                return
        parts = [{'type': 'text', 'text': ST_INSTRUCTIONS[self.direction]}]
        parts += [part(samples[i:i+480000], self.settings.model_audio_format)
                  for i in range(0, len(samples), 480000)]
        messages = [{'role': 'user', 'content': parts}]
        penalty = self.settings.st_repetition_penalty
        kwargs = {'max_tokens': self.settings.st_max_tokens, 'repetition_penalty': penalty, 'st': True}
        if context and self.mode == 'source_target':
            source_reply = await self.api.generate(messages, prefix=SRC + context['source'],
                                                   source_phase=True, **kwargs)
            prefix = SRC + context['source'] + source_reply.source + TGT + context['text']
            target = await self.api.generate(messages, prefix=prefix, target_only=True, **kwargs)
            reply = Reply(source_reply.source, target.text, target.raw, target.ids, target.stop,
                          target.prompt_tokens, target.completion_tokens, target.finish_reason)
        elif context:
            reply = await self.api.generate(messages, prefix=context['text'], target_only=True, **kwargs)
        else:
            reply = await self.api.generate(messages, **kwargs)
        self.st_count += 1
        event.update(st_attempted=True, st_context_used=context is not None, recovery_trigger=trigger,
            st_audio_ms=sum(b[2] for b in blocks)//16, st_trimmed_silence_ms=trimmed_ms,
            st_generated_token_ids=reply.ids, st_stop_reason=reply.stop,
            st_prompt_tokens=reply.prompt_tokens, st_completion_tokens=reply.completion_tokens,
            st_context_audio_ms=sum(b[2] for b in context['audio'])//16 if context else 0)
        self.publish_source(reply.source, event, blocks)
        guard_stats=guard_stats if guard_stats is not None else st_vad
        if self.settings.st_guard_enabled and guard_stats is not None:
            policy=short_speech_policy(guard_stats,self.direction,has_long_chunk,trigger)
            limit=policy['st_short_speech_token_limit']
            visible=(await self.api.tokenize(reply.text))['count'] if reply.text else 0
            event.update(**policy,st_short_speech_output_tokens=visible)
            if limit is not None and visible>limit:
                self.rejected_voice_revision=self.voice_revision
                event.update(st_output_rejected=True,
                    st_output_reject_reason='short_speech_excess_tokens',
                    st_rejected_text=reply.text)
                return
        if reply.finish_reason == 'length':
            event['st_output_truncated'] = True
        if not reply.text.strip():
            event.update(st_empty=True, unresolved_audio_ms=sum(b[2] for b in blocks)//16)
            if self.settings.reset_after_empty_st:
                self.discard(self.pending, event, 'st_empty_reset')
                self.pending = []
                self.empty_st_resets += 1
                self.reset(event, 'st_empty_reset')
            return
        self.commit(reply, event, context=context, recovered_audio=blocks)
        event['recovery'] = 'st'
        should_reset = ((trigger == 'audible_wait' and self.settings.reset_after_audible_st)
                        or (trigger == 'trailing_silence' and self.settings.reset_after_silence_st)
                        or reply.finish_reason == 'length')
        if should_reset:
            reason = 'st_audible_reset' if trigger == 'audible_wait' else 'st_silence_reset'
            self.reset(event, 'st_max_tokens' if reply.finish_reason == 'length' else reason)
        else:
            if self.messages and self.messages[-1]['role'] == 'assistant':
                self.messages[-1]['content'] = (SRC + reply.source + TGT + reply.text
                    if self.mode == 'source_target' else reply.text)
            event['recovery_context_retained'] = True
            # ST can add more text than an ordinary turn; count the next prompt
            # defensively before any future generation.

    async def finish(self):
        if self.eof:
            return
        self.eof = True
        if len(self.buffer):
            valid = len(self.buffer)
            size = (self.settings.first_chunk_ms if self.index == 0 else self.settings.chunk_ms)*16
            data = np.pad(self.buffer, (0, size-valid))
            self.buffer = self.buffer[:0]
            yield await self.step(data, valid, final=True)
        elif self.pending:
            event = {'event': 'flush', 'text': '', 'action': 'WAIT', 'end_ms': self.cursor//16}
            await self.flush(event, force=True, trigger='eof')
            yield event
        event = {'event': 'done', 'full_text': self.full_text, 'duration_ms': self.cursor//16,
                 'num_chunks': self.index, 'window_resets': self.run,
                 'st_recovery_count': self.st_count, 'empty_st_reset_count': self.empty_st_resets,
                 'skipped_untranslated_audio_ms': self.skipped_ms,
                 'unresolved_audio_ms': sum(b[2] for b in self.pending if not b[3])//16}
        if self.mode == 'source_target':
            event['source_text'] = self.full_source
        yield event

    async def close(self):
        self.closed = True
        await self.api.close()


class ServiceSession:
    """Optional adapter for an existing /translate/ws deployment."""
    def __init__(self, model, settings, mode, direction):
        self.model, self.settings, self.mode, self.direction = model, settings, mode, direction
        self.ws = None

    async def connect(self):
        url = self.model.base_url.rstrip('/').replace('https://', 'wss://').replace('http://', 'ws://')
        self.ws = await websockets.connect(url + '/translate/ws', max_size=16 * 1024 * 1024,
            ping_interval=None, additional_headers=auth(self.model.api_key_env),
            open_timeout=self.settings.timeout_seconds)
        await self.ws.send(json.dumps({"direction": self.direction}))
        ready = json.loads(await asyncio.wait_for(self.ws.recv(), self.settings.timeout_seconds))
        if ready.get('event') != 'ready':
            raise RuntimeError('翻译上游未就绪: ' + str(ready))
        return {**ready, "backend": "service"}

    async def send(self, samples):
        await self.ws.send(np.asarray(samples, dtype='<f4').tobytes())

    async def finish_input(self):
        await self.ws.send(json.dumps({'eof': True}))

    async def events(self):
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), self.settings.timeout_seconds)
            e = json.loads(raw)
            if e.get('final_recovery'):
                e['event'] = 'flush'
            if self.mode == 'target_only':
                e.pop('source_text', None)
            yield e
            if e.get('event') in ('done', 'error'):
                break

    async def close(self):
        if self.ws:
            await self.ws.close()


class MockSession:
    """Explicit protocol demo only; generated text is not a translation."""
    def __init__(self, model, settings, mode, direction):
        self.settings, self.mode, self.direction = settings, mode, direction
        self.buffer = np.empty(0, np.float32)
        self.cursor = self.index = 0
        self.text = ''

    def ready(self):
        return {'event':'ready', 'backend':'mock', 'demo_only':True,
                'chunk_ms':self.settings.chunk_ms,'first_chunk_ms':1600}

    async def push(self, samples):
        self.buffer = np.concatenate((self.buffer, samples))
        while len(self.buffer):
            n = (self.settings.first_chunk_ms if self.index == 0 else self.settings.chunk_ms) * 16
            if len(self.buffer) < n:
                break
            block, self.buffer = self.buffer[:n], self.buffer[n:]
            yield self.step(block)

    def step(self, samples):
        start = self.cursor; self.cursor += len(samples)
        voiced = len(samples) and np.max(np.abs(samples)) > .001
        text = (' [demo translation]' if self.direction == 'zh-en' else '【演示译文】') if voiced else ''
        self.text += text
        e = {'event':'chunk','index':self.index,'start_ms':start // 16,'end_ms':self.cursor // 16,
             'text':text,'action':'EMIT' if text else 'WAIT','demo_only':True}
        if self.mode == 'source_target':
            e['source_text'] = '【演示源文】' if voiced else ''
        self.index += 1
        return e

    async def finish(self):
        if len(self.buffer):
            yield self.step(self.buffer)
            self.buffer = self.buffer[:0]
        yield {'event':'done','full_text':self.text,'duration_ms':self.cursor // 16,
               'num_chunks':self.index,'demo_only':True}

    async def close(self):
        pass
