"""Optional CPU speech detection on generated PCM, independent of input VAD."""
import asyncio

import numpy as np

from .tts_audio import TTSStreamError


class SpeechTail:
    """Keep speech and short gaps; hold uncertain trailing non-speech.

    Only a generated segment's leading/trailing non-speech is discarded.
    Interior pauses are released unchanged if speech resumes before the timeout.
    A sustained non-speech run is an incomplete generation, never a fake EOS.
    """
    def __init__(self, rate, emit, *, detector=None, max_tail_seconds=2.4):
        self.rate, self.emit = rate, emit
        self.detector = detector
        self.max_tail = round(max_tail_seconds * rate)
        self.pad = round(.24 * rate)
        self.frame = round(.2 * rate)
        self.incoming = bytearray()
        self.pending = bytearray()
        self.position = self.pending_start = self.last_speech = 0
        self.opened = False

    async def start(self):
        if self.detector is None:
            from .vad_firered import FireRedStreamActivity
            self.detector = await asyncio.to_thread(FireRedStreamActivity, pad_ms=240)

    async def feed(self, raw):
        self.incoming.extend(raw)
        while len(self.incoming) >= self.frame * 2:
            frame = bytes(self.incoming[:self.frame*2])
            del self.incoming[:self.frame*2]
            await self._process(frame)

    async def _process(self, raw):
        samples = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768
        if not len(samples):
            return
        start = self.position
        self.pending.extend(raw)
        self.position += len(samples)
        converted = samples
        if self.rate != 16000:
            import soxr
            converted = soxr.resample(samples, self.rate, 16000)
        result = await asyncio.to_thread(self.detector.analyze, converted, round(start * 16000 / self.rate))
        voiced = np.flatnonzero(result['_vad_speech_mask'])
        if len(voiced):
            if not self.opened:
                first = start + round(int(voiced[0]) * self.rate / 16000)
                keep_start = max(self.pending_start, first - self.pad)
                del self.pending[:(keep_start-self.pending_start)*2]
                self.pending_start = keep_start
                self.opened = True
            self.last_speech = start + round((int(voiced[-1])+1) * self.rate / 16000)
        if self.opened:
            end = min(self.position, self.last_speech + self.pad)
            count = max(0, end-self.pending_start)*2
            if count:
                await self.emit(bytes(self.pending[:count]))
                del self.pending[:count]
                self.pending_start = end
            if self.position-self.last_speech >= self.max_tail:
                self.pending.clear()
                raise TTSStreamError('TTS stopped after sustained non-speech output; speech completion is unverified')
        else:
            # Bound silent leading output while keeping enough onset context.
            keep = self.pad * 2
            if len(self.pending) > keep:
                remove = len(self.pending)-keep
                del self.pending[:remove]
                self.pending_start += remove//2
            if self.position >= self.rate*6:
                raise TTSStreamError('TTS generated no speech before the leading-audio limit')

    async def finish(self):
        if self.incoming:
            raw = bytes(self.incoming)
            self.incoming.clear()
            await self._process(raw)
        self.pending.clear()
        if not self.opened:
            raise TTSStreamError('TTS returned no detected speech')
