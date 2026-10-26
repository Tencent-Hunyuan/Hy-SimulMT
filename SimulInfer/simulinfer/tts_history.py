"""Aligned text/audio reference history, isolated per synthesis conversation."""
from dataclasses import dataclass
import base64
import io
import wave


@dataclass(frozen=True)
class SpeechPair:
    text: str
    pcm: bytes
    sample_rate: int

    @property
    def seconds(self):
        return len(self.pcm) / (2 * self.sample_rate)


class AudioHistory:
    """Evict complete pairs; keep the newest complete utterance if it is long.

    A strict per-response safety limit is applied by the caller. The history
    window is soft: cropping PCM without an aligned transcript would lie to the
    next model request. Text and audio are retained as complete pairs.
    """
    def __init__(self, window_seconds=10, max_chars=240):
        self.window_seconds = window_seconds
        self.max_chars = max_chars
        self.entries = []
        self.key = None

    def prepare(self, language, voice):
        key = (language, voice)
        if self.key is not None and self.key != key:
            self.clear()
        self.key = key

    def clear(self):
        self.entries.clear()

    def commit(self, text, pcm, sample_rate, language, voice):
        self.prepare(language, voice)
        if not text.strip() or not pcm or len(pcm) % 2:
            raise ValueError('History requires nonempty text and complete PCM16 audio')
        if self.entries and self.entries[-1].sample_rate != sample_rate:
            self.clear()
        # Qwen Base references are bounded; retain neither a cropped transcript
        # pair nor an arbitrarily long generated reference. Reuse the anchor.
        if len(pcm) / (2 * sample_rate) > 30:
            self.clear()
            return
        self.entries.append(SpeechPair(text, bytes(pcm), sample_rate))
        while len(self.entries) > 1 and (
            self.seconds > self.window_seconds or sum(len(e.text) for e in self.entries) > self.max_chars
        ):
            self.entries.pop(0)

    @property
    def seconds(self):
        return sum(e.seconds for e in self.entries)

    def reference(self, language, voice):
        self.prepare(language, voice)
        if not self.entries:
            return None
        separator = ' ' if language == 'en' else ''
        return {'ref_audio': wav_data_url(b''.join(e.pcm for e in self.entries), self.entries[0].sample_rate),
                'ref_text': separator.join(e.text for e in self.entries)}


def wav_data_url(pcm, sample_rate):
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))
        wav.writeframes(pcm)
    return 'data:audio/wav;base64,' + base64.b64encode(output.getvalue()).decode('ascii')
