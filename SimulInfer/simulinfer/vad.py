"""CPU voice detection with session-local state and source-time decisions.

Translation can query the same audio twice while assembling a 1600 ms opening.
Only previously unseen samples advance the recurrent VAD. Decisions remain on
the source timeline even when the translation run resets.
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
import math
import time

import numpy as np

BUNDLED_MODEL = Path(__file__).parent / 'models/silero_vad_6.2.3/silero_vad_6.2.3.jit'
BUNDLED_FIRERED_MODEL = Path(__file__).parent / 'models/firered_vad'
BUNDLED_FIRERED_STREAM_MODEL = Path(__file__).parent / 'models/firered_stream_vad'


@lru_cache(maxsize=4)
def model_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


class SileroActivity:
    sample_rate = 16000
    frame_samples = 512

    def __init__(self, path: str, threshold: float = 0.5, release_threshold: float = 0.35,
                 pad_ms: float = 160, *, cpu_threads: int = 1, model=None):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Silero VAD 需要可选依赖：pip install 'simulinfer[vad]'") from exc
        if cpu_threads < 1:
            raise ValueError('vad_cpu_threads must be positive')
        if torch.get_num_threads() != cpu_threads:
            torch.set_num_threads(cpu_threads)

        if not 0 < release_threshold <= threshold < 1:
            raise ValueError('VAD requires 0 < release_threshold <= threshold < 1')
        if not math.isfinite(pad_ms) or not 0 <= pad_ms <= 1000:
            raise ValueError('VAD pad_ms must be between 0 and 1000')
        self.torch = torch
        # Sharing mutable TorchScript model instances would share recurrent
        # state. Cache only serialized weights and load an independent module.
        self.model = model if model is not None else torch.jit.load(
            io.BytesIO(model_bytes(resolve_model_path(path))), map_location='cpu')
        self.model.eval()
        self.model.reset_states()
        self.threshold = threshold
        self.release_threshold = release_threshold
        self.pad_samples = round(pad_ms * 16)
        self.frames = []  # (start, end, probability, padded_speech, raw_speech)
        self.end_sample = 0
        self.last_voice_end = 0
        self.voice_active = False
        self.tail_samples = 0
        self.frames_processed = 0

    def analyze(self, waveform, start_sample: int) -> dict:
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not np.isfinite(samples).all():
            raise ValueError('VAD input contains NaN or infinity; expected float32 PCM')
        start_sample = int(start_sample)
        end = start_sample + len(samples)
        if start_sample > self.end_sample:
            raise ValueError('VAD source timeline has a gap')
        if self.frames and start_sample < self.frames[0][0]:
            raise ValueError('VAD query is older than retained source decisions')
        t0 = time.monotonic()
        calls = 0
        if end > self.end_sample:
            if self.end_sample % self.frame_samples:
                raise ValueError('cannot append audio after a partial final VAD frame')
            with self.torch.inference_mode():
                for pos in range(self.end_sample, end, self.frame_samples):
                    segment = samples[pos-start_sample:pos-start_sample+self.frame_samples]
                    actual = len(segment)
                    block = np.zeros(self.frame_samples, dtype=np.float32)
                    block[:actual] = segment
                    prob = float(self.model(self.torch.from_numpy(block), self.sample_rate).item())
                    if not math.isfinite(prob):
                        raise RuntimeError('Silero VAD returned a non-finite probability')
                    if prob >= self.threshold:
                        self.voice_active = True
                    elif prob < self.release_threshold:
                        self.voice_active = False
                    if self.voice_active:
                        self.tail_samples = self.pad_samples
                        speech = True
                    else:
                        speech = self.tail_samples > 0
                        self.tail_samples = max(0, self.tail_samples-actual)
                    if speech:
                        self.last_voice_end = pos + actual
                    self.frames.append((pos, pos+actual, prob, speech, self.voice_active))
                    calls += 1
            self.end_sample = end
            self.frames_processed += calls

        relevant = [f for f in self.frames if f[1] > start_sample and f[0] < end]
        speech_mask = np.zeros(len(samples), dtype=np.bool_)
        raw_speech_mask = np.zeros(len(samples), dtype=np.bool_)
        for a, b, _, speech, raw_speech in relevant:
            if speech:
                left = max(start_sample, a) - start_sample
                right = min(end, b) - start_sample
                speech_mask[left:right] = True
            if raw_speech:
                left = max(start_sample, a) - start_sample
                right = min(end, b) - start_sample
                raw_speech_mask[left:right] = True
        speech_samples = sum(
            min(end,b)-max(start_sample,a) for a,b,_,speech,_ in relevant if speech)
        probabilities = [p for _,_,p,_,_ in relevant]
        # Keep enough past decisions for opening aggregation, not the entire
        # call. The model's state already contains its own small audio context.
        cutoff = self.end_sample - 32*self.sample_rate
        if cutoff > 0:
            self.frames = [f for f in self.frames if f[1] > cutoff]
        return {
            'vad_mode': 'silero',
            'vad_speech_samples': speech_samples,
            'vad_speech_ms': round(speech_samples/16, 3),
            'vad_probability_max': round(max(probabilities, default=0.0), 6),
            'vad_probability_mean': round(float(np.mean(probabilities)) if probabilities else 0.0, 6),
            'vad_threshold': self.threshold,
            'vad_trailing_silence_samples': max(0, end-self.last_voice_end),
            'vad_trailing_silence_ms': max(0, end-self.last_voice_end)//16,
            'vad_new_frames': calls,
            'vad_seconds': round(time.monotonic()-t0, 6),
            'is_silence': speech_samples == 0,
            '_vad_speech_mask': speech_mask,
            '_vad_raw_speech_mask': raw_speech_mask,
        }


class FireRedNonstreamActivity:
    """Whole-span FireRedVAD gate applied immediately before model ST."""
    sample_rate = 16000

    def __init__(self, path: str = '', threshold: float = 0.4, *, detector=None, cpu_threads=1):
        if detector is None:
            try:
                from fireredvad import FireRedVad, FireRedVadConfig
            except ImportError as exc:
                raise RuntimeError("FireRedVAD需要可选依赖：pip install 'simulinfer[vad]'") from exc
            import torch
            torch.set_num_threads(cpu_threads)
            model_dir = resolve_firered_model_path(path)
            config = FireRedVadConfig(use_gpu=False, smooth_window_size=5,
                speech_threshold=float(threshold), min_speech_frame=20,
                max_speech_frame=2000, min_silence_frame=20,
                merge_silence_frame=0, extend_speech_frame=0, chunk_max_frame=30000)
            detector = FireRedVad.from_pretrained(model_dir, config)
        self.detector = detector
        self.threshold = float(threshold)
        self.cpu_threads=cpu_threads

    def analyze(self, waveform) -> dict:
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not np.isfinite(samples).all():
            raise ValueError('ST VAD input contains NaN or infinity')
        pcm16 = np.clip(np.rint(np.clip(samples, -1, 1)*32768), -32768, 32767).astype('<i2')
        started = time.monotonic()
        # The feature extractor needs a 25ms window; fewer samples contain no
        # analyzable frame. Do not fabricate model input from a padded fragment.
        if len(pcm16)<400:
            result, probabilities={'timestamps':[]},np.empty(0,dtype=np.float32)
        else:
            import torch
            with torch.inference_mode():
                if len(pcm16)<560 and hasattr(self.detector,'vad_postprocessor'):
                    # FireRed 0.0.2 squeezes a single frame to a scalar before
                    # postprocessing. Normalize that one-frame result to a list.
                    _, probabilities = self.detector.detect(pcm16,do_postprocess=False)
                    decisions=self.detector.vad_postprocessor.process(probabilities.reshape(-1).tolist())
                    result={'timestamps':self.detector.vad_postprocessor.decision_to_segment(
                        decisions,len(pcm16)/self.sample_rate)}
                else:
                    result, probabilities = self.detector.detect(pcm16)
        segments = [(float(a), float(b)) for a,b in result.get('timestamps', []) if float(b)>float(a)]
        total = sum(b-a for a,b in segments)
        longest = max((b-a for a,b in segments), default=0.0)
        probs = np.asarray(probabilities.detach().cpu() if hasattr(probabilities, 'detach') else probabilities)
        return {'st_vad_mode':'firered_nonstream','st_vad_threshold':self.threshold,
            'st_vad_audio_ms':len(samples)//16,'st_vad_speech_ms':round(total*1000,3),
            'st_vad_longest_speech_ms':round(longest*1000,3),
            'st_vad_segments':[[round(a,3),round(b,3)] for a,b in segments],
            'st_vad_probability_max':round(float(probs.max()),6) if probs.size else 0.0,
            'st_vad_probability_mean':round(float(probs.mean()),6) if probs.size else 0.0,
            'st_vad_seconds':round(time.monotonic()-started,6),'st_vad_accepted':bool(segments)}


def resolve_model_path(path: str) -> str:
    """An explicit path overrides the bundled, versioned JIT model.

    Empty paths work offline without installing/importing the silero-vad
    distribution. Each session loads its own recurrent model instance.
    """
    candidate = Path(path).expanduser().resolve() if path else BUNDLED_MODEL.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f'Silero VAD 模型不存在: {candidate}')
    return str(candidate)


def resolve_firered_model_path(path: str) -> str:
    candidate = Path(path).expanduser().resolve() if path else BUNDLED_FIRERED_MODEL.resolve()
    if not (candidate/'model.pth.tar').is_file() or not (candidate/'cmvn.ark').is_file():
        raise FileNotFoundError(f'FireRedVAD模型不完整: {candidate}')
    return str(candidate)


def resolve_firered_stream_path(path: str) -> str:
    candidate = Path(path).expanduser().resolve() if path else BUNDLED_FIRERED_STREAM_MODEL.resolve()
    if not (candidate/'model.pth.tar').is_file() or not (candidate/'cmvn.ark').is_file():
        raise FileNotFoundError(f'FireRedStreamVAD模型不完整: {candidate}')
    return str(candidate)


class EnergyActivity:
    def __init__(self, threshold: float, *, disabled=False):
        self.threshold = threshold
        self.disabled = disabled

    def analyze(self, waveform, start_sample: int) -> dict:
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not np.isfinite(samples).all():
            raise ValueError('VAD input contains NaN or infinity; expected float32 PCM')
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
        return {'vad_mode': 'off' if self.disabled else 'energy',
                'rms': rms, 'energy_is_silence': rms <= self.threshold,
                'is_silence': False if self.disabled else rms <= self.threshold,
                '_vad_speech_mask': np.ones(len(samples), dtype=np.bool_) if self.disabled or rms > self.threshold
                                    else np.zeros(len(samples), dtype=np.bool_)}


def create_activity(settings):
    if settings.vad_mode == 'firered':
        from .vad_firered import FireRedStreamActivity
        return FireRedStreamActivity(settings.vad_model_path,settings.vad_threshold,
                                     settings.vad_release_threshold,settings.vad_pad_ms,
                                     cpu_threads=settings.vad_cpu_threads)
    if settings.vad_mode == 'silero':
        return SileroActivity(settings.vad_model_path, settings.vad_threshold,
            settings.vad_release_threshold, settings.vad_pad_ms,
            cpu_threads=settings.vad_cpu_threads)
    return EnergyActivity(settings.silence_rms, disabled=settings.vad_mode == 'off')
