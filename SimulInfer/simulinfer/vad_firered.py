"""Optional streaming FireRedVAD; loaded only when selected."""
from pathlib import Path
import time
import numpy as np

class FireRedStreamActivity:
    """Official FireRedStreamVAD adapted to the service source timeline.

    FireRed uses a 25 ms analysis window with a 10 ms frame shift. Consecutive
    service chunks therefore share 15 ms of feature context. ``pending`` keeps
    that overlap while the official DFSMN cache and postprocessor keep their
    own per-session streaming state.
    """

    sample_rate = 16000
    frame_length = 400
    frame_shift = 160

    def __init__(self, path: str = '', threshold: float = 0.4, release_threshold: float = 0.35,
                 pad_ms: float = 240, *, model=None, cpu_threads=1):
        del release_threshold  # FireRed uses a state machine rather than hysteresis.
        try:
            from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
        except ImportError as exc:
            raise RuntimeError("FireRedVAD is not installed; install this project with the vad extra") from exc
        import torch
        if cpu_threads<1:raise ValueError('vad_cpu_threads must be positive')
        torch.set_num_threads(cpu_threads)
        self.cpu_threads=cpu_threads
        from .vad import resolve_firered_stream_path
        model_dir = resolve_firered_stream_path(path)
        config = FireRedStreamVadConfig(
            use_gpu=False,
            smooth_window_size=5,
            speech_threshold=float(threshold),
            pad_start_frame=5,
            min_speech_frame=8,
            max_speech_frame=2000,
            min_silence_frame=20,
            chunk_max_frame=30000,
        )
        self.detector = model if model is not None else FireRedStreamVad.from_pretrained(model_dir, config)
        self.threshold = float(threshold)
        self.pad_start_samples = max(0, round(float(pad_ms) * 16))
        self.end_sample = 0
        self.pending_start = 0
        self.pending = np.zeros(0, dtype=np.float32)
        self.frames = []  # (start, end, raw_prob, smoothed_prob, threshold_speech)
        self.intervals = []  # [start_sample, end_sample|None]
        self.last_voice_end = 0
        self.frames_processed = 0
        self.min_speech_frames = int(config.min_speech_frame)

    @staticmethod
    def _pcm16(samples):
        return np.clip(
            np.rint(np.clip(samples, -1.0, 1.0) * 32768.0), -32768, 32767
        ).astype('<i2')

    def _record_result(self, result):
        start = (int(result.frame_idx) - 1) * self.frame_shift
        end = start + self.frame_shift
        self.frames.append((start, end, float(result.raw_prob),
                            float(result.smoothed_prob), bool(result.is_speech)))
        if result.is_speech_start:
            interval_start = max(0, (int(result.speech_start_frame) - 1) * self.frame_shift)
            if self.intervals and self.intervals[-1][1] is None:
                self.intervals[-1][0] = min(self.intervals[-1][0], interval_start)
            else:
                self.intervals.append([interval_start, None])
        if result.is_speech_end:
            interval_end = max(0, int(result.speech_end_frame) * self.frame_shift)
            if self.intervals and self.intervals[-1][1] is None:
                self.intervals[-1][1] = interval_end
            else:
                # Defensive fallback for a max-duration boundary emitted in the
                # same result as the next start.
                self.intervals.append([max(0, interval_end-self.frame_shift), interval_end])
            self.last_voice_end = max(self.last_voice_end, interval_end)

    def _append_new_audio(self, samples, start_sample, end):
        if end <= self.end_sample:
            return 0
        offset = max(0, self.end_sample - start_sample)
        new = samples[offset:]
        if self.pending.size == 0:
            self.pending_start = self.end_sample
        self.pending = np.concatenate((self.pending, new))
        self.end_sample = end
        calls = 0
        if len(self.pending) >= self.frame_length:
            count = 1 + (len(self.pending)-self.frame_length)//self.frame_shift
            needed = self.frame_length + (count-1)*self.frame_shift
            pcm = self._pcm16(self.pending[:needed])
            import torch
            with torch.inference_mode():
                results = self.detector.detect_chunk(pcm)
            if len(results) != count:
                raise RuntimeError(
                    f'FireRedStreamVAD returned {len(results)} frames for expected {count}'
                )
            for result in results:
                self._record_result(result)
            consumed = count * self.frame_shift
            self.pending = self.pending[consumed:]
            self.pending_start += consumed
            calls = count
            self.frames_processed += count
        return calls

    def analyze(self, waveform, start_sample: int) -> dict:
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not np.isfinite(samples).all():
            raise ValueError('VAD input contains NaN or infinity; expected float32 PCM')
        start_sample = int(start_sample)
        end = start_sample + len(samples)
        if start_sample > self.end_sample:
            raise ValueError('VAD source timeline has a gap')
        t0 = time.monotonic()
        calls = self._append_new_audio(samples, start_sample, end)

        mask = np.zeros(len(samples), dtype=np.bool_)
        for left, right in self.intervals:
            interval_end = self.end_sample if right is None else right
            a = max(start_sample, left)
            b = min(end, interval_end)
            if b > a:
                mask[a-start_sample:b-start_sample] = True

        raw_mask=mask.copy()

        # A not-yet-confirmed onset at the live edge must not be destroyed. Keep
        # the raw-positive tail and a short lead-in until the 80 ms minimum
        # speech decision is available on the next call.
        uncertain = self.frames[-self.min_speech_frames:]
        positives = [frame for frame in uncertain if frame[4]]
        if positives:
            a = max(start_sample, positives[0][0]-self.pad_start_samples)
            b = min(end, self.end_sample)
            if b > a:
                mask[a-start_sample:b-start_sample] = True

        speech_samples = int(mask.sum())
        if speech_samples:
            positions = np.flatnonzero(mask)
            self.last_voice_end = max(self.last_voice_end, start_sample+int(positions[-1])+1)
        relevant = [f for f in self.frames if f[1] > start_sample and f[0] < end]
        raw = [f[2] for f in relevant]
        smooth = [f[3] for f in relevant]
        cutoff = self.end_sample - 32*self.sample_rate
        if cutoff > 0:
            self.frames = [f for f in self.frames if f[1] > cutoff]
            self.intervals = [x for x in self.intervals if x[1] is None or x[1] > cutoff]
        return {
            'vad_mode': 'firered_stream',
            'vad_device': 'cpu',
            'vad_cpu_threads':self.cpu_threads,
            'vad_speech_samples': speech_samples,
            'vad_speech_ms': round(speech_samples/16, 3),
            'vad_probability_max': round(max(raw, default=0.0), 6),
            'vad_probability_mean': round(float(np.mean(raw)) if raw else 0.0, 6),
            'vad_smoothed_probability_max': round(max(smooth, default=0.0), 6),
            'vad_threshold': self.threshold,
            'vad_trailing_silence_samples': max(0, end-self.last_voice_end),
            'vad_trailing_silence_ms': max(0, end-self.last_voice_end)//16,
            'vad_new_frames': calls,
            'vad_seconds': round(time.monotonic()-t0, 6),
            'is_silence': speech_samples == 0,
            '_vad_speech_mask': mask,
            '_vad_raw_speech_mask': raw_mask,
        }
