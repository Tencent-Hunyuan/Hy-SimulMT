import base64
import io
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
MIN_INPUT_RATE = 8000
MAX_INPUT_RATE = 192000
MAX_INPUT_CHANNELS = 8


def decode_file(data: bytes, max_samples: int) -> np.ndarray:
    return decode_file_with_info(data, max_samples)[0]


def validate_input_format(sample_rate, channels):
    if type(sample_rate) is not int or not MIN_INPUT_RATE <= sample_rate <= MAX_INPUT_RATE:
        raise ValueError(f'输入采样率需为{MIN_INPUT_RATE}～{MAX_INPUT_RATE}Hz的整数')
    if type(channels) is not int or not 1 <= channels <= MAX_INPUT_CHANNELS:
        raise ValueError(f'输入声道数需为1～{MAX_INPUT_CHANNELS}的整数')


def decode_file_with_info(data: bytes, max_samples: int):
    """Decode supported audio containers, downmix, and resample to model format.

    The limit is counted in output mono16k samples, independent of the uploaded
    sample rate/channel count. Read in bounded blocks rather than expanding an
    arbitrary multichannel file in memory first.
    """
    if type(max_samples) is not int or max_samples < 1:
        raise ValueError('max_samples must be a positive integer')
    if not data:
        raise ValueError('音频为空')
    try:
        audio = sf.SoundFile(io.BytesIO(data))
    except (RuntimeError, ValueError, sf.LibsndfileError):
        return _decode_ffmpeg(data, max_samples)
    with audio:
        validate_input_format(audio.samplerate, audio.channels)
        if len(audio) * SAMPLE_RATE > max_samples * audio.samplerate:
            raise ValueError('解码后的音频超过配置时长上限')
        converter = PCM('f32le', audio.samplerate, audio.channels)
        pieces = []
        for block in audio.blocks(blocksize=audio.samplerate, dtype='float32', always_2d=True):
            pieces.append(converter.feed(np.asarray(block, dtype='<f4').tobytes()))
        pieces.append(converter.finish())
        samples = np.concatenate(pieces)
        info = converter.info()
        info['container'] = audio.format
        info['decoder'] = 'soundfile'
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError('音频为空或含NaN/Inf')
    if len(samples) > max_samples:
        raise ValueError('解码后的音频超过配置时长上限')
    return samples, info


def _decode_ffmpeg(data, max_samples):
    """Fallback for compressed containers, with bounded output/time and no URLs.

    ffmpeg identifies the container sample rate and channels itself. Its decoder
    converts directly to mono16k; no second resampling is applied afterward.
    """
    import imageio_ffmpeg
    from pathlib import Path
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix='simulst-audio-') as directory:
        source = Path(directory) / 'input.audio'
        output = Path(directory) / 'decoded.f32'
        source.write_bytes(data)
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error', '-y',
                   '-protocol_whitelist', 'file,pipe', '-i', str(source), '-map', '0:a:0',
                   '-vn', '-sn', '-dn', '-ac', '1', '-ar', str(SAMPLE_RATE),
                   '-t', str((max_samples + 16) / SAMPLE_RATE), '-f', 'f32le', str(output)]
        try:
            with (Path(directory) / 'decode.log').open('wb') as errors:
                completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=errors,
                                           timeout=120, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ValueError('音频解码超时') from exc
        if completed.returncode or not output.exists():
            raise ValueError('无法解码音频；请提供有效的WAV、FLAC、MP3、M4A或WebM等音频文件')
        size = output.stat().st_size
        if size > max_samples * 4:
            raise ValueError('解码后的音频超过配置时长上限')
        if not size or size % 4:
            raise ValueError('音频为空或解码结果不完整')
        samples = np.fromfile(output, dtype='<f4')
        if not np.isfinite(samples).all():
            raise ValueError('音频含NaN/Inf')
        return samples, {'sample_rate': None, 'channels': None, 'input_format': 'container',
                         'output_sample_rate': SAMPLE_RATE, 'output_channels': 1,
                         'output_samples': len(samples), 'duration_ms': len(samples)*1000/SAMPLE_RATE,
                         'decoder': 'ffmpeg', 'converted': True}


def float_wav(samples):
    buf = io.BytesIO()
    sf.write(buf, np.asarray(samples, dtype=np.float32), SAMPLE_RATE, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def pcm16_wav(samples):
    """Encode normalized samples as little-endian PCM16 without mutating them.

    Use 32768 and round before casting so PCM16 decoded as float32 round-trips
    exactly, including -32768. Values outside [-1, 1] saturate, never wrap.
    """
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("模型音频需为有限值的单声道PCM")
    pcm = np.clip(np.rint(np.clip(values, -1.0, 1.0) * 32768.0),
                  -32768, 32767).astype('<i2')
    buf = io.BytesIO()
    sf.write(buf, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16", endian="LITTLE")
    return buf.getvalue()


class AudioPart(dict):
    """JSON-compatible input_audio with local-only sample-count metadata."""
    def __init__(self, samples, audio_format="pcm16"):
        self.num_samples = len(samples)
        if audio_format not in {"pcm16", "float32"}:
            raise ValueError("model_audio_format 必须为 pcm16 或 float32")
        data = pcm16_wav(samples) if audio_format == "pcm16" else float_wav(samples)
        super().__init__(type="input_audio", input_audio={
            "data": base64.b64encode(data).decode("ascii"), "format": "wav"})


def part(samples, audio_format="pcm16"):
    return AudioPart(samples, audio_format)


class PCM:
    def __init__(self, fmt, sample_rate=SAMPLE_RATE, channels=1):
        if fmt not in {"s16le", "f32le"}:
            raise ValueError("input_format 必须为 s16le 或 f32le")
        validate_input_format(sample_rate, channels)
        self.fmt, self.width, self.tail = fmt, 2 if fmt == "s16le" else 4, b""
        self.sample_rate, self.channels = sample_rate, channels
        self.input_frames = self.output_samples = 0
        self.finished = False
        self.resampler = None
        if sample_rate != SAMPLE_RATE:
            import soxr
            self.resampler = soxr.ResampleStream(sample_rate, SAMPLE_RATE, 1, dtype='float32', quality='HQ')

    def feed(self, data):
        if self.finished:
            raise ValueError('EOF后不得继续转换音频')
        data = self.tail + data
        frame_bytes = self.width * self.channels
        n = len(data) // frame_bytes * frame_bytes
        self.tail = data[n:]
        a = np.frombuffer(data[:n], dtype="<i2" if self.width == 2 else "<f4").astype(np.float32)
        if self.width == 2:
            a /= 32768.0
        if not np.isfinite(a).all():
            raise ValueError("PCM含NaN/Inf")
        if not a.size:
            return a
        if self.channels > 1:
            # Float64 accumulation avoids overflow for finite f32 multichannel
            # inputs. Equal-weight downmix preserves amplitude, no AGC/gating.
            a = a.reshape(-1, self.channels).mean(axis=1, dtype=np.float64).astype(np.float32)
        self.input_frames += len(a)
        if self.resampler is not None:
            a = self.resampler.resample_chunk(a, last=False)
        if not np.isfinite(a).all():
            raise ValueError('重采样输出含非有限数值，请检查输入幅度')
        self.output_samples += len(a)
        return a

    def finish(self):
        if self.finished:
            return np.empty(0, dtype=np.float32)
        if self.tail:
            raise ValueError("音频在不完整PCM采样处结束")
        self.finished = True
        tail = np.empty(0, dtype=np.float32)
        if self.resampler is not None and self.input_frames:
            tail = self.resampler.resample_chunk(tail, last=True)
        if not np.isfinite(tail).all():
            raise ValueError('重采样尾部含非有限数值，请检查输入幅度')
        self.output_samples += len(tail)
        return tail

    def info(self):
        return {'sample_rate': self.sample_rate, 'channels': self.channels, 'input_format': self.fmt,
                'output_sample_rate': SAMPLE_RATE, 'output_channels': 1,
                'input_frames': self.input_frames, 'output_samples': self.output_samples,
                'duration_ms': self.input_frames * 1000 / self.sample_rate,
                'converted': self.sample_rate != SAMPLE_RATE or self.channels != 1}
