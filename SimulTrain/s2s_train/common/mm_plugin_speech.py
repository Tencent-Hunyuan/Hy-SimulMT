"""Hy-SimulMT 语音数据插件。

职责：
1. 按 Adapter 输出长度展开 `<audio>` 占位符。
2. 产出 `input_features` 与 `feature_attention_mask`，
   由 LlamaFactory 的 collator 自动并入 batch。

FSK Adapter 使用变长 token；Q-Former 模式仍支持固定 token 数。

音频解码不走 torchaudio：本机未安装 FFmpeg 共享库，torchcodec 后端
不可用。这里改用 imageio-ffmpeg 自带的 ffmpeg 可执行文件直接解码为
单声道 16k float32。
"""

import json
import subprocess
import wave
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from llamafactory.data.mm_plugin import AUDIO_PLACEHOLDER, BasePlugin
from typing_extensions import override

if TYPE_CHECKING:
    import torch
    from llamafactory.data.mm_plugin import AudioInput, ImageInput, MMProcessor, VideoInput


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_audio(path: str, sampling_rate: int) -> np.ndarray:
    """用 ffmpeg 解码任意音频为单声道 float32 波形。"""
    command = [
        _ffmpeg_exe(),
        "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sampling_rate),
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


AUDIO_LEN_INDEX_NAME = "audio_lens.json"

# 路径 -> 采样点数 的索引，按数据集目录缓存。
# 键是 index 文件所在目录的绝对路径，值是 {相对路径: 采样点数}。
_LEN_INDEX_CACHE: dict[str, dict[str, int]] = {}


def load_audio_len_index(media_dir: str) -> dict[str, int]:
    """加载 `<media_dir>/audio_lens.json`，没有则返回空 dict。

    为什么要这个索引：变长模式下 tokenize 阶段就要知道每条音频多长，
    而 mm_plugin 只拿到路径。实测共享盘（apdcephfs）冷读一次 wav 文件头
    要 **36.97 ms** —— 341 万条 ÷ 8 进程 = 4.4 小时，而 tokenize 本身
    只要 4 分钟。整体载入一个 400 MB 的 dict 换掉这 4.4 小时 IO。

    索引由 dataset/data_new/build_audio_len_index.py 生成。
    没有索引时会回落到逐条读文件头 —— 能跑，但在共享盘上会非常慢，
    所以第一次用会打一条明显的警告。
    """
    key = str(Path(media_dir).resolve())
    if key in _LEN_INDEX_CACHE:
        return _LEN_INDEX_CACHE[key]

    index_path = Path(key) / AUDIO_LEN_INDEX_NAME
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        print(
            f"[mm_plugin] 已载入音频长度索引 {index_path.name}："
            f"{len(index):,} 条（避免逐条读文件头）"
        )
    else:
        index = {}
        print(
            f"[mm_plugin] 警告：没找到 {index_path}，将逐条读音频文件头。\n"
            f"[mm_plugin] 共享盘上这会让 tokenize 慢几十倍（实测 37 ms/条）。\n"
            f"[mm_plugin] 建议先跑：python dataset/data_new/build_audio_len_index.py {key}"
        )

    _LEN_INDEX_CACHE[key] = index
    return index


@lru_cache(maxsize=1 << 16)
def audio_num_samples(path: str, sampling_rate: int) -> int:
    """音频的采样点数。变长模式下用来推算语音 token 数。

    只读 wav 的 44 字节文件头，不完整解码（实测两者结果一致，8/8 差 0）。
    非 wav（或文件头异常）回落到完整解码 —— 慢但一定对。

    注意：在共享盘上单次冷读约 37 ms，341 万条会要几小时。批量场景
    应该走 load_audio_len_index() 查表，这个函数只用于索引缺失时兜底
    与单条推理。lru_cache 覆盖同一条被重复读的情况。
    """
    if str(path).lower().endswith(".wav"):
        try:
            with wave.open(str(path)) as handle:
                if handle.getframerate() == sampling_rate:
                    return handle.getnframes()
                # 采样率不同要按比例换算，否则帧数会错
                return int(handle.getnframes() * sampling_rate / handle.getframerate())
        except (wave.Error, EOFError, OSError):
            pass  # 落到下面的完整解码

    return len(decode_audio(str(path), sampling_rate))


@dataclass
class SpeechAudioPlugin(BasePlugin):
    # 定长模式（Q-Former）的每条语音 token 数
    num_speech_tokens: int = 30
    # 变长模式：不为 None 时按每条音频的真实时长展开占位符。
    # 值是一个 callable(mel_frames) -> int，由 register.py 传入
    # config.speech_tokens_for —— 必须与 Adapter 的输出长度算法同源，
    # 否则占位符数量与 embedding 数量对不上，forward 会直接报错。
    speech_tokens_for: Optional[object] = None
    sampling_rate: int = 16000
    # 音频长度索引 {相对路径: 采样点数}，由 load_audio_len_index() 填。
    # 惰性加载：第一次遇到路径时按其所在目录往上找 audio_lens.json。
    _len_index: Optional[dict] = None

    def _lookup_num_samples(self, path: str) -> int:
        """查采样点数：先索引，再回落到读文件头。

        索引键是相对 media_dir 的路径，而这里拿到的是 LlamaFactory 拼好的
        绝对路径（converter.py:57 `os.path.join(media_dir, ...)`）。
        用后缀匹配来对应 —— 比反推 media_dir 稳，因为 plugin 拿不到 data_args。
        """
        if self._len_index is None:
            # 索引文件与数据集同目录。从音频路径往上找，最多 6 层
            # （音频通常是 <数据集>/audio/<子集>/<分片>/xxx.wav）。
            probe = Path(path).parent
            for _ in range(6):
                if (probe / AUDIO_LEN_INDEX_NAME).is_file():
                    self._len_index = load_audio_len_index(str(probe))
                    break
                if probe.parent == probe:
                    break
                probe = probe.parent
            else:
                self._len_index = {}
            if self._len_index is None:
                self._len_index = {}

        if self._len_index:
            text = str(path)
            # 常见情形：索引键就是绝对路径的后缀，逐段剥前缀来找
            parts = Path(text).parts
            for start in range(len(parts)):
                candidate = "/".join(parts[start:])
                hit = self._len_index.get(candidate)
                if hit is not None:
                    return hit

        return audio_num_samples(str(path), self.sampling_rate)

    def _tokens_for_audio(self, audio: "AudioInput") -> int:
        """这条音频要展开多少个占位符。

        走 config.speech_tokens_for_samples()（采样数 -> token 数），
        那里统一做 mel 帧的向上取整。不要在这里自己算 //160 ——
        曾因此与 Adapter 输出差 1 而训练报错（详见那个方法的注释）。
        """
        if self.speech_tokens_for is None:
            return self.num_speech_tokens

        if isinstance(audio, np.ndarray):
            n_samples = len(audio)
        else:
            n_samples = self._lookup_num_samples(str(audio))

        return int(self.speech_tokens_for(n_samples))

    @override
    def _regularize_audios(self, audios: list["AudioInput"], sampling_rate: float, **kwargs):
        results, sampling_rates = [], []
        for audio in audios:
            if not isinstance(audio, np.ndarray):
                audio = decode_audio(audio, int(sampling_rate))

            results.append(audio)
            sampling_rates.append(sampling_rate)

        return {"audios": results, "sampling_rates": sampling_rates}

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)

        # 每个 <audio> 按出现顺序对应 audios 里的一条，逐条算展开长度。
        # 定长模式下所有条都是 num_speech_tokens；变长模式随时长变。
        audio_index = 0
        for message in messages:
            content = message["content"]
            while AUDIO_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    seqlen = self._tokens_for_audio(audios[audio_index])
                else:
                    seqlen = 1
                content = content.replace(AUDIO_PLACEHOLDER, self.audio_token * seqlen, 1)
                audio_index += 1

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)
