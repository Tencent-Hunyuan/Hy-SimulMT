# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hy-SimulMT audio feature extraction and processor integration."""

from typing import Optional, Union

import numpy as np
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TensorType, is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class HySimulMTAudioFeatureExtractor(SequenceFeatureExtractor):
    """Convert mono waveforms to the 128-bin log-mel input used by Hy-SimulMT."""

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        padding_value: float = 0.0,
        dither: float = 0.0,
        return_attention_mask: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.dither = dither
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def _np_extract_fbank_features(self, waveform_batch: np.ndarray, device: str) -> np.ndarray:
        if device != "cpu":
            raise ValueError("NumPy feature extraction only supports device='cpu'.")

        log_spec_batch = []
        for waveform in waveform_batch:
            log_spec = spectrogram(
                waveform,
                window_function(self.n_fft, "hann"),
                frame_length=self.n_fft,
                hop_length=self.hop_length,
                power=2.0,
                dither=self.dither,
                mel_filters=self.mel_filters,
                log_mel="log10",
            )
            log_spec = log_spec[:, :-1]
            log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
            log_spec_batch.append((log_spec + 4.0) / 4.0)

        return np.asarray(log_spec_batch)

    def _torch_extract_fbank_features(self, waveform: np.ndarray, device: str = "cpu") -> np.ndarray:
        waveform = torch.from_numpy(waveform).to(device, torch.float32)
        window = torch.hann_window(self.n_fft, device=device)
        if self.dither != 0.0:
            waveform += self.dither * torch.randn_like(waveform)

        stft = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        log_spec = torch.clamp(mel_filters.T @ magnitudes, min=1e-10).log10()
        if waveform.dim() == 2:
            max_value = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_value - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)

        log_spec = (log_spec + 4.0) / 4.0
        if device != "cpu":
            log_spec = log_spec.detach().cpu()
        return log_spec.numpy()

    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: list[np.ndarray],
        attention_mask: list[np.ndarray] | None,
        padding_value: float = 0.0,
    ) -> list[np.ndarray]:
        if attention_mask is None:
            return [(item - item.mean()) / np.sqrt(item.var() + 1e-7) for item in input_values]

        attention_mask = np.asarray(attention_mask, np.int32)
        normalized = []
        for vector, length in zip(input_values, attention_mask.sum(-1)):
            item = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
            if length < item.shape[0]:
                item[length:] = padding_value
            normalized.append(item)
        return normalized

    def __call__(
        self,
        raw_speech: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        do_normalize: Optional[bool] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ) -> BatchFeature:
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected {self.sampling_rate} Hz audio, received {sampling_rate} Hz."
            )
        if sampling_rate is None:
            logger.warning(
                "Pass sampling_rate explicitly to avoid silently processing audio at the wrong rate."
            )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and raw_speech.ndim > 1
        if is_batched_numpy and raw_speech.ndim > 2:
            raise ValueError("Only mono-channel audio is supported.")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple))
            and len(raw_speech) > 0
            and isinstance(raw_speech[0], (np.ndarray, tuple, list))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif raw_speech.dtype == np.float64:
            raw_speech = raw_speech.astype(np.float32)

        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        padded_inputs = self.pad(
            BatchFeature({"input_features": raw_speech}),
            padding=padding,
            max_length=max_length or self.n_samples,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=bool(return_attention_mask or do_normalize),
        )

        if do_normalize:
            padded_inputs["input_features"] = np.stack(
                self.zero_mean_unit_var_norm(
                    padded_inputs["input_features"],
                    attention_mask=padded_inputs["attention_mask"],
                    padding_value=self.padding_value,
                ),
                axis=0,
            )

        input_features = padded_inputs["input_features"].transpose(2, 0, 1)
        extractor = (
            self._torch_extract_fbank_features
            if is_torch_available()
            else self._np_extract_fbank_features
        )
        input_features = extractor(input_features[0], device)
        padded_inputs["input_features"] = input_features

        if return_attention_mask:
            attention_mask = padded_inputs["attention_mask"][:, :: self.hop_length]
            if padded_inputs["attention_mask"].shape[1] % self.hop_length != 0:
                attention_mask = attention_mask[:, :-1]
            padded_inputs["attention_mask"] = attention_mask

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)
        return padded_inputs


class HySimulMTProcessor(ProcessorMixin):
    """Tokenizer plus Hy-SimulMT audio feature extractor."""

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "HySimulMTAudioFeatureExtractor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None) -> None:
        super().__init__(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        self.audio_sampling_rate = feature_extractor.sampling_rate

    @property
    def model_input_names(self) -> list[str]:
        return list(dict.fromkeys(self.feature_extractor.model_input_names + self.tokenizer.model_input_names))


def register_audio_processing(config_class) -> None:
    """Register local processor classes before LlamaFactory calls AutoProcessor."""
    from transformers import AutoFeatureExtractor, AutoProcessor

    AutoFeatureExtractor.register(
        config_class,
        HySimulMTAudioFeatureExtractor,
        exist_ok=True,
    )
    AutoProcessor.register(config_class, HySimulMTProcessor, exist_ok=True)


__all__ = [
    "HySimulMTAudioFeatureExtractor",
    "HySimulMTProcessor",
    "register_audio_processing",
]
