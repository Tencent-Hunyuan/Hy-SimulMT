#!/usr/bin/env python3
"""Verify the exact LlamaFactory token/label protocol without loading the 30B LLM."""

from __future__ import annotations

import argparse
import importlib.machinery
import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


def _install_torchaudio_stub_if_missing() -> None:
    try:
        import torchaudio  # noqa: F401
    except ModuleNotFoundError:
        # This check never decodes audio.  The stub only lets LlamaFactory import
        # its generic multimodal module in minimal CI environments.
        module = ModuleType("torchaudio")
        module.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
        module.functional = SimpleNamespace()
        sys.modules["torchaudio"] = module


def _contains_supervised_span(input_ids: list[int], labels: list[int], span: list[int]) -> bool:
    return any(
        input_ids[index : index + len(span)] == span
        and labels[index : index + len(span)] == span
        for index in range(len(input_ids) - len(span) + 1)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    _install_torchaudio_stub_if_missing()
    # The check only exercises tokenizer/template/data code.  Allow it to run
    # in a minimal CI image even if training-only packages are older there.
    os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

    from transformers import AutoTokenizer

    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    from s2s_train.common.register import _register_plugin_and_template
    from s2s_train.common.source_target_mask import patch_source_token_loss
    from s2s_train.variants import hy_v3_speech_merged as variant

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    _register_plugin_and_template(
        model_type=variant.MODEL_TYPE,
        template_name=variant.TEMPLATE_NAME,
        audio_token=variant.AUDIO_TOKEN,
        num_speech_tokens=0,
        template_kwargs=variant.build_template_kwargs(),
        speech_tokens_for=lambda num_samples: 7,
    )
    patch_source_token_loss(variant.SOURCE_TOKEN_ID)
    data_args = DataArguments(template=variant.TEMPLATE_NAME, cutoff_len=512, packing=False)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    processor = SupervisedDatasetProcessor(
        template=template,
        tokenizer=tokenizer,
        processor=SimpleNamespace(feature_extractor=object()),
        data_args=data_args,
    )

    source = "This is the source transcript."
    target = "这是目标译文。"
    user = (
        "把这段语音翻译成中文，要全部输出，不要分块。"
        "<｜hy_place▁holder▁no▁6｜><audio><｜hy_place▁holder▁no▁8｜>"
    )
    assistant = variant.SOURCE_TOKEN + source + variant.TARGET_TOKEN + target
    input_ids, labels = processor._encode_data_example(
        prompt=[{"role": "user", "content": user}],
        response=[{"role": "assistant", "content": assistant}],
        system=None,
        tools=None,
        images=[],
        videos=[],
        audios=[np.zeros(16_000, dtype=np.float32)],
    )

    source_positions = [index for index, token in enumerate(input_ids) if token == variant.SOURCE_TOKEN_ID]
    target_positions = [index for index, token in enumerate(input_ids) if token == variant.TARGET_TOKEN_ID]
    assert len(source_positions) == 1
    assert labels[source_positions[0]] == -100
    assert len(target_positions) == 1
    assert labels[target_positions[0]] == variant.TARGET_TOKEN_ID
    assert _contains_supervised_span(input_ids, labels, tokenizer.encode(source, add_special_tokens=False))
    assert _contains_supervised_span(input_ids, labels, tokenizer.encode(target, add_special_tokens=False))
    assert input_ids.count(tokenizer.eos_token_id) == 1
    assert input_ids[-1] == tokenizer.eos_token_id
    assert labels[-1] == tokenizer.eos_token_id

    print(
        {
            "sequence_length": len(input_ids),
            "audio_placeholders": input_ids.count(variant.AUDIO_TOKEN_ID),
            "source_marker_masked": True,
            "source_text_supervised": True,
            "target_marker_supervised": True,
            "target_text_supervised": True,
            "single_supervised_eos": True,
            "passed": True,
        }
    )


if __name__ == "__main__":
    main()
