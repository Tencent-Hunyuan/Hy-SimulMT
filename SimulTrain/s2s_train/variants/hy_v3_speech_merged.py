"""Variant for the merged A3B speech checkpoint using the original protocol."""

from ..common.configuration_speech_llm import SpeechLLMConfig


MODEL_TYPE = "speech_hy_v3_merged"
TEMPLATE_NAME = "hy_v3_merged_audio"

AUDIO_TOKEN = "<｜hy_place▁holder▁no▁5｜>"
AUDIO_TOKEN_ID = 120023
SOURCE_TOKEN = "<｜hy_place▁holder▁no▁790｜>"
SOURCE_TOKEN_ID = 120808
TARGET_TOKEN = "<｜hy_place▁holder▁no▁791｜>"
TARGET_TOKEN_ID = 120809


class SpeechHyV3MergedConfig(SpeechLLMConfig):
    model_type = MODEL_TYPE


CONFIG_CLASS = SpeechHyV3MergedConfig


def build_template_kwargs() -> dict:
    from llamafactory.data.formatter import EmptyFormatter, StringFormatter

    return {
        "format_user": StringFormatter(slots=["<｜hy_User｜>{{content}}"]),
        "format_assistant": StringFormatter(
            # efficient_eos=True appends the single conversation-final EOS in
            # SupervisedDatasetProcessor.  Do not also put EOS here or every
            # sample ends with two consecutive EOS tokens.
            slots=["<｜hy_Assistant｜><think></think>{{content}}"]
        ),
        "format_system": StringFormatter(slots=["{{content}}"]),
        "format_prefix": EmptyFormatter(slots=[{"bos_token"}]),
        "stop_words": ["<eos:6124c78e>"],
        "efficient_eos": True,
    }


def verify_tokens(tokenizer) -> None:
    expected = {
        AUDIO_TOKEN: AUDIO_TOKEN_ID,
        SOURCE_TOKEN: SOURCE_TOKEN_ID,
        TARGET_TOKEN: TARGET_TOKEN_ID,
    }
    for token, expected_id in expected.items():
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids != [expected_id]:
            raise RuntimeError(f"special token mismatch: {token!r} -> {ids}, expected [{expected_id}]")


def register(num_speech_tokens: int | None = None, config=None) -> None:
    from ..common.hy_v3_patches import apply_hy_v3_patches
    from ..common.register import register_variant
    from ..common.source_target_mask import patch_source_token_loss

    apply_hy_v3_patches()
    speech_tokens_for = None
    if config is not None:
        if config.adapter_type == "qformer":
            num_speech_tokens = config.num_speech_tokens
        else:
            num_speech_tokens = 0
            speech_tokens_for = config.speech_tokens_for_samples
    if num_speech_tokens is None:
        raise ValueError("register() requires config= or num_speech_tokens=")

    register_variant(
        model_type=MODEL_TYPE,
        config_class=CONFIG_CLASS,
        template_name=TEMPLATE_NAME,
        audio_token=AUDIO_TOKEN,
        num_speech_tokens=num_speech_tokens,
        template_kwargs=build_template_kwargs(),
        speech_tokens_for=speech_tokens_for,
    )
    patch_source_token_loss(SOURCE_TOKEN_ID)


__all__ = [
    "AUDIO_TOKEN",
    "AUDIO_TOKEN_ID",
    "CONFIG_CLASS",
    "MODEL_TYPE",
    "SOURCE_TOKEN",
    "SOURCE_TOKEN_ID",
    "TARGET_TOKEN",
    "TARGET_TOKEN_ID",
    "TEMPLATE_NAME",
    "register",
    "verify_tokens",
]
