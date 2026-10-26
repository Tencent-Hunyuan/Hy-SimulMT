"""Count the exact shipped chat template plus simulst_mm audio expansion.

Only text is sent to /tokenize; audio decoding/prefill is not repeated here.
Generation usage is checked against this count to detect incompatible models.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, StrictUndefined


class ContextBudgetError(ValueError):
    pass


@dataclass(frozen=True)
class PromptBudget:
    prompt_tokens: int
    max_model_len: int
    audio_items: int

    def fits(self, output_tokens):
        return self.prompt_tokens + output_tokens <= self.max_model_len


def speech_tokens(num_samples):
    # simulst_mm's 16kHz / hop160 / 100-mel-frame convolution grid / FSK2.
    # Keep this small pure arithmetic independent of torch/simulst_mm.
    if not 0 < num_samples <= 480000:
        raise ValueError('单个音频item需为1～480000采样（最多30秒）')
    frames = (num_samples + 159) // 160
    tail = (frames % 100 + 7) // 8
    return (frames // 100 * 13 + tail + 1) // 2


@lru_cache(maxsize=1)
def template():
    return Environment(undefined=StrictUndefined).from_string(
        (Path(__file__).parent/'templates/simulst.jinja').read_text())


def render(messages, mode, prefix=None):
    kwargs = {'messages': messages, 'output_mode': mode, 'add_generation_prompt': True}
    if prefix is not None:
        kwargs['harness_prefix'] = prefix
    extra = items = 0
    for message in messages:
        if message['role'] != 'user':
            continue
        for part in message['content']:
            if part['type'] == 'input_audio':
                count = getattr(part, 'num_samples', None)
                if count is None:
                    raise ValueError('音频计数缺失：请使用audio.part构造模型输入')
                extra += speech_tokens(count) - 1
                items += 1
    return template().render(**kwargs), extra, items
