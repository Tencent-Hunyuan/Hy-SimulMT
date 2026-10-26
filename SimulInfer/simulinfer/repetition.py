"""Separate ST/SimulST token n-gram filters with request-local state."""
from __future__ import annotations

import json
from pathlib import Path

from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

SIZE_KEY = 'hy_st_no_repeat_ngram_size'
PREFIX_KEY = 'hy_st_prefix_token_ids'
BOUNDARY_KEY = 'hy_st_boundary_token_ids'
SIMULST_SIZE_KEY = 'hy_simulst_no_repeat_ngram_size'
SIMULST_BOUNDARY_KEY = 'hy_simulst_boundary_token_ids'


class _Ngrams:
    def __init__(self, size, prefix, boundaries):
        self.size = size
        self.prefix = tuple(prefix)
        self.boundaries = set(boundaries)
        self.reset()

    def reset(self):
        self.seen = {}
        self.tail = []
        self.processed = 0
        for token in self.prefix:
            self.append(token)

    def append(self, token):
        if token in self.boundaries:
            self.seen.clear()
            self.tail.clear()
            return
        if len(self.tail) == self.size-1:
            self.seen.setdefault(tuple(self.tail), set()).add(token)
        self.tail.append(token)
        if len(self.tail) >= self.size:
            del self.tail[0]

    def __call__(self, output_ids, logits):
        if len(output_ids) < self.processed:
            self.reset()
        for token in output_ids[self.processed:]:
            self.append(token)
        self.processed = len(output_ids)
        if len(self.tail) == self.size-1:
            blocked = self.seen.get(tuple(self.tail), set()) - self.boundaries
            if blocked:
                logits[list(blocked)] = -float('inf')
        return logits


class STNoRepeatNgramProcessor(AdapterLogitsProcessor):
    """vLLM's adapter owns row moves/removal and per-request live output IDs.

    Only explicit ST tags enable this. Prompt instructions/audio placeholders
    are excluded; exact assistant prefix and new output IDs are included.
    """
    size_key = SIZE_KEY
    prefix_key = PREFIX_KEY
    boundary_key = BOUNDARY_KEY

    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)
        architectures=getattr(getattr(vllm_config.model_config,'hf_config',None),'architectures',None)
        self.enabled=architectures is None or 'HYV3SpeechForCausalLM' in architectures
        if not self.enabled:
            self.model_boundaries=set()
            return
        data = json.loads((Path(vllm_config.model_config.model)/'tokenizer.json').read_text())
        wanted = {'<｜hy_EOT｜>', '<｜hy_end▁of▁sentence｜>',
                  '<｜hy_place▁holder▁no▁790｜>', '<｜hy_place▁holder▁no▁791｜>'}
        self.model_boundaries = {x['id'] for x in data.get('added_tokens', []) if x['content'] in wanted}

    @classmethod
    def validate_params(cls, params):
        extra = params.extra_args or {}
        size = extra.get(cls.size_key, 0)
        if type(size) is not int or not 0 <= size <= 32:
            raise ValueError('invalid no-repeat ngram size')
        if size:
            for key in (cls.prefix_key, cls.boundary_key):
                if key is None:
                    continue
                values = extra.get(key, [])
                if not isinstance(values, list) or any(type(x) is not int or x < 0 for x in values):
                    raise ValueError(f'{key} must contain non-negative token IDs')

    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, params):
        if not self.enabled:return None
        self.validate_params(params)
        extra = params.extra_args or {}
        size = extra.get(self.size_key, 0)
        if not size:
            return None
        boundaries = list(self.model_boundaries) + list(params.stop_token_ids or [])
        prefix = extra.get(self.prefix_key, []) if self.prefix_key is not None else []
        return _Ngrams(size, prefix, boundaries)


class SimulstNoRepeatNgramProcessor(STNoRepeatNgramProcessor):
    """Current chunk only; each vLLM turn starts with a fresh n-gram index.

    Neither earlier assistant text nor a prefix supplied through ST tags is
    consulted. Source/target markers separate fields; EOT/EOS stay allowed.
    """
    size_key = SIMULST_SIZE_KEY
    prefix_key = None
    boundary_key = SIMULST_BOUNDARY_KEY
