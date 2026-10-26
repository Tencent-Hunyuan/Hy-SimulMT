"""Request-local target-only Chinese token mask for vLLM V1.

Source transcription is unconstrained. This processor is active only for
zh->en requests explicitly tagged by the serving engine, and only AFTER the
current assistant's target separator. Historical target separators do not
activate the mask for a new source field.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from vllm.v1.sample.logits_processor.interface import (
    LogitsProcessor, MoveDirectionality,
)

PARAM_KEY = 'hy_target_language'
HAN = re.compile('[\u2e80-\u2fdf\u3007\u31c0-\u31ef\u3400-\u4dbf'
                 '\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003347f]')


def byte_decoder():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(map(chr, chars), values))


def blocked_token_ids(tokenizer_json: dict) -> list[int]:
    if tokenizer_json.get('decoder', {}).get('type') != 'ByteLevel':
        raise ValueError('target language mask requires the verified ByteLevel tokenizer')
    decode = byte_decoder()
    blocked = set()
    for token, token_id in tokenizer_json['model']['vocab'].items():
        try:
            text = bytes(decode[c] for c in token).decode('utf-8', errors='strict')
        except (KeyError, UnicodeDecodeError):
            # Never let individually incomplete byte tokens reconstruct a Han
            # character across decoding steps. Complete UTF-8 Latin accents,
            # ASCII text, punctuation and numbers remain available.
            blocked.add(token_id)
            continue
        if HAN.search(text):
            blocked.add(token_id)
    for item in tokenizer_json.get('added_tokens', []):
        if not item.get('special') and HAN.search(item['content']):
            blocked.add(item['id'])
    return sorted(blocked)


class NoChineseTargetProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        root = Path(vllm_config.model_config.model)
        data = json.loads((root / 'tokenizer.json').read_text())
        manifest = json.loads((root / 'export_manifest.json').read_text())
        fmt = manifest.get('source_target_format', {})
        special = {x['content']: x['id'] for x in data.get('added_tokens', [])}
        self.target_id = special.get(fmt.get('target_token'))
        blocked = set(blocked_token_ids(data))
        # Once inside TARGET, prevent reopening SOURCE or injecting another
        # field boundary. EOT/EOS remain allowed to end generation normally.
        for field in ('source_token', 'target_token'):
            if fmt.get(field) in special:
                blocked.add(special[fmt[field]])
        self.blocked = torch.tensor(sorted(blocked), dtype=torch.long, device=device)
        self.requests = {}

    @classmethod
    def validate_params(cls, params):
        value = (params.extra_args or {}).get(PARAM_KEY)
        if value not in (None, 'after_target', 'target'):
            raise ValueError('invalid target language mask mode')

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.requests.pop(index, None)
        for index, params, _, output_ids in batch_update.added:
            self.requests.pop(index, None)
            mode = (params.extra_args or {}).get(PARAM_KEY)
            if mode:
                # Keep vLLM's live output list, never a copy or global history.
                self.requests[index] = {'output': output_ids, 'seen': 0,
                                        'active': mode == 'target'}
        for src, dst, direction in batch_update.moved:
            a = self.requests.pop(src, None)
            b = self.requests.pop(dst, None)
            if a is not None:
                self.requests[dst] = a
            if direction == MoveDirectionality.SWAP and b is not None:
                self.requests[src] = b

    def apply(self, logits):
        for row, state in self.requests.items():
            if not state['active']:
                output = state['output']
                if self.target_id in output[state['seen']:]:
                    state['active'] = True
                state['seen'] = len(output)
            if state['active']:
                logits[row].index_fill_(0, self.blocked, -float('inf'))
        return logits
