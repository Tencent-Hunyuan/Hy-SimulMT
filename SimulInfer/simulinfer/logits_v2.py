"""Request-slot-aware output constraints for upstream vLLM Model Runner V2.

The lightweight HTTP SimulInfer never imports this module. It is loaded only in
the vLLM environment. Histories are read from the current request slot each
step, so preemption, slot recycling and speculative rows cannot reuse another
request's n-gram or target-language state.
"""
from __future__ import annotations

import json
from pathlib import Path

from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

BOUNDARY_TEXTS = {'<｜hy_EOT｜>', '<｜hy_end▁of▁sentence｜>',
                  '<｜hy_place▁holder▁no▁790｜>', '<｜hy_place▁holder▁no▁791｜>'}


def _visible_histories(req_states, ctx, enabled):
    """Yield each active row's output prefix, including its speculative prefix.

    This eager reference implementation synchronizes small history views to the
    CPU. It favors correctness over a fused GPU kernel; measure its overhead
    before enabling speculative decoding/high concurrency in production.
    """
    if not enabled or not any(int(slot) in enabled for slot in ctx.idx_mapping_np):
        return
    mapping = ctx.expanded_idx_mapping.detach().cpu().tolist()
    positions = ctx.pos.detach().cpu().tolist()
    input_ids = ctx.input_ids.detach().cpu().tolist()
    cache = {}
    for slot in set(mapping).intersection(enabled):
        prompt = int(req_states.prompt_len.np[slot])
        total = int(req_states.total_len.gpu[slot].item())
        limit = max(pos+1 for pos,s in zip(positions,mapping) if s==slot)
        committed = min(total, limit)
        tokens = req_states.all_token_ids.gpu[slot,prompt:committed].detach().cpu().tolist()
        cache[slot] = (prompt, committed, tokens)
    draft = {}
    for slot,pos,token in zip(mapping,positions,input_ids):
        draft.setdefault(slot,{})[pos] = token
    for row, (slot,pos) in enumerate(zip(mapping,positions)):
        if slot not in cache:
            continue
        prompt, committed, tokens = cache[slot]
        end = max(prompt,pos+1)
        prefix = tokens[:max(0,min(end,committed)-prompt)]
        if end > committed:
            for position in range(max(committed,prompt),end):
                if position not in draft[slot]:
                    raise RuntimeError('V2 logits context is missing speculative output tokens')
                prefix.append(draft[slot][position])
        yield row, slot, prefix


def _ngram_bans(tokens, size, boundaries):
    seen = {};tail=[]
    for token in tokens:
        if token in boundaries:
            seen.clear();tail.clear();continue
        if len(tail)==size-1:
            seen.setdefault(tuple(tail),set()).add(token)
        tail.append(token)
        if len(tail)>=size:
            del tail[0]
    return seen.get(tuple(tail),set())-boundaries if len(tail)==size-1 else set()


class STNoRepeatNgramProcessor(LogitsProcessor):
    size_key = 'hy_st_no_repeat_ngram_size'
    prefix_key = 'hy_st_prefix_token_ids'

    def __init__(self, vllm_config, req_states):
        self.req_states=req_states;self.requests={}
        architectures=getattr(getattr(vllm_config.model_config,'hf_config',None),'architectures',None)
        self.enabled=architectures is None or 'HYV3SpeechForCausalLM' in architectures
        self.boundaries=set()
        if self.enabled:
            data=json.loads((Path(vllm_config.model_config.model)/'tokenizer.json').read_text())
            self.boundaries={x['id'] for x in data.get('added_tokens',[]) if x['content'] in BOUNDARY_TEXTS}

    @classmethod
    def validate_params(cls, params):
        extra=params.extra_args or {};size=extra.get(cls.size_key,0)
        if type(size) is not int or not 0<=size<=32:
            raise ValueError('invalid no-repeat ngram size')
        prefix=extra.get(cls.prefix_key,[]) if cls.prefix_key else []
        if not isinstance(prefix,list) or any(type(x) is not int or x<0 for x in prefix):
            raise ValueError('ngram prefix must be non-negative token IDs')

    def add_request(self, req_idx, sampling_params):
        self.requests.pop(req_idx,None);self.validate_params(sampling_params)
        if not self.enabled:return False
        extra=sampling_params.extra_args or {};size=extra.get(self.size_key,0)
        if not size:return False
        prefix=tuple(extra.get(self.prefix_key,[])) if self.prefix_key else ()
        self.requests[req_idx]=(size,prefix,self.boundaries | set(sampling_params.stop_token_ids or []))
        return True

    def apply(self, logits, ctx):
        for row,slot,tokens in _visible_histories(self.req_states,ctx,self.requests):
            size,prefix,boundaries=self.requests[slot]
            banned=_ngram_bans(list(prefix)+tokens,size,boundaries)
            if banned:logits[row,list(banned)]=-float('inf')
        return logits


class SimulstNoRepeatNgramProcessor(STNoRepeatNgramProcessor):
    size_key='hy_simulst_no_repeat_ngram_size'
    prefix_key=None


class NoChineseTargetProcessor(LogitsProcessor):
    def __init__(self,vllm_config,req_states):
        import torch
        # Import the existing, tested tokenizer mask only in the model process.
        from .target_language import blocked_token_ids
        self.req_states=req_states;self.requests={}
        root=Path(vllm_config.model_config.model)
        data=json.loads((root/'tokenizer.json').read_text())
        manifest=root/'export_manifest.json'
        fmt=(json.loads(manifest.read_text()).get('source_target_format',{}) if manifest.is_file() else {})
        special={x['content']:x['id'] for x in data.get('added_tokens',[])}
        self.target_id=special.get(fmt.get('target_token'),special.get('<｜hy_place▁holder▁no▁791｜>'))
        blocked=set(blocked_token_ids(data))
        for field, fallback in (('source_token','<｜hy_place▁holder▁no▁790｜>'),
                                ('target_token','<｜hy_place▁holder▁no▁791｜>')):
            token_id=special.get(fmt.get(field),special.get(fallback))
            if token_id is not None:blocked.add(token_id)
        if self.target_id is None:
            raise ValueError('Target-language mask requires the model TGT separator')
        self.blocked=torch.tensor(sorted(blocked),dtype=torch.long,device=req_states.device)

    @classmethod
    def validate_params(cls,params):
        if (params.extra_args or {}).get('hy_target_language') not in (None,'target','after_target'):
            raise ValueError('invalid target language mask mode')

    def add_request(self,req_idx,sampling_params):
        self.requests.pop(req_idx,None);self.validate_params(sampling_params)
        mode=(sampling_params.extra_args or {}).get('hy_target_language')
        if mode:self.requests[req_idx]=mode
        return bool(mode)

    def apply(self,logits,ctx):
        for row,slot,tokens in _visible_histories(self.req_states,ctx,self.requests):
            if self.requests[slot]=='target' or self.target_id in tokens:
                logits[row].index_fill_(0,self.blocked,-float('inf'))
        return logits
