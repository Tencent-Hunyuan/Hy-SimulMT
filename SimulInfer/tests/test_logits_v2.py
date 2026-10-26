"""V2 slot reuse, preemption and per-speculative-row output constraints."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('vllm.v1.worker.gpu.sample.logits_processor',reason='Requires upstream vLLM V2')
from simulinfer.logits_v2 import (SimulstNoRepeatNgramProcessor, STNoRepeatNgramProcessor,
                                     NoChineseTargetProcessor)
from vllm import SamplingParams


def setup(tmp_path):
    data={'decoder':{'type':'ByteLevel'},'model':{'vocab':{'a':1,'b':2,'c':3,'中':4}},
          'added_tokens':[{'id':99,'content':'<｜hy_EOT｜>','special':True},
                          {'id':98,'content':'<｜hy_end▁of▁sentence｜>','special':True},
                          {'id':97,'content':'<｜hy_place▁holder▁no▁791｜>','special':True}]}
    (tmp_path/'tokenizer.json').write_text(json.dumps(data))
    (tmp_path/'export_manifest.json').write_text(json.dumps({'source_target_format':{
        'target_token':'<｜hy_place▁holder▁no▁791｜>'}}))
    states=SimpleNamespace(device=torch.device('cpu'),max_num_reqs=4,vocab_size=100,
        prompt_len=SimpleNamespace(np=np.zeros(4,dtype=int)),
        total_len=SimpleNamespace(gpu=torch.zeros(4,dtype=torch.long)),
        all_token_ids=SimpleNamespace(gpu=torch.zeros(4,100,dtype=torch.long)))
    return SimpleNamespace(model_config=SimpleNamespace(model=str(tmp_path))),states


def history(s,slot,prompt,out):
    tokens=prompt+out;s.prompt_len.np[slot]=len(prompt)
    s.total_len.gpu[slot]=len(tokens);s.all_token_ids.gpu[slot,:len(tokens)]=torch.tensor(tokens)


def context(slots,positions,inputs):
    return SimpleNamespace(expanded_idx_mapping=torch.tensor(slots),idx_mapping_np=np.unique(slots),
        pos=torch.tensor(positions),input_ids=torch.tensor(inputs))


def test_reordered_slots_and_recycled_request_do_not_mix_ngrams(tmp_path):
    cfg,s=setup(tmp_path);p=SimulstNoRepeatNgramProcessor(cfg,s)
    params=SamplingParams(extra_args={'hy_simulst_no_repeat_ngram_size':3})
    p.add_request(0,params);p.add_request(1,params)
    history(s,0,[10],[1,2,3,1,2]);history(s,1,[20],[1,2,4,1,2])
    c=context([1,0],[5,5],[2,2]);logits=p.apply(torch.zeros(2,100),c)
    assert torch.isneginf(logits[0,4]) and torch.isfinite(logits[0,3])
    assert torch.isneginf(logits[1,3]) and torch.isfinite(logits[1,4])
    p.add_request(0,SamplingParams());history(s,0,[3,4,5],[])
    assert torch.isfinite(p.apply(torch.zeros(1,100),context([0],[2],[5]))).all()


def test_st_prefix_separate_and_boundaries_reset(tmp_path):
    cfg,s=setup(tmp_path);p=STNoRepeatNgramProcessor(cfg,s)
    ps=SamplingParams(extra_args={'hy_st_no_repeat_ngram_size':3,'hy_st_prefix_token_ids':[1,2,3]})
    p.add_request(2,ps);history(s,2,[10,11],[1,2])
    assert torch.isneginf(p.apply(torch.zeros(1,100),context([2],[3],[2]))[0,3])
    history(s,2,[10,11],[97,1,2])
    assert torch.isfinite(p.apply(torch.zeros(1,100),context([2],[4],[2]))).all()


def test_speculative_rows_use_only_their_visible_prefix(tmp_path):
    cfg,s=setup(tmp_path);p=SimulstNoRepeatNgramProcessor(cfg,s)
    p.add_request(0,SamplingParams(extra_args={'hy_simulst_no_repeat_ngram_size':3}))
    history(s,0,[10],[1,2,3,1])
    c=context([0,0],[4,5],[1,2]);logits=p.apply(torch.zeros(2,100),c)
    assert torch.isfinite(logits[0]).all()
    assert torch.isneginf(logits[1,3]) and torch.isfinite(logits[1,2])
    # Resuming after preemption carries generated IDs in all_token_ids; no
    # duplicate prefix or stale incremental state should change the result.
    p.add_request(0,SamplingParams(extra_args={'hy_simulst_no_repeat_ngram_size':3}))
    again=p.apply(torch.zeros(2,100),c)
    assert torch.equal(torch.isneginf(logits),torch.isneginf(again))


def test_target_mask_ignores_historical_prompt_and_handles_speculative_boundary(tmp_path):
    cfg,s=setup(tmp_path);p=NoChineseTargetProcessor(cfg,s)
    ps=SamplingParams(extra_args={'hy_target_language':'after_target'})
    p.add_request(0,ps);history(s,0,[97,10],[1])
    c=context([0,0],[2,3],[1,97]);logits=p.apply(torch.zeros(2,100),c)
    assert torch.isfinite(logits[0]).all()
    assert torch.isneginf(logits[1,4]) and torch.isfinite(logits[1,99])
    p.add_request(0,SamplingParams());assert torch.isfinite(p.apply(torch.zeros(2,100),c)).all()


def test_invalid_processor_params(tmp_path):
    cfg,s=setup(tmp_path);p=SimulstNoRepeatNgramProcessor(cfg,s)
    with pytest.raises(ValueError):p.add_request(0,SamplingParams(extra_args={'hy_simulst_no_repeat_ngram_size':-1}))


def test_source_chinese_allowed_target_chinese_blocked_slots_independent(tmp_path):
    cfg,s=setup(tmp_path);p=NoChineseTargetProcessor(cfg,s)
    p.add_request(0,SamplingParams(extra_args={'hy_target_language':'after_target'}))
    p.add_request(1,SamplingParams(extra_args={'hy_target_language':'target'}))
    p.add_request(2,SamplingParams())
    history(s,0,[97,4],[4])  # A prior prompt TGT must not constrain current source.
    history(s,1,[97,4],[])
    history(s,2,[97,4],[97])
    logits=p.apply(torch.zeros(3,100),context([2,0,1],[2,2,1],[97,4,4]))
    assert torch.isfinite(logits[0]).all()  # Untagged reverse direction.
    assert torch.isfinite(logits[1,4]) and torch.isfinite(logits[1,97])
    assert torch.isneginf(logits[2,4]) and torch.isneginf(logits[2,97])
    history(s,0,[97,4],[4,97])
    logits=p.apply(torch.zeros(1,100),context([0],[3],[97]))
    assert torch.isneginf(logits[0,4]) and torch.isfinite(logits[0,99])
    p.add_request(0,SamplingParams(extra_args={'hy_target_language':'after_target'}))
    history(s,0,[97,4],[])
    assert torch.isfinite(p.apply(torch.zeros(1,100),context([0],[1],[4]))[0,4])


def test_auto_ngram_plugins_leave_non_simulst_models_untouched(tmp_path):
    cfg,s=setup(tmp_path)
    cfg.model_config.hf_config=SimpleNamespace(architectures=['Qwen3TTSTalker'])
    (tmp_path/'tokenizer.json').unlink()
    for cls in (SimulstNoRepeatNgramProcessor,STNoRepeatNgramProcessor):
        p=cls(cfg,s)
        assert p.add_request(0,SamplingParams()) is False
        history(s,0,[10],[1,1,1])
        result=p.apply(torch.zeros(1,100),context([0],[3],[1]))
        assert torch.isfinite(result).all()
