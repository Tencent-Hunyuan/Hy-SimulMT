import json
from types import SimpleNamespace

import pytest


def processor(tmp_path, st=False):
    torch=pytest.importorskip('torch')
    pytest.importorskip('vllm')
    from simulinfer.repetition import SimulstNoRepeatNgramProcessor, STNoRepeatNgramProcessor
    (tmp_path/'tokenizer.json').write_text(json.dumps({'added_tokens':[
        {'content':'<｜hy_EOT｜>','id':99}, {'content':'<｜hy_end▁of▁sentence｜>','id':98},
        {'content':'<｜hy_place▁holder▁no▁791｜>','id':97}]}))
    cls=STNoRepeatNgramProcessor if st else SimulstNoRepeatNgramProcessor
    return cls(SimpleNamespace(model_config=SimpleNamespace(model=str(tmp_path))),torch.device('cpu'),False)


def test_normal_st_and_untagged_rows_are_isolated(tmp_path):
    normal=processor(tmp_path);st=processor(tmp_path,True)
    import torch
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor.interface import BatchUpdate,MoveDirectionality
    a=[1,2,3,1,2];b=[1,2]
    pn=SamplingParams(extra_args={'hy_simulst_no_repeat_ngram_size':3})
    ps=SamplingParams(extra_args={'hy_st_no_repeat_ngram_size':3,'hy_st_prefix_token_ids':[1,2,3]})
    change=BatchUpdate(3,[],[(0,pn,[],a),(1,ps,[],b),(2,SamplingParams(),[],a)],[])
    for p in (normal,st):p.update_state(change)
    result=st.apply(normal.apply(torch.zeros(3,100)))
    assert torch.isneginf(result[0,3]) and torch.isneginf(result[1,3])
    assert torch.isfinite(result[2]).all() and torch.isfinite(result[:,[97,98,99]]).all()
    move=BatchUpdate(3,[],[],[(0,1,MoveDirectionality.SWAP)])
    for p in (normal,st):p.update_state(move)
    assert set(normal.req_info)=={1} and set(st.req_info)=={0}
    normal.update_state(BatchUpdate(1,[1],[(0,pn,[1,2,3],[1,2])],[]))
    assert torch.isfinite(normal.apply(torch.zeros(1,100))[0,3])


def test_field_boundaries_reset_ngrams_but_eos_stays_available(tmp_path):
    p=processor(tmp_path)
    import torch
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor.interface import BatchUpdate
    out=[1,2,3,97,1,2]
    params=SamplingParams(extra_args={'hy_simulst_no_repeat_ngram_size':3})
    p.update_state(BatchUpdate(1,[],[(0,params,[],out)],[]))
    assert torch.isfinite(p.apply(torch.zeros(1,100))[0,3])
    out.extend([3,1,2])
    result=p.apply(torch.zeros(1,100))
    assert torch.isneginf(result[0,3]) and torch.isfinite(result[0,99])


@pytest.mark.parametrize('size',[1,2,3,5])
def test_matches_reference_without_cross_turn_memory(tmp_path,size):
    p=processor(tmp_path)
    import torch
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor.interface import BatchUpdate
    out=[]
    params=SamplingParams(extra_args={'hy_simulst_no_repeat_ngram_size':size})
    p.update_state(BatchUpdate(1,[],[(0,params,[],out)],[]))
    for token in [1,2,3,1,2,3,1,2,4,1,2]:
        context=tuple(out[-(size-1):]) if size>1 else ()
        expected={out[i+size-1] for i in range(len(out)-size+1) if tuple(out[i:i+size-1])==context}
        found=set(torch.where(torch.isneginf(p.apply(torch.zeros(1,100))[0]))[0].tolist())
        assert found==expected
        out.append(token)
