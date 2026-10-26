import json
from pathlib import Path

import httpx
import pytest

from simulinfer.config import Settings, Model, load
from simulinfer.translation import ChatAPI


def test_shipped_inference_configs_enable_both_ngram_filters():
    root=Path(__file__).parents[1]
    for path in (root/'config.example.toml',root/'simulinfer/default_config.toml',
                 root/'configs/noise_resistant.toml',root/'configs/vllm.toml'):
        s=load(path)
        assert s.no_repeat_ngram_size==s.st_no_repeat_ngram_size==3
        assert s.model_logits_processors


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled',[True,False])
async def test_preflight_rejects_backend_that_ignores_ngram_flag(enabled):
    bodies=[]
    def answer(req):
        d=json.loads(req.content)
        if req.url.path.endswith('/tokenize'):
            return httpx.Response(200,json={'tokens':[1 if d['prompt']=='a' else 2], 'count':1,'max_model_len':8192})
        bodies.append(d)
        ids=[1,1,1,2] if enabled else [1,1,1,1]
        return httpx.Response(200,json={'choices':[{'token_ids':ids}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as http:
        api=ChatAPI(Model(),Settings(),'source_target','zh-en',http=http)
        if enabled:
            result=await api.check_repetition_processors()
            assert result=={'hy_simulst_no_repeat_ngram_size':3,'hy_st_no_repeat_ngram_size':3}
            assert len(bodies)==2
        else:
            with pytest.raises(RuntimeError,match='missing or inactive'):
                await api.check_repetition_processors()
