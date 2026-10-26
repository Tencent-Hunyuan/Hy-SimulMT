import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from simulinfer.app import build_app
from simulinfer.config import Settings,Model,load
from simulinfer.model_discovery import discover,declared_format,checkpoint_format,SOURCE_TOKEN,TARGET_TOKEN


def manifest(enabled):
    return {'source_target_format':{'enabled':enabled,'source_token':SOURCE_TOKEN,'target_token':TARGET_TOKEN}}


def root_model(tmp_path,enabled,name='checkpoint'):
    path=tmp_path/name;path.mkdir();(path/'export_manifest.json').write_text(json.dumps(manifest(enabled)))
    return path


@pytest.mark.parametrize('enabled,mode',[(False,'target_only'),(True,'source_target')])
@pytest.mark.asyncio
async def test_local_export_metadata_selects_mode_without_inference(tmp_path,enabled,mode):
    path=root_model(tmp_path,enabled)
    calls=[]
    def transport(req):
        calls.append(req.url.path)
        assert req.url.path=='/v1/models'
        return httpx.Response(200,json={'data':[{'id':'custom-alias','root':str(path)}]})
    cfg=Settings(models={},model_endpoints=(Model(model=''),))
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        detected=await discover(cfg,http=client)
    assert list(detected.models)==[mode] and detected.models[mode].model=='custom-alias'
    assert detected.models[mode].chat_protocol=='standard' and not detected.model_endpoints
    assert cfg.models=={} and len(cfg.model_endpoints)==1
    assert calls==['/v1/models']


@pytest.mark.asyncio
async def test_two_services_discover_distinct_formats(tmp_path):
    a=root_model(tmp_path,False,'one');b=root_model(tmp_path,True,'two')
    def handler(req):
        return httpx.Response(200,json={'data':[{'id':req.url.host,'root':str(a if req.url.host=='target' else b)}]})
    cfg=Settings(models={},model_endpoints=(Model(base_url='http://target/v1',model=''),Model(base_url='http://source/v1',model='')))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        detected=await discover(cfg,http=client)
    assert set(detected.models)=={'target_only','source_target'}
    assert detected.models['source_target'].model=='source'


@pytest.mark.asyncio
async def test_huggingface_metadata_uses_hf_auth_not_model_api_auth(monkeypatch):
    monkeypatch.setenv('HF_TOKEN','hf-test-value');monkeypatch.setenv('VLLM_API_KEY','model-test-value')
    calls=[]
    def handler(req):
        calls.append(str(req.url))
        if req.url.host=='model-host':
            assert req.headers['Authorization']=='Bearer model-test-value'
            return httpx.Response(200,json={'data':[{'id':'alias','root':'example/speech-model'}]})
        assert str(req.url)=='https://huggingface.co/example/speech-model/resolve/main/export_manifest.json'
        assert req.headers['Authorization']=='Bearer hf-test-value'
        return httpx.Response(200,json=manifest(True))
    cfg=Settings(models={},model_endpoints=(Model(base_url='http://model-host/v1',model=''),))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        detected=await discover(cfg,http=client)
    assert list(detected.models)==['source_target'] and len(calls)==2


@pytest.mark.parametrize('name,mode',[('A3b_tgt_only_1019_bf16','target_only'),('A3b_src_tgt_1019_bf16','source_target'),
    ('Hy-SimulMT-30B-A3B','target_only'),('Hy-SimulMT-30B-A3B-Interleave','source_target')])
@pytest.mark.asyncio
async def test_provided_checkpoint_names_work_for_unshared_server_paths(name,mode):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(404))) as client:
        actual,source=await checkpoint_format('/unshared/checkpoints/'+name,client)
    assert actual==mode and source=='provided_model'


@pytest.mark.asyncio
async def test_unknown_alias_is_not_guessed(tmp_path):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(404))) as client:
        with pytest.raises(ValueError,match='无法识别'):
            await checkpoint_format('/unknown/arbitrary_src_tgt_name',client)
        # Explicit metadata has priority even if a directory's name suggests otherwise.
        path=root_model(tmp_path,False,'A3b_src_tgt_1019_bf16')
        assert (await checkpoint_format(str(path),client))[0]=='target_only'


@pytest.mark.asyncio
async def test_missing_or_bad_metadata_rejected():
    for data in ({},{'source_target_format':{'enabled':'false'}},{'source_target_format':{'enabled':True,'source_token':'wrong','target_token':'wrong'}}):
        with pytest.raises(ValueError):declared_format(data)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(403))) as client:
        with pytest.raises(ValueError,match='403'):await checkpoint_format('example/private-model',client)


@pytest.mark.asyncio
async def test_explicit_checkpoint_reference_and_named_model(tmp_path):
    path=root_model(tmp_path,True)
    def handler(_):return httpx.Response(200,json={'data':[{'id':'alias','root':'/server/opaque'},{'id':'other','root':'/server/other'}]})
    cfg=Settings(models={},model_endpoints=(Model(model='alias',checkpoint=str(path)),))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        detected=await discover(cfg,http=client)
        assert list(detected.models)==['source_target']
        with pytest.raises(ValueError,match='唯一'):
            await discover(replace(cfg,model_endpoints=(Model(model='',checkpoint=str(path)),)),http=client)
        with pytest.raises(ValueError,match='不匹配'):
            await discover(replace(cfg,model_endpoints=(Model(model='absent'),)),http=client)


@pytest.mark.asyncio
async def test_duplicate_output_types_are_not_silently_overwritten():
    def handler(_):return httpx.Response(200,json={'data':[{'id':'a','root':'/remote/A3b_tgt_only_1019_bf16'},{'id':'b','root':'/remote/A3b_tgt_only_1019_bf16'}]})
    cfg=Settings(models={},model_endpoints=(Model(model=''),))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError,match='多个相同'):await discover(cfg,http=client)


def test_simple_config_and_endpoint_list(tmp_path):
    path=root_model(tmp_path,False)
    cfg=tmp_path/'config.toml';cfg.write_text('[model]\nbase_url="http://model:8800/v1"\ncheckpoint="./checkpoint"\n')
    s=load(cfg);assert not s.models and s.model_endpoints[0].checkpoint==str(path)
    assert s.model_endpoints[0].model==''
    cfg.write_text('[[models]]\nbase_url="http://one/v1"\n[[models]]\nbase_url="http://two/v1"\n')
    assert len(load(cfg).model_endpoints)==2
    cfg.write_text('[model]\nbase_url="http://one/v1"\n[models.source_target]\nmodel="two"\n')
    with pytest.raises(ValueError,match='不能'):load(cfg)


def test_web_app_lists_only_discovered_model(monkeypatch):
    import simulinfer.model_discovery as module
    async def fake(settings):
        return replace(settings,models={'source_target':Model(model='published-source-model')},model_endpoints=())
    monkeypatch.setattr(module,'discover',fake)
    cfg=Settings(models={},model_endpoints=(Model(model=''),))
    with TestClient(build_app(cfg,preflight_result={'models':[]})) as c:
        public=c.get('/config').json()
        assert public['model_types']==['source_target']
        assert public['defaults']['model_type']=='source_target'
        assert public['model_labels']['source_target']=='published-source-model'
        assert c.post('/translate/stream',data={'model_type':'target_only'},files={'file':('x.wav',b'x')}).status_code==400
