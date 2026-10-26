import io
import json
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from simulinfer.app import build_app,session_settings
from simulinfer.config import Settings,Model,load


def wav():
    b=io.BytesIO();sf.write(b,np.full(59200,.1),16000,format='WAV');return b.getvalue()


@pytest.mark.parametrize('cadence,expected',[('short_chunk',800),('standard',1600)])
def test_named_default_and_legacy_numeric_config(tmp_path,cadence,expected):
    p=tmp_path/'config.toml'
    p.write_text(f'[models.target_only]\nmodel="demo"\n[defaults]\nresponse_mode="{cadence}"\n')
    s=load(p);assert s.chunk_ms==expected and s.first_chunk_ms==1600
    p.write_text(f'[models.target_only]\nmodel="demo"\n[inference]\nchunk_ms={expected}\n')
    assert load(p).response_mode==cadence
    p.write_text(p.read_text()+f'[defaults]\nresponse_mode="{cadence}"\n')
    assert load(p).response_mode==cadence
    p.write_text(p.read_text().replace(f'chunk_ms={expected}',f'chunk_ms={2400-expected}'))
    with pytest.raises(ValueError,match='不一致'):load(p)


@pytest.mark.parametrize('transport',['stream','ws'])
@pytest.mark.parametrize('model_type',['target_only','source_target'])
@pytest.mark.parametrize('cadence,ends',[('short_chunk',[1600,2400,3200,3700]),('standard',[1600,3200,3700])])
def test_request_selection_drives_actual_chunk_timeline(transport,model_type,cadence,ends):
    cfg=load(mock=True)
    with TestClient(build_app(cfg)) as c:
        options={'model_type':model_type,'response_mode':cadence,'vad':False,'tts':False}
        if transport=='stream':
            r=c.post('/translate/stream',data={k:str(v).lower() if isinstance(v,bool) else v for k,v in options.items()},files={'file':('x.wav',wav(),'audio/wav')})
            assert r.status_code==200
            es=[json.loads(l[5:]) for l in r.text.splitlines() if l.startswith('data:')]
        else:
            with c.websocket_connect('/translate/ws') as ws:
                ws.send_json({**options,'input_format':'f32le'})
                es=[ws.receive_json()]
                samples=np.full(59200,.1,dtype='<f4')
                for offset in range(0,len(samples),1600):ws.send_bytes(samples[offset:offset+1600].tobytes())
                ws.send_json({'eof':True})
                while es[-1]['event']!='done':es.append(ws.receive_json())
        assert es[0]['response_mode']==es[-1]['response_mode']==cadence
        assert es[0]['first_chunk_ms']==1600
        assert [e['end_ms'] for e in es if e['event']=='chunk']==ends
        assert es[-1]['duration_ms']==3700
        assert c.get('/config').json()['defaults']['response_mode']=='standard'
    assert cfg.chunk_ms==1600


def test_two_live_sessions_keep_separate_cadences():
    with TestClient(build_app(load(mock=True))) as c:
        with c.websocket_connect('/translate/ws') as fast,c.websocket_connect('/translate/ws') as slow:
            for ws,mode in [(fast,'short_chunk'),(slow,'standard')]:
                ws.send_json({'response_mode':mode,'vad':False,'input_format':'f32le'})
                assert ws.receive_json()['response_mode']==mode
                ws.send_bytes(np.full(59200,.1,dtype='<f4').tobytes());ws.send_json({'eof':True})
            for ws,expected in [(fast,4),(slow,3)]:
                es=[]
                while not es or es[-1]['event']!='done':es.append(ws.receive_json())
                assert es[-1]['num_chunks']==expected


@pytest.mark.parametrize('value',['quick','slow','fast',800,False,{},[]])
def test_invalid_cadence_rejected(value):
    with pytest.raises(ValueError,match='response_mode'):session_settings(Settings(),True,value)
    with TestClient(build_app(load(mock=True))) as c:
        with c.websocket_connect('/translate/ws') as ws:
            ws.send_json({'response_mode':value});assert ws.receive_json()['event']=='error'


def test_request_override_preserves_noise_processing_and_config_default():
    cfg=Settings(audio_mode='noise_resistant',chunk_ms=1600)
    fast=session_settings(cfg,True,'short_chunk','target_only')
    assert fast.chunk_ms==800 and fast.st_guard_enabled and fast.st_crop_enabled
    assert fast.audio_mode=='noise_resistant' and cfg.chunk_ms==1600
    assert session_settings(fast,True,'standard').chunk_ms==1600


def test_upstream_service_cadence_not_silently_ignored():
    cfg=Settings(models={'target_only':Model(backend='service')})
    with pytest.raises(ValueError,match='上游service'):session_settings(cfg,True,'short_chunk','target_only')
    with TestClient(build_app(cfg)) as c:
        assert c.get('/config').json()['response_mode_available']['target_only'] is False


def test_omitted_request_uses_fast_config_default():
    cfg=replace(load(mock=True),chunk_ms=800)
    with TestClient(build_app(cfg)) as c:
        assert c.get('/config').json()['defaults']['response_mode']=='short_chunk'
        r=c.post('/translate/stream',data={'vad':'false'},files={'file':('x.wav',wav(),'audio/wav')})
        es=[json.loads(l[5:]) for l in r.text.splitlines() if l.startswith('data:')]
        assert es[-1]['response_mode']=='short_chunk' and es[-1]['num_chunks']==4




@pytest.mark.parametrize('removed',['slow','fast'])
def test_removed_cadence_names_rejected_in_config_and_file_requests(tmp_path,removed):
    p=tmp_path/'config.toml'
    p.write_text('[model]\nbase_url="http://127.0.0.1:8800/v1"\n[defaults]\nresponse_mode="'+removed+'"\n')
    with pytest.raises(ValueError,match='response_mode'):load(p)
    with TestClient(build_app(load(mock=True))) as c:
        response=c.post('/translate/stream',data={'response_mode':removed},files={'file':('x.wav',wav(),'audio/wav')})
        assert response.status_code==400
