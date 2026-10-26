import pytest
from simulinfer.config import Model, Settings, load
from simulinfer.app import request_options, session_settings


def test_request_can_select_either_configured_model_type():
    models={'target_only':Model(model='target'),'source_target':Model(model='source-target')}
    s=Settings(models=models,default_model_type='source_target')
    assert request_options(s)==('source_target',True,False)
    assert request_options(s,model_type='target_only',vad=False)==('target_only',False,False)
    with pytest.raises(ValueError,match='defaults.model_type'):Settings(models=models,default_model_type='unknown')


def test_firered_uses_bundled_pair_by_default():
    assert Settings().vad_mode=='firered'
    assert Settings().st_vad_mode=='firered_nonstream'
    assert Settings().vad_model_path==''



def test_existing_config_aliases_and_relative_paths(tmp_path):
    p=tmp_path/'config.toml'
    p.write_text('''[defaults]
model_type="source_target"
vad=true
tts=false
[models.target_only]
base_url="http://model:8800/v1"
name="target"
[models.source_target]
base_url="http://model:8801/v1"
name="source-target"
[inference]
wait_fallback_seconds=6.4
[vad]
enabled=true
stream_model_path="models/stream"
nonstream_model_path="models/nonstream"
zero_fill_silence=false
short_speech_guard=false
[tts]
enabled=false
backend="openai"
base_url="http://tts:9000/v1"
''')
    s=load(p)
    assert s.default_model_type=='source_target' and s.models['target_only'].model=='target'
    assert s.vad_mode=='firered' and s.st_vad_mode=='firered_nonstream'
    assert s.vad_model_path==str(tmp_path/'models/stream')
    assert s.st_vad_model_path==str(tmp_path/'models/nonstream')
    assert s.wait_fallback_seconds==6.4 and s.tts_backend=='none'
    assert s.post_speech_silence_mode=='passthrough'


def test_request_settings_do_not_mutate_global_policy():
    cfg=Settings(st_vad_mode='firered_nonstream',st_short_speech_guard=True,
                 post_speech_silence_mode='zero_fill')
    off=session_settings(cfg,False)
    assert off.vad_mode==off.st_vad_mode=='off'
    assert not off.st_short_speech_guard and off.post_speech_silence_mode=='passthrough'
    assert cfg.vad_mode=='firered' and cfg.st_short_speech_guard
    with pytest.raises(ValueError):request_options(cfg,model_type='source_target',mode='target_only')
    with pytest.raises(ValueError):request_options(cfg,vad='false')
    with pytest.raises(ValueError):request_options(cfg,tts=True)


@pytest.mark.parametrize('transport',['stream','ws'])
def test_request_model_type_vad_and_tts_end_to_end(transport):
    import io,json
    import numpy as np
    import soundfile as sf
    from fastapi.testclient import TestClient
    from simulinfer.app import build_app
    cfg=load(mock=True)
    with TestClient(build_app(cfg)) as client:
        public=client.get('/config').json()
        assert set(public['model_types'])=={'target_only','source_target'}
        if transport=='ws':
            with client.websocket_connect('/translate/ws') as ws:
                ws.send_json({'model_type':'source_target','vad':False,'tts':True,'input_format':'f32le','audio_transport':'base64'})
                ready=ws.receive_json();assert 'vad_enabled' not in ready
                ws.send_bytes(np.full(51200,.1,dtype='<f4').tobytes());ws.send_json({'eof':True})
                events=[ready]
                while events[-1]['event']!='done':events.append(ws.receive_json())
        else:
            buf=io.BytesIO();sf.write(buf,np.full(51200,.1),16000,format='WAV')
            response=client.post('/translate/stream',data={'model_type':'source_target','vad':'false','tts':'true'},files={'file':('x.wav',buf.getvalue(),'audio/wav')})
            assert response.status_code==200
            events=[json.loads(x[5:]) for x in response.text.splitlines() if x.startswith('data:')]
        assert events[-1]['event']=='done' and events[-1]['model_type']=='source_target'
        assert events[-1]['tts_enabled'] and 'vad_enabled' not in events[-1]
        assert any(e['event']=='audio' for e in events)
