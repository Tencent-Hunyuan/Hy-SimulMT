"""Target-only inference shares the same audio, ST and TTS policy controls."""
import base64
import io
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pytest
import soundfile as sf

from simulinfer.app import session_settings
from simulinfer.config import Settings, Model, load
from simulinfer.tts import Speech
from simulinfer.translation import VLLMSession, TGT


def test_target_preset_retains_shared_mechanisms():
    root=Path(__file__).parents[1]
    target=load(root/'configs/target_only.toml')
    interleave=load(root/'configs/source_target.toml')
    assert list(target.models)==['target_only'] and target.no_chinese_target
    assert replace(target,models=interleave.models,no_chinese_target=interleave.no_chinese_target)==interleave
    assert target.first_chunk_ms==target.chunk_ms==1600
    assert target.no_repeat_ngram_size==target.st_no_repeat_ngram_size==3
    assert target.tts_speed_zh_en==1.2
    ws=session_settings(target,True,model_type='target_only',transport='ws')
    assert ws.audio_mode=='noise_resistant' and ws.st_guard_enabled
    assert ws.st_crop_enabled and not ws.simulst_crop_enabled
    assert ws.st_wait_chunks==4 and ws.trailing_silence_seconds==2.4


@pytest.mark.parametrize('mode',['target_only','source_target'])
@pytest.mark.parametrize('direction,expected',[('zh-en',1.2),('en-zh',1.1)])
def test_tts_speed_depends_on_direction_not_output_model(mode,direction,expected):
    settings=Settings(models={mode:Model()},tts_backend='openai',tts_speed=1.1)
    speech=Speech(settings,None,direction)
    assert speech.speed==expected and speech.info()['speed']==expected
    assert speech._speech_body('speech')['speed']==expected


@pytest.mark.parametrize('speed',[0,float('nan'),float('inf'),True,'1.2',2.1])
def test_directional_speed_validation(speed):
    with pytest.raises(ValueError,match='tts_speed_zh_en'):
        Settings(tts_speed_zh_en=speed)


def test_directional_speed_override_and_general_speed_config(tmp_path):
    p=tmp_path/'config.toml';p.write_text('[model]\nbase_url="http://model:8800/v1"\n[tts]\ntts_speed=1.1\ntts_speed_zh_en=1.3\n')
    cfg=load(p)
    assert Speech(cfg,None,'zh-en').speed==1.3
    assert Speech(cfg,None,'en-zh').speed==1.1


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['target_only','source_target'])
@pytest.mark.parametrize('direction,expected',[('zh-en',1.2),('en-zh',1.1)])
async def test_streaming_tempo_matches_direction_and_history_stays_original(mode,direction,expected):
    rate=24000
    pcm=(np.sin(2*np.pi*330*np.arange(rate*2)/rate)*6000).astype('<i2').tobytes()
    from simulinfer.tts_history import wav_data_url
    reference=wav_data_url(pcm,rate)
    calls=[];frames=[]
    def respond(req):
        if req.method=='GET':return httpx.Response(200,json={'source_voice_history':'source-voice-history-v1'})
        calls.append(json.loads(req.content));return httpx.Response(200,content=pcm)
    async def emit(e):
        if e['event']=='audio':frames.append(base64.b64decode(e['data']))
    settings=Settings(models={mode:Model()},tts_backend='vllm_omni',tts_task_type='Base',
                      tts_clone_source=True,tts_speed=1.1,tts_min_chars=1)
    speech=Speech(settings,emit,direction)
    client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with patch('simulinfer.tts.httpx.AsyncClient',return_value=client):await speech.start()
    try:
        speech.feed_source(np.full(38400,.1,np.float32))
        await speech.text('First sentence.');await speech.text('Second sentence.');await speech.finish()
        assert len(calls)==2
        assert all(r['stream'] and r['stream_format']=='audio' and 'speed' not in r for r in calls)
        assert calls[1]['ref_audio']==reference and calls[1]['ref_text']=='First sentence.'
        assert abs(len(b''.join(frames))/2/rate-4/expected)<.12
        assert speech.failed_segments==0
    finally:await speech.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['target_only','source_target'])
@pytest.mark.parametrize('chunk',[800,1600])
async def test_model_output_modes_share_actual_chunk_and_ngram_request_fields(mode,chunk):
    calls=[]
    def respond(req):
        body=json.loads(req.content);calls.append(body)
        return httpx.Response(200,json={'choices':[{'message':{'content':('源文'+TGT if mode=='source_target' else '')+'translation'},'finish_reason':'stop','stop_reason':120008}],
                                      'usage':{'prompt_tokens':80,'completion_tokens':4}})
    cfg=Settings(models={mode:Model()},chunk_ms=chunk,no_chinese_target=True,
                 vad_mode='off',st_vad_mode='off',default_vad=False,st_crop_speech=False)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        session=VLLMSession(Model(),cfg,mode,'zh-en',http=http)
        events=[e async for e in session.push(np.full(51200,.1,np.float32))]
        assert [e['end_ms'] for e in events]==([1600,2400,3200] if chunk==800 else [1600,3200])
        audios=[]
        for body in calls:
            data=body['messages'][-2 if mode=='source_target' else -1]['content'][-1]['input_audio']['data']
            samples,_=sf.read(io.BytesIO(base64.b64decode(data)),dtype='float32');audios.append(len(samples))
            assert body['vllm_xargs']['hy_simulst_no_repeat_ngram_size']==3
            assert body['vllm_xargs']['hy_target_language']==('target' if mode=='target_only' else 'after_target')
        assert audios==([25600,12800,12800] if chunk==800 else [25600,25600])
        await session.close()


def test_deployment_example_preserves_inference_controls(tmp_path):
    from argparse import Namespace
    from simulinfer.deploy import load_options, write_config, model_command
    source=Path(__file__).parents[1]/'configs/deploy/target_only.toml'
    local=tmp_path/'deploy.toml'
    local.write_text(source.read_text().replace('chunk_ms = 1600\nno_repeat', 'chunk_ms = 800\nno_repeat'))
    opts,tts=load_options(Namespace(config=str(local)))
    config=tmp_path/'generated.toml';write_config(config,opts,'http://model:8800/v1',tts)
    settings=load(config)
    assert settings.chunk_ms==800 and settings.first_chunk_ms==1600
    assert settings.no_chinese_target and settings.st_crop_enabled
    assert not settings.simulst_crop_enabled
    assert settings.no_repeat_ngram_size==settings.st_no_repeat_ngram_size==3
    cmd=model_command(opts,'python',Path('/operator/model'))
    assert cmd[-2:]==['--logits-processors','simulinfer.logits_plugins:NoChineseTargetProcessor']
    opts['inference']['no_chinese_target']=False
    assert '--logits-processors' not in model_command(opts,'python',Path('/operator/model'))
