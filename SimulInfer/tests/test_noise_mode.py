import base64
import io
import json
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from simulinfer.config import Settings as RuntimeSettings,Model,load
from simulinfer.app import session_settings
from simulinfer.noise import speech_audio
from simulinfer.translation import VLLMSession


def Settings(**kwargs):
    return RuntimeSettings(**{'vad_mode':'silero','st_vad_mode':'off',**kwargs})


def response(text=''):
    return httpx.Response(200,json={'choices':[{'message':{'content':text},
        'finish_reason':'stop','stop_reason':120008,'token_ids':[1]}],
        'usage':{'prompt_tokens':80,'completion_tokens':1}})


def audio_parts(body):
    return [sf.read(io.BytesIO(base64.b64decode(p['input_audio']['data'])),dtype='float32')[0]
            for m in body['messages'] if m['role']=='user'
            for p in m['content'] if p['type']=='input_audio']


class Detector:
    def __init__(self,masks):self.masks=iter(masks)
    def analyze(self,samples,start):
        mask=np.asarray(next(self.masks),dtype=np.bool_)
        assert len(samples)==len(mask)
        tail=len(mask)-np.flatnonzero(mask)[-1]-1 if mask.any() else len(mask)
        return {'is_silence':not mask.any(),'vad_trailing_silence_samples':tail,
                '_vad_speech_mask':mask,'_vad_raw_speech_mask':mask}


def config(**kwargs):
    return replace(Settings(),audio_mode='noise_resistant',vad_leading_pad_ms=0,
                   model_audio_format='float32',**{'st_short_speech_guard':False,'simulst_crop_speech':True,**kwargs})


def test_compaction_keeps_only_selected_samples_and_source_spans():
    x=np.arange(20,dtype=np.float32);mask=np.zeros(20,bool);mask[3:6]=True;mask[10:13]=True
    y,spans=speech_audio(x,mask,start_sample=100,leading_samples=2)
    np.testing.assert_array_equal(y,np.r_[x[1:6],x[8:13]])
    assert spans==[(101,106),(108,113)]
    with pytest.raises(ValueError):speech_audio(x,mask[:10],start_sample=0)


@pytest.mark.asyncio
async def test_translation_and_st_share_compact_audio_and_original_timeline():
    calls=[]
    def answer(request):
        body=json.loads(request.content);calls.append(body)
        return response('recovered' if body['max_tokens']==256 else '')
    x=np.arange(25600,dtype=np.float32)/25600
    mask=np.zeros(25600,bool);mask[1000:5000]=True;mask[10000:16000]=True
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        s=VLLMSession(Model(),config(st_wait_chunks=1),'target_only','zh-en',http=client)
        s.activity=Detector([mask]);e=[e async for e in s.push(x)][0]
    assert len(calls)==2
    np.testing.assert_array_equal(audio_parts(calls[0])[-1],x[mask])
    np.testing.assert_array_equal(audio_parts(calls[1])[-1],x[mask])
    assert e['end_ms']==1600 and e['st_audio_ms']==1600
    assert e['model_audio_ms']==e['st_model_audio_ms']==625
    assert e['vad_removed_ms']==e['st_vad_removed_ms']==975
    assert e['retained_source_spans_ms']==e['st_retained_source_spans_ms']==[[62.5,312.5],[625,1000]]


@pytest.mark.asyncio
async def test_silent_tail_triggers_st_without_uploading_noise():
    calls=[]
    def answer(request):
        body=json.loads(request.content);calls.append(body)
        return response('tail' if body['max_tokens']==256 else '')
    mask=np.zeros(25600,bool);mask[4000:20000]=True
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        s=VLLMSession(Model(),config(st_wait_chunks=0,trailing_silence_seconds=1.6),'target_only','zh-en',http=client)
        s.activity=Detector([mask,np.zeros(25600,bool)])
        events=[e async for e in s.push(np.r_[np.full(25600,.2),np.full(25600,.05)].astype(np.float32))]
    assert len(calls)==2 and not events[-1]['model_invoked']
    assert events[-1]['recovery_trigger']=='trailing_silence'
    assert events[-1]['st_audio_ms']==3200 and events[-1]['st_model_audio_ms']==1000
    assert np.all(audio_parts(calls[-1])[0]==np.float32(.2))
    assert [(e['start_ms'],e['end_ms']) for e in events]==[(0,1600),(1600,3200)]


@pytest.mark.asyncio
async def test_entire_noise_and_partial_eof_never_invoke_model():
    def fail(_):raise AssertionError('noise reached model')
    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        s=VLLMSession(Model(),config(),'target_only','zh-en',http=client)
        s.activity=Detector([np.zeros(25600,bool),np.zeros(3333,bool)])
        es=[e async for e in s.push(np.full(28933,.05,np.float32))]+[e async for e in s.finish()]
    assert es[-1]['duration_ms']==28933//16 and es[-1]['num_chunks']==2
    assert all(not e['model_invoked'] and e['model_audio_ms']==0 for e in es[:-1])


@pytest.mark.asyncio
async def test_st_prefix_reuses_cropped_previous_emit_and_current_audio():
    calls=[]
    def answer(request):
        b=json.loads(request.content);calls.append(b)
        return response('first ' if len(calls)==1 else 'second' if b['max_tokens']==256 else '')
    m1=np.zeros(25600,bool);m1[100:8100]=True
    m2=np.zeros(25600,bool);m2[16000:24000]=True
    x=np.r_[np.full(25600,.2),np.full(25600,.3)].astype(np.float32)
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        s=VLLMSession(Model(),config(st_wait_chunks=1,st_context=True,st_no_repeat_ngram_size=0),'target_only','zh-en',http=client)
        s.activity=Detector([m1,m2]);es=[e async for e in s.push(x)]
    assert len(calls)==3 and calls[-1]['continue_final_message']
    np.testing.assert_array_equal(audio_parts(calls[-1])[0],np.r_[x[:25600][m1],x[25600:][m2]])
    assert es[-1]['st_model_audio_ms']==1000
    assert es[-1]['st_retained_source_spans_ms']==[[6.25,506.25],[2600,3100]]


@pytest.mark.asyncio
async def test_speech_partial_eof_does_not_restore_padding():
    calls=[]
    def answer(request):calls.append(json.loads(request.content));return response('ok')
    mask=np.zeros(1111,bool);mask[101:1001]=True
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        s=VLLMSession(Model(),config(),'target_only','zh-en',http=client)
        s.activity=Detector([mask]);es=[e async for e in s.push(np.arange(1111,dtype=np.float32)/1111)]
        es += [e async for e in s.finish()]
    assert len(audio_parts(calls[0])[0])==900
    assert es[0]['final_padding_ms']==0 and es[0]['model_audio_ms']==56.25
    assert es[-1]['duration_ms']==1111//16


@pytest.mark.asyncio
async def test_default_no_fixed_time_rollover():
    cfg=replace(Settings(),vad_mode='energy',st_wait_chunks=0)
    assert cfg.window_seconds==0
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:response('x'))) as client:
        s=VLLMSession(Model(),cfg,'target_only','zh-en',http=client)
        es=[e async for e in s.push(np.full(640000,.1,np.float32))]
    assert s.run==0 and not any(e.get('window_cut_trigger') for e in es)


def test_noise_request_can_disable_vad_without_changing_server_settings():
    cfg=config();off=session_settings(cfg,False)
    assert off.audio_mode=='standard' and off.vad_mode=='off'
    assert cfg.audio_mode=='noise_resistant'


@pytest.mark.parametrize('kwargs',[{'vad_mode':'off','default_vad':False},
    {'vad_mode':'energy'},{'post_speech_silence_mode':'zero_fill'},{'vad_preroll_enabled':True}])
def test_noise_mode_rejects_conflicting_settings(kwargs):
    with pytest.raises(ValueError):config(**kwargs)


def test_noise_preset_and_readme_keep_internal_options_in_config():
    root=Path(__file__).parents[1]
    cfg=load(root/'configs/noise_resistant.toml')
    assert cfg.audio_mode=='noise_resistant' and cfg.window_seconds==0
    assert cfg.st_guard_enabled and not cfg.token_budget_enabled
    for name in ('README.md','README.zh-CN.md'):
        text=(root/name).read_text()
        assert all(word not in text for word in ('token_budget_enabled','st_short_speech_guard','max_tokens','token-budget','64/256'))


@pytest.mark.asyncio
async def test_context_rollover_keeps_current_cropped_audio():
    from simulinfer.budget import PromptBudget
    from simulinfer.translation import Reply
    class API:
        def __init__(self):self.audios=[]
        async def budget(self,messages,**kwargs):
            return PromptBudget(2000 if len(messages)>1 else 100,2048,1)
        async def generate(self,messages,**kwargs):
            self.audios.append(audio_parts({'messages':messages})[-1])
            return Reply('','ok','ok',[1],120008,100,1,'stop')
    cfg=config(token_budget_enabled=True,context_reserve_tokens=256)
    s=VLLMSession(Model(),cfg,'target_only','zh-en');await s.api.close();s.api=API()
    m=np.zeros(25600,bool);m[1000:9000]=True
    s.activity=Detector([m,m])
    es=[e async for e in s.push(np.r_[np.full(25600,.1),np.full(25600,.2)].astype(np.float32))]
    assert es[-1]['window_cut_trigger']=='token_budget_preflight'
    assert len(s.api.audios[-1])==8000
    assert np.all(s.api.audios[-1]==np.float32(.2))


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['target_only','source_target'])
async def test_noise_default_preserves_voiced_chunk_and_only_crops_st(mode):
    from simulinfer.translation import TGT
    calls=[]
    def answer(request):
        body=json.loads(request.content);calls.append(body)
        return response(('中文源文'+TGT if mode=='source_target' else '')+'recovered' if body['max_tokens']==256 else '')
    cfg=Settings(models={mode:Model()},audio_mode='noise_resistant',model_audio_format='float32',
                 vad_leading_pad_ms=160,vad_pad_ms=240,st_wait_chunks=1,st_short_speech_guard=False)
    assert not cfg.simulst_crop_enabled
    x=np.arange(25600,dtype=np.float32)/25600
    mask=np.zeros(25600,bool);mask[8000:12000]=True
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        s=VLLMSession(Model(),cfg,mode,'zh-en',http=client)
        s.activity=Detector([mask]);e=[e async for e in s.push(x)][0]
    assert len(calls)==2
    np.testing.assert_array_equal(audio_parts(calls[0])[-1],x)
    np.testing.assert_array_equal(audio_parts(calls[1])[-1],x[5440:15840])
    assert e['model_audio_ms']==1600 and e['vad_removed_ms']==0
    assert e['st_model_audio_ms']==650 and e['st_vad_removed_ms']==950
    assert e['end_ms']==1600 and not e['silence_zero_filled']
    assert e['text']=='recovered'


def test_ws_noise_default_keeps_stream_chunks_intact():
    cfg=RuntimeSettings()
    ws=session_settings(cfg,True,model_type='target_only',transport='ws')
    assert ws.audio_mode=='noise_resistant'
    assert not ws.simulst_crop_enabled and ws.st_crop_enabled
    assert ws.st_guard_enabled and ws.st_vad_mode=='firered_nonstream'
