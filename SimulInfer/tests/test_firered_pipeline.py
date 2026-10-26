import hashlib
import json
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest

from simulinfer.app import session_settings
from simulinfer.config import Settings, Model, load
from simulinfer.noise import detect_st_blocks, padded_st_audio, short_speech_policy, speech_stats
from simulinfer.translation import VLLMSession, SRC, TGT
from simulinfer.vad import FireRedNonstreamActivity, create_activity, resolve_firered_model_path, resolve_firered_stream_path
from tests.test_noise_mode import Detector, audio_parts, response


class Nonstream:
    def __init__(self,segments):self.segments=segments;self.inputs=[]
    def analyze(self,audio):
        self.inputs.append(audio.copy())
        return {'st_vad_segments':self.segments,'st_vad_accepted':bool(self.segments),
                'st_vad_threshold':.4,'st_vad_probability_max':.9,'st_vad_seconds':0}


def test_defaults_bundle_both_models_and_preserve_explicit_alternative():
    cfg=Settings()
    assert cfg.vad_mode=='firered' and cfg.st_vad_mode=='firered_nonstream'
    assert cfg.vad_threshold==.4 and cfg.vad_pad_ms==240
    for path in (resolve_firered_model_path(''),resolve_firered_stream_path('')):
        root=Path(path);manifest=json.loads((root/'manifest.json').read_text())
        for name,entry in manifest['files'].items():
            data=(root/name).read_bytes()
            assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
        assert (root/'LICENSE').is_file()
    assert resolve_firered_model_path('')!=resolve_firered_stream_path('')
    off=session_settings(cfg,False)
    assert off.vad_mode==off.st_vad_mode=='off' and not off.st_crop_enabled
    preset=load(Path(__file__).parents[1]/'configs/noise_resistant.toml')
    assert preset.vad_mode=='firered' and preset.st_vad_mode=='firered_nonstream'
    assert preset.st_guard_enabled and preset.window_seconds==0


@pytest.mark.parametrize('audio_mode',['standard','noise_resistant'])
@pytest.mark.parametrize('output_mode',['target_only','source_target'])
@pytest.mark.parametrize('crop',[False,True])
@pytest.mark.asyncio
async def test_st_redetects_original_instead_of_stream_cropped_audio(audio_mode,output_mode,crop):
    calls=[]
    def handle(req):
        body=json.loads(req.content);calls.append(body)
        return response(('源文'+TGT if output_mode=='source_target' else '')+'translation' if body['max_tokens']==256 else '')
    cfg=Settings(models={output_mode:Model()},audio_mode=audio_mode,simulst_crop_speech=crop,model_audio_format='float32',
                 st_wait_chunks=1,st_short_speech_guard=False)
    x=np.arange(25600,dtype=np.float32)/25600
    streaming_mask=np.zeros(25600,bool);streaming_mask[4000:12000]=True
    full=Nonstream([[.8,1.0]])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        s=VLLMSession(Model(),cfg,output_mode,'zh-en',http=client)
        s.activity=Detector([streaming_mask]);s.st_activity=full
        e=[e async for e in s.push(x)][0]
    np.testing.assert_array_equal(full.inputs[0],x)
    before=3840 if audio_mode=='standard' else 2560
    expected=x[12800-before:16000+3840]
    np.testing.assert_array_equal(audio_parts(calls[-1])[0],expected)
    stream=audio_parts(calls[0])[0]
    np.testing.assert_array_equal(stream,x if audio_mode=='standard' or not crop else x[1440:12000])
    assert e['st_vad_mode']=='firered_nonstream'
    assert e['st_model_audio_ms']==len(expected)/16 and e['end_ms']==1600
    assert not e['silence_zero_filled']
    assert e['text']=='translation'
    if output_mode=='source_target':assert e['source_text']=='源文'


@pytest.mark.asyncio
async def test_no_speech_in_original_blocks_prevents_st_generation():
    calls=[]
    def handle(req):calls.append(req);return response('')
    cfg=Settings(audio_mode='noise_resistant',st_wait_chunks=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        s=VLLMSession(Model(),cfg,'target_only','zh-en',http=client)
        s.activity=Detector([np.ones(25600,bool)]);s.st_activity=Nonstream([])
        e=[e async for e in s.push(np.full(25600,.1,np.float32))][0]
    assert len(calls)==1 and e['st_vad_rejected'] and not e.get('st_attempted')


@pytest.mark.asyncio
async def test_source_gaps_are_not_removed_before_detection_or_padding():
    def block(start,value):
        a=np.full(16000,value,np.float32)
        return (start,a,16000,False,[],{'original':a,'speech_spans':[]})
    detector=Nonstream([[.2,.5]])
    blocks,stats=await detect_st_blocks([block(0,.1),block(32000,.2)],detector)
    assert len(detector.inputs)==2
    assert stats['st_vad_segments']==[[.2,.5],[2.2,2.5]]
    audio,kept=padded_st_audio(blocks,3840)
    assert kept==[(0,11840),(32000,43840)]
    assert len(audio)==23680


@pytest.mark.parametrize('speech,longest,direction,expected',[
    (300,300,'zh-en',4),(400,400,'zh-en',6),(400,400,'en-zh',5),
    (600,500,'zh-en',9),(600,500,'en-zh',8),
    (900,800,'zh-en',12),(900,800,'en-zh',10),
    (1400,1000,'zh-en',None),(2600,800,'zh-en',None),
])
def test_guard_uses_direction_and_continuity(speech,longest,direction,expected):
    stats={'st_vad_speech_ms':speech,'st_vad_longest_speech_ms':longest,'st_vad_segments':[[0,speech/1000]]}
    policy=short_speech_policy(stats,direction)
    assert policy['st_short_speech_token_limit']==expected
    assert policy['st_has_reliable_continuous_speech']==(longest>=1000)
    assert short_speech_policy(stats,direction,True)['st_short_speech_token_limit'] is None
    assert short_speech_policy(stats,direction,False,'time_window')['st_short_speech_token_limit'] is None


@pytest.mark.parametrize('mode',['target_only','source_target'])
@pytest.mark.parametrize('context',[False,True])
@pytest.mark.asyncio
async def test_guard_counts_new_target_only_using_nonstream_current_speech(mode,context):
    counted=[];calls=[];source='very long source '*30;target='one two three four five'
    def handle(req):
        body=json.loads(req.content)
        if req.url.path.endswith('/tokenize'):
            counted.append(body['prompt']);ids=list(range(len(body['prompt'].split())))
            return httpx.Response(200,json={'count':len(ids),'tokens':ids,'max_model_len':8192})
        calls.append(body)
        if len(calls)==1 and context:return response((source+TGT if mode=='source_target' else '')+'old')
        if body['max_tokens']!=256:return response('')
        if context and mode=='source_target' and TGT in body['stop']:
            d=json.loads(response(source).content);d['choices'][0]['stop_reason']=TGT
            return httpx.Response(200,json=d)
        return response((source+TGT if mode=='source_target' and not context else '')+target)
    cfg=Settings(models={mode:Model()},audio_mode='noise_resistant',st_context=context,st_wait_chunks=1,vad_full_chunk_speech_ms=0)
    mask=np.zeros(25600,bool);mask[:3200]=True
    masks=([np.ones(25600,bool)] if context else [])+[mask]
    segments=[[0,1.6],[1.6,1.8]] if context else [[0,.2]]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        s=VLLMSession(Model(),cfg,mode,'zh-en',http=client)
        s.activity=Detector(masks);s.st_activity=Nonstream(segments)
        es=[e async for e in s.push(np.full(25600*len(masks),.1,np.float32))]
    assert counted[-1]==target
    assert all(t != target for t in counted[:-1])
    assert es[-1]['st_short_speech_token_limit']==4 and es[-1]['st_output_rejected']
    assert es[-1]['text']=='' and s.pending


@pytest.mark.asyncio
async def test_rejected_short_speech_waits_for_new_voice_but_retries_at_eof():
    calls=[]
    def handle(req):
        body=json.loads(req.content)
        if req.url.path.endswith('/tokenize'):
            return httpx.Response(200,json={'count':9,'tokens':list(range(9)),'max_model_len':8192})
        calls.append(body);return response('one two three four five six seven eight nine' if body['max_tokens']==256 else '')
    cfg=Settings(audio_mode='noise_resistant',st_wait_chunks=1,trailing_silence_seconds=1.6,vad_full_chunk_speech_ms=0)
    m=np.zeros(25600,bool);m[:3200]=True
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        s=VLLMSession(Model(),cfg,'target_only','zh-en',http=client)
        s.activity=Detector([m,np.zeros(25600,bool)]);s.st_activity=Nonstream([[0,.2]])
        es=[e async for e in s.push(np.full(51200,.1,np.float32))]
        assert es[0]['st_output_rejected'] and es[1]['st_output_rejected_audio_unchanged']
        assert sum(b['max_tokens']==256 for b in calls)==1
        es += [e async for e in s.finish()]
        assert sum(b['max_tokens']==256 for b in calls)==2
        assert es[-1]['unresolved_audio_ms']==1600


def test_real_firered_pair_loads_without_network_and_handles_short_audio():
    pytest.importorskip('fireredvad')
    v=create_activity(Settings())
    assert v.__class__.__name__=='FireRedStreamActivity'
    assert v.analyze(np.zeros(25600,np.float32),0)['is_silence']
    assert v.analyze(np.zeros(12800,np.float32),25600)['vad_new_frames']==80
    st=FireRedNonstreamActivity()
    for count in (1,399,400,559,560,25600):
        assert not st.analyze(np.zeros(count,np.float32))['st_vad_accepted']
