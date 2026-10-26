import json
from dataclasses import replace

import httpx
import numpy as np
import pytest

from simulinfer.config import Settings as RuntimeSettings,Model
from simulinfer.noise import padded_st_audio,speech_stats
from simulinfer.translation import VLLMSession,SRC,TGT
from tests.test_noise_mode import Detector,audio_parts,response


def Settings(**kwargs):
    return RuntimeSettings(**{'vad_mode':'silero','st_vad_mode':'off',**kwargs})


def block(start,x,spans):
    return (start,x,len(x),not spans,None,{'original':x,'speech_spans':spans})


def test_padding_merges_overlap_across_chunks_and_keeps_exact_samples():
    x=np.arange(48000,dtype=np.float32)
    blocks=[block(0,x[:25600],[(8000,9000),(12800,14000),(25000,25600)]),
            block(25600,x[25600:],[(25600,27000),(47000,48000)])]
    actual,spans=padded_st_audio(blocks,3840)
    assert spans==[(4160,17840),(21160,30840),(43160,48000)]
    expected=np.concatenate([x[a:b] for a,b in spans])
    np.testing.assert_array_equal(actual,expected)
    assert len(np.unique(actual))==len(actual)
    # A boundary through one continuous speech segment is not two segments.
    assert len(speech_stats(blocks)['st_vad_segments'])==4


def test_padding_clamps_to_available_audio_and_never_invents_source_gaps():
    blocks=[block(0,np.arange(5000,dtype=np.float32),[(0,1000)]),
            block(10000,np.arange(10000,15000,dtype=np.float32),[(14000,15000)])]
    actual,spans=padded_st_audio(blocks,3840)
    assert spans==[(0,4840),(10160,15000)]
    assert not np.any((actual>=5000)&(actual<10000))


@pytest.mark.parametrize('mode',['target_only','source_target'])
@pytest.mark.parametrize('audio_mode',['standard','noise_resistant'])
@pytest.mark.parametrize('crop',[False,True])
@pytest.mark.asyncio
async def test_both_formats_get_correct_stream_audio_and_padded_st(mode,audio_mode,crop):
    calls=[]
    def answer(req):
        b=json.loads(req.content);calls.append(b)
        return response(('源文'+TGT if mode=='source_target' else '')+'translated' if b['max_tokens']==256 else '')
    cfg=Settings(audio_mode=audio_mode,simulst_crop_speech=crop,models={mode:Model()},model_audio_format='float32',
                 st_wait_chunks=1,st_short_speech_guard=False,vad_leading_pad_ms=0)
    x=np.arange(25600,dtype=np.float32)/25600
    mask=np.zeros(25600,bool);mask[8000:12000]=True;mask[16000:18000]=True
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        s=VLLMSession(Model(),cfg,mode,'zh-en',http=client);s.activity=Detector([mask])
        e=[e async for e in s.push(x)][0]
    expected_stream=x if audio_mode=='standard' or not crop else x[mask]
    # The two padded segments overlap and therefore merge once.
    expected_st=(x[4160:21840] if audio_mode=='standard' else x[mask] if crop
                 else np.r_[x[8000:15840],x[16000:21840]])
    np.testing.assert_array_equal(audio_parts(calls[0])[-1],expected_stream)
    np.testing.assert_array_equal(audio_parts(calls[1])[-1],expected_st)
    assert e['text']=='translated' and e['end_ms']==1600
    assert e.get('source_text')==('源文' if mode=='source_target' else None)
    assert e['st_segment_pad_ms']==(240 if audio_mode=='standard' else None)


@pytest.mark.asyncio
async def test_standard_st_joins_speech_across_chunk_boundary_before_padding():
    calls=[]
    def answer(req):
        b=json.loads(req.content);calls.append(b);return response('ok' if b['max_tokens']==256 else '')
    cfg=Settings(model_audio_format='float32',st_wait_chunks=2)
    masks=[np.zeros(25600,bool),np.zeros(25600,bool)];masks[0][24000:]=True;masks[1][:2000]=True
    x=np.arange(51200,dtype=np.float32)/51200
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        s=VLLMSession(Model(),cfg,'target_only','zh-en',http=client);s.activity=Detector(masks)
        es=[e async for e in s.push(x)]
    np.testing.assert_array_equal(audio_parts(calls[-1])[0],x[20160:31440])
    assert es[-1]['st_retained_source_spans_ms']==[[1260,1965]]


@pytest.mark.parametrize('mode',['target_only','source_target'])
@pytest.mark.parametrize('context',[False,True])
@pytest.mark.parametrize('long_answer',[False,True])
@pytest.mark.asyncio
async def test_noise_guard_counts_new_target_only_for_both_formats(mode,context,long_answer):
    calls=[];counted=[];source='很长的源文'*30
    target='one two three four five' if long_answer else 'okay'
    def answer(req):
        b=json.loads(req.content)
        if req.url.path.endswith('/tokenize'):
            counted.append(b['prompt']);ids=list(range(len(b['prompt'].split())))
            return httpx.Response(200,json={'count':len(ids),'tokens':ids,'max_model_len':8192})
        calls.append(b)
        if len(calls)==1 and context:return response((source+TGT if mode=='source_target' else '')+'old ')
        if b['max_tokens']!=256:return response('')
        if mode=='source_target' and context and TGT in b['stop']:
            r=response(source);d=json.loads(r.content);d['choices'][0]['stop_reason']=TGT
            return httpx.Response(200,json=d)
        return response((source+TGT if mode=='source_target' and not context else '')+target)
    cfg=Settings(audio_mode='noise_resistant',models={mode:Model()},st_wait_chunks=1,st_context=context,vad_full_chunk_speech_ms=0)
    assert cfg.st_guard_enabled
    # Short current speech cannot gain a larger allowance from long context.
    mask=np.zeros(25600,bool);mask[0:3200]=True
    masks=([np.ones(25600,bool)] if context else [])+[mask]
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        s=VLLMSession(Model(),cfg,mode,'zh-en',http=client);s.activity=Detector(masks)
        es=[e async for e in s.push(np.full(25600*len(masks),.1,np.float32))]
    e=es[-1]
    assert counted[-1]==target,counted
    assert all(t != target for t in counted[:-1])
    assert e['st_short_speech_token_limit']==4
    assert e.get('st_output_rejected',False)==long_answer
    if long_answer:assert e['text']=='' and s.pending
    else:
        assert e['text']==target
        if mode=='source_target':assert e['source_text']==source


def test_mode_default_and_explicit_guard_override():
    assert Settings().st_crop_enabled and not Settings().st_guard_enabled
    assert Settings(audio_mode='noise_resistant').st_guard_enabled
    assert not Settings(audio_mode='noise_resistant',st_short_speech_guard=False).st_guard_enabled
