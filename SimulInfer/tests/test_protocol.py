import asyncio
import base64
from dataclasses import replace
import io
import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf
from jinja2 import Environment

from simulinfer.app import build_app, translate
from simulinfer.audio import PCM, float_wav, part
from simulinfer.config import load
from tests.helpers import Model
from tests.helpers import Settings, model_transport
from simulinfer.translation import ChatAPI, VLLMSession, SRC, TGT, EOT, split_reply
from simulinfer.tts import Speech


def wav(seconds=3.7, value=.1):
    return float_wav(np.full(round(seconds*16000),value,np.float32))


def mock_settings():return load(mock=True)


def response(text='',stop=None,finish='stop'):
    return httpx.Response(200,json={'choices':[{'message':{'content':text},
        'finish_reason':finish,'stop_reason':stop,'token_ids':[1,2]}]})


def test_no_training_or_vllm_required_by_app():
    import ast
    root=Path(__file__).parents[1]/'simulinfer'
    for name in ['app.py','translation.py','audio.py','config.py','tts.py','client.py']:
        tree=ast.parse((root/name).read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any(m.startswith(('torch','vllm','hytuner','simulst_infer')) for m in imports)


@pytest.mark.parametrize('fmt,dtype,width',[('s16le','<i2',2),('f32le','<f4',4)])
def test_pcm_split_at_arbitrary_byte_boundary(fmt,dtype,width):
    a=np.arange(4000,dtype=dtype)
    raw=a.tobytes();p=PCM(fmt);pieces=[]
    for offset in range(0,len(raw),303):pieces.append(p.feed(raw[offset:offset+303]))
    p.finish();out=np.concatenate(pieces)
    np.testing.assert_array_equal(out,a.astype(np.float32)/(32768 if width==2 else 1))
    bad=PCM(fmt);bad.feed(b'1')
    with pytest.raises(ValueError):bad.finish()


def test_float_wav_bit_exact():
    x=np.random.default_rng(1).uniform(-1,1,25600).astype(np.float32)
    encoded=part(x, 'float32')['input_audio']['data']
    y,sr=sf.read(io.BytesIO(base64.b64decode(encoded)),dtype='float32')
    assert sr==16000
    np.testing.assert_array_equal(x,y)


@pytest.mark.asyncio
async def test_old_assistant_has_no_duplicate_end_or_source_marker():
    calls=[]
    def transport(request):
        calls.append(json.loads(request.content))
        return response(SRC+'原文'+TGT+'target'+EOT)
    client=httpx.AsyncClient(transport=model_transport(transport))
    session=VLLMSession(Model(),Settings(models={'source_target':Model()}),'source_target','zh-en',http=client)
    async for _ in session.push(np.full(51200,.1,np.float32)):pass
    history=calls[-1]['messages'][1]['content']
    assert history==SRC+'原文'+TGT+'target'
    await client.aclose()


def test_split_reply_validates_model_format():
    assert split_reply('hello','target_only')==('','hello')
    assert split_reply(SRC+'你好'+TGT+'hello','source_target')==('你好','hello')
    assert split_reply(TGT,'source_target')==('','')
    assert split_reply('','source_target')==('','')
    with pytest.raises(ValueError):split_reply('not a structured answer','source_target')
    with pytest.raises(ValueError):split_reply(SRC+'x'+TGT+'y','target_only')


@pytest.mark.asyncio
async def test_chat_request_preserves_wait_and_language_fields():
    calls=[]
    def transport(request):
        calls.append(json.loads(request.content));return response('hello')
    client=httpx.AsyncClient(transport=model_transport(transport))
    s=Settings(models={'target_only':Model()},window_seconds=30,st_wait_chunks=0)
    session=VLLMSession(Model(),s,'target_only','zh-en',http=client)
    events=[e async for e in session.push(np.full(25600,.1,np.float32))]
    assert events[0]['text']=='hello'
    assert calls[0]['messages'][0]['content'][1]['type']=='input_audio'
    assert calls[0]['temperature']==0 and calls[0]['max_tokens']==64
    assert calls[0]['chat_template_kwargs']['output_mode']=='target_only'
    assert 'multi_modal_data' not in calls[0]
    await client.aclose()


@pytest.mark.asyncio
async def test_post_speech_silence_zero_fill_preserves_timeline_and_voice():
    calls=[]
    def transport(request):
        body=json.loads(request.content)
        latest=body['messages'][-1]['content'][-1]['input_audio']['data']
        audio,_=sf.read(io.BytesIO(base64.b64decode(latest)),dtype='float32')
        calls.append(audio)
        return response('spoken' if len(calls)==1 else '')
    class Activity:
        def __init__(self):self.index=0
        def analyze(self,samples,start):
            silent=self.index==1;self.index+=1
            return {'is_silence':silent,'audio_rms':0.0 if silent else .1,
                    'vad_trailing_silence_samples':len(samples) if silent else 0}
    client=httpx.AsyncClient(transport=model_transport(transport))
    settings=Settings(post_speech_silence_mode='zero_fill',st_wait_chunks=0,
                      trailing_silence_seconds=0)
    session=VLLMSession(Model(),settings,'target_only','zh-en',http=client)
    session.activity=Activity()
    events=[]
    for value in (.1,.2,.3):
        events += [e async for e in session.push(np.full(25600,value,np.float32))]
    assert len(calls)==3
    assert np.any(calls[0]) and not np.any(calls[1]) and np.any(calls[2])
    assert not events[0]['silence_zero_filled']
    assert events[1]['silence_zero_filled'] and events[1]['model_audio_rms']==0
    assert not events[2]['silence_zero_filled']
    assert [(e['start_ms'],e['end_ms']) for e in events]==[(0,1600),(1600,3200),(3200,4800)]
    await client.aclose()


@pytest.mark.asyncio
async def test_post_speech_silence_passthrough_keeps_original_audio():
    calls=[]
    def transport(request):
        body=json.loads(request.content)
        latest=body['messages'][-1]['content'][-1]['input_audio']['data']
        audio,_=sf.read(io.BytesIO(base64.b64decode(latest)),dtype='float32')
        calls.append(audio)
        return response('spoken' if len(calls)==1 else '')
    class Activity:
        def __init__(self):self.index=0
        def analyze(self,samples,start):
            silent=self.index==1;self.index+=1
            return {'is_silence':silent,'audio_rms':0.0 if silent else .1,
                    'vad_trailing_silence_samples':len(samples) if silent else 0}
    client=httpx.AsyncClient(transport=model_transport(transport))
    settings=Settings(post_speech_silence_mode='passthrough',st_wait_chunks=0,
                      trailing_silence_seconds=0)
    session=VLLMSession(Model(),settings,'target_only','zh-en',http=client)
    session.activity=Activity()
    events=[]
    for value in (.1,.2):
        events += [e async for e in session.push(np.full(25600,value,np.float32))]
    assert np.any(calls[1])
    assert not events[1]['silence_zero_filled']
    await client.aclose()


@pytest.mark.asyncio
async def test_st_keeps_zero_filled_silence_duration():
    calls=[]
    def transport(request):
        body=json.loads(request.content);calls.append(body)
        return response('recovered' if body['max_tokens']==256 else '')
    class Activity:
        def __init__(self):self.index=0
        def analyze(self,samples,start):
            silent=self.index==1;self.index+=1
            return {'is_silence':silent,'audio_rms':0.0 if silent else .1,
                    'vad_trailing_silence_samples':len(samples) if silent else 0}
    client=httpx.AsyncClient(transport=model_transport(transport))
    settings=Settings(post_speech_silence_mode='zero_fill',st_wait_chunks=0,
                      trailing_silence_seconds=1.6,
                      st_trim_trailing_silence=False)
    session=VLLMSession(Model(),settings,'target_only','zh-en',http=client)
    session.activity=Activity()
    events=[]
    events += [e async for e in session.push(np.full(25600,.25,np.float32))]
    events += [e async for e in session.push(np.full(25600,.05,np.float32))]
    st=next(c for c in calls if c['max_tokens']==256)
    encoded=st['messages'][0]['content'][1]['input_audio']['data']
    audio,sr=sf.read(io.BytesIO(base64.b64decode(encoded)),dtype='float32')
    assert sr==16000 and len(audio)==51200
    assert np.allclose(audio[:25600],.25,atol=1/32768)
    assert not np.any(audio[25600:])
    assert events[-1]['st_audio_ms']==3200
    assert events[-1]['st_trimmed_silence_ms']==0
    assert events[-1].get('st_trimmed_silence_chunks',0)==0
    await client.aclose()


@pytest.mark.asyncio
async def test_mixed_chunk_keeps_speech_and_zeros_silent_frames_for_simulst_and_st():
    calls=[]
    def transport(request):
        body=json.loads(request.content);calls.append(body)
        return response('recovered' if body['max_tokens']==256 else '')
    class Activity:
        def __init__(self):self.index=0
        def analyze(self,samples,start):
            mask=np.ones(len(samples),dtype=np.bool_)
            if self.index==1:mask[len(samples)//2:]=False
            silent=not mask.any();self.index+=1
            return {'is_silence':silent,'audio_rms':.1,
                    'vad_trailing_silence_samples':len(samples)//2 if self.index==2 else 0,
                    '_vad_speech_mask':mask}
    client=httpx.AsyncClient(transport=model_transport(transport))
    settings=Settings(post_speech_silence_mode='zero_fill',st_wait_chunks=2,
                      trailing_silence_seconds=0,st_trim_trailing_silence=False)
    session=VLLMSession(Model(),settings,'target_only','zh-en',http=client)
    session.activity=Activity()
    events=[]
    events += [e async for e in session.push(np.full(25600,.25,np.float32))]
    events += [e async for e in session.push(np.full(25600,.5,np.float32))]
    normal=calls[1]['messages'][-1]['content'][-1]['input_audio']['data']
    normal_audio,_=sf.read(io.BytesIO(base64.b64decode(normal)),dtype='float32')
    assert np.allclose(normal_audio[:12800],.5,atol=1/32768)
    assert not np.any(normal_audio[12800:])
    assert events[1]['silence_partially_zero_filled']
    assert events[1]['silence_zero_filled_ms']==800
    st=next(c for c in calls if c['max_tokens']==256)
    encoded=st['messages'][0]['content'][1]['input_audio']['data']
    st_audio,_=sf.read(io.BytesIO(base64.b64decode(encoded)),dtype='float32')
    assert len(st_audio)==51200
    assert np.allclose(st_audio[25600:38400],.5,atol=1/32768)
    assert not np.any(st_audio[38400:])
    await client.aclose()


@pytest.mark.asyncio
async def test_long_continuous_speech_passes_whole_chunk_and_short_speech_keeps_padding():
    calls=[]
    def transport(request):
        body=json.loads(request.content);calls.append(body);return response('')
    class Activity:
        def __init__(self):self.index=0
        def analyze(self,samples,start):
            raw=np.zeros(len(samples),dtype=np.bool_)
            speech=np.zeros(len(samples),dtype=np.bool_)
            if self.index==0:
                raw[4800:22400]=True; speech[4800:24000]=True
            else:
                raw[8000:12800]=True; speech[8000:14400]=True
            self.index+=1
            return {'is_silence':False,'audio_rms':.1,
                    'vad_trailing_silence_samples':0,
                    '_vad_speech_mask':speech,'_vad_raw_speech_mask':raw}
    client=httpx.AsyncClient(transport=model_transport(transport))
    settings=Settings(post_speech_silence_mode='zero_fill',st_wait_chunks=0,
        trailing_silence_seconds=0,vad_leading_pad_ms=160,
        vad_full_chunk_speech_ms=1000)
    session=VLLMSession(Model(),settings,'target_only','zh-en',http=client)
    session.activity=Activity()
    events=[]
    events += [e async for e in session.push(np.full(25600,.25,np.float32))]
    events += [e async for e in session.push(np.full(25600,.5,np.float32))]
    first=sf.read(io.BytesIO(base64.b64decode(
        calls[0]['messages'][-1]['content'][-1]['input_audio']['data'])),dtype='float32')[0]
    second=sf.read(io.BytesIO(base64.b64decode(
        calls[1]['messages'][-1]['content'][-1]['input_audio']['data'])),dtype='float32')[0]
    assert events[0]['vad_full_chunk_passthrough'] and np.allclose(first,.25,atol=1/32768)
    assert not events[1]['vad_full_chunk_passthrough']
    assert np.allclose(second[5440:14400],.5,atol=1/32768)
    assert not np.any(second[:5440]) and not np.any(second[14400:])
    await client.aclose()


@pytest.mark.asyncio
async def test_short_speech_st_guard_rejects_long_answer_without_resetting_pending_audio():
    calls=[]
    def transport(request):
        body=json.loads(request.content);calls.append(body)
        return response('abcdefgh' if body['max_tokens']==256 else '')
    class STActivity:
        def analyze(self,samples):
            return {'st_vad_mode':'firered_nonstream','st_vad_threshold':.4,
                'st_vad_audio_ms':len(samples)//16,'st_vad_speech_ms':300.0,
                'st_vad_longest_speech_ms':300.0,'st_vad_segments':[[0.1,.4]],
                'st_vad_probability_max':.9,'st_vad_probability_mean':.2,
                'st_vad_seconds':.01,'st_vad_accepted':True}
    client=httpx.AsyncClient(transport=model_transport(transport))
    settings=Settings(st_wait_chunks=1,trailing_silence_seconds=0,
        st_vad_mode='firered_nonstream',st_short_speech_guard=True,vad_full_chunk_speech_ms=0)
    session=VLLMSession(Model(),settings,'target_only','zh-en',http=client)
    session.st_activity=STActivity()
    events=[e async for e in session.push(np.full(25600,.1,np.float32))]
    event=events[-1]
    assert event['st_output_rejected']
    assert event['st_short_speech_token_limit']==4
    assert event['st_short_speech_output_tokens']==8
    assert event['text']=='' and event['action']=='WAIT'
    assert len(session.pending)==1 and session.run==0
    await client.aclose()


@pytest.mark.asyncio
async def test_st_vad_rejects_no_speech_before_model_st_call():
    calls=[]
    def transport(request):
        body=json.loads(request.content);calls.append(body);return response('')
    class STActivity:
        def analyze(self,samples):
            return {'st_vad_mode':'firered_nonstream','st_vad_threshold':.4,
                'st_vad_audio_ms':len(samples)//16,'st_vad_speech_ms':0.0,
                'st_vad_longest_speech_ms':0.0,'st_vad_segments':[],
                'st_vad_probability_max':.1,'st_vad_probability_mean':.01,
                'st_vad_seconds':.01,'st_vad_accepted':False}
    client=httpx.AsyncClient(transport=model_transport(transport))
    settings=Settings(st_wait_chunks=1,trailing_silence_seconds=0,
        st_vad_mode='firered_nonstream',st_short_speech_guard=True,vad_full_chunk_speech_ms=0)
    session=VLLMSession(Model(),settings,'target_only','zh-en',http=client)
    session.st_activity=STActivity()
    event=[e async for e in session.push(np.full(25600,.1,np.float32))][-1]
    assert event['st_vad_rejected'] and event['st_fallback_skipped']=='nonstream_vad_no_speech'
    assert len([c for c in calls if c.get('max_tokens')==256])==0
    await client.aclose()


@pytest.mark.asyncio
async def test_fourth_wait_flushes_all_four_audio_blocks():
    calls=[]
    def transport(request):
        d=json.loads(request.content);calls.append(d)
        return response('recovered' if d['max_tokens']==256 else '')
    client=httpx.AsyncClient(transport=model_transport(transport))
    session=VLLMSession(Model(),Settings(),'target_only','zh-en',http=client)
    es=[]
    for i in range(4):
        es += [e async for e in session.push(np.full(25600,.1,np.float32))]
        assert len(calls)==(5 if i==3 else i+1)
    assert [e['action'] for e in es]==['WAIT']*3+['EMIT']
    st=calls[-1];a=st['messages'][0]['content'][1]['input_audio']['data']
    audio,sr=sf.read(io.BytesIO(base64.b64decode(a)),dtype='float32')
    assert len(audio)==102400 and sr==16000
    assert es[-1]['st_audio_ms']==6400
    assert session.messages==[]  # successful recovery starts a fresh run
    assert es[-1]['restart_reason']=='st_audible_reset'
    await client.aclose()


@pytest.mark.asyncio
async def test_context_st_prefix_and_audio_alignment():
    calls=[]
    def transport(request):
        d=json.loads(request.content);calls.append(d)
        return response(' old' if len(calls)==1 else ' new' if d['max_tokens']==256 else '')
    client=httpx.AsyncClient(transport=model_transport(transport))
    s=Settings(st_context=True)
    session=VLLMSession(Model(),s,'target_only','zh-en',http=client)
    x=np.full(25600,.2,np.float32)
    es=[e async for e in session.push(x)]
    for _ in range(4):es += [e async for e in session.push(np.full(25600,.3,np.float32))]
    st=calls[-1]
    assert st['chat_template_kwargs']['harness_prefix']==' old'
    assert es[-1]['text']==' new' and session.full_text==' old new'
    assert es[-1]['st_context_audio_ms']==1600 and es[-1]['st_audio_ms']==6400
    a,_=sf.read(io.BytesIO(base64.b64decode(st['messages'][0]['content'][1]['input_audio']['data'])),dtype='float32')
    np.testing.assert_array_equal(a[:25600],x)
    assert len(a)==128000
    await client.aclose()


@pytest.mark.asyncio
async def test_source_target_st_has_correct_two_prefixes():
    calls=[]
    def transport(request):
        d=json.loads(request.content);calls.append(d)
        if len(calls)==1:return response('旧源'+TGT+'old')
        if TGT in d['stop']:return response('新源',stop=TGT)
        if d['max_tokens']==256:return response(' new')
        return response('')
    client=httpx.AsyncClient(transport=model_transport(transport))
    s=Settings(models={'source_target':Model()},st_context=True,no_chinese_target=True)
    session=VLLMSession(Model(),s,'source_target','zh-en',http=client)
    for _ in range(5):es=[e async for e in session.push(np.full(25600,.1,np.float32))]
    assert calls[-2]['chat_template_kwargs']['harness_prefix']==SRC+'旧源'
    assert 'hy_target_language' not in calls[-2].get('vllm_xargs', {})
    assert calls[-1]['chat_template_kwargs']['harness_prefix']==SRC+'旧源新源'+TGT+'old'
    assert calls[-1]['vllm_xargs']['hy_target_language']=='target'
    assert es[-1]['source_text']=='新源' and es[-1]['text']==' new'
    await client.aclose()


@pytest.mark.asyncio
async def test_empty_st_does_not_pretend_success_or_lose_source():
    def transport(request):return response('')
    client=httpx.AsyncClient(transport=model_transport(transport))
    session=VLLMSession(Model(),Settings(reset_after_empty_st=False),'target_only','zh-en',http=client)
    es=[e async for e in session.push(np.full(6*16000,.1,np.float32))]
    es += [e async for e in session.finish()]
    cs=[e for e in es if e['event']=='chunk']
    assert len(cs)==4 and cs[-1]['end_ms']==6000 and cs[-1]['final_padding_ms']==400
    assert all(c['action']=='WAIT' for c in cs)
    assert es[-1]['unresolved_audio_ms']==6000 and session.pending
    await client.aclose()


@pytest.mark.asyncio
async def test_generation_length_is_reported_for_run_reset():
    client=httpx.AsyncClient(transport=model_transport(lambda _:response('partial',finish='length')))
    api=ChatAPI(Model(),Settings(),'target_only','zh-en',client)
    reply=await api.generate([])
    assert reply.finish_reason=='length' and reply.text=='partial'
    await client.aclose()


def test_template_has_exact_turn_order_and_open_source_prefix():
    path=Path(__file__).parents[1]/'simulinfer/templates/simulst.jinja'
    template=Environment().from_string(path.read_text())
    messages=[{'role':'user','content':[{'type':'text','text':'instruction'},{'type':'audio'}]},
              {'role':'assistant','content':[{'type':'text','text':SRC+'原文'+TGT+'target'}]},
              {'role':'user','content':[{'type':'audio'}]}]
    out=template.render(messages=messages,add_generation_prompt=True,output_mode='source_target')
    assert out.count('<｜hy_place▁holder▁no▁5｜>')==2
    assert out.count(EOT)==1
    assert out.endswith('<｜hy_Assistant｜><think></think>'+SRC)
    assert SRC+'原文'+TGT+'target'+EOT in out
    out=template.render(messages=messages[:1],add_generation_prompt=True,output_mode='target_only',harness_prefix='old target')
    assert out.endswith('<think></think>old target') and EOT not in out


def test_stream_mock_tts_and_source_timeline():
    with TestClient(build_app(mock_settings())) as c:
        r=c.post('/translate/stream',data={'mode':'source_target','tts':'true'},files={'file':('x.wav',wav(),'audio/wav')})
        es=[json.loads(line[5:]) for line in r.text.splitlines() if line.startswith('data:')]
        assert es[0]['event']=='ready' and es[-1]['event']=='done'
        cs=[e for e in es if e['event']=='chunk']
        assert [(e['start_ms'],e['end_ms']) for e in cs]==[(0,1600),(1600,3200),(3200,3700)]
        assert any(e.get('source_text') for e in cs)
        audio=[e for e in es if e['event']=='audio']
        assert audio and all(base64.b64decode(e['data']) for e in audio)
        assert es[-1]['tts_audio']['bytes']>0
        assert c.get('/health').json()['active_sessions']==0


def test_ws_binary_audio_and_input_fragments():
    with TestClient(build_app(mock_settings())) as client:
        with client.websocket_connect('/translate/ws') as ws:
            ws.send_json({'mode':'target_only','tts':True,'input_format':'s16le'})
            assert ws.receive_json()['event']=='ready'
            raw=np.full(59200,4000,dtype='<i2').tobytes()
            for i in range(0,len(raw),3201):ws.send_bytes(raw[i:i+3201])
            ws.send_json({'eof':True})
            got_audio=False
            while True:
                e=ws.receive_json()
                assert e['event']!='error',e
                if e['event']=='audio':
                    assert len(ws.receive_bytes())==e['bytes'];got_audio=True
                if e['event']=='done':
                    assert e['duration_ms']==3700;break
            assert got_audio


def test_bad_inputs_return_errors():
    with TestClient(build_app(mock_settings())) as c:
        assert c.post('/translate/stream',data={'mode':'wrong'},files={'file':('x.wav',wav(),'audio/wav')}).status_code==400
        with c.websocket_connect('/translate/ws') as ws:
            ws.send_json({'input_format':'wrong'});assert ws.receive_json()['event']=='error'
        with c.websocket_connect('/translate/ws') as ws:
            ws.send_json({});assert ws.receive_json()['event']=='ready'
            ws.send_bytes(b'1');ws.send_json({'eof':True});assert ws.receive_json()['event']=='error'


def test_health_has_no_api_keys(monkeypatch):
    monkeypatch.setenv('VLLM_API_KEY','unit-test-secret')
    with TestClient(build_app(mock_settings())) as c:
        assert 'unit-test-secret' not in c.get('/health').text


@pytest.mark.asyncio
async def test_tts_failure_cannot_deadlock_producer():
    s=replace(mock_settings(),tts_min_chars=1)
    async def emit(e):
        if e['event']=='audio':raise RuntimeError('sink failed')
    speech=Speech(s,emit,'zh-en')
    await speech.start()
    with pytest.raises(RuntimeError,match='sink failed'):
        await speech.text('test')
        await asyncio.sleep(.01)
        await asyncio.wait_for(speech.finish(),2)
    await speech.close()


@pytest.mark.asyncio
async def test_window_reset_reports_unresolved_audio_and_keeps_every_chunk():
    client=httpx.AsyncClient(transport=model_transport(lambda _:response('')))
    s=Settings(window_seconds=3.2,st_wait_chunks=0)
    session=VLLMSession(Model(),s,'target_only','zh-en',http=client)
    es=[e async for e in session.push(np.full(5*16000,.1,np.float32))]
    es += [e async for e in session.finish()]
    cs=[e for e in es if e['event']=='chunk']
    assert [(c['start_ms'],c['end_ms']) for c in cs]==[(0,1600),(1600,3200),(3200,4800),(4800,5000)]
    assert cs[1]['window_reset'] and cs[1]['skipped_untranslated_audio_ms']==3200
    assert es[-1]['skipped_untranslated_audio_ms']==5000
    assert es[-1]['unresolved_audio_ms']==0
    assert es[-1]['empty_st_reset_count']==2
    await client.aclose()


def test_disconnect_releases_session_slot():
    with TestClient(build_app(replace(mock_settings(),max_sessions=1))) as c:
        with c.websocket_connect('/translate/ws') as ws:
            ws.send_json({});assert ws.receive_json()['event']=='ready'
            ws.send_bytes(np.full(25600,1234,dtype='<i2').tobytes())
            assert ws.receive_json()['event']=='chunk'
            # Wait for server cleanup before TestClient cancels its ASGI task.
            # Older Starlette versions otherwise race fut.result() at exit.
            ws.close()
            assert ws.receive()['type']=='websocket.close'
        assert c.get('/health').json()['active_sessions']==0
        with c.websocket_connect('/translate/ws') as ws:
            ws.send_json({});assert ws.receive_json()['event']=='ready'
            ws.send_json({'eof':True});assert ws.receive_json()['event']=='done'


def test_launcher_passes_vllm_scheduler_and_audio_template(tmp_path):
    from types import SimpleNamespace
    from simulinfer.launch import command
    (tmp_path/'config.json').write_text(json.dumps({'architectures':['HYV3SpeechForCausalLM']}))
    (tmp_path/'preprocessor_config.json').write_text('{}')
    from simulinfer.prepare_model import TEMPLATE
    (tmp_path/'chat_template.jinja').write_bytes(TEMPLATE.read_bytes())
    args=SimpleNamespace(model_dir=str(tmp_path),host='127.0.0.1',port=8800,model_name='simulst',
        tp=1,max_model_len=8192,max_num_seqs=4,max_num_batched_tokens=16384,
        gpu_memory_utilization=.9,audio_items=256,no_chinese_target=True,logits_api='v1')
    cmd=command(args)
    assert cmd[cmd.index('--max-num-seqs')+1]=='4'
    assert cmd[cmd.index('--max-model-len')+1]=='8192'
    assert cmd[cmd.index('--chat-template-content-format')+1]=='openai'
    assert 'simulinfer.target_language:NoChineseTargetProcessor' in cmd
    assert Path(cmd[cmd.index('--chat-template')+1]).is_file()
    args.logits_api='v2'
    modern=command(args)
    assert 'simulinfer.logits_v2:NoChineseTargetProcessor' in modern
    assert 'simulinfer.logits_v2:STNoRepeatNgramProcessor' not in modern  # auto-loaded entry point


@pytest.mark.parametrize('version,setting,requested,expected',[
    ('0.13.0',None,'auto','v1'),
    ('0.31.0+cu129',None,'auto','v2'),
    ('0.32.0rc1',None,'auto','v2'),
    ('0.31.0','0','auto','v1'),
    ('0.31.0','0','v2','v2'),
])
def test_launcher_runner_selection(monkeypatch,version,setting,requested,expected):
    from simulinfer import launch
    monkeypatch.setattr(launch.metadata,'version',lambda name:version)
    monkeypatch.delenv('VLLM_USE_V2_MODEL_RUNNER',raising=False)
    if setting is not None:monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER',setting)
    assert launch.logits_api(requested)==expected


@pytest.mark.asyncio
async def test_tts_openai_request_pcm_order():
    seen=[];events=[]
    s=replace(mock_settings(),tts_backend='openai',tts_model='demo-model',tts_voice='voice',tts_min_chars=1)
    speech=Speech(s,events.append,'zh-en')
    async def emit(e):events.append(e)
    speech.emit=emit
    await speech.start()
    await speech.client.aclose()
    def responder(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,content=b'\x01\x00'*100)
    speech.client=httpx.AsyncClient(transport=httpx.MockTransport(responder))
    await speech.text('one');await speech.text('two');await speech.finish();await speech.close()
    assert [r['input'] for r in seen]==['one','two']
    assert all(r['response_format']=='pcm' for r in seen)
    assert [e['seq'] for e in events if e['event']=='audio']==[1,2]
    assert events[-1]['event']=='tts_done'


@pytest.mark.asyncio
async def test_service_final_recovery_is_flush_not_extra_audio_chunk(monkeypatch):
    from simulinfer.translation import ServiceSession
    class WS:
        async def recv(self):return json.dumps({'event':'chunk','final_recovery':True,'text':'tail','end_ms':1600})
    s=ServiceSession(Model(backend='service'),Settings(),'target_only','zh-en');s.ws=WS()
    events=s.events()
    e=await anext(events)
    assert e['event']=='flush' and e['text']=='tail'
    await events.aclose()


@pytest.mark.asyncio
async def test_tts_http_failure_keeps_translation_and_reports_partial_audio(monkeypatch):
    from simulinfer.tts import Speech
    real_start=Speech.start
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'\x01\x00'*100
            raise httpx.RemoteProtocolError('peer closed connection without sending complete message body')
    async def start(self):
        await real_start(self);await self.client.aclose()
        self.client=httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(200,stream=Broken())))
    monkeypatch.setattr(Speech,'start',start)
    settings=replace(mock_settings(),tts_backend='openai',tts_min_chars=1)
    events=[]
    async def source():
        for _ in range(3):yield np.ones(25600,dtype=np.float32)*.1
    async def emit(e):events.append(e)
    await translate(settings,'target_only','zh-en',source(),emit,tts=True)
    assert events[-1]['event']=='done' and events[-1]['full_text']
    assert not any(e['event']=='error' for e in events)
    assert any(e['event']=='tts_error' for e in events)
    assert events[-1]['tts_audio']['complete'] is False
    assert events[-1]['tts_audio']['failed_segments']>0


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,direction,source_phase,target_only,prefix,expected',[
    ('source_target','zh-en',False,False,None,'after_target'),
    ('source_target','zh-en',False,False,SRC+'中文源文'+TGT+'English ','target'),
    ('source_target','zh-en',True,False,SRC+'中文源文',None),
    ('target_only','zh-en',False,False,None,'target'),
    ('source_target','en-zh',False,False,None,None),
])
@pytest.mark.parametrize('st',[False,True])
async def test_target_language_tag_matches_current_field_only(mode,direction,source_phase,target_only,prefix,expected,st):
    calls=[]
    def transport(request):
        body=json.loads(request.content);calls.append(body)
        if source_phase:return response('中文源文',stop=TGT)
        text='中文源文'+TGT+'English' if mode=='source_target' else 'English'
        return response(text,stop=120008)
    cfg=Settings(models={mode:Model()},model_logits_processors=True,no_chinese_target=True)
    async with httpx.AsyncClient(transport=model_transport(transport)) as client:
        api=ChatAPI(Model(),cfg,mode,direction,http=client)
        await api.generate([{'role':'user','content':[{'type':'text','text':'test'}]}],prefix=prefix,
                           source_phase=source_phase,target_only=target_only,st=st)
    assert calls[-1].get('vllm_xargs',{}).get('hy_target_language')==expected
