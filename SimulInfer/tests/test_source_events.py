"""Preserve model-generated source (including punctuation) during target WAIT."""
import json

import httpx
import numpy as np
import pytest

from simulinfer.config import Settings, Model
from simulinfer.events import public_event
from simulinfer.translation import VLLMSession, TGT


def cfg(**kwargs):
    return Settings(models={'source_target':Model()},default_model_type='source_target',
                    vad_mode='off',st_vad_mode='off',default_vad=False,st_crop_speech=False,
                    trailing_silence_seconds=0,**kwargs)


def respond(text):
    return httpx.Response(200,json={'choices':[{'message':{'content':text},'finish_reason':'stop',
        'stop_reason':120008,'token_ids':[]}],'usage':{'prompt_tokens':100,'completion_tokens':5}})


def displayed(events):
    text=''
    for raw in events:
        e=public_event(raw)
        if 'source_text_full' in e:text=e['source_text_full']
        elif e['event']=='done':text=e.get('source_text',text)
        else:text+=e.get('source_text','')
    return text


@pytest.mark.asyncio
async def test_source_only_sentence_end_is_published_without_completing_translation():
    replies=iter(['你好'+TGT+'Hello','。'+TGT,''])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:respond(next(replies)))) as http:
        s=VLLMSession(Model(),cfg(st_wait_chunks=0),'source_target','zh-en',http=http)
        events=[e async for e in s.push(np.ones(51200,np.float32)*.1)]
        assert events[1]['source_text']=='。' and events[1]['action']=='WAIT'
        assert s.full_source=='你好。' and s.full_text=='Hello'
        assert s.voiced_waits==1 and len(s.pending)==1
        events += [e async for e in s.finish()]
        assert events[-1]['unresolved_audio_ms']==1600
        assert displayed(events)=='你好。'
        assert events[-1]['source_text']=='你好。'
        await s.close()


@pytest.mark.asyncio
async def test_st_replaces_retranscribed_source_without_duplicate_words_or_punctuation():
    replies=iter(['他说'+TGT,'你好。'+TGT,'他说，你好。'+TGT+'He said hello.'])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:respond(next(replies)))) as http:
        s=VLLMSession(Model(),cfg(st_wait_chunks=2),'source_target','zh-en',http=http)
        events=[e async for e in s.push(np.ones(51200,np.float32)*.1)]
        assert events[0]['source_text']=='他说' and events[0]['action']=='WAIT'
        assert events[1]['source_text_full']=='他说，你好。'
        assert events[1]['source_text']==''
        assert events[1]['text']=='He said hello.'
        assert s.last_emit['source']=='他说，你好。'
        assert displayed(events)=='他说，你好。'
        assert s.full_source=='他说，你好。' and not s.pending
        await s.close()


@pytest.mark.asyncio
async def test_repeated_empty_st_source_updates_do_not_duplicate_or_mark_audio_translated():
    replies=iter(['甲'+TGT,'甲。'+TGT,'甲。'+TGT])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:respond(next(replies)))) as http:
        s=VLLMSession(Model(),cfg(st_wait_chunks=1),'source_target','zh-en',http=http)
        events=[e async for e in s.push(np.ones(25600,np.float32)*.1)]
        assert events[0]['source_text']=='甲。'
        assert events[0]['action']=='WAIT' and s.pending
        events += [e async for e in s.finish()]
        assert displayed(events)=='甲。'
        assert events[-1]['unresolved_audio_ms']==1600
        assert s.full_text==''
        await s.close()


@pytest.mark.asyncio
async def test_source_only_words_are_preserved_when_later_target_has_no_source():
    replies=iter(['完整源文。'+TGT,TGT+'Complete translation.'])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:respond(next(replies)))) as http:
        s=VLLMSession(Model(),cfg(st_wait_chunks=0),'source_target','zh-en',http=http)
        events=[e async for e in s.push(np.ones(51200,np.float32)*.1)]
        assert displayed(events)=='完整源文。'
        assert s.last_emit['source']=='完整源文。'
        assert s.full_text=='Complete translation.' and not s.pending
        await s.close()


@pytest.mark.asyncio
async def test_eof_source_correction_preserves_previous_committed_audio_interval():
    replies=iter(['旧句。'+TGT+'Old.','新句'+TGT,'新句。'+TGT+'New.'])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:respond(next(replies)))) as http:
        s=VLLMSession(Model(),cfg(st_wait_chunks=0),'source_target','zh-en',http=http)
        events=[e async for e in s.push(np.ones(51200,np.float32)*.1)]
        events += [e async for e in s.finish()]
        assert displayed(events)=='旧句。新句。'
        assert s.full_text=='Old.New.'
        await s.close()


@pytest.mark.parametrize('transport',['stream','ws'])
def test_source_only_punctuation_reaches_public_client(transport,monkeypatch):
    from fastapi.testclient import TestClient
    from simulinfer.app import build_app
    from simulinfer.audio import float_wav
    from simulinfer.translation import Reply, ChatAPI
    replies=iter([Reply('你好','Hello','你好'+TGT+'Hello',[],120008),
                  Reply('。','','。'+TGT,[],120008),Reply('','','',[],120008)])
    async def generate(self,*args,**kwargs):return next(replies)
    monkeypatch.setattr(ChatAPI,'generate',generate)
    with TestClient(build_app(cfg(st_wait_chunks=0),preflight_result={})) as client:
        if transport=='stream':
            r=client.post('/translate/stream',data={'direction':'zh-en','tts':'false'},
                files={'file':('a.wav',float_wav(np.ones(51200,np.float32)*.1),'audio/wav')})
            events=[json.loads(l[5:]) for l in r.text.splitlines() if l.startswith('data:')]
        else:
            with client.websocket_connect('/translate/ws') as ws:
                ws.send_json({'direction':'zh-en','tts':False,'input_format':'f32le'})
                events=[ws.receive_json()]
                ws.send_bytes(np.ones(51200,dtype='<f4').tobytes());ws.send_json({'eof':True})
                while events[-1]['event']!='done':events.append(ws.receive_json())
        assert any(e.get('source_text')=='。' and e.get('text')=='' for e in events)
        assert events[-1]['source_text']=='你好。'
        assert events[-1]['full_text']=='Hello'
        assert events[-1]['complete'] is False
