import asyncio
import base64
from dataclasses import replace
import io
import json
from pathlib import Path
import wave

import httpx
import numpy as np
import pytest
import soundfile as sf

from simulinfer.config import Settings, load
from simulinfer.tts import Speech
from simulinfer.tts_history import AudioHistory


def pcm(seconds, rate=24000, value=123):
    return np.full(round(seconds*rate), value, dtype='<i2').tobytes()


def decode_reference(ref):
    raw=base64.b64decode(ref['ref_audio'].split(',',1)[1])
    with wave.open(io.BytesIO(raw)) as f:
        return f.readframes(f.getnframes()),f.getframerate()


def test_history_drops_complete_pairs_and_keeps_long_last_pair():
    h=AudioHistory(10,240)
    h.commit('first',pcm(6),24000,'en','a')
    h.commit('second',pcm(6,value=456),24000,'en','a')
    ref=h.reference('en','a')
    assert ref['ref_text']=='second' and decode_reference(ref)==(pcm(6,value=456),24000)
    h.commit('long',pcm(15),24000,'en','a')
    assert h.seconds==15 and h.reference('en','a')['ref_text']=='long'
    h.commit('new',pcm(1),24000,'en','a')
    assert h.seconds==1 and h.reference('en','a')['ref_text']=='new'


def test_history_language_voice_and_rate_isolation():
    h=AudioHistory()
    h.commit('one',pcm(1),24000,'en','a')
    assert h.reference('zh','a') is None
    h.commit('二',pcm(1),24000,'zh','a')
    assert h.reference('zh','b') is None
    h.commit('三',pcm(1),24000,'zh','b')
    h.commit('四',pcm(1,rate=16000),16000,'zh','b')
    assert h.reference('zh','b')['ref_text']=='四'
    h.commit('too long',pcm(31,rate=16000),16000,'zh','b')
    assert h.reference('zh','b') is None


def test_history_text_and_waveform_order():
    h=AudioHistory()
    h.commit('first,',pcm(1),24000,'en','a')
    h.commit('second',pcm(2,value=456),24000,'en','a')
    ref=h.reference('en','a')
    assert ref['ref_text']=='first, second'
    assert decode_reference(ref)[0]==pcm(1)+pcm(2,value=456)


@pytest.fixture
def omni_settings(tmp_path):
    ref=tmp_path/'reference.wav';sf.write(ref,np.ones(24000,dtype=np.float32)*.02,24000)
    return Settings(tts_backend='vllm_omni',tts_task_type='Base',tts_history_mode='paired',
        tts_reference_audio=str(ref),tts_reference_text='reference words',tts_min_chars=1,tts_speed_zh_en=1)


class Fragmented(httpx.AsyncByteStream):
    def __init__(self,raw,error=False):self.raw=raw;self.error=error
    async def __aiter__(self):
        for i in range(0,len(self.raw),333):
            yield self.raw[i:i+333]
        if self.error:raise httpx.ReadError('broken stream')


async def make_speech(settings,responder):
    events=[]
    async def emit(e):events.append(e)
    s=Speech(settings,emit,'zh-en')
    # For session mode the initial open is part of start().
    if settings.tts_backend=='session':
        from unittest.mock import patch
        client=httpx.AsyncClient(transport=httpx.MockTransport(responder))
        with patch('simulinfer.tts.httpx.AsyncClient',return_value=client):await s.start()
    else:
        await s.start();await s.client.aclose()
        s.client=httpx.AsyncClient(transport=httpx.MockTransport(responder))
    return s,events


@pytest.mark.asyncio
async def test_omni_second_request_uses_first_complete_audio_and_text(omni_settings):
    requests=[]
    def respond(req):
        requests.append(json.loads(req.content))
        return httpx.Response(200,headers={'content-type':'audio/pcm','x-audio-sample-rate':'24000','x-audio-channels':'1'},
                              stream=Fragmented(pcm(1,value=len(requests))))
    s,events=await make_speech(omni_settings,respond)
    try:
        await s.text('First sentence,');await s.text('continuing it.');await s.finish()
        assert requests[0]['ref_text']=='reference words'
        assert requests[1]['ref_text']=='First sentence,'
        assert decode_reference(requests[1])==(pcm(1,value=1),24000)
        assert all(r['stream_format']=='audio' and r['stream'] and r['task_type']=='Base' for r in requests)
        assert all('voice' not in r and r['x_vector_only_mode'] is False for r in requests)
        data=b''.join(base64.b64decode(e['data']) for e in events if e['event']=='audio')
        assert data==pcm(1,value=1)+pcm(1,value=2)
        assert s.history.seconds==2 and events[-1]['event']=='tts_done'
    finally:await s.close()
    assert not s.history.entries and s.anchor is None


@pytest.mark.asyncio
async def test_conversations_never_share_history(omni_settings):
    bodies=[]
    def respond(req):
        bodies.append(json.loads(req.content));return httpx.Response(200,content=pcm(.1))
    a,_=await make_speech(omni_settings,respond);b,_=await make_speech(omni_settings,respond)
    try:
        await a.text('A');await a.finish();await b.text('B');await b.finish()
        assert [r['ref_text'] for r in bodies]==['reference words','reference words']
    finally:await a.close();await b.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind',['broken','odd','empty','json','sse','wav','rate','stereo'])
async def test_failed_audio_is_not_history(omni_settings,kind):
    def respond(req):
        headers={'content-type':{'json':'application/json','sse':'text/event-stream','wav':'audio/wav'}.get(kind,'audio/pcm')}
        if kind=='rate':headers['x-audio-sample-rate']='16000'
        if kind=='stereo':headers['x-audio-channels']='2'
        data=b'' if kind=='empty' else pcm(.1)+(b'\x00' if kind=='odd' else b'')
        return httpx.Response(200,headers=headers,stream=Fragmented(data,error=kind=='broken'))
    s,events=await make_speech(omni_settings,respond)
    try:
        await s.text('bad');await s.finish()
        assert any(e['event']=='tts_error' for e in events)
        assert events[-1]['event']=='tts_done' and not events[-1]['complete']
        assert not s.history.entries
    finally:await s.close()


@pytest.mark.asyncio
async def test_audio_size_limit(omni_settings):
    s,events=await make_speech(replace(omni_settings,tts_max_audio_seconds=1),
                         lambda req:httpx.Response(200,content=pcm(1.1)))
    try:
        await s.text('too long');await s.finish()
        assert any(e['event']=='tts_error' for e in events)
        assert not s.history.entries
    finally:await s.close()


@pytest.mark.asyncio
async def test_cancelled_stream_does_not_commit(omni_settings):
    entered=asyncio.Event()
    class Blocked(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield pcm(.1);entered.set();await asyncio.Event().wait()
    s,_=await make_speech(omni_settings,lambda req:httpx.Response(200,stream=Blocked()))
    await s.text('cancelled');await asyncio.wait_for(entered.wait(),2)
    await asyncio.wait_for(s.close(),2)
    assert not s.history.entries


@pytest.mark.asyncio
async def test_session_window_speed_sid_and_cleanup():
    calls=[]
    def respond(req):
        body=json.loads(req.content) if req.method=='POST' else dict(req.url.params)
        calls.append((req.url.path,body))
        if req.url.path=='/session_open':
            return httpx.Response(200,json={'sid':'private-session','sample_rate':24000,'continuity_version':'paired'})
        if req.url.path=='/session_close':return httpx.Response(200,json={'ok':True})
        return httpx.Response(200,content=pcm(.1),headers={'x-sample-rate':'24000'})
    s,events=await make_speech(Settings(tts_backend='session',tts_history_mode='paired',
        tts_history_window_s=8,tts_speed=1.1,tts_min_chars=1),respond)
    await s.text('first');await s.text('second');await s.finish();await s.close();await s.close()
    assert calls[0][1]['window_s']==8 and calls[0][1]['speed']==1.2
    assert [body['sid'] for path,body in calls if path=='/session_say']==['private-session']*2
    assert sum(path=='/session_close' for path,_ in calls)==1
    assert events[-1]['continuity']=='backend_paired_history'


def test_configuration_rejects_unsupported_history_and_speed(omni_settings,tmp_path):
    with pytest.raises(ValueError):Settings(tts_backend='openai',tts_history_mode='paired')
    with pytest.raises(ValueError):Settings(tts_backend='vllm_omni',tts_history_mode='paired')
    assert replace(omni_settings,tts_speed=1.1).tts_speed==1.1
    with pytest.raises(ValueError):replace(omni_settings,tts_reference_text='')
    path=tmp_path/'config.toml'
    path.write_text('[models.target_only]\nbackend="mock"\n[tts]\ntts_backend="vllm_omni"\ntts_task_type="Base"\ntts_reference_audio="reference.wav"\ntts_reference_text="words"\n')
    assert load(str(path)).tts_reference_audio==str(tmp_path/'reference.wav')


@pytest.mark.asyncio
async def test_short_text_flushes_before_more_text_or_eof(omni_settings):
    called=asyncio.Event();requests=[]
    def respond(req):
        requests.append(json.loads(req.content));called.set()
        return httpx.Response(200,content=pcm(.2))
    s,_=await make_speech(replace(omni_settings,tts_min_chars=40,tts_batch_ms=30),respond)
    try:
        await s.text('Short phrase')
        await asyncio.wait_for(called.wait(),.5)
        assert requests[0]['input']=='Short phrase'
        await s.finish()
    finally:await s.close()


@pytest.mark.asyncio
async def test_english_sentence_does_not_wait_for_batch_timer(omni_settings):
    called=asyncio.Event()
    def respond(req):called.set();return httpx.Response(200,content=pcm(.2))
    s,_=await make_speech(replace(omni_settings,tts_min_chars=40,tts_batch_ms=1000),respond)
    try:
        await s.text('Hello.')
        await asyncio.wait_for(called.wait(),.3)
        await s.finish()
    finally:await s.close()


@pytest.mark.asyncio
async def test_batching_preserves_fragments_and_follows_eof(omni_settings):
    requests=[]
    def respond(req):requests.append(json.loads(req.content));return httpx.Response(200,content=pcm(.2))
    s,_=await make_speech(replace(omni_settings,tts_min_chars=40),respond)
    try:
        await s.text('inter');await s.text('national ');await s.text('cooperation.')
        await s.finish()
        assert ''.join(r['input'] for r in requests)=='international cooperation.'
        assert len(requests)==1
    finally:await s.close()


@pytest.mark.asyncio
async def test_broken_segment_does_not_abort_next_or_poison_history(omni_settings):
    requests=[]
    def respond(req):
        requests.append(json.loads(req.content))
        return httpx.Response(200,stream=Fragmented(pcm(.2,value=len(requests)),error=len(requests)==1))
    s,events=await make_speech(omni_settings,respond)
    try:
        await s.text('first');await s.text('second');await s.finish()
        assert len(requests)==2 and requests[1]['ref_text']=='reference words'
        assert [e['seq'] for e in events if e['event']=='tts_error']==[1]
        assert [e['seq'] for e in events if e['event']=='tts_end']==[2]
        assert events[-1]['failed_segments']==1 and not events[-1]['complete']
        assert s.history.reference('en','')['ref_text']=='second'
    finally:await s.close()


@pytest.mark.asyncio
async def test_pcm_forwarded_without_waiting_for_large_buffer(omni_settings):
    received=asyncio.Event();release=asyncio.Event()
    class SmallFirstPacket(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield pcm(.02)
            await release.wait()
            yield pcm(.02)
    s,events=await make_speech(omni_settings,lambda req:httpx.Response(200,stream=SmallFirstPacket()))
    async def emit(e):
        events.append(e)
        if e['event']=='audio':received.set()
    s.emit=emit
    try:
        await s.text('first packet')
        await asyncio.wait_for(received.wait(),.5)
        release.set();await s.finish()
    finally:release.set();await s.close()


@pytest.mark.asyncio
async def test_tempo_stream_preserves_pitch_and_original_history(omni_settings):
    from simulinfer.tts_audio import TempoStream
    rate=24000;seconds=4
    raw=(np.sin(2*np.pi*440*np.arange(rate*seconds)/rate)*6000).astype('<i2').tobytes()
    frames=[];ready=asyncio.Event()
    async def emit(data):frames.append(data);ready.set()
    tempo=TempoStream(rate,1.1,emit)
    await tempo.start()
    try:
        await tempo.write(raw[:rate*4])
        # Must produce PCM before stdin/utterance closes.
        await asyncio.wait_for(ready.wait(),2)
        await tempo.write(raw[rate*4:]);await tempo.finish()
    finally:await tempo.close()
    adjusted=np.frombuffer(b''.join(frames),dtype='<i2')
    assert abs(len(adjusted)/rate-seconds/1.1)<.06
    spectrum=abs(np.fft.rfft(adjusted.astype(float)))
    peak=np.fft.rfftfreq(len(adjusted),1/rate)[spectrum.argmax()]
    assert abs(peak-440)<3
    requests=[]
    def respond(req):requests.append(json.loads(req.content));return httpx.Response(200,content=raw)
    s,events=await make_speech(replace(omni_settings,tts_speed=1.1,tts_speed_zh_en=1.1),respond)
    try:
        await s.text('first');await s.text('second');await s.finish()
        assert all('speed' not in r for r in requests)  # backend raw streaming accepts only speed 1
        assert decode_reference(requests[1])==(raw,rate)
        assert s.bytes < 2*len(raw) and events[-1]['complete']
    finally:await s.close()


@pytest.mark.asyncio
async def test_stalled_segment_times_out_and_next_segment_still_runs(omni_settings):
    calls=[]
    class Stalled(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield pcm(.02)
            await asyncio.Event().wait()
    def respond(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200,stream=Stalled()) if len(calls)==1 else httpx.Response(200,content=pcm(.2))
    s,events=await make_speech(replace(omni_settings,tts_timeout_seconds=1),respond)
    try:
        await s.text('first');await s.text('second')
        await asyncio.wait_for(s.finish(),2)
        assert len(calls)==2
        assert any(e['event']=='tts_error' and e['seq']==1 for e in events)
        assert any(e['event']=='tts_end' and e['seq']==2 for e in events)
        assert not s.history.reference('en','')['ref_text'].startswith('first')
    finally:await s.close()


@pytest.mark.asyncio
async def test_first_clone_starts_when_reference_ready_without_next_translation():
    from unittest.mock import patch
    requests=[];started=asyncio.Event();events=[]
    def respond(req):
        if req.method=='GET':return httpx.Response(200,json={'source_voice_history':'source-voice-history-v1'})
        requests.append(json.loads(req.content));started.set()
        return httpx.Response(200,content=pcm(1.2))
    settings=Settings(tts_backend='vllm_omni',tts_task_type='Base',tts_clone_source=True,
                      tts_min_chars=1)
    async def emit(e):events.append(e)
    speech=Speech(settings,emit,'zh-en')
    client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with patch('simulinfer.tts.httpx.AsyncClient',return_value=client):await speech.start()
    try:
        speech.feed_source(np.full(25600,.1,np.float32))
        await speech.text('Okay.')
        assert not requests
        speech.feed_source(np.full(12800,.1,np.float32))
        await speech.flush_pending()
        await asyncio.wait_for(started.wait(),.5)
        assert requests[0]['input']=='Okay.'
        assert requests[0]['x_vector_only_mode'] is True
        assert requests[0]['non_streaming_mode'] is True
        assert requests[0]['stream'] is True and requests[0]['stream_format']=='audio'
        while not speech.history.entries:await asyncio.sleep(.001)
        await speech.text('The next sentence.');await speech.finish()
        assert requests[1]['ref_text']=='Okay.'
        assert requests[1]['x_vector_only_mode'] is False
        assert requests[1]['non_streaming_mode'] is False
    finally:await speech.close()


@pytest.mark.asyncio
async def test_pending_clone_and_next_text_are_neither_lost_nor_duplicated():
    from unittest.mock import patch
    calls=[]
    def respond(req):
        if req.method=='GET':return httpx.Response(200,json={'source_voice_history':'source-voice-history-v1'})
        calls.append(json.loads(req.content));return httpx.Response(200,content=pcm(.4))
    async def emit(e):pass
    s=Speech(Settings(tts_backend='vllm_omni',tts_task_type='Base',tts_clone_source=True,
                      tts_min_chars=1),emit,'zh-en')
    client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with patch('simulinfer.tts.httpx.AsyncClient',return_value=client):await s.start()
    try:
        await s.text('First ')
        s.feed_source(np.full(38400,.1,np.float32))
        await asyncio.gather(s.flush_pending(),s.text('second.'))
        await s.finish()
        assert ''.join(r['input'] for r in calls)=='First second.'
    finally:await s.close()


@pytest.mark.asyncio
async def test_short_history_accumulates_before_becoming_clone_reference():
    from unittest.mock import patch
    requests=[]
    def respond(req):
        if req.method=='GET':return httpx.Response(200,json={'source_voice_history':'source-voice-history-v1'})
        requests.append(json.loads(req.content))
        return httpx.Response(200,content=pcm(.6))
    async def emit(e):pass
    s=Speech(Settings(tts_backend='vllm_omni',tts_task_type='Base',tts_clone_source=True,
                      tts_min_chars=1),emit,'zh-en')
    client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with patch('simulinfer.tts.httpx.AsyncClient',return_value=client):await s.start()
    try:
        s.feed_source(np.full(38400,.1,np.float32))
        await s.text('Okay.');await s.text('Next.');await s.text('Third.');await s.finish()
        assert requests[0]['x_vector_only_mode'] is True
        assert requests[1]['x_vector_only_mode'] is True
        assert requests[0]['non_streaming_mode'] is requests[1]['non_streaming_mode'] is True
        assert requests[2]['x_vector_only_mode'] is False
        assert requests[2]['non_streaming_mode'] is False
        assert requests[2]['ref_text']=='Okay. Next.'
        assert len(decode_reference(requests[2])[0])/48000==1.2
    finally:await s.close()
