import io
import base64
import json

import httpx
import numpy as np
import pytest
import soundfile as sf

from simulinfer.config import Settings as Defaults
from tests.helpers import Model
from simulinfer.translation import MockSession, VLLMSession
from tests.helpers import Settings, model_transport


def answer(text):
    return httpx.Response(200, json={'choices':[{'message':{'content':text},
        'finish_reason':'stop', 'stop_reason':120008, 'token_ids':[11]}]})


@pytest.mark.parametrize('mode', ['target_only', 'source_target'])
@pytest.mark.parametrize('chunk_ms', [800, 1600])
@pytest.mark.asyncio
async def test_native_100ms_input_uses_selected_grid_and_real_eof(mode, chunk_ms):
    from simulinfer.translation import SRC, TGT
    text = 'hello' if mode == 'target_only' else SRC+'源文'+TGT+'hello'
    async with httpx.AsyncClient(transport=model_transport(lambda _: answer(text))) as client:
        settings = Settings(chunk_ms=chunk_ms, models={mode:Model()})
        session = VLLMSession(Model(), settings, mode, 'zh-en', http=client)
        events = []
        wave = np.full(3700*16, .1, np.float32)
        for pos in range(0, len(wave), 1600):
            events.extend([e async for e in session.push(wave[pos:pos+1600])])
        events.extend([e async for e in session.finish()])
    chunks = [e for e in events if e['event']=='chunk']
    expected = [(0,1600),(1600,2400),(2400,3200),(3200,3700)] if chunk_ms==800 else [(0,1600),(1600,3200),(3200,3700)]
    assert [(e['start_ms'],e['end_ms']) for e in chunks] == expected
    assert chunks[0]['model_audio_ms']==1600
    assert all(e['model_audio_ms']==chunk_ms for e in chunks[1:])
    assert chunks[-1]['final_padding_ms']==(300 if chunk_ms==800 else 1100)
    assert events[-1]['duration_ms']==3700


@pytest.mark.parametrize('chunk_ms,trigger_ms', [(800,4000),(1600,6400)])
@pytest.mark.asyncio
async def test_four_wait_blocks_include_first_1600ms_once(chunk_ms, trigger_ms):
    calls = []
    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        return answer('recovered' if body['max_tokens']==256 else '')
    async with httpx.AsyncClient(transport=model_transport(respond)) as client:
        session = VLLMSession(Model(),Settings(chunk_ms=chunk_ms),'target_only','zh-en',http=client)
        events = [e async for e in session.push(np.full(trigger_ms*16,.1,np.float32))]
    assert len(events)==4 and [e['audible_wait_chunks'] for e in events]==[1,2,3,4]
    assert events[-1]['restart_reason']=='st_audible_reset'
    assert events[-1]['st_audio_ms']==trigger_ms
    assert len(calls)==5
    audio = calls[-1]['messages'][0]['content'][1]['input_audio']['data']
    data, rate = sf.read(io.BytesIO(base64.b64decode(audio)),dtype='float32')
    assert rate==16000 and len(data)==trigger_ms*16


@pytest.mark.parametrize('chunk_ms', [800,1600])
@pytest.mark.asyncio
async def test_mock_matches_first_chunk_and_default_is_1600(chunk_ms):
    session = MockSession(Model(backend='mock'),Defaults(chunk_ms=chunk_ms),'target_only','zh-en')
    assert [e async for e in session.push(np.zeros(800*16,np.float32))]==[]
    first=[e async for e in session.push(np.zeros(800*16,np.float32))]
    assert first[0]['end_ms']==1600
    next_events=[e async for e in session.push(np.zeros(chunk_ms*16,np.float32))]
    assert next_events[0]['start_ms']==1600 and next_events[0]['end_ms']==1600+chunk_ms
    assert Defaults().first_chunk_ms==Defaults().chunk_ms==1600
