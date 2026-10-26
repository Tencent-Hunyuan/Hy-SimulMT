import base64
import io
import json

import httpx
import numpy as np
import pytest
import soundfile as sf

from simulinfer.audio import part, pcm16_wav
from simulinfer.budget import render
from simulinfer.config import Settings as Defaults
from tests.helpers import Model
from simulinfer.translation import VLLMSession, SRC, TGT
from tests.helpers import Settings, model_transport


def decode(audio_part, dtype='float32'):
    raw = base64.b64decode(audio_part['input_audio']['data'])
    with sf.SoundFile(io.BytesIO(raw)) as f:
        metadata = (f.samplerate, f.channels, f.subtype, len(f))
        waveform = f.read(dtype=dtype)
    return waveform, metadata


def test_all_pcm16_values_roundtrip_exactly():
    original = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)
    samples = original.astype(np.float32) / 32768
    before = samples.copy()
    audio = part(samples)
    recovered, metadata = decode(audio, 'int16')
    assert metadata == (16000, 1, 'PCM_16', 65536)
    np.testing.assert_array_equal(recovered, original)
    np.testing.assert_array_equal(samples, before)
    recovered_float, _ = decode(audio)
    np.testing.assert_array_equal(recovered_float, samples)


def test_float32_to_pcm16_error_is_bounded_without_normalizing_volume():
    samples = np.random.default_rng(51).uniform(-.999, .999, 160000).astype(np.float32)
    decoded, _ = decode(part(samples))
    assert np.max(np.abs(samples - decoded)) <= 1/65536
    np.testing.assert_array_equal(decoded, np.rint(samples*32768).astype(np.int16).astype(np.float32)/32768)
    values = np.asarray([-4., -1., 0., 1., 4.], np.float32)
    clamped, _ = decode(part(values), 'int16')
    np.testing.assert_array_equal(clamped, [-32768, -32768, 0, 32767, 32767])


def test_sub_lsb_input_can_be_quantized_to_zero_and_float32_option_preserves_it():
    samples = np.full(25600, 1e-6, np.float32)
    pcm, _ = decode(part(samples))
    unchanged, info = decode(part(samples, 'float32'))
    assert not pcm.any() and info[2] == 'FLOAT'
    np.testing.assert_array_equal(unchanged, samples)


@pytest.mark.parametrize('samples', [np.asarray([float('nan')]), np.asarray([float('inf')]), np.zeros((4,2))])
def test_invalid_pcm_input_is_rejected(samples):
    with pytest.raises(ValueError):
        pcm16_wav(samples)


def test_transport_size_is_halved_while_tokens_and_sampling_rate_stay_the_same():
    samples = np.zeros(25600, np.float32)
    pcm, floating = part(samples), part(samples, 'float32')
    assert len(pcm['input_audio']['data']) < len(floating['input_audio']['data'])*.501
    assert pcm.num_samples == floating.num_samples == 25600
    assert pcm['input_audio']['format'] == floating['input_audio']['format'] == 'wav'
    one = render([{'role':'user','content':[pcm]}], 'target_only')
    two = render([{'role':'user','content':[floating]}], 'target_only')
    assert one == two
    assert Defaults().model_audio_format == 'pcm16'
    with pytest.raises(ValueError, match='model_audio_format'):
        Defaults(model_audio_format='invalid')


@pytest.mark.parametrize('mode', ['target_only','source_target'])
@pytest.mark.parametrize('audio_format', ['pcm16','float32'])
@pytest.mark.parametrize('chunk_ms', [800,1600])
@pytest.mark.asyncio
async def test_normal_history_st_and_eof_all_use_selected_format(mode, audio_format, chunk_ms):
    calls = []
    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        text = ''
        if body['max_tokens'] == 256:
            text = 'recovered' if mode == 'target_only' else SRC+'源文'+TGT+'recovered'
        return httpx.Response(200,json={'choices':[{'message':{'content':text},
            'finish_reason':'stop','stop_reason':120008,'token_ids':[11]}]})
    async with httpx.AsyncClient(transport=model_transport(respond)) as client:
        settings = Settings(models={mode:Model()}, model_audio_format=audio_format, chunk_ms=chunk_ms)
        session = VLLMSession(Model(),settings,mode,'zh-en',http=client)
        # Four voiced WAITs, followed by a partial EOF tail after the reset.
        count = (1600+3*chunk_ms+300)*16
        samples = np.full(count,.1,np.float32)
        events = [e async for e in session.push(samples)]
        events.extend([e async for e in session.finish()])
        assert session.ready()['model_audio_format'] == audio_format
    assert events[-1]['event'] == 'done' and events[-1]['num_chunks'] == 5
    assert events[-1]['duration_ms'] == count//16
    assert events[3]['st_audio_ms'] == 1600+3*chunk_ms
    assert events[-2]['st_audio_ms'] == 300
    assert events[-2]['final_padding_ms'] == chunk_ms-300
    expected_subtype = 'PCM_16' if audio_format == 'pcm16' else 'FLOAT'
    for request in calls:
        for message in request['messages']:
            if message['role'] != 'user': continue
            for item in message['content']:
                if item['type'] != 'input_audio': continue
                waveform, info = decode(item)
                assert info[:3] == (16000,1,expected_subtype)
                assert len(waveform) == info[3]
    # The already-encoded history is reused byte-for-byte, not re-quantized.
    assert calls[0]['messages'][0]['content'][-1] == calls[1]['messages'][0]['content'][-1]
