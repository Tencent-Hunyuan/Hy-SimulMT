import builtins
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from simulinfer.app import build_app
from simulinfer.audio import float_wav
from simulinfer.config import load
from tests.helpers import Model
from tests.helpers import Settings, model_transport
from simulinfer.translation import Reply, VLLMSession
from simulinfer.vad import SileroActivity, create_activity, resolve_model_path


@pytest.mark.parametrize('options', [
    {'vad_mode': 'bad'}, {'vad_threshold': 0}, {'vad_threshold': float('nan')},
    {'vad_release_threshold': .6}, {'vad_pad_ms': -1}, {'vad_pad_ms': float('inf')},
    {'vad_cpu_threads': 0}, {'vad_cpu_threads': 1.5}, {'silence_rms': float('nan')},
])
def test_invalid_vad_config_is_rejected(options):
    with pytest.raises(ValueError):
        Settings(**options)


def test_config_examples_and_relative_model_path(tmp_path, monkeypatch):
    for path in (Path(__file__).parents[1]/'configs').glob('*.toml'):
        load(str(path))
    config = tmp_path/'app.toml'
    config.write_text('[models.target_only]\nbackend="vllm"\n[inference]\n'
                      'vad_mode="silero"\nvad_model_path="weights/vad.jit"\n')
    monkeypatch.chdir('/')
    assert load(str(config)).vad_model_path == str(tmp_path/'weights/vad.jit')


def test_energy_and_off_require_no_torch(monkeypatch):
    original = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name.startswith(('torch', 'silero_vad')):
            raise ImportError('optional dependency not installed')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_torch)
    silent = np.zeros(25600, np.float32)
    assert create_activity(Settings()).analyze(silent, 0)['is_silence']
    assert not create_activity(Settings(vad_mode='off')).analyze(silent, 0)['is_silence']
    with pytest.raises(RuntimeError, match='可选依赖'):
        create_activity(Settings(vad_mode='silero'))


def test_explicit_path_overrides_bundled_default(tmp_path, monkeypatch):
    import simulinfer.vad as vad
    model = tmp_path/'vad.jit'
    model.write_bytes(b'test')
    assert resolve_model_path(str(model)) == str(model)
    assert Path(resolve_model_path('')).is_file()
    assert Path(resolve_model_path('')).name == 'silero_vad_6.2.3.jit'
    assert resolve_model_path('') != str(model)
    monkeypatch.setattr(vad, 'BUNDLED_MODEL', model)
    assert resolve_model_path('') == str(model)
    model.unlink()
    with pytest.raises(FileNotFoundError):
        resolve_model_path(str(model))
    with pytest.raises(FileNotFoundError):
        resolve_model_path('')


def fake_vad(monkeypatch):
    torch = pytest.importorskip('torch')
    import simulinfer.vad as vad
    models = []
    class ProbabilityModel:
        def eval(self): return self
        def reset_states(self): self.calls = 0
        def __call__(self, samples, sample_rate):
            self.calls += 1
            return torch.tensor([.9 if samples.abs().max() > .05 else .02])
    def new_model(*args, **kwargs):
        model = ProbabilityModel()
        models.append(model)
        return model
    monkeypatch.setattr(vad, 'resolve_model_path', lambda _: 'test-only-model')
    monkeypatch.setattr(vad, 'model_bytes', lambda _: b'test')
    monkeypatch.setattr(torch.jit, 'load', new_model)
    return models


def test_silero_isolation_repeated_query_and_partial_eof(monkeypatch):
    models = fake_vad(monkeypatch)
    a, b = SileroActivity(''), SileroActivity('')
    voice = np.full(25600, .1, np.float32)
    a.analyze(voice, 0)
    assert a.analyze(voice, 0)['vad_new_frames'] == 0
    assert models[0].calls == 50
    out = b.analyze(np.zeros(25600, np.float32), 0)
    assert out['is_silence'] and out['vad_speech_ms'] == 0
    assert models[0] is not models[1]
    partial = a.analyze(np.full(1601, .1, np.float32), 25600)
    assert partial['vad_speech_samples'] == 1601
    assert partial['vad_new_frames'] == 4
    assert a.end_sample == 27201


def test_silero_hysteresis_and_tail_padding(monkeypatch):
    torch = pytest.importorskip('torch')
    class Model:
        def eval(self): return self
        def reset_states(self): self.values = iter([.6, .4] + [.2]*6)
        def __call__(self, *args): return torch.tensor([next(self.values)])
    detector = SileroActivity('', model=Model(), pad_ms=160)
    states = [detector.analyze(np.zeros(512, np.float32), i*512)['is_silence'] for i in range(8)]
    assert states == [False]*7 + [True]


@pytest.mark.asyncio
async def test_off_mode_forwards_silence_and_keeps_wait_fallback():
    calls = []
    def response(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={'choices': [{'message': {'content': ''}, 'finish_reason': 'stop'}]})
    async with httpx.AsyncClient(transport=model_transport(response)) as client:
        session = VLLMSession(Model(), Settings(vad_mode='off'), 'target_only', 'zh-en', http=client)
        events = [e async for e in session.push(np.zeros(102400, np.float32))]
    assert len(calls) == 5
    assert all(not e['is_silence'] and e['model_invoked'] for e in events)
    assert events[-1]['st_audio_ms'] == 6400


@pytest.mark.asyncio
async def test_silero_keeps_state_across_window_and_counts_actual_eof(monkeypatch):
    models = fake_vad(monkeypatch)
    client = httpx.AsyncClient(transport=model_transport(lambda _: httpx.Response(200,
        json={'choices': [{'message': {'content': 'translated'}, 'finish_reason': 'stop'}]})))
    session = VLLMSession(Model(), Settings(vad_mode='silero', window_seconds=1.6),
                          'target_only', 'zh-en', http=client)
    try:
        waveform = np.concatenate([np.zeros(25600, np.float32), np.full(27201, .1, np.float32)])
        events = [e async for e in session.push(waveform)]
        events += [e async for e in session.finish()]
        assert events[0]['silence_skipped'] and not events[0]['model_invoked']
        assert events[1]['window_reset']
        assert len(models) == 1 and models[0].calls == 104
        assert events[-2]['vad_speech_samples'] == 1601
        assert events[-1]['duration_ms'] == len(waveform)//16
        assert events[-1]['skipped_untranslated_audio_ms'] == 0
    finally:
        await session.close()
        await client.aclose()


@pytest.mark.parametrize('transport', ['stream', 'ws'])
def test_silero_shared_path_for_upload_and_fragmented_ws(monkeypatch, transport):
    models = fake_vad(monkeypatch)
    async def generate(self, *args, **kwargs): return Reply('', 'hello', 'hello', [], None)
    monkeypatch.setattr('simulinfer.translation.ChatAPI.generate', generate)
    async def budget(self, *args, **kwargs):
        from simulinfer.budget import PromptBudget
        return PromptBudget(100,8192,1)
    monkeypatch.setattr('simulinfer.translation.ChatAPI.budget', budget)
    samples = np.concatenate([np.zeros(25600, np.float32), np.full(27201, .1, np.float32)])
    with TestClient(build_app(Settings(vad_mode='silero'),preflight_result={'models':[]})) as client:
        if transport == 'stream':
            r = client.post('/translate/stream', files={'file': ('test.wav', float_wav(samples), 'audio/wav')})
            events = [json.loads(line[5:]) for line in r.text.splitlines() if line.startswith('data:')]
        else:
            with client.websocket_connect('/translate/ws') as ws:
                ws.send_json({'input_format': 'f32le'})
                events = [ws.receive_json()]
                raw = samples.astype('<f4').tobytes()
                for pos in range(0, len(raw), 6399): ws.send_bytes(raw[pos:pos+6399])
                ws.send_json({'eof': True})
                while events[-1]['event'] not in ('done', 'error'): events.append(ws.receive_json())
        assert 'vad_mode' not in events[0] and 'vad_cpu_threads' not in events[0]
        assert events[-1]['event'] == 'done', events
        chunks = [e for e in events if e['event'] == 'chunk']
        assert len(chunks) == 3 and chunks[0]['action']=='WAIT'
        assert 'vad_speech_samples' not in chunks[-1]
        assert len(models) == 1 and models[0].calls == 104


def test_missing_silero_is_reported_before_ready(monkeypatch):
    def missing(_): raise RuntimeError('missing Silero model')
    monkeypatch.setattr('simulinfer.translation.create_activity', missing)
    with TestClient(build_app(Settings(vad_mode='silero'),preflight_result={'models':[]})) as client:
        with client.websocket_connect('/translate/ws') as ws:
            ws.send_json({})
            event = ws.receive_json()
            assert event['event'] == 'error' and 'missing Silero' in event['detail']
        assert client.get('/health').json()['active_sessions'] == 0
