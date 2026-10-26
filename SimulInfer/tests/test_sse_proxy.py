"""SSE must remain flushable through proxies while waiting for inference."""
import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import simulinfer.app as app_module
from simulinfer.audio import float_wav
from simulinfer.config import load


@pytest.mark.parametrize('failure', [False, True])
def test_sse_keepalives_preserve_events_and_disable_proxy_compression(monkeypatch, failure):
    async def delayed_translation(settings, mode, direction, source, emit, *args):
        # Simulate waiting for a slot/model before ready, then a delayed turn.
        await asyncio.sleep(.04)
        await emit({'event': 'ready', 'model_type': mode, 'direction': direction})
        await asyncio.sleep(.04)
        if failure:
            raise RuntimeError('inference failed')
        await emit({'event': 'done', 'full_text': 'translated', 'complete': True,
                    'duration_ms': 1600, 'num_chunks': 1})

    monkeypatch.setattr(app_module, 'translate', delayed_translation)
    monkeypatch.setattr(app_module, 'SSE_KEEPALIVE_SECONDS', .01)
    app = app_module.build_app(load(mock=True))
    with TestClient(app) as client:
        response = client.post('/translate/stream',
            headers={'Accept-Encoding': 'gzip, br'},
            files={'file': ('sample.wav', float_wav(np.zeros(25600, np.float32)), 'audio/wav')})
        assert response.status_code == 200
        assert response.headers['cache-control'] == 'no-cache, no-transform'
        assert response.headers['x-accel-buffering'] == 'no'
        assert 'content-encoding' not in response.headers
        frames = response.text.strip().split('\n\n')
        assert frames[0] == ': connected'
        assert ': keep-alive' in frames
        events = [json.loads(f[6:]) for f in frames if f.startswith('data: ')]
        assert [e['event'] for e in events] == ['ready', 'error' if failure else 'done']
        if failure:
            assert events[-1]['detail'] == 'inference failed'
        else:
            assert events[-1]['full_text'] == 'translated' and events[-1]['complete']
        assert client.get('/health').json()['active_sessions'] == 0
