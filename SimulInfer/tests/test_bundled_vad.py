import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simulinfer.config import Settings
from simulinfer.vad import create_activity, resolve_model_path


def test_bundled_weights_license_and_parameter_manifest():
    model = Path(resolve_model_path(''))
    manifest = json.loads((model.parent/'manifest.json').read_text())
    assert manifest['version'] == '6.2.3' and manifest['license'] == 'MIT'
    for name, info in manifest['files'].items():
        raw = (model.parent/name).read_bytes()
        assert len(raw) == info['bytes']
        assert hashlib.sha256(raw).hexdigest() == info['sha256']
    settings = Settings(vad_mode="silero",vad_threshold=.5,vad_pad_ms=160)
    for key, value in manifest['harness_defaults'].items():
        assert getattr(settings, key) == value
    assert manifest['harness_frame_samples'] == 512
    assert 'Silero Team' in (model.parent/'LICENSE').read_text()


def test_real_bundled_vad_needs_no_silero_distribution_or_network(monkeypatch):
    pytest.importorskip('torch')
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith('silero_vad'):
            raise AssertionError('bundled VAD must not import the Silero distribution')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    first, second = create_activity(Settings(vad_mode="silero",vad_threshold=.5,vad_pad_ms=160)), create_activity(Settings(vad_mode="silero",vad_threshold=.5,vad_pad_ms=160))
    a = first.analyze(np.zeros(25600, np.float32), 0)
    b = second.analyze(np.zeros(12800, np.float32), 0)
    assert a['is_silence'] and b['is_silence']
    assert first.model is not second.model
    assert a['vad_new_frames'] == 50 and b['vad_new_frames'] == 25
