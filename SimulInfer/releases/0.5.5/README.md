# SimulInfer 0.5.5

For a full deployment from the repository root, follow the
[quick start](../../../README.md#deploy-inference) or [中文快速部署](../../../README.zh-CN.md#部署推理服务).

To install only the inference service in a prepared Python 3.11+ environment:

```bash
python -m pip install 'torch>=2.2' --index-url https://download.pytorch.org/whl/cpu
python -m pip install './simulinfer-0.5.5-py3-none-any.whl[vad]'
simulinfer --init-config config.toml
# Set the model service URL, then start.
simulinfer --config config.toml
```

Use `simulinfer-client audio.wav --url http://127.0.0.1:8080` to send a file.
Model deployment and TTS setup are covered in the [detailed guide](../../README.md).
