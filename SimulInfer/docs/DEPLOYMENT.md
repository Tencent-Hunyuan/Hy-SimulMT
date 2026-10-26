# Deployment script options

The script requires Python 3.12+. From the repository root:

```bash
cd SimulInfer
bash deploy.sh --install-server --model tencent/Hy-SimulMT-30B-A3B-Interleave
```

Run the commands below from `SimulInfer/`. To save deployment settings, copy
`deploy.example.toml` to `deploy.toml` in this directory. By default, configuration
is read from `SimulInfer/deploy.toml` and runtime files are stored under
`SimulInfer/.simulinfer/`.

| Option | Use |
|---|---|
| `--model` | Hugging Face model ID or local checkpoint directory |
| `--install-server` | Create and install a vLLM model environment |
| `--server-python` | Use an existing model environment's Python |
| `--backend-url` | Connect an already running model API; starts only SimulInfer |
| `--gpus` | Model GPU IDs, such as `0` or `0,1` |
| `--tensor-parallel-size` | Number of GPUs for the model |
| `--gpu-memory-utilization` | Model memory allocation fraction; default `0.9` |
| `--model-port` | Local model API port; default `8800` |
| `--port` | SimulInfer web/API port; default `8080` |
| `--host` | SimulInfer listening address; default `0.0.0.0` |
| `--max-num-seqs` | Model scheduler concurrency; default `4` |
| `--max-sessions` | SimulInfer session concurrency; default `4` |
| `--runtime-dir` | Environments, model files, configuration and logs; default `.simulinfer` |
| `--startup-timeout` | Startup timeout in seconds; default `900` |
| `--config` | Deployment TOML file; default `deploy.toml` when present |
| `--dry-run` | Print the selected settings without changing anything |

CLI options override `[deployment]` settings. Relative paths in a deployment
configuration are resolved from its directory. Keep the runtime directory and
local source checkpoint available while using the deployment.

## Reuse environments

Activate the desired environment or set `--server-python`. Install `simulst`
there before using it:

```bash
/path/to/model-env/bin/python -m pip install --no-deps simulst
bash deploy.sh --server-python /path/to/model-env/bin/python --model /path/to/model
```

For an existing deployment, `--no-install` skips dependency installation.
`--app-python` can select an already prepared SimulInfer Python when combined
with `--no-install`.

```bash
bash deploy.sh --no-install --backend-url http://model-host:8800/v1 \
  --app-python /path/to/infer-env/bin/python
```

## TTS

Deploy a Qwen Base clone service using the [TTS guide](TTS.en.md), then add its
connection settings to `deploy.toml`:

```toml
[deployment]
backend_url = "http://model-host:8800/v1"

[tts]
tts_backend = "vllm_omni"
tts_url = "http://tts-host:8091/v1"
tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
tts_task_type = "Base"
tts_clone_source = true
tts_history_mode = "paired"
tts_history_window_s = 10
```

```bash
bash deploy.sh --config deploy.toml
```

The script connects to the configured TTS service and enables speech output.
It does not install or start a TTS model unless you separately follow the TTS guide.

## Logs and stopping

Model startup output is written to `model.log`; SimulInfer output is in `app.log`
inside the runtime directory. Press Ctrl+C in the deployment terminal to stop
the processes it started. An external model or TTS service continues running.

If the settings change, choose a new `--runtime-dir`; an existing generated
configuration is never silently replaced. If a previous launcher was forcibly
terminated, check for remaining processes before removing its `deploy.lock`.
