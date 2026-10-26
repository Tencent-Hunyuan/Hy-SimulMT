# SimulInfer

**English** | [简体中文](README.zh-CN.md) | [Hy-SimulMT](../README.md)

Use Hy-SimulMT for speech translation through a web page, audio file uploads or
real-time WebSocket. Choose a translation-only model or a model that returns
both source text and translation. TTS speech output is optional.

For installation through a single deployment command, start with the
[project quick start](../README.md#deploy-inference). This guide covers manual
model deployment, separate machines, client options and TTS.

From the repository root, enter `SimulInfer/` and start the deployment:

```bash
cd SimulInfer
bash deploy.sh --install-server --model tencent/Hy-SimulMT-30B-A3B-Interleave
```

For a saved configuration, copy `deploy.example.toml` to `deploy.toml` in this directory.

[Deployment script options](docs/DEPLOYMENT.md)

[Configuration](docs/CONFIGURATION.md) · [Audio input](docs/AUDIO_INPUT.en.md) · [TTS](docs/TTS.en.md)

## Requirements

- SimulInfer: Python 3.11+. Install the `[vad]` extra for the default audio features.
  SimulInfer can run on a CPU machine separate from the model server.
- Model server: vLLM >=0.31 with `HYV3ForCausalLM` support, plus compatible
  PyTorch, CUDA, GPU drivers and audio dependencies.
- A Hy-SimulMT model.

## llama.cpp

> Hy-SimulMT support has not yet been merged into the official llama.cpp repository. Support is expected soon; we recommend waiting for a new llama.cpp release that supports this model before using it.

Use this option when an audio-capable llama.cpp server has already loaded your
Hy-SimulMT GGUF model on port **8080**. In a Python 3.11+ environment, run the
following from the repository root:

```bash
cd SimulInfer
python -m pip install .
export MODEL=/path/to/hymt-A3b-tgt_only-1019-Q8_0.gguf
MODEL_TYPE=target_only python serve_llamacpp.py
```

Replace `MODEL` with your own GGUF path, exactly as loaded by the llama.cpp server.
Use `MODEL_TYPE=target_only` for a translation-only model, or `source_target` for
a model that outputs both source text and translation.

For `serve_llamacpp.py`, open **http://127.0.0.1:8094/** for the web page.
This is the bridge web/API port; **8080** is the default llama.cpp model port.
The separate `deploy.sh` / `simulinfer` launch path defaults to web/API port **8080**.
For the llama.cpp bridge, file uploads and WebSocket use
`/translate/stream` and `/translate/ws` on port **8094**. The script connects to
`http://127.0.0.1:8080` by default; set `LLAMA_URL` for a different model server
and `PORT` for a different web/API port. For remote microphone access, use an
HTTPS port-forwarding address.

## Manual model deployment

In an existing compatible vLLM environment:

```bash
git clone https://github.com/Tencent-Hunyuan/Hy-SimulMT.git Hy-SimulMT
cd Hy-SimulMT/SimulInfer
```

Install the model plugin:

```bash
python -m pip install --no-deps --index-url https://pypi.org/simple/ simulst
```

For a new Python 3.12 environment, install vLLM and audio dependencies together:

```bash
python -m pip install --index-url https://pypi.org/simple/ 'simulst[server]'
```

Use the first command for a custom vLLM or CUDA build. If audio dependencies
are missing, install `simulst[audio]` in the model environment.

In the model environment, install SimulInfer and prepare the model directory once:

```bash
python -m pip install --no-deps .
MODEL_ID="tencent/Hy-SimulMT-30B-A3B-Interleave"
MODEL_DIR="./models/hy-simulmt"
hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
simulinfer-prepare-model --model "$MODEL_DIR"
```

For an existing local model, set `MODEL_DIR` to its directory and skip the download.
Preparation writes the current inference template and backs up the replaced files.
Run it once before starting vLLM:

```bash
vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 --port 8800 --dtype bfloat16 \
  --max-num-seqs 4 --gpu-memory-utilization 0.9 \
  --enable-prefix-caching --enable-chunked-prefill \
  --limit-mm-per-prompt '{"audio":1024}' --chat-template-content-format openai
```

Use that same `MODEL_DIR` in both commands. Adjust the resource settings to your
GPU. After preparing an already deployed model, restart its vLLM service.

## Install and start SimulInfer

```bash
git clone https://github.com/Tencent-Hunyuan/Hy-SimulMT.git Hy-SimulMT
cd Hy-SimulMT/SimulInfer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.2' --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[vad]'
simulinfer --init-config config.toml
```

Set the model service address in `config.toml`:

```toml
[model]
base_url = "http://model-host:8800/v1"
```

```bash
simulinfer --config config.toml
```

Open `http://127.0.0.1:8080/`. For access from another machine, start with
`--host 0.0.0.0`. A wheel is also available in `releases/0.5.9/`.

## Use the web page

1. Select the translation direction.
2. Upload an audio file, or click **Start real-time conversation** and allow microphone access.
3. Click **Stop and finish** when you finish speaking.
4. Enable speech output if a TTS service is configured.

The page automatically uses the deployed model and shows source text when that
model provides it. No output-mode selection is needed. After changing the model,
restart SimulInfer and refresh the page.

Remote microphone access requires HTTPS. When using a forwarded port, open the
complete address, such as `https://your-host/proxy/8080/`. The proxy must support
WebSocket connections. See [configuration](docs/CONFIGURATION.md) for server settings.

## Input cadence

| Option | Setting | Use |
|---|---|---|
| Standard — recommended and default | `standard` | 1600ms blocks for everyday use |
| Short-chunk | `short_chunk` | **First block must be 1600ms**; subsequent blocks 800ms |

**The first input block must be 1600ms in both modes.** Only subsequent blocks
can use 800ms. Configure the cadence in `config.toml`; the web page follows
the service configuration and does not offer an 800ms selector:

```toml
[defaults]
response_mode = "standard"
```

## Standard and noise-resistant audio modes

- **Standard mode**: enabled by default for everyday use.
- **Noise-resistant mode**: intended for noisy environments to help reduce
  noise interference with translation.

Both modes support file uploads and real-time input. To create a noise-resistant configuration:

```bash
simulinfer --init-config noise.toml --profile noise_resistant
# Set the model service address in noise.toml, then start.
simulinfer --config noise.toml
```

For an existing configuration, change this setting and restart SimulInfer:

```toml
[inference]
audio_mode = "noise_resistant"
```

Use `audio_mode = "standard"` to return to standard mode.

## File upload and WebSocket

Upload a file:

```bash
curl -N http://127.0.0.1:8080/translate/stream \
  -F direction=zh-en -F file=@audio.wav
```

Use the real-time client:

```bash
simulinfer-client audio.wav --url http://127.0.0.1:8080 \
  --transport ws --response-mode standard --realtime
```

For a custom client, connect to `ws://127.0.0.1:8080/translate/ws` and send:

```json
{"direction":"zh-en","input_format":"s16le","input_sample_rate":48000,"input_channels":1}
```

Wait for `ready`, then send binary PCM matching the declared format. Send
`{"eof":true}` when finished and read responses until `done`. See
[audio input](docs/AUDIO_INPUT.en.md) for accepted formats and [configuration](docs/CONFIGURATION.md)
for request options. Interactive API documentation is available at `/docs`.

## Optional TTS

Deploy Qwen3-TTS with vLLM-Omni in a separate model environment:

```bash
python -m pip install --no-deps /path/to/simulinfer-0.5.9-py3-none-any.whl
simulinfer-tts-omni --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --host 0.0.0.0 --port 8091
```

Add its connection settings to `config.toml`:

```toml
[tts]
tts_backend = "vllm_omni"
tts_url = "http://127.0.0.1:8091/v1"
tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
tts_task_type = "Base"
tts_clone_source = true
tts_history_mode = "paired"
tts_history_window_s = 10
```

This setup clones the voice from your uploaded audio or microphone input.
No reference file or transcript is required.

Restart SimulInfer, then enable speech output on the page or set `tts=true`
on a request. This connects directly to the deployed TTS service; Docker is
optional. See [TTS setup](docs/TTS.en.md) for details.
