# Qwen TTS setup

[简体中文](TTS.zh-CN.md)

Deploy Qwen3-TTS with vLLM-Omni, then connect SimulInfer to its service URL.
Docker is optional. Use the launcher below to enable input voice cloning.
For an existing plain vLLM-Omni endpoint, use the fixed-reference setup below.

## 1. Start the TTS service

In a separate Python environment, install compatible vLLM and vLLM-Omni
versions with Qwen3-TTS support. Follow that release's PyTorch, CUDA and driver
requirements. The clone launcher requires the Qwen TTS adapter API provided
by vLLM-Omni 0.31 or a compatible build. Install `soxr` and `soundfile` in that
environment if missing.

```bash
python -m pip install --no-deps /path/to/simulinfer-0.5.9-py3-none-any.whl
simulinfer-tts-omni --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --host 0.0.0.0 --port 8091
```

You may replace the model ID with a local model directory. For custom deployment
settings, pass your Omni configuration with `--deploy-config`.

## 2. Connect SimulInfer

Add to `config.toml`:

```toml
[tts]
tts_backend = "vllm_omni"
tts_url = "http://127.0.0.1:8091/v1"
tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
tts_task_type = "Base"
tts_sample_rate = 24000
tts_clone_source = true
tts_history_mode = "paired"
tts_history_window_s = 10
tts_speed = 1.0
```

Use the model name returned by the TTS service's `/v1/models`.
Input voice cloning uses your uploaded audio or microphone; no reference file
or transcript is needed. It starts after a short sample and refreshes every
10 seconds while retaining previous speech for continuity.

A complete example is available in `configs/tts_vllm_omni.toml`.

## 3. Enable speech output

Restart SimulInfer, then select **Clone voice and read translation** on the page, use
`--tts` in the CLI, or set `tts=true` in an API request.

```bash
simulinfer-client audio.wav --url http://127.0.0.1:8080 \
  --transport ws --tts --tts-output translated.wav
```

## Voice continuity

For Base models, `tts_history_mode="paired"` enables previous-speech reference
for subsequent segments. Adjust `tts_history_window_s` to set the approximate
history duration in seconds; use `tts_history_mode="none"` to disable it.

## Fixed reference voice

For a fixed voice, use a regular vLLM-Omni Base service and set:

```toml
[tts]
tts_backend = "vllm_omni"
tts_url = "http://tts-host:8091/v1"
tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
tts_task_type = "Base"
tts_clone_source = false
tts_reference_audio = "/path/to/reference.wav"
tts_reference_text = "The exact reference transcript."
tts_history_mode = "paired"
```

The reference file must be on the SimulInfer machine.

## Other voices and services

For a CustomVoice checkpoint, set its model name, `tts_task_type="CustomVoice"`
and a supported `tts_voice`. Remove the Base reference-audio/text fields.

Other service connection examples are in [optional TTS backends](TTS_ALTERNATIVES.en.md).
