# Optional TTS connections

For Qwen3-TTS through vLLM-Omni, use the [TTS setup guide](TTS.en.md).

## OpenAI-compatible speech endpoint

```toml
[tts]
tts_backend = "openai"
tts_url = "http://tts-host:8003/v1"
tts_model = "your-tts-model"
tts_voice = "your-voice"
tts_sample_rate = 24000
```

The service must accept `/v1/audio/speech` and return raw mono PCM16.
Set the sample rate and voice to values supported by that service.

## Native Qwen adapter

Install and run in a separate GPU environment:

```bash
pip install '.[tts-qwen]'
simulinfer-tts-qwen --model /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --reference-audio /path/to/reference.wav \
  --reference-text 'The exact reference transcript.' \
  --host 0.0.0.0 --port 8003
```

Connect SimulInfer with:

```toml
[tts]
tts_backend = "session"
tts_url = "http://tts-host:8003"
tts_sample_rate = 24000
tts_history_mode = "paired"
tts_history_window_s = 10
```

Restart SimulInfer and enable speech output in the page or API request.
