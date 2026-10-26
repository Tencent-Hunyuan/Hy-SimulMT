# Audio input

## Upload a file

Use the web page or upload WAV, FLAC, MP3, M4A/AAC or WebM/Opus:

```bash
curl -N http://127.0.0.1:8080/translate/stream \
  -F direction=zh-en -F file=@audio.wav
```

Supported input ranges are 8–192kHz and 1–8 channels. The service converts audio
automatically; you do not need to prepare 16kHz mono files. File size is limited
by `[server] max_upload_mb` (100MB by default).

## Use the microphone

Open the page, select a translation direction, click **Start real-time
conversation**, and allow microphone access. Click **Stop and finish** when done.
Use HTTPS when accessing the page from another machine.

## Use the command-line client

```bash
simulinfer-client audio.wav --url http://127.0.0.1:8080 \
  --transport ws --response-mode standard --realtime
```

Use `--transport stream` for a file upload. Add `--tts` to request speech output,
`--tts-output translated.wav` to save it, and `--events events.json` to save responses.

## Connect a WebSocket client

1. Connect to `ws://127.0.0.1:8080/translate/ws` (use `wss://` for HTTPS).
2. Send the input settings:

```json
{"direction":"zh-en","input_format":"s16le","input_sample_rate":48000,"input_channels":2}
```

3. Wait for the `ready` event.
4. Send binary PCM frames matching the declared sample rate and channel count.
5. Send `{"eof":true}` when finished.
6. Read responses until `done`.

`input_format` accepts `s16le` (little-endian PCM16) or `f32le` (little-endian
float32 PCM). Multichannel samples must be interleaved. A 100ms packet of 48kHz
stereo PCM16 is 19200 bytes. Do not send a WAV or MP3 file as raw PCM; use the
file-upload endpoint instead. When omitted, the format defaults to 16kHz mono PCM16.

Responses include `ready`, `chunk`, `flush` and `done`, or `error` on failure.
Translation is in `text` and the final `full_text`; models that provide source
text also return `source_text`. Final `complete=false` means some audio could
not be translated; `unresolved_audio_ms` and `skipped_untranslated_audio_ms`
report its duration. These events contain results and client progress only.

To receive TTS, set `tts=true`. Use `audio_transport="base64"` for JSON audio
with a `data` field, or `"binary"` (default for WS) for an `audio` metadata event
followed by a binary audio frame. Play the audio using its declared sample rate.
