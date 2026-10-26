# Optional container deployment

From the `SimulInfer/` directory:

```bash
docker build -t simulinfer .
cp config.example.toml config.toml
# Set reachable translation and TTS service URLs in config.toml.
docker run --rm -p 8080:8080 -v "$PWD/config.toml:/app/config.toml:ro" simulinfer
```

Open `http://127.0.0.1:8080/`. Translation and TTS models should already be deployed
as separate services. To use a reference voice file, mount its directory and
set `tts_reference_audio` to the path inside the container.
