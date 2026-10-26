# Silero VAD 6.2.3

Optional Silero weights. See `LICENSE` for the MIT license and `manifest.json`
for source information.

To use the Silero configuration:

```bash
python -m pip install '.[vad-silero]'
cp configs/silero.toml config.toml
# Set the model service URL in config.toml, then start.
simulinfer --config config.toml
```

In this configuration, an empty `vad_model_path` selects the bundled weights.
