#!/usr/bin/env bash
set -euo pipefail
exec python -m simulinfer.tts_backends.qwen "$@"
