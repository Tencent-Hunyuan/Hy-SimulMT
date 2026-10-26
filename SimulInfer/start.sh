#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config.toml}"
exec simulinfer --config "$CONFIG_PATH"
