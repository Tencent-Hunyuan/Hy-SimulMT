#!/usr/bin/env bash
set -euo pipefail
# CUDA_VISIBLE_DEVICES在调用此脚本之前设置；剩余参数由launch透传给vLLM。
exec python -m simulinfer.launch "$@"
