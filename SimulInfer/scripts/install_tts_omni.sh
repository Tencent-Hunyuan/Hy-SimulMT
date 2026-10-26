#!/usr/bin/env bash
set -euo pipefail
# Keep the TTS GPU environment separate from the translation model and SimulInfer.
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate a new TTS virtualenv/conda environment first." >&2
  exit 1
fi
# Pick the pair documented by the Omni release. Prerelease suffixes need not
# match: for example Omni 0.31.0rc1 targets vLLM 0.31.0.
release="${VLLM_OMNI_VERSION:-}"
if [[ -z "$release" || -z "${VLLM_VERSION:-}" ]]; then
  echo "Set VLLM_OMNI_VERSION and VLLM_VERSION to a compatible pair with Qwen TTS support." >&2
  exit 1
fi
engine_release="${VLLM_VERSION:-$release}"
python -m pip install "vllm==$engine_release" "vllm-omni==$release"
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip install "$package_dir"
python -c 'from simulinfer.tts_launch import check_environment; print(check_environment(clone_source=True))'
