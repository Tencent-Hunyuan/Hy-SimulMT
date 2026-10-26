#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  cp .env.example .env
  bash train_merged.sh [key=value ...]

Required environment variables:
  MERGED_MODEL_PATH  read-only HYV3SpeechForCausalLM export
  DATASET_DIR        directory containing dataset_info.json
  DATASET            dataset name in dataset_info.json
  OUTPUT_DIR         new training output directory

Common overrides:
  GPU=0,1,2,3,4,5,6,7 BATCH_SIZE=1 GRAD_ACCUM=2 LR=1e-5
  MAX_STEPS=10 bash train_merged.sh save_strategy=no
EOF
  exit 0
fi

# Multi-rank imports from a venv on Ceph/FUSE can stall every worker.  Localize
# the installed environment once per node before bootstrap selects Python.
if [[ -z "${S2S_PYTHON:-}" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
  source "$ROOT_DIR/scripts/sync_venv.sh"
  previous_hostfile="${HOSTFILE-}"
  HOSTFILE=/dev/null s2s_sync_venv "$ROOT_DIR/.venv" "${VENV_LOCAL:-/tmp/simultrain-venv}"
  if [[ -n "$previous_hostfile" ]]; then
    export HOSTFILE="$previous_hostfile"
  else
    unset HOSTFILE
  fi
  export S2S_PYTHON="${VENV_LOCAL:-/tmp/simultrain-venv}/bin/python"
fi

source "$ROOT_DIR/scripts/bootstrap.sh"

if [[ "${A3B_RUNTIME_LOCALIZED:-0}" != "1" ]]; then
  source "$ROOT_DIR/scripts/sync_runtime.sh"
  local_runtime="${RUNTIME_LOCAL:-/tmp/simultrain-code}"
  a3b_sync_runtime "$ROOT_DIR" "$local_runtime"
  exec env A3B_RUNTIME_LOCALIZED=1 S2S_PYTHON="$PYTHON_BIN" \
    bash "$local_runtime/train_merged.sh" "$@"
fi

MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-}"
DATASET_DIR="${DATASET_DIR:-}"
MEDIA_DIR="${MEDIA_DIR:-$DATASET_DIR}"
DATASET="${DATASET:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
GPU="${GPU:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-1.0e-5}"
EPOCHS="${EPOCHS:-1.0}"
MAX_STEPS="${MAX_STEPS:-}"

missing=()
for name in MERGED_MODEL_PATH DATASET_DIR DATASET OUTPUT_DIR; do
  [[ -n "${!name}" ]] || missing+=("$name")
done
if (( ${#missing[@]} )); then
  echo "missing required variables: ${missing[*]}" >&2
  exit 2
fi
if [[ "$MERGED_MODEL_PATH" == "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR must not be the model directory" >&2
  exit 2
fi
if [[ ! "$GPU" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "invalid GPU list: $GPU" >&2
  exit 2
fi

args=(
  merged_model_path="$MERGED_MODEL_PATH"
  dataset="$DATASET"
  dataset_dir="$DATASET_DIR"
  media_dir="$MEDIA_DIR"
  output_dir="$OUTPUT_DIR"
  per_device_train_batch_size="$BATCH_SIZE"
  gradient_accumulation_steps="$GRAD_ACCUM"
  learning_rate="$LR"
  num_train_epochs="$EPOCHS"
)
[[ -n "$MAX_STEPS" ]] && args+=(max_steps="$MAX_STEPS")
args+=("$@")

exec bash "$ROOT_DIR/train.sh" merged_a3b_sft "$GPU" "${args[@]}"
