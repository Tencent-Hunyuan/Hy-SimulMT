#!/usr/bin/env bash
# SimulTrain launcher. Usage:
#   bash train.sh merged_a3b_sft 0,1 merged_model_path=/path/to/model \
#     dataset_dir=/path/to/data dataset=my_dataset output_dir=/path/to/output
# Environment: GPUS/GPU, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT,
# MIN_FREE_MIB, ALLOW_BUSY_GPU, S2S_PYTHON.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 读 .env、选解释器、把 s2s_train 与 vendor 里的 llamafactory 挂进 PYTHONPATH
source "$ROOT_DIR/scripts/bootstrap.sh"

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <配置> [GPU列表] [key=value ...]" >&2
  echo "可用配置:" >&2
  for f in "$ROOT_DIR"/configs/*.yaml; do
    echo "  $(basename "$f" .yaml)" >&2
  done
  exit 2
fi

# ---------- 解析配置 ----------
CONFIG="$1"
shift
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$ROOT_DIR/configs/${CONFIG%.yaml}.yaml"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "配置文件不存在：$CONFIG" >&2
  exit 1
fi

# 第二个位置参数若是 GPU 编号列表则取用，否则沿用 GPUS/GPU/GPU_ID 环境变量
# （GPU 通常来自 .env）
if [[ "${1:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  GPUS="$1"
  shift
fi
GPUS="${GPUS:-${GPU:-${GPU_ID:-}}}"
if [[ -z "$GPUS" ]]; then
  echo "请指定 GPU：$0 <配置> 3   或   GPUS=0,1 $0 <配置>   或在 .env 里配 GPU=" >&2
  exit 2
fi
if [[ ! "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU 列表格式错误：$GPUS（应形如 3 或 2,3,6,7）" >&2
  exit 2
fi

# ---------- 从配置+命令行推断阶段描述与显存需求 ----------
# 合并逻辑与 launcher 保持一致：命令行覆盖 YAML
IFS=$'\t' read -r STAGE_LABEL AUTO_MIN_FREE OUTPUT_DIR <<<"$(
  "$PYTHON_BIN" - "$CONFIG" "$@" <<'PY'
import sys, yaml
from pathlib import Path

# PYTHONPATH 已由 bootstrap.sh 设好，可直接导入
from s2s_train.launcher import _parse_value

config_path = Path(sys.argv[1])
raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
for item in sys.argv[2:]:
    if "=" in item:
        key, value = item.split("=", 1)
        # 与 launcher 用同一套解析，避免两处推断出的结果不一致
        raw[key.strip()] = _parse_value(value)

variant = raw.get("variant", "unknown")
ft = str(raw.get("finetuning_type", "full"))
frozen_llm = bool(raw.get("freeze_language_model", False))
frozen_adapter = bool(raw.get("freeze_multi_modal_projector", False))

if frozen_llm:
    trainable, min_free = "仅 Adapter", 30000
elif frozen_adapter:
    # Adapter 冻结但 LLM 全参在训，显存需求与全参同级（1.8B 的优化器状态照样要）
    trainable, min_free = "仅 LLM 全参（Adapter 冻结）", 40000
elif ft == "full":
    trainable, min_free = "Adapter + LLM 全参", 40000
else:
    trainable, min_free = f"Adapter + LLM {ft.upper()}", 30000

label = f"{config_path.stem}  variant={variant}  可训练={trainable}"
# 用制表符分隔，避免中文标签里的空格被 shell 拆错字段
print("\t".join([label, str(min_free), str(raw.get("output_dir", ""))]))
PY
)"

MIN_FREE_MIB="${MIN_FREE_MIB:-$AUTO_MIN_FREE}"

if [[ -z "$OUTPUT_DIR" || "$OUTPUT_DIR" == "PLEASE_OVERRIDE" ]]; then
  echo "必须提供 output_dir=<路径>（本包不在自身目录下产生输出）" >&2
  exit 2
fi

# ---------- 资源检查 ----------
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC=$(awk -F, '{print NF}' <<<"$GPUS")

for id in ${GPUS//,/ }; do
  IFS=, read -r free_mib utilization <<<"$(nvidia-smi --id="$id" \
    --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
  if (( free_mib < MIN_FREE_MIB )); then
    echo "GPU $id 空闲显存不足：${free_mib} MiB（至少需要 ${MIN_FREE_MIB} MiB）" >&2
    exit 1
  fi
  if (( utilization > 20 )) && [[ "${ALLOW_BUSY_GPU:-0}" != "1" ]]; then
    echo "GPU $id 当前利用率 ${utilization}%，为避免干扰其他任务已停止。设置 ALLOW_BUSY_GPU=1 可明确覆盖。" >&2
    exit 1
  fi
done

# ---------- 启动 ----------
# PYTHONPATH 与 TOKENIZERS_PARALLELISM 已由 bootstrap.sh 设好
mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "配置: $STAGE_LABEL"
echo "GPU: $GPUS (每节点 $NPROC 个进程)  最低显存要求: ${MIN_FREE_MIB} MiB"
echo "输出目录: $OUTPUT_DIR"
cd "$ROOT_DIR"

if (( NPROC == 1 && NNODES == 1 )); then
  "$PYTHON_BIN" -m s2s_train.launcher "$CONFIG" "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"
else
  # 注意：不能用 llamafactory-cli，它会重新拉起自己的 torchrun 而绕过本框架注册
  export TORCH_NCCL_AVOID_RECORD_STREAMS=1
  echo "分布式: nnodes=$NNODES node_rank=$NODE_RANK"
  "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --nproc_per_node "$NPROC" \
    --master_addr "${MASTER_ADDR:-127.0.0.1}" \
    --master_port "${MASTER_PORT:-29500}" \
    -m s2s_train.launcher "$CONFIG" "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"
fi
