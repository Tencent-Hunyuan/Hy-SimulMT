#!/usr/bin/env bash
# ============================================================
# 一键安装
# ============================================================
#
#   bash install.sh
#
# 做的事：
#   1. 检查 Python 与已装好的 torch
#   2. 建 .venv（带 --system-site-packages，复用系统里已编译好的
#      torch / flash-attn，避免重复下载约 2GB 的 CUDA wheel）
#   3. 装本框架依赖 + vendor/LlamaFactory
#   4. 跑一遍导入自检
#
# 环境变量：
#   PYTHON=/path/to/python   指定用哪个解释器建环境（默认 python3）
#   NO_VENV=1                不建.venv，直接装进当前环境
#   PIP_INDEX_URL=...        换国内源，例如
#                            PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
#
# 在SimulTrain目录执行；指定PYTHON可选择已有的GPU环境。
# ============================================================

set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# This repository supports one line only: the merged A3B speech export.
VENV_NAME=".venv"
PYTHON="${PYTHON:-python3}"
echo "==> 目标环境: $VENV_NAME（merged A3B）"

if ! command -v "$PYTHON" >/dev/null && [[ ! -x "$PYTHON" ]]; then
  echo "找不到 Python：$PYTHON（可用 PYTHON=/path/to/python bash install.sh 指定）" >&2
  exit 1
fi

echo "==> 解释器: $("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# ---------- 1. torch 必须先由用户自己装好 ----------
# torch/torchaudio/flash-attn 与 CUDA 版本强绑定，让 pip 自己解会拉到不匹配的
# wheel（典型症状：torchaudio undefined symbol），所以这里只检查不安装。
if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
  cat >&2 <<'EOF'
未检测到 torch。请先按你的CUDA 版本手动安装，例如：

  # CUDA 12.8
  pip install torch --index-url https://download.pytorch.org/whl/cu128

装好后重新执行 bash install.sh。
EOF
  exit 1
fi

TORCH_INFO="$("$PYTHON" - <<'PY'
import torch
print(f"{torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
PY
)"
echo "==> torch: $TORCH_INFO"

# ---------- 2. 建 venv ----------
if [[ "${NO_VENV:-0}" == "1" ]]; then
  echo "==> NO_VENV=1，直接装进当前环境"
  PIP=("$PYTHON" -m pip)
else
  if [[ ! -x "$ROOT_DIR/$VENV_NAME/bin/python" ]]; then
    echo "==> 创建 $VENV_NAME（--system-site-packages，复用系统 torch / flash-attn）"
    "$PYTHON" -m venv --system-site-packages "$ROOT_DIR/$VENV_NAME"
  else
    echo "==> 复用已有的 $VENV_NAME"
  fi
  PIP=("$ROOT_DIR/$VENV_NAME/bin/python" -m pip)
fi

"${PIP[@]}" install --upgrade pip setuptools wheel

# ---------- 3. 装依赖 ----------
# 关键：--no-deps 装 LlamaFactory，避免它把 torch/transformers 重新解一遍版本
echo "==> 安装 merged A3B 依赖（requirements.txt）"
"${PIP[@]}" install -r "$ROOT_DIR/requirements.txt"

echo "==> 安装本训练包（editable, --no-deps）"
"${PIP[@]}" install -e "$ROOT_DIR" --no-deps

echo "==> 安装 vendor/LlamaFactory（--no-deps，防止覆盖已装好的 torch）"
if [[ -d "$ROOT_DIR/vendor/LlamaFactory" ]]; then
  "${PIP[@]}" install -e "$ROOT_DIR/vendor/LlamaFactory" --no-deps
else
  echo "缺少 vendor/LlamaFactory，仓库不完整" >&2
  exit 1
fi

# ---------- 4. 自检 ----------
echo "==> 自检"
if [[ "${NO_VENV:-0}" == "1" ]]; then
  CHECK_PY="$PYTHON"
else
  CHECK_PY="$ROOT_DIR/$VENV_NAME/bin/python"
fi

"$CHECK_PY" - <<'PY'
import importlib

mods = ["torch", "torchaudio", "transformers", "accelerate", "datasets", "peft", "llamafactory", "yaml"]
for name in mods:
    m = importlib.import_module(name)
    print(f"  {name:14s} {getattr(m, '__version__', 'n/a')}")

# 变体注册链路是最容易出问题的一环，这里实际走一遍
from s2s_train.variants import load_variant

variant = load_variant("hy_v3_speech_merged")
print(f"  变体注册       ok (model_type={variant.MODEL_TYPE})")

# 音频解码依赖 imageio-ffmpeg 自带的二进制，确认能找到
import imageio_ffmpeg

print(f"  ffmpeg         {imageio_ffmpeg.get_ffmpeg_exe()}")
PY

cat <<'EOF'

==> 安装完成

下一步：
  1. cp .env.example .env
  2. python scripts/preflight.py /path/to/A3b_src_tgt_1019_bf16
  3. bash train_merged.sh
EOF
