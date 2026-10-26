#!/usr/bin/env bash
# 被 train.sh / infer.sh source，负责统一解决三件事：
#   1. 读取仓库根目录下的 .env（若存在），用于配置模型与数据路径
#   2. 挑选 Python 解释器：优先 .venv，其次当前环境的 python3
#   3. 保证 s2s_train 与 llamafactory 可被导入（没 pip install 也能跑）
#
# 依赖调用方先定义 ROOT_DIR（仓库根目录）。

if [[ -z "${ROOT_DIR:-}" ]]; then
  echo "bootstrap.sh 需要调用方先设置 ROOT_DIR" >&2
  exit 1
fi

# ---------- 1. 读取 .env ----------
# .env 只放路径这类跟机器相关的配置，已被 .gitignore 排除。
# 里面的写法是 KEY=value；已存在的同名环境变量优先（不会被覆盖）。
if [[ -f "$ROOT_DIR/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%$'\r'}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key// /}"
    # 去掉值两端可能出现的引号
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
    # 命令行/环境里已经给过就不覆盖。
    # 注意用 -v 判断「是否已定义」而不是「是否非空」：
    # MEDIA_DIR= bash train_merged.sh 传的是空字符串，语义是「显式关闭」，
    # 若按非空判断就会被 .env 里的值覆盖，导致关不掉。
    if [[ ! -v "$key" ]]; then
      export "$key=$value"
    fi
  done <"$ROOT_DIR/.env"
fi

# ---------- 2. 选Python ----------
# S2S_PYTHON 可强制指定解释器；否则优先用仓库内的 .venv
if [[ -n "${S2S_PYTHON:-}" ]]; then
  PYTHON_BIN="$S2S_PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi

if [[ -z "$PYTHON_BIN" ]] || ! "$PYTHON_BIN" -c '' 2>/dev/null; then
  echo "找不到可用的 Python。先执行 bash install.sh，或用 S2S_PYTHON=/path/to/python 指定。" >&2
  exit 1
fi
export PYTHON_BIN

# ---------- 3. 组装 PYTHONPATH ----------
# Always prefer the source tree beside this runtime snapshot. Editable installs
# may point back to the shared Ceph checkout; allowing that would reintroduce
# the multi-rank FUSE import stall that sync_runtime.sh is designed to avoid.
VENDOR_LF_SRC="$ROOT_DIR/vendor/LlamaFactory/src"
export PYTHONPATH="$VENDOR_LF_SRC:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PYTHON_BIN" -c 'import llamafactory' 2>/dev/null; then
  echo "无法导入 llamafactory。先执行 bash install.sh 安装依赖。" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
