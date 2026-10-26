#!/usr/bin/env bash
set -euo pipefail
infer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
deploy_python="${SIMULINFER_PYTHON:-python3}"
if ! "$deploy_python" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))'; then
  echo "Use Python 3.12+ (for example: SIMULINFER_PYTHON=python3.12 bash deploy.sh ...)." >&2
  exit 1
fi
exec "$deploy_python" "$infer_root/simulinfer/deploy.py" "$@"
