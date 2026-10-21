#!/bin/bash
# T15: full inference in the order 05 -> 07 -> 06; each stops the chain on failure.
set -e
# Project root = parent of v1/ (this script lives in v1/scripts/); all paths below are relative to it.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/v1/work/logs"
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=v1/.venv/bin/python
for s in 05_predict 07_stream 06_perturb; do
  echo "$(date '+%F %T') start $s"
  $PY v1/python/$s.py > v1/work/logs/t15_$s.log 2>&1
  echo "$(date '+%F %T') done  $s"
done
