#!/bin/bash
# Gate G2 pilot: fold-0, both architectures, 20-epoch cosine, inner-validation split (T8 --pilot)
# Project root = parent of v1/ (this script lives in v1/scripts/); all paths below are relative to it.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/v1/work/logs"
for m in unet segformer; do
  echo "$(date '+%F %T') start $m" >> v1/work/logs/pilot_run.log
  caffeinate -i v1/.venv/bin/python v1/python/04_train.py --model $m --fold 0 --epochs 20 --pilot > v1/work/logs/pilot_$m.log 2>&1
  echo "$(date '+%F %T') end $m exit=$?" >> v1/work/logs/pilot_run.log
done
