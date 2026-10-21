#!/bin/bash
# T16: real-data R statistics in lock order; stops on the first failure.
set -e
# Project root = parent of v1/ (this script lives in v1/scripts/); all paths below are relative to it.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/v1/work/logs"
for s in 03_association 04b_external_eval 05_transfer 06_perturb 07_stream 08_sensitivity; do
  echo "$(date '+%F %T') start $s"
  Rscript v1/R/$s.R > v1/work/logs/r_$s.log 2>&1
  echo "$(date '+%F %T') done  $s"
done
