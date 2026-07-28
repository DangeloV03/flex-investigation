#!/usr/bin/env bash
# Reset and rerun critical-region homo Δμ=1 coex combos:
#   ε = -1.775, -1.780, -1.785, -1.790, -1.800 (NaN fit)
#       -1.765 (bad fit)
#
# Run from the flex-investigation repo root on the workstation.
#
#   ./coex/rerun_critical_homo_dmu1p0.sh --dry-run   # preview
#   ./coex/rerun_critical_homo_dmu1p0.sh             # reset + enqueue 60 jobs
#
# Stop tmux first if dispatcher/analyzer are mid-cycle on these ε:
#   tmux kill-session -t coex-dmu1p0
# Then rerun this script and restart daemons (or leave them stopped and start fresh).

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR"
export PYTHONUNBUFFERED=1

python -u coex/rerun_combos.py \
  --manage coex_manage_homo_dmu1p0.csv \
  --manifest coex_homo_dmu1p0_queue.json \
  --samples coex_samples/homo_dmu1p0 \
  --results susceptibility/results/coex_homo_dmu1p0 \
  "$@"
