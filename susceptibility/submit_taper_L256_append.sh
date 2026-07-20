#!/usr/bin/env bash
# Submit L=256 append jobs for the homo Δμ=1 tapered susceptibility campaign.
#
# Appends 16 replicas per ε into the EXISTING results folder (same paths as L≤128).
#
# Usage (from repo root on Della):
#   ./susceptibility/submit_taper_L256_append.sh --dry-run
#   ./susceptibility/submit_taper_L256_append.sh
#
# Override paths if needed:
#   RESULTS_BASE=susceptibility_results/homo_dmu1p0_taper_2026-07-19 \
#   MU_SOURCE=coex_manage_homo_dmu1p0.csv \
#     ./susceptibility/submit_taper_L256_append.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR"
export PYTHONUNBUFFERED=1

RESULTS_BASE="${RESULTS_BASE:-susceptibility_results/homo_dmu1p0_taper_2026-07-19}"
MU_SOURCE="${MU_SOURCE:-coex_manage_homo_dmu1p0.csv}"
SCRIPT=susceptibility/run_susceptibility_homo_dmu1p0_tapered_L256.sh

if [[ ! -d "$RESULTS_BASE" ]]; then
  echo "ERROR: results folder not found: $RESULTS_BASE" >&2
  exit 1
fi
if [[ ! -f "$MU_SOURCE" ]]; then
  echo "ERROR: mu source not found: $MU_SOURCE" >&2
  exit 1
fi
chmod +x "$SCRIPT"

mkdir -p slurm_reports

python susceptibility/sweep_susceptibility.py \
  --script "$SCRIPT" \
  --mu-source "$MU_SOURCE" \
  --scheme homo --delta-f 0.0 --delta-mu 1.0 --k 1.0 \
  --eps-min -2.0 --eps-max -1.4 --eps-step 0.005 \
  --results-base "$RESULTS_BASE" \
  --label homo_dmu1p0_taper_L256 \
  "$@"

echo ""
echo "Append target: $RESULTS_BASE/susceptibility_256x256_*_epsilon*/susceptibility_data.csv"
echo "Monitor: squeue -u \$USER -n susc_taper_L256"
