#!/usr/bin/env bash
# Mentor item 4 on Della, on the CURRENT coex data (the preview_* folder used the 2026-10-16 snapshot).
#   bash experiments/2026-11-10_flex_scheme_check/run.sh            # all COEX_RUNS_* -> FLEX_SCHEME_CHECK/
#   bash experiments/2026-11-10_flex_scheme_check/run.sh eps_c      # item 2 table (no data needed)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/WJACOBS/vd7294/flex-investigation}"
OUT="${OUT:-$DATA_ROOT/FLEX_SCHEME_CHECK}"
module load anaconda3/2024.10 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FLEX_CONDA_ENV:-lattice}"
case "${1:-check}" in
  check) python -W ignore "$HERE/flex_scheme_check.py" --runs-root "$DATA_ROOT" --outdir "$OUT" ;;
  eps_c) python -W ignore "$HERE/eps_c_shift.py" --outdir "$OUT" ;;
  *) sed -n '2,4p' "$0"; exit 2 ;;
esac
