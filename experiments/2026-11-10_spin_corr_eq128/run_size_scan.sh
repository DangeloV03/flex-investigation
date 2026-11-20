#!/usr/bin/env bash
# Spin-spin correlation size scan (sheet "Correlation Length Investigation"), eps = -1.7627 and -1.65,
# 4 replicas each, same eq 1e6 / prod 2e7 / snapshot every 1000 as the L=128 runs.
#   eq     homo dmu=0, exact mu,            L = 24 32 48 64 96      -> SPIN_CORR_EQ<L>
#   neg3   negative_drive dmu=3 k=1 df=0,   L = 24 32 48 64 96 128  -> SPIN_CORR_NEG_DMU3_EQEPS_<L>
#          mu from a warm-side cubic fit of COEX_RUNS_C3D/ly40 over eps [-1.85,-1.55]
#          (SPIN_CORR_NEG_DMU3_EQEPS_fit/fit.json; the SUSC_RUNS_NEG_DMU3 fit stops at -1.78).
#
#   bash experiments/2026-11-10_spin_corr_eq128/run_size_scan.sh fit                 # neg3 mu fit (once)
#   bash experiments/2026-11-10_spin_corr_eq128/run_size_scan.sh launch|status|analyze [eq|neg3] [L ...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/WJACOBS/vd7294/flex-investigation}"
export EPS_LIST="${EPS_LIST:--1.7627 -1.65}"
FIT_DIR="$DATA_ROOT/SPIN_CORR_NEG_DMU3_EQEPS_fit"
action="${1:-}"; shift || true

if [[ "$action" == fit ]]; then
  cd "$DATA_ROOT"; export FLEX_OUTPUT_ROOT="$DATA_ROOT"
  set +u; module load anaconda3/2024.10 2>/dev/null || true
  source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${FLEX_CONDA_ENV:-lattice}"; set -u
  python "$REPO/run/susc.py" refit fit --coex COEX_RUNS_C3D/ly40/manage.csv --degree 3 \
    --delta-f 0 --delta-mu 3 --k 1 --scheme negative_drive --fit-min -1.85 --fit-max -1.55 \
    --eps-min -1.7627 --eps-max -1.65 --eps-step 0.1127 --out "$FIT_DIR"
  exit 0
fi
[[ "$action" =~ ^(launch|status|analyze)$ ]] || { sed -n '2,13p' "$0"; exit 2; }

groups=("${1:-eq neg3}"); [[ $# -gt 0 ]] && shift
sizes=("$@")
for g in ${groups[*]}; do
  case "$g" in
    eq)   Ls=(${sizes[@]:-24 32 48 64 96}) ;;
    neg3) Ls=(${sizes[@]:-24 32 48 64 96 128}) ;;
    *) echo "unknown group $g"; exit 2 ;;
  esac
  for L in "${Ls[@]}"; do
    echo "=== $g L=$L"
    if [[ "$g" == eq ]]; then
      ( export LSIZE=$L OUT="$DATA_ROOT/SPIN_CORR_EQ$L" JOB_NAME="spincorr_eq$L"
        export SCHEME=homo DELTA_MU=0.0 DELTA_F=0.0 K=1.0 MU_FIT_JSON=
        bash "$HERE/submit.sh" "$action" )
    else
      [[ -f "$FIT_DIR/fit.json" ]] || { echo "run '$0 fit' first"; exit 1; }
      ( export LSIZE=$L OUT="$DATA_ROOT/SPIN_CORR_NEG_DMU3_EQEPS_$L" JOB_NAME="spincorr_n3_$L"
        export SCHEME=negative_drive DELTA_MU=3.0 DELTA_F=0.0 K=1.0 MU_FIT_JSON="$FIT_DIR/fit.json"
        bash "$HERE/submit.sh" "$action" )
    fi
  done
done
