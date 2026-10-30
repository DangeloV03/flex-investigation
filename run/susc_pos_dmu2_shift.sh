#!/usr/bin/env bash
# More POS_DMU2 susceptibility runs, colder, where the critical point actually is.
#
#   bash run/susc_pos_dmu2_shift.sh fit       # cubic fit of D3C Ly=40 coex mu(eps) -> fitted_mu.csv, fit.png
#   DRY_RUN=1 bash run/susc_pos_dmu2_shift.sh launch   # preview the jobs
#   bash run/susc_pos_dmu2_shift.sh launch    # submit the sweep (top-ups chain automatically)
#   bash run/susc_pos_dmu2_shift.sh status    # Slurm queue + latest jump report
#   bash run/susc_pos_dmu2_shift.sh analyze   # old + new runs, |<m>| > 0.05 dropped -> plots, FSS
#
# Why: SUSC_RUNS_POS_DMU2 ran eps -1.630 ... -1.430 around the Ly=16 eps_c (-1.530).
# For every L >= 32 chi is largest at its coldest eps and still rising, so the peak
# was missed. The D3C Ly=40 BC estimate and the mentor's phase diagram put eps_c
# near -1.70.
#
# What: positive_drive, beta*delta_f = 0, k = 1, beta*delta_mu = 2, L = 16 ... 128.
# New eps -1.800042 ... -1.635042 at step 0.005 (34 eps), the old campaign's grid
# continued colder, so the two join at -1.630042 with no gap. Each runs at mu from a
# cubic fit of COEX_RUNS_D3C/ly40 mu_coex(eps) over eps in [-1.85, -1.55].
#
# The finished campaign is not touched. New runs go to SUSC_RUNS_POS_DMU2_shift/runs.
# `analyze` links the old and new points with |<m>| <= 0.05 into
# SUSC_RUNS_POS_DMU2_shift/combined and plots from there. Dropped points stay on disk
# and are listed in combined/dropped_points.csv.
#
# Paths are absolute (DATA_ROOT), so this runs from any clone and any account.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/WJACOBS/vd7294/flex-investigation}"
export FLEX_OUTPUT_ROOT="$DATA_ROOT"
cd "$DATA_ROOT"
umask 002                                         # group-writable for both accounts

CAMPAIGN=SUSC_RUNS_POS_DMU2
COEX=COEX_RUNS_D3C/ly40/manage.csv
WORK="${CAMPAIGN}_shift"
RUNS="$WORK/runs"
COMBINED="$WORK/combined"
DEGREE="${DEGREE:-3}"
FIT_MIN="${FIT_MIN:--1.85}"
FIT_MAX="${FIT_MAX:--1.55}"
EPS_MIN="${EPS_MIN:--1.800042}"
EPS_MAX="${EPS_MAX:--1.635042}"
EPS_STEP=0.005
SIZES="${SIZES:-16 32 48 64 96 128}"
J_THRESHOLD="${J_THRESHOLD:-10}"                  # jumps per (eps, L) before a run counts as done
M_CUT="${M_CUT:-0.05}"                            # analyze drops points with |<m>| above this
PHYSICS=(--delta-f 0.0 --delta-mu 2.0 --k 1.0 --scheme positive_drive)

refit() { python "$REPO/run/susc.py" refit "$@"; }
need() { [[ -f "$1" ]] || { echo "Missing $1 — run '$0 $2' first." >&2; exit 1; }; }

do_fit() {
  refit fit --campaign "$CAMPAIGN" --coex "$COEX" --degree "$DEGREE" \
      --fit-min "$FIT_MIN" --fit-max "$FIT_MAX" \
      --eps-min "$EPS_MIN" --eps-max "$EPS_MAX" --eps-step "$EPS_STEP" --out "$WORK"
}

do_launch() {
  need "$WORK/fitted_mu.csv" fit
  if compgen -G "$RUNS/*/_*/susceptibility_data.csv" >/dev/null && [[ "${FORCE:-0}" != 1 ]]; then
    echo "$RUNS already has data; refusing to submit again (FORCE=1 to override)." >&2
    exit 1
  fi
  local args=(sweep --eps-min "$EPS_MIN" --eps-max "$EPS_MAX" --eps-step "$EPS_STEP"
              --sizes $SIZES --threshold "$J_THRESHOLD" --results-base "$RUNS"
              --mu-source "$WORK/fitted_mu.csv" "${PHYSICS[@]}")
  [[ "${DRY_RUN:-0}" == 1 ]] && args+=(--dry-run)
  python "$REPO/run/susc.py" "${args[@]}" | tee -a "$WORK/launch.log"
}

do_status() {
  squeue -u "$USER" -o "%.10i %.16j %.8T %.10M %R" 2>/dev/null || true
  if [[ -f "$RUNS/jump_report.md" ]]; then
    sed -n '1,8p' "$RUNS/jump_report.md"
    grep -E '^\| [0-9]' "$RUNS/jump_report.md" | grep -v '✓' || echo "(all listed pairs pass)"
  else
    echo "No jump report yet in $RUNS (written by the first check job)."
  fi
}

do_analyze() {
  refit combine --base "$CAMPAIGN" --base "$RUNS" --max-abs-m "$M_CUT" --out "$COMBINED"
  python "$REPO/run/plots.py" susc --results "$COMBINED" --outdir "$WORK/plots"
  python "$REPO/run/plots.py" fss --results "$COMBINED" --outdir "$WORK/plots/fss"
  echo "Plots in $DATA_ROOT/$WORK/plots (dropped points: $COMBINED/dropped_points.csv)"
}

case "${1:-}" in
  fit)     do_fit ;;
  launch)  do_launch ;;
  status)  do_status ;;
  analyze) do_analyze ;;
  *)       sed -n '2,8p' "$0"; exit 2 ;;
esac
