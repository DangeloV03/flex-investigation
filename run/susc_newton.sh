#!/usr/bin/env bash
# Newton search for coexistence mu, per (eps, L), then chi at that mu (mentor's notes, sec. 3.1).
#
#   bash run/susc_newton.sh fit        # cubic fit of D3C Ly=40 coex mu(eps) -> $WORK/fitted_mu.csv, fit.png
#   DRY_RUN=1 bash run/susc_newton.sh launch   # preview the jobs
#   bash run/susc_newton.sh launch     # one chain per (eps, L); each resubmits itself until done
#   bash run/susc_newton.sh status     # table: iteration, mu - mu_fit, <m>, SE, J per (eps, L)
#   bash run/susc_newton.sh kick       # restart chains whose Slurm jobs are gone (node failure etc.)
#   bash run/susc_newton.sh stop       # every chain stops at its next decision (touches $WORK/runs/STOP)
#   bash run/susc_newton.sh analyze    # link converged points -> combined/, plots + FSS, mu shift plot
#
#   SMOKE=1 bash run/susc_newton.sh fit|launch|status|analyze
#       small test: 2 eps, L = 16/32, short segments, in NEWTON_SMOKE/ (separate from the campaign)
#
# Each (eps, L) starts at the cubic-fit mu (robust to single bad coex points). After every
# segment (no minimum jump count, no standard-error test):
#   |<m>| > 0.05             ->  damped Newton step  beta*mu -= tau <m> / (d<m>/d beta*mu),
#                                new iteration folder started from the last lattices
#   otherwise                ->  more data at the same mu until J >= 10 with |<m>| <= 0.05
# d<m>/d(beta*mu) comes from the fluctuations of signed m (N Cov(m, rho_B + rho_I)), later from
# the secant between iterations. Details: src/flexsim/susc/newton.py.
#
# First target: POS_DMU2 (positive_drive, beta*delta_f = 0, k = 1, beta*delta_mu = 2), eps
# -1.800042 ... -1.635042 (the grid of SUSC_RUNS_POS_DMU2_shift), L = 16 ... 128.
#
# Paths are absolute (DATA_ROOT), so this runs from any clone and any account.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/WJACOBS/vd7294/flex-investigation}"
export FLEX_OUTPUT_ROOT="$DATA_ROOT"
cd "$DATA_ROOT"
umask 002                                         # group-writable for both accounts

# Defaults are POS_DMU2. Wrappers (e.g. run/susc_newton_homo_1.0.sh) set these from the environment.
CAMPAIGN="${CAMPAIGN:-SUSC_RUNS_POS_DMU2}"        # old campaign; `fit` only reads its physics
COEX="${COEX:-COEX_RUNS_D3C/ly40/manage.csv}"     # coex mu(eps) points the cubic is fitted to
DEGREE="${DEGREE:-3}"
FIT_MIN="${FIT_MIN:--1.85}"
FIT_MAX="${FIT_MAX:--1.55}"
PHYSICS=(--delta-f "${DELTA_F:-0.0}" --delta-mu "${DELTA_MU:-2.0}" --k "${K:-1.0}" --scheme "${SCHEME:-positive_drive}")
EXTRA=()                                          # newton start overrides (see newton.py start --help)

if [[ "${SMOKE:-0}" == 1 ]]; then
  WORK="${WORK:-NEWTON_SMOKE}"
  EPS_MIN=-1.700042; EPS_MAX=-1.650042; EPS_STEP=0.05
  SIZES="16 32"
  EXTRA=(--eq0 1e5 --prod0 2e5 --seg-prod 2e5 --eq-iter 2e4)
else
  WORK="${WORK:-SUSC_RUNS_POS_DMU2_newton}"
  EPS_MIN="${EPS_MIN:--1.800042}"; EPS_MAX="${EPS_MAX:--1.635042}"; EPS_STEP="${EPS_STEP:-0.005}"
  [[ -n "${MAX_ITER:-}" ]] && EXTRA+=(--max-iter "$MAX_ITER")
  SIZES="${SIZES:-16 32 48 64 96 128}"
fi
RUNS="$WORK/runs"
COMBINED="$WORK/combined"

newton() { python "$REPO/run/susc.py" newton "$@"; }
need() { [[ -f "$1" ]] || { echo "Missing $1 — run '$0 $2' first." >&2; exit 1; }; }

do_fit() {
  python "$REPO/run/susc.py" refit fit --campaign "$CAMPAIGN" --coex "$COEX" --degree "$DEGREE" \
      --fit-min "$FIT_MIN" --fit-max "$FIT_MAX" \
      --eps-min "$EPS_MIN" --eps-max "$EPS_MAX" --eps-step "$EPS_STEP" --out "$WORK"
}

do_launch() {
  need "$WORK/fitted_mu.csv" fit
  local args=(start --base "$RUNS" --mu-source "$WORK/fitted_mu.csv"
              --eps-min "$EPS_MIN" --eps-max "$EPS_MAX" --eps-step "$EPS_STEP"
              --sizes $SIZES "${PHYSICS[@]}" ${EXTRA[@]+"${EXTRA[@]}"})
  [[ -n "${ACCOUNT:-}" ]] && args+=(--account "$ACCOUNT")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    newton "${args[@]}" --dry-run
  else
    newton "${args[@]}" | tee -a "$WORK/launch.log"
  fi
}

do_status() {
  squeue -u "$USER" -h -o "%j %T" 2>/dev/null | awk '$1 ~ /^nwt_/ {n[$2]++} END {for (s in n) print "slurm " s ": " n[s]}' || true
  newton status --base "$RUNS" ${ONLY:+--only "$ONLY"}
}

do_analyze() {
  newton combine --base "$RUNS" --out "$COMBINED"
  python "$REPO/run/plots.py" susc --results "$COMBINED" --outdir "$WORK/plots"
  python "$REPO/run/plots.py" fss --results "$COMBINED" --outdir "$WORK/plots/fss"
  echo "Plots in $DATA_ROOT/$WORK/plots; mu shifts in $COMBINED/mu_shift_vs_eps.png"
}

case "${1:-}" in
  fit)     do_fit ;;
  launch)  do_launch ;;
  status)  do_status ;;
  kick)    k=(kick --base "$RUNS"); [[ "${DRY_RUN:-0}" == 1 ]] && k+=(--dry-run)
           [[ ! -e "$RUNS/STOP" ]] && k+=(--resume-stopped)
           newton "${k[@]}" ;;
  stop)    touch "$RUNS/STOP"; echo "Chains stop at their next decision. To resume: rm $RUNS/STOP, then '$0 kick'." ;;
  analyze) do_analyze ;;
  *)       sed -n '2,13p' "$0"; exit 2 ;;
esac
