#!/usr/bin/env bash
# Susceptibility campaign at mu from a cubic fit of Ly = 40 coex mu(eps). Plain smart sweep
# (top-ups until every (eps, L) has J >= 10), no Newton correction of mu.
#
#   bash <wrapper> fit        # cubic fit of the coex mu(eps) -> $WORK/fitted_mu.csv, fit.png, fit.json
#   DRY_RUN=1 bash <wrapper> launch   # preview the jobs
#   bash <wrapper> launch     # submit the sweep; top-up rounds chain automatically
#   bash <wrapper> status     # Slurm queue + latest jump report
#   bash <wrapper> analyze    # chi/U4 plots + FSS; points with |<m>| > 0.05 dropped (kept on disk)
#
# Wrappers set the campaign through the environment (see run/susc_fitted_coex_neg_dmu2.sh):
#   WORK          output folder, e.g. SUSC_RUNS_NEG_DMU2_FITTED_COEX (runs go to $WORK/runs)
#   PHYS_CAMPAIGN optional: an existing SUSC_RUNS base with the same physics. fit checks the
#                 physics against it and lists its old mu next to the new one. Leave it unset
#                 (or point it at an empty folder) for physics that has never had a susc run.
#   COEX          coex manage.csv with mu_coex_FITTED(_error)
#   FIT_MIN/MAX   eps range of coex points used in the fit
#   EPS_MIN/MAX/STEP  susc eps grid;  SIZES  L values
#   DELTA_F DELTA_MU K SCHEME   physics
#
# Paths are absolute (DATA_ROOT), so this runs from any clone and any account.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/WJACOBS/vd7294/flex-investigation}"
export FLEX_OUTPUT_ROOT="$DATA_ROOT"
cd "$DATA_ROOT"
umask 002                                         # group-writable for both accounts

: "${WORK:?set by the wrapper}" "${COEX:?}" "${FIT_MIN:?}" "${FIT_MAX:?}"
: "${EPS_MIN:?}" "${EPS_MAX:?}" "${DELTA_MU:?}" "${SCHEME:?}"
DEGREE="${DEGREE:-3}"
EPS_STEP="${EPS_STEP:-0.005}"
SIZES="${SIZES:-16 32 48 64}"
J_THRESHOLD="${J_THRESHOLD:-10}"
M_CUT="${M_CUT:-0.05}"
PHYSICS=(--delta-f "${DELTA_F:-0.0}" --delta-mu "$DELTA_MU" --k "${K:-1.0}" --scheme "$SCHEME")
RUNS="$WORK/runs"

need() { [[ -f "$1" ]] || { echo "Missing $1 — run '$0 $2' first." >&2; exit 1; }; }

do_fit() {
  local camp=()
  [[ -n "${PHYS_CAMPAIGN:-}" ]] && camp=(--campaign "$PHYS_CAMPAIGN")
  python "$REPO/run/susc.py" refit fit "${camp[@]}" --coex "$COEX" --degree "$DEGREE" "${PHYSICS[@]}" \
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
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python "$REPO/run/susc.py" "${args[@]}" --dry-run
  else
    python "$REPO/run/susc.py" "${args[@]}" | tee -a "$WORK/launch.log"
  fi
}

do_status() {
  squeue -u "$USER" -h -o "%j %T" 2>/dev/null | awk '$1 ~ /^susc_/ {n[$1" "$2]++} END {for (s in n) print n[s], s}' | sort -k2 || true
  if [[ -f "$RUNS/jump_report.md" ]]; then
    sed -n '1,8p' "$RUNS/jump_report.md"
    grep -E '^\| [0-9]' "$RUNS/jump_report.md" | grep -v '✓' || echo "(all listed pairs pass)"
  else
    echo "No jump report yet in $RUNS (written by the first check job)."
  fi
}

do_analyze() {
  python "$REPO/run/susc.py" refit combine --base "$RUNS" --max-abs-m "$M_CUT" --out "$WORK/combined"
  python "$REPO/run/plots.py" susc --results "$WORK/combined" --outdir "$WORK/plots"
  python "$REPO/run/plots.py" fss --results "$WORK/combined" --outdir "$WORK/plots/fss"
  echo "Plots in $DATA_ROOT/$WORK/plots (dropped points: $WORK/combined/dropped_points.csv)"
}

case "${1:-}" in
  fit)     do_fit ;;
  launch)  do_launch ;;
  status)  do_status ;;
  analyze) do_analyze ;;
  *)       sed -n '2,9p' "$0"; exit 2 ;;
esac
