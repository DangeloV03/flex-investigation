#!/usr/bin/env bash
# Refit the coex mu(eps) of a susc campaign to a smooth function, rerun every eps
# whose <m> is off zero at that fitted mu, then swap the reruns into the campaign.
#
#   bash run/coex_refit_rerun.sh fit       # fit mu_coex(eps) -> <work>/fitted_mu.csv, fit.png
#   bash run/coex_refit_rerun.sh select    # eps with any L |<m>| >= M_THRESHOLD -> rerun_eps.csv
#   bash run/coex_refit_rerun.sh launch    # smart sweep of those eps at the fitted mu (DRY_RUN=1 to preview)
#   bash run/coex_refit_rerun.sh all       # fit + select + launch
#   bash run/coex_refit_rerun.sh status    # Slurm queue + latest jump report of the reruns
#   bash run/coex_refit_rerun.sh compare   # <m> before vs after, per L -> <work>/compare/compare.png
#   bash run/coex_refit_rerun.sh promote [--yes]
#                                          # once converged: old eps folders -> archive, reruns -> campaign
#
# Why: each eps of a campaign ran at its own slab coex fit. Those fits scatter by
# ~1e-4 to 1e-3 in mu, and the susc runs respond to that scatter one-to-one, so
# <m> != 0 at many eps. A smooth fit of mu(eps) removes the eps-to-eps scatter.
#
# Defaults are the S2A campaign (homo, beta*delta_f = 0, k = 1, beta*delta_mu = 1,
# mu from COEX_RUNS_C2A/ly40). Override any setting below from the environment,
# e.g.  DEGREE=2 bash run/coex_refit_rerun.sh fit
#
# Nothing in the campaign changes until `promote --yes`. Reruns run in
# <work>/runs (a separate SUSC_RUNS base, so the jump check never touches the
# campaign's cold-end runs). promote refuses unless every rerun has J >= 10, and
# moves the replaced folders into <work>/archived_runs instead of deleting them.
#
# Paths are relative to the output root. On Della the campaigns live in the repo
# root, so that is the default here; set FLEX_OUTPUT_ROOT to change it.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export FLEX_OUTPUT_ROOT="${FLEX_OUTPUT_ROOT:-$REPO}"
cd "$FLEX_OUTPUT_ROOT"

CAMPAIGN="${CAMPAIGN:-SUSC_RUNS_S2A}"                       # campaign to fix
COEX="${COEX:-COEX_RUNS_C2A/ly40/manage.csv}"                # coex fits it was run from
DEGREE="${DEGREE:-1}"                                        # 1 = linear fit of mu(eps)
MAX_FIT_ERR="${MAX_FIT_ERR:-1e-3}"                           # coex fits with larger error are not used
M_THRESHOLD="${M_THRESHOLD:-0.05}"                           # rerun eps with any L |<m>| >= this
MIN_EPS="${MIN_EPS:--1.775}"                                 # skip colder eps (large L never converges); "none" = keep all
EPS_STEP="${EPS_STEP:-0.005}"                                # the campaign's eps grid step
SIZES="${SIZES:-16 32 48 64 96 128}"
J_THRESHOLD="${J_THRESHOLD:-10}"                             # jumps per (eps, L) before a run counts as done
WORK="${WORK:-${CAMPAIGN}_refit}"                            # fit, selection, reruns, archive, plots
RUNS="$WORK/runs"
ARCHIVE="$WORK/archived_runs"

refit() { python "$REPO/run/susc.py" refit "$@"; }

need() { [[ -f "$1" ]] || { echo "Missing $1 — run '$0 $2' first." >&2; exit 1; }; }

do_fit() {
  refit fit --campaign "$CAMPAIGN" --coex "$COEX" --degree "$DEGREE" \
      --max-fit-err "$MAX_FIT_ERR" --out "$WORK"
}

do_select() {
  local cold=()
  [[ "$MIN_EPS" != none ]] && cold=(--min-eps "$MIN_EPS")
  refit select --campaign "$CAMPAIGN" --threshold "$M_THRESHOLD" ${cold[@]+"${cold[@]}"} --out "$WORK"
}

do_launch() {
  need "$WORK/fitted_mu.csv" fit
  need "$WORK/rerun_eps.csv" select
  if [[ -d "$RUNS" ]] && compgen -G "$RUNS/*/_*/susceptibility_data.csv" >/dev/null && [[ "${FORCE:-0}" != 1 ]]; then
    echo "$RUNS already has rerun data; refusing to submit again (FORCE=1 to override)." >&2
    exit 1
  fi
  # eps grid spanning the rerun list, with everything not on the list skipped,
  # plus the physics flags the fit recorded.
  local plan
  plan=$(python - "$WORK" "$EPS_STEP" <<'PY'
import csv, json, sys
from flexsim.grid import frange
work, step = sys.argv[1], float(sys.argv[2])
eps = sorted(round(float(r["epsilon"]), 6) for r in csv.DictReader(open(f"{work}/rerun_eps.csv")))
if not eps:
    print("NONE"); sys.exit()
grid = [round(e, 6) for e in frange(eps[0], eps[-1], step)]
missing = [e for e in eps if e not in grid]
if missing:
    sys.exit(f"rerun eps {missing} are not on the {step} grid from {eps[0]}")
skip = [e for e in grid if e not in eps]
phys = json.load(open(f"{work}/fit.json"))["physics"]
print(eps[0], eps[-1], phys["delta_f"], phys["delta_mu"], phys["k"], phys["scheme"],
      " ".join(f"{e:g}" for e in skip))
PY
)
  if [[ "$plan" == NONE ]]; then echo "No eps to rerun."; return; fi
  read -r EMIN EMAX DF DMU K SCHEME SKIP <<<"$plan"
  local args=(sweep --eps-min "$EMIN" --eps-max "$EMAX" --eps-step "$EPS_STEP"
              --sizes $SIZES --threshold "$J_THRESHOLD" --results-base "$RUNS"
              --mu-source "$WORK/fitted_mu.csv"
              --delta-f "$DF" --delta-mu "$DMU" --k "$K" --scheme "$SCHEME")
  [[ -n "${SKIP:-}" ]] && args+=(--skip-eps $SKIP)
  [[ "${DRY_RUN:-0}" == 1 ]] && args+=(--dry-run)
  mkdir -p "$WORK"
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

do_compare() {
  need "$WORK/rerun_eps.csv" select
  if [[ -d "$ARCHIVE" ]]; then
    refit compare --before "$ARCHIVE" --after "$CAMPAIGN" --eps-file "$WORK/rerun_eps.csv" \
        --threshold "$M_THRESHOLD" --out "$WORK/compare"
  else
    refit compare --before "$CAMPAIGN" --after "$RUNS" --eps-file "$WORK/rerun_eps.csv" \
        --threshold "$M_THRESHOLD" --out "$WORK/compare"
  fi
}

do_promote() {
  need "$WORK/rerun_eps.csv" select
  refit promote --campaign "$CAMPAIGN" --staging "$RUNS" --archive "$ARCHIVE" \
      --eps-file "$WORK/rerun_eps.csv" --threshold "$J_THRESHOLD" "$@"
}

case "${1:-}" in
  fit)     do_fit ;;
  select)  do_select ;;
  launch)  do_launch ;;
  all)     do_fit; do_select; do_launch ;;
  status)  do_status ;;
  compare) do_compare ;;
  promote) shift; do_promote "$@" ;;
  *)       sed -n '2,14p' "$0"; exit 2 ;;
esac
