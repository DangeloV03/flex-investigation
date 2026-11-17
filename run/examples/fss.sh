#!/usr/bin/env bash
# Example: finite-size-scaling (FSS) analysis of the five positive-drive fitted-coex campaigns.
#
#   bash run/examples/fss.sh run              # read every replica's m(t), fit, plot (a few min per campaign)
#   bash run/examples/fss.sh refit [options]  # redo fits + plots from the cached tables (seconds)
#   bash run/examples/fss.sh summary          # one line per campaign: eps_c, 1/nu, gamma/nu, beta/nu
#
# 'refit' passes extra options to run/fss_runner.py, e.g.
#   bash run/examples/fss.sh refit --fix-nu 1.0           # hold 1/nu at the 2D Ising value
#   bash run/examples/fss.sh refit --sizes 32 48 64       # drop L = 16
#   bash run/examples/fss.sh refit --min-jumps 0          # keep unconverged points too
# All the physics (observables, cuts, collapse fit) is explained at the top of run/fss_runner.py.
#
# Needs numpy/scipy/pandas/matplotlib, e.g. on Della:
#   module load anaconda3/2024.10 && conda activate lattice
# 'run' reads ~1000 time-series files per campaign: use a compute node (salloc) or raise WORKERS.
#
# Output: one folder per campaign under $FSS_OUT (default ~/fss_example_output), each with
#   observables_vs_eps.png  fss_collapse.png  chi_peak_scaling.png  fss_table.csv  fss_fit_results.json

set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/WJACOBS/vd7294/flex-investigation}"
FSS_OUT="${FSS_OUT:-$HOME/fss_example_output}"
WORKERS="${WORKERS:-4}"
# short name : campaign folder under DATA_ROOT (edit this list to analyse other campaigns)
CAMPAIGNS=(
  "pos_dmu-2:SUSC_RUNS_POS_DMU-2_FITTED_COEX"
  "pos_dmu-1:SUSC_RUNS_POS_DMU-1_FITTED_COEX"
  "pos_dmu1:SUSC_RUNS_POS_DMU1_FITTED_COEX"
  "pos_dmu2:SUSC_RUNS_POS_DMU2_FITTED_COEX"
  "pos_dmu3:SUSC_RUNS_POS_DMU3_FITTED_COEX"
)

cmd="${1:-}"; shift || true
case "$cmd" in
  run)
    for c in "${CAMPAIGNS[@]}"; do
      name="${c%%:*}"; folder="${c#*:}"
      echo "######## $name"
      python run/fss_runner.py "$DATA_ROOT/$folder/runs" --out "$FSS_OUT/$name" --title "$folder" \
          --workers "$WORKERS" "$@"
    done
    ;;
  refit)
    for c in "${CAMPAIGNS[@]}"; do
      name="${c%%:*}"; folder="${c#*:}"
      echo "######## $name"
      python run/fss_runner.py --table "$FSS_OUT/$name/fss_table.csv" --out "$FSS_OUT/$name" \
          --title "$folder" "$@"
    done
    ;;
  summary)
    python - "$FSS_OUT" "${CAMPAIGNS[@]%%:*}" <<'EOF'
import json, os, sys
root, names = sys.argv[1], sys.argv[2:]
print(f"{'campaign':12s} {'kept':>9s} {'eps_c(U4)':>10s} {'1/nu':>6s} {'gamma/nu':>9s} {'beta/nu':>8s}")
for n in names:
    p = os.path.join(root, n, "fss_fit_results.json")
    if not os.path.isfile(p):
        print(f"{n:12s} (no results yet)"); continue
    r = json.load(open(p)); f = r["fits"]
    g = lambda o, k, s=1: f"{s * f[o][k]:.3f}" if o in f and f[o][k] is not None else "-"
    eps = f"{f['U4']['eps_c']:.4f}" if "U4" in f else "-"
    print(f"{n:12s} {r['n_kept']:>4d}/{r['n_points']:<4d} {eps:>10s} {g('U4', 'inv_nu'):>6s} "
          f"{g('chi', 'b', -1):>9s} {g('absm', 'b'):>8s}")
EOF
    ;;
  *)
    sed -n '2,7p' "$0"
    exit 2
    ;;
esac
