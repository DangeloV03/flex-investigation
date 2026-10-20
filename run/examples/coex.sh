#!/usr/bin/env bash
# Example coexistence campaign: find mu_coex for a few epsilon on Della.
#
#   bash run/examples/coex.sh launch   # make the jobs, start dispatcher + analyzer in tmux
#   bash run/examples/coex.sh status   # Slurm queue, queue file, fitted mu_coex so far
#   bash run/examples/coex.sh stop     # stop the tmux session and cancel ALL your flex_sim jobs
#   bash run/examples/coex.sh plots    # FLEX vs simulated mu_coex
#
# Physics: homo scheme, beta*delta_f = 0, k = 1, beta*delta_mu = 1, on a
# 160 x 16 slab, for epsilon = -2.0 ... -1.6 (step 0.1). Each epsilon starts with
# 10 mu values around the FLEX prediction (50 jobs). The analyzer fits mu_coex
# from phi(mu) and queues extra mu points where the fit needs them.
#
# The dispatcher and analyzer run in the tmux session "flex-investigation"
# (`tmux attach -t flex-investigation`, detach with Ctrl-b d). Only one coex
# campaign can use that session at a time.
#
# Output: <output root>/examples/COEX_RUNS_EXAMPLE/ly16 (manage.csv, samples/,
# results/, slurm_reports/) and examples/plots/flex_vs_sim. The output root is
# <repo>/output unless FLEX_OUTPUT_ROOT is set.
# Afterwards, run/examples/criticality.sh works on these results.

set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${FLEX_OUTPUT_ROOT:-$PWD/output}"
CAMPAIGN="$OUT/examples/COEX_RUNS_EXAMPLE/ly16"

# The coex commands use fixed file names (manage.csv, samples/, results/) in
# the output root, so point the output root at this campaign's folder.
coex() { FLEX_OUTPUT_ROOT="$CAMPAIGN" python run/coex.py "$@"; }

case "${1:-}" in
  launch)
    coex generate --ly 16 \
        --eps-min -2.0 --eps-max -1.6 --eps-step 0.1 \
        --dmu-min 1 --dmu-max 1 --dmu-step 1
    coex start
    ;;
  status)
    echo "flex_sim jobs in Slurm: $(squeue -u "$USER" -n flex_sim -h 2>/dev/null | wc -l)"
    python - "$CAMPAIGN" <<'PY'
import json, sys
import pandas as pd
campaign = sys.argv[1]
m = json.load(open(f"{campaign}/run_all_queue.json"))
print(f"queue: {len(m.get('pending', []))} pending, {len(m.get('in_flight', {}))} in flight")
cols = ["epsilon", "delta_mu", "mu_coex_FLEX", "mu_coex_FITTED", "mu_coex_FITTED_error",
        "RequestForAdditionalData", "isAnalyzed"]
print(pd.read_csv(f"{campaign}/manage.csv")[cols].to_string(index=False))
PY
    ;;
  stop)
    coex stop --slurm
    ;;
  plots)
    python run/plots.py flex-vs-sim --runs-root examples --outdir examples/plots/flex_vs_sim
    echo "Plots in $OUT/examples/plots/flex_vs_sim"
    ;;
  *)
    sed -n '2,7p' "$0"
    exit 2
    ;;
esac
