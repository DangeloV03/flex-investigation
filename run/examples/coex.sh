#!/usr/bin/env bash
# Example coexistence campaign: find mu_coex for a few epsilon on Della.
#
#   bash run/examples/coex.sh quick    # ~30 s smoke test on the login node, no Slurm or tmux
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
  quick)
    # One eps on a small 80 x 8 slab, run right here (no Slurm, no tmux). Alternates
    # "run the queued jobs" and "analyze" until mu_coex is fitted (usually 2-4 rounds).
    QUICK="$OUT/examples/COEX_QUICK"
    rm -rf "$QUICK"
    q() { FLEX_OUTPUT_ROOT="$QUICK" python run/coex.py "$@"; }
    q generate --ly 8 --eps-min -1.9 --eps-max -1.9 --eps-step 0.1 \
        --dmu-min 1 --dmu-max 1 --dmu-step 1 >/dev/null
    for round in 1 2 3 4 5 6 7 8 9 10; do
      q dispatch --local --max-concurrent 5 --interval 2 >>"$QUICK/dispatch.log" 2>&1
      q analyze --once >>"$QUICK/analyze.log" 2>&1
      if python - "$QUICK/manage.csv" <<'PY'
import csv, sys
r = next(csv.DictReader(open(sys.argv[1])))
ok = r["isAnalyzed"].strip() != ""
if ok:
    print(f"mu_coex_FITTED = {float(r['mu_coex_FITTED']):.4f} +- {float(r['mu_coex_FITTED_error']):.4f}"
          f"   (FLEX estimate {float(r['mu_coex_FLEX']):.4f})")
sys.exit(0 if ok else 1)
PY
      then
        echo "OK: coex quick test finished in $round rounds. Data in $QUICK"
        exit 0
      fi
      echo "round $round: analyzer asked for more mu points"
    done
    echo "coex quick test did not converge in 10 rounds; see $QUICK/analyze.log" >&2
    exit 1
    ;;
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
