#!/usr/bin/env bash
# Example susceptibility campaign: a small smart sweep on Della.
#
#   bash run/examples/susc.sh quick    # ~1 min smoke test on the login node, no Slurm
#   bash run/examples/susc.sh launch   # submit the sweep (6 Slurm jobs + a check job)
#   bash run/examples/susc.sh status   # Slurm queue + latest jump report
#   bash run/examples/susc.sh plots    # chi, <|m|>, c, U4 vs epsilon and jump counts
#
# Physics: Ising limit (runner defaults: beta*delta_f = -20, k = 0), run at the
# exact equilibrium coexistence mu. epsilon_c is about -1.763 here, so the three
# epsilon values (-1.85, -1.75, -1.65) sit below, at and above it.
#
# Size: L = 16 and 32 only, one job per (epsilon, L). L=16 takes ~5 min and
# L=32 ~25 min. The check job counts magnetization jumps and submits top-up runs
# until every (epsilon, L) has >= 2 jumps (a real campaign uses the default 10).
#
# Output: <output root>/examples/SUSC_RUNS_EXAMPLE and examples/plots/susc.
# The output root is <repo>/output unless FLEX_OUTPUT_ROOT is set.

set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${FLEX_OUTPUT_ROOT:-$PWD/output}"
RESULTS=examples/SUSC_RUNS_EXAMPLE      # relative to the output root
PLOTS=examples/plots/susc

case "${1:-}" in
  quick)
    # One (eps, L = 16) point, 4 replicas, run right here (no Slurm), ~1 minute.
    # Checks the lattice engine, the runner and the output files.
    rm -rf "$OUT/examples/SUSC_QUICK"
    python run/susc.py run-one --epsilon -1.75 --L 16 --cpus 4 \
        --eq-time 500000 --prod-time 1000000 --prod-chunks 10000 \
        --results-base examples/SUSC_QUICK
    echo "OK: susc quick test finished. Data in $OUT/examples/SUSC_QUICK"
    ;;
  launch)
    python run/susc.py sweep \
        --eps-min -1.85 --eps-max -1.65 --eps-step 0.1 \
        --sizes 16 32 \
        --mu-exact \
        --threshold 2 \
        --results-base "$RESULTS"
    ;;
  status)
    squeue -u "$USER" -o "%.10i %.16j %.8T %.10M %R" || true
    if [[ -f "$OUT/$RESULTS/jump_report.md" ]]; then
      cat "$OUT/$RESULTS/jump_report.md"
    else
      echo "No jump report yet (written by the check job after the sweep finishes)."
    fi
    ;;
  plots)
    python run/plots.py susc --results "$RESULTS" --outdir "$PLOTS"
    python run/plots.py jumps --results "$RESULTS" --all-eps --outdir "$PLOTS/jumps"
    echo "Plots in $OUT/$PLOTS"
    ;;
  *)
    sed -n '2,7p' "$0"
    exit 2
    ;;
esac
