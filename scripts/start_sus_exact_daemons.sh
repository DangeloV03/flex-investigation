#!/usr/bin/env bash
# Start susceptibility exact-mu dispatcher in tmux.
#
# Runs square L×L lattices at mu = 2*epsilon (analytic coexistence) as a
# control experiment to isolate whether errors come from the coex measurement.
#
# Usage (on Della):
#   python generate_susceptibility_exact.py   # run first, once
#   ./scripts/start_sus_exact_daemons.sh
#   tmux attach -t sus-exact

set -euo pipefail

SESSION="sus-exact"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "$PROJECT_DIR/scripts/env.sh" ]]; then
  DAEMON_SETUP="source scripts/env.sh"
else
  DAEMON_SETUP='module load anaconda3/2024.10 2>/dev/null; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate lattice; export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; export PYTHONUNBUFFERED=1'
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists."
  echo "  attach:  tmux attach -t $SESSION"
  echo "  kill:    tmux kill-session -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" -n exact-dispatch -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:exact-dispatch" \
  "${DAEMON_SETUP}; python -u run_susceptibility_all.py --phase exact" C-m

echo "Started tmux session '$SESSION' (exact-mu dispatcher)"
echo "  attach:  tmux attach -t $SESSION"
echo "  detach:  Ctrl-b then d"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER -n flex_sim"
echo "  find susceptibility_results/exact -name susceptibility_data.csv | wc -l"
echo ""
echo "Plot when done:"
echo "  python plot_susceptibility.py --results susceptibility_results/exact --outdir plots/exact"
