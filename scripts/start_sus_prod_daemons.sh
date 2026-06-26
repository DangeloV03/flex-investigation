#!/usr/bin/env bash
# Start susceptibility prod dispatcher in tmux.
#
# Prod phase runs square L×L lattices at mu_coex_SIM from the coex phase.
# No analyzer needed — jobs are one-shot (no refinement loop).
#
# Usage (on Della):
#   python generate_susceptibility_jobs.py   # run first, once
#   ./scripts/start_sus_prod_daemons.sh
#   tmux attach -t sus-prod

set -euo pipefail

SESSION="sus-prod"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "$PROJECT_DIR/scripts/env.sh" ]]; then
  DAEMON_SETUP="source scripts/env.sh"
else
  DAEMON_SETUP='module load anaconda3/2024.10 2>/dev/null; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate lattice; export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; export PYTHONUNBUFFERED=1'
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "WARNING: sbatch not in PATH — prod dispatcher will use LOCAL mode."
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists."
  echo "  attach:  tmux attach -t $SESSION"
  echo "  kill:    tmux kill-session -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" -n prod-dispatch -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:prod-dispatch" \
  "${DAEMON_SETUP}; python -u run_susceptibility_all.py --phase prod" C-m

# Second window: re-run generate_susceptibility_jobs.py every 10 min so newly
# finished coex eps values are queued automatically without manual intervention.
tmux new-window -t "$SESSION" -n job-gen -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:job-gen" \
  "${DAEMON_SETUP}; while true; do echo \"[\$(date '+%H:%M:%S')] generating jobs...\"; python generate_susceptibility_jobs.py; echo 'sleeping 600s'; sleep 600; done" C-m

echo "Started tmux session '$SESSION' (prod dispatcher + job generator)"
echo "  attach:  tmux attach -t $SESSION"
echo "  detach:  Ctrl-b then d"
echo "  windows: prod-dispatch (Ctrl-b 0)  job-gen (Ctrl-b 1)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER -n flex_sim"
echo "  find susceptibility_results -name susceptibility_data.csv | wc -l"
