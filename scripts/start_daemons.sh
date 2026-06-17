#!/usr/bin/env bash
# Start run_all.py and analyzer.py in a detached tmux session.
#
# Usage (on Della login node):
#   ./scripts/start_daemons.sh
#   tmux attach -t flex-investigation   # reattach later
#   Ctrl-b d                            # detach without stopping

set -euo pipefail

SESSION="flex-investigation"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists."
  echo "  attach:  tmux attach -t $SESSION"
  echo "  stop:    ./scripts/stop_daemons.sh"
  exit 1
fi

tmux new-session -d -s "$SESSION" -n run_all -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:run_all" \
  "module load anaconda3/2024.10 2>/dev/null; conda activate lattice; export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\"; python -u run_all.py" C-m

tmux new-window -t "$SESSION" -n analyzer -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:analyzer" \
  "module load anaconda3/2024.10 2>/dev/null; conda activate lattice; export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\"; python -u analyzer.py" C-m

echo "Started tmux session '$SESSION' with windows: run_all, analyzer"
echo "  attach:  tmux attach -t $SESSION"
echo "  list:    tmux list-windows -t $SESSION"
echo "  detach:  Ctrl-b then d"
