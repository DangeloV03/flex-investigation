#!/usr/bin/env bash
# Start susceptibility coex dispatcher + analyzer in tmux.
#
# The coex dispatcher must stay alive while the analyzer enqueues mu refinements.
#
# Usage (on Della):
#   ./scripts/start_sus_coex_daemons.sh
#   tmux attach -t sus-coex
#   Ctrl-b d

set -euo pipefail

SESSION="sus-coex"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "$PROJECT_DIR/scripts/env.sh" ]]; then
  DAEMON_SETUP="source scripts/env.sh"
else
  DAEMON_SETUP='module load anaconda3/2024.10 2>/dev/null; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate lattice; export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; export PYTHONUNBUFFERED=1'
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists."
  if ! pgrep -f "run_susceptibility_all.py --phase coex" >/dev/null \
     || ! pgrep -f "susceptibility_coex_queue.json" >/dev/null; then
    echo "  WARNING: sus-coex tmux exists but dispatcher and/or analyzer process is missing."
    echo "  The session is probably stale. Kill and restart:"
    echo "    tmux kill-session -t $SESSION"
    echo "    $0"
  fi
  echo "  attach:  tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" -n coex-dispatch -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:coex-dispatch" \
  "${DAEMON_SETUP}; python -u run_susceptibility_all.py --phase coex" C-m

tmux new-window -t "$SESSION" -n analyzer -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:analyzer" \
  "${DAEMON_SETUP}; python -u analyzer.py \
  --manage susceptibility_manage.csv \
  --results susceptibility_results/coex \
  --samples susceptibility_samples/coex \
  --manifest susceptibility_coex_queue.json \
  --depth-first" C-m

echo "Started tmux session '$SESSION' (coex dispatcher + analyzer)"
echo "  attach:  tmux attach -t $SESSION"
echo "  detach:  Ctrl-b then d"
