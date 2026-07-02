#!/usr/bin/env bash
# Stop the flex-investigation tmux session (run_all + analyzer).
#
# Usage:
#   ./coex/stop_daemons.sh          # kill tmux session only
#   ./coex/stop_daemons.sh --slurm  # also scancel your flex_sim jobs

set -euo pipefail

SESSION="flex-investigation"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
  echo "Killed tmux session '$SESSION'"
else
  echo "No tmux session '$SESSION' (already stopped?)"
fi

# Fallback: kill stray processes not in tmux
if pkill -f "python.*run_all.py" 2>/dev/null; then
  echo "Killed stray run_all.py"
fi
if pkill -f "python.*analyzer.py" 2>/dev/null; then
  echo "Killed stray analyzer.py"
fi

if [[ "${1:-}" == "--slurm" ]]; then
  scancel -u "$(whoami)" -n flex_sim 2>/dev/null || scancel -u "$(whoami)"
  echo "Cancelled Slurm flex_sim jobs for $(whoami)"
fi
