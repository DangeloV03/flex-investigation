#!/usr/bin/env bash
# Refine the driven-homo Δμ=1 coex campaign (Δf=0, k=1, scheme=homo, flex-scheme 1)
# from ε step 0.01 to 0.005 -- i.e. fill in the 60 midpoint ε values.
#
# Reuses the EXISTING isolated dmu1p0 paths, so the generator dedups the 61
# points already at 0.01 and only adds the new 0.005 midpoints
# (ε = -1.995, -1.985, ... -1.405 -> 60 combos x 10 mu = 600 new jobs).
# Existing rows / results / mu_coex_FITTED are left untouched (append + dedup).
#
# Usage (on Della login node):
#   ./coex/run_dmu1p0_refine.sh generate   # add the 600 midpoint jobs (idempotent)
#   ./coex/run_dmu1p0_refine.sh daemons     # dispatcher + analyzer in a new tmux session
#   ./coex/run_dmu1p0_refine.sh all         # generate, then launch (default)
#   tmux attach -t coex-dmu1p0-refine        # watch; Ctrl-b d to detach
#   tmux kill-session -t coex-dmu1p0-refine  # stop
#
# NOTE: make sure no OLD dmu1p0 dispatcher/analyzer is still running against these
# same paths (a second dispatcher on the same queue would double-submit).

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# --- driven-homo Δμ=1 campaign identity + isolated paths ---
SCHEME=homo
FLEX_SCHEME=1
DELTA_F=0.0
K=1.0
DELTA_MU=1.0
LY=16
EPS_MIN=-2.0
EPS_MAX=-1.4
EPS_STEP=0.005

SAMPLES=coex_samples/homo_dmu1p0
RESULTS=susceptibility_results/coex_homo_dmu1p0
MANAGE=coex_manage_homo_dmu1p0.csv
QUEUE=coex_homo_dmu1p0_queue.json
SESSION=coex-dmu1p0-refine

generate() {
  export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR"
  echo "== filling in ε step $EPS_STEP for $MANAGE (dedups existing 0.01 pts) =="
  python -u coex/generate_susceptibility_coex.py \
    --scheme "$SCHEME" --flex-scheme "$FLEX_SCHEME" \
    --delta-f "$DELTA_F" --k "$K" --delta-mu "$DELTA_MU" --ly "$LY" \
    --eps-min "$EPS_MIN" --eps-max "$EPS_MAX" --eps-step "$EPS_STEP" \
    --samples-dir "$SAMPLES" \
    --results-dir "$RESULTS" \
    --manage "$MANAGE" \
    --manifest "$QUEUE"
}

# Non-interactive tmux panes need conda.sh sourced before `conda activate`.
# $PROJECT_DIR is expanded now; the \$(...) / \${...} stay literal for the pane.
DAEMON_SETUP="module load anaconda3/2024.10 2>/dev/null; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate lattice; export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\"; export PYTHONPATH=\"$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR\"; export PYTHONUNBUFFERED=1"

daemons() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists."
    echo "  attach: tmux attach -t $SESSION"
    echo "  stop:   tmux kill-session -t $SESSION"
    exit 1
  fi

  tmux new-session -d -s "$SESSION" -n dispatcher -c "$PROJECT_DIR"
  tmux send-keys -t "$SESSION:dispatcher" \
    "${DAEMON_SETUP}; python -u coex/run_all.py --manifest $QUEUE" C-m

  tmux new-window -t "$SESSION" -n analyzer -c "$PROJECT_DIR"
  tmux send-keys -t "$SESSION:analyzer" \
    "${DAEMON_SETUP}; python -u coex/analyzer.py --results $RESULTS --manage $MANAGE --samples $SAMPLES --manifest $QUEUE" C-m

  echo "Started tmux session '$SESSION' (windows: dispatcher, analyzer)."
  echo "  attach: tmux attach -t $SESSION"
  echo "  stop:   tmux kill-session -t $SESSION"
}

case "${1:-all}" in
  generate) generate ;;
  daemons)  daemons ;;
  all)      generate; daemons ;;
  *) echo "usage: $0 [generate|daemons|all]"; exit 1 ;;
esac
