#!/usr/bin/env bash
# Scheme-3 (negative_drive) coexistence campaign: Δf=0.0, k=0.1, flex-scheme 3.
# Five isolated Δμ sweeps (1.0, 2.0, 3.0, 4.0, 4.5) over ε ∈ [-2.5, -1.6] step 0.01.
#
# Layout produced (repo-root relative):
#   scheme3/dmu<X>/{samples,results,manage.csv,queue.json}
#
# Each sweep gets its OWN dispatcher + analyzer so queues/manage never collide.
# Per-sweep --max-concurrent is capped so 5 dispatchers stay within one campaign's
# Slurm budget (5 * 20 = 100, matching run_all.py's default MAX_CONCURRENT).
#
# Usage (on Della login node):
#   ./coex/run_scheme3_campaign.sh generate   # write the 5 sweeps (idempotent)
#   ./coex/run_scheme3_campaign.sh daemons     # launch tmux dispatchers + analyzers
#   ./coex/run_scheme3_campaign.sh all         # generate, then launch (default)
#   tmux attach -t scheme3-coex                # watch; Ctrl-b d to detach
#   tmux kill-session -t scheme3-coex          # stop everything

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

SCHEME=negative_drive
FLEX_SCHEME=3
DELTA_F=0.0
K=0.1
EPS_MIN=-2.5
EPS_MAX=-1.6
EPS_STEP=0.01
DMUS=(1.0 2.0 3.0 4.0 4.5)
SESSION=scheme3-coex
MAX_CONCURRENT=20

dmu_tag() { echo "dmu$(echo "$1" | sed 's/\./p/')"; }

generate() {
  export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR"
  for dmu in "${DMUS[@]}"; do
    base="scheme3/$(dmu_tag "$dmu")"
    echo "== generating $base (delta_mu=$dmu) =="
    python -u coex/generate_susceptibility_coex.py \
      --scheme "$SCHEME" --flex-scheme "$FLEX_SCHEME" \
      --delta-f "$DELTA_F" --k "$K" --delta-mu "$dmu" \
      --eps-min "$EPS_MIN" --eps-max "$EPS_MAX" --eps-step "$EPS_STEP" \
      --samples-dir "$base/samples" \
      --results-dir "$base/results" \
      --manage "$base/manage.csv" \
      --manifest "$base/queue.json"
  done
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

  tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR"
  local first=1
  for dmu in "${DMUS[@]}"; do
    tag=$(dmu_tag "$dmu")
    base="scheme3/$tag"

    if [ "$first" -eq 1 ]; then
      tmux rename-window -t "$SESSION" "disp-$tag"
      first=0
    else
      tmux new-window -t "$SESSION" -n "disp-$tag" -c "$PROJECT_DIR"
    fi
    tmux send-keys -t "$SESSION:disp-$tag" \
      "${DAEMON_SETUP}; python -u coex/run_all.py --manifest $base/queue.json --max-concurrent $MAX_CONCURRENT" C-m

    tmux new-window -t "$SESSION" -n "anlz-$tag" -c "$PROJECT_DIR"
    tmux send-keys -t "$SESSION:anlz-$tag" \
      "${DAEMON_SETUP}; python -u coex/analyzer.py --results $base/results --manage $base/manage.csv --samples $base/samples --manifest $base/queue.json" C-m
  done

  echo "Started tmux session '$SESSION' (disp-*/anlz-* windows per Δμ)."
  echo "  attach: tmux attach -t $SESSION"
  echo "  stop:   tmux kill-session -t $SESSION"
}

case "${1:-all}" in
  generate) generate ;;
  daemons)  daemons ;;
  all)      generate; daemons ;;
  *) echo "usage: $0 [generate|daemons|all]"; exit 1 ;;
esac
