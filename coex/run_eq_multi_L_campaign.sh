#!/usr/bin/env bash
# Equilibrium (Ising-limit) coexistence campaign at multiple slab sizes.
#
# Physics identity: scheme=homo, Δf=-20, Δμ=0, k=0, flex-scheme=1
# Grid: ε ∈ [-1.8, -1.6] step 0.005 (same fineness as sus), initial μ window
# ±0.05 (half of the historical ±0.1), 10 μ points. Lx = 10 * Ly.
#
# Default sizes: Ly ∈ {16, 20, 40} → 160x16, 200x20, 400x40.
# Each Ly gets its OWN samples/results/manage/queue so dispatchers never collide.
#
# Layout (repo-root relative):
#   coex/coex_eq/ly<N>/{samples,results,manage.csv,queue.json}
#
# Usage (on Della login node, from repo root after sourcing env.sh):
#   ./coex/run_eq_multi_L_campaign.sh generate           # seed all default L
#   ./coex/run_eq_multi_L_campaign.sh generate 20 40     # only new sizes
#   ./coex/run_eq_multi_L_campaign.sh daemons            # dispatcher+analyzer per L
#   ./coex/run_eq_multi_L_campaign.sh daemons 20 40
#   ./coex/run_eq_multi_L_campaign.sh all 20 40          # generate + daemons
#   ./coex/run_eq_multi_L_campaign.sh status
#   tmux attach -t coex-eq-multiL                        # watch; Ctrl-b d to detach
#   tmux kill-session -t coex-eq-multiL                  # stop everything
#
# After coex finishes, criticality per L (example for Ly=20):
#   ./coex/run_eq_multi_L_criticality.sh 20
#   # -> criticality/equilibrium_multi_L_testing/ly20/
#
# Multi-L FSS-style comparison once all three are done:
#   ./coex/run_eq_multi_L_criticality.sh compare
#   # -> criticality/equilibrium_multi_L_testing/multi_L/

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

SCHEME=homo
FLEX_SCHEME=1
DELTA_F=-20.0
DELTA_MU=0.0
K=0.0
EPS_MIN=-1.8
EPS_MAX=-1.6
EPS_STEP=0.005
MU_WINDOW=0.05
N_MU_POINTS=10
DEFAULT_LYS=(16 20 40)
SESSION=coex-eq-multiL
# Cap per-L concurrent so 3 dispatchers stay near run_all's default budget.
MAX_CONCURRENT=30

CMD="${1:-all}"
shift || true
if [ "$#" -gt 0 ]; then
  LYS=("$@")
else
  LYS=("${DEFAULT_LYS[@]}")
fi

ly_tag() { echo "ly$1"; }

base_for() { echo "coex/coex_eq/$(ly_tag "$1")"; }

generate() {
  export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR"
  for ly in "${LYS[@]}"; do
    base="$(base_for "$ly")"
    mkdir -p "$base/samples" "$base/results"
    echo "== generating $base (Ly=$ly, Lx=$((10 * ly)), mu_window=±$MU_WINDOW) =="
    python -u coex/generate_susceptibility_coex.py \
      --scheme "$SCHEME" --flex-scheme "$FLEX_SCHEME" \
      --delta-f "$DELTA_F" --delta-mu "$DELTA_MU" --k "$K" \
      --ly "$ly" \
      --eps-min "$EPS_MIN" --eps-max "$EPS_MAX" --eps-step "$EPS_STEP" \
      --mu-window "$MU_WINDOW" --n-mu-points "$N_MU_POINTS" \
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
  for ly in "${LYS[@]}"; do
    tag="$(ly_tag "$ly")"
    base="$(base_for "$ly")"

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

  echo "Started tmux session '$SESSION' (disp-*/anlz-* windows per Ly)."
  echo "  attach: tmux attach -t $SESSION"
  echo "  stop:   tmux kill-session -t $SESSION"
}

status() {
  export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR"
  for ly in "${LYS[@]}"; do
    base="$(base_for "$ly")"
    python - "$base" "$ly" <<'PY'
import csv, json, os, sys
base, ly = sys.argv[1], sys.argv[2]
manage = os.path.join(base, "manage.csv")
results = os.path.join(base, "results")
queue = os.path.join(base, "queue.json")
n_manage = n_analyzed = n_results = n_pending = n_inflight = 0
if os.path.isfile(manage):
    rows = list(csv.DictReader(open(manage, newline="")))
    n_manage = len(rows)
    n_analyzed = sum(1 for r in rows if str(r.get("isAnalyzed", "")).strip())
if os.path.isdir(results):
    n_results = sum(1 for e in os.listdir(results) if os.path.isdir(os.path.join(results, e)))
if os.path.isfile(queue):
    m = json.load(open(queue))
    n_pending = len(m.get("pending", []))
    n_inflight = len(m.get("in_flight", {}))
print(
    f"Ly={ly} ({base}): manage={n_manage} analyzed={n_analyzed} "
    f"result_dirs={n_results} pending={n_pending} in_flight={n_inflight}"
)
PY
  done
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION': RUNNING"
    tmux list-windows -t "$SESSION"
  else
    echo "tmux session '$SESSION': not running"
  fi
}

case "$CMD" in
  generate) generate ;;
  daemons)  daemons ;;
  status)   status ;;
  all)      generate; daemons ;;
  *) echo "usage: $0 [generate|daemons|status|all] [Ly ...]"; exit 1 ;;
esac
