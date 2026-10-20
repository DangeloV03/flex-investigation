#!/usr/bin/env bash
# Equilibrium coex benchmark (experiments branch, 2026-10-19).
#
# Why: fitted slab coex mu gave <m> != 0 in the susc runs. Here the exact answer
# is known: with delta_mu = 0 the model is in equilibrium and
#   mu_coex = -ln(exp(2|eps|) - exp(delta_f)).
# We run the same slab coex protocol as C2A at Ly 16/20/40 and compare.
#
# Physics: scheme=homo, beta*delta_f=0, k=1, beta*delta_mu=0, flex-scheme=1
# Sizes:   Ly in {16, 20, 40}, Lx = 10*Ly
# Grid:    eps in [-1.9, -1.5] step 0.02 (21), mu +-0.05 around exact, 10 points
#
# Copied from coex/run_c2a_coex_campaign.sh. Code runs from the main Della
# checkout (REPO_DIR, not edited); data goes to CAMPAIGN_ROOT outside it.
#
# Usage (Della login node, any directory):
#   bash run_eq_coex.sh generate 16      # call ONCE per Ly (2nd call wipes fits)
#   bash run_eq_coex.sh daemons          # tmux session coex-EQ
#   bash run_eq_coex.sh status

set -euo pipefail

PROJECT_DIR=/scratch/gpfs/WJACOBS/vd7294/flex-investigation
cd "$PROJECT_DIR"

CAMPAIGN_ROOT=/scratch/gpfs/WJACOBS/vd7294/flex-experiments/COEX_RUNS_EQ_DMU0
SCHEME=homo
FLEX_SCHEME=1
DELTA_F=0.0
DELTA_MU=0.0
K=1.0
EPS_MIN=-1.9
EPS_MAX=-1.5
EPS_STEP=0.02
MU_WINDOW=0.05
N_MU_POINTS=10
DEFAULT_LYS=(16 20 40)
SESSION=coex-EQ
# Cap per-L concurrent so 3 dispatchers stay near run_all's default budget.
MAX_CONCURRENT=30

CMD="${1:-status}"
shift || true
if [ "$#" -gt 0 ]; then
  LYS=("$@")
else
  LYS=("${DEFAULT_LYS[@]}")
fi

base_for() { echo "$CAMPAIGN_ROOT/ly$1"; }

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
  for ly in "${LYS[@]}"; do
    if [ ! -f "$(base_for "$ly")/queue.json" ]; then
      echo "No $(base_for "$ly")/queue.json — run '$0 generate $ly' first."
      exit 1
    fi
  done

  tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR"
  local first=1
  for ly in "${LYS[@]}"; do
    base="$(base_for "$ly")"
    if [ "$first" -eq 1 ]; then
      tmux rename-window -t "$SESSION" "disp-ly$ly"
      first=0
    else
      tmux new-window -t "$SESSION" -n "disp-ly$ly" -c "$PROJECT_DIR"
    fi
    tmux send-keys -t "$SESSION:disp-ly$ly" \
      "${DAEMON_SETUP}; python -u coex/run_all.py --manifest $base/queue.json --max-concurrent $MAX_CONCURRENT" C-m

    tmux new-window -t "$SESSION" -n "anlz-ly$ly" -c "$PROJECT_DIR"
    tmux send-keys -t "$SESSION:anlz-ly$ly" \
      "${DAEMON_SETUP}; python -u coex/analyzer.py --results $base/results --manage $base/manage.csv --samples $base/samples --manifest $base/queue.json" C-m
  done

  echo "Started tmux session '$SESSION' on $(hostname -s) (disp-ly*/anlz-ly* per Ly: ${LYS[*]})."
  echo "  attach: tmux attach -t $SESSION   (from $(hostname -s))"
  echo "  stop:   tmux kill-session -t $SESSION"
}

status() {
  for ly in "${LYS[@]}"; do
    python - "$(base_for "$ly")" "$ly" <<'PY'
import csv, json, os, sys
base, ly = sys.argv[1], sys.argv[2]
manage = os.path.join(base, "manage.csv")
results = os.path.join(base, "results")
queue = os.path.join(base, "queue.json")
n_manage = n_analyzed = n_fitted = n_results = n_pending = n_inflight = 0
if os.path.isfile(manage):
    rows = list(csv.DictReader(open(manage, newline="")))
    n_manage = len(rows)
    n_analyzed = sum(1 for r in rows if str(r.get("isAnalyzed", "")).strip())
    n_fitted = sum(1 for r in rows
                   if str(r.get("mu_coex_FITTED", "")).strip().lower() not in ("", "nan"))
if os.path.isdir(results):
    n_results = sum(1 for e in os.listdir(results) if os.path.isdir(os.path.join(results, e)))
if os.path.isfile(queue):
    m = json.load(open(queue))
    n_pending = len(m.get("pending", []))
    n_inflight = len(m.get("in_flight", {}))
print(
    f"Ly={ly} ({base}): manage={n_manage} analyzed={n_analyzed} "
    f"mu_coex_fitted={n_fitted} result_dirs={n_results} pending={n_pending} in_flight={n_inflight}"
)
PY
  done
  if command -v tmux >/dev/null && tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION': RUNNING on $(hostname -s)"
    tmux list-windows -t "$SESSION"
  else
    echo "tmux session '$SESSION': not running on $(hostname -s)"
  fi
}

case "$CMD" in
  generate) generate ;;
  daemons)  daemons ;;
  status)   status ;;
  all)      generate; daemons ;;
  *) echo "usage: $0 [generate|daemons|status|all] [Ly ...]"; exit 1 ;;
esac
