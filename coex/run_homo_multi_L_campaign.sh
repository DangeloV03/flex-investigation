#!/usr/bin/env bash
# Experiment C2A: homogeneous driven coexistence at multiple slab sizes.
#
# Physics identity: scheme=homo, Δf=0, k=1, flex-scheme=1; Δμ via --dmu (default 1.0)
# Grid: ε ∈ [-1.9, -1.5] step 0.005, μ window ±0.05, 10 μ points. Lx = 10*Ly.
#   ε window: the single-L (Ly=16) homo Δμ=1 run put the BC transition at
#   βε ≈ -1.80…-1.66 (sigmoid ε_c = -1.705) with BC = 5/9 near -1.57, so this
#   brackets it with room for ε_c to shift at Ly=20/40.
#   μ window: FLEX was within 0.013 of the fitted μ_coex at Ly=16, so ±0.05 is
#   ample; the analyzer extends the sweep if a sign change is not bracketed.
#
# Default sizes: Ly ∈ {16, 20, 40}
#
# Layout:
#   homo/dmu<X>_multiL/ly<N>/{samples,results,manage.csv,queue.json}
#
# Usage (Della login node, repo root, after env.sh):
#   ./coex/run_homo_multi_L_campaign.sh all                  # Δμ=1.0, Ly 16 20 40
#   ./coex/run_homo_multi_L_campaign.sh all --lys 20 40
#   ./coex/run_homo_multi_L_campaign.sh generate --dmu 1.0 --lys 16 20 40
#   ./coex/run_homo_multi_L_campaign.sh daemons
#   ./coex/run_homo_multi_L_campaign.sh status
#   tmux attach -t homo-multiL
#   tmux kill-session -t homo-multiL
#
# After coex finishes:
#   ./coex/run_homo_multi_L_criticality.sh
#   ./coex/run_homo_multi_L_criticality.sh compare

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

SCHEME=homo
FLEX_SCHEME=1
DELTA_F=0.0
K=1.0
EPS_MIN=-1.9
EPS_MAX=-1.5
EPS_STEP=0.005
MU_WINDOW=0.05
N_MU_POINTS=10
DEFAULT_LYS=(16 20 40)
DEFAULT_DMUS=(1.0)
SESSION=homo-multiL
# Cap per (Δμ, L) dispatcher; 3 dispatchers at Δμ=1 stay near run_all's budget.
MAX_CONCURRENT=30

CMD="${1:-all}"
shift || true

DMUS=()
LYS=()
MODE=lys
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dmu) MODE=dmu; shift ;;
    --lys) MODE=lys; shift ;;
    *)
      if [ "$MODE" = "dmu" ]; then DMUS+=("$1"); else LYS+=("$1"); fi
      shift
      ;;
  esac
done
[ "${#DMUS[@]}" -eq 0 ] && DMUS=("${DEFAULT_DMUS[@]}")
[ "${#LYS[@]}" -eq 0 ] && LYS=("${DEFAULT_LYS[@]}")

dmu_tag() { echo "dmu$(echo "$1" | sed 's/\./p/')"; }
ly_tag() { echo "ly$1"; }
coex_root_for() { echo "homo/$(dmu_tag "$1")_multiL"; }
base_for() { echo "$(coex_root_for "$1")/$(ly_tag "$2")"; }

generate() {
  export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR"
  for dmu in "${DMUS[@]}"; do
    for ly in "${LYS[@]}"; do
      base="$(base_for "$dmu" "$ly")"
      mkdir -p "$base/samples" "$base/results"
      echo "== generating $base (Ly=$ly, Lx=$((10 * ly)), Δμ=$dmu, mu_window=±$MU_WINDOW) =="
      python -u coex/generate_susceptibility_coex.py \
        --scheme "$SCHEME" --flex-scheme "$FLEX_SCHEME" \
        --delta-f "$DELTA_F" --delta-mu "$dmu" --k "$K" \
        --ly "$ly" \
        --eps-min "$EPS_MIN" --eps-max "$EPS_MAX" --eps-step "$EPS_STEP" \
        --mu-window "$MU_WINDOW" --n-mu-points "$N_MU_POINTS" \
        --samples-dir "$base/samples" \
        --results-dir "$base/results" \
        --manage "$base/manage.csv" \
        --manifest "$base/queue.json"
    done
  done
}

# Non-interactive tmux panes need conda.sh sourced before `conda activate`.
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
    local dtag
    dtag="$(dmu_tag "$dmu")"
    for ly in "${LYS[@]}"; do
      local ltag base wdisp wanlz
      ltag="$(ly_tag "$ly")"
      base="$(base_for "$dmu" "$ly")"
      wdisp="disp-${dtag}-${ltag}"
      wanlz="anlz-${dtag}-${ltag}"

      if [ "$first" -eq 1 ]; then
        tmux rename-window -t "$SESSION" "$wdisp"
        first=0
      else
        tmux new-window -t "$SESSION" -n "$wdisp" -c "$PROJECT_DIR"
      fi
      tmux send-keys -t "$SESSION:$wdisp" \
        "${DAEMON_SETUP}; python -u coex/run_all.py --manifest $base/queue.json --max-concurrent $MAX_CONCURRENT" C-m

      tmux new-window -t "$SESSION" -n "$wanlz" -c "$PROJECT_DIR"
      tmux send-keys -t "$SESSION:$wanlz" \
        "${DAEMON_SETUP}; python -u coex/analyzer.py --results $base/results --manage $base/manage.csv --samples $base/samples --manifest $base/queue.json" C-m
    done
  done

  echo "Started tmux session '$SESSION' (disp-*/anlz-* per Δμ×Ly)."
  echo "  Δμ=${DMUS[*]}  Ly=${LYS[*]}"
  echo "  attach: tmux attach -t $SESSION"
  echo "  stop:   tmux kill-session -t $SESSION"
}

status() {
  export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR"
  for dmu in "${DMUS[@]}"; do
    for ly in "${LYS[@]}"; do
      base="$(base_for "$dmu" "$ly")"
      python - "$base" "$ly" "$dmu" <<'PY'
import csv, json, os, sys
base, ly, dmu = sys.argv[1], sys.argv[2], sys.argv[3]
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
    f"Δμ={dmu} Ly={ly} ({base}): manage={n_manage} analyzed={n_analyzed} "
    f"mu_coex_fitted={n_fitted} result_dirs={n_results} pending={n_pending} in_flight={n_inflight}"
)
PY
    done
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
  *)
    echo "usage: $0 [generate|daemons|status|all] [--dmu 1.0 ...] [--lys 16 20 40]"
    exit 1
    ;;
esac
