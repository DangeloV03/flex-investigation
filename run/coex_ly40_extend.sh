#!/usr/bin/env bash
# Extend a Ly = 40 coex campaign to more eps, appending to its existing ledger.
# Called through a per-campaign script (run/coex_ly40_extend_d3c.sh etc.), which sets
# EXP and DELTA_MU. Steps:
#
#   generate   back up manage.csv + queue.json, then add the new eps x mu jobs to them
#   start      run dispatcher + analyzer in tmux session coex-<EXP>-ly40 (own name, so
#              several campaigns can run side by side under one account)
#   all        generate + start
#   status     how many eps in the range have a fitted mu_coex; queue; tmux
#   stop       kill this campaign's tmux session only (queued Slurm jobs keep running)
#
# Grid: eps from EPS_MIN to EPS_MAX at EPS_STEP, Ly = 40 (Lx = 400), 10 mu values within
# +-0.05 of the FLEX prediction per eps (the analyzer adds more where needed). Same
# protocol as the campaign's existing Ly = 40 points.
#
# All paths are absolute (DATA_ROOT), so this works from any clone and any account.

set -euo pipefail
umask 002                                    # files stay group-writable for both accounts

: "${EXP:?set by the per-campaign script}"
: "${DELTA_MU:?set by the per-campaign script}"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/WJACOBS/vd7294/flex-investigation}"
SCHEME="${SCHEME:-positive_drive}"
FLEX_SCHEME="${FLEX_SCHEME:-2}"
DELTA_F="${DELTA_F:-0.0}"
K="${K:-1.0}"
LY=40
EPS_MIN="${EPS_MIN:--2.1}"
EPS_MAX="${EPS_MAX:--1.6}"
EPS_STEP="${EPS_STEP:-0.01}"
MU_WINDOW="${MU_WINDOW:-0.05}"
N_MU_POINTS="${N_MU_POINTS:-10}"
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="${WRAPPER:?set by the per-campaign script}"
BASE="$DATA_ROOT/COEX_RUNS_$EXP/ly$LY"
SESSION="coex-$EXP-ly$LY"
MARKER="$BASE/.extend_${EPS_MIN}_${EPS_MAX}_${EPS_STEP}"
export FLEX_OUTPUT_ROOT="$DATA_ROOT"

coex() { python "$REPO/run/coex.py" "$@"; }

[[ -d "$BASE" ]] || { echo "No $BASE (wrong DATA_ROOT or EXP?)" >&2; exit 1; }

# Fraction of the new eps that are analyzed: prints "<analyzed> <fitted> <total>".
progress() {
  python - "$BASE/manage.csv" "$EPS_MIN" "$EPS_MAX" <<'PY'
import csv, sys
lo, hi = float(sys.argv[2]) - 1e-9, float(sys.argv[3]) + 1e-9
rows = [r for r in csv.DictReader(open(sys.argv[1])) if lo <= float(r["epsilon"]) <= hi]
ana = sum(1 for r in rows if r.get("isAnalyzed", "").strip())
fit = sum(1 for r in rows if (r.get("mu_coex_FITTED") or "").strip().lower() not in ("", "nan"))
print(ana, fit, len(rows))
PY
}

do_generate() {
  if [[ -f "$MARKER" && "${FORCE:-0}" != 1 ]]; then
    echo "Already generated this grid ($MARKER). FORCE=1 to add it again." >&2
    exit 1
  fi
  local stamp; stamp=$(date +%Y%m%d_%H%M%S)
  cp -p "$BASE/manage.csv" "$BASE/manage.csv.bak_extend_$stamp"
  [[ -f "$BASE/queue.json" ]] && cp -p "$BASE/queue.json" "$BASE/queue.json.bak_extend_$stamp"
  mkdir -p "$BASE/samples" "$BASE/results"
  coex generate-susc --scheme "$SCHEME" --flex-scheme "$FLEX_SCHEME" \
      --delta-f "$DELTA_F" --delta-mu "$DELTA_MU" --k "$K" --ly "$LY" \
      --eps-min "$EPS_MIN" --eps-max "$EPS_MAX" --eps-step "$EPS_STEP" \
      --mu-window "$MU_WINDOW" --n-mu-points "$N_MU_POINTS" \
      --samples-dir "$BASE/samples" --results-dir "$BASE/results" \
      --manage "$BASE/manage.csv" --manifest "$BASE/queue.json"
  date > "$MARKER"
  echo "Backups: $BASE/manage.csv.bak_extend_$stamp (and queue.json)"
}

# tmux panes run these two (through the per-campaign script, so the settings carry over).
pane_env() {
  module load anaconda3/2024.10 2>/dev/null || true
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate lattice
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" PYTHONUNBUFFERED=1
  cd "$DATA_ROOT"
}

dispatch_loop() {
  pane_env
  # The dispatcher exits when the queue is momentarily empty, but the analyzer keeps
  # adding mu points; restart it until every new eps is analyzed.
  while true; do
    coex dispatch --manifest "$BASE/queue.json" --max-concurrent "$MAX_CONCURRENT"
    read -r ana fit total < <(progress)
    echo "[$(date +%H:%M)] analyzed $ana/$total eps in range (fitted $fit)"
    (( total > 0 && ana >= total )) && { echo "All eps in range analyzed. Dispatcher done."; break; }
    sleep 120
  done
}

analyze_loop() {
  pane_env
  coex analyze --results "$BASE/results" --manage "$BASE/manage.csv" \
      --samples "$BASE/samples" --manifest "$BASE/queue.json"
}

do_start() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session $SESSION already running (tmux attach -t $SESSION)."; exit 1
  fi
  [[ -f "$MARKER" ]] || { echo "Run '$SELF generate' first." >&2; exit 1; }
  tmux new-session -d -s "$SESSION" -n dispatcher -c "$DATA_ROOT"
  tmux send-keys -t "$SESSION:dispatcher" "bash $SELF _dispatch_loop" C-m
  tmux new-window -t "$SESSION" -n analyzer -c "$DATA_ROOT"
  tmux send-keys -t "$SESSION:analyzer" "bash $SELF _analyze_loop" C-m
  echo "Started tmux session $SESSION on $(hostname -s)."
  echo "  watch: tmux attach -t $SESSION   (from $(hostname -s); detach with Ctrl-b d)"
}

do_status() {
  read -r ana fit total < <(progress)
  echo "$EXP Ly=$LY, eps $EPS_MIN..$EPS_MAX (incl. any older points in that range): $ana/$total analyzed, $fit fitted"
  python - "$BASE/queue.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"queue: {len(m.get('pending', []))} pending, {len(m.get('in_flight', {}))} in flight")
PY
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux $SESSION: running on $(hostname -s)"
  else
    echo "tmux $SESSION: not running on $(hostname -s) (it may be on another login node)"
  fi
  echo "your flex_sim jobs in Slurm (all campaigns): $(squeue -u "$USER" -n flex_sim -h 2>/dev/null | wc -l)"
}

do_stop() {
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "Stopped $SESSION." || echo "No $SESSION here."
}

case "${1:-}" in
  generate) do_generate ;;
  start)    do_start ;;
  all)      do_generate; do_start ;;
  status)   do_status ;;
  stop)     do_stop ;;
  _dispatch_loop) dispatch_loop ;;
  _analyze_loop)  analyze_loop ;;
  *) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
