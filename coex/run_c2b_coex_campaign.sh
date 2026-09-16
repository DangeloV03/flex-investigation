#!/usr/bin/env bash
# Experiment C2B: coexistence μ-sweeps at multiple slab sizes, negative drive.
#
# Physics: scheme=homo, βΔf=0, k=1, βΔμ=-1, flex-scheme=1
# Sizes:   Ly ∈ {16, 20}, Lx = 10*Ly
#
# C2B has no simulated anchor yet (C2A's ε window came from the Ly=16 Δμ=+1 run
# that put ε_c at -1.705), so this runs in two phases:
#
#   scout   Ly=16 only, ε ∈ [-2.0, -1.3] step 0.02 (36 ε × 10 μ = 360 jobs).
#           Coarse enough to finish on the workstation and locate ε_c.
#   refine  Both Ly, ε ∈ [ε_c-0.2, ε_c+0.2] step 0.005 (81 ε × 10 μ per Ly),
#           matching C2A's resolution. Pass ε_c from the scout's criticality fit:
#             ./coex/run_c2b_coex_campaign.sh refine -1.42
#
# Everything for C2B lives under COEX_RUNS_C2B/ (repo root):
#   COEX_RUNS_C2B/ly<N>/{samples,results,manage.csv,queue.json}   coex (this script)
#   COEX_RUNS_C2B/criticality/ly<N>/  and  .../multi_L/            ./coex/run_c2b_criticality.sh
# Each Ly has its own queue, dispatcher and analyzer, so sizes never collide.
#
# This campaign targets the wjacobs workstation, which has no Slurm: run_all.py
# sees no sbatch and runs each job as a local subprocess. Each job forks
# num_parallel_runs=8 replica processes, so the dispatcher cap is sized to keep
# RESERVED_CORES cores free for interactive work.
#
# Usage (workstation, repo root):
#   ./coex/run_c2b_coex_campaign.sh scout            # generate + daemons, Ly=16 coarse
#   ./coex/run_c2b_coex_campaign.sh status
#   ./coex/run_c2b_coex_campaign.sh refine -1.42     # generate + daemons, Ly 16 20 fine
#   tmux attach -t coex-C2B                          # Ctrl-b d to detach
#   tmux kill-session -t coex-C2B                    # stop dispatchers/analyzers
#
# Overrides: RESERVED_CORES=4 MAX_CONCURRENT= LYS= EPS_MIN= EPS_MAX= EPS_STEP=
#
# After every Ly shows mu_coex_fitted equal to its ε count:
#   ./coex/run_c2b_criticality.sh && ./coex/run_c2b_criticality.sh compare

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CAMPAIGN_ROOT=COEX_RUNS_C2B
SCHEME=homo
FLEX_SCHEME=1
DELTA_F=0.0
DELTA_MU=-1.0
K=1.0
MU_WINDOW=0.05
N_MU_POINTS=10
SESSION=coex-C2B

# json_runner.py forks this many replicas per job (generate's num_parallel_runs).
REPLICAS_PER_JOB=8
# Cores deliberately left idle for interactive work on the workstation.
RESERVED_CORES="${RESERVED_CORES:-4}"

# Scout defaults; refine overrides both the window and the Ly list.
SCOUT_LYS=(16)
SCOUT_EPS_MIN=-2.0
SCOUT_EPS_MAX=-1.3
SCOUT_EPS_STEP=0.02
REFINE_LYS=(16 20)
REFINE_HALF_WIDTH=0.2
REFINE_EPS_STEP=0.005

core_count() {
  if command -v nproc >/dev/null; then
    nproc
  else
    sysctl -n hw.ncpu
  fi
}

# Total concurrent jobs across every dispatcher, then per-Ly share (min 1).
concurrency_per_ly() {
  local n_lys="$1" cores usable total
  if [ -n "${MAX_CONCURRENT:-}" ]; then
    total="$MAX_CONCURRENT"
  else
    cores="$(core_count)"
    usable=$((cores - RESERVED_CORES))
    total=$((usable / REPLICAS_PER_JOB))
  fi
  if [ "$total" -lt 1 ]; then
    total=1
  fi
  local per=$((total / n_lys))
  if [ "$per" -lt 1 ]; then
    per=1
  fi
  echo "$per"
}

base_for() { echo "$CAMPAIGN_ROOT/ly$1"; }

generate() {
  export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR"
  for ly in "${LYS[@]}"; do
    base="$(base_for "$ly")"
    mkdir -p "$base/samples" "$base/results"
    echo "== generating $base (Ly=$ly, Lx=$((10 * ly)), ε ∈ [$EPS_MIN, $EPS_MAX] step $EPS_STEP) =="
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
DAEMON_SETUP="source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate lattice; export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\"; export PYTHONPATH=\"$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR\"; export PYTHONUNBUFFERED=1"

daemons() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists."
    echo "  attach: tmux attach -t $SESSION"
    echo "  stop:   tmux kill-session -t $SESSION"
    exit 1
  fi
  for ly in "${LYS[@]}"; do
    if [ ! -f "$(base_for "$ly")/queue.json" ]; then
      echo "No $(base_for "$ly")/queue.json — generate first."
      exit 1
    fi
  done

  local per_ly
  per_ly="$(concurrency_per_ly "${#LYS[@]}")"
  echo "Cores: $(core_count) total, $RESERVED_CORES reserved, ${REPLICAS_PER_JOB} per job"
  echo "  -> --max-concurrent $per_ly per Ly (${#LYS[@]} dispatcher(s))"

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
      "${DAEMON_SETUP}; python -u coex/run_all.py --manifest $base/queue.json --max-concurrent $per_ly" C-m

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

CMD="${1:-status}"
shift || true

case "$CMD" in
  scout)
    LYS=("${SCOUT_LYS[@]}")
    EPS_MIN="${EPS_MIN:-$SCOUT_EPS_MIN}"
    EPS_MAX="${EPS_MAX:-$SCOUT_EPS_MAX}"
    EPS_STEP="${EPS_STEP:-$SCOUT_EPS_STEP}"
    generate
    daemons
    ;;
  refine)
    EPS_C="${1:-}"
    if [ -z "$EPS_C" ]; then
      echo "usage: $0 refine <epsilon_c>   (from the scout's criticality.csv)"
      exit 1
    fi
    LYS=("${REFINE_LYS[@]}")
    EPS_MIN="${EPS_MIN:-$(python -c "print(round($EPS_C - $REFINE_HALF_WIDTH, 6))")}"
    EPS_MAX="${EPS_MAX:-$(python -c "print(round($EPS_C + $REFINE_HALF_WIDTH, 6))")}"
    EPS_STEP="${EPS_STEP:-$REFINE_EPS_STEP}"
    generate
    daemons
    ;;
  status)
    LYS=(${LYS:-16 20})
    status
    ;;
  *)
    echo "usage: $0 [scout|refine <epsilon_c>|status]"
    exit 1
    ;;
esac
