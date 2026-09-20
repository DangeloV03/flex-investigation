#!/usr/bin/env bash
# Experiments C3A–C3E: negative-drive coexistence μ-sweeps at multiple slab sizes.
#
# Physics identity (all five): scheme=negative_drive, flex-scheme=3, βΔf=0, k=1
# Sizes: Ly ∈ {16, 20} on the workstation, Ly = 40 on Della. Lx = 10*Ly.
#
# ε_c has never been measured for negative drive at k=1, so this runs in two
# phases rather than guessing a window:
#
#   scout   Ly=16 only, step 0.05 over a deliberately wide window (~40 ε,
#           400 jobs per experiment). Cheap, and wide enough that it does not
#           depend on any extrapolation being right.
#   refine  All Ly, ε_c ± 0.3 step 0.005 (121 ε, 1210 jobs per Ly).
#           Pass ε_c from the scout's criticality fit:
#             ./coex/run_c3_coex_campaign.sh C3A refine -1.83
#
# Why ε_c ± 0.3 at 0.005: that is exactly the production susceptibility grid.
# susceptibility/sweep_susceptibility.py defaults to ε ∈ [-2.0, -1.4] step 0.005
# (121 values; PIPELINE.md calls this the standard 121-job run), and that window
# is centred on -1.7 — the measured ε_c of -1.705 for homo k=1 Δμ=+1. So the
# convention is ε_c ± 0.3 at 0.005, and coex must cover every ε that production
# will later need, since mu_coex_SIM from this phase feeds the square-L prod runs
# at the same ε. Note this is WIDER than C2A/C2B's ±0.2 — those were criticality
# and FSS experiments that never had to feed a susceptibility sweep.
#
# Scout windows: upper edge is the hard FLEX cutoff. Negative drive uses
# k_eff = min(1, exp(-Δf-Δμ-2ε))·k, which at Δf=0, k=1 saturates to 1 for
# ε < -Δμ/2, and the generator drops any ε whose μ_coex_FLEX > 0 — a boundary
# sitting just above that same point (-0.655/-1.060/-1.520/-2.005 for Δμ=1/2/3/4;
# Δμ=-1 never crosses). Above the cutoff the generator silently skips, so there
# is nothing to be gained by scanning there. Lower edges are generous.
#
#   Exp    βΔμ    scout ε window      #ε   scout jobs   FLEX cutoff
#   C3A    +1.0   [-2.60, -0.70]      39      390         -0.655
#   C3B    -1.0   [-2.80, -1.00]      37      370         (none)
#   C3C    +2.0   [-3.00, -1.10]      39      390         -1.060
#   C3D    +3.0   [-3.50, -1.55]      40      400         -1.520
#   C3E    +4.0   [-4.00, -2.05]      40      400         -2.005
#
# Everything for an experiment lives under COEX_RUNS_<EXP>/ (repo root):
#   COEX_RUNS_C3A/ly<N>/{samples,results,manage.csv,queue.json}   coex (this script)
#   COEX_RUNS_C3A/criticality/ly<N>/  and  .../multi_L/           ./coex/run_c3_criticality.sh
# The layout is identical on both machines, so once Della's ly40 is done a plain
#   rsync -a della:.../COEX_RUNS_C3A/ly40 ./COEX_RUNS_C3A/
# merges all three sizes under one root for the FSS step.
#
# Host handling is automatic: with sbatch on PATH (Della) run_all.py submits to
# Slurm; without it (wjacobs workstation) each job runs as a local subprocess
# forking num_parallel_runs=8 replicas, and the dispatcher cap is sized to leave
# RESERVED_CORES free for interactive work.
#
# Usage (repo root, after env.sh):
#   ./coex/run_c3_coex_campaign.sh C3A scout          # Ly=16 coarse, then daemons
#   ./coex/run_c3_coex_campaign.sh C3A status
#   ./coex/run_c3_criticality.sh   C3A 16             # read ε_c off the scout
#   ./coex/run_c3_coex_campaign.sh C3A refine -1.83   # all Ly, production grid
#   tmux attach -t coex-C3A                           # Ctrl-b d to detach
#   tmux kill-session -t coex-C3A                     # stop dispatchers/analyzers
#
# The scout and the refine reuse the same tmux session name, so kill the scout's
# session before launching the refine.
#
# Run ONE experiment's refine at a time on the workstation — five at once is
# ~12k jobs at 8 replicas each. Scouts are small enough to run together.
#
# Overrides: RESERVED_CORES= MAX_CONCURRENT= LYS= EPS_MIN= EPS_MAX= EPS_STEP=
#
# After every Ly shows mu_coex_fitted = #ε:
#   ./coex/run_c3_criticality.sh C3A && ./coex/run_c3_criticality.sh C3A compare

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

SCHEME=negative_drive
FLEX_SCHEME=3
DELTA_F=0.0
K=1.0
MU_WINDOW=0.05
N_MU_POINTS=10

SCOUT_EPS_STEP=0.05
SCOUT_LYS=(16)
# Matches the production susceptibility grid: 0.6 wide at 0.005 = 121 ε.
REFINE_HALF_WIDTH=0.3
REFINE_EPS_STEP=0.005

# json_runner.py forks this many replicas per job (generate's num_parallel_runs).
REPLICAS_PER_JOB=8
# Cores deliberately left idle for interactive work on the workstation.
RESERVED_CORES="${RESERVED_CORES:-4}"
# Per-dispatcher cap on Della, where Slurm queues the overflow anyway.
SLURM_MAX_CONCURRENT=30

EXP="${1:-}"
if [ -z "$EXP" ]; then
  echo "usage: $0 <C3A|C3B|C3C|C3D|C3E> [scout|refine <eps_c>|generate|daemons|status] [--lys N ...]"
  exit 1
fi
shift
EXP="$(echo "$EXP" | tr '[:lower:]' '[:upper:]')"

# Per-experiment Δμ, scout window, and the ε above which the generator skips.
case "$EXP" in
  C3A) DELTA_MU=1.0;  SCOUT_MIN=-2.60; SCOUT_MAX=-0.70; FLEX_CUTOFF=-0.655 ;;
  C3B) DELTA_MU=-1.0; SCOUT_MIN=-2.80; SCOUT_MAX=-1.00; FLEX_CUTOFF= ;;
  C3C) DELTA_MU=2.0;  SCOUT_MIN=-3.00; SCOUT_MAX=-1.10; FLEX_CUTOFF=-1.060 ;;
  C3D) DELTA_MU=3.0;  SCOUT_MIN=-3.50; SCOUT_MAX=-1.55; FLEX_CUTOFF=-1.520 ;;
  C3E) DELTA_MU=4.0;  SCOUT_MIN=-4.00; SCOUT_MAX=-2.05; FLEX_CUTOFF=-2.005 ;;
  *) echo "Unknown experiment '$EXP' (expected C3A..C3E)"; exit 1 ;;
esac

CAMPAIGN_ROOT="COEX_RUNS_$EXP"
SESSION="coex-$EXP"

has_slurm() { command -v sbatch >/dev/null 2>&1; }

default_lys() {
  # Della takes the expensive size; the workstation takes the two small ones.
  if has_slurm; then echo "40"; else echo "16 20"; fi
}

core_count() {
  if command -v nproc >/dev/null; then nproc; else sysctl -n hw.ncpu; fi
}

# Total concurrent jobs across every dispatcher, then per-Ly share (min 1).
concurrency_per_ly() {
  local n_lys="$1" cores usable total per
  if [ -n "${MAX_CONCURRENT:-}" ]; then
    total="$MAX_CONCURRENT"
  elif has_slurm; then
    total=$((SLURM_MAX_CONCURRENT * n_lys))
  else
    cores="$(core_count)"
    usable=$((cores - RESERVED_CORES))
    total=$((usable / REPLICAS_PER_JOB))
  fi
  [ "$total" -lt 1 ] && total=1
  per=$((total / n_lys))
  [ "$per" -lt 1 ] && per=1
  echo "$per"
}

base_for() { echo "$CAMPAIGN_ROOT/ly$1"; }

n_eps() { python -c "print(int(round(($EPS_MAX - ($EPS_MIN)) / $EPS_STEP)) + 1)"; }

generate() {
  export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR"
  echo "== $EXP: scheme=$SCHEME flex=$FLEX_SCHEME Δf=$DELTA_F Δμ=$DELTA_MU k=$K =="
  echo "   ε ∈ [$EPS_MIN, $EPS_MAX] step $EPS_STEP ($(n_eps) values) × $N_MU_POINTS μ points"
  if [ -n "$FLEX_CUTOFF" ]; then
    if python -c "import sys; sys.exit(0 if $EPS_MAX >= $FLEX_CUTOFF else 1)"; then
      echo "   !! ε_max=$EPS_MAX is at/above the FLEX cutoff $FLEX_CUTOFF —"
      echo "      the generator will skip those ε (μ_coex_FLEX > 0). Lower --eps-max."
    fi
  fi
  for ly in "${LYS[@]}"; do
    base="$(base_for "$ly")"
    mkdir -p "$base/samples" "$base/results"
    echo "== generating $base (Ly=$ly, Lx=$((10 * ly))) =="
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
    echo "Session '$SESSION' already exists (the scout's, if this is the refine)."
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
  if has_slurm; then
    echo "Slurm detected: --max-concurrent $per_ly per Ly (${#LYS[@]} dispatcher(s))"
  else
    echo "No Slurm (local subprocesses): $(core_count) cores, $RESERVED_CORES reserved, ${REPLICAS_PER_JOB} per job"
    echo "  -> --max-concurrent $per_ly per Ly (${#LYS[@]} dispatcher(s))"
  fi

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
  echo "$EXP (Δμ=$DELTA_MU)"
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

CLI_LYS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --lys) shift ;;
    *) CLI_LYS+=("$1"); shift ;;
  esac
done

set_lys() {
  if [ "${#CLI_LYS[@]}" -gt 0 ]; then
    LYS=("${CLI_LYS[@]}")
  elif [ -n "${LYS:-}" ]; then
    # shellcheck disable=SC2206
    LYS=(${LYS})
  else
    # shellcheck disable=SC2206
    LYS=($(default_lys))
  fi
}

case "$CMD" in
  scout)
    LYS=("${SCOUT_LYS[@]}")
    [ "${#CLI_LYS[@]}" -gt 0 ] && LYS=("${CLI_LYS[@]}")
    EPS_MIN="${EPS_MIN:-$SCOUT_MIN}"
    EPS_MAX="${EPS_MAX:-$SCOUT_MAX}"
    EPS_STEP="${EPS_STEP:-$SCOUT_EPS_STEP}"
    generate
    daemons
    ;;
  refine)
    EPS_C="${CLI_LYS[0]:-}"
    if [ -z "$EPS_C" ]; then
      echo "usage: $0 $EXP refine <epsilon_c>   (from the scout's criticality.csv)"
      echo "  e.g. $0 $EXP refine \"\$(awk -F, 'NR==2{print \$9}' $CAMPAIGN_ROOT/criticality/ly16/criticality.csv)\""
      exit 1
    fi
    CLI_LYS=("${CLI_LYS[@]:1}")
    set_lys
    EPS_MIN="${EPS_MIN:-$(python -c "print(round($EPS_C - $REFINE_HALF_WIDTH, 6))")}"
    EPS_MAX="${EPS_MAX:-$(python -c "print(round($EPS_C + $REFINE_HALF_WIDTH, 6))")}"
    EPS_STEP="${EPS_STEP:-$REFINE_EPS_STEP}"
    echo "refine around ε_c=$EPS_C -> production grid ε ∈ [$EPS_MIN, $EPS_MAX] step $EPS_STEP"
    generate
    daemons
    ;;
  generate|daemons|status)
    set_lys
    EPS_MIN="${EPS_MIN:-$SCOUT_MIN}"
    EPS_MAX="${EPS_MAX:-$SCOUT_MAX}"
    EPS_STEP="${EPS_STEP:-$SCOUT_EPS_STEP}"
    "$CMD"
    ;;
  *)
    echo "usage: $0 <C3A|C3B|C3C|C3D|C3E> [scout|refine <eps_c>|generate|daemons|status] [--lys N ...]"
    exit 1
    ;;
esac
