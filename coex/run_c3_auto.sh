#!/usr/bin/env bash
# Experiments C3A–C3E, unattended. One command, no babysitting.
#
# Runs the whole pipeline for each experiment in turn:
#
#   scout (coarse, smallest Ly)  ->  criticality  ->  ε_c
#                                                      |
#   criticality + FSS  <-  refine (ε_c ± 0.3, all Ly) <+
#
# and then moves to the next experiment. Nothing here needs a human between
# steps, and nothing needs the other machine: each host derives its own ε_c from
# its own scout, so the workstation and Della run the identical command with no
# handoff. Same physics as ./coex/run_c3_coex_campaign.sh — this just drives it.
#
# Physics: scheme=negative_drive, flex-scheme=3, βΔf=0, k=1, Lx=10*Ly.
#   C3A Δμ=+1   C3B Δμ=-1   C3C Δμ=+2   C3D Δμ=+3   C3E Δμ=+4
# Refine sizes come from the host: Ly 16+20 without Slurm (workstation), Ly 40
# with it (Della). The scout always runs at Ly=16 — it only has to centre a
# 0.6-wide window, so there is no reason to pay for it at Ly=40.
#
# Why it can run unattended. coex/run_all.py exits by itself once the queue is
# drained AND nothing is in flight; coex/analyzer.py has --once. The analyzer can
# re-enqueue work (it extends μ windows when a sign change is not bracketed), so
# each phase alternates dispatch and analyze until the queue stays empty. That
# terminates because the analyzer allows at most MAX_ADDITIONAL_REQUESTS=10
# extensions per combo before writing NaN.
#
# Cores are budgeted ONCE for the whole campaign and experiments run strictly one
# at a time, so there is no way to oversubscribe the workstation by launching
# several sessions — the failure mode of driving run_c3_coex_campaign.sh by hand.
#
# Usage (repo root, after `source env.sh`):
#   ./coex/run_c3_auto.sh reset --yes     # archive previous C3 runs, kill sessions
#   ./coex/run_c3_auto.sh scouts          # scouts only -> eps_c for all five
#   ./coex/run_c3_auto.sh start           # full pipeline (scout + refine)
#   ./coex/run_c3_auto.sh start C3A C3B   # just these
#   ./coex/run_c3_auto.sh status          # progress + last log lines
#   ./coex/run_c3_auto.sh stop            # halt the campaign
#   tail -f logs/c3_auto_*.log            # watch it work
#
# reset ARCHIVES rather than deletes: COEX_RUNS_C3* move to .trash/<timestamp>/.
#
# Overrides: RESERVED_CORES= MAX_CONCURRENT= DISPATCH_INTERVAL= LYS= SCOUT_LY=
#            SCOUT_STEP= REFINE_HALF_WIDTH=

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

ALL_EXPS=(C3A C3B C3C C3D C3E)
SESSION=c3-auto
LOG_DIR="$PROJECT_DIR/logs"

SCHEME=negative_drive
FLEX_SCHEME=3
DELTA_F=0.0
K=1.0
MU_WINDOW=0.05
N_MU_POINTS=10

SCOUT_STEP="${SCOUT_STEP:-0.05}"
REFINE_HALF_WIDTH="${REFINE_HALF_WIDTH:-0.3}"
REFINE_STEP=0.005          # production susceptibility grid (sweep_susceptibility.py)
REPLICAS_PER_JOB=8         # json_runner.py forks this many per job
RESERVED_CORES="${RESERVED_CORES:-4}"
# run_all.py polls every POLL_INTERVAL=30s by default. A coex job here is only
# ~8s of compute, so with the stock interval the box sat ~93% idle: measured
# 1.6 of 24 cores busy, bursts of r=16 for ~8s then ~22s of nothing. Refill
# promptly instead.
DISPATCH_INTERVAL="${DISPATCH_INTERVAL:-3}"
SLURM_MAX_CONCURRENT=30
MAX_PHASE_ROUNDS=40        # dispatch/analyze rounds before calling a phase stuck

has_slurm() { command -v sbatch >/dev/null 2>&1; }

# Per-experiment Δμ, scout window, and the ε at/above which the generator skips
# (μ_coex_FLEX > 0). See run_c3_coex_campaign.sh for the derivation.
exp_params() {
  case "$1" in
    C3A) DELTA_MU=1.0;  SCOUT_MIN=-2.60; SCOUT_MAX=-0.70; FLEX_CUTOFF=-0.655 ;;
    C3B) DELTA_MU=-1.0; SCOUT_MIN=-2.80; SCOUT_MAX=-1.00; FLEX_CUTOFF= ;;
    C3C) DELTA_MU=2.0;  SCOUT_MIN=-3.00; SCOUT_MAX=-1.10; FLEX_CUTOFF=-1.060 ;;
    C3D) DELTA_MU=3.0;  SCOUT_MIN=-3.50; SCOUT_MAX=-1.55; FLEX_CUTOFF=-1.520 ;;
    C3E) DELTA_MU=4.0;  SCOUT_MIN=-4.00; SCOUT_MAX=-2.05; FLEX_CUTOFF=-2.005 ;;
    *) echo "Unknown experiment '$1'" >&2; return 1 ;;
  esac
}

target_lys() {
  if [ -n "${LYS:-}" ]; then echo "$LYS"
  elif has_slurm; then echo "40"
  else echo "16 20"; fi
}

core_count() {
  if command -v nproc >/dev/null; then nproc; else sysctl -n hw.ncpu; fi
}

# Total concurrent jobs for the campaign. Only one experiment runs at a time, so
# this whole budget belongs to whichever phase is active.
total_budget() {
  local cores usable total
  if [ -n "${MAX_CONCURRENT:-}" ]; then
    echo "$MAX_CONCURRENT"; return
  fi
  if has_slurm; then echo "$SLURM_MAX_CONCURRENT"; return; fi
  cores="$(core_count)"
  usable=$((cores - RESERVED_CORES))
  total=$((usable / REPLICAS_PER_JOB))
  [ "$total" -lt 1 ] && total=1
  echo "$total"
}

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- phase driver

# True when the queue is empty and every manage row has been analyzed.
# Prints "<pending> <in_flight> <analyzed> <total>" for progress logging.
phase_state() {
  python - "$1" <<'PY'
import csv, json, os, sys
base = sys.argv[1]
q = os.path.join(base, "queue.json")
m = os.path.join(base, "manage.csv")
pending = inflight = analyzed = total = 0
if os.path.isfile(q):
    d = json.load(open(q))
    pending = len(d.get("pending", []))
    inflight = len(d.get("in_flight", {}))
if os.path.isfile(m):
    rows = list(csv.DictReader(open(m, newline="")))
    total = len(rows)
    analyzed = sum(1 for r in rows if str(r.get("isAnalyzed", "")).strip())
print(pending, inflight, analyzed, total)
PY
}

# Alternate dispatch and analyze until the queue stays empty and nothing new
# gets analyzed. The analyzer re-enqueues μ extensions, so one pass is not enough.
drive_phase() {
  local base="$1" mc="$2" label="$3"
  local round=0 prev_analyzed=-1
  while :; do
    round=$((round + 1))
    if [ "$round" -gt "$MAX_PHASE_ROUNDS" ]; then
      log "  !! $label: still churning after $MAX_PHASE_ROUNDS rounds — moving on"
      return 1
    fi
    python -u coex/run_all.py --manifest "$base/queue.json" --max-concurrent "$mc" \
      --interval "$DISPATCH_INTERVAL" \
      || { log "  !! $label: dispatcher failed"; return 1; }
    python -u coex/analyzer.py --results "$base/results" --manage "$base/manage.csv" \
      --samples "$base/samples" --manifest "$base/queue.json" --once \
      || { log "  !! $label: analyzer failed"; return 1; }

    read -r pending inflight analyzed total <<<"$(phase_state "$base")"
    log "  $label round $round: pending=$pending in_flight=$inflight analyzed=$analyzed/$total"

    if [ "$pending" -eq 0 ] && [ "$inflight" -eq 0 ]; then
      if [ "$analyzed" -ge "$total" ]; then
        log "  $label complete ($analyzed/$total analyzed)"
        return 0
      fi
      if [ "$analyzed" -eq "$prev_analyzed" ]; then
        # Queue drained and the analyzer made no further progress: terminal.
        log "  $label settled at $analyzed/$total analyzed (queue empty, no progress)"
        return 0
      fi
    fi
    prev_analyzed="$analyzed"
  done
}

gen_grid() {
  local exp="$1" ly="$2" emin="$3" emax="$4" estep="$5"
  local base="COEX_RUNS_$exp/ly$ly"
  mkdir -p "$base/samples" "$base/results"
  python -u coex/generate_susceptibility_coex.py \
    --scheme "$SCHEME" --flex-scheme "$FLEX_SCHEME" \
    --delta-f "$DELTA_F" --delta-mu "$DELTA_MU" --k "$K" \
    --ly "$ly" \
    --eps-min "$emin" --eps-max "$emax" --eps-step "$estep" \
    --mu-window "$MU_WINDOW" --n-mu-points "$N_MU_POINTS" \
    --samples-dir "$base/samples" --results-dir "$base/results" \
    --manage "$base/manage.csv" --manifest "$base/queue.json"
}

run_criticality() {
  local exp="$1"; shift
  ./coex/run_c3_criticality.sh "$exp" "$@"
}

# epsilon_c_estimate from a criticality.csv, by column name. Empty if unusable.
read_eps_c() {
  python - "$1" <<'PY'
import csv, math, os, sys
p = sys.argv[1]
if not os.path.isfile(p):
    raise SystemExit(0)
vals = []
with open(p, newline="") as f:
    for r in csv.DictReader(f):
        raw = str(r.get("epsilon_c_estimate", "")).strip()
        try:
            v = float(raw)
        except ValueError:
            continue
        if math.isfinite(v):
            vals.append(v)
if vals:
    print(f"{vals[0]:.6f}")
PY
}

# ------------------------------------------------------------------- pipeline

pipeline() {
  local exps=("$@")
  local lys budget scout_ly
  # shellcheck disable=SC2206
  lys=($(target_lys))
  # The scout only locates ε_c well enough to CENTRE a 0.6-wide refine window,
  # so it always runs at the cheap size — even on Della, where the refine is
  # Ly=40 and scouting there would cost ~6x the sites for the same number.
  # Finite-size drift in ε_c between L=16 and L=40 is far inside ±0.3, and C2A
  # set its window for Ly 16/20/40 from the Ly=16 measurement the same way.
  scout_ly="${SCOUT_LY:-16}"
  budget="$(total_budget)"

  log "host=$(hostname -s)  slurm=$(has_slurm && echo yes || echo no)  cores=$(core_count)"
  log "target Ly: ${lys[*]}   scout Ly: $scout_ly   budget: $budget concurrent jobs"
  log "experiments: ${exps[*]}"
  echo

  for exp in "${exps[@]}"; do
    exp_params "$exp" || continue
    local root="COEX_RUNS_$exp"
    log "=============== $exp (Δμ=$DELTA_MU) ==============="

    # ---- scout ----
    local scout_base="$root/ly$scout_ly"
    if [ -f "$root/.scout_done" ]; then
      log "scout: already done, skipping"
    else
      log "scout: Ly=$scout_ly  ε ∈ [$SCOUT_MIN, $SCOUT_MAX] step $SCOUT_STEP"
      gen_grid "$exp" "$scout_ly" "$SCOUT_MIN" "$SCOUT_MAX" "$SCOUT_STEP"
      # Only mark complete on a clean finish: a phase that got stuck stays
      # unmarked so re-running `start` retries it instead of skipping ahead.
      if drive_phase "$scout_base" "$budget" "scout"; then
        touch "$root/.scout_done"
      else
        log "  scout did not finish cleanly — continuing on what analyzed so far"
      fi
    fi

    # ---- eps_c ----
    log "criticality on scout (Ly=$scout_ly)"
    run_criticality "$exp" "$scout_ly" || log "  !! criticality failed"
    local eps_c
    eps_c="$(read_eps_c "$root/criticality/ly$scout_ly/criticality.csv")"
    if [ -z "$eps_c" ]; then
      log "!! $exp: no usable ε_c from the scout — skipping refine, continuing campaign"
      echo
      continue
    fi
    log "ε_c = $eps_c"

    if [ "${SCOUTS_ONLY:-0}" = "1" ]; then
      log "$exp scout done (scouts-only mode — no refine)"
      echo
      continue
    fi

    # ---- refine on the production grid, clamped below the FLEX cutoff ----
    local emin emax
    emin="$(python -c "print(round($eps_c - $REFINE_HALF_WIDTH, 6))")"
    emax="$(python -c "print(round($eps_c + $REFINE_HALF_WIDTH, 6))")"
    if [ -n "$FLEX_CUTOFF" ] \
       && python -c "import sys; sys.exit(0 if $emax >= $FLEX_CUTOFF else 1)"; then
      emax="$(python -c "print(round($FLEX_CUTOFF - $REFINE_STEP, 6))")"
      log "refine: ε_max clamped to $emax (FLEX cutoff $FLEX_CUTOFF; above it jobs are skipped)"
    fi

    local per_ly=$((budget / ${#lys[@]}))
    [ "$per_ly" -lt 1 ] && per_ly=1
    for ly in "${lys[@]}"; do
      if [ -f "$root/.refine_done_ly$ly" ]; then
        log "refine Ly=$ly: already done, skipping"
        continue
      fi
      log "refine Ly=$ly: ε ∈ [$emin, $emax] step $REFINE_STEP"
      gen_grid "$exp" "$ly" "$emin" "$emax" "$REFINE_STEP"
    done
    for ly in "${lys[@]}"; do
      [ -f "$root/.refine_done_ly$ly" ] && continue
      if drive_phase "$root/ly$ly" "$per_ly" "refine-ly$ly"; then
        touch "$root/.refine_done_ly$ly"
      else
        log "  refine Ly=$ly did not finish cleanly — leaving unmarked for retry"
      fi
    done

    # ---- criticality per size, then FSS across whatever this host has ----
    log "criticality on refine (${lys[*]})"
    run_criticality "$exp" "${lys[@]}" || log "  !! criticality failed"
    if [ "${#lys[@]}" -gt 1 ]; then
      run_criticality "$exp" compare "${lys[@]}" || log "  !! compare failed"
    fi
    log "$exp done"
    echo
  done

  if [ "${SCOUTS_ONLY:-0}" = "1" ]; then
    log "SCOUTS COMPLETE — ε_c per experiment:"
    for exp in "${exps[@]}"; do
      local c="COEX_RUNS_$exp/criticality/ly$scout_ly/criticality.csv"
      printf '    %-4s %s\n' "$exp" "$(read_eps_c "$c" 2>/dev/null || true)"
    done
    log "Refine with: $0 start   (or per experiment: $0 start C3A)"
  else
    log "CAMPAIGN COMPLETE: ${exps[*]}"
  fi
}

# ------------------------------------------------------------------- commands

cmd_reset() {
  if [ "${1:-}" != "--yes" ]; then
    echo "This will stop all C3 tmux sessions and archive:"
    for e in "${ALL_EXPS[@]}"; do
      [ -d "COEX_RUNS_$e" ] && echo "  COEX_RUNS_$e  ($(du -sh "COEX_RUNS_$e" 2>/dev/null | cut -f1))"
    done
    tmux has-session -t "$SESSION" 2>/dev/null && echo "  tmux session $SESSION"
    echo
    echo "Nothing is deleted — roots move to .trash/<timestamp>/."
    echo "Re-run with --yes to proceed."
    return 0
  fi
  local stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  for e in "${ALL_EXPS[@]}"; do
    tmux kill-session -t "coex-$e" 2>/dev/null || true
  done
  mkdir -p ".trash/$stamp"
  local moved=0
  for e in "${ALL_EXPS[@]}"; do
    if [ -d "COEX_RUNS_$e" ]; then
      mv "COEX_RUNS_$e" ".trash/$stamp/"
      echo "archived COEX_RUNS_$e -> .trash/$stamp/"
      moved=$((moved + 1))
    fi
  done
  [ "$moved" -eq 0 ] && rmdir ".trash/$stamp" 2>/dev/null || true
  echo "Reset complete. Start with: $0 start"
}

cmd_start() {
  local exps=("$@")
  [ "${#exps[@]}" -eq 0 ] && exps=("${ALL_EXPS[@]}")
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already running. Stop it first: $0 stop"
    exit 1
  fi
  mkdir -p "$LOG_DIR"
  local logf="$LOG_DIR/c3_auto_$(date -u +%Y%m%dT%H%M%SZ).log"
  local setup="module load anaconda3/2024.10 2>/dev/null; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate lattice; export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\"; export PYTHONPATH=\"$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR\"; export PYTHONUNBUFFERED=1"
  tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR"
  tmux send-keys -t "$SESSION" \
    "${setup}; SCOUTS_ONLY=${SCOUTS_ONLY:-0} ./coex/run_c3_auto.sh _pipeline ${exps[*]} 2>&1 | tee -a '$logf'" C-m
  if [ "${SCOUTS_ONLY:-0}" = "1" ]; then
    echo "Started '$SESSION' on $(hostname -s): SCOUTS ONLY for ${exps[*]}"
  else
    echo "Started '$SESSION' on $(hostname -s): ${exps[*]}"
  fi
  echo "  log:    tail -f $logf"
  echo "  attach: tmux attach -t $SESSION   (Ctrl-b d to detach)"
  echo "  stop:   $0 stop"
}

cmd_status() {
  local lys scout_ly shown
  # shellcheck disable=SC2206
  lys=($(target_lys))
  scout_ly="${SCOUT_LY:-16}"
  # The scout size is not always one of the refine sizes (on Della it is not),
  # so show it alongside them.
  shown=("$scout_ly")
  for ly in "${lys[@]}"; do
    [ "$ly" = "$scout_ly" ] || shown+=("$ly")
  done
  for e in "${ALL_EXPS[@]}"; do
    [ -d "COEX_RUNS_$e" ] || continue
    echo "== $e =="
    for ly in "${shown[@]}"; do
      [ -d "COEX_RUNS_$e/ly$ly" ] || continue
      read -r p i a t <<<"$(phase_state "COEX_RUNS_$e/ly$ly")"
      local tag="refine"
      [ "$ly" = "$scout_ly" ] && tag="scout/refine"
      printf "  Ly=%-3s %-13s pending=%-6s in_flight=%-4s analyzed=%s/%s\n" \
        "$ly" "$tag" "$p" "$i" "$a" "$t"
    done
    local c="COEX_RUNS_$e/criticality/ly$scout_ly/criticality.csv"
    if [ -f "$c" ]; then
      echo "  ε_c(scout) = $(read_eps_c "$c")"
    fi
  done
  echo
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION': RUNNING on $(hostname -s)"
  else
    echo "session '$SESSION': not running on $(hostname -s)"
  fi
  # pipefail would make a failing ls (no logs yet) abort the script
  local latest; latest="$(ls -t "$LOG_DIR"/c3_auto_*.log 2>/dev/null | head -1 || true)"
  if [ -n "$latest" ]; then
    echo "--- $latest ---"
    tail -8 "$latest"
  fi
  return 0
}

cmd_stop() {
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "Stopped '$SESSION'." \
    || echo "No session '$SESSION' running."
}

CMD="${1:-}"
shift || true
case "$CMD" in
  reset)     cmd_reset "$@" ;;
  start)     cmd_start "$@" ;;
  scouts)    SCOUTS_ONLY=1 cmd_start "$@" ;;
  status)    cmd_status ;;
  stop)      cmd_stop ;;
  _pipeline) pipeline "$@" ;;   # internal: runs inside tmux
  *)
    echo "usage: $0 <reset [--yes]|scouts [EXP ...]|start [EXP ...]|status|stop>"
    exit 1
    ;;
esac
