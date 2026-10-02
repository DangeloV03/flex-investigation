#!/usr/bin/env bash
# Experiments C3A–C3E (negative_drive, βΔf=0, k=1), unattended, Ly=16 on Della.
#
# For each experiment, with no human between steps:
#
#   scout (Ly=16, ε ∈ [-3.0,-0.6] step 0.05) -> criticality -> ε_c
#                                                              |
#   criticality  <-  refine (Ly=16, ε_c ± 0.3 step 0.005)  <---+
#
# The five experiments run as five independent pipelines in parallel (one tmux
# session each, c3-auto-C3A .. C3E), each capped at 30 concurrent Slurm jobs, so
# the whole set finishes together. Same physics and file layout as
# ./coex/run_c3_coex_campaign.sh; this just drives it.
#
# Physics: scheme=negative_drive, flex-scheme=2, βΔf=0, k=1, Lx=10*Ly.
#   C3A Δμ=+1   C3B Δμ=-1   C3C Δμ=+2   C3D Δμ=+3   C3E Δμ=+4
# Sizes: Ly=16 only (LYS="16 20 40" would add more; the sheet's 20/40 rows come later).
#
# Why it can run unattended. coex/run_all.py exits by itself once the queue is
# drained AND nothing is in flight; coex/analyzer.py has --once. The analyzer can
# re-enqueue work (it extends μ windows when a sign change is not bracketed), so
# each phase alternates dispatch and analyze until the queue stays empty. That
# terminates because the analyzer allows at most MAX_ADDITIONAL_REQUESTS=10
# extensions per combo before writing NaN.
#
# If the scout yields no usable ε_c (criticality fit fails), the refine does NOT
# stop: it falls back to FALLBACK_EPS_C (default -1.705, the measured homo k=1
# Δμ=1 value) and logs that loudly, so there is still Ly=16 data at morning.
#
# Usage (repo root, after `source env.sh`):
#   ./coex/run_c3_ly2040.sh start [EXP ...]   # all five, or just those named
#   ./coex/run_c3_ly2040.sh status            # progress + last log lines
#   ./coex/run_c3_ly2040.sh stop [EXP ...]    # halt
#   ./coex/run_c3_ly2040.sh reset --yes       # archive COEX_RUNS_C3* to .trash/, kill sessions
#   tail -f logs/c3x_<EXP>_*.log
#
# Overrides: MAX_CONCURRENT= DISPATCH_INTERVAL= LYS= SCOUT_LY= SCOUT_STEP=
#            REFINE_HALF_WIDTH= FALLBACK_EPS_C=

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

ALL_EXPS=(C3B C3C)
SESSION_PREFIX=c3x   # one tmux session per experiment: c3-auto-C3A, ...
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
FALLBACK_EPS_C="${FALLBACK_EPS_C:--1.705}"
MAX_PHASE_ROUNDS=40        # dispatch/analyze rounds before calling a phase stuck

has_slurm() { command -v sbatch >/dev/null 2>&1; }

# Per-experiment Δμ, scout window, and the ε at/above which the generator skips
# (μ_coex_FLEX > 0). See run_c3_coex_campaign.sh for the derivation.
exp_params() {
  case "$1" in
    C3B) DELTA_MU=-1.0; FLEX_CUTOFF=;         ANCHOR="${ANCHOR_C3B:--1.6613}" ;;
    C3C) DELTA_MU=2.0;  FLEX_CUTOFF=-1.060;   ANCHOR="${ANCHOR_C3C:--1.7201}" ;;
    *) echo "Unknown experiment '$1' (C3B, C3C only)" >&2; return 1 ;;
  esac
}

target_lys() {
  if [ -n "${LYS:-}" ]; then echo "$LYS"
  else echo "16"; fi
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

# A hung sbatch (busy controller) blocked one dispatcher for 5+ minutes on the
# first night's launch. Wrap sbatch in a timeout so a hang becomes a failure that
# run_all.py re-queues, rather than a stall.
install_sbatch_shim() {
  has_slurm || return 0
  local real; real="$(command -v sbatch)"
  mkdir -p "$PROJECT_DIR/.shims"
  printf '#!/bin/sh\nexec timeout 180 %s "$@"\n' "$real" > "$PROJECT_DIR/.shims/sbatch.$$"
  chmod +x "$PROJECT_DIR/.shims/sbatch.$$"
  mv -f "$PROJECT_DIR/.shims/sbatch.$$" "$PROJECT_DIR/.shims/sbatch"
  export PATH="$PROJECT_DIR/.shims:$PATH"
}

# Unfinished samples/*.json (finished ones are archived to samples/done/) that
# are in neither pending nor in_flight were dropped by a failed submit: re-queue.
# Only call between dispatcher runs, never while one is active.
recover_orphans() {
  python - "$1" <<'PY'
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "coex"))
import queue_manifest as qm
base = sys.argv[1]
qpath = os.path.join(base, "queue.json")
sdir = os.path.join(base, "samples")
if not (os.path.isfile(qpath) and os.path.isdir(sdir)):
    raise SystemExit(0)
m = json.load(open(qpath))
known = {os.path.basename(p) for p in m.get("pending", [])}
known |= {os.path.basename(p) for p in m.get("in_flight", {}).values()}
orphans = [os.path.join(sdir, f) for f in sorted(os.listdir(sdir))
           if f.endswith(".json") and f not in known]
if orphans:
    qm.merge_pending(orphans, path=qpath)
    print(f"  recovered {len(orphans)} orphaned job(s) into {qpath}")
PY
}

# Alternate dispatch and analyze until the queue stays empty and nothing new
# gets analyzed. The analyzer re-enqueues μ extensions, so one pass is not enough.
# A failed dispatcher/analyzer is retried (transient Slurm trouble), not treated
# as "phase finished": moving on with a half-run scout gave a bogus ε_c.
drive_phase() {
  local base="$1" mc="$2" label="$3"
  local round=0 prev_analyzed=-1 fails=0
  while :; do
    round=$((round + 1))
    if [ "$round" -gt "$MAX_PHASE_ROUNDS" ]; then
      log "  !! $label: still churning after $MAX_PHASE_ROUNDS rounds — moving on"
      return 1
    fi
    recover_orphans "$base" || true
    if ! python -u coex/run_all.py --manifest "$base/queue.json" --max-concurrent "$mc"          --interval "$DISPATCH_INTERVAL"        || ! python -u coex/analyzer.py --results "$base/results" --manage "$base/manage.csv"          --samples "$base/samples" --manifest "$base/queue.json" --once; then
      fails=$((fails + 1))
      if [ "$fails" -gt 30 ]; then
        log "  !! $label: dispatcher/analyzer failed $fails times — giving up"
        return 1
      fi
      log "  !! $label: dispatcher/analyzer failed (#$fails) — retrying in 60s"
      round=$((round - 1))
      sleep 60
      continue
    fi

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

  install_sbatch_shim
  log "host=$(hostname -s)  slurm=$(has_slurm && echo yes || echo no)  cores=$(core_count)"
  log "target Ly: ${lys[*]}   scout Ly: $scout_ly   budget: $budget concurrent jobs"
  log "experiments: ${exps[*]}"
  echo

  for exp in "${exps[@]}"; do
    exp_params "$exp" || continue
    local root="COEX_RUNS_$exp"
    log "=============== $exp (Δμ=$DELTA_MU) ==============="

    # ε_c anchor is the raw BC=5/9 linear crossing of the Ly=16 scout (the
    # automated fit was inconsistent with it for C3B/C3C); see ANCHOR_<EXP>.
    local eps_c="$ANCHOR"
    log "ε_c anchor = $eps_c (window ± $REFINE_HALF_WIDTH)"

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

session_for() { echo "${SESSION_PREFIX}-$1"; }

cmd_reset() {
  if [ "${1:-}" != "--yes" ]; then
    echo "This will stop all C3 tmux sessions and archive:"
    for e in "${ALL_EXPS[@]}"; do
      [ -d "COEX_RUNS_$e" ] && echo "  COEX_RUNS_$e  ($(du -sh "COEX_RUNS_$e" 2>/dev/null | cut -f1))"
    done
    echo "Nothing is deleted — roots move to .trash/<timestamp>/. Re-run with --yes."
    return 0
  fi
  local stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  for e in "${ALL_EXPS[@]}"; do
    tmux kill-session -t "$(session_for "$e")" 2>/dev/null || true
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
  mkdir -p "$LOG_DIR"
  local setup="module load anaconda3/2024.10 2>/dev/null; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate lattice; export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\"; export PYTHONPATH=\"$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR\"; export PYTHONUNBUFFERED=1; export LYS='${LYS:-}' REFINE_HALF_WIDTH='${REFINE_HALF_WIDTH:-0.3}' MAX_CONCURRENT='${MAX_CONCURRENT:-}' SCOUT_LY='${SCOUT_LY:-16}'"
  for e in "${exps[@]}"; do
    local s; s="$(session_for "$e")"
    if tmux has-session -t "$s" 2>/dev/null; then
      echo "Session '$s' already running — skipping $e (stop it first: $0 stop $e)"
      continue
    fi
    local logf="$LOG_DIR/c3x_${e}_$(date -u +%Y%m%dT%H%M%SZ).log"
    tmux new-session -d -s "$s" -c "$PROJECT_DIR"
    tmux send-keys -t "$s" \
      "${setup}; SCOUTS_ONLY=${SCOUTS_ONLY:-0} ./coex/run_c3_ly2040.sh _pipeline $e 2>&1 | tee -a '$logf'" C-m
    echo "Started '$s' on $(hostname -s)   log: $logf"
  done
}

cmd_status() {
  local lys scout_ly shown
  # shellcheck disable=SC2206
  lys=($(target_lys))
  scout_ly="${SCOUT_LY:-16}"
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
      printf "  Ly=%-3s pending=%-6s in_flight=%-4s analyzed=%s/%s\n" "$ly" "$p" "$i" "$a" "$t"
    done
    [ -f "COEX_RUNS_$e/.scout_done" ] && echo "  scout done"
    [ -f "COEX_RUNS_$e/.refine_done_ly16" ] && echo "  refine done (Ly=16)"
    local c="COEX_RUNS_$e/criticality/ly$scout_ly/criticality.csv"
    [ -f "$c" ] && echo "  ε_c = $(read_eps_c "$c")"
    if tmux has-session -t "$(session_for "$e")" 2>/dev/null; then
      echo "  session $(session_for "$e"): RUNNING"
    else
      echo "  session $(session_for "$e"): not running"
    fi
    local latest; latest="$(ls -t "$LOG_DIR"/c3x_${e}_*.log 2>/dev/null | head -1 || true)"
    [ -n "$latest" ] && tail -2 "$latest" | sed 's/^/    | /'
  done
  return 0
}

cmd_stop() {
  local exps=("$@")
  [ "${#exps[@]}" -eq 0 ] && exps=("${ALL_EXPS[@]}")
  for e in "${exps[@]}"; do
    tmux kill-session -t "$(session_for "$e")" 2>/dev/null && echo "Stopped $(session_for "$e")." \
      || echo "No session $(session_for "$e") running."
  done
}

CMD="${1:-}"
shift || true
case "$CMD" in
  reset)     cmd_reset "$@" ;;
  start)     cmd_start "$@" ;;
  scouts)    SCOUTS_ONLY=1 cmd_start "$@" ;;
  status)    cmd_status ;;
  stop)      cmd_stop "$@" ;;
  _pipeline) pipeline "$@" ;;   # internal: runs inside tmux
  *)
    echo "usage: $0 <reset [--yes]|scouts [EXP ...]|start [EXP ...]|status|stop [EXP ...]>"
    exit 1
    ;;
esac
