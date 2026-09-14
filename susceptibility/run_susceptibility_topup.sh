#!/bin/bash
#SBATCH --job-name=susc_topup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=512M
#SBATCH --time=23:59:59
#SBATCH --output=slurm_reports/%x_%j.out
#SBATCH --error=slurm_reports/%x_%j.err
#
# Top-up run: continue an existing SUSC_RUNS campaign for one epsilon by
# loading the final lattice of each prior replica and running 10^6 extra
# production steps.  Submitted with --dependency=afterok:... by smart_sweep.py
# check when any (eps, L) pair has mean J < threshold.
#
# Args:
#   $1  epsilon        (required)
#   $2  results_base   (required; same SUSC_RUNS root used for the original run)
#   $3  failing_sizes  (space-separated; e.g. "48 64 96"; omit = all sizes)
#   remaining positional args are also treated as additional L sizes (for
#   sbatch which takes each word as a separate token in the arg list)

set -euo pipefail

EPS=$1
RESULTS_BASE=$2
shift 2

# Remaining args are the L sizes that need topping up.
if [[ $# -gt 0 ]]; then
    SIZES=("$@")
else
    SIZES=(16 32 48 64 96 128)
fi

N=${SLURM_CPUS_PER_TASK:-2}

# 10^6 per round (L=128 ~2 h on a healthy node, well inside the 24 h wall).
# Each round writes a new full-history m_timeseries file per replica, so disk
# grows with (rounds x history): small increments multiply rounds, check
# overhead, and storage for the same statistics.  A 2026-09-08 round that ran
# ~29x slow was confined to two nodes (della-h17n16/h17n19), not the physics.
TOPUP_PROD_TIME=1000000.0
TOPUP_PROD_CHUNKS=10000
SEED_BASE=7000

if command -v module >/dev/null 2>&1; then
    module load anaconda3/2024.10
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate lattice
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONUNBUFFERED=1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")/.."
fi
export PYTHONPATH="$PWD/coex:$PWD/susceptibility:$PWD${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LAUNCH=(srun python -u)
else
    LAUNCH=(python -u)
fi

JOB_ID="${SLURM_JOB_ID:-local}"
TIMING_DIR="$RESULTS_BASE/timing"
mkdir -p "$TIMING_DIR"
TIMING_CSV="$TIMING_DIR/${JOB_ID}.csv"
if [[ ! -f "$TIMING_CSV" ]]; then
    echo "phase,job_id,epsilon,L,wall_seconds,ncpus,finished_at" > "$TIMING_CSV"
fi

for SIZE in "${SIZES[@]}"; do
    echo "=== TOP-UP epsilon=${EPS} L=${SIZE} ==="

    # Discover the existing run directory and the physics it was produced with.
    # Do NOT rebuild the path from assumed parameters: campaigns differ
    # (S1A: Δf=-20, k=0; S1B: Δf=0, k=1; …) and a wrong guess silently skips
    # every top-up, leaving the check/top-up loop spinning forever.
    RESOLVED=$(python -c "
import sys
sys.path[:0] = ['susceptibility', 'coex']
from susceptibility_paths import find_susc_run_dir, read_run_physics

run_dir = find_susc_run_dir('$RESULTS_BASE', $SIZE, float('$EPS'))
if run_dir is None:
    sys.exit(3)
physics = read_run_physics(run_dir)
flags = []
for flag, key in (('--delta-f', 'delta_f'), ('--delta-mu', 'delta_mu'),
                  ('--k', 'k'), ('--scheme', 'scheme'), ('--mu', 'mu')):
    if key in physics:
        flags += [flag, str(physics[key])]
print(run_dir)
print(' '.join(flags))
") || {
        echo "WARNING: no run found for epsilon=${EPS} L=${SIZE} under ${RESULTS_BASE}, skipping"
        continue
    }

    OUTDIR=$(echo "$RESOLVED" | sed -n 1p)
    read -r -a PHYSICS_ARGS <<< "$(echo "$RESOLVED" | sed -n 2p)"

    if [[ ! -d "$OUTDIR" ]]; then
        echo "WARNING: $OUTDIR does not exist, skipping L=${SIZE}"
        continue
    fi
    echo "    run dir: $OUTDIR"
    echo "    physics: ${PHYSICS_ARGS[*]:-<runner defaults>}"

    SECONDS=0
    "${LAUNCH[@]}" susceptibility/susceptibility_runner.py \
        --epsilon "$EPS" --L "$SIZE" \
        --cpus "$N" \
        --outdir "$OUTDIR" \
        --resume-dir "$OUTDIR" \
        --eq-time 0 \
        --prod-time "$TOPUP_PROD_TIME" \
        --prod-chunks "$TOPUP_PROD_CHUNKS" \
        --seed-base "$SEED_BASE" \
        --results-base "$RESULTS_BASE" \
        ${PHYSICS_ARGS[@]+"${PHYSICS_ARGS[@]}"}
    echo ">>> TOP-UP epsilon=${EPS} L=${SIZE} took ${SECONDS}s ($((SECONDS/60))m$((SECONDS%60))s)"
    echo "topup,${JOB_ID},${EPS},${SIZE},${SECONDS},${N},$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$TIMING_CSV"
done
