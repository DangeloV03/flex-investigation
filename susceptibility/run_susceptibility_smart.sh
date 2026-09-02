#!/bin/bash
#SBATCH --job-name=susc_smart
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=512M
#SBATCH --time=23:59:59
#SBATCH --output=slurm_reports/%x_%j.out
#SBATCH --error=slurm_reports/%x_%j.err
#
# Smart-sweep susceptibility run for a single epsilon, all square L sizes.
# Writes output to SUSC_RUNS/_{Lx}_{Ly}_S{n}_.../_eps/ (new layout).
# Submitted by smart_sweep.py sweep; a check job is chained afterwards.
#
# Args:
#   $1  epsilon       (required)
#   $2  results_base  (required; e.g. SUSC_RUNS)
#   $3  num_batches   (optional, default 1)
#   $4  mu            (optional; empty => runner default mu = 2*epsilon)
#   $5  delta_f       (optional; empty => runner default)
#   $6  delta_mu      (optional; empty => runner default)
#   $7  k             (optional; empty => runner default)
#   $8  scheme        (optional; empty => runner default)

set -euo pipefail

EPS=$1
RESULTS_BASE=$2
NUM_BATCHES=${3:-1}
MU=${4:-}
DELTA_F=${5:-}
DELTA_MU=${6:-}
K_ARG=${7:-}
SCHEME=${8:-}
N=${SLURM_CPUS_PER_TASK:-2}

EXTRA_ARGS=()
[[ -n "$MU"       ]] && EXTRA_ARGS+=(--mu "$MU")
[[ -n "$DELTA_F"  ]] && EXTRA_ARGS+=(--delta-f "$DELTA_F")
[[ -n "$DELTA_MU" ]] && EXTRA_ARGS+=(--delta-mu "$DELTA_MU")
[[ -n "$K_ARG"    ]] && EXTRA_ARGS+=(--k "$K_ARG")
[[ -n "$SCHEME"   ]] && EXTRA_ARGS+=(--scheme "$SCHEME")

# 1M eq + 1M prod; chunk_time = 100 (same cadence as standard runner).
EQ_TIME=1000000.0
PROD_TIME=1000000.0
PROD_CHUNKS=10000
SEED_BASE=7000
INITIAL_FRACTION=0.8

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

SIZES=(16 32 48 64 96 128)

JOB_ID="${SLURM_JOB_ID:-local}"
TIMING_DIR="$RESULTS_BASE/timing"
mkdir -p "$TIMING_DIR"
TIMING_CSV="$TIMING_DIR/${JOB_ID}.csv"
if [[ ! -f "$TIMING_CSV" ]]; then
    echo "phase,job_id,epsilon,L,wall_seconds,ncpus,finished_at" > "$TIMING_CSV"
fi

for SIZE in "${SIZES[@]}"; do
    echo "=== epsilon=${EPS} L=${SIZE} (cpus=${N}, batches=${NUM_BATCHES}) ==="
    SECONDS=0
    "${LAUNCH[@]}" susceptibility/susceptibility_runner.py \
        --epsilon "$EPS" --L "$SIZE" \
        --cpus "$N" --num-batches "$NUM_BATCHES" \
        --eq-time "$EQ_TIME" --prod-time "$PROD_TIME" --prod-chunks "$PROD_CHUNKS" \
        --seed-base "$SEED_BASE" --initial-fraction "$INITIAL_FRACTION" \
        --results-base "$RESULTS_BASE" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    echo ">>> epsilon=${EPS} L=${SIZE} took ${SECONDS}s ($((SECONDS/60))m$((SECONDS%60))s)"
    echo "sweep,${JOB_ID},${EPS},${SIZE},${SECONDS},${N},$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$TIMING_CSV"
done
