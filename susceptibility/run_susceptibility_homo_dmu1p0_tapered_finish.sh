#!/bin/bash
#SBATCH --job-name=susc_taper_fin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=512M
#SBATCH --time=2-23:59:59
#SBATCH --output=slurm_reports/%x_%j.out
#SBATCH --error=slurm_reports/%x_%j.err
#
# Finish partial homo Δμ=1 tapered runs: large L only (default 64 96 128).
# Same ε-tapered times as run_susceptibility_homo_dmu1p0_tapered.sh.
#
# Override sizes at submit time, e.g. only L=96,128 for a near-complete ε:
#   sbatch --export=ALL,TAPER_SIZES="96 128" susceptibility/run_susceptibility_homo_dmu1p0_tapered_finish.sh ...
#
# Args: same as run_susceptibility_homo_dmu1p0_tapered.sh ($1 eps, $2 results_base, ...)

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

eval "$(python -u susceptibility/taper_times.py --epsilon "$EPS" --format shell)"

echo "tapered times for eps=${EPS}: EQ_TIME=${EQ_TIME} PROD_TIME=${PROD_TIME} PROD_CHUNKS=${PROD_CHUNKS}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LAUNCH=(srun python -u)
else
    LAUNCH=(python -u)
fi

if [[ -n "${TAPER_SIZES:-}" ]]; then
    read -ra SIZES <<< "$TAPER_SIZES"
else
    SIZES=(64 96 128)
fi

echo "finish sizes for eps=${EPS}: ${SIZES[*]}"

for SIZE in "${SIZES[@]}"; do
    echo "=== epsilon=${EPS} L=${SIZE} (cpus=${N}, batches=${NUM_BATCHES}, eq=${EQ_TIME}, prod=${PROD_TIME}) ==="
    SECONDS=0
    "${LAUNCH[@]}" susceptibility/susceptibility_runner.py \
        --epsilon "$EPS" --L "$SIZE" \
        --cpus "$N" --num-batches "$NUM_BATCHES" \
        --eq-time "$EQ_TIME" --prod-time "$PROD_TIME" --prod-chunks "$PROD_CHUNKS" \
        --seed-base "$SEED_BASE" --initial-fraction "$INITIAL_FRACTION" \
        --results-base "$RESULTS_BASE" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    echo ">>> epsilon=${EPS} L=${SIZE} took ${SECONDS}s ($((SECONDS/60))m$((SECONDS%60))s)"
done
