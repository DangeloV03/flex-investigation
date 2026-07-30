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
# loading the final lattice of each prior replica and running 10^5 extra
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

for SIZE in "${SIZES[@]}"; do
    echo "=== TOP-UP epsilon=${EPS} L=${SIZE} ==="

    # Compute the run directory for this (eps, L) combo using the path helpers.
    OUTDIR=$(python -c "
import sys
sys.path.insert(0, 'susceptibility')
from susceptibility_paths import susc_run_dir
params = dict(
    epsilon=float('$EPS'),
    Lx=$SIZE, Ly=$SIZE,
    delta_f=-20.0, delta_mu=0.0, k=0.0, scheme='homo'
)
print(susc_run_dir(params, '$RESULTS_BASE'))
")

    if [[ ! -d "$OUTDIR" ]]; then
        echo "WARNING: $OUTDIR does not exist, skipping L=${SIZE}"
        continue
    fi

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
        --results-base "$RESULTS_BASE"
    echo ">>> TOP-UP epsilon=${EPS} L=${SIZE} took ${SECONDS}s ($((SECONDS/60))m$((SECONDS%60))s)"
done
