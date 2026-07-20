#!/bin/bash
#SBATCH --job-name=susc_taper_L256
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=512M
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_reports/%x_%j.out
#SBATCH --error=slurm_reports/%x_%j.err
#
# Append L=256 replicas to an existing homo_dmu1p0_taper campaign folder.
# Same ε-tapered eq/prod times and fitted μ as run_susceptibility_homo_dmu1p0_tapered.sh,
# but only L=256 (susceptibility_runner appends new replica rows if the dir exists).
#
# Args (passed by sbatch/bash):
#   $1  epsilon       (required)
#   $2  results_base  (required; existing taper folder)
#   $3  num_batches   (optional, default 1)
#   $4  mu            (optional; fitted mu_coex from manage CSV)
#   $5..$8  delta_f, delta_mu, k, scheme (optional)

set -euo pipefail

EPS=$1
RESULTS_BASE=$2
NUM_BATCHES=${3:-1}
MU=${4:-}
DELTA_F=${5:-}
DELTA_MU=${6:-}
K_ARG=${7:-}
SCHEME=${8:-}
N=${SLURM_CPUS_PER_TASK:-16}

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

echo "L=256 append  eps=${EPS}  results=${RESULTS_BASE}"
echo "tapered times: EQ_TIME=${EQ_TIME} PROD_TIME=${PROD_TIME} PROD_CHUNKS=${PROD_CHUNKS}"

SIZE=256
echo "=== epsilon=${EPS} L=${SIZE} (cpus=${N}, batches=${NUM_BATCHES}, eq=${EQ_TIME}, prod=${PROD_TIME}) ==="
SECONDS=0
python -u susceptibility/susceptibility_runner.py \
    --epsilon "$EPS" --L "$SIZE" \
    --cpus "$N" --num-batches "$NUM_BATCHES" \
    --eq-time "$EQ_TIME" --prod-time "$PROD_TIME" --prod-chunks "$PROD_CHUNKS" \
    --seed-base "$SEED_BASE" --initial-fraction "$INITIAL_FRACTION" \
    --results-base "$RESULTS_BASE" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
echo ">>> epsilon=${EPS} L=${SIZE} took ${SECONDS}s ($((SECONDS/60))m$((SECONDS%60))s)"
