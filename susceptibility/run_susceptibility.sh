#!/bin/bash
#SBATCH --job-name=susc_exact
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --time=23:59:59
# Logs go to ~/slurm_reports (same place as the coex jobs). SLURM does not expand
# "~", so use /home/%u; %x = job name, %j = job id. The dir must already exist.
#SBATCH --output=/home/%u/slurm_reports/%x_%j.out
#SBATCH --error=/home/%u/slurm_reports/%x_%j.err
#
# Exact-mu (mu = 2*epsilon) susceptibility run for a single epsilon, looping over
# all square sizes L = 16 .. 256. Submitted once per epsilon by
# sweep_susceptibility.py. Uses SLURM_CPUS_PER_TASK parallel replicas per job and
# appends replicas to each per-(L, eps) susceptibility_data.csv.
#
# Args (passed by sbatch):
#   $1  epsilon       (required)
#   $2  results_base  (required; e.g. susceptibility_results/exact_2026-07-02)
#   $3  num_batches   (optional, default 1)

set -euo pipefail

EPS=$1
RESULTS_BASE=$2
NUM_BATCHES=${3:-1}
N=${SLURM_CPUS_PER_TASK:-2}     # 16 under SLURM -> num_parallel_runs; 2 locally

# ---- Tunables (kept here so they are easy to change) ------------------------
EQ_TIME=100000.0
PROD_TIME=200000.0
PROD_CHUNKS=2000
SEED_BASE=7000
INITIAL_FRACTION=0.5

# ---- Environment (mirrors slurm_config.yml setup_cmds; skipped off-SLURM) ---
if command -v module >/dev/null 2>&1; then
    module load anaconda3/2024.10
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate lattice
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONUNBUFFERED=1

# Run from the repo root so relative results paths land there and PYTHONPATH
# resolves both source folders. Under SLURM the batch script is copied to a spool
# dir, so $0 is useless — use SLURM_SUBMIT_DIR (the dir sbatch was launched from).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")/.."
fi
export PYTHONPATH="$PWD/coex:$PWD/susceptibility:$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Use srun under SLURM; run python directly when testing locally.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LAUNCH=(srun python -u)
else
    LAUNCH=(python -u)
fi

SIZES=(16 32 48 64 96 128 256)

for SIZE in "${SIZES[@]}"; do
    echo "=== epsilon=${EPS} L=${SIZE} (cpus=${N}, batches=${NUM_BATCHES}) ==="
    "${LAUNCH[@]}" susceptibility/susceptibility_runner.py \
        --epsilon "$EPS" --L "$SIZE" \
        --cpus "$N" --num-batches "$NUM_BATCHES" \
        --eq-time "$EQ_TIME" --prod-time "$PROD_TIME" --prod-chunks "$PROD_CHUNKS" \
        --seed-base "$SEED_BASE" --initial-fraction "$INITIAL_FRACTION" \
        --results-base "$RESULTS_BASE"
done
