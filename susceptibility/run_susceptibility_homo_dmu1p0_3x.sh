#!/bin/bash
#SBATCH --job-name=susc_homo_dmu1p0_3x
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=512M
#SBATCH --time=23:59:59
# Logs go to an in-repo slurm_reports/ (repo is on /scratch, large quota) to keep
# stdout/err off the small /home quota — /home filling makes jobs die with
# "Disk quota exceeded" at their first print(). Relative path resolves to the
# sbatch submit dir (repo root). %x = job name, %j = job id.
#SBATCH --output=slurm_reports/%x_%j.out
#SBATCH --error=slurm_reports/%x_%j.err
#
# Driven homo scheme-1 susceptibility run at 3x the 1x baseline eq+prod time,
# for the driven-time study (does the non-equilibrium regime depend on run time?).
# Single epsilon, looping square sizes L = 16 .. 128. Submitted once per epsilon by
# sweep_susceptibility.py. Uses SLURM_CPUS_PER_TASK parallel replicas per job and
# appends replicas to each per-(L, eps) susceptibility_data.csv.
#
# Args (passed by sbatch/bash):
#   $1  epsilon       (required)
#   $2  results_base  (required; e.g. susceptibility/results/exact_2026-07-02)
#   $3  num_batches   (optional, default 1)
#   $4  mu            (optional; empty => runner default mu = 2*epsilon)
#   $5  delta_f       (optional; empty => runner default)
#   $6  delta_mu      (optional; empty => runner default)
#   $7  k             (optional; empty => runner default)
#   $8  scheme        (optional; empty => runner default)
# The optional args let sweep_susceptibility.py --mu-source drive fitted-mu runs
# for an arbitrary scheme; with all empty this reproduces the exact-mu Ising run.

set -euo pipefail

EPS=$1
RESULTS_BASE=$2
NUM_BATCHES=${3:-1}
MU=${4:-}
DELTA_F=${5:-}
DELTA_MU=${6:-}
K_ARG=${7:-}
SCHEME=${8:-}
N=${SLURM_CPUS_PER_TASK:-2}     # 16 under SLURM -> num_parallel_runs; 2 locally

# Optional runner flags: only added when the corresponding arg is non-empty.
EXTRA_ARGS=()
[[ -n "$MU"       ]] && EXTRA_ARGS+=(--mu "$MU")
[[ -n "$DELTA_F"  ]] && EXTRA_ARGS+=(--delta-f "$DELTA_F")
[[ -n "$DELTA_MU" ]] && EXTRA_ARGS+=(--delta-mu "$DELTA_MU")
[[ -n "$K_ARG"    ]] && EXTRA_ARGS+=(--k "$K_ARG")
[[ -n "$SCHEME"   ]] && EXTRA_ARGS+=(--scheme "$SCHEME")

# ---- Tunables (kept here so they are easy to change) ------------------------
# 3x the driven 1x baseline (EQ 100000, PROD 1000000, CHUNKS 10000). Both eq and
# prod time triple; PROD_CHUNKS triples too so chunk_time = PROD/CHUNKS = 100 is
# unchanged (more samples at the same cadence, longer equilibration).
EQ_TIME=300000.0
PROD_TIME=3000000.0
PROD_CHUNKS=30000
SEED_BASE=7000
# High-side fill: even run_ids seed 80% BONDING (0.8), odd run_ids seed the
# mirror 20% (0.2), so replicas sample both magnetization branches.
INITIAL_FRACTION=0.8

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

SIZES=(16 32 48 64 96 128)

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
done
