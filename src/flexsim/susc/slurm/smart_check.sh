#!/bin/bash
#SBATCH --job-name=susc_check
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_reports/%x_%j.out
#SBATCH --error=slurm_reports/%x_%j.err
#
# Thin wrapper: run `smart_sweep check` and chain another round if any
# (ε, L) pairs still haven't met the jump threshold.  Submitted with
# --dependency=afterok:... by `smart_sweep sweep` and by itself when it
# finds failing pairs and submits top-up jobs.
#
# Args:
#   $1  results_base  (required; SUSC_RUNS directory path)
#   $2  threshold     (optional, default 10.0)
#   $3  check_script  (optional; path to this script, for self-scheduling)

set -euo pipefail

RESULTS_BASE=$1
THRESHOLD=${2:-10.0}
# Default check/top-up scripts come from susc/smart_sweep.py (this package's slurm/ dir).
CHECK_ARGS=()
[[ -n "${3:-}" ]] && CHECK_ARGS+=(--check-script "$3")

if command -v module >/dev/null 2>&1; then
    module load anaconda3/2024.10
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate lattice
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONUNBUFFERED=1

# Run from the output root (the dir sbatch was called from; run/*.py chdir there)
# so relative results paths land in it. Locally we are already there.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p slurm_reports

python -u -m flexsim.susc.smart_sweep check \
    --results-base "$RESULTS_BASE" \
    --threshold "$THRESHOLD" \
    ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"}
