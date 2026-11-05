#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=512M
#SBATCH --time=23:59:59
#SBATCH --output=sweep_%j.out
#
# Run every (L, epsilon, mu) row of a CSV on --cpus-per-task CPUs:
#     sbatch sweep.sh mu_database.csv
# Fixed parameters and replicas per row (M) are set at the top of sweep.py.

set -euo pipefail
CSV=${1:?usage: sbatch sweep.sh mu_database.csv}

module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lattice
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$SLURM_SUBMIT_DIR"
python -u src/flexsim/susc/sweep.py "$CSV"
