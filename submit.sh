#!/bin/bash
#SBATCH --job-name=susc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=512M
#SBATCH --time=23:59:59
#SBATCH --output=susc_%j.out
#
# Run one susceptibility job on Slurm:
#     sbatch submit.sh params.json
# Set "n_replicas" in the JSON to at most --cpus-per-task.

set -euo pipefail
PARAMS=${1:?usage: sbatch submit.sh params.json}

set +u  # module/conda scripts reference unset vars (e.g. PS1) in batch shells
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lattice
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$SLURM_SUBMIT_DIR"
python -u src/flexsim/susc/replica.py "$PARAMS"
