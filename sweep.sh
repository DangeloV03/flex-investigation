#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=512M
#SBATCH --time=23:59:59
#SBATCH -o /home/mo9089/slurm_reports/%A_%a.out
#SBATCH --error=/home/mo9089/slurm_reports/%A_%a.err
#SBATCH --array=0
#
# Run every (L, epsilon, mu) row of MU_DATABASE on --cpus-per-task CPUs:
#     sbatch sweep.sh
# All parameters are set below; they are written to $OUTROOT/sweep_params.json,
# which sweep.py reads. Afterwards average.py writes mu_database_iteration1.csv
# (next to MU_DATABASE) with <m> and chi per row.

set -euo pipefail

# ---- sweep parameters (L, epsilon, mu come from the CSV rows) ----
MU_DATABASE=mu_database.csv   # CSV: lattice_size,epsilon,mu (e.g. from setup_sweep.py)
OUTROOT=/scratch/gpfs/WJACOBS/MLO/3SM_CORRECTIONS
N_REPLICAS=16                 # independent trajectories per row (must be even)
DELTA_MU=1.0
DELTA_F=0.0
K=1.0
SCHEME=homo
BETA=1.0
EQ_TIME=2e6
PROD_TIME=5e6
PROD_CHUNKS=2000
INITIAL_M=0.8
SEED=$RANDOM
# ------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUTROOT"
PARAMS="$OUTROOT/sweep_params.json"
cat > "$PARAMS" <<EOF
{
  "mu_database": "$MU_DATABASE",
  "outroot": "$OUTROOT",
  "n_replicas": $N_REPLICAS,
  "delta_mu": $DELTA_MU,
  "delta_f": $DELTA_F,
  "k": $K,
  "scheme": "$SCHEME",
  "beta": $BETA,
  "eq_time": $EQ_TIME,
  "prod_time": $PROD_TIME,
  "prod_chunks": $PROD_CHUNKS,
  "initial_m": $INITIAL_M,
  "seed": $SEED
}
EOF

set +u  # module/conda scripts reference unset vars (e.g. PS1) in batch shells
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lattice
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u src/flexsim/susc/sweep.py "$PARAMS"
python -u src/flexsim/susc/average.py "$MU_DATABASE" "$OUTROOT"
