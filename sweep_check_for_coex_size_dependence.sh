#!/bin/bash
#SBATCH --job-name=coex_size_dependence
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=256M
#SBATCH --time=02:00:00
#SBATCH -o /home/mo9089/slurm_reports/%A_%a.out
#SBATCH --error=/home/mo9089/slurm_reports/%A_%a.err
#SBATCH --array=0

set -euo pipefail

# This script is meant to find out about the size dependence of mu_coex(epsilon;L). In order to calculate that I only want <m>(mu) which 
# should reach steady-state reasonably fast even near criticality. Therefore, 4x EQ + 1x PROD is enough.
# Also four replicas should be enough!

# ---- sweep parameters (L, epsilon, mu come from the CSV rows) ----
MU_DATABASE=mu_database_sparse_neg_drive_dmu3_L24.csv   # CSV: lattice_size,epsilon,mu (e.g. from setup_sweep.py)
OUTROOT=/scratch/gpfs/WJACOBS/MLO/3SM_COEX_SIZE_DEPENDENCE
N_REPLICAS=4                 # independent trajectories per row (must be even)
DELTA_MU=3.0
DELTA_F=0.0
K=1.0
SCHEME=negative_drive
BETA=1.0
EQ_TIME=4e6
PROD_TIME=1e6
PROD_CHUNKS=500
INITIAL_M=0.0
SEED=$RANDOM
CONTINUATION=false            # true: restart every replica from its final_lattice_<i>.npy in
                              # OUTROOT (no equilibration) and append to m_timeseries_<i>.csv
# ------------------------------------------------------------------

OUTROOT="$OUTROOT/SCHEME_${SCHEME}_DF_${DELTA_F}_K_${K}_DMU_${DELTA_MU}"

CONTINUATION=$(echo "$CONTINUATION" | tr '[:upper:]' '[:lower:]')  # JSON needs true/false
[[ $CONTINUATION == true || $CONTINUATION == false ]] || { echo "CONTINUATION must be true or false"; exit 1; }

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
  "seed": $SEED,
  "continuation": $CONTINUATION
}
EOF

set +u  # module/conda scripts reference unset vars (e.g. PS1) in batch shells
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lattice
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u src/flexsim/susc/sweep.py "$PARAMS"