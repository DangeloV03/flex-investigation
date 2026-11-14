#!/bin/bash
#SBATCH --job-name=sweep_rev_dfeff_dmu3_rest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-cpu=512M
#SBATCH --time=48:00:00
#SBATCH -o /home/mo9089/slurm_reports/%A_%a.out
#SBATCH --error=/home/mo9089/slurm_reports/%A_%a.err
#SBATCH --array=0
#
# Run every (L, epsilon, mu) row of MU_DATABASE on --cpus-per-task CPUs:
#     sbatch sweep_rev_dfeff_dmu3_rest.sh
# Reruns the 7 L=48 points (eps -1.71..-1.65) left unfinished when job 17222205 hit its 48h limit.
# All parameters are set below; they are written to $OUTROOT/sweep_params_rest.json,
# which sweep.py reads. Afterwards average.py writes <m> and chi per row, renamed to
# mu_database_rev_dfeff_dmu3_rest_iteration1.csv so other sweeps' results aren't overwritten.

set -euo pipefail

# ---- sweep parameters (L, epsilon, mu come from the CSV rows) ----
MU_DATABASE=mu_database_rev_dfeff_dmu3_rest.csv   # CSV: lattice_size,epsilon,mu (e.g. from setup_sweep.py)
OUTROOT=/scratch/gpfs/WJACOBS/MLO/3SM_CORRECTIONS
N_REPLICAS=16                 # independent trajectories per row (must be even)
DELTA_MU=0.0                   # reversible map of homo DMU=3, DF=0, K=1 (flex_mappings/)
DELTA_F=2.61912               # = calc_df_eff(0, 3, 1); mu = coex from estimate_coex_df0.py
K=1.0
SCHEME=homo
BETA=1.0
EQ_TIME=2e6
PROD_TIME=5e6
PROD_CHUNKS=2000
INITIAL_M=0.8
RANDOM_INITIAL=false          # true: ignore INITIAL_M, draw each initial m uniformly from [-1, 1]
SEED=$RANDOM
CONTINUATION=false            # true: restart every replica from its final_lattice_<i>.npy in
                              # OUTROOT (no equilibration) and append to m_timeseries_<i>.csv
# ------------------------------------------------------------------

OUTROOT="$OUTROOT/SCHEME_${SCHEME}_DF_${DELTA_F}_K_${K}_DMU_${DELTA_MU}"

CONTINUATION=$(echo "$CONTINUATION" | tr '[:upper:]' '[:lower:]')  # JSON needs true/false
[[ $CONTINUATION == true || $CONTINUATION == false ]] || { echo "CONTINUATION must be true or false"; exit 1; }
RANDOM_INITIAL=$(echo "$RANDOM_INITIAL" | tr '[:upper:]' '[:lower:]')
[[ $RANDOM_INITIAL == true || $RANDOM_INITIAL == false ]] || { echo "RANDOM_INITIAL must be true or false"; exit 1; }

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUTROOT"

cat >> "$OUTROOT/claude_intent.md" <<INTENT_EOF
# Claude Intent Log (sweep_rev_dfeff_dmu3_rest.sh)

## Why this job exists
Please check if all the jobs finished within the last two days have finished succefully.
-> job 17222205 (sweep_rev_dfeff_dmu3.sh) hit its 48h limit with 41/48 points done.
yes, set up the rerun for those 7 points and submit
-> reruns the 7 unfinished L=48 points from scratch (16 replicas each, 56 CPUs, 2 rounds).

## Parameters used
MU_DATABASE=${MU_DATABASE}
$(cat "$MU_DATABASE")
N_REPLICAS=${N_REPLICAS}  DELTA_MU=${DELTA_MU}  DELTA_F=${DELTA_F}  K=${K}  SCHEME=${SCHEME}  BETA=${BETA}
EQ_TIME=${EQ_TIME}  PROD_TIME=${PROD_TIME}  PROD_CHUNKS=${PROD_CHUNKS}  INITIAL_M=${INITIAL_M}
RANDOM_INITIAL=${RANDOM_INITIAL}  SEED=${SEED}  CONTINUATION=${CONTINUATION}
Generated: $(date)
Slurm job: ${SLURM_JOB_NAME}, job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}

INTENT_EOF

PARAMS="$OUTROOT/sweep_params_rest.json"
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
  "random_initial": $RANDOM_INITIAL,
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
python -u src/flexsim/susc/average.py "$MU_DATABASE" "$OUTROOT"
mv "$(dirname "$MU_DATABASE")/mu_database_iteration1.csv" "${MU_DATABASE%.csv}_iteration1.csv"
