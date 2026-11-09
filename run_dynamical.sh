#!/bin/bash
#SBATCH --job-name=dynamical
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=256M
#SBATCH --time=07:59:59
#SBATCH -o /home/mo9089/slurm_reports/%A_%a.out
#SBATCH --error=/home/mo9089/slurm_reports/%A_%a.err
#SBATCH --array=0
#
# Relaxation m(t) from the ordered state: no equilibration, 32 trajectories per
# (L, epsilon, mu) row starting at |m| = 1 (even replicas +1, odd -1), sampled every
# 10 kMC time units (PROD_TIME / PROD_CHUNKS) over 1e6.
# Two cases run one after the other, each over L = 16, 24, 32 at epsilon = -1.6:
#   case 1: dmu=0, k=0, df=-20, mu=-3.2       (mu_database_dynamical_dmu0.csv)
#   case 2: dmu=1, k=1, df=0,   mu=-2.65564   (mu_database_dynamical_dmu1.csv, coexistence)
#     sbatch run_dynamical.sh
# Each case writes $OUTROOT/SCHEME_..._DF_..._K_..._DMU_.../sweep_params.json, which
# sweep.py reads; m_timeseries_<i>.csv in each L*_eps*_mu* folder holds m(t).

set -euo pipefail

# ---- fixed parameters (L, epsilon, mu come from the CSV rows) ----
OUTROOT_BASE=/scratch/gpfs/WJACOBS/MLO/3SM_DYNAMICAL
N_REPLICAS=32                 # independent trajectories per row (must be even)
SCHEME=homo
BETA=1.0
EQ_TIME=0
PROD_TIME=1e6
PROD_CHUNKS=100000            # chunk length PROD_TIME/PROD_CHUNKS = 10 kMC time
INITIAL_M=1.0
RANDOM_INITIAL=false
CONTINUATION=false
# ---- cases: MU_DATABASE DELTA_MU K DELTA_F ----
CASES=(
    "mu_database_dynamical_dmu0.csv 0.0 0.0 -20.0"
    "mu_database_dynamical_dmu1.csv 1.0 1.0 0.0"
)
# ------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR"

set +u  # module/conda scripts reference unset vars (e.g. PS1) in batch shells
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lattice
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

for CASE in "${CASES[@]}"; do
    read -r MU_DATABASE DELTA_MU K DELTA_F <<< "$CASE"
    SEED=$RANDOM
    OUTROOT="$OUTROOT_BASE/SCHEME_${SCHEME}_DF_${DELTA_F}_K_${K}_DMU_${DELTA_MU}"
    mkdir -p "$OUTROOT"

    cat > "$OUTROOT/claude_intent.md" <<'PROMPT_EOF'
# Claude Intent Log

## Why this job exists
I want to rework the @run_dynamical.sh set up a bit. I want 32 trajectories on 32 CPUS to start at m=1. No eq time and then we measure m(t) in time steps of 100 kMC time over a production time of 10^6. Within one script do that for two cases: at epsilon=-1.6, L=32. The first case is \Delta \mu=0, k=0, \Delta f=-20, and mu=-3.2. The second case is k=1, \Delta f=0, \Delta \mu=1.0 and the corresponding coexistence chemical potential which is -2.65564. Ask me any questions you might have before starting. You can set it up so that it overwrites the data in the current directory, that is fine

Follow-up: 1. The +-m split is fine. 2. Homo scheme please. 3. Run the two cases one after another they are fast enough. 4. Yes. Also please not only do it for L=32 but also for L=16 and L=24

Follow-up: Okay I checked the results and I think we need to change it slightly so that we measure every 10kMC time units. The relaxation process is very fast.
PROMPT_EOF

    cat >> "$OUTROOT/claude_intent.md" <<PARAMS_EOF

## Parameters used
MU_DATABASE=${MU_DATABASE}
$(cat "$MU_DATABASE")
N_REPLICAS=${N_REPLICAS}
DELTA_MU=${DELTA_MU}
DELTA_F=${DELTA_F}
K=${K}
SCHEME=${SCHEME}
BETA=${BETA}
EQ_TIME=${EQ_TIME}
PROD_TIME=${PROD_TIME}
PROD_CHUNKS=${PROD_CHUNKS}
INITIAL_M=${INITIAL_M}
RANDOM_INITIAL=${RANDOM_INITIAL}
SEED=${SEED}
CONTINUATION=${CONTINUATION}
Generated: $(date)
Slurm job: ${SLURM_JOB_NAME}, job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}
PARAMS_EOF

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
  "random_initial": $RANDOM_INITIAL,
  "seed": $SEED,
  "continuation": $CONTINUATION
}
EOF

    echo "=== case: $MU_DATABASE (dmu=$DELTA_MU, k=$K, df=$DELTA_F) ==="
    python -u src/flexsim/susc/sweep.py "$PARAMS"
done
