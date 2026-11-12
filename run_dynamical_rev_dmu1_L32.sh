#!/bin/bash
#SBATCH --job-name=dynamical_rev_dmu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=256M
#SBATCH --time=07:59:59
#SBATCH -o /home/mo9089/slurm_reports/%A_%a.out
#SBATCH --error=/home/mo9089/slurm_reports/%A_%a.err
#SBATCH --array=0
#
# Relaxation m(t) from the ordered state for the reversible map of the dmu=1 case:
# k=1, df=0.7633 (= df_eff of df=0, dmu=1, k=1), dmu=0, at L=32, epsilon=-1.6,
# mu=-3.10849 (coex from flex_mappings/estimate_coex_df0.py -1.6 0.7633 0.0 1.0).
# No equilibration, 512 trajectories (on 16 CPUs, 32 rounds) starting at |m| = 1
# (even replicas +1, odd -1), sampled every 10 kMC time units over 1e6.
#     sbatch run_dynamical_rev_dmu1_L32.sh
# Writes $OUTROOT/sweep_params.json, which sweep.py reads; m_timeseries_<i>.csv in the
# L32_eps-1.6_mu-3.10849 folder holds m(t). Refuses to run if $OUTROOT already exists.

set -euo pipefail

# ---- fixed parameters (L, epsilon, mu come from the CSV row) ----
MU_DATABASE=mu_database_dynamical_rev_dmu1_L32.csv
OUTROOT_BASE=/scratch/gpfs/WJACOBS/MLO/3SM_DYNAMICAL
N_REPLICAS=512                # independent trajectories (must be even)
DELTA_MU=0.0
K=1.0
DELTA_F=0.7633
SCHEME=homo
BETA=1.0
EQ_TIME=0
PROD_TIME=1e6
PROD_CHUNKS=100000            # chunk length PROD_TIME/PROD_CHUNKS = 10 kMC time
INITIAL_M=1.0
RANDOM_INITIAL=false
CONTINUATION=false
# ------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR"

set +u  # module/conda scripts reference unset vars (e.g. PS1) in batch shells
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lattice
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

SEED=$RANDOM
OUTROOT="$OUTROOT_BASE/SCHEME_${SCHEME}_DF_${DELTA_F}_K_${K}_DMU_${DELTA_MU}"
if [[ -e "$OUTROOT" ]]; then echo "$OUTROOT already exists, refusing to overwrite"; exit 1; fi
mkdir -p "$OUTROOT"

cat > "$OUTROOT/claude_intent.md" <<'PROMPT_EOF'
# Claude Intent Log

## Why this job exists
Please run a @run_dynamical.sh for L=32 and epsilon=-1.6 for k=1.0, \Delta f = 0.7633, \Delta \mu =0.0. You can get the correct chemical potential from @flex_mappings/. Get 512 replicas on 16 cpus.
-> mu = -3.10849 from flex_mappings/estimate_coex_df0.py -1.6 0.7633 0.0 1.0 (same script gives -2.65565 for the dmu=1 case).
Same protocol as run_dynamical.sh: no eq time, start at |m|=1 (+/- split), homo scheme, m(t) every 10 kMC time over 1e6.
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

python -u src/flexsim/susc/sweep.py "$PARAMS"
