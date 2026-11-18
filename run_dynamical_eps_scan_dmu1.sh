#!/bin/bash
#SBATCH --job-name=dyn_eps_scan_dmu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=256M
#SBATCH --time=24:00:00
#SBATCH -o /home/mo9089/slurm_reports/%A_%a.out
#SBATCH --error=/home/mo9089/slurm_reports/%A_%a.err
#SBATCH --array=0
#
# Relaxation m(t) from the ordered state, epsilon scan, case "dmu1": driven reactions (k=1, df=0, dmu=1)
# L = 32, 24, 16 x epsilon = -1.65, -1.70, -1.75 (9 rows, L=32 first so the
# slowest trajectories start first), mu at coexistence from
# flex_mappings/estimate_coex_df0.py <eps> 0.0 1.0 1.0  (rounded to 5 decimals).
# No equilibration, 512 trajectories per row on 32 CPUs, starting at |m| = 1 (even
# replicas +1, odd -1), sampled every 100 kMC time units over 1e6.
#     sbatch run_dynamical_eps_scan_dmu1.sh
# Writes into the existing case folder $OUTROOT (shared with the eps=-1.6 runs), so the
# provenance goes to claude_intent_eps_scan.md / sweep_params_eps_scan.json there and the
# job refuses to run if any of its own L*_eps*_mu* point folders already exists.

set -euo pipefail

# ---- fixed parameters (L, epsilon, mu come from the CSV rows) ----
MU_DATABASE=mu_database_dynamical_eps_scan_dmu1.csv
OUTROOT_BASE=/scratch/gpfs/WJACOBS/MLO/3SM_DYNAMICAL
N_REPLICAS=512                # independent trajectories per row (must be even)
DELTA_MU=1.0
K=1.0
DELTA_F=0.0
SCHEME=homo
BETA=1.0
EQ_TIME=0
PROD_TIME=1e6
PROD_CHUNKS=10000             # chunk length PROD_TIME/PROD_CHUNKS = 100 kMC time
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
# point folder names exactly as sweep.py builds them (Python float repr)
python - "$MU_DATABASE" "$OUTROOT" <<'PY'
import csv, os, sys
with open(sys.argv[1]) as f:
    dirs = [os.path.join(sys.argv[2], f"L{int(r['lattice_size'])}_eps{float(r['epsilon'])}_mu{float(r['mu'])}")
            for r in csv.DictReader(f)]
clash = [d for d in dirs if os.path.exists(d)]
if clash:
    sys.exit("refusing to overwrite existing point folders:\n" + "\n".join(clash))
PY
mkdir -p "$OUTROOT"

INTENT="$OUTROOT/claude_intent_eps_scan.md"
cat > "$INTENT" <<'PROMPT_EOF'
# Claude Intent Log

## Why this job exists
Lets set up a large scale simulation on della together. Our starting point will be the @run_dynamical_rev_dmu1_L32.sh approach and similar. For L=16,24,32 and epsilon=-1.65, -1.70, -1.75, -1.8 do the following: create individual slurm files for no chemical reactions (k=0, \Delta f = -20, \Delta \mu =0.0), driven chemical reactions (k=1, \Delta f=0.0, \Delta mu=1.0, homo scheme) and reversible chemical reactions (k=1, \Delta f=0.7633, \Delta \mu=0.0) with N=512 independent replicas on 32 CPUs. The coexistence points can be calculated from the @flex_mappings/ folder. Ask me any questions you might have before starting. I want one job per reaction scheme. 3 sizes times 4 epsilon values should easily fit within a 24h job. Also ask for 256MB ram per cpu.

Follow-up answers: reuse the existing case folders in 3SM_DYNAMICAL (new L/eps point folders only), keep 24h, keep sampling every 10 kMC.
Follow-up: removed the epsilon=-1.8 case (L = 16, 24, 32 x epsilon = -1.65, -1.70, -1.75).
Follow-up: sample every 100 kMC instead (PROD_CHUNKS=10000).
-> Same protocol as run_dynamical_rev_dmu1_L32.sh (homo, no eq, |m|=1 +/- split, m(t) over 1e6).
-> This file: driven reactions (k=1, df=0, dmu=1); mu from flex_mappings/estimate_coex_df0.py <eps> 0.0 1.0 1.0.
PROMPT_EOF

cat >> "$INTENT" <<PARAMS_EOF

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

PARAMS="$OUTROOT/sweep_params_eps_scan.json"
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
