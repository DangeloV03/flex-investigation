#!/usr/bin/env bash
# Submit autocorrelation test as 9 Slurm jobs (one per skip value) plus a
# final mega-plot job that runs only after all 9 succeed.
#
# Usage (from project root on Della):
#   ./scripts/submit_correlation_test.sh

set -euo pipefail
cd "$(dirname "$0")/.."

SKIPS=(0 1 2 3 5 7 10 12 15)
RESULTS="susceptibility_results/exact"
OUTDIR="plots/susceptibility/correlation_test"
PARTITION="cpu"
TIME="02:00:00"
MEM="16G"

SETUP='module load anaconda3/2024.10; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate lattice; export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; export PYTHONUNBUFFERED=1'

mkdir -p "$OUTDIR"
mkdir -p ~/slurm_reports

job_ids=()
for skip in "${SKIPS[@]}"; do
    job_id=$(sbatch \
        --job-name="corr_skip${skip}" \
        --partition="$PARTITION" \
        --cpus-per-task=1 \
        --mem="$MEM" \
        --time="$TIME" \
        --output="$HOME/slurm_reports/%j.out" \
        --error="$HOME/slurm_reports/%j.err" \
        --parsable \
        --wrap="${SETUP}; cd $(pwd) && python -u plot_correlation_test.py --results ${RESULTS} --skips ${skip} --outdir ${OUTDIR}")
    echo "Submitted skip=${skip} → job ${job_id}"
    job_ids+=("$job_id")
done

# Mega plot runs only after all 9 succeed
dep=$(IFS=:; echo "${job_ids[*]}")
mega_id=$(sbatch \
    --job-name="corr_mega" \
    --partition="$PARTITION" \
    --cpus-per-task=1 \
    --mem="4G" \
    --time="00:10:00" \
    --output="$HOME/slurm_reports/%j.out" \
    --error="$HOME/slurm_reports/%j.err" \
    --dependency="afterok:${dep}" \
    --parsable \
    --wrap="${SETUP}; cd $(pwd) && python -u plot_correlation_test.py --mega-only --outdir ${OUTDIR}")
echo "Submitted mega plot → job ${mega_id} (runs after all skip jobs finish)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    cat ~/slurm_reports/<jobid>.out"
