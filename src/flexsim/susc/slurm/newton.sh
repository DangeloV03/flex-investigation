#!/bin/bash
#SBATCH --job-name=susc_newton
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=512M
#SBATCH --time=23:59:59
#SBATCH --output=slurm_reports/%x_%j.out
#SBATCH --error=slurm_reports/%x_%j.err
#
# One step of a Newton mu chain (susc/newton.py) for one (eps, L).
#
#   newton.sh segment <pair_dir>   run the next simulation segment (16 replicas)
#   newton.sh decide  <pair_dir>   measure <m>, pick the next action, submit it
#
# newton.py submits both: the segment, then decide with --dependency=afterany on it
# (and --cpus-per-task=1 --time=00:30:00 on the command line), so a segment that
# hits the wall still gets its chain continued.

set -eo pipefail     # not -u: conda activate reads unset PS1 in some shells

MODE=$1
PAIR_DIR=$2
N=${SLURM_CPUS_PER_TASK:-2}

if command -v module >/dev/null 2>&1; then
    module load anaconda3/2024.10
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate lattice
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONUNBUFFERED=1

# Run from the output root (where sbatch was called), so the next sbatch from the
# decide job writes its slurm_reports/ logs there too.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
fi

case "$MODE" in
    segment)
        echo "=== Newton segment: $PAIR_DIR (cpus=$N) ==="
        SECONDS=0
        python -u -m flexsim.susc.newton segment --pair-dir "$PAIR_DIR" --cpus "$N"
        echo ">>> segment took ${SECONDS}s ($((SECONDS/60))m$((SECONDS%60))s)"
        ;;
    decide)
        echo "=== Newton decide: $PAIR_DIR ==="
        python -u -m flexsim.susc.newton decide --pair-dir "$PAIR_DIR"
        ;;
    *)
        echo "usage: newton.sh segment|decide <pair_dir>" >&2
        exit 2
        ;;
esac
