#!/usr/bin/env bash
# Equilibrium 128x128 spin-spin correlation runs: eps = -1.7627 (eps_c) and -1.65,
# 4 replicas each = 8 Slurm array tasks (1 CPU each). Run from the repo on Della:
#
#   bash experiments/2026-11-10_spin_corr_eq128/submit.sh smoke    # ~1 min test on the login node
#   bash experiments/2026-11-10_spin_corr_eq128/submit.sh launch   # submit the 8 tasks + 3 resume rounds
#   bash experiments/2026-11-10_spin_corr_eq128/submit.sh status   # queue + progress per replica
#   bash experiments/2026-11-10_spin_corr_eq128/submit.sh analyze  # C(r), xi, autocorrelation, plots
#
# Every task is resumable: it checkpoints after each chunk. 'launch' also queues RESUME_ROUNDS
# follow-up arrays (afterany) that resume anything that ran out of wall time; they exit
# immediately for replicas that are done. Override defaults with env vars, e.g.
#   PROD_TIME=40000000 RESUME_ROUNDS=5 bash .../submit.sh launch

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/WJACOBS/vd7294/flex-investigation}"
OUT="${OUT:-$DATA_ROOT/SPIN_CORR_EQ128}"
EPS_LIST=(${EPS_LIST:--1.7627 -1.65})
NREP="${NREP:-4}"
EQ_TIME="${EQ_TIME:-1000000}"
PROD_TIME="${PROD_TIME:-20000000}"
CHUNK_TIME="${CHUNK_TIME:-1000}"
WALL="${WALL:-23:59:00}"
RESUME_ROUNDS="${RESUME_ROUNDS:-3}"
CONDA_ENV="${FLEX_CONDA_ENV:-lattice}"

activate() {
  set +u   # module/conda scripts reference unset variables (e.g. PS1)
  module load anaconda3/2024.10 2>/dev/null || true
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  set -u
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONUNBUFFERED=1
}

do_smoke() {
  activate
  local tmp="$OUT/_smoke"
  rm -rf "$tmp"
  python "$HERE/spin_corr_run.py" --eps -1.7627 --replica 0 --out "$tmp" --smoke
  # second call must resume and report "already done"
  python "$HERE/spin_corr_run.py" --eps -1.7627 --replica 0 --out "$tmp" --smoke
  python "$HERE/spin_corr_analyze.py" --out "$tmp" --plots "$tmp/plots"
  echo "Smoke test OK. Output in $tmp (safe to delete)."
}

do_launch() {
  mkdir -p "$OUT/logs"
  local n=$(( ${#EPS_LIST[@]} * NREP ))
  cp "$HERE/spin_corr_run.py" "$OUT/spin_corr_run.py"   # freeze the code this campaign ran with
  local script="$OUT/task.sh"
  cat > "$script" <<EOF
#!/bin/bash
#SBATCH --job-name=spincorr128
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=$WALL
#SBATCH --array=0-$((n - 1))
#SBATCH --output=$OUT/logs/%x_%A_%a.out
#SBATCH --error=$OUT/logs/%x_%A_%a.err
module load anaconda3/2024.10
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
EPS_LIST=(${EPS_LIST[*]})
EPS=\${EPS_LIST[\$((SLURM_ARRAY_TASK_ID / $NREP))]}
REP=\$((SLURM_ARRAY_TASK_ID % $NREP))
echo "[task] \$(hostname) eps=\$EPS replica=\$REP"
python $OUT/spin_corr_run.py --eps "\$EPS" --replica "\$REP" --out $OUT \\
  --eq-time $EQ_TIME --prod-time $PROD_TIME --chunk-time $CHUNK_TIME
EOF
  local jid; jid=$(sbatch --parsable "$script")
  echo "round 0: array job $jid ($n tasks)" | tee -a "$OUT/launch.log"
  for r in $(seq 1 "$RESUME_ROUNDS"); do
    jid=$(sbatch --parsable --dependency=afterany:"$jid" "$script")
    echo "round $r (resume): array job $jid" | tee -a "$OUT/launch.log"
  done
  echo "eps=${EPS_LIST[*]} NREP=$NREP eq=$EQ_TIME prod=$PROD_TIME chunk=$CHUNK_TIME -> $OUT" | tee -a "$OUT/launch.log"
}

do_status() {
  squeue -u "$USER" -h -n spincorr128 -o "%i %T %M %R" || true
  for f in "$OUT"/eps*/replica_*/progress.json; do
    [[ -f "$f" ]] && echo "$(dirname "${f#$OUT/}"): $(cat "$f")"
  done
}

do_analyze() {
  activate
  python "$HERE/spin_corr_analyze.py" --out "$OUT" --plots "$OUT/plots"
}

case "${1:-}" in
  smoke) do_smoke ;;
  launch) do_launch ;;
  status) do_status ;;
  analyze) do_analyze ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
