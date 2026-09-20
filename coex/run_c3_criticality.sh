#!/usr/bin/env bash
# Experiments C3A–C3E: bimodality criticality per slab size, then FSS across sizes.
#
# Physics: scheme=negative_drive, βΔf=0, k=1, βΔμ per experiment (see table below).
# Reads  COEX_RUNS_<EXP>/ly<N>/{results,manage.csv}  from ./coex/run_c3_coex_campaign.sh
# Writes COEX_RUNS_<EXP>/criticality/ly<N>/{bc_vs_beta_epsilon.csv,criticality.csv,*.png}
#        COEX_RUNS_<EXP>/criticality/multi_L/   (βε_c and βμ_coex(ε_c) vs L)
#
# Usage:
#   ./coex/run_c3_criticality.sh C3A              # this host's default Ly
#   ./coex/run_c3_criticality.sh C3A 16 20 40     # explicit sizes
#   ./coex/run_c3_criticality.sh C3A compare      # ε_c table + FSS plots
#
# `compare` is the merged step: run it only after Della's ly40 has been rsynced
# in next to the workstation's ly16/ly20, so all three sizes sit under one root.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR:$PROJECT_DIR/criticality"

SCHEME=negative_drive
DELTA_F=0.0
K=1.0
CRIT_PREFIX=ly

EXP="${1:-}"
if [ -z "$EXP" ]; then
  echo "usage: $0 <C3A|C3B|C3C|C3D|C3E> [run|compare] [Ly ...]"
  exit 1
fi
shift
EXP="$(echo "$EXP" | tr '[:lower:]' '[:upper:]')"

case "$EXP" in
  C3A) DELTA_MU=1.0 ;;
  C3B) DELTA_MU=-1.0 ;;
  C3C) DELTA_MU=2.0 ;;
  C3D) DELTA_MU=3.0 ;;
  C3E) DELTA_MU=4.0 ;;
  *) echo "Unknown experiment '$EXP' (expected C3A..C3E)"; exit 1 ;;
esac

CAMPAIGN_ROOT="COEX_RUNS_$EXP"
CRIT_ROOT="$CAMPAIGN_ROOT/criticality"

CMD="${1:-run}"
case "$CMD" in
  run|compare) shift || true ;;
  [0-9]*) CMD=run ;;
  *) echo "usage: $0 <C3A..C3E> [run|compare] [Ly ...]"; exit 1 ;;
esac

if [ "$#" -gt 0 ]; then
  LYS=("$@")
elif [ "$CMD" = "compare" ]; then
  # compare runs after the merge, so it wants every size.
  LYS=(16 20 40)
elif command -v sbatch >/dev/null 2>&1; then
  LYS=(40)
else
  LYS=(16 20)
fi

if [ "$CMD" = "compare" ]; then
  python - "$CRIT_ROOT" "$CRIT_PREFIX" <<'PY'
import csv, sys
from pathlib import Path
root, prefix = Path(sys.argv[1]), sys.argv[2]
rows = []
for p in sorted(root.glob(f"{prefix}*/criticality.csv")):
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            r["_src"] = str(p)
            rows.append(r)
if not rows:
    raise SystemExit(f"No {root}/{prefix}*/criticality.csv — run without 'compare' first.")
print("L_short\tL_long\tepsilon_c\tfit_uncertainty\trecommended_uncertainty\tsource")
for r in sorted(rows, key=lambda x: int(float(x["L_short"]))):
    print(
        f"{r['L_short']}\t{r['L_long']}\t{float(r['epsilon_c_estimate']):.6f}\t"
        f"{r.get('fit_uncertainty','')}\t{r.get('recommended_uncertainty','')}\t{r['_src']}"
    )
PY
  echo
  echo "== clearing ${CRIT_ROOT}/multi_L and writing FSS plots =="
  rm -rf "${CRIT_ROOT}/multi_L"
  python -u criticality/plot_eq_L_scaling.py \
    --lys "${LYS[@]}" \
    --coex-root "$CAMPAIGN_ROOT" \
    --crit-root "$CRIT_ROOT" \
    --crit-prefix "$CRIT_PREFIX" \
    --out-dir "${CRIT_ROOT}/multi_L"
  exit 0
fi

mkdir -p "$CRIT_ROOT"
for ly in "${LYS[@]}"; do
  lx=$((10 * ly))
  base="$CAMPAIGN_ROOT/ly${ly}"
  out="$CRIT_ROOT/${CRIT_PREFIX}${ly}"
  if [ ! -d "$base/results" ]; then
    echo "[skip] missing $base/results"
    continue
  fi
  n_dirs=$(find "$base/results" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n_dirs" -eq 0 ]; then
    echo "[skip] no combo dirs yet in $base/results"
    continue
  fi
  echo "== $EXP criticality Ly=$ly (Lx=$lx) from $base/results -> $out =="
  mkdir -p "$out"
  # phase-diagram appends; clear stale CSVs so re-runs don't duplicate rows
  rm -f "$out/bc_vs_beta_epsilon.csv" "$out/criticality.csv"
  # --delta-mus takes nargs="*"; the equals form keeps a negative value from
  # being read as an option name.
  python -u criticality/bimodality.py phase-diagram \
    --base-dir "$base/results" \
    --scheme "$SCHEME" --delta-f "$DELTA_F" --k "$K" \
    --Lx "$lx" --Ly "$ly" --delta-mus="$DELTA_MU" \
    --mu-reduction zero_mean \
    --out-dir "$out" \
    --manage-csv "$base/manage.csv"
done

echo
echo "Done. Compare with:  $0 $EXP compare"
