#!/usr/bin/env bash
# Experiment C2B: bimodality criticality per slab size, then FSS across sizes.
#
# Physics: scheme=homo, βΔf=0, k=1, βΔμ=-1
# Reads  COEX_RUNS_C2B/ly<N>/{results,manage.csv}  from ./coex/run_c2b_coex_campaign.sh
# Writes COEX_RUNS_C2B/criticality/ly<N>/{bc_vs_beta_epsilon.csv,criticality.csv,*.png}
#        COEX_RUNS_C2B/criticality/multi_L/   (βε_c and βμ_coex(ε_c) vs L)
#
# Usage:
#   ./coex/run_c2b_criticality.sh 16          # after the scout finishes: read ε_c
#   ./coex/run_c2b_criticality.sh             # Ly 16 20, after refine finishes
#   ./coex/run_c2b_criticality.sh compare     # ε_c table + FSS plots
#
# The scout's ε_c feeds the refine phase:
#   ./coex/run_c2b_coex_campaign.sh refine "$(awk -F, 'NR==2{print $9}' \
#       COEX_RUNS_C2B/criticality/ly16/criticality.csv)"

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR:$PROJECT_DIR/criticality"

CAMPAIGN_ROOT=COEX_RUNS_C2B
CRIT_ROOT="$CAMPAIGN_ROOT/criticality"
CRIT_PREFIX=ly
SCHEME=homo
DELTA_F=0.0
DELTA_MU=-1.0
K=1.0
DEFAULT_LYS=(16 20)

CMD="${1:-run}"
case "$CMD" in
  run|compare) shift || true ;;
  [0-9]*) CMD=run ;;
  *) echo "usage: $0 [run|compare] [Ly ...]"; exit 1 ;;
esac
if [ "$#" -gt 0 ]; then
  LYS=("$@")
else
  LYS=("${DEFAULT_LYS[@]}")
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
  echo "== criticality Ly=$ly (Lx=$lx) from $base/results -> $out =="
  mkdir -p "$out"
  # phase-diagram appends; clear stale CSVs so re-runs don't duplicate rows
  rm -f "$out/bc_vs_beta_epsilon.csv" "$out/criticality.csv"
  # --delta-mus takes nargs="*"; the equals form keeps the negative value from
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
echo "Done. Compare with:  ./coex/run_c2b_criticality.sh compare"
