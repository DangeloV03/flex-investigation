#!/usr/bin/env bash
# Run bimodality criticality for each finished equilibrium coex size.
#
# Expects coex/coex_eq/ly<N>/results from ./coex/run_eq_multi_L_campaign.sh.
# Writes under criticality/equilibrium_multi_L_testing/:
#   ly<N>/{bc_vs_beta_epsilon.csv,criticality.csv,*.png}
#   multi_L/   (FSS / mentor compare plots)
#
# Usage (after coex is analyzed):
#   ./coex/run_eq_multi_L_criticality.sh           # all of 16 20 40
#   ./coex/run_eq_multi_L_criticality.sh 20 40     # subset
#   ./coex/run_eq_multi_L_criticality.sh compare   # table + βε_c / βμ_coex vs L plots
#
# One-time migrate of old flat paths (eq_ly*, eq_multi_L, eq_validation):
#   ./coex/migrate_criticality_layout.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR:$PROJECT_DIR/criticality"

DEFAULT_LYS=(16 20 40)
SCHEME=homo
DELTA_F=-20.0
K=0.0
DELTA_MU=0.0
COEX_ROOT=coex/coex_eq
CRIT_ROOT=criticality/equilibrium_multi_L_testing
CRIT_PREFIX=ly

CMD="${1:-run}"
if [ "$CMD" = "compare" ]; then
  shift || true
  if [ "$#" -gt 0 ]; then
    LYS_COMPARE=("$@")
  else
    LYS_COMPARE=("${DEFAULT_LYS[@]}")
  fi
  python - <<PY
import csv
from pathlib import Path
rows = []
root = Path("${CRIT_ROOT}")
for p in sorted(root.glob("${CRIT_PREFIX}*/criticality.csv")):
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            r["_src"] = str(p)
            rows.append(r)
if not rows:
    raise SystemExit(
        f"No {root}/${CRIT_PREFIX}*/criticality.csv found — "
        "run this script without 'compare' first "
        "(or ./coex/migrate_criticality_layout.sh if data is still at criticality/eq_ly*)."
    )
print("L_short\tL_long\tepsilon_c\tfit_uncertainty\trecommended_uncertainty\tsource")
for r in sorted(rows, key=lambda x: int(float(x["L_short"]))):
    print(
        f"{r['L_short']}\t{r['L_long']}\t{float(r['epsilon_c_estimate']):.6f}\t"
        f"{r.get('fit_uncertainty','')}\t{r.get('recommended_uncertainty','')}\t{r['_src']}"
    )
PY
  echo
  echo "== clearing ${CRIT_ROOT}/multi_L and writing FSS plot =="
  rm -rf "${CRIT_ROOT}/multi_L"
  python -u criticality/plot_eq_L_scaling.py \
    --lys "${LYS_COMPARE[@]}" \
    --coex-root "$COEX_ROOT" \
    --crit-root "$CRIT_ROOT" \
    --crit-prefix "$CRIT_PREFIX" \
    --out-dir "${CRIT_ROOT}/multi_L"
  exit 0
fi

if [ "$CMD" = "run" ]; then
  shift || true
fi
if [ "$#" -gt 0 ]; then
  LYS=("$@")
else
  LYS=("${DEFAULT_LYS[@]}")
fi

mkdir -p "$CRIT_ROOT"

for ly in "${LYS[@]}"; do
  lx=$((10 * ly))
  base="${COEX_ROOT}/ly${ly}"
  out="${CRIT_ROOT}/${CRIT_PREFIX}${ly}"
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
  python -u criticality/bimodality.py phase-diagram \
    --base-dir "$base/results" \
    --scheme "$SCHEME" --delta-f "$DELTA_F" --k "$K" \
    --Lx "$lx" --Ly "$ly" --delta-mus "$DELTA_MU" \
    --mu-reduction zero_mean \
    --out-dir "$out" \
    --manage-csv "$base/manage.csv"
done

echo
echo "Done. Compare with:  ./coex/run_eq_multi_L_criticality.sh compare"
