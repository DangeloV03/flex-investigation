#!/usr/bin/env bash
# Bimodality criticality for Scheme-3 (negative_drive) multi-L coex at Δμ=1.0.
#
# Expects scheme3/dmu1p0_multiL/ly<N>/results from
#   ./coex/run_scheme3_multi_L_campaign.sh
# Writes criticality/s3_dmu1_ly<N>/{bc_vs_beta_epsilon.csv,criticality.csv,*.png}.
#
# Usage (after coex is analyzed):
#   ./coex/run_scheme3_multi_L_criticality.sh           # all of 16 20 40
#   ./coex/run_scheme3_multi_L_criticality.sh 20 40
#   ./coex/run_scheme3_multi_L_criticality.sh compare

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR:$PROJECT_DIR/criticality"

DEFAULT_LYS=(16 20 40)
SCHEME=negative_drive
DELTA_F=0.0
K=0.1
DELTA_MU=1.0
COEX_ROOT=scheme3/dmu1p0_multiL
CRIT_PREFIX=s3_dmu1_ly

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
for p in sorted(Path("criticality").glob("${CRIT_PREFIX}*/criticality.csv")):
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            r["_src"] = str(p)
            rows.append(r)
if not rows:
    raise SystemExit(
        "No criticality/${CRIT_PREFIX}*/criticality.csv found — "
        "run this script without 'compare' first."
    )
print("L_short\tL_long\tepsilon_c\tfit_uncertainty\trecommended_uncertainty\tsource")
for r in sorted(rows, key=lambda x: int(float(x["L_short"]))):
    print(
        f"{r['L_short']}\t{r['L_long']}\t{float(r['epsilon_c_estimate']):.6f}\t"
        f"{r.get('fit_uncertainty','')}\t{r.get('recommended_uncertainty','')}\t{r['_src']}"
    )
PY
  echo
  echo "== clearing criticality/s3_dmu1_multi_L and writing FSS plots =="
  rm -rf criticality/s3_dmu1_multi_L
  python -u criticality/plot_eq_L_scaling.py \
    --lys "${LYS_COMPARE[@]}" \
    --coex-root "$COEX_ROOT" \
    --crit-prefix "$CRIT_PREFIX" \
    --out-dir criticality/s3_dmu1_multi_L
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

for ly in "${LYS[@]}"; do
  lx=$((10 * ly))
  base="${COEX_ROOT}/ly${ly}"
  out="criticality/${CRIT_PREFIX}${ly}"
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
echo "Done. Compare with:  ./coex/run_scheme3_multi_L_criticality.sh compare"
