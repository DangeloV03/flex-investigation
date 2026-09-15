#!/usr/bin/env bash
# Experiment C2A: bimodality criticality for homogeneous driven multi-L coex.
#
# Expects homo/dmu<X>_multiL/ly<N>/results from ./coex/run_homo_multi_L_campaign.sh
# Writes under criticality/homo_multi_L_testing/dmu<X>/:
#   ly<N>/{bc_vs_beta_epsilon.csv,criticality.csv,*.png}
#   multi_L/   (FSS: βε_c and βμ_coex(ε_c) vs L)
#
# Usage (after coex is analyzed):
#   ./coex/run_homo_multi_L_criticality.sh                    # Δμ=1.0, Ly 16 20 40
#   ./coex/run_homo_multi_L_criticality.sh run --lys 20 40
#   ./coex/run_homo_multi_L_criticality.sh compare            # table + FSS plots

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR:$PROJECT_DIR/criticality"

DEFAULT_LYS=(16 20 40)
DEFAULT_DMUS=(1.0)
SCHEME=homo
DELTA_F=0.0
K=1.0
CRIT_PREFIX=ly

CMD="${1:-run}"
case "$CMD" in
  run|compare) shift || true ;;
  --dmu|--lys) CMD=run ;;
  *) echo "usage: $0 [run|compare] [--dmu ...] [--lys ...]"; exit 1 ;;
esac

DMUS=()
LYS=()
MODE=lys
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dmu) MODE=dmu; shift ;;
    --lys) MODE=lys; shift ;;
    *)
      if [ "$MODE" = "dmu" ]; then DMUS+=("$1"); else LYS+=("$1"); fi
      shift
      ;;
  esac
done
[ "${#DMUS[@]}" -eq 0 ] && DMUS=("${DEFAULT_DMUS[@]}")
[ "${#LYS[@]}" -eq 0 ] && LYS=("${DEFAULT_LYS[@]}")

dmu_tag() { echo "dmu$(echo "$1" | sed 's/\./p/')"; }
coex_root_for() { echo "homo/$(dmu_tag "$1")_multiL"; }
crit_root_for() { echo "criticality/homo_multi_L_testing/$(dmu_tag "$1")"; }

run_one_dmu() {
  local dmu="$1"
  local coex_root crit_root
  coex_root="$(coex_root_for "$dmu")"
  crit_root="$(crit_root_for "$dmu")"
  mkdir -p "$crit_root"

  for ly in "${LYS[@]}"; do
    local lx=$((10 * ly))
    local base="${coex_root}/ly${ly}"
    local out="${crit_root}/${CRIT_PREFIX}${ly}"
    if [ ! -d "$base/results" ]; then
      echo "[skip] missing $base/results"
      continue
    fi
    local n_dirs
    n_dirs=$(find "$base/results" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_dirs" -eq 0 ]; then
      echo "[skip] no combo dirs yet in $base/results"
      continue
    fi
    echo "== criticality Δμ=$dmu Ly=$ly (Lx=$lx) from $base/results -> $out =="
    mkdir -p "$out"
    # phase-diagram appends; clear stale CSVs so re-runs don't duplicate rows
    rm -f "$out/bc_vs_beta_epsilon.csv" "$out/criticality.csv"
    python -u criticality/bimodality.py phase-diagram \
      --base-dir "$base/results" \
      --scheme "$SCHEME" --delta-f "$DELTA_F" --k "$K" \
      --Lx "$lx" --Ly "$ly" --delta-mus "$dmu" \
      --mu-reduction zero_mean \
      --out-dir "$out" \
      --manage-csv "$base/manage.csv"
  done
}

compare_one_dmu() {
  local dmu="$1"
  local coex_root crit_root
  coex_root="$(coex_root_for "$dmu")"
  crit_root="$(crit_root_for "$dmu")"

  python - "$crit_root" "$CRIT_PREFIX" "$dmu" <<'PY'
import csv, sys
from pathlib import Path
root, prefix, dmu = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = []
for p in sorted(root.glob(f"{prefix}*/criticality.csv")):
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            r["_src"] = str(p)
            rows.append(r)
if not rows:
    raise SystemExit(f"No {root}/{prefix}*/criticality.csv for Δμ={dmu} — run without 'compare' first.")
print(f"=== Δμ={dmu} ===")
print("L_short\tL_long\tepsilon_c\tfit_uncertainty\trecommended_uncertainty\tsource")
for r in sorted(rows, key=lambda x: int(float(x["L_short"]))):
    print(
        f"{r['L_short']}\t{r['L_long']}\t{float(r['epsilon_c_estimate']):.6f}\t"
        f"{r.get('fit_uncertainty','')}\t{r.get('recommended_uncertainty','')}\t{r['_src']}"
    )
PY

  echo "== clearing ${crit_root}/multi_L and writing FSS plots (Δμ=$dmu) =="
  rm -rf "${crit_root}/multi_L"
  python -u criticality/plot_eq_L_scaling.py \
    --lys "${LYS[@]}" \
    --coex-root "$coex_root" \
    --crit-root "$crit_root" \
    --crit-prefix "$CRIT_PREFIX" \
    --out-dir "${crit_root}/multi_L"
}

for dmu in "${DMUS[@]}"; do
  if [ "$CMD" = "compare" ]; then compare_one_dmu "$dmu"; else run_one_dmu "$dmu"; fi
done

if [ "$CMD" = "run" ]; then
  echo
  echo "Done. Compare with:  ./coex/run_homo_multi_L_criticality.sh compare --dmu ${DMUS[*]}"
fi
