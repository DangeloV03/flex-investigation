#!/usr/bin/env bash
# Bimodality criticality for Scheme-3 (negative_drive) multi-L coex.
#
# Expects scheme3/dmu<X>_multiL/ly<N>/results from
#   ./coex/run_scheme3_multi_L_campaign.sh
# Writes under criticality/scheme3_multi_L_testing/dmu<X>/:
#   ly<N>/{bc_vs_beta_epsilon.csv,criticality.csv,*.png}
#   multi_L/   (FSS / mentor compare plots)
#
# Usage (after coex is analyzed):
#   ./coex/run_scheme3_multi_L_criticality.sh --dmu 2.0 3.0 4.0 4.5
#   ./coex/run_scheme3_multi_L_criticality.sh compare --dmu 2.0 3.0 4.0 4.5
#   ./coex/run_scheme3_multi_L_criticality.sh --dmu 2.0 --lys 16 20 40
#
# Note: older Δμ=1.0 criticality may still sit at
#   criticality/scheme3_multi_L_testing/ly*/  (flat). Re-run with --dmu 1.0 to
#   place it under .../dmu1p0/ like the other Δμ.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/coex:$PROJECT_DIR/susceptibility:$PROJECT_DIR:$PROJECT_DIR/criticality"

DEFAULT_LYS=(16 20 40)
DEFAULT_DMUS=(2.0 3.0 4.0 4.5)
SCHEME=negative_drive
DELTA_F=0.0
K=0.1
CRIT_PREFIX=ly

CMD="${1:-run}"
shift || true

DMUS=()
LYS=()
MODE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dmu) MODE=dmu; shift ;;
    --lys) MODE=lys; shift ;;
    *)
      if [ "$MODE" = "dmu" ]; then
        DMUS+=("$1")
      elif [ "$MODE" = "lys" ]; then
        LYS+=("$1")
      else
        LYS+=("$1")
      fi
      shift
      ;;
  esac
done

if [ "${#DMUS[@]}" -eq 0 ]; then
  if [ "$MODE" = "" ] && [ "${#LYS[@]}" -gt 0 ]; then
    DMUS=(1.0)
  else
    DMUS=("${DEFAULT_DMUS[@]}")
  fi
fi
if [ "${#LYS[@]}" -eq 0 ]; then
  LYS=("${DEFAULT_LYS[@]}")
fi

dmu_tag() { echo "dmu$(echo "$1" | sed 's/\./p/')"; }
coex_root_for() { echo "scheme3/$(dmu_tag "$1")_multiL"; }
crit_root_for() { echo "criticality/scheme3_multi_L_testing/$(dmu_tag "$1")"; }

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
    raise SystemExit(
        f"No {root}/{prefix}*/criticality.csv for Δμ={dmu} — run without 'compare' first."
    )
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

if [ "$CMD" = "compare" ]; then
  for dmu in "${DMUS[@]}"; do
    compare_one_dmu "$dmu"
  done
  exit 0
fi

if [ "$CMD" != "run" ]; then
  # allow: ./script --dmu 2.0   (CMD was eaten as --dmu if user forgot 'run')
  if [ "$CMD" = "--dmu" ] || [ "$CMD" = "--lys" ]; then
    echo "usage: $0 [run|compare] [--dmu ...] [--lys ...]"
    exit 1
  fi
fi

for dmu in "${DMUS[@]}"; do
  run_one_dmu "$dmu"
done

echo
echo "Done. Compare with:"
echo "  ./coex/run_scheme3_multi_L_criticality.sh compare --dmu ${DMUS[*]}"
