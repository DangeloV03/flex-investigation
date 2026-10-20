#!/usr/bin/env bash
# Example criticality analysis on the outputs of the coex and susc examples.
#
#   bash run/examples/criticality.sh coex   # Binder/bimodality analysis of the coex slabs
#   bash run/examples/criticality.sh susc   # P(m) distributions from the susc runs
#
# coex: needs run/examples/coex.sh to have finished (or at least some epsilon
#   done). For each epsilon it picks the mu with <phi> closest to 0 and computes
#   Sarle's bimodality coefficient BC of the column order parameter phi_col.
#   BC > 5/9 means P(phi_col) is bimodal (two coexisting phases); the epsilon
#   where BC drops through 5/9 estimates epsilon_c for this slab size.
#   Writes bc_vs_beta_epsilon.csv, criticality.csv, the BC_max plot and P(phi_col)
#   histograms. Five epsilon on one size is only a smoke test; a real estimate
#   uses a fine epsilon grid and several sizes (`run/criticality.py bc-fss`).
#
# susc: needs run/examples/susc.sh to have finished. Pools m over all replicas
#   per (epsilon, L): a double peak that stays open as L grows means first order,
#   a single peak that narrows means continuous.
#
# Output: <output root>/examples/plots/criticality. The output root is
# <repo>/output unless FLEX_OUTPUT_ROOT is set.

set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${FLEX_OUTPUT_ROOT:-$PWD/output}"
COEX_RESULTS=examples/COEX_RUNS_EXAMPLE/ly16/results    # relative to the output root
SUSC_RESULTS=examples/SUSC_RUNS_EXAMPLE
PLOTS=examples/plots/criticality

case "${1:-}" in
  coex)
    COMMON=(--base-dir "$COEX_RESULTS" --scheme homo --delta-f 0 --k 1 --Lx 160 --Ly 16)
    python run/criticality.py inspect "${COMMON[@]}" --ref-step 0.1
    python run/criticality.py phase-diagram "${COMMON[@]}" --out-dir "$PLOTS/coex"
    python run/criticality.py histograms "${COMMON[@]}" --delta-mu 1 \
        --epsilons -2.0 -1.8 -1.6 --out-dir "$PLOTS/coex"
    echo "Results in $OUT/$PLOTS/coex"
    ;;
  susc)
    python run/criticality.py distributions --results "$SUSC_RESULTS" \
        --epsilons -1.85 -1.75 -1.65 --outdir "$PLOTS/susc"
    echo "Results in $OUT/$PLOTS/susc"
    ;;
  *)
    sed -n '2,5p' "$0"
    exit 2
    ;;
esac
