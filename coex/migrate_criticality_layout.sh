#!/usr/bin/env bash
# One-time migrate of criticality outputs into the multi-L testing layout.
#
# Equilibrium:
#   criticality/eq_ly<N>        -> criticality/equilibrium_multi_L_testing/ly<N>
#   criticality/eq_multi_L      -> criticality/equilibrium_multi_L_testing/multi_L
#   criticality/eq_validation   -> criticality/equilibrium_multi_L_testing/validation
#
# Scheme-3 (prep; safe if dirs do not exist yet):
#   criticality/s3_dmu1_ly<N>   -> criticality/scheme3_multi_L_testing/ly<N>
#   criticality/s3_dmu1_multi_L -> criticality/scheme3_multi_L_testing/multi_L
#
# Usage (repo root):
#   ./coex/migrate_criticality_layout.sh
#   ./coex/migrate_criticality_layout.sh --dry-run

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

DRY=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY=1
fi

move_one() {
  local src="$1" dst="$2"
  if [ ! -e "$src" ]; then
    echo "[skip] missing $src"
    return 0
  fi
  if [ -e "$dst" ]; then
    echo "[skip] destination exists: $dst  (left $src in place)"
    return 0
  fi
  echo "mv $src -> $dst"
  if [ "$DRY" -eq 0 ]; then
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
  fi
}

echo "== equilibrium multi-L =="
mkdir -p criticality/equilibrium_multi_L_testing
for ly in 16 20 40; do
  move_one "criticality/eq_ly${ly}" "criticality/equilibrium_multi_L_testing/ly${ly}"
done
move_one "criticality/eq_multi_L" "criticality/equilibrium_multi_L_testing/multi_L"
move_one "criticality/eq_validation" "criticality/equilibrium_multi_L_testing/validation"

echo
echo "== scheme3 multi-L (prep) =="
mkdir -p criticality/scheme3_multi_L_testing
for ly in 16 20 40; do
  move_one "criticality/s3_dmu1_ly${ly}" "criticality/scheme3_multi_L_testing/ly${ly}"
done
move_one "criticality/s3_dmu1_multi_L" "criticality/scheme3_multi_L_testing/multi_L"

echo
if [ "$DRY" -eq 1 ]; then
  echo "Dry run only — no files moved."
else
  echo "Done. New layout:"
  ls -la criticality/equilibrium_multi_L_testing 2>/dev/null || true
  ls -la criticality/scheme3_multi_L_testing 2>/dev/null || true
fi
