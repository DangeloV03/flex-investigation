#!/usr/bin/env bash
# Generate fine-ε jobs for L=256 in the critical region ε ∈ [-1.8, -1.7].
# Spacing: 0.0025 (half of the main campaign's 0.005).
# Replica settings: same as main campaign (4×24 batches = 96 total).
# Only adds intermediate ε values — existing ε points skip automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
python generate_susceptibility_exact.py \
    --eps-min -1.8 \
    --eps-max -1.7 \
    --eps-step 0.0025 \
    --L 256
echo ""
echo "Jobs queued. Start (or let the running dispatcher pick them up automatically):"
echo "  python run_susceptibility_all.py --phase exact"
