#!/usr/bin/env python3
"""
Move flat mu* run dirs into mu_sweeps/ under each combo folder.

Use this after the combo-folder migration (dc2a2eb) when mu runs still sit
directly under each combo dir instead of combo/mu_sweeps/mu*/.

Before:
    results/320x32_homo_.../mu3174482/output.csv
    results/320x32_homo_.../phi_psi.png

After:
    results/320x32_homo_.../phi_psi.png
    results/320x32_homo_.../mu_sweeps/mu3174482/output.csv

Usage (on Della, from project root):
    ./scripts/stop_daemons.sh
    python scripts/nest_mu_sweeps.py --dry-run
    python scripts/nest_mu_sweeps.py
    ./scripts/start_daemons.sh
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combo_paths import RESULTS_DIR, nest_flat_mu_dirs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nest flat mu* dirs under mu_sweeps/ in each combo folder",
    )
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n = nest_flat_mu_dirs(args.results, dry_run=args.dry_run)
    label = "Would nest" if args.dry_run else "Nested"
    print(f"\n{label} {n} mu run dir(s)")
    if args.dry_run and n:
        print("Re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
