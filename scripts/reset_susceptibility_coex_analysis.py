#!/usr/bin/env python3
"""
Clear coex analysis fields in susceptibility_manage.csv so analyzer can re-run.

Keeps coex simulation output.csv files; only resets manage.csv analysis state.

Usage:
    python scripts/reset_susceptibility_coex_analysis.py --dry-run
    python scripts/reset_susceptibility_coex_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset susceptibility coex analysis fields in manage.csv",
    )
    parser.add_argument("--manage", default="susceptibility_manage.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.manage, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    if not fields:
        raise SystemExit(f"No rows in {args.manage}")

    n_reset = 0
    for row in rows:
        if not str(row.get("isAnalyzed", "")).strip():
            continue
        n_reset += 1
        if args.dry_run:
            print(f"would reset eps={row.get('epsilon')}")
            continue
        row["isAnalyzed"] = ""
        row["mu_coex_SIM"] = ""
        row["mu_coex_SIM_error"] = ""
        row["RequestForAdditionalData"] = "0"

    if args.dry_run:
        print(f"\nWould reset {n_reset} analyzed row(s)")
        return

    with open(args.manage, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Reset {n_reset} analyzed row(s) in {args.manage}")
