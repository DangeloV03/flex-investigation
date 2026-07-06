#!/usr/bin/env python3
"""
Clear coex analysis fields in susceptibility_manage.csv so analyzer can re-run.

IMPORTANT: Stop the coex analyzer first or it will re-mark rows analyzed within
one poll cycle:
    tmux kill-session -t sus-coex

After reset, restart daemons (a long-running analyzer caches nothing in manage,
but restart clears pending_points wait state):
    ./coex/start_daemons.sh

Keeps coex simulation output.csv files; only resets manage.csv analysis state.

Usage:
    python coex/reset_susceptibility_coex_analysis.py --dry-run
    python coex/reset_susceptibility_coex_analysis.py
    python coex/reset_susceptibility_coex_analysis.py --bad-psi-only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzer import (
    MANAGE_FIELDS,
    build_curves,
    is_psi_minimum_acceptable,
    read_manage,
    write_manage,
)
from combo_paths import COMBO_KEY_FIELDS, discover_combo_results


def count_analyzed(rows: list[dict]) -> int:
    return sum(1 for row in rows if str(row.get("isAnalyzed", "")).strip())


def combo_key(row: dict) -> tuple[str, ...]:
    return tuple(str(row[f]) for f in COMBO_KEY_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset susceptibility coex analysis fields in manage.csv",
    )
    parser.add_argument("--manage", default="susceptibility_manage.csv")
    parser.add_argument("--results", default="susceptibility_results/coex")
    parser.add_argument(
        "--bad-psi-only",
        action="store_true",
        help="Only reset rows whose current min(psi) fails the PSI_COEX_MAX check",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_manage(args.manage)
    if not rows:
        raise SystemExit(f"No rows in {args.manage} (file missing or unreadable)")

    before = count_analyzed(rows)
    grouped = discover_combo_results(args.results) if args.bad_psi_only else {}

    n_reset = 0
    for row in rows:
        if args.bad_psi_only:
            data = grouped.get(combo_key(row))
            if data is None:
                continue
            _, _, _, psi_vals, _ = build_curves(data["points"])
            if is_psi_minimum_acceptable(psi_vals):
                continue
        elif not str(row.get("isAnalyzed", "")).strip():
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
        print(f"\nWould reset {n_reset} row(s) (currently analyzed: {before})")
        return

    if n_reset == 0:
        print(
            f"No rows matched reset criteria (analyzed={before}). "
            f"If analyzed>0 but you expected resets, stop tmux first:\n"
            f"  tmux kill-session -t sus-coex",
        )
        return

    write_manage(args.manage, rows)
    after_rows = read_manage(args.manage)
    after = count_analyzed(after_rows)
    expected_after = before - n_reset
    print(f"Reset {n_reset} row(s) in {args.manage} (isAnalyzed: {before} -> {after})")
    if after != expected_after:
        print(
            "WARNING: isAnalyzed count unexpected after reset. "
            "An analyzer may still be running — check:\n"
            "  pgrep -af 'analyzer.py.*susceptibility'\n"
            "  tmux kill-session -t sus-analyzer   # or sus-analysis, sus-coex",
        )
    elif args.bad_psi_only and after > 0:
        print(f"Left {after} row(s) analyzed (min(psi) already OK).")


if __name__ == "__main__":
    main()
