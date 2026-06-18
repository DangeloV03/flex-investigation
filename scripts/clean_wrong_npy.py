#!/usr/bin/env python3
"""
Delete final_lattice_*.npy (and leftover _scratch_* dirs) for wrongly analyzed combos.

Keeps output.csv so combos can be re-analyzed after resetting manage.csv.

Default target: rows with isAnalyzed set and mu_coex_SIM = NaN (premature or unstable).

Usage:
    python scripts/clean_wrong_npy.py --dry-run
    python scripts/clean_wrong_npy.py
    python scripts/clean_wrong_npy.py --mode premature   # NaN + RequestForAdditionalData >= 5
    python scripts/clean_wrong_npy.py --reset-manage     # also clear isAnalyzed for those rows
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combo_paths import COMBO_KEY_FIELDS, iter_output_csvs

MANAGE_FIELDS = COMBO_KEY_FIELDS + [
    "mu_coex_FLEX",
    "isSubmitted",
    "isRan",
    "isAnalyzed",
    "mu_coex_SIM",
    "mu_coex_SIM_error",
    "RequestForAdditionalData",
    "combo_path",
]


def combo_key_from_row(row: dict) -> tuple[str, ...]:
    return tuple(str(row[f]) for f in COMBO_KEY_FIELDS)


def read_manage(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_manage(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANAGE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANAGE_FIELDS})


def is_target_row(row: dict, mode: str) -> bool:
    if not str(row.get("isAnalyzed", "")).strip():
        return False
    sim = str(row.get("mu_coex_SIM", "")).strip().lower()
    if sim != "nan":
        return False
    if mode == "nan":
        return True
    if mode == "premature":
        try:
            req = int(row.get("RequestForAdditionalData") or 0)
        except ValueError:
            req = 0
        return req >= 5
    raise ValueError(f"unknown mode: {mode}")


def find_result_dirs(results_dir: Path, combo_key: tuple[str, ...]) -> list[Path]:
    combo = dict(zip(COMBO_KEY_FIELDS, combo_key))
    dirs: list[Path] = []
    for csv_path in iter_output_csvs(str(results_dir)):
        try:
            with open(csv_path, newline="") as f:
                row = next(csv.DictReader(f), None)
        except OSError:
            continue
        if row is None:
            continue
        if all(str(row[f]) == str(combo[f]) for f in COMBO_KEY_FIELDS):
            dirs.append(csv_path.parent)
    return dirs


def clean_dir(run_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Remove lattice npy and scratch dirs. Returns (n_npy, n_scratch)."""
    n_npy = 0
    n_scratch = 0
    for npy in run_dir.glob("final_lattice_*.npy"):
        if dry_run:
            print(f"  [dry-run] would delete {npy}")
        else:
            npy.unlink(missing_ok=True)
            print(f"  deleted {npy}")
        n_npy += 1
    for scratch in run_dir.glob("_scratch_*"):
        if not scratch.is_dir():
            continue
        if dry_run:
            print(f"  [dry-run] would delete {scratch}/")
        else:
            shutil.rmtree(scratch, ignore_errors=True)
            print(f"  deleted {scratch}/")
        n_scratch += 1
    return n_npy, n_scratch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete final_lattice_*.npy for wrongly/unstably analyzed combos."
    )
    parser.add_argument("--manage", default="manage.csv")
    parser.add_argument("--results", default="results")
    parser.add_argument(
        "--mode",
        choices=("nan", "premature"),
        default="nan",
        help="nan: all analyzed NaN rows; premature: NaN with RequestForAdditionalData >= 5",
    )
    parser.add_argument(
        "--reset-manage",
        action="store_true",
        help="Clear isAnalyzed / mu_coex_SIM / mu_coex_SIM_error / RequestForAdditionalData "
        "for targeted rows so analyzer can re-run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without deleting files or editing manage.csv",
    )
    args = parser.parse_args()

    rows = read_manage(args.manage)
    if not rows:
        print(f"No rows in {args.manage}", file=sys.stderr)
        return 1

    results_dir = Path(args.results)
    if not results_dir.is_dir():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    targets = [row for row in rows if is_target_row(row, args.mode)]
    if not targets:
        print(f"No manage.csv rows match mode={args.mode!r}")
        return 0

    print(f"Targeting {len(targets)} combo(s) (mode={args.mode!r}, dry_run={args.dry_run})")

    total_npy = 0
    total_scratch = 0
    total_dirs = 0

    for row in targets:
        key = combo_key_from_row(row)
        eps, dmu = row["epsilon"], row["delta_mu"]
        print(f"\ncombo eps={eps} dmu={dmu} (RequestForAdditionalData={row.get('RequestForAdditionalData', '')})")

        run_dirs = find_result_dirs(results_dir, key)
        if not run_dirs:
            print("  no result directories found")
            continue

        for run_dir in run_dirs:
            print(f"  {run_dir}")
            n_npy, n_scratch = clean_dir(run_dir, args.dry_run)
            total_npy += n_npy
            total_scratch += n_scratch
            total_dirs += 1

        if args.reset_manage:
            row["isAnalyzed"] = ""
            row["mu_coex_SIM"] = ""
            row["mu_coex_SIM_error"] = ""
            row["RequestForAdditionalData"] = "0"

    if args.reset_manage and not args.dry_run:
        write_manage(args.manage, rows)
        print(f"\nReset analysis fields in {args.manage} for {len(targets)} combo(s)")
    elif args.reset_manage and args.dry_run:
        print(f"\n[dry-run] would reset analysis fields in {args.manage} for {len(targets)} combo(s)")

    print(
        f"\nSummary: {len(targets)} combo(s), {total_dirs} run dir(s), "
        f"{total_npy} npy file(s), {total_scratch} scratch dir(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
