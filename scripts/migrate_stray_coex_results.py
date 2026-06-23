#!/usr/bin/env python3
"""
Move susceptibility coex mu outputs that landed under results/ into
susceptibility_results/coex/ (wrong results_base on old refinement jobs).

Usage:
    python scripts/migrate_stray_coex_results.py --dry-run
    python scripts/migrate_stray_coex_results.py
    python scripts/migrate_stray_coex_results.py --purge-done-json
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from combo_paths import COMBO_KEY_FIELDS, combo_dir, iter_output_csvs, mu_dir
from susceptibility_paths import COEX_RESULTS_DIR, COEX_SAMPLES_DIR, ISING_DELTA_F, ISING_K


def is_susceptibility_coex_row(row: dict) -> bool:
    try:
        return (
            float(row["delta_f"]) == float(ISING_DELTA_F)
            and float(row["k"]) == float(ISING_K)
            and int(row["Ly"]) == 16
            and int(row["Lx"]) == 160
        )
    except (KeyError, TypeError, ValueError):
        return False


def params_from_csv(csv_path: Path) -> dict | None:
    import pandas as pd

    try:
        df = pd.read_csv(csv_path, nrows=1)
        if df.empty:
            return None
        row = df.iloc[0].to_dict()
        if not is_susceptibility_coex_row(row):
            return None
        params = {f: row[f] for f in COMBO_KEY_FIELDS}
        params["mu"] = float(row["mu"])
        return params
    except Exception:
        return None


def merge_output_csv(src: Path, dst: Path) -> int:
    """Append rows from src into dst, skipping duplicate ids."""
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        with open(dst, newline="") as f:
            return sum(1 for _ in csv.DictReader(f))

    with open(dst, newline="") as f:
        existing_ids = {int(r["id"]) for r in csv.DictReader(f)}
    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        new_rows = [r for r in reader if int(r["id"]) not in existing_ids]
    if not new_rows:
        return 0
    with open(dst, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(new_rows)
    return len(new_rows)


def migrate_one(csv_path: Path, dest_base: str, *, dry_run: bool) -> tuple[str, int]:
    params = params_from_csv(csv_path)
    if params is None:
        return "skip", 0

    dest_dir = Path(mu_dir(params, base=dest_base))
    dest_csv = dest_dir / "output.csv"
    src_dir = csv_path.parent

    if dry_run:
        action = "merge" if dest_csv.is_file() else "move"
        return action, 1

    if dest_csv.is_file():
        n = merge_output_csv(csv_path, dest_csv)
        for npy in src_dir.glob("final_lattice_*.npy"):
            target = dest_dir / npy.name
            if not target.exists():
                shutil.copy2(npy, target)
        return "merge", n

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_dir), str(dest_dir))
    return "move", 1


def purge_done_refinement_json(samples_dir: Path, *, dry_run: bool) -> int:
    done_dir = samples_dir / "done"
    if not done_dir.is_dir():
        return 0
    n = 0
    for path in done_dir.glob("homo_Ly16_mu*.json"):
        n += 1
        if not dry_run:
            path.unlink()
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate stray susceptibility coex outputs from results/ to coex tree",
    )
    parser.add_argument("--src", default="results", help="Wrong results base to scan")
    parser.add_argument("--dest", default=COEX_RESULTS_DIR)
    parser.add_argument("--samples", default=COEX_SAMPLES_DIR)
    parser.add_argument(
        "--purge-done-json",
        action="store_true",
        help="Remove homo_Ly16_mu*.json from samples/coex/done/ (stale without results_base)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    moved = merged = skipped = 0
    rows_added = 0

    for csv_path in iter_output_csvs(args.src):
        action, n = migrate_one(Path(csv_path), args.dest, dry_run=args.dry_run)
        if action == "move":
            moved += n
        elif action == "merge":
            merged += 1
            rows_added += n
        else:
            skipped += 1

    prefix = "Would" if args.dry_run else "Done"
    print(
        f"{prefix}: move {moved} mu dir(s), merge into {merged} existing dir(s) "
        f"({rows_added} csv rows), skip {skipped}",
    )
    print(f"Destination base: {args.dest}")

    if args.purge_done_json:
        n_purge = purge_done_refinement_json(Path(args.samples), dry_run=args.dry_run)
        print(f"{prefix} purge {n_purge} stale JSON(s) from {args.samples}/done/")


if __name__ == "__main__":
    main()
