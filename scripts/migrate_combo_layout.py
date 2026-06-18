#!/usr/bin/env python3
"""
Migrate results/ and plots/ to the unified combo folder layout.

For campaigns already on the combo-folder layout, use nest_mu_sweeps.py instead.

Usage:
    python scripts/migrate_combo_layout.py --dry-run
    python scripts/migrate_combo_layout.py --replot
    python scripts/nest_mu_sweeps.py --dry-run   # mu_sweeps/ only (typical fix)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import read_manage, write_manage
from combo_paths import (
    COMBO_KEY_FIELDS,
    MU_SWEEPS_DIR,
    PHI_PSI_PNG,
    combo_dir,
    combo_dir_name,
    combo_key_from_dict,
    iter_output_csvs,
    legacy_plot_basenames,
    mu_dir_name,
    mu_sweeps_dir,
    nest_flat_mu_dirs,
    read_combo_from_output_csv,
)
from generate_samples import MANAGE_CSV, RESULTS_DIR


def find_legacy_plot(plots_dir: Path, params: dict) -> Path | None:
    if not plots_dir.is_dir():
        return None
    for name in legacy_plot_basenames(params):
        candidate = plots_dir / name
        if candidate.is_file():
            return candidate
    return None


def move_mu_dir(src: Path, dst: Path, *, dry_run: bool) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        if dry_run:
            print(f"  merge {src} -> {dst} (dst exists)")
            return
        for item in src.iterdir():
            target = dst / item.name
            if target.exists():
                continue
            shutil.move(str(item), str(target))
        if not any(src.iterdir()):
            src.rmdir()
        return
    if dry_run:
        print(f"  move {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def migrate_legacy_to_combo(results_dir: Path, plots_dir: Path, *, dry_run: bool) -> int:
    seen_mu: set[tuple[tuple[str, ...], str]] = set()
    moved = 0

    for csv_path in iter_output_csvs(str(results_dir)):
        params = read_combo_from_output_csv(csv_path)
        if params is None:
            continue
        combo_key = combo_key_from_dict(params)
        mu_name = mu_dir_name(params["mu"])
        key = (combo_key, mu_name)
        if key in seen_mu:
            continue
        seen_mu.add(key)

        src = csv_path.parent
        dst = Path(mu_sweeps_dir(params, str(results_dir))) / mu_name
        if src.resolve() == dst.resolve():
            continue

        tag = combo_dir_name(params)
        print(f"{tag}/{MU_SWEEPS_DIR}/{mu_name}: from {src}")
        move_mu_dir(src, dst, dry_run=dry_run)
        moved += 1

        dest_combo = dst.parent.parent
        if not dry_run:
            dest_combo.mkdir(parents=True, exist_ok=True)
            legacy_plot = find_legacy_plot(plots_dir, params)
            if legacy_plot and not (dest_combo / PHI_PSI_PNG).exists():
                shutil.copy2(legacy_plot, dest_combo / PHI_PSI_PNG)
                print(f"  copied plot {legacy_plot.name}")

    return moved


def migrate_manage_paths(manage_path: str, results_dir: str, *, dry_run: bool) -> int:
    rows = read_manage(manage_path)
    if not rows:
        return 0
    updated = 0
    for row in rows:
        combo = {f: row[f] for f in COMBO_KEY_FIELDS}
        path = combo_dir(combo, results_dir)
        if row.get("combo_path") != path:
            if dry_run:
                print(f"manage combo_path: eps={row['epsilon']} dmu={row['delta_mu']} -> {path}")
            row["combo_path"] = path
            updated += 1
    if updated and not dry_run:
        write_manage(manage_path, rows)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate to unified combo folder layout")
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--plots", default="plots", help="Legacy plots/ directory")
    parser.add_argument("--manage", default=MANAGE_CSV)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate phi_psi.png and phi_psi.csv from migrated output.csv data",
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    plots_dir = Path(args.plots)
    if not results_dir.is_dir():
        print(f"No results directory: {results_dir}", file=sys.stderr)
        return 1

    print("=== nesting flat mu dirs under mu_sweeps/ ===")
    n_nested = nest_flat_mu_dirs(str(results_dir), dry_run=args.dry_run)
    print(f"nested: {n_nested}")

    print("\n=== migrating legacy mu run directories ===")
    n_migrated = migrate_legacy_to_combo(results_dir, plots_dir, dry_run=args.dry_run)
    print(f"legacy mu dirs processed: {n_migrated}")

    print("\n=== backfilling manage.csv combo_path ===")
    n_manage = migrate_manage_paths(args.manage, args.results, dry_run=args.dry_run)
    print(f"manage rows updated: {n_manage}")

    if args.replot:
        import matplotlib
        matplotlib.use("Agg")
        from analyzer import build_curves, discover_combo_results, plot_combo

        print("\n=== regenerating phi/psi artifacts ===")
        grouped = discover_combo_results(args.results)
        for combo_key, data in sorted(grouped.items()):
            job = data["job"]
            tag = combo_dir_name(job)
            mu_vals, phi_vals, phi_errs, psi_vals, psi_errs = build_curves(data["points"])
            if args.dry_run:
                print(f"would replot {tag} ({len(mu_vals)} points)")
            else:
                plot_combo(
                    combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
                    results_dir=args.results,
                )

    if args.dry_run:
        print("\n(dry run — no files changed)")
    else:
        print("\nDone.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
