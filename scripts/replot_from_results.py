#!/usr/bin/env python3
"""
Regenerate phi/psi plots and CSV from existing results/ data.

Usage:
    python scripts/replot_from_results.py --dry-run
    python scripts/replot_from_results.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

from analyzer import (
    COMBO_KEY_FIELDS,
    build_curves,
    discover_combo_results,
    find_manage_row,
    plot_combo,
    read_manage,
)
from combo_paths import RESULTS_DIR, combo_dir_name


def main():
    parser = argparse.ArgumentParser(description="Replot phi/psi into combo folders")
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--manage", default="manage.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    grouped = discover_combo_results(args.results)
    rows = read_manage(args.manage)
    n = 0

    for combo_key, data in sorted(grouped.items()):
        job = data["job"]
        tag = combo_dir_name(job)
        mu_vals, phi_vals, phi_errs, psi_vals, psi_errs = build_curves(data["points"])

        mu_coex_sim = None
        combo = {f: job[f] for f in COMBO_KEY_FIELDS}
        idx = find_manage_row(rows, combo)
        if idx is not None:
            raw = rows[idx].get("mu_coex_SIM", "")
            if str(raw).strip() and str(raw).strip().lower() != "nan":
                try:
                    mu_coex_sim = float(raw)
                except ValueError:
                    pass

        out = os.path.join(args.results, tag, "phi_psi.png")
        print(f"{'would plot' if args.dry_run else 'plotting'}: {out}  ({len(mu_vals)} mu points)")
        if not args.dry_run:
            plot_combo(
                combo_key,
                mu_vals,
                phi_vals,
                phi_errs,
                psi_vals,
                psi_errs,
                mu_coex_sim=mu_coex_sim,
                results_dir=args.results,
            )
        n += 1

    print(f"\n{'Would write' if args.dry_run else 'Wrote'} artifacts for {n} combo(s)")


if __name__ == "__main__":
    main()
