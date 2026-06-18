#!/usr/bin/env python3
"""
Regenerate phi/psi plots from existing results/ data.

The live analyzer skips combos already marked isAnalyzed in manage.csv, so
plots are not recreated after plots/ is deleted. This script ignores that
gate and plots every combo that has output.csv data.

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

from analyzer import (  # noqa: E402
    COMBO_KEY_FIELDS,
    PLOTS_DIR,
    RESULTS_DIR,
    build_curves,
    combo_dir_tag,
    discover_combo_results,
    find_manage_row,
    plot_combo,
    read_manage,
)


def main():
    parser = argparse.ArgumentParser(description="Replot phi/psi from results/")
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--plots", default=PLOTS_DIR)
    parser.add_argument("--manage", default="manage.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    grouped = discover_combo_results(args.results)
    rows = read_manage(args.manage)
    n = 0

    for combo_key, data in sorted(grouped.items()):
        job = data["job"]
        tag = combo_dir_tag(job)
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

        out = os.path.join(args.plots, f"{tag}_phi_psi.png")
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
                plots_dir=args.plots,
            )
        n += 1

    print(f"\n{'Would write' if args.dry_run else 'Wrote'} {n} plot(s) to {args.plots}/")


if __name__ == "__main__":
    main()
