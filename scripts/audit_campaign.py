#!/usr/bin/env python3
"""
Compare manage.csv, results/, and plots/ counts.

Usage (on Della, from project root):
    python scripts/audit_campaign.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import COMBO_KEY_FIELDS, PLOTS_DIR, RESULTS_DIR, combo_dir_tag, discover_combo_results
from generate_samples import MANAGE_CSV, N_MU_POINTS, read_manage


def main():
    parser = argparse.ArgumentParser(description="Audit manage.csv vs results vs plots")
    parser.add_argument("--manage", default=MANAGE_CSV)
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--plots", default=PLOTS_DIR)
    args = parser.parse_args()

    rows = read_manage(args.manage)
    n_submitted = sum(1 for r in rows if str(r.get("isSubmitted", "")).strip())
    n_ran = sum(1 for r in rows if str(r.get("isRan", "")).strip())
    n_analyzed = sum(1 for r in rows if str(r.get("isAnalyzed", "")).strip())
    n_incomplete = n_submitted - n_analyzed

    grouped = discover_combo_results(args.results)
    n_combos_with_results = len(grouped)
    n_mu_points = sum(len(d["points"]) for d in grouped.values())
    n_initial_ready = sum(
        1 for d in grouped.values() if len(d["points"]) >= N_MU_POINTS
    )

    plot_dir = args.plots
    n_plots = 0
    if os.path.isdir(plot_dir):
        n_plots = len([f for f in os.listdir(plot_dir) if f.endswith("_phi_psi.png")])

    print(f"manage.csv rows:           {len(rows)}")
    print(f"  isSubmitted:             {n_submitted}")
    print(f"  isRan:                   {n_ran}")
    print(f"  isAnalyzed:              {n_analyzed}")
    print(f"  not yet analyzed:        {n_incomplete}")
    print()
    print(f"results/ combos w/ data:   {n_combos_with_results}")
    print(f"  total mu output.csv:     {n_mu_points}")
    print(f"  with >= {N_MU_POINTS} mu points:      {n_initial_ready}")
    print()
    print(f"plots/ *_phi_psi.png:      {n_plots}")
    print()

    if n_analyzed > n_plots:
        print(
            f"GAP: {n_analyzed - n_plots} combos marked isAnalyzed but no plot file.\n"
            "  Common causes:\n"
            "    - plots/ was deleted or never synced to this machine\n"
            "    - analyzer skips already-analyzed rows (no auto-replot)\n"
            "    - manage.csv outlived results/ (results lost, isAnalyzed kept)\n"
            "  Fix missing plots (where results still exist):\n"
            "    python scripts/replot_from_results.py"
        )

    if n_analyzed > n_combos_with_results:
        print(
            f"GAP: {n_analyzed - n_combos_with_results} isAnalyzed rows have no results/ data.\n"
            "  Those combos need re-simulation, not just replotting:\n"
            "    python scripts/requeue_incomplete.py --reset-ran"
        )

    # List combos with results but no plot
    missing_plot = []
    for combo_key, data in sorted(grouped.items()):
        job = data["job"]
        tag = combo_dir_tag(job)
        plot_path = os.path.join(plot_dir, f"{tag}_phi_psi.png")
        if not os.path.isfile(plot_path):
            missing_plot.append(tag)

    if missing_plot:
        print(f"\nCombos with results but no plot ({len(missing_plot)}):")
        for tag in missing_plot[:20]:
            print(f"  {tag}")
        if len(missing_plot) > 20:
            print(f"  ... and {len(missing_plot) - 20} more")


if __name__ == "__main__":
    main()
