#!/usr/bin/env python3
"""
Compare manage.csv, results/, and combo folder artifacts.

Usage:
    python coex/audit_campaign.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combo_paths import (
    PHI_PSI_CSV,
    PHI_PSI_PNG,
    RESULTS_DIR,
    combo_dir_name,
    discover_combo_results,
)
from generate_samples import MANAGE_CSV, N_MU_POINTS, read_manage


def main():
    parser = argparse.ArgumentParser(description="Audit manage.csv vs combo folders")
    parser.add_argument("--manage", default=MANAGE_CSV)
    parser.add_argument("--results", default=RESULTS_DIR)
    args = parser.parse_args()

    rows = read_manage(args.manage)
    n_submitted = sum(1 for r in rows if str(r.get("isSubmitted", "")).strip())
    n_ran = sum(1 for r in rows if str(r.get("isRan", "")).strip())
    n_analyzed = sum(1 for r in rows if str(r.get("isAnalyzed", "")).strip())
    n_with_path = sum(1 for r in rows if str(r.get("combo_path", "")).strip())

    grouped = discover_combo_results(args.results)
    n_combos_with_results = len(grouped)
    n_mu_points = sum(len(d["points"]) for d in grouped.values())
    n_initial_ready = sum(
        1 for d in grouped.values() if len(d["points"]) >= N_MU_POINTS
    )

    n_plots = 0
    n_csv = 0
    results_root = args.results
    if os.path.isdir(results_root):
        for name in os.listdir(results_root):
            combo_path = os.path.join(results_root, name)
            if not os.path.isdir(combo_path):
                continue
            if os.path.isfile(os.path.join(combo_path, PHI_PSI_PNG)):
                n_plots += 1
            if os.path.isfile(os.path.join(combo_path, PHI_PSI_CSV)):
                n_csv += 1

    print(f"manage.csv rows:           {len(rows)}")
    print(f"  isSubmitted:             {n_submitted}")
    print(f"  isRan:                   {n_ran}")
    print(f"  isAnalyzed:              {n_analyzed}")
    print(f"  combo_path set:          {n_with_path}")
    print()
    print(f"results/ combos w/ data:   {n_combos_with_results}")
    print(f"  total mu output.csv:     {n_mu_points}")
    print(f"  with >= {N_MU_POINTS} mu points:      {n_initial_ready}")
    print()
    print(f"combo folders w/ plot:     {n_plots}")
    print(f"combo folders w/ csv:      {n_csv}")
    print()

    missing_plot = []
    for combo_key, data in sorted(grouped.items()):
        job = data["job"]
        tag = combo_dir_name(job)
        plot_path = os.path.join(args.results, tag, PHI_PSI_PNG)
        if not os.path.isfile(plot_path):
            missing_plot.append(tag)

    if missing_plot:
        print(f"Combos with results but no {PHI_PSI_PNG} ({len(missing_plot)}):")
        for tag in missing_plot[:20]:
            print(f"  {tag}")
        if len(missing_plot) > 20:
            print(f"  ... and {len(missing_plot) - 20} more")
        print("\n  Fix: python coex/replot_from_results.py")

    missing_path = [
        row for row in rows
        if not str(row.get("combo_path", "")).strip()
    ]
    if missing_path:
        print(f"\n{len(missing_path)} manage row(s) missing combo_path")
        print("  Fix: python coex/migrate_combo_layout.py")


if __name__ == "__main__":
    main()
