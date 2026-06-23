#!/usr/bin/env python3
"""Print why susceptibility coex combos are or aren't ready for analyzer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzer import (
    N_INITIAL_MU_POINTS,
    PSI_COEX_MAX,
    build_curves,
    find_manage_row,
    has_phi_sign_change,
    is_psi_minimum_acceptable,
    min_psi_value,
    read_manage,
)
from combo_paths import COMBO_KEY_FIELDS, combo_dir_name, discover_combo_results


def print_queue_summary(manifest_path: str = "susceptibility_coex_queue.json") -> None:
    import json
    import os

    if not os.path.isfile(manifest_path):
        print(f"Queue manifest not found: {manifest_path}\n")
        return
    with open(manifest_path) as f:
        m = json.load(f)
    pending = m.get("pending", [])
    in_flight = m.get("in_flight", {})
    print(
        f"Queue {manifest_path}: pending={len(pending)}  "
        f"in_flight={len(in_flight)}\n"
    )


def count_stray_coex_outputs(results_dir: str = "results") -> int:
    """Count mu-level output.csv under default results/ (wrong dir for susceptibility)."""
    from combo_paths import iter_output_csvs

    n = 0
    for csv_path in iter_output_csvs(results_dir):
        try:
            import pandas as pd

            df = pd.read_csv(csv_path, nrows=1)
            if df.empty:
                continue
            if float(df["delta_f"].iloc[0]) == -20.0 and float(df["k"].iloc[0]) == 0.0:
                if int(df["Ly"].iloc[0]) == 16 and int(df["Lx"].iloc[0]) == 160:
                    n += 1
        except Exception:
            continue
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose susceptibility analyzer readiness")
    parser.add_argument("--manage", default="susceptibility_manage.csv")
    parser.add_argument("--results", default="susceptibility_results/coex")
    args = parser.parse_args()

    print_queue_summary("susceptibility_coex_queue.json")

    stray = count_stray_coex_outputs("results")
    if stray:
        print(
            f"WARNING: found {stray} susceptibility-like output.csv under results/ "
            f"(refinement jobs likely wrote to the wrong directory).\n"
        )

    grouped = discover_combo_results(args.results)
    rows = read_manage(args.manage)

    print(f"Discovered {len(grouped)} combo(s) under {args.results}")
    print(f"manage.csv rows: {len(rows)}\n")

    if not grouped:
        print("ERROR: discover_combo_results found nothing. Check --results path.")
        return

    for combo_key, data in sorted(grouped.items(), key=lambda x: float(x[1]["job"]["epsilon"])):
        job = data["job"]
        combo = {f: job[f] for f in COMBO_KEY_FIELDS}
        tag = combo_dir_name(job)
        n_points = len(data["points"])
        idx = find_manage_row(rows, combo)
        n_req = int(rows[idx].get("RequestForAdditionalData", 0)) if idx is not None else -1
        analyzed = bool(idx is not None and str(rows[idx].get("isAnalyzed", "")).strip())

        mu_vals, phi_vals, _, psi_vals, _ = build_curves(data["points"])
        sign_change = has_phi_sign_change(phi_vals) if n_points else False
        psi_min = min_psi_value(psi_vals) if n_points else float("nan")
        psi_ok = is_psi_minimum_acceptable(psi_vals) if n_points else False
        stalled = (
            n_req > 0
            and not analyzed
            and n_points <= N_INITIAL_MU_POINTS
        )

        print(f"{tag}")
        print(f"  eps={job['epsilon']}  n_mu={n_points}/{N_INITIAL_MU_POINTS}  "
              f"manage_row={idx}  n_requests={n_req}  analyzed={analyzed}  "
              f"phi_sign_change={sign_change}  min_psi={psi_min:.4f}  "
              f"psi_ok(<={PSI_COEX_MAX})={psi_ok}")
        if stalled:
            print(
                "  STALL: n_requests>0 but mu count not growing — "
                "pull latest analyzer.py fix and reset RequestForAdditionalData",
            )
        if idx is None and rows:
            sample = rows[0]
            print(f"  manage sample: eps={sample.get('epsilon')} Lx={sample.get('Lx')} "
                  f"Ly={sample.get('Ly')} k={sample.get('k')}")
            print(f"  job values:    eps={combo.get('epsilon')} Lx={combo.get('Lx')} "
                  f"Ly={combo.get('Ly')} k={combo.get('k')}")


if __name__ == "__main__":
    main()
