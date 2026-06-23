#!/usr/bin/env python3
"""Print why susceptibility coex combos are or aren't ready for analyzer."""

from __future__ import annotations

import argparse

from analyzer import (
    N_INITIAL_MU_POINTS,
    build_curves,
    find_manage_row,
    has_phi_sign_change,
    read_manage,
)
from combo_paths import COMBO_KEY_FIELDS, combo_dir_name, discover_combo_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose susceptibility analyzer readiness")
    parser.add_argument("--manage", default="susceptibility_manage.csv")
    parser.add_argument("--results", default="susceptibility_results/coex")
    args = parser.parse_args()

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

        mu_vals, phi_vals, _, _, _ = build_curves(data["points"])
        sign_change = has_phi_sign_change(phi_vals) if n_points else False

        print(f"{tag}")
        print(f"  eps={job['epsilon']}  n_mu={n_points}/{N_INITIAL_MU_POINTS}  "
              f"manage_row={idx}  n_requests={n_req}  analyzed={analyzed}  "
              f"phi_sign_change={sign_change}")
        if idx is None and rows:
            sample = rows[0]
            print(f"  manage sample: eps={sample.get('epsilon')} Lx={sample.get('Lx')} "
                  f"Ly={sample.get('Ly')} k={sample.get('k')}")
            print(f"  job values:    eps={combo.get('epsilon')} Lx={combo.get('Lx')} "
                  f"Ly={combo.get('Ly')} k={combo.get('k')}")


if __name__ == "__main__":
    main()
