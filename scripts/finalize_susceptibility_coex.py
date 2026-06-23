#!/usr/bin/env python3
"""
Force mu_coex_SIM = argmin(psi) for susceptibility coex combos with complete mu sweeps.

Use when the analyzer is stuck on 'refinement jobs already queued, waiting'
but you already have 10/10 initial mu points and phi sign change.

Usage:
    python scripts/finalize_susceptibility_coex.py
    python scripts/finalize_susceptibility_coex.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzer import (
    N_INITIAL_MU_POINTS,
    build_curves,
    combo_dir_name,
    finalize_combo,
    find_manage_row,
    has_phi_sign_change,
    interior_psi_minimum,
    read_manage,
)
from combo_paths import COMBO_KEY_FIELDS, discover_combo_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Force finalize susceptibility coex combos at argmin(psi)",
    )
    parser.add_argument("--manage", default="susceptibility_manage.csv")
    parser.add_argument("--results", default="susceptibility_results/coex")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    grouped = discover_combo_results(args.results)
    rows = read_manage(args.manage)
    n_done = 0

    for combo_key, data in sorted(grouped.items(), key=lambda x: float(x[1]["job"]["epsilon"])):
        job = data["job"]
        combo = {f: job[f] for f in COMBO_KEY_FIELDS}
        tag = combo_dir_name(job)
        n_points = len(data["points"])

        idx = find_manage_row(rows, combo)
        if idx is None:
            print(f"[skip] {tag}: no manage row")
            continue
        if str(rows[idx].get("isAnalyzed", "")).strip():
            print(f"[skip] {tag}: already analyzed")
            continue
        if n_points < N_INITIAL_MU_POINTS:
            print(f"[skip] {tag}: only {n_points}/{N_INITIAL_MU_POINTS} mu points")
            continue

        mu_vals, phi_vals, phi_errs, psi_vals, psi_errs = build_curves(data["points"])
        if not has_phi_sign_change(phi_vals):
            print(f"[skip] {tag}: no phi sign change")
            continue
        if not interior_psi_minimum(psi_vals):
            print(f"[skip] {tag}: min(psi) at edge of mu window")
            continue

        n_requests = int(rows[idx].get("RequestForAdditionalData", 0))
        if args.dry_run:
            min_idx = int(psi_vals.argmin())
            print(f"[dry-run] {tag}: would set mu_coex_SIM={mu_vals[min_idx]:.6f}")
            n_done += 1
            continue

        finalize_combo(
            combo_key,
            combo,
            tag,
            mu_vals,
            phi_vals,
            phi_errs,
            psi_vals,
            psi_errs,
            args.manage,
            args.results,
            n_requests,
            reason="forced argmin(psi) susceptibility coex",
        )
        rows = read_manage(args.manage)
        n_done += 1

    print(f"\nFinalized (or dry-run): {n_done} combo(s)")


if __name__ == "__main__":
    main()
