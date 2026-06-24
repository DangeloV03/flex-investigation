"""
analyzer_fitted_line.py

Post-hoc analysis: for each completed coex combo in susceptibility_manage.csv,
fit a weighted linear model to phi(mu) and find where it crosses zero.
Compares mu_coex_FITTED (linear fit zero-crossing) against mu_coex_SIM (argmin psi).

The linear fit uses only the mu points inside the sign-change bracket
(i.e., between the nearest positive and nearest negative phi values), weighted
by 1/phi_err^2 so noisier points contribute less.  When no sign change exists,
the row is skipped.

Usage:
    python analyzer_fitted_line.py
    python analyzer_fitted_line.py \\
        --manage susceptibility_manage.csv \\
        --results susceptibility_results/coex \\
        --out mu_coex_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

from analyzer import (
    build_curves,
    has_phi_sign_change,
    sign_change_bracket,
)
from combo_paths import COMBO_KEY_FIELDS, combo_key_from_dict, discover_combo_results
from generate_samples import read_manage
from susceptibility_paths import COEX_RESULTS_DIR, MANAGE_CSV


def fit_zero_crossing(
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
) -> float | None:
    """Weighted linear fit to phi(mu); return the zero crossing.

    Only points inside (and immediately outside) the sign-change bracket are
    used so the fit captures the local slope rather than distant non-linear tails.
    Returns None when fewer than 2 points are available or when the slope is flat.
    """
    bracket = sign_change_bracket(mu_vals, phi_vals)
    if bracket is None:
        return None

    mu_lo, mu_hi = bracket

    # Expand bracket by one neighbour on each side for better slope estimate
    sorted_mus = np.sort(mu_vals)
    lo_idx = int(np.searchsorted(sorted_mus, mu_lo))
    hi_idx = int(np.searchsorted(sorted_mus, mu_hi))
    expanded_lo = float(sorted_mus[max(0, lo_idx - 1)])
    expanded_hi = float(sorted_mus[min(len(sorted_mus) - 1, hi_idx + 1)])

    mask = (mu_vals >= expanded_lo - 1e-9) & (mu_vals <= expanded_hi + 1e-9)
    mu_fit = mu_vals[mask]
    phi_fit = phi_vals[mask]
    err_fit = phi_errs[mask]

    if len(mu_fit) < 2:
        return None

    # Weighted least squares: weight = 1/err^2 (fall back to uniform if err=0)
    w = np.where(err_fit > 0, 1.0 / (err_fit ** 2), 1.0)
    w = np.sqrt(w)  # polyfit expects sqrt(weight)

    try:
        coeffs = np.polyfit(mu_fit, phi_fit, deg=1, w=w)
    except (np.linalg.LinAlgError, ValueError):
        return None

    slope, intercept = coeffs
    if abs(slope) < 1e-12:
        return None

    return float(-intercept / slope)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit phi(mu) line and compare mu_coex_FITTED to mu_coex_SIM"
    )
    parser.add_argument("--manage", default=MANAGE_CSV, help="susceptibility_manage.csv path")
    parser.add_argument(
        "--results", default=COEX_RESULTS_DIR, help="Coex results directory"
    )
    parser.add_argument(
        "--out", default="mu_coex_comparison.csv", help="Output comparison CSV"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.manage):
        print(f"ERROR: {args.manage} not found", file=sys.stderr)
        sys.exit(1)

    rows = read_manage(args.manage)
    analyzed = [
        r for r in rows
        if str(r.get("mu_coex_SIM", "")).strip()
        and str(r.get("mu_coex_SIM", "")).strip().lower() != "nan"
        and str(r.get("isAnalyzed", "")).strip()
    ]
    print(f"Found {len(analyzed)} analyzed rows in {args.manage}")

    print(f"Scanning coex results in {args.results} ...")
    grouped = discover_combo_results(args.results)
    print(f"Found {len(grouped)} combos with output data")

    # Build lookup: combo_key -> (mu_vals, phi_vals, phi_errs)
    curves: dict[tuple, tuple] = {}
    for combo_key, data in grouped.items():
        mu_vals, phi_vals, phi_errs, _, _ = build_curves(data["points"])
        curves[combo_key] = (mu_vals, phi_vals, phi_errs)

    comparison_rows = []
    n_fitted = 0
    n_no_data = 0
    n_no_bracket = 0

    for row in analyzed:
        epsilon = float(row["epsilon"])
        mu_sim = float(row["mu_coex_SIM"])
        mu_flex = float(row.get("mu_coex_FLEX", "nan") or "nan")

        # Build the combo key to find matching results
        combo = {f: row[f] for f in COMBO_KEY_FIELDS if f in row}
        key = combo_key_from_dict(combo)

        if key not in curves:
            print(f"  [skip] eps={epsilon:.4f}: no result data found")
            n_no_data += 1
            comparison_rows.append({
                "epsilon": epsilon,
                "mu_coex_FLEX": mu_flex,
                "mu_coex_SIM": mu_sim,
                "mu_coex_FITTED": "",
                "delta_SIM_FITTED": "",
                "delta_FLEX_FITTED": "",
                "n_mu_points": "",
                "note": "no_data",
            })
            continue

        mu_vals, phi_vals, phi_errs = curves[key]
        n_pts = len(mu_vals)

        if not has_phi_sign_change(phi_vals):
            print(f"  [skip] eps={epsilon:.4f}: no sign change in phi(mu) ({n_pts} pts)")
            n_no_bracket += 1
            comparison_rows.append({
                "epsilon": epsilon,
                "mu_coex_FLEX": mu_flex,
                "mu_coex_SIM": mu_sim,
                "mu_coex_FITTED": "",
                "delta_SIM_FITTED": "",
                "delta_FLEX_FITTED": "",
                "n_mu_points": n_pts,
                "note": "no_sign_change",
            })
            continue

        mu_fitted = fit_zero_crossing(mu_vals, phi_vals, phi_errs)

        if mu_fitted is None:
            print(f"  [skip] eps={epsilon:.4f}: fit failed ({n_pts} pts)")
            comparison_rows.append({
                "epsilon": epsilon,
                "mu_coex_FLEX": mu_flex,
                "mu_coex_SIM": mu_sim,
                "mu_coex_FITTED": "",
                "delta_SIM_FITTED": "",
                "delta_FLEX_FITTED": "",
                "n_mu_points": n_pts,
                "note": "fit_failed",
            })
            continue

        delta = mu_fitted - mu_sim
        delta_flex = mu_fitted - mu_flex if not np.isnan(mu_flex) else ""
        print(
            f"  eps={epsilon:.4f}: mu_coex_SIM={mu_sim:.6f}  "
            f"mu_coex_FITTED={mu_fitted:.6f}  delta={delta:+.6f}  ({n_pts} pts)"
        )
        n_fitted += 1
        comparison_rows.append({
            "epsilon": epsilon,
            "mu_coex_FLEX": mu_flex,
            "mu_coex_SIM": mu_sim,
            "mu_coex_FITTED": round(mu_fitted, 8),
            "delta_SIM_FITTED": round(delta, 8),
            "delta_FLEX_FITTED": round(float(delta_flex), 8) if delta_flex != "" else "",
            "n_mu_points": n_pts,
            "note": "ok",
        })

    comparison_rows.sort(key=lambda r: float(r["epsilon"]))

    fieldnames = [
        "epsilon", "mu_coex_FLEX", "mu_coex_SIM",
        "mu_coex_FITTED", "delta_SIM_FITTED", "delta_FLEX_FITTED",
        "n_mu_points", "note",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"\nResults: {n_fitted} fitted, {n_no_data} no data, {n_no_bracket} no sign change")
    print(f"Wrote {args.out}")

    if n_fitted > 0:
        fitted_vals = [r for r in comparison_rows if r["note"] == "ok"]
        deltas = [abs(float(r["delta_SIM_FITTED"])) for r in fitted_vals]
        print(f"Mean |delta| = {np.mean(deltas):.6f},  max = {np.max(deltas):.6f}")


if __name__ == "__main__":
    main()
