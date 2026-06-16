"""
analyzer.py

Long-running watcher that:
  1. Polls results/ for completed output.csv files.
  2. Groups results by outer combo (scheme, Lx, Ly, epsilon, delta_f, delta_mu, k).
  3. For each combo with enough data, computes phi(mu) and psi(mu).
  4. Checks for sign change in phi:
       - Sign change found  -> refine: 10 new mu points between the bracketing pair.
       - No sign change     -> extend: jump window in the direction needed.
     Max MAX_ADDITIONAL_REQUESTS additional data requests per combo.
  5. Finds min(psi) -> mu_coex_SIM.
  6. Saves phi/psi plots per combo.
  7. Updates manage.csv with mu_coex_SIM, isAnalyzed, RequestForAdditionalData.

Usage:
    python analyzer.py [--results results] [--manage manage.csv] [--interval 10]
"""

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MANAGE_CSV = "manage.csv"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
SAMPLES_DIR = "samples"
POLL_INTERVAL = 10.0  # seconds
MAX_ADDITIONAL_REQUESTS = 5
N_REFINEMENT_POINTS = 10
COMBO_KEY_FIELDS = ["epsilon", "delta_f", "delta_mu", "k", "scheme", "Lx", "Ly"]
MANAGE_FIELDS = COMBO_KEY_FIELDS + [
    "mu_coex_flex",
    "isSubmitted",
    "isRan",
    "isAnalyzed",
    "mu_coex_SIM",
    "RequestForAdditionalData",
]


# ---------------------------------------------------------------------------
# manage.csv helpers
# ---------------------------------------------------------------------------

def read_manage(manage_path: str) -> list[dict]:
    if not os.path.isfile(manage_path):
        return []
    with open(manage_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_manage(manage_path: str, rows: list[dict]):
    with open(manage_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANAGE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def find_manage_row(rows: list[dict], combo: dict) -> int | None:
    """Return index of the manage row matching this combo, or None."""
    for i, row in enumerate(rows):
        if all(str(row[f]) == str(combo[f]) for f in COMBO_KEY_FIELDS):
            return i
    return None


def update_manage_field(manage_path: str, combo: dict, updates: dict):
    """Update specific fields on the matching combo row, only if currently empty
    (for timestamp fields) or always (for numeric fields like RequestForAdditionalData)."""
    rows = read_manage(manage_path)
    idx = find_manage_row(rows, combo)
    if idx is None:
        print(f"[analyzer] WARNING: no manage row found for {combo}", file=sys.stderr)
        return
    for field, value in updates.items():
        # Timestamp fields: only write if empty (first writer wins)
        if field in ("isRan", "isAnalyzed") and rows[idx].get(field, ""):
            continue
        rows[idx][field] = value
    write_manage(manage_path, rows)


# ---------------------------------------------------------------------------
# Results discovery
# ---------------------------------------------------------------------------

def discover_combo_results(results_dir: str, samples_dir: str = None) -> dict:
    """Scan results/ and group output.csv files by combo key.

    Reads all params including run_settings directly from output.csv columns.
    No JSON files needed.

    Returns: { combo_key_tuple: {"job": job, "points": [(mu, df), ...]} }
    """
    grouped = {}
    results_path = pathlib.Path(results_dir)

    for csv_path in sorted(results_path.glob("*/*/output.csv")):
        try:
            df = pd.read_csv(csv_path)
            if df.empty or "mu" not in df.columns:
                continue

            required = COMBO_KEY_FIELDS + ["mu"]
            if not all(col in df.columns for col in required):
                continue

            mu = float(df["mu"].iloc[0])
            if df["mu"].nunique() != 1:
                continue

            # Build job dict from CSV columns
            job = {f: df[f].iloc[0].item() if hasattr(df[f].iloc[0], "item")
                   else df[f].iloc[0] for f in COMBO_KEY_FIELDS}
            job["mu"] = mu

            # Reconstruct run_settings from CSV columns if present
            run_settings_fields = ["beta", "num_parallel_runs", "eq_time",
                                   "prod_time", "seed_base"]
            if all(col in df.columns for col in run_settings_fields):
                job["run_settings"] = {
                    f: df[f].iloc[0].item() if hasattr(df[f].iloc[0], "item")
                    else df[f].iloc[0]
                    for f in run_settings_fields
                }
                job["run_settings"]["initial_condition"] = "slab_half_active_half_empty"
            else:
                job["run_settings"] = None

            combo_key = tuple(str(job[f]) for f in COMBO_KEY_FIELDS)

        except Exception as e:
            print(f"[analyzer] Skipping {csv_path}: {e}", file=sys.stderr)
            continue

        if combo_key not in grouped:
            grouped[combo_key] = {"job": job, "points": []}
        grouped[combo_key]["points"].append((mu, df))

    return grouped


# ---------------------------------------------------------------------------
# Physics calculations
# ---------------------------------------------------------------------------

def calculate_phi_psi(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Compute phi = <rho_active - rho_inert - rho_empty> and psi = |phi|
    with error propagation."""
    rho_a = df["rho_active"].mean()
    rho_i = df["rho_inert"].mean()
    rho_e = df["rho_empty"].mean()

    std_a = df["rho_active"].std()
    std_i = df["rho_inert"].std()
    std_e = df["rho_empty"].std()

    phi = rho_a - rho_i - rho_e
    psi = abs(phi)

    # Error propagation for psi = |phi|
    # d(psi)/d(rho_a) = sign(phi), etc.
    if psi > 0:
        prefactor = -phi / psi  # = -sign(phi)
        psi_error = np.sqrt(
            (std_a * phi / psi) ** 2
            + (std_i * prefactor) ** 2
            + (std_e * prefactor) ** 2
        )
        phi_error = np.sqrt(std_a**2 + std_i**2 + std_e**2)
    else:
        psi_error = 0.0
        phi_error = 0.0

    return phi, phi_error, psi, psi_error


def build_curves(points: list[tuple]) -> tuple:
    """Given [(mu, df), ...], return sorted arrays of mu, phi, phi_err, psi, psi_err."""
    points_sorted = sorted(points, key=lambda x: x[0])
    mu_vals, phi_vals, phi_errs, psi_vals, psi_errs = [], [], [], [], []
    for mu, df in points_sorted:
        phi, phi_err, psi, psi_err = calculate_phi_psi(df)
        mu_vals.append(mu)
        phi_vals.append(phi)
        phi_errs.append(phi_err)
        psi_vals.append(psi)
        psi_errs.append(psi_err)
    return (
        np.array(mu_vals),
        np.array(phi_vals),
        np.array(phi_errs),
        np.array(psi_vals),
        np.array(psi_errs),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_combo(combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
               mu_coex_sim=None, plots_dir=PLOTS_DIR):
    os.makedirs(plots_dir, exist_ok=True)
    epsilon, scheme, Ly = combo_key[0], combo_key[4], combo_key[6]
    eps_tag = str(abs(float(epsilon))).replace(".", "")
    tag = f"{scheme}_eps{eps_tag}_Ly{Ly}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # phi plot
    ax1.errorbar(mu_vals, phi_vals, yerr=phi_errs, fmt="o-", capsize=4,
                 linewidth=1.2, markersize=5, label=r"$\phi$")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    if mu_coex_sim is not None:
        ax1.axvline(mu_coex_sim, color="red", linestyle="--", linewidth=1,
                    label=f"$\\mu_{{coex}}^{{SIM}}={mu_coex_sim:.4f}$")
    ax1.set_xlabel(r"$\mu$", fontsize=13)
    ax1.set_ylabel(r"$\phi$", fontsize=13)
    ax1.set_title(f"$\\phi$ vs $\\mu$ — {tag}", fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    # psi plot
    ax2.errorbar(mu_vals, psi_vals, yerr=psi_errs, fmt="o-", capsize=4,
                 linewidth=1.2, markersize=5, color="orange", label=r"$\psi$")
    if mu_coex_sim is not None:
        ax2.axvline(mu_coex_sim, color="red", linestyle="--", linewidth=1,
                    label=f"$\\mu_{{coex}}^{{SIM}}={mu_coex_sim:.4f}$")
    ax2.set_xlabel(r"$\mu$", fontsize=13)
    ax2.set_ylabel(r"$\psi$", fontsize=13)
    ax2.set_title(f"$\\psi$ vs $\\mu$ — {tag}", fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    plot_path = os.path.join(plots_dir, f"{tag}_phi_psi.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[analyzer] Plot saved: {plot_path}")


# ---------------------------------------------------------------------------
# Job dispatching
# ---------------------------------------------------------------------------

def make_job_json(job_template: dict, mu: float, samples_dir: str) -> str:
    """Write a new JSON job file for a given mu and return its path."""
    scheme = str(job_template["scheme"])
    Ly = int(job_template["Ly"])
    mu_tag = f"mu{round(abs(mu) * 1_000_000):07d}"
    filename = f"{scheme}_Ly{Ly}_{mu_tag}.json"
    os.makedirs(samples_dir, exist_ok=True)
    filepath = os.path.join(samples_dir, filename)

    # Coerce all values to native Python types for JSON serialization
    def to_native(v):
        if hasattr(v, "item"):  # catches all numpy scalars
            return v.item()
        return v

    job = {k: to_native(v) for k, v in job_template.items()}
    job["mu"] = round(float(mu), 6)

    with open(filepath, "w") as f:
        json.dump(job, f, indent=2)
    return filepath


def dispatch_jobs(mu_values: list[float], job_template: dict, samples_dir: str):
    """Write JSON files and call json_runner.py for each mu value sequentially."""
    for mu in mu_values:
        json_path = make_job_json(job_template, mu, samples_dir)
        print(f"[analyzer] Dispatching {json_path} (mu={mu:.6f})")
        result = subprocess.run([sys.executable, "json_runner.py", json_path])
        if result.returncode != 0:
            print(f"[analyzer] WARNING: json_runner.py failed for {json_path}",
                  file=sys.stderr)


# ---------------------------------------------------------------------------
# Core analysis logic per combo
# ---------------------------------------------------------------------------

def analyze_combo(combo_key: tuple, data: dict, manage_path: str,
                  plots_dir: str, samples_dir: str):
    job = data["job"]
    points = data["points"]

    combo = {f: job[f] for f in COMBO_KEY_FIELDS}
    epsilon, scheme, Ly = job["epsilon"], job["scheme"], job["Ly"]
    eps_tag = str(abs(float(epsilon))).replace(".", "")
    tag = f"{scheme}_eps{eps_tag}_Ly{Ly}"

    mu_vals, phi_vals, phi_errs, psi_vals, psi_errs = build_curves(points)

    if len(mu_vals) < 2:
        return  # not enough points yet

    # Check manage row for current request count
    rows = read_manage(manage_path)
    idx = find_manage_row(rows, combo)
    if idx is None:
        return
    n_requests = int(rows[idx].get("RequestForAdditionalData", 0))

    # Already analyzed
    if rows[idx].get("isAnalyzed", ""):
        return

    # Check for sign change in phi
    signs = np.sign(phi_vals)
    has_sign_change = not np.all(signs == signs[0])

    if has_sign_change:
        # Find bracketing pair: closest phi to 0 from each side
        pos_mask = phi_vals > 0
        neg_mask = phi_vals < 0

        mu_pos = mu_vals[pos_mask][np.argmin(np.abs(phi_vals[pos_mask]))]
        mu_neg = mu_vals[neg_mask][np.argmin(np.abs(phi_vals[neg_mask]))]

        mu_lo = min(mu_pos, mu_neg)
        mu_hi = max(mu_pos, mu_neg)

        # Check if we already have refined points in this bracket
        existing_in_bracket = np.sum((mu_vals >= mu_lo) & (mu_vals <= mu_hi))

        if existing_in_bracket < N_REFINEMENT_POINTS and n_requests < MAX_ADDITIONAL_REQUESTS:
            # Refine
            new_mus = list(np.linspace(mu_lo, mu_hi, N_REFINEMENT_POINTS))
            # Filter out mu values already sampled (within tolerance)
            new_mus = [m for m in new_mus
                       if not any(abs(m - existing) < 1e-6 for existing in mu_vals)]
            if new_mus:
                print(f"[analyzer] {tag}: sign change found, refining "
                      f"[{mu_lo:.4f}, {mu_hi:.4f}] with {len(new_mus)} points")
                n_requests += 1
                rows[idx]["RequestForAdditionalData"] = n_requests
                write_manage(manage_path, rows)
                dispatch_jobs(new_mus, job, samples_dir)
                return  # re-analyze after new data arrives

        # Find mu_coex_SIM as min of psi
        min_idx = np.argmin(psi_vals)
        mu_coex_sim = float(mu_vals[min_idx])
        print(f"[analyzer] {tag}: mu_coex_SIM = {mu_coex_sim:.6f}")

        # Save plot with mu_coex_SIM marked
        plot_combo(combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
                   mu_coex_sim=mu_coex_sim, plots_dir=plots_dir)

        # Update manage.csv
        update_manage_field(manage_path, combo, {
            "mu_coex_SIM": mu_coex_sim,
            "isAnalyzed": time.strftime("%Y-%m-%d %H:%M:%S"),
            "RequestForAdditionalData": n_requests,
        })

    else:
        # No sign change — need to extend the mu window
        if n_requests >= MAX_ADDITIONAL_REQUESTS:
            print(f"[analyzer] {tag}: max additional requests reached, "
                  f"cannot find sign change. Skipping.")
            # Still plot what we have
            plot_combo(combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
                       plots_dir=plots_dir)
            return

        window = mu_vals[-1] - mu_vals[0]
        if np.all(signs > 0):
            # All phi > 0: active phase dominates everywhere, need lower mu
            new_lo = mu_vals[0] - window
            new_hi = mu_vals[0]
            print(f"[analyzer] {tag}: all phi>0, extending window lower "
                  f"[{new_lo:.4f}, {new_hi:.4f}]")
        else:
            # All phi < 0: need higher mu
            new_lo = mu_vals[-1]
            new_hi = mu_vals[-1] + window
            print(f"[analyzer] {tag}: all phi<0, extending window higher "
                  f"[{new_lo:.4f}, {new_hi:.4f}]")

        new_mus = list(np.linspace(new_lo, new_hi, N_REFINEMENT_POINTS))
        new_mus = [m for m in new_mus
                   if not any(abs(m - existing) < 1e-6 for existing in mu_vals)]

        n_requests += 1
        rows[idx]["RequestForAdditionalData"] = n_requests
        write_manage(manage_path, rows)

        # Plot current state so we can see what's happening
        plot_combo(combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
                   plots_dir=plots_dir)

        dispatch_jobs(new_mus, job, samples_dir)


# ---------------------------------------------------------------------------
# Main watch loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Watch results/ and compute mu_coex_SIM per combo."
    )
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--manage", default=MANAGE_CSV)
    parser.add_argument("--plots", default=PLOTS_DIR)
    parser.add_argument("--samples", default=SAMPLES_DIR)
    parser.add_argument("--interval", type=float, default=POLL_INTERVAL)
    args = parser.parse_args()

    print(f"[analyzer] Watching '{args.results}' every {args.interval}s "
          f"(Ctrl-C to stop)")

    processed_combos = set()  # combo_keys that are fully analyzed

    while True:
        grouped = discover_combo_results(args.results)

        for combo_key, data in grouped.items():
            if combo_key in processed_combos:
                continue

            # Check if this combo is already marked analyzed in manage.csv
            rows = read_manage(args.manage)
            combo = {f: data["job"][f] for f in COMBO_KEY_FIELDS}
            idx = find_manage_row(rows, combo)
            if idx is not None and rows[idx].get("isAnalyzed", ""):
                processed_combos.add(combo_key)
                continue

            analyze_combo(combo_key, data, args.manage, args.plots, args.samples)

            # Re-check if now analyzed
            rows = read_manage(args.manage)
            if idx is not None and rows[idx].get("isAnalyzed", ""):
                processed_combos.add(combo_key)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()