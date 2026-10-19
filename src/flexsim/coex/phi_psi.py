"""phi(mu) and psi(mu) from simulation output, and the per-combo phi/psi plot."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flexsim.coex.layout import COMBO_KEY_FIELDS, RESULTS_DIR, combo_dir_name, phi_psi_png_path, write_phi_psi_csv


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

def _plot_mu_coex(mu_coex_sim) -> bool:
    if mu_coex_sim is None:
        return False
    if isinstance(mu_coex_sim, str) and mu_coex_sim.lower() == "nan":
        return False
    if isinstance(mu_coex_sim, float) and np.isnan(mu_coex_sim):
        return False
    return True


def mu_coex_for_plot(
    mu_vals: np.ndarray,
    psi_vals: np.ndarray,
    manage_row: dict | None = None,
) -> float | None:
    """Pick mu_coex to annotate plots: manage.csv fitted value, else argmin(psi).

    Returns None when manage.csv explicitly records mu_coex_FITTED=NaN (unstable).
    """
    if manage_row is not None:
        raw = str(manage_row.get("mu_coex_FITTED", "")).strip()
        if raw.lower() == "nan":
            return None
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    if len(mu_vals) == 0:
        return None
    return float(mu_vals[int(np.argmin(psi_vals))])


def plot_combo(combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
               mu_coex_sim=None, results_dir=RESULTS_DIR):
    job = dict(zip(COMBO_KEY_FIELDS, combo_key))
    tag = combo_dir_name(job)
    plot_path = phi_psi_png_path(job, results_dir)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    write_phi_psi_csv(
        job, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs, base=results_dir,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # phi plot
    ax1.errorbar(mu_vals, phi_vals, yerr=phi_errs, fmt="o-", capsize=4,
                 linewidth=1.2, markersize=5, label=r"$\phi$")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    if _plot_mu_coex(mu_coex_sim):
        ax1.axvline(mu_coex_sim, color="red", linestyle="--", linewidth=1,
                    label=f"$\\mu_{{coex}}^{{SIM}}={float(mu_coex_sim):.4f}$")
    ax1.set_xlabel(r"$\mu$", fontsize=13)
    ax1.set_ylabel(r"$\phi$", fontsize=13)
    ax1.set_title(f"$\\phi$ vs $\\mu$ — {tag}", fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    # psi plot
    ax2.errorbar(mu_vals, psi_vals, yerr=psi_errs, fmt="o-", capsize=4,
                 linewidth=1.2, markersize=5, color="orange", label=r"$\psi$")
    if _plot_mu_coex(mu_coex_sim):
        ax2.axvline(mu_coex_sim, color="red", linestyle="--", linewidth=1,
                    label=f"$\\mu_{{coex}}^{{SIM}}={float(mu_coex_sim):.4f}$")
    ax2.set_xlabel(r"$\mu$", fontsize=13)
    ax2.set_ylabel(r"$\psi$", fontsize=13)
    ax2.set_title(f"$\\psi$ vs $\\mu$ — {tag}", fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[analyzer] Plot saved: {plot_path}")
