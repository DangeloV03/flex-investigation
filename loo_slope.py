"""
loo_slope.py

LOO jackknife on chi_max vs L using the pre-computed CSV from
plot_susceptibility.py.  Reads ~7 rows — runs in under a second.

Usage:
    python loo_slope.py
    python loo_slope.py --csv plots/exact/peak_chi_vs_L_pooled.csv
    python loo_slope.py --csv plots/exact/peak_chi_vs_L_pooled.csv --fixed-A 0.095
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KD_GAMMA_NU     = 1.73
KD_GAMMA_NU_ERR = 0.01
KD_A            = 0.095


def fit_power_law(
    L_arr: np.ndarray,
    chi_arr: np.ndarray,
    fixed_A: float | None = None,
) -> tuple[float, float]:
    """OLS log-log fit. If fixed_A is given, intercept is fixed to log(fixed_A)."""
    log_L   = np.log(L_arr)
    log_chi = np.log(chi_arr)
    if fixed_A is not None:
        # Slope only: minimise sum((log_chi - log(A) - slope*log_L)^2)
        log_A = np.log(fixed_A)
        slope = float(np.dot(log_L, log_chi - log_A) / np.dot(log_L, log_L))
        return slope, fixed_A
    slope, intercept = np.polyfit(log_L, log_chi, 1)
    return float(slope), float(np.exp(intercept))


def main() -> None:
    parser = argparse.ArgumentParser(description="LOO jackknife on chi_max vs L")
    parser.add_argument("--csv", default="plots/exact/peak_chi_vs_L_pooled.csv")
    parser.add_argument("--outdir", default=None,
                        help="Directory for output plot (default: same dir as CSV)")
    parser.add_argument(
        "--fixed-A", type=float, default=None,
        help="Fix the prefactor A to this value and fit only gamma/nu. "
             "Use --fixed-A 0.095 to constrain to the K&D prefactor.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}\n"
                         "Run plot_susceptibility.py --pooled first to generate it.")

    df = pd.read_csv(args.csv)
    peaks = (
        df.sort_values("chi_mean", ascending=False)
          .drop_duplicates("L")
          .sort_values("L")
          .reset_index(drop=True)
    )

    Ls       = peaks["L"].to_numpy(float)
    chi_max  = peaks["chi_mean"].to_numpy(float)
    chi_err  = peaks["chi_stderr"].to_numpy(float)
    eps_star = peaks["epsilon"].to_numpy(float)
    n        = len(Ls)

    fixed_A = args.fixed_A

    # Full fit
    gnu_full, A_full = fit_power_law(Ls, chi_max, fixed_A)

    # LOO: drop each L in turn, refit
    gnu_loo = np.zeros(n)
    A_loo   = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        gnu_loo[i], A_loo[i] = fit_power_law(Ls[mask], chi_max[mask], fixed_A)

    gnu_loo_std = float(gnu_loo.std(ddof=1))
    gnu_loo_min = float(gnu_loo.min())
    gnu_loo_max = float(gnu_loo.max())
    gnu_err     = gnu_loo_std

    sigma_diff = abs(gnu_full - KD_GAMMA_NU) / np.sqrt(gnu_err**2 + KD_GAMMA_NU_ERR**2)
    within     = sigma_diff <= 2.0

    # ---- Print results ----
    A_label = f"fixed A={fixed_A}" if fixed_A is not None else "free A"
    print(f"\nData: {args.csv}  ({n} L values)  [{A_label}]")
    print(f"{'L':>6}  {'eps_peak':>9}  {'chi_max':>10}  {'±':>2}  {'chi_err':>8}  "
          f"{'γ/ν (LOO drop)':>16}")
    print("-" * 65)
    for i in range(n):
        print(f"{int(Ls[i]):>6}  {eps_star[i]:>9.4f}  {chi_max[i]:>10.4f}  "
              f"{'±':>2}  {chi_err[i]:>8.4f}  {gnu_loo[i]:>16.4f}")
    print("-" * 65)
    if fixed_A is not None:
        print(f"\nFull fit:      γ/ν = {gnu_full:.4f},  A = {A_full:.4f}  (fixed)")
    else:
        print(f"\nFull fit:      γ/ν = {gnu_full:.4f},  A = {A_full:.4f}")
    print(f"LOO slopes:    mean = {gnu_loo.mean():.4f},  "
          f"std = {gnu_loo_std:.4f},  "
          f"range = [{gnu_loo_min:.4f}, {gnu_loo_max:.4f}]")
    print(f"Our result:    γ/ν = {gnu_full:.4f} ± {gnu_err:.4f}  (std of LOO slopes)")
    print(f"K&D (2020):    γ/ν = {KD_GAMMA_NU} ± {KD_GAMMA_NU_ERR}")
    print(f"\nDifference:    {gnu_full - KD_GAMMA_NU:+.4f}  ({sigma_diff:.2f}σ combined)")
    print(f"Within 2σ of K&D: {'YES ✓' if within else 'NO ✗'}")

    # ---- Plot ----
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    L_fine = np.geomspace(Ls.min(), Ls.max(), 200)

    # Left: chi_max vs L
    ax = axes[0]
    ax.errorbar(Ls, chi_max, yerr=chi_err, fmt="o", color="black",
                markersize=7, capsize=4, zorder=5, label="data")
    ax.loglog(L_fine, A_full * L_fine ** gnu_full, "-", color="blue", linewidth=2,
              label=(rf"fit ($A={A_full:.3f}$ fixed): "
                     rf"$\gamma/\nu={gnu_full:.3f}\pm{gnu_err:.3f}$"
                     if fixed_A is not None else
                     rf"fit: $A={A_full:.3f}$, $\gamma/\nu={gnu_full:.3f}\pm{gnu_err:.3f}$"))
    ax.loglog(L_fine, KD_A * L_fine ** KD_GAMMA_NU, "--", color="red", linewidth=1.8,
              label=rf"K&D: $A={KD_A}$, $\gamma/\nu={KD_GAMMA_NU}\pm{KD_GAMMA_NU_ERR}$")
    ax.fill_between(L_fine,
                    A_full * L_fine ** gnu_loo_min,
                    A_full * L_fine ** gnu_loo_max,
                    color="blue", alpha=0.12, label="LOO range")
    ax.set_xlabel("$L$", fontsize=12)
    ax.set_ylabel(r"$\chi^{\rm max}(L)$", fontsize=12)
    ax.set_title(r"$\chi^{\rm max}$ vs $L$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # Right: γ/ν per dropped L
    ax = axes[1]
    ax.plot(Ls, gnu_loo, "o-", color="navy", markersize=8, zorder=4,
            label=r"$\gamma/\nu$ (L dropped)")
    ax.axhline(gnu_full, color="blue", linewidth=1.5,
               label=rf"Full fit: {gnu_full:.4f}")
    ax.fill_between([Ls[0] * 0.7, Ls[-1] * 1.4],
                    gnu_full - gnu_err, gnu_full + gnu_err,
                    color="blue", alpha=0.15)
    ax.axhline(KD_GAMMA_NU, color="red", linestyle="--", linewidth=1.5,
               label=rf"K&D: {KD_GAMMA_NU} ± {KD_GAMMA_NU_ERR}")
    ax.fill_between([Ls[0] * 0.7, Ls[-1] * 1.4],
                    KD_GAMMA_NU - KD_GAMMA_NU_ERR,
                    KD_GAMMA_NU + KD_GAMMA_NU_ERR,
                    color="red", alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("$L$ dropped", fontsize=12)
    ax.set_ylabel(r"$\gamma/\nu$", fontsize=12)
    ax.set_title(rf"LOO: {sigma_diff:.2f}σ from K&D  ({'within 2σ' if within else 'outside 2σ'})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(Ls[0] * 0.7, Ls[-1] * 1.4)

    suffix = f"_fixedA{fixed_A}".replace(".", "p") if fixed_A is not None else ""
    path = os.path.join(outdir, f"loo_slope{suffix}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {path}")


if __name__ == "__main__":
    main()
