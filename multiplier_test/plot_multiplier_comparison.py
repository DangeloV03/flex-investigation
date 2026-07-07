"""
plot_multiplier_comparison.py

Overlay peak-chi-vs-L curves from the production-time multiplier runs (2x, 3x, 4x,
5x) on a single log-log axis, to see whether chi^max(L) grows with production run
length (under-converged) or collapses onto one curve (time-converged).

It consumes the per-multiplier peak CSVs written by plot_susceptibility.py:
    <plots-root>/<mult>/peak_chi_vs_L_pooled.csv   (or peak_chi_vs_L.csv)
with columns L, epsilon, chi_mean, chi_stderr. Run plot_susceptibility.py once per
multiplier first (see multiplier_test/README.md).

Usage:
    python multiplier_test/plot_multiplier_comparison.py \
        --plots-root multiplier_test/plots \
        --out multiplier_test/plots/peak_chi_vs_L_comparison.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Default multiplier labels, in run-length order (matches folder names).
DEFAULT_MULTS = ["2x", "3x", "4x", "5x"]
# Kumar & Dasgupta (2020) reference slope, same as plot_susceptibility.py.
REF_A, REF_GNU = 0.095, 1.75


def load_peaks(plots_root: str, mult: str) -> pd.DataFrame | None:
    """Load a multiplier's peak CSV (prefer the pooled file), or None if absent."""
    for name in ("peak_chi_vs_L_pooled.csv", "peak_chi_vs_L.csv"):
        path = os.path.join(plots_root, mult, name)
        if os.path.exists(path):
            df = pd.read_csv(path).sort_values("L")
            return df
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plots-root", default="multiplier_test/plots",
                        help="Dir holding per-multiplier subfolders with peak CSVs")
    parser.add_argument("--out", default="multiplier_test/plots/peak_chi_vs_L_comparison.png",
                        help="Output PNG path")
    parser.add_argument("--mults", nargs="+", default=DEFAULT_MULTS,
                        help="Multiplier subfolder names to overlay (default: 2x 3x 4x 5x)")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(args.mults)))

    all_L: list[float] = []
    plotted = 0
    for mult, color in zip(args.mults, colors):
        peaks = load_peaks(args.plots_root, mult)
        if peaks is None or peaks.empty:
            print(f"[skip] {mult}: no peak CSV under {os.path.join(args.plots_root, mult)}")
            continue
        L = peaks["L"].to_numpy(dtype=float)
        chi = peaks["chi_mean"].to_numpy(dtype=float)
        err = peaks["chi_stderr"].to_numpy(dtype=float) if "chi_stderr" in peaks else None
        all_L.extend(L.tolist())

        # Per-multiplier slope fit (gamma/nu) over all its L points, for the legend.
        label = mult
        if len(L) >= 2:
            slope, intercept = np.polyfit(np.log(L), np.log(chi), 1)
            label = rf"{mult}  ($\gamma/\nu={slope:.3f}$)"
            print(f"[{mult}] gamma/nu={slope:.4f}, A={np.exp(intercept):.4f}, "
                  f"chi^max={chi.max():.2f} at L={int(L[np.argmax(chi)])}, n_L={len(L)}")
        ax.errorbar(L, chi, yerr=err, fmt="o-", color=color, markersize=6,
                    capsize=3, linewidth=1.3, label=label, zorder=3)
        plotted += 1

    if plotted == 0:
        print("ERROR: no multiplier peak CSVs found — run plot_susceptibility.py per "
              "multiplier first.", file=sys.stderr)
        return 1

    # Reference K&D slope across the observed L range for visual comparison.
    L_fine = np.geomspace(min(all_L), max(all_L), 200)
    ax.loglog(L_fine, REF_A * L_fine ** REF_GNU, "-", color="red", linewidth=1.2,
              alpha=0.8, label=rf"$\gamma/\nu={REF_GNU}$ (K&D 2020)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\chi^{\mathrm{max}}(L)$")
    ax.set_title(r"Peak $\chi$ vs $L$ across production-time multipliers")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
