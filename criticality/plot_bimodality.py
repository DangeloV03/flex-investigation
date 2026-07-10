"""
plot_bimodality.py

Sanity-check plots for the bimodality-based epsilon_c pipeline (bimodality.py):
  * BC vs epsilon per system size (the scalar crossover),
  * P(phi_col) histograms for a few representative epsilon (well below / near /
    well above the candidate epsilon_c) so the bimodal -> unimodal shape change
    can be confirmed by eye, not just inferred from BC.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BC_UNIMODAL = 1.0 / 3.0
BC_BIMODAL_CUTOFF = 5.0 / 9.0


def plot_bc_vs_epsilon(
    bc_csv: str,
    out_png: str,
    *,
    x_col: str = "beta_epsilon",
    crit: float | None = None,
) -> str:
    """Max BC vs the sweep axis (beta*epsilon by default), one line per system
    size, with the 1/3 and 5/9 reference lines and the criticality marker."""
    df = pd.read_csv(bc_csv)
    xlabel = r"$\beta\epsilon$" if x_col == "beta_epsilon" else r"$\epsilon$"
    fig, ax = plt.subplots(figsize=(6, 4))
    for L_long, sub in df.groupby("L_long"):
        sub = sub.sort_values(x_col)
        L_short = int(sub["L_short"].iloc[0])
        ax.plot(sub[x_col], sub["BC"], "o-", label=f"{int(L_long)}x{L_short}")
    ax.axhline(BC_UNIMODAL, ls=":", c="grey", lw=1, label="1/3 (Gaussian)")
    ax.axhline(BC_BIMODAL_CUTOFF, ls="--", c="grey", lw=1, label="5/9 (cutoff)")
    if crit is not None and np.isfinite(crit):
        ax.axvline(crit, ls="-", c="crimson", lw=1,
                   label=f"criticality={crit:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("max Sarle's BC")
    ax.set_title(r"Max bimodality coefficient of $P(\phi_{col})$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_bc_family(bc_csv: str, out_png: str, *, x_col: str = "beta_epsilon") -> str:
    """Thermal-phase-diagram family: max BC vs beta*epsilon, one curve per
    delta_mu (further split by system size if more than one L is present)."""
    df = pd.read_csv(bc_csv)
    xlabel = r"$\beta\epsilon$" if x_col == "beta_epsilon" else r"$\epsilon$"
    multi_L = df["L_long"].nunique() > 1
    fig, ax = plt.subplots(figsize=(7, 5))
    for keys, sub in df.groupby(["delta_mu", "L_long"] if multi_L else ["delta_mu"]):
        sub = sub.sort_values(x_col)
        if multi_L:
            dmu, L_long = keys
            label = f"$\\Delta\\mu$={dmu}, L={int(L_long)}"
        else:
            dmu = keys[0] if isinstance(keys, tuple) else keys
            label = f"$\\Delta\\mu$={dmu}"
        ax.plot(sub[x_col], sub["BC"], "o-", ms=4, label=label)
    ax.axhline(BC_UNIMODAL, ls=":", c="grey", lw=1)
    ax.axhline(BC_BIMODAL_CUTOFF, ls="--", c="grey", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Max Bimodality Coefficient")
    ax.set_title(r"Max bimodality coefficient of $P(\phi_{col})$ vs $\beta\epsilon$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_pooled_histograms(cache_dir: str, epsilons, out_png: str, *, bins: int = 60) -> str:
    """Overlay P(phi_col) histograms for a few epsilon from cached column ops.

    Reads <cache_dir>/*.npy (+ .json sidecar with 'epsilon'), selecting the cached
    chunks whose epsilon is closest to each requested value.
    """
    entries = []
    for jpath in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        with open(jpath) as f:
            meta = json.load(f)
        npy = jpath[:-5] + ".npy"
        if "epsilon" in meta and os.path.isfile(npy):
            entries.append((float(meta["epsilon"]), npy))
    if not entries:
        raise FileNotFoundError(f"no cached column ops with epsilon in {cache_dir}")

    fig, ax = plt.subplots(figsize=(6, 4))
    for target in epsilons:
        eps, npy = min(entries, key=lambda e: abs(e[0] - target))
        pooled = np.load(npy).reshape(-1)
        ax.hist(pooled, bins=bins, density=True, histtype="step",
                label=f"$\\epsilon$={eps:.3f}")
    ax.set_xlabel(r"$\phi_{col}$")
    ax.set_ylabel(r"$P(\phi_{col})$")
    ax.set_title("Column order-parameter distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
