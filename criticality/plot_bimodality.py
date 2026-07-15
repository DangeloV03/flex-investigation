"""
plot_bimodality.py

Sanity-check plots for the bimodality-based epsilon_c pipeline (bimodality.py):
  * BC vs epsilon per system size (the scalar crossover),
  * P(phi_col) histograms for a few representative epsilon (well below / near /
    well above the candidate epsilon_c) so the bimodal -> unimodal shape change
    can be confirmed by eye, not just inferred from BC.
"""

from __future__ import annotations

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
    delta_mu: float | None = None,
    title: str | None = None,
) -> str:
    """Max BC vs the sweep axis (beta*epsilon by default), one line per system
    size, with the 1/3 and 5/9 reference lines and an optional criticality marker.

    Pass delta_mu to restrict to a single Delta mu (Figure 2: sizes at fixed dmu).
    """
    df = pd.read_csv(bc_csv)
    if delta_mu is not None and "delta_mu" in df.columns:
        df = df[np.isclose(df["delta_mu"], delta_mu)]
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
    ax.set_title(title or r"Max bimodality coefficient of $P(\phi_{col})$")
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


def plot_pooled_histograms(data: list[dict], out_png: str, *, bins: int = 60,
                           title: str | None = None) -> str:
    """Figure 1: overlay P(phi_col) histograms at several epsilon (one size).

    `data` is a list of dicts (from bimodality.histogram_data), each with:
      'epsilon', 'pooled' (1D column-op sample at coexistence), and 'BC'.
    Curves are ordered and colored by epsilon so the bimodal (two humps, low
    epsilon) -> unimodal (one hump, high epsilon) change reads left-to-right.
    """
    data = sorted(data, key=lambda d: d["epsilon"])
    cmap = plt.get_cmap("viridis")
    n = max(len(data) - 1, 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, d in enumerate(data):
        pooled = np.asarray(d["pooled"]).reshape(-1)
        ax.hist(pooled, bins=bins, density=True, histtype="step", lw=1.8,
                color=cmap(i / n),
                label=f"$\\epsilon$={d['epsilon']:.3f}  (BC={d['BC']:.2f})")
    ax.set_xlabel(r"$\phi_{col}$  (liquid $\to +1$, gas $\to -1$)")
    ax.set_ylabel(r"$P(\phi_{col})$")
    ax.set_title(title or r"Column order-parameter distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
