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
    crit_row: dict | None = None,
    delta_mu: float | None = None,
    title: str | None = None,
) -> str:
    """Max BC vs the sweep axis (beta*epsilon by default), one line per system
    size, with the 1/3 and 5/9 reference lines and an optional criticality marker.

    Pass delta_mu to restrict to a single Delta mu (Figure 2: sizes at fixed dmu).
    Pass crit_row (from locate_epsilon_c) to shade the transition bracket.
    """
    df = pd.read_csv(bc_csv)
    if delta_mu is not None and "delta_mu" in df.columns:
        df = df[np.isclose(df["delta_mu"], delta_mu)]
    xlabel = r"$\beta\epsilon$" if x_col == "beta_epsilon" else r"$\epsilon$"
    fig, ax = plt.subplots(figsize=(6, 4))
    for L_long, sub in df.groupby("L_long"):
        sub = sub.sort_values(x_col)
        L_short = int(sub["L_short"].iloc[0])
        yerr = sub["BC_err"] if "BC_err" in sub.columns else None
        if yerr is not None:
            yerr = pd.to_numeric(yerr, errors="coerce")
            if not yerr.notna().any():
                yerr = None
        ax.errorbar(sub[x_col], sub["BC"], yerr=yerr, fmt="o-", capsize=2,
                    label=f"{int(L_long)}x{L_short}")
    _draw_transition_bracket(ax, crit_row, x_col=x_col)
    ax.axhline(BC_UNIMODAL, ls=":", c="grey", lw=1, label="1/3 (Gaussian)")
    ax.axhline(BC_BIMODAL_CUTOFF, ls="--", c="grey", lw=1, label="5/9 (cutoff)")
    if crit is not None and np.isfinite(crit):
        ax.axvline(crit, ls="-", c="crimson", lw=1.2,
                   label=rf"$\epsilon_c$ sigmoid={crit:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("max Sarle's BC")
    ax.set_title(title or r"Max bimodality coefficient of $P(\phi_{col})$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def _draw_transition_bracket(ax, crit_row: dict | None, *, x_col: str = "beta_epsilon") -> None:
    """Shade the transition region and draw BC contour lines from crit_row."""
    if not crit_row:
        return
    try:
        x_hi = float(crit_row["transition_x_high"])
        x_lo = float(crit_row["transition_x_low"])
        bc_hi = float(crit_row.get("transition_bc_high", TRANSITION_BC_HIGH))
        bc_lo = float(crit_row.get("transition_bc_low", TRANSITION_BC_LOW))
    except (KeyError, TypeError, ValueError):
        return
    if not (np.isfinite(x_hi) and np.isfinite(x_lo) and x_lo > x_hi):
        return
    ax.axhspan(bc_lo, bc_hi, color="#FEF3C7", alpha=0.35, zorder=0,
               label="BC transition levels")
    ax.axvspan(x_hi, x_lo, color="#FDBA74", alpha=0.22, zorder=0,
               label=r"$\beta\epsilon$ transition region")
    ax.axhline(bc_hi, ls="--", c="#D97706", lw=1, alpha=0.9)
    ax.axhline(bc_lo, ls="--", c="#D97706", lw=1, alpha=0.9)
    half = crit_row.get("transition_half_width")
    if half is not None and np.isfinite(float(half)):
        mid = 0.5 * (x_hi + x_lo)
        y_text = max(bc_lo - 0.08, 0.05)
        ax.annotate(
            rf"$\pm {float(half):.3f}$",
            xy=(mid, bc_lo), xytext=(mid, y_text),
            ha="center", fontsize=8, color="#92400E",
        )


# Default transition levels (mirrors bimodality.py).
TRANSITION_BC_HIGH = 0.85
TRANSITION_BC_LOW = 0.65

_INNER_TITLE = r"Max bimodality coefficient of $P(\phi_{col})$ vs $\beta\epsilon$"


def _plot_bc_curves_on_ax(
    ax,
    df: pd.DataFrame,
    *,
    x_col: str = "beta_epsilon",
    legend: bool = True,
) -> None:
    """One curve per delta_mu (and L if multiple sizes), with BC_err vertical bars."""
    xlabel = r"$\beta\epsilon$" if x_col == "beta_epsilon" else r"$\epsilon$"
    multi_L = df["L_long"].nunique() > 1
    for keys, sub in df.groupby(["delta_mu", "L_long"] if multi_L else ["delta_mu"]):
        sub = sub.sort_values(x_col)
        if multi_L:
            dmu, L_long = keys
            label = f"$\\Delta\\mu$={dmu}, L={int(L_long)}"
        else:
            dmu = keys[0] if isinstance(keys, tuple) else keys
            label = f"$\\Delta\\mu$={dmu}"
        yerr = sub["BC_err"] if "BC_err" in sub.columns else None
        if yerr is not None:
            yerr = pd.to_numeric(yerr, errors="coerce")
            if not yerr.notna().any():
                yerr = None
        ax.errorbar(sub[x_col], sub["BC"], yerr=yerr, fmt="o-", ms=4, capsize=3,
                    capthick=1.2, elinewidth=1.2, label=label, zorder=3)
    ax.axhline(BC_UNIMODAL, ls=":", c="grey", lw=1)
    ax.axhline(BC_BIMODAL_CUTOFF, ls="--", c="grey", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Max Bimodality Coefficient")
    if legend:
        ax.legend(fontsize=8, loc="upper right")


def plot_bc_scheme_comparison(
    panels: list[tuple[str, str]],
    out_png: str,
    *,
    x_col: str = "beta_epsilon",
) -> str:
    """Side-by-side BC_max-vs-(beta*epsilon) family plots (Scheme 1 vs Scheme 3).

    `panels` is a list of (bc_csv, panel_heading) pairs, e.g.
    [("criticality/scheme1/bc_vs_beta_epsilon.csv", "Scheme 1: Homogenous"), ...].
    Each CSV is the output of ``phase_diagram``; vertical bars use ``BC_err``.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (bc_csv, heading) in zip(axes, panels):
        df = pd.read_csv(bc_csv)
        _plot_bc_curves_on_ax(ax, df, x_col=x_col)
        ax.set_title(f"{heading}\n{_INNER_TITLE}", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_bc_family(
    bc_csv: str,
    out_png: str,
    *,
    x_col: str = "beta_epsilon",
    crit_csv: str | None = None,
    show_transition_bracket: bool = True,
) -> str:
    """Thermal-phase-diagram family: max BC vs beta*epsilon, one curve per
    delta_mu (further split by system size if more than one L is present).

    Vertical bars use ``BC_err`` from the CSV. If ``crit_csv`` is provided and
    ``show_transition_bracket`` is True, each curve also gets a sigmoid marker
    and a shaded transition bracket.
    """
    df = pd.read_csv(bc_csv)
    crit_df = None
    if show_transition_bracket and crit_csv and os.path.isfile(crit_csv):
        crit_df = pd.read_csv(crit_csv)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _plot_bc_curves_on_ax(ax, df, x_col=x_col)
    bracket_labeled = False
    if crit_df is not None:
        multi_L = df["L_long"].nunique() > 1
        for keys, sub in df.groupby(["delta_mu", "L_long"] if multi_L else ["delta_mu"]):
            dmu = keys[0] if isinstance(keys, tuple) else keys
            L_long = keys[1] if multi_L else int(sub["L_long"].iloc[0])
            match = crit_df[np.isclose(crit_df["delta_mu"], dmu)]
            if multi_L:
                match = match[np.isclose(match["L_long"], L_long)]
            if not len(match):
                continue
            crit_row = match.iloc[0].to_dict()
            x_c = crit_row.get("criticality_estimate")
            if x_c is not None and np.isfinite(float(x_c)):
                ax.axvline(float(x_c), ls="-", c="crimson", lw=1.2, alpha=0.85, zorder=2)
            if not bracket_labeled:
                _draw_transition_bracket(ax, crit_row, x_col=x_col)
                bracket_labeled = True
            else:
                try:
                    x_hi = float(crit_row["transition_x_high"])
                    x_lo = float(crit_row["transition_x_low"])
                    if np.isfinite(x_hi) and np.isfinite(x_lo) and x_lo > x_hi:
                        ax.axvspan(x_hi, x_lo, color="#FDBA74", alpha=0.18, zorder=0)
                        ax.axvline(float(crit_row["criticality_estimate"]), ls="-",
                                   c="crimson", lw=1.2, alpha=0.85, zorder=2)
                except (KeyError, TypeError, ValueError):
                    pass
    ax.set_title(_INNER_TITLE)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


BC_UNIMODAL = 1.0 / 3.0
BC_BIMODAL_CUTOFF = 5.0 / 9.0


def _phi_col_bin_edges(Ly: int | None, *, bins: int = 60) -> int | np.ndarray:
    """Discrete bin edges for averaged short-axis spins, or a fixed bin count."""
    if Ly and Ly > 0:
        step = 2.0 / Ly
        pad = step / 2.0
        return np.arange(-1.0 - pad, 1.0 + step + pad, step)
    return bins


def _panel_regime(d: dict) -> str:
    if d["BC"] < BC_BIMODAL_CUTOFF:
        return "homogeneous"
    balance = d.get("balance")
    if balance is None:
        frac_liq = d.get("frac_liq")
        frac_gas = d.get("frac_gas")
        if frac_liq is not None and frac_gas is not None:
            balance = min(frac_liq, frac_gas)
    if balance is not None and balance >= 0.15:
        return "phase-separated"
    return "near criticality"


def plot_pooled_histograms(data: list[dict], out_png: str, *, bins: int = 60,
                           title: str | None = None) -> str:
    """Figure 1: side-by-side P(phi_col) histograms at several epsilon (one size).

    `data` is a list of dicts (from bimodality.histogram_data), each with:
      'epsilon', 'pooled' (1D column-op sample at coexistence), and 'BC'.
    Panels are ordered left-to-right by epsilon so the bimodal (two humps, low
    epsilon) -> unimodal (one hump, high epsilon) change is easy to read.
    """
    data = sorted(data, key=lambda d: d["epsilon"])
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, d in zip(axes, data):
        pooled = np.asarray(d["pooled"]).reshape(-1)
        hist_bins = _phi_col_bin_edges(d.get("Ly"), bins=bins)
        ax.hist(
            pooled, bins=hist_bins, density=True, histtype="stepfilled",
            color="#4C72B0", alpha=0.55, edgecolor="#2F4A7A", linewidth=0.8,
        )
        ax.axvline(-1.0, ls="--", c="#888888", lw=0.8, alpha=0.8)
        ax.axvline(1.0, ls="--", c="#888888", lw=0.8, alpha=0.8)
        regime = _panel_regime(d)
        ax.set_title(
            rf"$\epsilon={d['epsilon']:.3f}$" + "\n"
            + rf"BC$={d['BC']:.2f}$, {regime.replace('-', ' ')}",
            fontsize=10,
        )
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(r"$P(\phi_{col})$")
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel(r"$\phi_{col}$  (liquid $\to +1$, gas $\to -1$)")
    if title:
        fig.suptitle(title, fontsize=11, y=1.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png
