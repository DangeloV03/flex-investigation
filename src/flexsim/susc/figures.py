"""Figures for susceptibility campaigns: observables vs epsilon, peak chi vs L, m histograms."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


L_PLOT_STYLE: dict[int, dict[str, str]] = {
    16: {"color": "black", "marker": "o"},
    32: {"color": "red", "marker": "s"},
    48: {"color": "#90EE90", "marker": "^"},
    64: {"color": "blue", "marker": "D"},
    96: {"color": "cyan", "marker": "v"},
    128: {"color": "saddlebrown", "marker": "<"},
    256: {"color": "orange", "marker": ">"},
}


def _plot_l_curves_vs_epsilon(
    agg: pd.DataFrame,
    outdir: str,
    *,
    y_col: str,
    yerr_col: str,
    ylabel: str,
    title: str,
    filename: str,
    log_y: bool = False,
    y_filter: pd.DataFrame | None = None,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    plot_df = y_filter if y_filter is not None else agg
    if plot_df.empty:
        raise ValueError(f"No data to plot for {filename}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for l_val, sub in plot_df.groupby("L"):
        l_int = int(l_val)
        style = L_PLOT_STYLE.get(l_int, {"color": "gray", "marker": "o"})
        color = style["color"]
        ax.errorbar(
            sub["epsilon"],
            sub[y_col],
            yerr=sub[yerr_col],
            fmt=f"{style['marker']}-",
            color=color,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.2,
            capsize=3,
            label=f"L = {l_int}",
        )
    if y_col == "m_mean":
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both" if log_y else "major", alpha=0.3)
    path = os.path.join(outdir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_chi_vs_epsilon(agg: pd.DataFrame, outdir: str, pooled: bool = False) -> None:
    suffix = " (pooled)" if pooled else ""
    ftag = "_pooled" if pooled else ""
    plot_df = agg[agg["chi_mean"] > 0].copy()
    _plot_l_curves_vs_epsilon(
        agg,
        outdir,
        y_col="chi_mean",
        yerr_col="chi_stderr",
        ylabel=r"$\chi$",
        title=r"Susceptibility vs $\varepsilon$" + suffix,
        filename=f"chi_vs_epsilon{ftag}.png",
        log_y=True,
        y_filter=plot_df,
    )


def plot_m_vs_epsilon(agg: pd.DataFrame, outdir: str, pooled: bool = False) -> None:
    suffix = " (pooled)" if pooled else ""
    ftag = "_pooled" if pooled else ""
    os.makedirs(outdir, exist_ok=True)

    plot_df = agg.copy()
    if "beta" not in plot_df.columns:
        plot_df["beta"] = 1.0
    plot_df["beta_epsilon"] = _beta_epsilon(plot_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    for l_val, sub in plot_df.sort_values("beta_epsilon").groupby("L"):
        l_int = int(l_val)
        style = L_PLOT_STYLE.get(l_int, {"color": "gray", "marker": "o"})
        color = style["color"]
        ax.errorbar(
            sub["beta_epsilon"],
            sub["abs_m_mean"],
            yerr=sub["abs_m_mean_stderr"],
            fmt=f"{style['marker']}-",
            color=color,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.2,
            capsize=3,
            label=f"L = {l_int}",
        )
    ax.set_xlabel(r"$\beta\varepsilon$")
    ax.set_ylabel(r"$\langle |m| \rangle$")
    ax.set_title(r"$\langle |m| \rangle$ vs $\beta\varepsilon$" + suffix)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    png_path = os.path.join(outdir, f"abs_m_vs_beta_epsilon{ftag}.png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {png_path}")

    csv_df = plot_df[["L", "epsilon", "beta", "beta_epsilon", "abs_m_mean", "abs_m_mean_stderr"]].copy()
    csv_df = csv_df.sort_values(["L", "beta_epsilon"])
    csv_path = os.path.join(outdir, f"abs_m_vs_beta_epsilon{ftag}.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


def plot_heat_capacity_vs_epsilon(agg: pd.DataFrame, outdir: str, pooled: bool = False) -> None:
    if "c_mean" not in agg.columns or agg["c_mean"].isna().all():
        print("Skipping heat capacity plot — no energy data found.")
        return
    suffix = " (pooled)" if pooled else ""
    ftag = "_pooled" if pooled else ""
    _plot_l_curves_vs_epsilon(
        agg,
        outdir,
        y_col="c_mean",
        yerr_col="c_stderr",
        ylabel=r"$c(T, L)$",
        title=r"Heat capacity vs $\varepsilon$" + suffix,
        filename=f"heat_capacity_vs_epsilon{ftag}.png",
    )


def plot_binder_vs_epsilon(agg: pd.DataFrame, outdir: str, pooled: bool = False) -> None:
    suffix = " (pooled)" if pooled else ""
    ftag = "_pooled" if pooled else ""
    _plot_l_curves_vs_epsilon(
        agg,
        outdir,
        y_col="u4",
        yerr_col="u4_err",
        ylabel=r"$U_4(T, L)$",
        title=r"Binder cumulant vs $\varepsilon$" + suffix,
        filename=f"binder_vs_epsilon{ftag}.png",
    )


def _beta_epsilon(df: pd.DataFrame) -> pd.Series:
    """Sweep axis βε; beta defaults to 1.0 when absent (legacy aggregates)."""
    beta = df["beta"] if "beta" in df.columns else 1.0
    return beta * df["epsilon"]


def plot_fig7_panels(agg_c: pd.DataFrame, agg_chi: pd.DataFrame, outdir: str) -> None:
    """Reproduce Fig. 7 of Kumar & Dasgupta (PRE 102, 052111): a two-panel figure with
    (a) specific heat c vs βε (linear-y, per-trajectory aggregation) and
    (b) susceptibility χ vs βε (log-y, pooled replica samples), each a family of L curves.
    """
    os.makedirs(outdir, exist_ok=True)

    have_c = "c_mean" in agg_c.columns and agg_c["c_mean"].notna().any()
    if not have_c:
        print("Fig 7 panel (a): no energy/heat-capacity data — plotting χ panel only.")

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 8.5), sharex=True)
    ax_c, ax_chi = axes

    def _series(ax, sub, x, y_col, yerr_col, color, style):
        ax.errorbar(
            x, sub[y_col], yerr=sub[yerr_col],
            fmt=f"{style['marker']}-", color=color,
            markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.2,
            capsize=3, markersize=5, linewidth=1.0, label=f"L = {int(sub['L'].iloc[0])}",
        )

    for l_val, sub in agg_c.sort_values("epsilon").groupby("L"):
        style = L_PLOT_STYLE.get(int(l_val), {"color": "gray", "marker": "o"})
        color = style["color"]
        c_sub = sub[sub["c_mean"].notna()]
        if have_c and not c_sub.empty:
            _series(ax_c, c_sub, _beta_epsilon(c_sub), "c_mean", "c_stderr", color, style)

    for l_val, sub in agg_chi.sort_values("epsilon").groupby("L"):
        style = L_PLOT_STYLE.get(int(l_val), {"color": "gray", "marker": "o"})
        color = style["color"]
        chi_sub = sub[sub["chi_mean"] > 0]
        if not chi_sub.empty:
            _series(ax_chi, chi_sub, _beta_epsilon(chi_sub), "chi_mean", "chi_stderr",
                    color, style)

    ax_c.set_ylabel(r"$c(\beta\varepsilon, L)$")
    ax_c.set_title(r"(a) Specific heat vs $\beta\varepsilon$ (per replica)", loc="left", fontsize=11)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=8, ncol=2)

    ax_chi.set_yscale("log")
    ax_chi.set_xlabel(r"$\beta\varepsilon$")
    ax_chi.set_ylabel(r"$\chi(\beta\varepsilon, L)$")
    ax_chi.set_title(r"(b) Susceptibility vs $\beta\varepsilon$ (log scale, pooled)",
                     loc="left", fontsize=11)
    ax_chi.grid(True, which="both", alpha=0.3)
    ax_chi.legend(fontsize=8, ncol=2)

    path = os.path.join(outdir, "fig7_c_chi_vs_beta_epsilon.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")

    c_cols = ["L", "epsilon", "beta", "c_mean", "c_stderr"]
    chi_cols = ["L", "epsilon", "beta", "chi_mean", "chi_stderr"]
    c_out = agg_c[[c for c in c_cols if c in agg_c.columns]].copy()
    chi_out = agg_chi[[c for c in chi_cols if c in agg_chi.columns]].copy()
    if "beta" not in c_out.columns:
        c_out["beta"] = 1.0
    if "beta" not in chi_out.columns:
        chi_out["beta"] = 1.0
    c_out = c_out.rename(columns={"c_mean": "c_mean_per_traj", "c_stderr": "c_stderr_per_traj"})
    chi_out = chi_out.rename(columns={"chi_mean": "chi_mean_pooled", "chi_stderr": "chi_stderr_pooled"})
    csv_df = pd.merge(c_out, chi_out, on=["L", "epsilon", "beta"], how="outer")
    csv_df["beta_epsilon"] = csv_df["beta"] * csv_df["epsilon"]
    csv_df = csv_df.sort_values(["L", "epsilon"])
    csv_path = os.path.join(outdir, "fig7_c_chi_vs_beta_epsilon.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


def draw_peak_chi_figure(
    peaks: pd.DataFrame, outdir: str, pooled: bool = False, fit_min_L: float = 0.0
) -> None:
    suffix = " (pooled)" if pooled else ""
    ftag = "_pooled" if pooled else ""
    os.makedirs(outdir, exist_ok=True)

    L_vals = peaks["L"].to_numpy(dtype=float)
    chi_vals = peaks["chi_mean"].to_numpy(dtype=float)
    L_fine = np.geomspace(L_vals.min(), L_vals.max(), 200)

    # Points used for the slope fit (drop small-L corrections-to-scaling if requested).
    fit_mask = L_vals >= fit_min_L
    if fit_mask.sum() < 2:
        raise ValueError(f"--fit-min-L={fit_min_L} leaves <2 points to fit.")

    fig, ax = plt.subplots(figsize=(6, 5))
    # Fitted points solid, excluded points hollow so it's obvious what drove the slope.
    ax.loglog(L_vals[fit_mask], chi_vals[fit_mask], "o", markersize=6, color="black",
              zorder=3, label="simulation (fitted)")
    if (~fit_mask).any():
        ax.loglog(L_vals[~fit_mask], chi_vals[~fit_mask], "o", markersize=6,
                  markerfacecolor="none", markeredgecolor="black", zorder=3,
                  label=f"simulation (excluded, L<{fit_min_L:g})")

    # Reference line from Kumar & Dasgupta (2020): A=0.095, gamma/nu=1.75
    REF_A, REF_GNU = 0.095, 1.75
    ax.loglog(
        L_fine, REF_A * L_fine**REF_GNU,
        "-", color="red", linewidth=1.5,
        label=rf"$A={REF_A}$,  $\gamma/\nu={REF_GNU}$ (K&D 2020)",
    )

    # Best fit to our data (over the fitted subset only)
    log_slope, log_intercept = np.polyfit(np.log(L_vals[fit_mask]), np.log(chi_vals[fit_mask]), 1)
    fit_A = np.exp(log_intercept)
    fit_label = rf"$A={fit_A:.3f}$,  $\gamma/\nu={log_slope:.3f}$ (fit"
    fit_label += rf", $L\geq{fit_min_L:g}$)" if fit_min_L > 0 else ")"
    ax.loglog(L_fine, fit_A * L_fine**log_slope, "--", color="blue", linewidth=1.5,
              label=fit_label)
    print(f"[peak_chi_vs_L] fit gamma/nu={log_slope:.4f}, A={fit_A:.4f} "
          f"over L={sorted(L_vals[fit_mask].astype(int))}")

    ax.legend(fontsize=9)

    ax.set_xlabel("L")
    ax.set_ylabel(r"$\chi^{\mathrm{max}}(L)$")
    ax.set_ylim(10, 1500)
    ax.set_title(r"Peak susceptibility vs $L$" + suffix)
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(outdir, f"peak_chi_vs_L{ftag}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_peak_chi_vs_L(
    agg: pd.DataFrame, outdir: str, pooled: bool = False, fit_min_L: float = 0.0
) -> None:
    ftag = "_pooled" if pooled else ""
    os.makedirs(outdir, exist_ok=True)
    peaks = (
        agg.loc[agg.groupby("L")["chi_mean"].idxmax()]
        .sort_values("L")
    )
    draw_peak_chi_figure(peaks, outdir, pooled=pooled, fit_min_L=fit_min_L)
    csv_path = os.path.join(outdir, f"peak_chi_vs_L{ftag}.csv")
    peaks[["L", "epsilon", "chi_mean", "chi_stderr"]].to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


def print_moments_summary(agg: pd.DataFrame, warn_threshold: float = 0.1) -> None:
    peaks = agg.loc[agg.groupby("L")["chi_mean"].idxmax()].sort_values("L")
    has_counts = "chunks_per_replica" in agg.columns
    print("\n=== Moments at peak chi per L ===")
    hdr = f"{'L':>5}  {'eps_peak':>9}  {'chi':>9}  {'<m>':>8}  {'<m^2>':>8}  {'<m^4>':>10}"
    if has_counts:
        hdr += f"  {'chunks/rep':>10}  {'n_samp':>8}"
    print(hdr)
    print("-" * (len(hdr) + 2))
    warned = False
    for _, row in peaks.iterrows():
        m1 = row["m_mean"]
        flag = "  *** |<m>| far from 0 ***" if abs(m1) > warn_threshold else ""
        if flag:
            warned = True
        line = (f"{int(row['L']):>5}  {row['epsilon']:>9.4f}  {row['chi_mean']:>9.2f}  "
                f"{m1:>8.4f}  {row['m2_mean']:>8.4f}  {row['m4_mean']:>10.6f}")
        if has_counts:
            line += f"  {row['chunks_per_replica']:>10.0f}  {int(row['n_samples']):>8}"
        print(line + flag)
    if warned:
        print(f"\nWARNING: one or more L values have |<m>| > {warn_threshold} at peak chi.")
        print("  This may indicate replicas trapped in one well (check initial_fraction).\n")
    else:
        print()


def plot_m_histograms_at_peak(
    traj_df: pd.DataFrame, agg: pd.DataFrame, outdir: str, pooled: bool = False
) -> None:
    """Histogram of per-replica <m> at the peak chi epsilon for each L."""
    ftag = "_pooled" if pooled else ""
    os.makedirs(outdir, exist_ok=True)
    Ls = sorted(int(v) for v in agg["L"].unique())
    peaks = agg.loc[agg.groupby("L")["chi_mean"].idxmax()].set_index("L")

    fig, axes = plt.subplots(len(Ls), 1, figsize=(6, 2.5 * len(Ls)), squeeze=False)
    for i, L in enumerate(Ls):
        peak_eps = float(peaks.loc[L, "epsilon"])
        sub = traj_df[traj_df["L"] == L]
        eps_vals = sub["epsilon"].to_numpy()
        closest_eps = float(eps_vals[np.argmin(np.abs(eps_vals - peak_eps))])
        m_vals = sub[np.isclose(sub["epsilon"], closest_eps, atol=1e-6)]["m_mean"].to_numpy()

        ax = axes[i][0]
        color = L_PLOT_STYLE.get(L, {}).get("color", "gray")
        ax.hist(m_vals, bins=min(20, max(5, len(m_vals) // 3)), color=color, alpha=0.75, density=True)
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
        m1 = float(np.mean(m_vals))
        warn = "  ⚠ |<m>| far from 0" if abs(m1) > 0.1 else ""
        ax.set_title(
            rf"L={L},  $\varepsilon$={closest_eps:.4f}   $\langle m\rangle$={m1:.3f}{warn}",
            fontsize=9,
        )
        ax.set_xlabel(r"per-replica $\langle m \rangle$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"Per-replica $\langle m \rangle$ at peak $\chi$ (criticality)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(outdir, f"m_hist_at_peak{ftag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")
