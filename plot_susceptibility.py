"""
plot_susceptibility.py

Plot susceptibility χ vs ε, order parameter m vs ε, and peak χ vs L from susceptibility_data.csv files.

Usage:
    python plot_susceptibility.py
    python plot_susceptibility.py --results susceptibility_results --outdir plots/susceptibility
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from susceptibility_paths import read_susceptibility_csv

# Open markers and colors matched to reference χ vs control-parameter figure.
L_PLOT_STYLE: dict[int, dict[str, str]] = {
    16: {"color": "black", "marker": "o"},
    32: {"color": "red", "marker": "s"},
    48: {"color": "#90EE90", "marker": "^"},
    64: {"color": "blue", "marker": "D"},
    96: {"color": "cyan", "marker": "v"},
    128: {"color": "saddlebrown", "marker": "<"},
    256: {"color": "orange", "marker": ">"},
}


def collect_susceptibility_data(results_dir: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, "**", "susceptibility_data.csv")
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        raise FileNotFoundError(f"No susceptibility_data.csv under {results_dir}")

    frames = [pd.DataFrame(read_susceptibility_csv(p)) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    for col in ("L", "epsilon", "chi", "chi_err", "m_mean", "m_mean_err"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    def _stderr(series: pd.Series) -> float:
        return float(series.std(ddof=1) / np.sqrt(len(series))) if len(series) > 1 else 0.0

    rows = []
    for (l_val, eps), sub in df.groupby(["L", "epsilon"]):
        chi_stderr = float(sub["chi_err"].mean()) if "chi_err" in sub.columns else _stderr(sub["chi"])
        rows.append({
            "L": l_val,
            "epsilon": eps,
            "chi_mean": float(sub["chi"].mean()),
            "chi_stderr": chi_stderr,
            "m_mean": float(sub["m_mean"].mean()),
            "m_mean_stderr": float(sub["m_mean_err"].mean()) if "m_mean_err" in sub.columns else _stderr(sub["m_mean"]),
            "n_replicas": len(sub),
        })
    return pd.DataFrame(rows).sort_values(["L", "epsilon"])


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


def plot_chi_vs_epsilon(agg: pd.DataFrame, outdir: str) -> None:
    plot_df = agg[agg["chi_mean"] > 0].copy()
    _plot_l_curves_vs_epsilon(
        agg,
        outdir,
        y_col="chi_mean",
        yerr_col="chi_stderr",
        ylabel=r"$\chi$",
        title=r"Susceptibility vs $\varepsilon$",
        filename="chi_vs_epsilon.png",
        log_y=True,
        y_filter=plot_df,
    )


def plot_m_vs_epsilon(agg: pd.DataFrame, outdir: str) -> None:
    _plot_l_curves_vs_epsilon(
        agg,
        outdir,
        y_col="m_mean",
        yerr_col="m_mean_stderr",
        ylabel=r"$m$",
        title=r"Order parameter vs $\varepsilon$",
        filename="m_vs_epsilon.png",
    )


def plot_peak_chi_vs_L(agg: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    peaks = (
        agg.loc[agg.groupby("L")["chi_mean"].idxmax()]
        .sort_values("L")
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(peaks["L"], peaks["chi_mean"], "o-", markersize=6)
    ax.set_xlabel("L")
    ax.set_ylabel(r"max($\chi$)")
    ax.set_title(r"Peak susceptibility vs $L$")
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(outdir, "peak_chi_vs_L.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")

    csv_path = os.path.join(outdir, "peak_chi_vs_L.csv")
    peaks[["L", "epsilon", "chi_mean", "chi_stderr"]].to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot susceptibility campaign results")
    parser.add_argument("--results", default="susceptibility_results")
    parser.add_argument("--outdir", default="plots/susceptibility")
    args = parser.parse_args()

    df = collect_susceptibility_data(args.results)
    agg = aggregate(df)
    plot_chi_vs_epsilon(agg, args.outdir)
    plot_m_vs_epsilon(agg, args.outdir)
    plot_peak_chi_vs_L(agg, args.outdir)


if __name__ == "__main__":
    main()
