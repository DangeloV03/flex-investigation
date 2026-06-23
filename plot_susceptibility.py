"""
plot_susceptibility.py

Plot susceptibility χ vs ε and peak χ vs L from susceptibility_data.csv files.

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


def collect_susceptibility_data(results_dir: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, "**", "susceptibility_data.csv")
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        raise FileNotFoundError(f"No susceptibility_data.csv under {results_dir}")

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["L", "epsilon"], as_index=False)
        .agg(
            chi_mean=("chi", "mean"),
            chi_stderr=("chi", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0),
            m_mean=("m_mean", "mean"),
            n_replicas=("chi", "count"),
        )
        .sort_values(["L", "epsilon"])
    )
    return grouped


def plot_chi_vs_epsilon(agg: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for l_val, sub in agg.groupby("L"):
        ax.errorbar(
            sub["epsilon"],
            sub["chi_mean"],
            yerr=sub["chi_stderr"],
            fmt="o-",
            capsize=3,
            label=f"L={int(l_val)}",
        )
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\chi$")
    ax.set_title(r"Susceptibility vs $\varepsilon$")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, "chi_vs_epsilon.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


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
    plot_peak_chi_vs_L(agg, args.outdir)


if __name__ == "__main__":
    main()
