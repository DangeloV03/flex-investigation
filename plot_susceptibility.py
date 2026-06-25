"""
plot_susceptibility.py

Plot susceptibility χ vs ε, order parameter m vs ε, heat capacity c vs ε, and Binder U4 vs ε.

Reads m_timeseries_{id}.csv files (raw time series per replica) and computes all observables
from scratch, following the paper's ordering:
  1. time averages ⟨m⟩, ⟨m²⟩, ⟨m⁴⟩, ⟨E_int⟩, ⟨E_int²⟩ per single trajectory
  2. per-trajectory observables: χ, c, U4
  3. average over replicas per (L, ε)

E_interact is recovered from the stored total energy and densities:
  e_interact = e_total − e_chem
  e_chem     = −β·μ·N·ρ_B − β·(μ+Δf)·N·ρ_I

Usage:
    python plot_susceptibility.py
    python plot_susceptibility.py --results susceptibility_results/exact --outdir plots/exact
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from susceptibility_paths import read_susceptibility_csv

L_PLOT_STYLE: dict[int, dict[str, str]] = {
    16: {"color": "black", "marker": "o"},
    32: {"color": "red", "marker": "s"},
    48: {"color": "#90EE90", "marker": "^"},
    64: {"color": "blue", "marker": "D"},
    96: {"color": "cyan", "marker": "v"},
    128: {"color": "saddlebrown", "marker": "<"},
    256: {"color": "orange", "marker": ">"},
}


def _compute_traj_stats(ts_path: str, meta: dict) -> dict | None:
    """Compute per-trajectory observables from a single m_timeseries CSV."""
    if not os.path.isfile(ts_path):
        return None

    ts = pd.read_csv(ts_path)
    if ts.empty:
        return None

    beta = float(meta["beta"])
    mu = float(meta["mu"])
    delta_f = float(meta["delta_f"])
    L = int(float(meta["L"]))
    N = L * L

    m_arr = ts["m"].values.astype(float)
    m_mean = float(np.mean(m_arr))
    m2_mean = float(np.mean(m_arr ** 2))
    m4_mean = float(np.mean(m_arr ** 4))

    chi = N * beta * (m2_mean - m_mean ** 2)
    u4 = 1.0 - m4_mean / (3.0 * m2_mean ** 2) if m2_mean != 0 else float("nan")

    result: dict = {
        "L": L,
        "epsilon": float(meta["epsilon"]),
        "m_mean": m_mean,
        "m2_mean": m2_mean,
        "m4_mean": m4_mean,
        "chi": chi,
        "u4": u4,
    }

    # Heat capacity from interaction energy only (strips out chemical potential terms).
    if (
        "energy" in ts.columns
        and "rho_bonding" in ts.columns
        and "rho_inert" in ts.columns
    ):
        rho_B = ts["rho_bonding"].values.astype(float)
        rho_I = ts["rho_inert"].values.astype(float)
        e_total = ts["energy"].values.astype(float)
        e_chem = -beta * mu * N * rho_B - beta * (mu + delta_f) * N * rho_I
        e_int = e_total - e_chem
        e_int_mean = float(np.mean(e_int))
        e_int2_mean = float(np.mean(e_int ** 2))
        result["c"] = (e_int2_mean - e_int_mean ** 2) / N

    return result


def aggregate(results_dir: str) -> pd.DataFrame:
    """
    Scan for susceptibility_data.csv files under results_dir, load each replica's
    m_timeseries CSV, compute per-trajectory observables, then average over replicas
    grouped by (L, ε).
    """
    pattern = os.path.join(results_dir, "**", "susceptibility_data.csv")
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        raise FileNotFoundError(f"No susceptibility_data.csv under {results_dir}")

    traj_records: list[dict] = []
    for csv_path in paths:
        dirpath = os.path.dirname(csv_path)
        meta_rows = read_susceptibility_csv(csv_path)
        for meta in meta_rows:
            run_id = str(meta.get("id", "")).strip()
            if not run_id:
                continue
            ts_path = os.path.join(dirpath, f"m_timeseries_{run_id}.csv")
            stats = _compute_traj_stats(ts_path, meta)
            if stats:
                traj_records.append(stats)

    if not traj_records:
        raise FileNotFoundError(
            "No timeseries files found — check that runs have completed."
        )

    def _stderr(s: pd.Series) -> float:
        return float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0

    df = pd.DataFrame(traj_records)
    rows_agg = []
    for (l_val, eps), sub in df.groupby(["L", "epsilon"]):
        row: dict = {
            "L": int(l_val),
            "epsilon": float(eps),
            "chi_mean": float(sub["chi"].mean()),
            "chi_stderr": _stderr(sub["chi"]),
            "m_mean": float(sub["m_mean"].mean()),
            "m_mean_stderr": _stderr(sub["m_mean"]),
            "m2_mean": float(sub["m2_mean"].mean()),
            "m4_mean": float(sub["m4_mean"].mean()),
            "u4": float(sub["u4"].mean()),
            "u4_err": _stderr(sub["u4"]),
            "n_replicas": int(len(sub)),
        }
        if "c" in sub.columns and sub["c"].notna().any():
            row["c_mean"] = float(sub["c"].mean())
            row["c_stderr"] = _stderr(sub["c"])
        else:
            row["c_mean"] = float("nan")
            row["c_stderr"] = float("nan")
        rows_agg.append(row)

    return pd.DataFrame(rows_agg).sort_values(["L", "epsilon"])


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


def plot_heat_capacity_vs_epsilon(agg: pd.DataFrame, outdir: str) -> None:
    if "c_mean" not in agg.columns or agg["c_mean"].isna().all():
        print("Skipping heat capacity plot — no energy data found.")
        return
    _plot_l_curves_vs_epsilon(
        agg,
        outdir,
        y_col="c_mean",
        yerr_col="c_stderr",
        ylabel=r"$c(T, L)$",
        title=r"Heat capacity vs $\varepsilon$",
        filename="heat_capacity_vs_epsilon.png",
    )


def plot_binder_vs_epsilon(agg: pd.DataFrame, outdir: str) -> None:
    _plot_l_curves_vs_epsilon(
        agg,
        outdir,
        y_col="u4",
        yerr_col="u4_err",
        ylabel=r"$U_4(T, L)$",
        title=r"Binder cumulant vs $\varepsilon$",
        filename="binder_vs_epsilon.png",
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

    agg = aggregate(args.results)
    plot_chi_vs_epsilon(agg, args.outdir)
    plot_m_vs_epsilon(agg, args.outdir)
    plot_binder_vs_epsilon(agg, args.outdir)
    plot_heat_capacity_vs_epsilon(agg, args.outdir)
    plot_peak_chi_vs_L(agg, args.outdir)


if __name__ == "__main__":
    main()
