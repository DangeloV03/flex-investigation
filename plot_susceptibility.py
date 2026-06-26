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
from collections import defaultdict

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

    # χ = (1/NT)(⟨M²⟩-⟨|M|⟩²) with M = N·m, |M| = N·|m|, T = 1/β.
    # Subtracting ⟨|M|⟩² (not ⟨M⟩²) removes the between-well m₀² term that flipping
    # small-L runs would otherwise pick up — the connected (FSS) susceptibility.
    M_arr = N * m_arr
    chi = beta / N * (float(np.mean(M_arr ** 2)) - float(np.mean(np.abs(M_arr))) ** 2)
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

    # c(T,L) = (1/NT²)(⟨E²⟩ - ⟨E⟩²).  At μ=2ε the Nm terms in e_interact and
    # e_chem cancel exactly so E_total is spin-invariant; use the stored energy directly.
    if "energy" in ts.columns:
        e_total = ts["energy"].values.astype(float)
        e2_mean = float(np.mean(e_total ** 2))
        e_mean  = float(np.mean(e_total))
        T = 1.0 / beta
        result["c"] = (e2_mean - e_mean ** 2) / (N * T ** 2)

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


def _load_traj_arrays(ts_path: str, meta: dict) -> dict | None:
    """Load one trajectory's raw per-chunk m (and recovered E_int) arrays."""
    if not os.path.isfile(ts_path):
        return None
    ts = pd.read_csv(ts_path)
    if ts.empty or "m" not in ts.columns:
        return None

    beta = float(meta["beta"])
    mu = float(meta["mu"])
    delta_f = float(meta["delta_f"])
    L = int(float(meta["L"]))
    N = L * L

    m = ts["m"].to_numpy(float)
    e_int = None
    if {"energy", "rho_bonding", "rho_inert"}.issubset(ts.columns):
        rho_B = ts["rho_bonding"].to_numpy(float)
        rho_I = ts["rho_inert"].to_numpy(float)
        e_total = ts["energy"].to_numpy(float)
        e_chem = -beta * mu * N * rho_B - beta * (mu + delta_f) * N * rho_I
        e_int = e_total - e_chem

    return {"L": L, "epsilon": float(meta["epsilon"]), "beta": beta, "N": N, "m": m, "e_int": e_int}


def _jackknife(arrays: list[np.ndarray], stat_fn) -> tuple[float, float]:
    """Leave-one-replica-out jackknife of a pooled statistic.

    stat_fn maps a pooled sample array -> scalar. Returns (full_estimate, stderr),
    where the error reflects between-replica variation — the right scale here.
    """
    n = len(arrays)
    full = float(stat_fn(np.concatenate(arrays)))
    if n < 2:
        return full, 0.0
    partials = np.array([
        float(stat_fn(np.concatenate([arrays[j] for j in range(n) if j != i])))
        for i in range(n)
    ])
    mean = partials.mean()
    err = float(np.sqrt((n - 1) / n * np.sum((partials - mean) ** 2)))
    return full, err


def aggregate_pooled(results_dir: str) -> pd.DataFrame:
    """Pool every replica's samples per (L, ε), then compute χ, c, U4 once.

    Workaround for non-ergodic short runs: combining replicas reconstructs the full
    P(m) the per-trajectory estimator misses. Errors are leave-one-replica-out jackknife.
    """
    pattern = os.path.join(results_dir, "**", "susceptibility_data.csv")
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        raise FileNotFoundError(f"No susceptibility_data.csv under {results_dir}")

    groups: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for csv_path in paths:
        dirpath = os.path.dirname(csv_path)
        for meta in read_susceptibility_csv(csv_path):
            run_id = str(meta.get("id", "")).strip()
            if not run_id:
                continue
            ts_path = os.path.join(dirpath, f"m_timeseries_{run_id}.csv")
            rec = _load_traj_arrays(ts_path, meta)
            if rec:
                groups[(rec["L"], rec["epsilon"])].append(rec)

    if not groups:
        raise FileNotFoundError("No timeseries files found — check that runs have completed.")

    rows = []
    for (l_val, eps), recs in groups.items():
        beta = recs[0]["beta"]
        N = recs[0]["N"]
        m_arrays = [r["m"] for r in recs]
        pooled_m = np.concatenate(m_arrays)

        chi, chi_err = _jackknife(
            m_arrays, lambda a, N=N, beta=beta: N * beta * (np.mean(a ** 2) - np.mean(a) ** 2)
        )
        m_mean, m_mean_err = _jackknife(m_arrays, lambda a: float(np.mean(a)))
        u4, u4_err = _jackknife(
            m_arrays,
            lambda a: 1.0 - np.mean(a ** 4) / (3.0 * np.mean(a ** 2) ** 2)
            if np.mean(a ** 2) != 0 else float("nan"),
        )

        row: dict = {
            "L": int(l_val),
            "epsilon": float(eps),
            "chi_mean": chi,
            "chi_stderr": chi_err,
            "m_mean": m_mean,
            "m_mean_stderr": m_mean_err,
            "m2_mean": float(np.mean(pooled_m ** 2)),
            "m4_mean": float(np.mean(pooled_m ** 4)),
            "u4": u4,
            "u4_err": u4_err,
            "n_replicas": len(recs),
        }

        e_arrays = [r["e_int"] for r in recs if r["e_int"] is not None]
        if e_arrays:
            c, c_err = _jackknife(e_arrays, lambda a, N=N: (np.mean(a ** 2) - np.mean(a) ** 2) / N)
            row["c_mean"] = c
            row["c_stderr"] = c_err
        else:
            row["c_mean"] = float("nan")
            row["c_stderr"] = float("nan")
        rows.append(row)

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
    _plot_l_curves_vs_epsilon(
        agg,
        outdir,
        y_col="m_mean",
        yerr_col="m_mean_stderr",
        ylabel=r"$m$",
        title=r"Order parameter vs $\varepsilon$" + suffix,
        filename=f"m_vs_epsilon{ftag}.png",
    )


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


def plot_peak_chi_vs_L(agg: pd.DataFrame, outdir: str, pooled: bool = False) -> None:
    suffix = " (pooled)" if pooled else ""
    ftag = "_pooled" if pooled else ""
    os.makedirs(outdir, exist_ok=True)
    peaks = (
        agg.loc[agg.groupby("L")["chi_mean"].idxmax()]
        .sort_values("L")
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(peaks["L"], peaks["chi_mean"], "o-", markersize=6)
    ax.set_xlabel("L")
    ax.set_ylabel(r"max($\chi$)")
    ax.set_title(r"Peak susceptibility vs $L$" + suffix)
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(outdir, f"peak_chi_vs_L{ftag}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")

    csv_path = os.path.join(outdir, f"peak_chi_vs_L{ftag}.csv")
    peaks[["L", "epsilon", "chi_mean", "chi_stderr"]].to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot susceptibility campaign results")
    parser.add_argument("--results", default="susceptibility_results")
    parser.add_argument("--outdir", default="plots/susceptibility")
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="Pool all replica samples per (L, ε) before computing χ/c/U4 "
        "(vs per-trajectory then averaged). Writes *_pooled.png alongside the originals.",
    )
    args = parser.parse_args()

    if args.pooled:
        print("Aggregation: POOLED (all replica samples combined before χ/c/U4)")
        agg = aggregate_pooled(args.results)
    else:
        print("Aggregation: per-trajectory then averaged")
        agg = aggregate(args.results)

    plot_chi_vs_epsilon(agg, args.outdir, pooled=args.pooled)
    plot_m_vs_epsilon(agg, args.outdir, pooled=args.pooled)
    plot_binder_vs_epsilon(agg, args.outdir, pooled=args.pooled)
    plot_heat_capacity_vs_epsilon(agg, args.outdir, pooled=args.pooled)
    plot_peak_chi_vs_L(agg, args.outdir, pooled=args.pooled)


if __name__ == "__main__":
    main()
