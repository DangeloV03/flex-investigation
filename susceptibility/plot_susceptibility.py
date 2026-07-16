"""
plot_susceptibility.py

Plot susceptibility χ vs ε, order parameter |m| vs ε, heat capacity c vs ε, and Binder U4 vs ε.

Reads m_timeseries_{id}.csv files (raw time series per replica) and computes all observables
from scratch, following the paper's ordering:
  1. time averages ⟨m⟩, ⟨|m|⟩, ⟨m²⟩, ⟨m⁴⟩, ⟨E_int⟩, ⟨E_int²⟩ per single trajectory
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
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from susceptibility_paths import find_susceptibility_csvs, read_susceptibility_csv

L_PLOT_STYLE: dict[int, dict[str, str]] = {
    16: {"color": "black", "marker": "o"},
    32: {"color": "red", "marker": "s"},
    48: {"color": "#90EE90", "marker": "^"},
    64: {"color": "blue", "marker": "D"},
    96: {"color": "cyan", "marker": "v"},
    128: {"color": "saddlebrown", "marker": "<"},
    256: {"color": "orange", "marker": ">"},
}


def _tail_slice(n: int, tail_fraction: float) -> slice:
    """Index slice keeping the last `tail_fraction` of an n-length series (>=1 sample)."""
    if tail_fraction >= 1.0:
        return slice(None)
    keep = max(1, int(round(n * tail_fraction)))
    return slice(n - keep, None)


def _compute_traj_stats(ts_path: str, meta: dict, tail_fraction: float = 1.0) -> dict | None:
    """Compute per-trajectory observables from a single m_timeseries CSV.

    tail_fraction < 1 keeps only the last fraction of each series (equilibrated tail).
    """
    if not os.path.isfile(ts_path):
        return None

    ts = pd.read_csv(ts_path)
    if ts.empty:
        return None
    ts = ts.iloc[_tail_slice(len(ts), tail_fraction)]

    beta = float(meta["beta"])
    mu = float(meta["mu"])
    delta_f = float(meta["delta_f"])
    L = int(float(meta["L"]))
    N = L * L

    m_arr = ts["m"].values.astype(float)
    m_mean = float(np.mean(m_arr))
    abs_m_mean = float(np.mean(np.abs(m_arr)))
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
        "abs_m_mean": abs_m_mean,
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
        epsilon = float(meta["epsilon"])
        result["c"] = (e2_mean - e_mean ** 2) / (N * T ** 2 * abs(epsilon))

    return result


def aggregate(results_dir: str, tail_fraction: float = 1.0) -> pd.DataFrame:
    """
    Scan for susceptibility_data.csv files under results_dir, load each replica's
    m_timeseries CSV, compute per-trajectory observables, then average over replicas
    grouped by (L, ε). tail_fraction < 1 uses only each series' last fraction.
    """
    paths = find_susceptibility_csvs(results_dir)
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
            stats = _compute_traj_stats(ts_path, meta, tail_fraction=tail_fraction)
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
            "abs_m_mean": float(sub["abs_m_mean"].mean()),
            "abs_m_mean_stderr": _stderr(sub["abs_m_mean"]),
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

    return pd.DataFrame(rows_agg).sort_values(["L", "epsilon"]), pd.DataFrame(traj_records)


def _load_traj_arrays(ts_path: str, meta: dict, tail_fraction: float = 1.0) -> dict | None:
    """Load one trajectory's raw per-chunk m (and recovered E_int) arrays.

    tail_fraction < 1 keeps only the last fraction of each series (equilibrated tail).
    """
    if not os.path.isfile(ts_path):
        return None
    ts = pd.read_csv(ts_path)
    if ts.empty or "m" not in ts.columns:
        return None
    ts = ts.iloc[_tail_slice(len(ts), tail_fraction)]

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


def aggregate_pooled(results_dir: str, tail_fraction: float = 1.0) -> pd.DataFrame:
    """Pool every replica's samples per (L, ε), then compute χ, c, U4 once.

    Workaround for non-ergodic short runs: combining replicas reconstructs the full
    P(m) the per-trajectory estimator misses. Errors are leave-one-replica-out jackknife.
    tail_fraction < 1 pools only each series' last fraction (equilibrated tail).
    """
    paths = find_susceptibility_csvs(results_dir)
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
            rec = _load_traj_arrays(ts_path, meta, tail_fraction=tail_fraction)
            if rec:
                groups[(rec["L"], rec["epsilon"])].append(rec)

    if not groups:
        raise FileNotFoundError("No timeseries files found — check that runs have completed.")

    rows = []
    traj_rows: list[dict] = []
    for (l_val, eps), recs in groups.items():
        beta = recs[0]["beta"]
        N = recs[0]["N"]
        m_arrays = [r["m"] for r in recs]
        pooled_m = np.concatenate(m_arrays)
        for m_arr in m_arrays:
            traj_rows.append({
                "L": int(l_val),
                "epsilon": float(eps),
                "m_mean": float(np.mean(m_arr)),
                "m2_mean": float(np.mean(m_arr ** 2)),
                "m4_mean": float(np.mean(m_arr ** 4)),
            })

        chi, chi_err = _jackknife(
            m_arrays, lambda a, N=N, beta=beta: N * beta * (np.mean(a ** 2) - np.mean(np.abs(a)) ** 2)
        )
        m_mean, m_mean_err = _jackknife(m_arrays, lambda a: float(np.mean(a)))
        abs_m_mean, abs_m_mean_err = _jackknife(
            m_arrays, lambda a: float(np.mean(np.abs(a)))
        )
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
            "abs_m_mean": abs_m_mean,
            "abs_m_mean_stderr": abs_m_mean_err,
            "m2_mean": float(np.mean(pooled_m ** 2)),
            "m4_mean": float(np.mean(pooled_m ** 4)),
            "u4": u4,
            "u4_err": u4_err,
            "n_replicas": len(recs),
            "n_samples": int(pooled_m.size),
            "chunks_per_replica": float(np.mean([a.size for a in m_arrays])),
        }

        e_arrays = [r["e_int"] for r in recs if r["e_int"] is not None]
        if e_arrays:
            # Match the per-trajectory convention: c = (⟨E²⟩-⟨E⟩²)/(N·T²·|ε|), T = 1/β.
            T = 1.0 / beta
            denom = N * T ** 2 * abs(float(eps))
            c, c_err = _jackknife(
                e_arrays, lambda a, denom=denom: (np.mean(a ** 2) - np.mean(a) ** 2) / denom
            )
            row["c_mean"] = c
            row["c_stderr"] = c_err
        else:
            row["c_mean"] = float("nan")
            row["c_stderr"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["L", "epsilon"]), pd.DataFrame(traj_rows)


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
        y_col="abs_m_mean",
        yerr_col="abs_m_mean_stderr",
        ylabel=r"$\langle |m| \rangle$",
        title=r"$\langle |m| \rangle$ vs $\varepsilon$" + suffix,
        filename=f"abs_m_vs_epsilon{ftag}.png",
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


def plot_fig7_panels(agg: pd.DataFrame, outdir: str, pooled: bool = False) -> None:
    """Reproduce Fig. 7 of Kumar & Dasgupta (PRE 102, 052111): a two-panel figure with
    (a) specific heat c vs ε (linear-y) and (b) susceptibility χ vs ε (log-y), each a
    family of L curves. Here ε plays the role of temperature T in the paper.
    """
    ftag = "_pooled" if pooled else ""
    suffix = " (pooled)" if pooled else ""
    os.makedirs(outdir, exist_ok=True)

    have_c = "c_mean" in agg.columns and agg["c_mean"].notna().any()
    if not have_c:
        print("Fig 7 panel (a): no energy/heat-capacity data — plotting χ panel only.")

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 8.5), sharex=True)
    ax_c, ax_chi = axes

    def _series(ax, sub, y_col, yerr_col, color, style):
        ax.errorbar(
            sub["epsilon"], sub[y_col], yerr=sub[yerr_col],
            fmt=f"{style['marker']}-", color=color,
            markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.2,
            capsize=3, markersize=5, linewidth=1.0, label=f"L = {int(sub['L'].iloc[0])}",
        )

    for l_val, sub in agg.sort_values("epsilon").groupby("L"):
        style = L_PLOT_STYLE.get(int(l_val), {"color": "gray", "marker": "o"})
        color = style["color"]
        if have_c:
            c_sub = sub[sub["c_mean"].notna()]
            if not c_sub.empty:
                _series(ax_c, c_sub, "c_mean", "c_stderr", color, style)
        chi_sub = sub[sub["chi_mean"] > 0]
        if not chi_sub.empty:
            _series(ax_chi, chi_sub, "chi_mean", "chi_stderr", color, style)

    ax_c.set_ylabel(r"$c(\varepsilon, L)$")
    ax_c.set_title(r"(a) Specific heat vs $\varepsilon$" + suffix, loc="left", fontsize=11)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=8, ncol=2)

    ax_chi.set_yscale("log")
    ax_chi.set_xlabel(r"$\varepsilon$")
    ax_chi.set_ylabel(r"$\chi(\varepsilon, L)$")
    ax_chi.set_title(r"(b) Susceptibility vs $\varepsilon$ (log scale)" + suffix,
                     loc="left", fontsize=11)
    ax_chi.grid(True, which="both", alpha=0.3)
    ax_chi.legend(fontsize=8, ncol=2)

    path = os.path.join(outdir, f"fig7_c_chi_vs_epsilon{ftag}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def _draw_peak_chi_figure(
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
    _draw_peak_chi_figure(peaks, outdir, pooled=pooled, fit_min_L=fit_min_L)
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
    parser.add_argument(
        "--peak-only",
        action="store_true",
        help="Skip aggregation and replot peak_chi_vs_L from the existing CSV in --outdir.",
    )
    parser.add_argument(
        "--tail-fraction", type=float, default=1.0,
        help="Use only the last fraction of each replica's series (e.g. 0.2 = last 20%%, "
             "the equilibrated tail). Default 1.0 = full series. Point --outdir somewhere "
             "separate to keep tail plots apart from full-data plots.",
    )
    parser.add_argument(
        "--fig7-only",
        action="store_true",
        help="Only reproduce the Fig. 7 two-panel figure (c vs ε and χ vs ε) and exit. "
             "Skips the individual per-observable plots, peak-χ fit, and histograms.",
    )
    parser.add_argument(
        "--fit-min-L", type=float, default=0.0,
        help="Fit the χ^max vs L slope (γ/ν) using only L >= this value, excluding "
             "small-L corrections-to-scaling that bias the slope low. Default 0 = all L.",
    )
    args = parser.parse_args()
    if not 0.0 < args.tail_fraction <= 1.0:
        parser.error("--tail-fraction must be in (0, 1].")

    if args.peak_only:
        ftag = "_pooled" if args.pooled else ""
        csv_path = os.path.join(args.outdir, f"peak_chi_vs_L{ftag}.csv")
        peaks = pd.read_csv(csv_path)
        _draw_peak_chi_figure(peaks, args.outdir, pooled=args.pooled, fit_min_L=args.fit_min_L)
        return

    if args.tail_fraction < 1.0:
        print(f"Using last {args.tail_fraction:.0%} of each replica's series (equilibrated tail)")

    if args.pooled:
        print("Aggregation: POOLED (all replica samples combined before χ/c/U4)")
        agg, traj_df = aggregate_pooled(args.results, tail_fraction=args.tail_fraction)
    else:
        print("Aggregation: per-trajectory then averaged")
        agg, traj_df = aggregate(args.results, tail_fraction=args.tail_fraction)

    if args.fig7_only:
        plot_fig7_panels(agg, args.outdir, pooled=args.pooled)
        return

    print_moments_summary(agg)
    plot_fig7_panels(agg, args.outdir, pooled=args.pooled)
    plot_chi_vs_epsilon(agg, args.outdir, pooled=args.pooled)
    plot_m_vs_epsilon(agg, args.outdir, pooled=args.pooled)
    plot_binder_vs_epsilon(agg, args.outdir, pooled=args.pooled)
    plot_heat_capacity_vs_epsilon(agg, args.outdir, pooled=args.pooled)
    plot_peak_chi_vs_L(agg, args.outdir, pooled=args.pooled, fit_min_L=args.fit_min_L)
    plot_m_histograms_at_peak(traj_df, agg, args.outdir, pooled=args.pooled)


if __name__ == "__main__":
    main()
