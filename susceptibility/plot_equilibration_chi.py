"""
plot_equilibration_chi.py

Equilibration test: χ^max(L) recomputed from only the *first* f fraction of each
replica's time series, for f = 10%, 20%, …, 100%.

Rationale: if a run is equilibrated (stationary) the peak susceptibility does not
depend on how much of the trajectory you include — the first 10% of samples give the
same χ^max as the full run. So each L's curve should be flat across f. A curve that is
still rising or falling as f→1 means the estimator is still absorbing burn-in / drift,
i.e. that L had not equilibrated over the plotted window.

Produces one line per square size L (7 curves). x-axis = fraction of time series used,
y-axis = χ^max(L) = max over ε of the connected susceptibility.

χ is the same connected estimator used in plot_susceptibility (pooled path):
    χ = N·β·(⟨m²⟩ − ⟨|m|⟩²)
computed over the pooled, truncated samples of all replicas at each (L, ε).

Usage:
    python plot_equilibration_chi.py --results susceptibility_results/exact --outdir plots/exact
    python plot_equilibration_chi.py --results susceptibility_results --n-fractions 10
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from susceptibility_paths import find_susceptibility_csvs, read_susceptibility_csv
from plot_susceptibility import L_PLOT_STYLE, _load_traj_arrays, _jackknife


def _chi_stat(m: np.ndarray, N: int, beta: float) -> float:
    """Connected susceptibility of a pooled m sample (matches aggregate_pooled)."""
    return float(N * beta * (np.mean(m ** 2) - np.mean(np.abs(m)) ** 2))


def load_groups(results_dir: str) -> dict[tuple[int, float], list[dict]]:
    """Group every replica's raw m array (+ beta, N) by (L, ε)."""
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
            rec = _load_traj_arrays(ts_path, meta)
            if rec is not None and rec["m"].size > 0:
                groups[(rec["L"], rec["epsilon"])].append(rec)
    if not groups:
        raise FileNotFoundError("No timeseries files found — check that runs have completed.")
    return groups


def compute_equilibration_curves(
    groups: dict[tuple[int, float], list[dict]], fractions: np.ndarray
) -> pd.DataFrame:
    """For each L and fraction f, find χ^max over ε using the first f of each replica.

    Returns tidy rows: L, fraction, epsilon_peak, chi_max, chi_max_err, n_replicas.
    """
    rows: list[dict] = []
    Ls = sorted({key[0] for key in groups})

    for L in Ls:
        eps_keys = sorted(eps for (l_val, eps) in groups if l_val == L)
        for f in fractions:
            best: dict | None = None
            for eps in eps_keys:
                recs = groups[(L, eps)]
                beta, N = recs[0]["beta"], recs[0]["N"]
                # Truncate each replica to its own first-f prefix, then pool.
                trunc = [r["m"][: max(1, int(round(r["m"].size * f)))] for r in recs]
                trunc = [a for a in trunc if a.size > 0]
                if not trunc:
                    continue
                chi, chi_err = _jackknife(trunc, lambda a, N=N, beta=beta: _chi_stat(a, N, beta))
                if best is None or chi > best["chi_max"]:
                    best = {
                        "L": L,
                        "fraction": float(f),
                        "epsilon_peak": float(eps),
                        "chi_max": chi,
                        "chi_max_err": chi_err,
                        "n_replicas": len(recs),
                    }
            if best is not None:
                rows.append(best)
    return pd.DataFrame(rows)


def plot_equilibration_curves(df: pd.DataFrame, outdir: str, log_y: bool = False) -> None:
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for L_val, sub in df.groupby("L"):
        L = int(L_val)  # type: ignore[arg-type]
        sub = sub.sort_values("fraction")
        style = L_PLOT_STYLE.get(L, {"color": "gray", "marker": "o"})
        color = style["color"]
        ax.errorbar(
            sub["fraction"] * 100.0,
            sub["chi_max"],
            yerr=sub["chi_max_err"],
            fmt=f"{style['marker']}-",
            color=color,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.2,
            capsize=3,
            label=f"L = {L}",
        )
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("fraction of time series used (first %)")
    ax.set_ylabel(r"$\chi^{\mathrm{max}}(L)$")
    ax.set_title(r"Equilibration test: $\chi^{\mathrm{max}}$ vs time-series fraction")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both" if log_y else "major", alpha=0.3)
    path = os.path.join(outdir, "max_chi_vs_time.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="χ^max vs time-series fraction (equilibration test)")
    parser.add_argument("--results", default="susceptibility_results")
    parser.add_argument("--outdir", default="plots/susceptibility")
    parser.add_argument(
        "--n-fractions", type=int, default=10,
        help="Number of evenly spaced prefixes from (1/n) to 1.0 (default 10 → 10%%,20%%,…,100%%).",
    )
    parser.add_argument("--log-y", action="store_true", help="Log-scale the χ^max axis.")
    args = parser.parse_args()

    fractions = np.arange(1, args.n_fractions + 1) / args.n_fractions
    groups = load_groups(args.results)
    df = compute_equilibration_curves(groups, fractions)
    if df.empty:
        raise SystemExit("No curves computed — no usable (L, ε) groups found.")

    plot_equilibration_curves(df, args.outdir, log_y=args.log_y)
    csv_path = os.path.join(args.outdir, "max_chi_vs_time.csv")
    df.sort_values(["L", "fraction"]).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
