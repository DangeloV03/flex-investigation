"""
plot_equilibration_chi.py

Two equilibration diagnostics from the raw m(t) time series.

1. Expanding-fraction χ^max(L) (``--n-fractions``): χ^max recomputed from a fraction
   f of each replica's series, for f = 10%, 20%, …, 100%. If a run is equilibrated the
   peak susceptibility does not depend on how much of the trajectory you include, so
   each L's curve is flat across f. Two versions are written:
     * max_chi_vs_time.png  — the *first* f% (prefix). Starts contaminated by any
       burn-in and settles as more data is added.
     * max_chi_from_back.png — the *last* f% (suffix), accumulating from the
       equilibrated tail. Flat across f until it starts pulling in early burn-in;
       a rise only near f→1 pinpoints how much of the head is still un-equilibrated.

2. Rolling-window χ vs time (``--window``): instead of a cumulative prefix, a
   fixed-width window slides along the trajectory and χ is recomputed inside each
   window. The x-axis is *actual MC time* (not "use up to t%"), the y-axis is χ
   evaluated locally around that time. Because early burn-in falls out the back of
   the window instead of being permanently averaged in, drift is much more visible:
   a flat line ⇒ stationary/equilibrated, a sloping or decaying line ⇒ not yet.

Both use the same connected estimator as plot_susceptibility (pooled path):
    χ = N·β·(⟨m²⟩ − ⟨|m|⟩²)
computed over the pooled samples of all replicas. For the rolling plot each L is
shown at its peak ε (the ε that maximises the full-series χ).

Usage:
    python plot_equilibration_chi.py --results susceptibility/results/exact --outdir plots/exact
    python plot_equilibration_chi.py --results susceptibility/results --n-fractions 10
    python plot_equilibration_chi.py --results susceptibility/results --window 100 --stride 20
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


def _mean_sem(arrays: list[np.ndarray], stat_fn) -> tuple[float, float]:
    """Mean and standard error of a per-replica statistic.

    Computes stat_fn on each replica's array separately, then returns the mean over
    replicas and the standard error of that mean (std/√N, ddof=1). Unlike the pooled
    jackknife this treats each replica as one independent measurement of the statistic.
    """
    vals = np.array([float(stat_fn(a)) for a in arrays if a.size > 0])
    n = vals.size
    if n == 0:
        return float("nan"), 0.0
    mean = float(vals.mean())
    sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, sem


def _meta_float(meta: dict, key: str) -> float:
    """Parse a meta field to float, returning 0.0 for missing/blank/non-numeric."""
    try:
        return float(str(meta.get(key, "")).strip())
    except (TypeError, ValueError):
        return 0.0


def load_groups(results_dir: str) -> dict[tuple[int, float], list[dict]]:
    """Group every replica's raw m array (+ beta, N) by (L, ε)."""
    paths = find_susceptibility_csvs(results_dir)
    if not paths:
        raise FileNotFoundError(f"No susceptibility_data.csv under {results_dir}")

    n_csv = len(paths)
    print(f"[load] {n_csv} susceptibility_data.csv files under {results_dir}", flush=True)
    groups: dict[tuple[int, float], list[dict]] = defaultdict(list)
    n_reps = 0
    for i, csv_path in enumerate(paths, 1):
        dirpath = os.path.dirname(csv_path)
        for meta in read_susceptibility_csv(csv_path):
            run_id = str(meta.get("id", "")).strip()
            if not run_id:
                continue
            ts_path = os.path.join(dirpath, f"m_timeseries_{run_id}.csv")
            rec = _load_traj_arrays(ts_path, meta)
            if rec is not None and rec["m"].size > 0:
                rec["chunk_time"] = _meta_float(meta, "prod_time") and (
                    _meta_float(meta, "prod_time") / (_meta_float(meta, "prod_chunks") or 1.0)
                )
                rec["eq_time"] = _meta_float(meta, "eq_time")
                groups[(rec["L"], rec["epsilon"])].append(rec)
                n_reps += 1
        if i % 25 == 0 or i == n_csv:
            print(f"[load] {i}/{n_csv} dirs read, {n_reps} replicas, "
                  f"{len(groups)} (L,ε) groups", flush=True)
    if not groups:
        raise FileNotFoundError("No timeseries files found — check that runs have completed.")
    return groups


def compute_equilibration_curves(
    groups: dict[tuple[int, float], list[dict]], fractions: np.ndarray,
    from_back: bool = False,
) -> pd.DataFrame:
    """For each L and fraction f, find χ^max over ε using a fraction f of each replica.

    from_back=False: use the *first* f (prefix). from_back=True: use the *last* f
    (suffix) — accumulating from the equilibrated tail, so a curve still rising as
    f→1 means the earlier data being pulled in is still carrying burn-in.

    Returns tidy rows: L, fraction, epsilon_peak, chi_max, chi_max_err, n_replicas.
    """
    rows: list[dict] = []
    Ls = sorted({key[0] for key in groups})
    tag = "suffix" if from_back else "prefix"
    print(f"[{tag}] computing χ^max over {len(fractions)} fractions for L={Ls}", flush=True)

    for L in Ls:
        print(f"[{tag}] L={L} …", flush=True)
        eps_keys = sorted(eps for (l_val, eps) in groups if l_val == L)
        for f in fractions:
            best: dict | None = None
            for eps in eps_keys:
                recs = groups[(L, eps)]
                beta, N = recs[0]["beta"], recs[0]["N"]
                # Truncate each replica to its own first-f prefix (or last-f suffix), then pool.
                k = lambda size: max(1, int(round(size * f)))
                if from_back:
                    trunc = [r["m"][-k(r["m"].size):] for r in recs]
                else:
                    trunc = [r["m"][: k(r["m"].size)] for r in recs]
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


def _peak_epsilon(groups: dict[tuple[int, float], list[dict]], L: int) -> float | None:
    """ε that maximises the full-series pooled χ for this L."""
    best_eps, best_chi = None, -np.inf
    for eps in sorted(eps for (l_val, eps) in groups if l_val == L):
        recs = groups[(L, eps)]
        beta, N = recs[0]["beta"], recs[0]["N"]
        chi = _chi_stat(np.concatenate([r["m"] for r in recs]), N, beta)
        if chi > best_chi:
            best_eps, best_chi = eps, chi
    return best_eps


def compute_rolling_chi(
    groups: dict[tuple[int, float], list[dict]], window: int, stride: int
) -> pd.DataFrame:
    """Rolling-window χ vs time for each L at its peak ε.

    A fixed-width window of ``window`` chunks slides (step ``stride``) along the
    trajectory. At each position χ = Nβ(⟨m²⟩−⟨|m|⟩²) is computed on each replica's
    windowed m; the plotted point is the mean over replicas and the bar is the
    standard error of that mean (std/√N). Window position is reported both as a
    chunk index (center) and, when chunk timing is known, as absolute MC time.

    Returns tidy rows: L, epsilon_peak, window, center_chunk, time, chi, chi_err,
    n_replicas.
    """
    rows: list[dict] = []
    Ls = sorted({key[0] for key in groups})
    print(f"[rolling] window={window} stride={stride} chunks for L={Ls}", flush=True)

    for L in Ls:
        eps = _peak_epsilon(groups, L)
        if eps is None:
            continue
        recs = groups[(L, eps)]
        beta, N = recs[0]["beta"], recs[0]["N"]
        chunk_time = recs[0].get("chunk_time") or 0.0
        eq_time = recs[0].get("eq_time") or 0.0
        T = min(r["m"].size for r in recs)  # common length across replicas
        w = min(window, T)
        if w < 1:
            continue
        starts = range(0, T - w + 1, max(1, stride))
        print(f"[rolling] L={L} peak ε={eps:.2f}, {len(recs)} replicas, "
              f"{len(starts)} windows", flush=True)
        for start in starts:
            wins = [r["m"][start : start + w] for r in recs]
            chi, chi_err = _mean_sem(wins, lambda a, N=N, beta=beta: _chi_stat(a, N, beta))
            center = start + w / 2.0
            # Absolute MC time at the window center (production begins after eq_time).
            time = eq_time + center * chunk_time if chunk_time else center
            rows.append({
                "L": L,
                "epsilon_peak": float(eps),
                "window": int(w),
                "center_chunk": float(center),
                "time": float(time),
                "chi": chi,
                "chi_err": chi_err,
                "n_replicas": len(recs),
            })
    return pd.DataFrame(rows)


def plot_rolling_chi(df: pd.DataFrame, outdir: str, log_y: bool = False) -> None:
    os.makedirs(outdir, exist_ok=True)
    has_time = bool(df["time"].to_numpy().any()) and not np.allclose(
        df["time"], df["center_chunk"]
    )
    xcol = "time" if has_time else "center_chunk"
    xlabel = "MC sweeps (production)" if has_time else "chunk index (window center)"

    fig, ax = plt.subplots(figsize=(8, 5))
    for L_val, sub in df.groupby("L"):
        L = int(L_val)  # type: ignore[arg-type]
        sub = sub.sort_values(xcol)
        style = L_PLOT_STYLE.get(L, {"color": "gray", "marker": "o"})
        color = style["color"]
        eps = sub["epsilon_peak"].iloc[0]
        ax.errorbar(
            sub[xcol],
            sub["chi"],
            yerr=sub["chi_err"],
            fmt=f"{style['marker']}-",
            color=color,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.2,
            markersize=4,
            capsize=2,
            elinewidth=0.8,
            label=f"L = {L} (ε={eps:.2f})",
        )
    if log_y:
        ax.set_yscale("log")
    w = int(df["window"].iloc[0]) if not df.empty else 0
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\chi$ (rolling window)")
    ax.set_title(rf"Rolling-window $\chi$ vs time (window = {w} chunks)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both" if log_y else "major", alpha=0.3)
    path = os.path.join(outdir, "rolling_chi_vs_time.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_equilibration_curves(
    df: pd.DataFrame, outdir: str, log_y: bool = False,
    from_back: bool = False, filename: str = "max_chi_vs_time.png",
) -> None:
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
    which = "last" if from_back else "first"
    ax.set_xlabel(f"fraction of time series used ({which} %)")
    ax.set_ylabel(r"$\chi^{\mathrm{max}}(L)$")
    ax.set_title(rf"Equilibration test: $\chi^{{\mathrm{{max}}}}$ vs time-series fraction ({which} %)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both" if log_y else "major", alpha=0.3)
    path = os.path.join(outdir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="χ^max vs time-series fraction (equilibration test)")
    parser.add_argument("--results", default="susceptibility/results")
    parser.add_argument("--outdir", default="plots/susceptibility")
    parser.add_argument(
        "--n-fractions", type=int, default=10,
        help="Expanding-prefix plot: number of evenly spaced prefixes from (1/n) to 1.0 "
             "(default 10 → 10%%,20%%,…,100%%).",
    )
    parser.add_argument(
        "--window", type=int, default=0,
        help="Rolling-window plot: window width in chunks (fixed sample count). "
             "0 (default) → auto = 10%% of the series length.",
    )
    parser.add_argument(
        "--stride", type=int, default=0,
        help="Rolling-window plot: step between window starts in chunks. "
             "0 (default) → auto = window/4.",
    )
    parser.add_argument("--log-y", action="store_true", help="Log-scale the χ axis.")
    args = parser.parse_args()

    groups = load_groups(args.results)

    # Expanding-prefix χ^max(L) vs fraction — first f% (original diagnostic).
    fractions = np.arange(1, args.n_fractions + 1) / args.n_fractions
    df = compute_equilibration_curves(groups, fractions)
    if df.empty:
        raise SystemExit("No curves computed — no usable (L, ε) groups found.")
    plot_equilibration_curves(df, args.outdir, log_y=args.log_y)
    csv_path = os.path.join(args.outdir, "max_chi_vs_time.csv")
    df.sort_values(["L", "fraction"]).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Same, accumulating from the equilibrated tail — last f% (10%, 20%, …).
    df_back = compute_equilibration_curves(groups, fractions, from_back=True)
    plot_equilibration_curves(
        df_back, args.outdir, log_y=args.log_y,
        from_back=True, filename="max_chi_from_back.png",
    )
    back_csv = os.path.join(args.outdir, "max_chi_from_back.csv")
    df_back.sort_values(["L", "fraction"]).to_csv(back_csv, index=False)
    print(f"Wrote {back_csv}")

    # Rolling-window χ vs actual MC time.
    series_len = min(min(r["m"].size for r in recs) for recs in groups.values())
    window = args.window if args.window > 0 else max(1, series_len // 10)
    stride = args.stride if args.stride > 0 else max(1, window // 4)
    roll = compute_rolling_chi(groups, window, stride)
    if roll.empty:
        print("No rolling-window curves computed — skipping rolling plot.")
        return
    plot_rolling_chi(roll, args.outdir, log_y=args.log_y)
    roll_csv = os.path.join(args.outdir, "rolling_chi_vs_time.csv")
    roll.sort_values(["L", "center_chunk"]).to_csv(roll_csv, index=False)
    print(f"Wrote {roll_csv}")


if __name__ == "__main__":
    main()
