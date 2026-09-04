"""
count_mag_jumps.py

Count magnetization well-crossings ("jumps") in susceptibility production
timeseries and plot the average ⟨J⟩ vs L.

A jump is counted when m(t) leaves the high well (above 75% of the trace maximum)
and enters the low well (below 75% of the trace minimum), or vice versa.  For
normalized m ∈ [-1, 1] at full saturation this is equivalent to crossing ±0.75.

By default only ε values within eps_crit ± eps_window are included (see
post_presentation_notes.md).  Pass --epsilon to analyse a single ε instead.

Usage:
    python susceptibility/count_mag_jumps.py \\
        --results susceptibility/results/eq_12p5x_<date> \\
        --outdir plots/eq_timetest/12p5x

    python susceptibility/count_mag_jumps.py \\
        --results susceptibility/results/eq_12p5x_<date> \\
        --epsilon -1.85 \\
        --outdir plots/eq_timetest/12p5x

    python susceptibility/count_mag_jumps.py \\
        --results susceptibility/results/eq_12p5x_<date> \\
        --L 96 --all-eps \\
        --outdir plots/eq_timetest/12p5x

    python susceptibility/count_mag_jumps.py --list \\
        --results susceptibility/results/eq_12p5x_<date>
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import csv

from analyze_chi_max_scaling import load_replica_groups
from plot_susceptibility import L_PLOT_STYLE, resolve_repo_path
from susceptibility_paths import find_susc_run_csvs, parse_susc_run_dir

L_COLOR = {L: style["color"] for L, style in L_PLOT_STYLE.items()}


def count_jumps(m: np.ndarray, frac: float = 0.75) -> int:
    """Count well crossings in a single m(t) trace.

    Thresholds are 75% of that trace's max and min (``frac`` of extrema).
    Intermediate samples between wells do not change the latched state.
    """
    m = np.asarray(m, dtype=float)
    if m.size < 2:
        return 0

    m_max = float(np.max(m))
    m_min = float(np.min(m))
    if m_max <= m_min:
        return 0

    thresh_hi = frac * m_max
    thresh_lo = frac * m_min
    if thresh_hi <= thresh_lo:
        return 0

    state = 0  # +1 high well, -1 low well, 0 undecided
    jumps = 0
    for val in m:
        if val >= thresh_hi:
            new_state = 1
        elif val <= thresh_lo:
            new_state = -1
        else:
            continue
        if state != 0 and new_state != state:
            jumps += 1
        state = new_state
    return jumps


def _stderr(vals: np.ndarray) -> float:
    n = vals.size
    if n <= 1:
        return float("nan")
    return float(np.std(vals, ddof=1) / np.sqrt(n))


def _eps_tag(eps: float) -> str:
    return f"eps{abs(eps):.3f}".replace(".", "p")


def pick_epsilon(groups: dict[tuple[int, float], dict], target: float) -> float:
    """Snap *target* to the nearest ε present in *groups*."""
    eps_values = np.sort([eps for _, eps in groups])
    return float(eps_values[int(np.argmin(np.abs(eps_values - target)))])


def analyze_jumps(
    groups: dict[tuple[int, float], dict],
    *,
    eps_crit: float,
    eps_window: float,
    epsilon: float | None,
    frac: float,
    L_filter: int | None = None,
    all_eps: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float | None]:
    """Return (summary by L×ε, summary by L, per-replica detail, selected ε or None)."""
    if epsilon is not None:
        eps_use = pick_epsilon(groups, epsilon)
        selected_eps = eps_use
    else:
        selected_eps = None

    detail_rows: list[dict] = []
    for (L, eps), g in sorted(groups.items()):
        if L_filter is not None and L != L_filter:
            continue
        if epsilon is not None:
            if not np.isclose(eps, eps_use):
                continue
        elif not all_eps:
            eps_lo = eps_crit - eps_window
            eps_hi = eps_crit + eps_window
            if not (eps_lo <= eps <= eps_hi):
                continue
        for rep_idx, m_arr in enumerate(g["replicas"]):
            detail_rows.append({
                "L": L,
                "epsilon": eps,
                "replica": rep_idx,
                "J": count_jumps(m_arr, frac=frac),
                "n_chunks": int(m_arr.size),
            })

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, selected_eps

    summary = (
        detail.groupby(["L", "epsilon"], as_index=False)
        .agg(J_mean=("J", "mean"), J_std=("J", "std"), n_replicas=("J", "size"))
    )
    summary["J_stderr"] = summary.apply(
        lambda r: r["J_std"] / np.sqrt(r["n_replicas"]) if r["n_replicas"] > 1 else float("nan"),
        axis=1,
    )

    by_L = (
        detail.groupby("L", as_index=False)
        .agg(J_mean=("J", "mean"), n_replicas=("J", "size"))
    )
    for L in by_L["L"]:
        vals = detail.loc[detail["L"] == L, "J"].to_numpy(float)
        by_L.loc[by_L["L"] == L, "J_stderr"] = _stderr(vals)
    by_L = by_L.sort_values("L")

    return summary, by_L, detail, selected_eps


def plot_J_vs_L(
    by_L: pd.DataFrame,
    out_path: str,
    *,
    eps_crit: float,
    eps_window: float,
    epsilon: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    L = by_L["L"].to_numpy(float)
    J = by_L["J_mean"].to_numpy(float)
    err = by_L["J_stderr"].to_numpy(float)

    colors = [L_COLOR.get(int(l), "gray") for l in L]
    ax.errorbar(L, J, yerr=err, fmt="o-", color="tab:blue", markersize=7,
                capsize=3, linewidth=1.4, zorder=3)
    for l, j, c in zip(L, J, colors):
        ax.plot(l, j, "o", color=c, markersize=8, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\langle J \rangle$")
    if epsilon is not None:
        title_eps = rf"$\varepsilon = {epsilon:.3f}$"
    else:
        title_eps = rf"$\varepsilon \in [{eps_crit - eps_window:.2f},\, {eps_crit + eps_window:.2f}]$"
    ax.set_title(rf"Average magnetization jumps vs $L$ ({title_eps})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_J_vs_eps_single_L(summary: pd.DataFrame, L: int, out_path: str) -> None:
    """⟨J⟩ vs ε for one system size."""
    sub = summary.loc[summary["L"] == L].sort_values("epsilon")
    if sub.empty:
        return

    style = L_PLOT_STYLE.get(L, {"color": "tab:blue", "marker": "o"})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        sub["epsilon"], sub["J_mean"], yerr=sub["J_stderr"],
        fmt=f'{style["marker"]}-', color=style["color"],
        capsize=3, linewidth=1.3, markersize=6,
    )
    ax.set_xlabel(r"$\varepsilon$ (bonding strength)")
    ax.set_ylabel(r"$\langle J \rangle$")
    ax.set_title(rf"Average magnetization jumps vs $\varepsilon$ at $L={L}$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_J_vs_eps(summary: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for L in sorted(summary["L"].unique()):
        sub = summary.loc[summary["L"] == L].sort_values("epsilon")
        style = L_PLOT_STYLE.get(int(L), {"color": "gray", "marker": "o"})
        ax.errorbar(
            sub["epsilon"], sub["J_mean"], yerr=sub["J_stderr"],
            fmt=f'{style["marker"]}-', color=style["color"], label=f"L={int(L)}",
            capsize=2, linewidth=1.1, markersize=5,
        )
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\langle J \rangle$")
    ax.set_title("Average jumps vs chemical potential")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def load_susc_runs_groups(base: str) -> dict[tuple[int, float], dict]:
    """Load replica groups from a SUSC_RUNS base directory.

    Parses L and ε from the path structure  base/*/_*/susceptibility_data.csv
    and loads the m_timeseries_{id}.csv for each replica.

    Returns { (L, eps): {"beta": float, "N": int, "replicas": [m_array, ...]} }
    Only uses the most up-to-date row per (L, eps) group (highest prod_time).
    """
    csv_paths = find_susc_run_csvs(base)
    if not csv_paths:
        return {}

    from susceptibility_paths import read_susceptibility_csv

    groups: dict[tuple[int, float], dict] = {}
    n_paths = len(csv_paths)
    print(f"[check] found {n_paths} run dirs; reading timeseries …", flush=True)

    for i, data_csv in enumerate(csv_paths, start=1):
        if i == 1 or i % 25 == 0 or i == n_paths:
            print(f"[check] loading {i}/{n_paths} …", flush=True)
        run_dir = os.path.dirname(data_csv)
        L, eps = parse_susc_run_dir(run_dir)
        if L is None or eps is None:
            continue

        rows = read_susceptibility_csv(data_csv)
        if not rows:
            continue

        # Only keep the most recent generation (max prod_time).
        max_pt = max(float(r.get("prod_time", 0) or 0) for r in rows)
        latest_rows = [r for r in rows
                       if abs(float(r.get("prod_time", 0) or 0) - max_pt) < 1.0]

        beta = float(latest_rows[0].get("beta", 1.0) or 1.0)
        n_sites = L * L

        m_arrays: list[np.ndarray] = []
        for row in latest_rows:
            run_id = row.get("id", "")
            if not run_id:
                continue
            ts_path = os.path.join(run_dir, f"m_timeseries_{run_id}.csv")
            if not os.path.isfile(ts_path):
                continue
            try:
                m_arr = pd.read_csv(ts_path, usecols=["m"])["m"].to_numpy(dtype=float)
            except (ValueError, KeyError, OSError):
                continue
            if m_arr.size > 0:
                m_arrays.append(m_arr)

        if not m_arrays:
            continue

        key = (L, round(eps, 8))
        groups[key] = {"beta": beta, "N": n_sites, "replicas": m_arrays}

    print(f"[check] loaded {len(groups)} (L, ε) pairs; counting jumps …", flush=True)
    return groups


def compute_jump_summary(
    results_base: str,
    frac: float = 0.75,
    threshold: float = 10.0,
) -> pd.DataFrame:
    """Compute average jump count per (L, ε) from a SUSC_RUNS directory.

    Loads timeseries from the new-layout  SUSC_RUNS/*/_*/ path structure.
    Returns DataFrame with columns: L, epsilon, J_mean, J_std, J_stderr, n_replicas.
    Adds a boolean  passes  column (True if J_mean >= threshold).
    """
    groups = load_susc_runs_groups(results_base)
    _empty = pd.DataFrame({"L": pd.Series(dtype=int), "epsilon": pd.Series(dtype=float),
                           "J_mean": pd.Series(dtype=float), "J_std": pd.Series(dtype=float),
                           "J_stderr": pd.Series(dtype=float), "n_replicas": pd.Series(dtype=int),
                           "passes": pd.Series(dtype=bool)})
    if not groups:
        return _empty

    summary, _, _, _ = analyze_jumps(
        groups,
        eps_crit=0.0,   # unused when all_eps=True
        eps_window=0.0,
        epsilon=None,
        frac=frac,
        all_eps=True,
    )

    if summary.empty:
        return _empty

    summary = summary.rename(columns={"J_stderr": "J_stderr"}).copy()
    summary["passes"] = summary["J_mean"] >= threshold
    return summary.sort_values(["L", "epsilon"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="susceptibility/results",
                   help="Susceptibility results tree (e.g. susceptibility/results/eq_12p5x_2026-07-20)")
    p.add_argument("--outdir", default="plots/eq_timetest/12p5x")
    p.add_argument("--eps-crit", type=float, default=-1.75,
                   help="Central ε for the analysis window (default: -1.75; ignored with --epsilon)")
    p.add_argument("--eps-window", type=float, default=0.1,
                   help="Half-width of ε window around eps_crit (default: 0.1; ignored with --epsilon)")
    p.add_argument("--epsilon", type=float, default=None,
                   help="Analyse a single ε only (snaps to nearest available); writes tagged outputs")
    p.add_argument("--L", type=int, default=None,
                   help="Restrict analysis to one system size (e.g. 96 for ⟨J⟩ vs ε scan)")
    p.add_argument("--all-eps", action="store_true",
                   help="Include every ε in the results tree (ignore eps_crit ± window)")
    p.add_argument("--frac", type=float, default=0.75,
                   help="Fraction of trace max/min for well thresholds (default: 0.75)")
    p.add_argument("--list", action="store_true", help="List available L and ε, then exit")
    args = p.parse_args()

    args.results = resolve_repo_path(args.results)
    args.outdir = resolve_repo_path(args.outdir)

    try:
        groups = load_replica_groups(args.results)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.list:
        eps_vals = sorted({eps for _, eps in groups})
        L_vals = sorted({L for L, _ in groups})
        print(f"Results: {args.results}")
        print(f"L values ({len(L_vals)}): {L_vals}")
        print(f"ε range: {eps_vals[0]:.4f} … {eps_vals[-1]:.4f} ({len(eps_vals)} values)")
        lo, hi = args.eps_crit - args.eps_window, args.eps_crit + args.eps_window
        in_window = [e for e in eps_vals if lo <= e <= hi]
        print(f"ε in [{lo:.3f}, {hi:.3f}]: {len(in_window)} values")
        return 0

    summary, by_L, detail, selected_eps = analyze_jumps(
        groups,
        eps_crit=args.eps_crit,
        eps_window=args.eps_window,
        epsilon=args.epsilon,
        frac=args.frac,
        L_filter=args.L,
        all_eps=args.all_eps,
    )

    if detail.empty:
        if args.epsilon is not None:
            print(
                f"ERROR: no replicas at ε={args.epsilon:.4f} "
                f"(nearest available: {selected_eps}). "
                f"Use --list to inspect available data.",
                file=sys.stderr,
            )
        elif args.L is not None:
            lo, hi = args.eps_crit - args.eps_window, args.eps_crit + args.eps_window
            scope = "all ε" if args.all_eps else f"ε ∈ [{lo:.3f}, {hi:.3f}]"
            print(
                f"ERROR: no replicas for L={args.L} in {scope}. "
                f"Use --list to inspect available data.",
                file=sys.stderr,
            )
        else:
            lo, hi = args.eps_crit - args.eps_window, args.eps_crit + args.eps_window
            print(
                f"ERROR: no replicas in ε ∈ [{lo:.3f}, {hi:.3f}]. "
                f"Use --list to inspect available data or adjust --eps-crit / --eps-window.",
                file=sys.stderr,
            )
        return 1

    if args.epsilon is not None and not np.isclose(selected_eps, args.epsilon):
        print(f"Note: ε={args.epsilon} not found; using nearest available ε={selected_eps:.4f}")

    tag = _eps_tag(selected_eps) if selected_eps is not None else ""
    suffix = f"_{tag}" if tag else ""
    if args.L is not None:
        suffix = f"{suffix}_L{args.L}" if suffix else f"_L{args.L}"

    os.makedirs(args.outdir, exist_ok=True)
    detail.to_csv(os.path.join(args.outdir, f"J_per_replica{suffix}.csv"), index=False)
    summary.to_csv(os.path.join(args.outdir, f"J_vs_epsilon{suffix}.csv"), index=False)
    if args.L is None or args.epsilon is not None or not args.all_eps:
        by_L.to_csv(os.path.join(args.outdir, f"J_vs_L{suffix}.csv"), index=False)

    if args.L is not None and args.all_eps and args.epsilon is None:
        print(f"\n⟨J⟩ vs ε at L={args.L} (n={detail.shape[0]} replicas):")
        print(f"{'epsilon':>10} {'⟨J⟩':>8} {'stderr':>8} {'n_rep':>6}")
        sub = summary.loc[summary["L"] == args.L].sort_values("epsilon")
        for _, row in sub.iterrows():
            print(f"{row['epsilon']:>10.4f} {row['J_mean']:>8.2f} {row['J_stderr']:>8.2f} "
                  f"{int(row['n_replicas']):>6}")
        plot_J_vs_eps_single_L(
            summary, args.L,
            os.path.join(args.outdir, f"J_vs_epsilon{suffix}.png"),
        )
        return 0

    if selected_eps is not None:
        eps_label = f"ε = {selected_eps:.3f}"
    else:
        eps_label = (
            f"ε ∈ [{args.eps_crit - args.eps_window:.3f}, "
            f"{args.eps_crit + args.eps_window:.3f}]"
        )
    print(f"\n⟨J⟩ vs L  ({eps_label}, n={detail.shape[0]} replicas):")
    print(f"{'L':>6} {'⟨J⟩':>8} {'stderr':>8} {'n_rep':>6}")
    for _, row in by_L.iterrows():
        print(f"{int(row['L']):>6} {row['J_mean']:>8.2f} {row['J_stderr']:>8.2f} "
              f"{int(row['n_replicas']):>6}")

    plot_J_vs_L(
        by_L,
        os.path.join(args.outdir, f"J_vs_L{suffix}.png"),
        eps_crit=args.eps_crit,
        eps_window=args.eps_window,
        epsilon=selected_eps,
    )
    if selected_eps is None:
        plot_J_vs_eps(summary, os.path.join(args.outdir, "J_vs_epsilon.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
