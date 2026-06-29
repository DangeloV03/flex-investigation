"""
plot_correlation_test.py

Autocorrelation diagnostic: thin the m timeseries at different rates and check
whether peak chi grows — if it does, the raw chunks are autocorrelated.

For skip=s, takes every (s+1)th chunk from each replica's timeseries before
computing chi = (beta/N)(< |M|^2 > - < |M| >^2).

Produces in plots/susceptibility/correlation_test/:
  chi_vs_eps_skip{s}.png      — chi vs epsilon for each skip value
  peak_chi_vs_L_skip{s}.png   — peak chi vs L with K&D reference + our fit
  peak_chi_vs_L_all_skips.png — all skip levels overlaid (mega plot)

Usage:
    python plot_correlation_test.py --results susceptibility_results/exact
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from susceptibility_paths import read_susceptibility_csv

SKIP_VALUES = [0, 1, 2, 3, 5, 7, 10, 12, 15]

REF_A, REF_GNU = 0.095, 1.75  # Kumar & Dasgupta (2020)

L_STYLES: dict[int, dict] = {
    16:  {"color": "tab:blue",   "marker": "o"},
    32:  {"color": "tab:orange", "marker": "s"},
    48:  {"color": "tab:green",  "marker": "^"},
    64:  {"color": "tab:red",    "marker": "D"},
    96:  {"color": "tab:purple", "marker": "v"},
    128: {"color": "tab:brown",  "marker": "<"},
    256: {"color": "tab:olive",  "marker": ">"},
}


def _thinned_chi(ts_path: str, meta: dict, skip: int) -> dict | None:
    if not os.path.isfile(ts_path):
        return None
    ts = pd.read_csv(ts_path)
    if ts.empty or "m" not in ts.columns:
        return None

    ts = ts.iloc[::skip + 1]
    if len(ts) < 2:
        return None

    beta = float(meta["beta"])
    L = int(float(meta["L"]))
    N = L * L
    M = N * ts["m"].to_numpy(float)
    chi = beta / N * (np.mean(M ** 2) - np.mean(np.abs(M)) ** 2)

    return {"L": L, "epsilon": float(meta["epsilon"]), "chi": float(chi)}


def aggregate_thinned(results_dir: str, skip: int) -> pd.DataFrame:
    paths = glob.glob(os.path.join(results_dir, "**", "susceptibility_data.csv"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No susceptibility_data.csv under {results_dir}")

    records: list[dict] = []
    for csv_path in paths:
        dirpath = os.path.dirname(csv_path)
        for meta in read_susceptibility_csv(csv_path):
            run_id = str(meta.get("id", "")).strip()
            if not run_id:
                continue
            ts_path = os.path.join(dirpath, f"m_timeseries_{run_id}.csv")
            r = _thinned_chi(ts_path, meta, skip)
            if r:
                records.append(r)

    if not records:
        raise FileNotFoundError("No timeseries files found — check that runs have completed.")

    df = pd.DataFrame(records)

    rows = []
    for (l_val, eps), sub in df.groupby(["L", "epsilon"]):
        n = len(sub)
        rows.append({
            "L": int(l_val),
            "epsilon": float(eps),
            "chi_mean": float(sub["chi"].mean()),
            "chi_stderr": float(sub["chi"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["L", "epsilon"])


def _reference_and_fit(ax, L_vals: np.ndarray, chi_vals: np.ndarray) -> None:
    L_fine = np.geomspace(L_vals.min(), L_vals.max(), 200)
    ax.plot(L_fine, REF_A * L_fine ** REF_GNU, "-", color="red", linewidth=1.5,
            label=rf"K&D 2020: $A={REF_A}$, $\gamma/\nu={REF_GNU}$")
    slope, intercept = np.polyfit(np.log(L_vals), np.log(chi_vals), 1)
    fit_A = np.exp(intercept)
    ax.plot(L_fine, fit_A * L_fine ** slope, "--", color="blue", linewidth=1.5,
            label=rf"fit: $A={fit_A:.3f}$, $\gamma/\nu={slope:.3f}$")


def plot_chi_vs_eps(agg: pd.DataFrame, outdir: str, skip: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for l_val in sorted(agg["L"].unique()):
        sub = agg[agg["L"] == l_val].sort_values("epsilon")
        sty = L_STYLES.get(int(l_val), {})
        ax.errorbar(sub["epsilon"], sub["chi_mean"], yerr=sub["chi_stderr"],
                    fmt=f"{sty.get('marker', 'o')}-", markersize=3,
                    color=sty.get("color"), label=f"L={l_val}")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\chi$")
    ax.set_title(rf"$\chi$ vs $\varepsilon$ — skip {skip} (every {skip+1}th chunk)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, f"chi_vs_eps_skip{skip}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")


def plot_peak_chi_vs_L(agg: pd.DataFrame, outdir: str, skip: int) -> pd.DataFrame:
    peaks = agg.loc[agg.groupby("L")["chi_mean"].idxmax()].sort_values("L")
    L_vals = peaks["L"].to_numpy(float)
    chi_vals = peaks["chi_mean"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(L_vals, chi_vals, yerr=peaks["chi_stderr"].to_numpy(float),
                fmt="o", markersize=6, color="black", zorder=3, label="data")
    _reference_and_fit(ax, L_vals, chi_vals)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\chi^{\mathrm{max}}(L)$")
    ax.set_title(rf"Peak $\chi$ vs $L$ — skip {skip} (every {skip+1}th chunk)")
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(outdir, f"peak_chi_vs_L_skip{skip}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")
    return peaks


def plot_mega(all_peaks: dict[int, pd.DataFrame], outdir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.get_cmap("plasma")
    colors = cmap(np.linspace(0.05, 0.85, len(all_peaks)))

    all_L: list[float] = []
    for (skip, peaks), color in zip(all_peaks.items(), colors):
        L_vals = peaks["L"].to_numpy(float)
        all_L.extend(L_vals.tolist())
        ax.errorbar(peaks["L"], peaks["chi_mean"], yerr=peaks["chi_stderr"].to_numpy(float),
                    fmt="o-", markersize=4, color=color, label=f"skip {skip}")

    L_fine = np.geomspace(min(all_L), max(all_L), 200)
    ax.plot(L_fine, REF_A * L_fine ** REF_GNU, "-", color="red", linewidth=2,
            label=rf"K&D 2020: $\gamma/\nu={REF_GNU}$", zorder=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\chi^{\mathrm{max}}(L)$")
    ax.set_title(r"Peak $\chi$ vs $L$ — all skip values")
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(outdir, "peak_chi_vs_L_all_skips.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autocorrelation diagnostic via timeseries thinning")
    parser.add_argument("--results", default="susceptibility_results/exact",
                        help="Directory containing susceptibility results")
    parser.add_argument("--outdir", default="plots/susceptibility/correlation_test")
    parser.add_argument("--skips", type=int, nargs="+", default=SKIP_VALUES,
                        help="Skip values to test (default: 0 1 2 3 5 7 10 12 15)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    all_peaks: dict[int, pd.DataFrame] = {}

    for skip in args.skips:
        n_kept = f"every {skip + 1}th chunk"
        print(f"\n[skip={skip}] {n_kept}")
        agg = aggregate_thinned(args.results, skip)
        plot_chi_vs_eps(agg, args.outdir, skip)
        peaks = plot_peak_chi_vs_L(agg, args.outdir, skip)
        all_peaks[skip] = peaks

    print("\n[mega plot]")
    plot_mega(all_peaks, args.outdir)
    print("\nDone.")


if __name__ == "__main__":
    main()
