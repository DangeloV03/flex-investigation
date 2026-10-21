"""One worked example: a bad coex fit (S2A, eps = -1.74) vs a good neighbour (eps = -1.735).

Top row: the C2A Ly = 40 coex data phi(mu) and the analyzer's own zero-crossing fit
(coex/analyzer.py fit_zero_crossing_with_error), which set the mu that S2A ran at.
Bottom row: distribution of m (all replicas, all time) in the S2A run at that mu.

Run on Della from the main repo root (needs coex/ on PYTHONPATH, e.g. `source env.sh`):
    python plot_bad_fit_case.py --out <dir> [--L 64]
"""
import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyzer import fit_zero_crossing_with_error, sign_change_bracket

C2A = "COEX_RUNS_C2A/ly40/results/400x40_homo_deltaF0p0_dmu1p0_epsilon{tag}/phi_psi.csv"
S2A = "SUSC_RUNS_S2A/_{L}_{L}_S1_DF0.0_DMU1.0_K1.0/_{eps}"
CASES = [(-1.74, "m1p74", "Bad fit"), (-1.735, "m1p735", "Good fit")]
DATA = "#2a78d6"
EMPH = "#eb6834"
INK2 = "#52514e"
TRACE = "#86b6ef"


def fit_points(mu, phi):
    """Same point selection as fit_zero_crossing_with_error (bracket +- one neighbour)."""
    lo, hi = sign_change_bracket(mu, phi)
    s = np.sort(mu)
    lo_i, hi_i = int(np.searchsorted(s, lo)), int(np.searchsorted(s, hi))
    a, b = s[max(0, lo_i - 1)], s[min(len(s) - 1, hi_i + 1)]
    return (mu >= a - 1e-9) & (mu <= b + 1e-9)


def latest_replicas(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "susceptibility_data.csv"))))
    pt = max(float(r["prod_time"]) for r in rows)
    return [r for r in rows if float(r["prod_time"]) == pt]


def m_by_L(eps):
    out = {}
    for d in sorted(glob.glob(S2A.format(L="*", eps=eps)), key=lambda p: int(p.split("/")[1].split("_")[1])):
        L = int(d.split("/")[1].split("_")[1])
        ms = [float(r["m_mean"]) for r in latest_replicas(d)]
        out[L] = np.mean(ms)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=64)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for col, (eps, tag, name) in enumerate(CASES):
        pp = pd.read_csv(C2A.format(tag=tag)).sort_values("mu")
        mu, phi, err = pp["mu"].to_numpy(), pp["phi"].to_numpy(), pp["phi_err"].to_numpy()
        mu0, mu0_err = fit_zero_crossing_with_error(mu, phi, err)
        used = fit_points(mu, phi)

        # Fit line, recomputed the same way as the analyzer.
        w = 1.0 / err[used] ** 2
        slope, icpt = np.polyfit(mu[used], phi[used], 1, w=np.sqrt(w))

        ax = axes[0, col]
        lo = min(mu[used].min(), mu0) - 1.0e-3
        hi = max(mu[used].max(), mu0) + 1.0e-3
        win = (mu > lo) & (mu < hi)
        ax.axhline(0, color=INK2, lw=1)
        ax.errorbar(mu[win & ~used], phi[win & ~used], yerr=err[win & ~used], fmt="o", ms=5,
                    color=DATA, mfc="white", elinewidth=1, capsize=1.5, label="φ(μ), not in fit")
        ax.errorbar(mu[used], phi[used], yerr=err[used], fmt="o", ms=6, color=DATA,
                    elinewidth=1.2, capsize=1.5, label="φ(μ), used in fit")
        xs = np.linspace(lo, hi, 50)
        ax.plot(xs, slope * xs + icpt, color=EMPH, lw=2, label="analyzer's linear fit")
        ax.axvline(mu0, color=EMPH, lw=1.5, ls="--")
        ax.text(0.02, 0.97, f"μ used (dashed) = {mu0:.5f}\nanalyzer error = ± {mu0_err:.1e}",
                transform=ax.transAxes, color=EMPH, fontsize=9, va="top")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("βμ")
        ax.set_ylabel("φ (slab order parameter)")
        ax.set_title(f"({'ab'[col]}) {name}: C2A Ly = 40 coex at βε = {eps}", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.locator_params(axis="x", nbins=5)

        # Bottom: distribution of m over all replicas and all time at that mu.
        ax = axes[1, col]
        run = S2A.format(L=args.L, eps=eps)
        reps = latest_replicas(run)
        allm = np.concatenate([pd.read_csv(os.path.join(run, f"m_timeseries_{r['id']}.csv"))["m"].to_numpy()
                               for r in reps])
        bins = np.linspace(-0.9, 0.9, 61)
        ax.hist(allm, bins=bins, weights=np.full(len(allm), 1.0 / len(allm)), color=TRACE,
                edgecolor="white", linewidth=0.5)
        ax.axvline(0, color=INK2, lw=1)
        mbar = np.mean([float(r["m_mean"]) for r in reps])
        ax.axvline(mbar, color=EMPH, lw=2, label=f"⟨m⟩ = {mbar:+.3f}")
        ax.set_xlim(-0.9, 0.9)
        ax.set_xlabel("m")
        ax.set_ylabel("fraction of time")
        ax.set_title(f"({'cd'[col]}) S2A L = {args.L} run at that μ: where m spends its time "
                     f"({len(reps)} replicas)", loc="left", fontsize=11)
        byL = m_by_L(eps)
        ax.text(0.02, 0.97, "⟨m⟩ by L:\n" + "\n".join(f"L = {L:>3}: {v:+.2f}" for L, v in byL.items()),
                transform=ax.transAxes, fontsize=8.5, color=INK2, va="top", family="monospace",
                bbox=dict(facecolor="white", edgecolor="#d9d8d3", boxstyle="round,pad=0.4"), zorder=5)
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.suptitle("One bad coex fit → system pushed into one phase  "
                 "(homo, βΔf = 0, k = 1, βΔμ = 1)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "bad_fit_case_eps1p74.png")
    fig.savefig(path, dpi=160)
    print("wrote", path)


if __name__ == "__main__":
    main()
