"""Two independent estimates of each eps's mu error, per campaign.

x: "mu jump" = the coex mu used minus a robust smooth cubic through mu(eps) (coex side).
y: "offset felt" = 2<m>/chi from the susc runs at the largest L (susc side), chi = N * pooled var(m).
If the jump in the coex mu is what pushes <m> off zero, points fall on y = x.

Run on Della from the main repo root:  python plot_jump_vs_felt.py --out <dir>
"""
import argparse
import csv
import glob
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CAMPAIGNS = ["S2A", "POS_DMU1", "POS_DMU2", "POS_DMU3", "NEG_DMU1", "NEG_DMU2", "NEG_DMU3"]
DATA = "#2a78d6"
EMPH = "#eb6834"
INK2 = "#52514e"
LIM = 2.2e-3


def load(camp):
    out = {}
    for f in glob.glob(f"SUSC_RUNS_{camp}/_*/_*/susceptibility_data.csv"):
        rows = list(csv.DictReader(open(f)))
        pt = max(float(r["prod_time"]) for r in rows)
        rr = [r for r in rows if float(r["prod_time"]) == pt]
        L, e = int(rr[0]["L"]), round(float(rr[0]["epsilon"]), 6)
        ms = np.array([float(r["m_mean"]) for r in rr])
        m2 = np.array([float(r["m2_mean"]) for r in rr])
        m, var = ms.mean(), m2.mean() - ms.mean() ** 2
        se = ms.std(ddof=1) / math.sqrt(len(ms))
        chi = L * L * var
        out[(L, e)] = dict(mu=float(rr[0]["mu"]), felt=2 * m / chi, felt_err=2 * se / chi)
    return out


def robust_jump(e, mu):
    keep = np.ones(len(e), bool)
    for _ in range(3):
        p = np.polyfit(e[keep], mu[keep], 3)
        r = mu - np.polyval(p, e)
        mad = np.median(np.abs(r[keep])) * 1.4826
        keep = np.abs(r) < 4 * mad
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, camp in zip(axes, CAMPAIGNS):
        R = load(camp)
        eps = np.array(sorted({e for _, e in R}))
        mu = np.array([next(v["mu"] for (L, e2), v in R.items() if e2 == e) for e in eps])
        jump = robust_jump(eps, mu)
        Lbig = max(L for L, _ in R)
        idx = [i for i, e in enumerate(eps) if (Lbig, e) in R]
        x = jump[idx]
        y = np.array([R[(Lbig, eps[i])]["felt"] for i in idx])
        yerr = np.array([R[(Lbig, eps[i])]["felt_err"] for i in idx])
        ok = np.isfinite(y) & (np.abs(y) < 5e-3)
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        slope = np.polyfit(x[ok], y[ok], 1)[0]

        ax.axhline(0, color="#d9d8d3", lw=0.8)
        ax.axvline(0, color="#d9d8d3", lw=0.8)
        ax.plot([-LIM, LIM], [-LIM, LIM], color=EMPH, lw=1.5, ls="--", zorder=1)
        ax.errorbar(np.clip(x, -LIM, LIM), np.clip(y, -LIM, LIM), yerr=yerr, fmt="o", ms=5,
                    color=DATA, mec="white", mew=0.8, elinewidth=0.8, capsize=0, zorder=2)
        ax.set_title(f"{camp}  (L = {Lbig})", loc="left", fontsize=11)
        ax.text(0.04, 0.96, f"r = {r:+.2f}\nslope = {slope:.2f}", transform=ax.transAxes,
                va="top", fontsize=9.5, color=INK2)
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    key = axes[-1]
    key.axis("off")
    key.text(0.0, 0.95,
             "Each dot = one ε.\n\n"
             "x: μ jump — the coex μ used,\n    minus a smooth curve through\n    all ε (coex side).\n\n"
             "y: μ offset the susc run felt,\n    2⟨m⟩/χ at the largest L\n    (susc side).\n\n"
             "Dashed: y = x. On the line, the\ncoex jump fully explains ⟨m⟩ ≠ 0.\n\n"
             "Points beyond ±2.2×10⁻³ are\ndrawn at the edge.",
             va="top", fontsize=10, color=INK2, transform=key.transAxes)
    for ax in axes[4:7]:
        ax.set_xlabel("μ jump vs neighbours (coex)")
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("μ offset felt, 2⟨m⟩/χ (susc)")
    fig.suptitle("Coex μ jumps predict the μ offset the susc runs respond to (y = x ⇒ jumps are fit errors)",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "mu_jump_vs_offset_felt.png")
    fig.savefig(path, dpi=160)
    print("wrote", path)


if __name__ == "__main__":
    main()
