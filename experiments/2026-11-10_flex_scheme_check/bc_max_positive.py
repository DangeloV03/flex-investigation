"""Max BC (BC at the mu with <phi> closest to 0) vs beta*eps for all positive-drive coex campaigns.

Reads plots/bc_all_coex/<EXP>/ly<N>/bc_vs_beta_epsilon.csv (one.sh output) under --root.
Figure 1 (bc_max_positive_by_Ly): one panel per Ly, one curve per dmu; homo C2A (dmu=+1) dashed
          reference, BC = 5/9 guide and the equilibrium eps_c = -1.7627.
Figure 2 (bc_max_positive_by_dmu): one panel per dmu, Ly 16/20/40 overlaid (crossings ~ eps_c).
Also writes bc_max_positive.csv (all points) and bc_crossing_5_9.csv (sigmoid 5/9 crossing).

Usage:  python bc_max_positive.py --root /scratch/gpfs/WJACOBS/vd7294/flex-investigation --out <dir>
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

EXPS = {"D3F": -2.0, "D3B": -1.0, "D3A": 1.0, "D3C": 2.0, "D3D": 3.0, "D3E": 4.0}
COLOR = {-2.0: "#2a78d6", -1.0: "#eb6834", 1.0: "#1baf7a", 2.0: "#eda100", 3.0: "#e87ba4", 4.0: "#008300"}
MARK = {-2.0: "v", -1.0: "s", 1.0: "o", 2.0: "D", 3.0: "^", 4.0: "P"}
LY_COLOR = {16: "#2a78d6", 20: "#eb6834", 40: "#1baf7a"}
LY_MARK = {16: "o", 20: "s", 40: "^"}
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#9a9893"
EPS_C = -2 * np.log(1 + np.sqrt(2))
T = 5 / 9


def load(root, exp, ly):
    f = os.path.join(root, "plots", "bc_all_coex", exp, f"ly{ly}", "bc_vs_beta_epsilon.csv")
    if not os.path.isfile(f):
        return None
    d = pd.read_csv(f).dropna(subset=["BC"]).sort_values("beta_epsilon")
    return d[["beta_epsilon", "BC", "BC_err"]].assign(experiment=exp, Ly=ly)


def crossing(d):
    sig = lambda x, a, b, x0, w: a + b * np.tanh((x - x0) / w)
    x, y = d.beta_epsilon.values, d.BC.values
    try:
        p, _ = curve_fit(sig, x, y, p0=[0.45, -0.2, x[np.argmin(abs(y - T))], 0.05], maxfev=20000)
        z = (T - p[0]) / p[1]
        xc = p[2] + p[3] * np.arctanh(z) if abs(z) < 1 else np.nan
        return xc if x.min() - 0.02 <= xc <= x.max() + 0.02 else np.nan
    except Exception:
        return np.nan


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.axhline(T, color=REF, lw=0.8, ls=":")
    ax.axvline(EPS_C, color=INK, lw=0.8, ls=":")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xmin", type=float, default=-2.1)
    ap.add_argument("--xmax", type=float, default=-1.3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frames = [load(a.root, e, ly) for e in list(EXPS) + ["C2A"] for ly in (16, 20, 40)]
    data = pd.concat([f for f in frames if f is not None], ignore_index=True)
    data["delta_mu"] = data.experiment.map({**EXPS, "C2A": 1.0})
    data.to_csv(os.path.join(a.out, "bc_max_positive.csv"), index=False)
    win = data[(data.beta_epsilon >= a.xmin) & (data.beta_epsilon <= a.xmax)]

    cross = []
    for (e, ly), g in data.groupby(["experiment", "Ly"]):
        cross.append(dict(experiment=e, scheme="homo" if e == "C2A" else "positive_drive",
                          delta_mu=g.delta_mu.iloc[0], Ly=ly, n=len(g), eps_lo=g.beta_epsilon.min(),
                          eps_hi=g.beta_epsilon.max(), eps_cross_5_9=crossing(g)))
    cross = pd.DataFrame(cross).sort_values(["Ly", "delta_mu", "experiment"])
    cross.to_csv(os.path.join(a.out, "bc_crossing_5_9.csv"), index=False)
    print(cross.round(4).to_string(index=False))

    # Figure 1: panels by Ly
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, ly in zip(axes, (16, 20, 40)):
        ref = win[(win.experiment == "C2A") & (win.Ly == ly)]
        if len(ref):
            ax.plot(ref.beta_epsilon, ref.BC, "--", color=REF, lw=1.4, label="homo βΔμ=+1 (C2A)")
        for e, dmu in EXPS.items():
            g = win[(win.experiment == e) & (win.Ly == ly)]
            if g.empty:
                continue
            ax.plot(g.beta_epsilon, g.BC, "-", color=COLOR[dmu], lw=1.2, alpha=0.8)
            ax.plot(g.beta_epsilon, g.BC, MARK[dmu], ms=4, color=COLOR[dmu], mec="white", mew=0.5,
                    label=f"{e}: βΔμ={dmu:+g}")
        style(ax)
        ax.set_title(f"$L_y$ = {ly}", fontsize=10)
        ax.set_xlabel(r"$\beta\varepsilon$")
        ax.set_xlim(a.xmin, a.xmax)
    axes[0].set_ylabel("max BC  (BC at μ with ⟨φ⟩ ≈ 0)")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="center right", frameon=False, fontsize=9)
    fig.suptitle("Positive drive: max BC vs βε  (βΔf = 0, k = 1; dotted: BC = 5/9 and equilibrium ε_c = −1.7627)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.86, 0.94))
    fig.savefig(os.path.join(a.out, "bc_max_positive_by_Ly.png"), dpi=180)
    fig.savefig(os.path.join(a.out, "bc_max_positive_by_Ly.pdf"))
    plt.close(fig)

    # Figure 2: panels by dmu, Ly overlaid
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    for ax, (e, dmu) in zip(axes.ravel(), EXPS.items()):
        for ly in (16, 20, 40):
            g = win[(win.experiment == e) & (win.Ly == ly)]
            if g.empty:
                continue
            ax.plot(g.beta_epsilon, g.BC, "-", color=LY_COLOR[ly], lw=1.1, alpha=0.8)
            ax.plot(g.beta_epsilon, g.BC, LY_MARK[ly], ms=3.5, color=LY_COLOR[ly], mec="white", mew=0.4,
                    label=f"$L_y$={ly}")
        style(ax)
        ax.set_title(f"{e}: positive drive, βΔμ={dmu:+g}", fontsize=10)
        ax.set_xlim(a.xmin, a.xmax)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel(r"$\beta\varepsilon$")
    for ax in axes[:, 0]:
        ax.set_ylabel("max BC")
    fig.suptitle("Positive drive: max BC vs βε by Ly  (dotted: BC = 5/9, equilibrium ε_c)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(a.out, "bc_max_positive_by_dmu.png"), dpi=180)
    plt.close(fig)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
