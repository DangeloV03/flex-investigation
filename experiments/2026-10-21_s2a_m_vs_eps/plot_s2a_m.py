"""S2A: <m> vs eps for every L, with the coex-mu error that drives it.

S2A = homo, beta*delta_f = 0, k = 1, beta*delta_mu = 1, mu taken from the C2A Ly = 40 coex fits.

Inputs (Della):
  --data    s2a_m_vs_eps.csv (L, epsilon, mu, n, m_mean, m_se, jump_status, ...)
  --coex    COEX_RUNS_C2A/ly40/manage.csv (mu_coex_FITTED, mu_coex_FITTED_error)
Output: s2a_m_vs_eps.png in --out.

<m> per (L, eps) is the mean over replicas of the latest generation; the error bar is the
standard error over replicas. Open markers = pair never passed the J >= 10 jump check.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

L_COLORS = {16: "#86b6ef", 32: "#5598e7", 48: "#2a78d6", 64: "#1c5cab", 96: "#104281", 128: "#06223f"}
L_MARKERS = {16: "o", 32: "s", 48: "^", 64: "D", 96: "v", 128: "P"}
INK2 = "#52514e"
BAND = "#ecebe6"
EMPH = "#eb6834"
BAD_MU = 4e-4  # |mu - smooth fit| above this marks an eps as a "bad coex mu" point
ZOOM_EPS = -1.736  # panel (b) shows eps above this (all pairs converged there)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--coex", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    d["converged"] = d["jump_status"].str.contains("✓")
    c = pd.read_csv(args.coex)
    c["epsilon"] = c["epsilon"].round(4)
    c = c.set_index("epsilon")

    mu = d.groupby("epsilon")["mu"].first().reset_index()
    mu["err"] = [c.loc[round(e, 4), "mu_coex_FITTED_error"] for e in mu["epsilon"]]
    w = 1.0 / np.maximum(mu["err"].to_numpy(), 1e-5)
    p = np.polyfit(mu["epsilon"], mu["mu"], 3, w=w)
    mu["resid"] = mu["mu"] - np.polyval(p, mu["epsilon"])
    bad = mu[mu["resid"].abs() > BAD_MU]["epsilon"].to_numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 11.5), gridspec_kw={"height_ratios": [1.25, 1, 0.8]})
    ax3.sharex(ax1)
    Ls = sorted(d["L"].unique())
    offs = dict(zip(Ls, np.linspace(-0.0013, 0.0013, len(Ls))))

    for ax in (ax1, ax2, ax3):
        for e in bad:
            ax.axvspan(e - 0.0025, e + 0.0025, color=BAND, lw=0, zorder=0)
        ax.axhline(0, color=INK2, lw=1, zorder=1)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax in (ax1, ax2):
        for L in Ls:
            g = d[d["L"] == L].sort_values("epsilon")
            if ax is ax2:
                g = g[g["epsilon"] > ZOOM_EPS]
            x = g["epsilon"] + offs[L]
            ax.plot(x, g["m_mean"], color=L_COLORS[L], lw=1.4, alpha=0.8, zorder=2)
            for conv, face in ((True, L_COLORS[L]), (False, "white")):
                h = g[g["converged"] == conv]
                ax.errorbar(h["epsilon"] + offs[L], h["m_mean"], yerr=h["m_se"], fmt=L_MARKERS[L],
                            ms=5.5, color=L_COLORS[L], mfc=face, mec=L_COLORS[L], mew=1.2,
                            elinewidth=1.1, capsize=1.5, zorder=3,
                            label=f"L = {L}" if conv else None)

    ax1.set_ylabel("⟨m⟩")
    ax1.tick_params(labelbottom=False)
    ax1.set_title("(a) ⟨m⟩ vs βε, full range", loc="left", fontsize=11)
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, color=INK2))
    labels.append("not converged (J < 10)")
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND))
    labels.append("ε with |μ error| > 4×10⁻⁴")
    ax1.legend(handles, labels, frameon=False, fontsize=9, ncol=4, loc="upper right")

    ax2.set_xlim(ZOOM_EPS - 0.002, d["epsilon"].max() + 0.004)
    ax2.set_ylim(-0.06, 0.06)
    ax2.set_ylabel("⟨m⟩")
    ax2.set_xlabel("βε")
    ax2.set_title("(b) Zoom on βε > −1.736 (all converged); the point at −1.665 is off-scale (+0.14)",
                  loc="left", fontsize=11)

    ax3.errorbar(mu["epsilon"], mu["resid"], yerr=mu["err"], fmt="o", ms=5, color=INK2,
                 elinewidth=1.1, capsize=1.5, zorder=3, label="μ used − smooth fit (± C2A fit error)")
    flag = mu["epsilon"].isin(bad)
    ax3.plot(mu.loc[flag, "epsilon"], mu.loc[flag, "resid"], "o", ms=7, mfc=EMPH, mec="white",
             mew=1, zorder=4, label="|μ error| > 4×10⁻⁴")
    ax3.set_ylim(-3e-3, 3e-3)
    ax3.set_ylabel("Δμ")
    ax3.set_xlabel("βε")
    ax3.set_title("(c) Error in the coex μ we ran at (C2A Ly = 40 fit minus smooth cubic in ε)",
                  loc="left", fontsize=11)
    ax3.text(0.01, 0.95, "μ too high → pushed dense → ⟨m⟩ > 0;  μ too low → ⟨m⟩ < 0.  "
             "Error bars at −1.74 and −1.665 run off-scale (±8e-3, ±3e-3).",
             transform=ax3.transAxes, fontsize=9, color=INK2, va="top")
    ax3.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle("S2A: ⟨m⟩ vs βε   (homo, βΔf = 0, k = 1, βΔμ = 1; μ from C2A Ly = 40 coex)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "s2a_m_vs_eps.png")
    fig.savefig(path, dpi=160)
    mu.to_csv(os.path.join(args.out, "s2a_mu_residuals.csv"), index=False)
    print("bad-mu eps:", ", ".join(f"{e:g}" for e in bad))
    print("wrote", path)


if __name__ == "__main__":
    main()
