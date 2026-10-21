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
BAD_BAND = "#fbe3d7"
EMPH_TEXT = "#b4461b"
EMPH = "#eb6834"
BAD_MU = 4e-4  # |mu - smooth fit| above this marks an eps as a "bad coex mu" point


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
        3, 1, figsize=(12, 12), sharex=True, gridspec_kw={"height_ratios": [1.1, 1, 0.8]})
    Ls = sorted(d["L"].unique())
    xlo, xhi = d["epsilon"].min() - 0.004, d["epsilon"].max() + 0.004
    # Cold end: eps where some L never passed the jump check (ignore the isolated bad-mu eps).
    unconv = sorted(set(d.loc[~d["converged"], "epsilon"]) - set(bad))
    cold_edge = max(unconv) + 0.0025 if unconv else None

    for ax in (ax1, ax2, ax3):
        if cold_edge is not None:
            ax.axvspan(xlo, cold_edge, facecolor="none", edgecolor="#c9c8c2", hatch="///", lw=0, zorder=0)
        for e in bad:
            ax.axvspan(e - 0.0024, e + 0.0024, color=BAD_BAND, lw=0, zorder=0)
        ax.axhline(0, color=INK2, lw=1, zorder=1)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("βε")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax1.set_xlim(xlo, xhi)

    for ax in (ax1, ax2):
        for L in Ls:
            g = d[d["L"] == L].sort_values("epsilon")
            ax.plot(g["epsilon"], g["m_mean"], color=L_COLORS[L], lw=1.2, alpha=0.75, zorder=2)
            for conv, face in ((True, L_COLORS[L]), (False, "white")):
                h = g[g["converged"] == conv]
                ax.errorbar(h["epsilon"], h["m_mean"], yerr=h["m_se"], fmt=L_MARKERS[L],
                            ms=5, color=L_COLORS[L], mfc=face, mec=L_COLORS[L], mew=1.1,
                            elinewidth=1, capsize=1.5, zorder=3,
                            label=f"L = {L}" if conv else None)

    # Label the bad-fit bands once, above panel (a).
    groups = []
    for e in sorted(bad):
        if groups and e - groups[-1][-1] < 0.0075:
            groups[-1].append(e)
        else:
            groups.append([e])
    for g in groups:
        ax1.annotate(" / ".join(f"{e:g}" for e in g) + "\nbad coex fit",
                     xy=(np.mean(g), 1.0), xycoords=("data", "axes fraction"),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.5, color=EMPH_TEXT)
    if cold_edge is not None:
        ax1.annotate("large-L runs\nnot converged", xy=((xlo + cold_edge) / 2, 1.0),
                     xycoords=("data", "axes fraction"), xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, color=INK2)

    ax1.set_ylabel("⟨m⟩")
    ax1.set_title("(a) ⟨m⟩ vs βε, full range", loc="left", fontsize=11, pad=34)
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, color=INK2))
    labels.append("open = not converged")
    ax1.legend(handles, labels, frameon=False, fontsize=9, ncol=4, loc="upper right")

    ax2.set_ylim(-0.06, 0.06)
    ax2.set_ylabel("⟨m⟩")
    ax2.set_title("(b) Same data, y zoomed to ±0.06 (bad-fit points run off the top/bottom)",
                  loc="left", fontsize=11)

    ax3.errorbar(mu["epsilon"], mu["resid"], yerr=mu["err"], fmt="o", ms=5, color=INK2,
                 elinewidth=1, capsize=1.5, zorder=3, label="μ used − smooth fit (± coex fit error)")
    flag = mu["epsilon"].isin(bad)
    ax3.plot(mu.loc[flag, "epsilon"], mu.loc[flag, "resid"], "o", ms=8, mfc=EMPH, mec="white",
             mew=1, zorder=4, label="bad coex fit (|μ error| > 4×10⁻⁴)")
    ax3.set_ylim(-3e-3, 3e-3)
    ax3.set_ylabel("μ error")
    ax3.set_title("(c) Error in the coex μ each run used (C2A Ly = 40 fit minus a smooth curve)",
                  loc="left", fontsize=11)
    ax3.text(0.01, 0.95, "above 0: μ too high → pushed dense → ⟨m⟩ > 0      "
             "below 0: μ too low → ⟨m⟩ < 0",
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
