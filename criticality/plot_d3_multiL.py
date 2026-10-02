"""D3A-E (positive drive): Max BC vs beta*eps at Ly=16/20/40, plus FSS of beta*eps_c vs 1/L."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "criticality/d3_positive_drive_family"
EXPS = {"D3A": 1, "D3B": -1, "D3C": 2, "D3D": 3, "D3E": 4}
LYS = (16, 20, 40)
order = sorted(EXPS, key=lambda e: EXPS[e])
col = {e: f"C{i}" for i, e in enumerate(order)}

def bracketed(e, ly):
    d = pd.read_csv(f"criticality/{e}/ly{ly}/bc_vs_beta_epsilon.csv")
    return d.BC.min() < 5 / 9 < d.BC.max() and ((d.sort_values("beta_epsilon").BC.values[0] >= 5 / 9))

for ly in LYS:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for e in order:
        d = pd.read_csv(f"criticality/{e}/ly{ly}/bc_vs_beta_epsilon.csv").sort_values("beta_epsilon")
        ax.errorbar(d.beta_epsilon, d.BC, yerr=d.BC_err, fmt="o-", ms=3.5, capsize=2, color=col[e],
                    label=rf"$\Delta\mu$={EXPS[e]:+d}")
        c = pd.read_csv(f"criticality/{e}/ly{ly}/criticality.csv").epsilon_c_estimate.iloc[0]
        if d.beta_epsilon.min() <= c <= d.beta_epsilon.max():
            ax.axvline(c, ls="--", lw=1, color=col[e], alpha=.8)
    ax.axhline(5 / 9, ls="--", c="grey", lw=1); ax.axhline(1 / 3, ls=":", c="grey", lw=1)
    if ly == 16: ax.set_xlim(-2.1, -1.2); ax.set_ylim(0.33, 0.95)
    ax.set_xlabel(r"$\beta\epsilon$"); ax.set_ylabel("Max Bimodality Coefficient")
    ax.set_title(rf"Max BC vs $\beta\epsilon$ — positive drive, {10*ly}x{ly}")
    ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()
    fig.savefig(f"{OUT}/bc_max_vs_beta_epsilon_positive_Ly{ly}.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5.5))
rows = []
for e in order:
    s = pd.read_csv(f"criticality/{e}/multi_L/eq_scaling_vs_L.csv").sort_values("L_short")
    f = pd.read_csv(f"criticality/{e}/multi_L/eq_fss_fit.csv").iloc[0]
    invL = 1 / s.L_short.values; y = s.beta_epsilon_c.values
    ax.plot(invL, y, "o", color=col[e], label=rf"$\Delta\mu$={EXPS[e]:+d}: $\infty$={f.beta_eps_c_infty:.3f}±{f.beta_eps_c_infty_err:.3f}")
    xx = np.linspace(0, 1 / 16 * 1.05, 50); ax.plot(xx, f.beta_eps_c_infty + f.invL_slope * xx, "-", color=col[e], lw=1)
    ax.plot(0, f.beta_eps_c_infty, "s", color=col[e], mfc="none")
    for ly, v in zip(s.L_short, y):
        if ly == 40 or ly == 20:
            if not bracketed(e, ly): ax.plot(1 / ly, v, "o", ms=11, mfc="none", mec="k", lw=1)
ax.plot([], [], "o", ms=11, mfc="none", mec="k", ls="", label="BC=5/9 not bracketed in window (extrapolated)")
ax.set_xlim(-0.003, 0.068); ax.set_xlabel("1/L  (L = Ly)"); ax.set_ylabel(r"$\beta\epsilon_c$ (BC = 5/9)")
ax.set_title(r"FSS: $\beta\epsilon_c$ vs 1/L, linear fit, positive drive"); ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fss_beta_eps_c_vs_invL_positive.png", dpi=150)
