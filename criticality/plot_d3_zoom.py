"""Zoomed Max BC vs beta*epsilon for D3A-E: BC=5/9 crossing (dashed)."""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "criticality/d3_positive_drive_family"
df = pd.read_csv(f"{d}/bc_vs_beta_epsilon.csv")
crit = pd.read_csv(f"{d}/criticality.csv")
fig, ax = plt.subplots(figsize=(8.5, 6))
for i, (dmu, sub) in enumerate(df.groupby("delta_mu")):
    c = f"C{i}"
    sub = sub.sort_values("beta_epsilon")
    x, y = sub["beta_epsilon"].to_numpy(), sub["BC"].to_numpy()
    ax.errorbar(x, y, yerr=sub["BC_err"], fmt="o-", ms=4, capsize=3, color=c, label=rf"$\Delta\mu$={dmu:g}", zorder=3)
    xc = crit.loc[np.isclose(crit.delta_mu, dmu), "criticality_estimate"].iloc[0]
    ax.axvline(xc, ls="--", c=c, lw=1.2, alpha=0.8)
    print(f"dmu={dmu:g}: eps_c(BC=5/9)={xc:.4f} ")
ax.axhline(5 / 9, ls="--", c="grey", lw=1); ax.axhline(1 / 3, ls=":", c="grey", lw=1)
ax.plot([], [], "--", c="grey", label=r"$\epsilon_c$ (BC=5/9)")
ax.set_xlim(-1.95, -1.3); ax.set_ylim(0.38, 0.95)
ax.set_xlabel(r"$\beta\epsilon$"); ax.set_ylabel("Max Bimodality Coefficient")
ax.set_title(r"Max BC of $P(\phi_{col})$ vs $\beta\epsilon$ — D3A–E zoom (160x16)")
ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()
out = f"{d}/bc_max_vs_beta_epsilon_D3A-E_zoom.png"; fig.savefig(out, dpi=150); print(out)
