"""Max BC vs beta*eps, negative drive (C3), Ly=16. C3E excluded (data off)."""
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPS = {"C3B": -1, "C3A": 1, "C3C": 2, "C3D": 3}
fig, ax = plt.subplots(figsize=(8.5, 6))
for i, (e, dmu) in enumerate(EXPS.items()):
    d = pd.read_csv(f"criticality/{e}/ly16/bc_vs_beta_epsilon.csv").sort_values("beta_epsilon")
    ax.errorbar(d.beta_epsilon, d.BC, yerr=d.BC_err, fmt="o-", ms=3.5, capsize=2, color=f"C{i}",
                label=rf"$\Delta\mu$={dmu:+d} ({e})")
ax.axhline(5 / 9, ls="--", c="grey", lw=1); ax.axhline(1 / 3, ls=":", c="grey", lw=1)
ax.set_xlabel(r"$\beta\epsilon$"); ax.set_ylabel("Max Bimodality Coefficient")
ax.set_title(r"Max BC of $P(\phi_{col})$ vs $\beta\epsilon$ — negative drive, 160x16")
ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()
fig.savefig("criticality/c3_negative_drive_family/bc_max_vs_beta_epsilon_negative_Ly16.png", dpi=150)
