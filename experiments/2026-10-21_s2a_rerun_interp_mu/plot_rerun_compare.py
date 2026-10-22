"""Original S2A <m> vs the rerun at linearly interpolated mu, per eps and L.

Original: SUSC_RUNS_S2A (mu from C2A Ly = 40 coex fits).
Rerun:    <exp>/SUSC_RUNS_S2A_INTERP (mu = linear interpolation between the nearest good
          S2A neighbours; -1.735 is a control rerun at its original mu with new seeds).
Both use the latest generation of replicas; error bar = standard error over replicas.

Run on Della from the main repo root:
    python plot_rerun_compare.py --exp <experiment dir>
Writes rerun_compare.png and rerun_compare.csv into --exp.
"""
import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORIG = "SUSC_RUNS_S2A"
RUN_DIR = "_{L}_{L}_S1_DF0.0_DMU1.0_K1.0/_{eps}"
LS = [16, 32, 48, 64, 96, 128]
THRESH = 0.05
OLD = "#8a8984"
NEW = "#2a78d6"
INK2 = "#52514e"
BAND = "#e3efe3"


def summarize(base, eps, L):
    path = os.path.join(base, RUN_DIR.format(L=L, eps=f"{eps:g}"), "susceptibility_data.csv")
    if not os.path.isfile(path):
        return None
    rows = list(csv.DictReader(open(path)))
    pt = max(float(r["prod_time"]) for r in rows)
    ms = np.array([float(r["m_mean"]) for r in rows if float(r["prod_time"]) == pt])
    return dict(m=ms.mean(), se=ms.std(ddof=1) / math.sqrt(len(ms)), n=len(ms),
                mu=float(rows[0]["mu"]), prod_time=pt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    new_base = os.path.join(args.exp, "SUSC_RUNS_S2A_INTERP")
    plan = list(csv.DictReader(open(os.path.join(args.exp, "mu_source.csv"))))
    plan.sort(key=lambda r: float(r["epsilon"]))

    table = []
    fig, axes = plt.subplots(1, len(plan), figsize=(3.3 * len(plan), 4.6), sharey=False)
    for ax, p in zip(axes, plan):
        eps = float(p["epsilon"])
        control = p["method"].startswith("control")
        for label, base, color, face, dx in (("original S2A", ORIG, OLD, "white", -1.5),
                                            ("rerun", new_base, NEW, NEW, 1.5)):
            pts = [(L, summarize(base, eps, L)) for L in LS]
            pts = [(L, s) for L, s in pts if s]
            x = [L + dx for L, _ in pts]
            ax.errorbar(x, [s["m"] for _, s in pts], yerr=[s["se"] for _, s in pts], fmt="o-",
                        ms=6, lw=1.5, color=color, mfc=face, mec=color, mew=1.3, capsize=2,
                        label=label)
            for L, s in pts:
                table.append(dict(epsilon=eps, L=L, run=label, mu=s["mu"], m_mean=s["m"],
                                  m_se=s["se"], n=s["n"], prod_time=s["prod_time"],
                                  within_0p05=abs(s["m"]) < THRESH))
        ax.axhspan(-THRESH, THRESH, color=BAND, lw=0, zorder=0)
        ax.axhline(0, color=INK2, lw=1, zorder=1)
        new_ok = all(abs(t["m_mean"]) < THRESH for t in table if t["epsilon"] == eps and t["run"] == "rerun")
        dmu = float(p["mu_coex_FITTED"]) - float(p["mu_original"])
        head = f"βε = {eps:g}" + ("  (control)" if control else "")
        sub = "μ unchanged, new seeds" if control else f"μ shifted {dmu:+.1e}"
        verdict = "all L within ±0.05" if new_ok else "some L outside ±0.05"
        ax.set_title(f"{head}\n{sub}\nrerun: {verdict}", loc="left", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(LS)
        ax.set_xticklabels([str(L) for L in LS])
        ax.set_xlabel("L")
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -0.08), max(hi, 0.08))
    axes[0].set_ylabel("⟨m⟩")
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("S2A rerun at linearly interpolated μ: ⟨m⟩ before vs after (green band: |⟨m⟩| < 0.05)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.exp, "rerun_compare.png"), dpi=160)

    with open(os.path.join(args.exp, "rerun_compare.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"{'eps':>7} {'L':>4} | {'original <m>':>16} | {'rerun <m>':>16} | within 0.05")
    for eps in sorted({t['epsilon'] for t in table}):
        for L in LS:
            o = next((t for t in table if t["epsilon"] == eps and t["L"] == L and t["run"] != "rerun"), None)
            n = next((t for t in table if t["epsilon"] == eps and t["L"] == L and t["run"] == "rerun"), None)
            if o and n:
                print(f"{eps:7.3f} {L:4d} | {o['m_mean']:+.3f} ± {o['m_se']:.3f} | "
                      f"{n['m_mean']:+.3f} ± {n['m_se']:.3f} | {'yes' if n['within_0p05'] else 'NO'}")
    print("wrote", os.path.join(args.exp, "rerun_compare.png"))


if __name__ == "__main__":
    main()
