"""Equilibrium vs driven 128x128 spin-spin correlations, from spin_corr_analyze.py output.

Each run is given as PLOTS_DIR:OUT_DIR (OUT_DIR holds eps*/replica_*/timeseries.csv) plus its eps_c.
Points are compared at equal distance from their own critical point, d_eps = eps - eps_c(scheme).

  python compare_eq_driven.py --eq SPIN_CORR_EQ128/plots:SPIN_CORR_EQ128 \
      --driven SPIN_CORR_NEG_DMU3_128/plots:SPIN_CORR_NEG_DMU3_128 --eps-c-driven -1.958 \
      --label-driven "neg drive βΔμ=3" --out compare/

Figure (compare_eq_vs_driven.png), four panels:
  a  near eps_c: full <s0 sr> on log-log, power-law fit r = 2-16 (eta_eff), <|m|>^2 plateau
  b  disordered side: connected G(r) = <s0 sr> - <s>^2 on log-linear, 2D OZ fit e^{-r/xi}/sqrt(r)
  c  xi vs d_eps: measured OZ xi for both, exact 2D Ising xi(d_eps) above and below eps_c as reference
  d  P(m) in production for every run
Table: compare_eq_vs_driven.csv.
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS_C_EQ = -2 * np.log(1 + np.sqrt(2))
INK2, GRID = "#52514e", "#e4e3df"
EQ_COL = ["#1f4e9c", "#6fa0e0"]       # eq: near eps_c, disordered
DRV_COL = ["#b8410f", "#f0915a"]      # driven


def ising_xi(d_eps):
    """Exact 2D Ising axis correlation length at eps = eps_c + d_eps (K = |eps|/4).
    Above Tc 1/xi = ln coth K - 2K; below Tc the connected correlation decays with 1/xi = 2(2K - ln coth K)."""
    K = np.abs(EPS_C_EQ + np.asarray(d_eps, float)) / 4
    a = np.log(1 / np.tanh(K)) - 2 * K
    return np.where(a > 0, 1 / np.where(a > 0, a, np.nan), 1 / (2 * np.where(a < 0, -a, np.nan)))


def load(spec, eps_c, label):
    plots, out = spec.split(":")
    summ = pd.read_csv(os.path.join(plots, "summary.csv"))
    runs = []
    for r in summ.to_dict("records"):
        tag = "eps" + f"{r['eps']:.4f}".replace("-", "m").replace(".", "p")
        df = pd.read_csv(os.path.join(plots, f"corr_{tag}.csv"))
        ms = []
        for ts in sorted(glob.glob(os.path.join(out, tag, "replica_*", "timeseries.csv"))):
            t = pd.read_csv(ts).drop_duplicates(["phase", "chunk"], keep="last")
            ms.append(t[t.phase == "prod"].m.values)
        runs.append(dict(r, d_eps=r["eps"] - eps_c, eps_c=eps_c, label=label, df=df,
                         m=np.concatenate(ms) if ms else np.array([]), tag=tag))
    return runs


def power_fit(df, r_min=2, r_max=16):
    k = (df.r >= r_min) & (df.r <= r_max) & (df.S_full > 0)
    sl, ic = np.polyfit(np.log(df.r[k]), np.log(df.S_full[k]), 1)
    return -sl, np.exp(ic)


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eq", required=True)
    ap.add_argument("--driven", required=True, nargs="+", help="one or more PLOTS:OUT (e.g. finished + partial eps)")
    ap.add_argument("--eps-c-driven", type=float, required=True)
    ap.add_argument("--label-driven", default="driven")
    ap.add_argument("--partial", nargs="*", type=float, default=[], help="driven eps still running (marked prelim.)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    eq = load(a.eq, EPS_C_EQ, "equilibrium")
    drv = [r for spec in a.driven for r in load(spec, a.eps_c_driven, a.label_driven)]
    for r in drv:
        r["prelim"] = any(abs(r["eps"] - e) < 1e-6 for e in a.partial)
    for r in eq:
        r["prelim"] = False
    # "near-critical" = the point of each set closest to its own eps_c; the other is the disordered-side point
    def split(runs):
        runs = sorted(runs, key=lambda r: abs(r["d_eps"]))
        return runs[0], [r for r in runs[1:] if r["d_eps"] > 0]
    eq_c, eq_dis = split(eq)
    dr_c, dr_dis = split(drv)

    def name(r):
        s = f"{r['label']}, βε = {r['eps']:g} (Δε = {r['d_eps']:+.3f})"
        return s + " [prelim.]" if r["prelim"] else s

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))
    rows = []

    # a: near eps_c, full <s0 sr>
    ax = axes[0, 0]
    for r, col in ((eq_c, EQ_COL[0]), (dr_c, DRV_COL[0])):
        df = r["df"]
        k = (df.r >= 1) & (df.S_full > 0)
        eta, A = power_fit(df)
        r["eta_eff"] = eta
        ax.errorbar(df.r[k], df.S_full[k], yerr=df.S_full_err[k], fmt="o", ms=3, color=col, elinewidth=0.7,
                    label=f"{name(r)}\n   fit r = 2–16: η_eff = {eta:.3f}")
        rr = np.arange(2, 17)
        ax.plot(rr, A * rr ** -eta, "-", color=col, lw=1.2)
        ax.axhline(r["absm"] ** 2, color=col, ls="--", lw=0.9)
    ax.text(0.02, 0.03, "dashed: ⟨|m|⟩² of each run (finite-size plateau)\nexact 2D Ising at ε_c: η = 1/4",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=INK2)
    lo = min(r["absm"] ** 2 for r in (eq_c, dr_c))
    ax.set_ylim(bottom=lo * 0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("r (lattice units)"); ax.set_ylabel(r"$\langle s_0 s_r\rangle$")
    ax.set_title("a  closest to ε_c: full spin correlation", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    style(ax)

    # b: disordered side, connected G with OZ fit
    ax = axes[0, 1]
    for r, col in [(x, EQ_COL[1]) for x in eq_dis] + [(x, DRV_COL[1]) for x in dr_dis]:
        df = r["df"]
        G = df.S_full - (2 * r["rho"] - 1) ** 2
        rs = int(r["r_signal"]) if np.isfinite(r["r_signal"]) else 40
        k = (df.r >= 1) & (df.r <= rs) & (G > 0)
        ax.errorbar(df.r[k], G[k], yerr=df.S_full_err[k], fmt="o", ms=3, color=col, elinewidth=0.7,
                    label=f"{name(r)}\n   2D OZ fit (r = 10–{min(rs, 40)}): ξ = {r['xi_oz']:.2f} ± {r['xi_oz_err']:.2f}")
        rr = np.linspace(1, rs, 200)
        if np.isfinite(r["xi_oz"]):
            ax.plot(rr, r["oz_A"] * np.exp(-rr / r["xi_oz"]) / np.sqrt(rr), "-", color=col, lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("r (lattice units)"); ax.set_ylabel(r"$G(r)=\langle s_0 s_r\rangle-\langle s\rangle^2$")
    ax.set_title("b  disordered side: connected correlation, shown while G > 5σ", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    style(ax)

    # c: xi vs distance from own eps_c
    ax = axes[1, 0]
    d = np.linspace(-0.2, 0.2, 801)
    xi = ising_xi(d)
    ax.plot(d[d > 0], xi[d > 0], "-", color=INK2, lw=1.2, label="exact Ising ξ, disordered")
    ax.plot(d[d < 0], xi[d < 0], "--", color=INK2, lw=1.2, label="exact Ising ξ, ordered (connected)")
    for runs, cols, mk in ((eq, EQ_COL, "o"), (drv, DRV_COL, "s")):
        for r in runs:
            if np.isfinite(r.get("xi_oz", np.nan)) and r["d_eps"] > 0:
                ax.errorbar(r["d_eps"], r["xi_oz"], yerr=r["xi_oz_err"], fmt=mk, ms=8,
                            mfc="white" if r["prelim"] else cols[1], mec=cols[1], mew=1.6, color=cols[1],
                            label=f"{'eq' if r['label'] == 'equilibrium' else 'driven'} βε = {r['eps']:g}: "
                                  f"ξ = {r['xi_oz']:.2f}" + (" (prelim.)" if r["prelim"] else ""))
    ax.axvline(0, color=GRID, lw=1)
    ax.axhline(64, color=GRID, lw=1, ls=":")
    ax.text(0.195, 58, "L/2", fontsize=7.5, color=INK2, ha="right", va="top")
    ax.set_yscale("log"); ax.set_ylim(1, 150)
    ax.set_xlabel(r"Δε = βε − βε_c (own scheme; ε_c eq = −1.7627, driven = %g)" % a.eps_c_driven)
    ax.set_ylabel("ξ (lattice units)")
    ax.set_title("c  correlation length at equal distance from criticality", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    style(ax)

    # d: P(m)
    ax = axes[1, 1]
    bins = np.linspace(-0.9, 0.9, 121)
    for r, col in ((eq_c, EQ_COL[0]), (dr_c, DRV_COL[0])) + tuple((x, EQ_COL[1]) for x in eq_dis) \
            + tuple((x, DRV_COL[1]) for x in dr_dis):
        if len(r["m"]):
            ax.hist(r["m"], bins=bins, density=True, histtype="step", lw=1.5, color=col,
                    label=f"{name(r)}: ⟨m⟩ = {r['m'].mean():+.3f}, ⟨|m|⟩ = {np.abs(r['m']).mean():.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("m (production, all replicas)"); ax.set_ylabel("P(m)")
    ax.set_title("d  magnetization distribution", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.2, loc="lower center")
    style(ax)

    fig.suptitle(f"Spin-spin correlations, L = 128: equilibrium vs {a.label_driven}", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(a.out, "compare_eq_vs_driven.png"), dpi=170)
    plt.close(fig)

    for r in eq + drv:
        rows.append(dict(set=r["label"], eps=r["eps"], eps_c=r["eps_c"], d_eps=r["d_eps"], prelim=r["prelim"],
                         n_snap=r["n_snap"], m=r["m"].mean() if len(r["m"]) else np.nan,
                         absm=np.abs(r["m"]).mean() if len(r["m"]) else np.nan, tau_absm=r["tau_absm"],
                         n_eff=r["n_eff_total"], eta_eff=r.get("eta_eff", np.nan), xi_oz=r["xi_oz"],
                         xi_oz_err=r["xi_oz_err"], xi_exp=r["xi_exp"], r_signal=r["r_signal"],
                         xi_ising_at_d_eps=float(ising_xi(r["d_eps"]))))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out, "compare_eq_vs_driven.csv"), index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
