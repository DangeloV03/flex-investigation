"""Active-active spatial correlation C(r) from the spin_corr_run.py snapshots.

n_i = 1 if site i is BONDING (active), else 0.  For every eps folder:
  C_glob(r) = <n_i n_j> - rho^2        rho = mean density over ALL snapshots of that eps
  C_snap(r) = <n_i n_j - rho_s^2>      rho_s = density of each snapshot (the summer analyzer's
                                       convention; it suppresses the long-range tail near eps_c)
<n_i n_j> is computed with an FFT for every displacement, averaged over sites and snapshots,
then radially binned (integer r) and also kept along the two lattice axes.
Error bars: standard error across replicas (each replica averaged first).

Also from timeseries.csv: integrated autocorrelation time tau of m and |m| (in time units)
and the effective number of independent snapshots, n_eff = T_prod / (2 tau) per replica.

Exact 2D Ising references (spin s = 2n - 1, so C = G_spin / 4, K = |eps|/4):
  eps_c:  G_spin(r) ~ r^(-1/4)
  eps > eps_c:  axis correlation length 1/xi = ln(coth K) - 2K   (xi ~ 8.7 at eps = -1.65)

Outputs in --plots: corr_<eps>.csv, summary.csv, C_r.png, C_r_loglog.png, m_timeseries.png
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

BONDING = 2


def tau_int(x, c=5.0):
    """Integrated autocorrelation time in samples (Sokal automatic window)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 10 or np.var(x) == 0:
        return np.nan
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (np.arange(n, 0, -1) * np.var(x))
    tau = 0.5
    for w in range(1, n):
        tau += acf[w]
        if w >= c * tau:
            break
    return tau


def replica_sums(snaps):
    """Sum over snapshots of the 2D <n_i n_{i+d}> map, of rho, and of rho^2 (per snapshot)."""
    acc, rho_sum, rho2_sum = None, 0.0, 0.0
    for p in snaps:
        n = (np.load(p) == BONDING).astype(float)
        N = n.size
        F = np.fft.fft2(n)
        c2 = np.fft.ifft2(F * np.conj(F)).real / N
        acc = c2 if acc is None else acc + c2
        r = n.mean()
        rho_sum += r
        rho2_sum += r * r
    return acc, rho_sum, rho2_sum, len(snaps)


def radial(c2):
    L = c2.shape[0]
    d = np.minimum(np.arange(L), L - np.arange(L))
    dx, dy = np.meshgrid(d, d, indexing="ij")
    r = np.rint(np.sqrt(dx ** 2 + dy ** 2)).astype(int)
    rmax = L // 2
    m = r <= rmax
    num = np.bincount(r[m], weights=c2[m], minlength=rmax + 1)
    cnt = np.bincount(r[m], minlength=rmax + 1)
    return np.arange(rmax + 1), num / np.maximum(cnt, 1)


def axis_avg(c2):
    L = c2.shape[0]
    return 0.5 * (c2[: L // 2 + 1, 0] + c2[0, : L // 2 + 1])


def analyze_eps(edir):
    reps = sorted(glob.glob(os.path.join(edir, "replica_*")))
    meta = json.load(open(os.path.join(reps[0], "meta.json")))
    per_rep, tot = [], dict(acc=0.0, rho=0.0, rho2=0.0, n=0)
    taus = []
    for rd in reps:
        snaps = sorted(glob.glob(os.path.join(rd, "snap_*.npy")))
        if not snaps:
            continue
        acc, rs, r2s, n = replica_sums(snaps)
        per_rep.append((acc / n, rs / n, r2s / n))
        tot["acc"] = tot["acc"] + acc
        tot["rho"] += rs
        tot["rho2"] += r2s
        tot["n"] += n
        ts = read_ts(rd)
        pr = ts[ts.phase == "prod"]
        dt = meta["chunk_time"]
        t_m = tau_int(pr.m.values) * dt
        t_am = tau_int(np.abs(pr.m.values)) * dt
        taus.append(dict(replica=os.path.basename(rd), n_snap=n, m_mean=pr.m.mean(),
                         absm_mean=pr.m.abs().mean(), tau_m=t_m, tau_absm=t_am,
                         T_prod=pr.t.iloc[-1] - ts[ts.phase == "eq"].t.iloc[-1] if (ts.phase == "eq").any() else np.nan))
    if not per_rep:
        return None
    rho = tot["rho"] / tot["n"]
    nn = tot["acc"] / tot["n"]
    r, C_glob = radial(nn - rho ** 2)
    _, C_snap = radial(nn - tot["rho2"] / tot["n"])
    C_axis = axis_avg(nn - rho ** 2)
    # replica spread (each replica vs the common rho, so the errors describe C_glob)
    reps_C = np.array([radial(a - rho ** 2)[1] for a, _, _ in per_rep])
    err = reps_C.std(axis=0, ddof=1) / np.sqrt(len(reps_C)) if len(reps_C) > 1 else np.full_like(C_glob, np.nan)
    df = pd.DataFrame(dict(r=r, C_glob=C_glob, C_glob_err=err, C_snap=C_snap, C_axis=C_axis,
                           G_spin=4 * C_glob))
    tau = pd.DataFrame(taus)
    tau["n_eff"] = tau.T_prod / (2 * tau.tau_absm)
    return meta, rho, df, tau


def read_ts(rd):
    """timeseries.csv; a job killed mid-chunk repeats that chunk on resume, keep the last row."""
    ts = pd.read_csv(os.path.join(rd, "timeseries.csv"))
    ts = ts.drop_duplicates(["phase", "chunk"], keep="last")
    order = ts.phase.map({"eq": 0, "prod": 1})
    return ts.assign(_o=order).sort_values(["_o", "chunk"]).drop(columns="_o").reset_index(drop=True)


def ising_xi(eps):
    if abs(eps + 2 * np.log(1 + np.sqrt(2))) < 1e-3:
        return np.inf
    K = abs(eps) / 4
    inv = np.log(1 / np.tanh(K)) - 2 * K
    return 1 / inv if inv > 0 else np.inf


def fit_xi(df, L):
    m = (df.r >= 2) & (df.r <= L / 4) & (df.C_glob > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    f = lambda r, A, xi: A * np.exp(-r / xi) / np.sqrt(r)   # Ornstein-Zernike form in 2D
    try:
        p, cov = curve_fit(f, df.r[m], df.C_glob[m], p0=[df.C_glob[m].iloc[0], 5.0], maxfev=20000)
        return p[1], np.sqrt(cov[1, 1])
    except Exception:
        return np.nan, np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="campaign folder with eps*/replica_*")
    ap.add_argument("--plots", required=True)
    a = ap.parse_args()
    os.makedirs(a.plots, exist_ok=True)

    rows, curves = [], {}
    for edir in sorted(glob.glob(os.path.join(a.out, "eps*"))):
        res = analyze_eps(edir)
        if res is None:
            continue
        meta, rho, df, tau = res
        eps, L = meta["eps"], meta["L"]
        tag = os.path.basename(edir)
        df.to_csv(os.path.join(a.plots, f"corr_{tag}.csv"), index=False)
        tau.to_csv(os.path.join(a.plots, f"tau_{tag}.csv"), index=False)
        xi, xi_err = fit_xi(df, L)
        rows.append(dict(eps=eps, L=L, n_snap=int(tau.n_snap.sum()), rho=rho, m=2 * rho - 1,
                         absm=tau.absm_mean.mean(), tau_m=tau.tau_m.mean(), tau_absm=tau.tau_absm.mean(),
                         n_eff_total=tau.n_eff.sum(), xi_fit=xi, xi_fit_err=xi_err,
                         xi_ising_exact=ising_xi(eps)))
        curves[eps] = (df, L)
        print(f"\n== {tag}  rho={rho:.4f}  snapshots={int(tau.n_snap.sum())}")
        print(tau.round(4).to_string(index=False))

    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.plots, "summary.csv"), index=False)
    print("\n" + summ.round(4).to_string(index=False))

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"]
    for logy, name in [(True, "C_r.png"), ("loglog", "C_r_loglog.png")]:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for (eps, (df, L)), c in zip(sorted(curves.items()), colors):
            d = df[(df.r >= 1) & (df.C_glob > 0)]
            ax.errorbar(d.r, d.C_glob, yerr=d.C_glob_err, fmt="o", ms=3, color=c, label=f"βε = {eps:g}")
            s = df[(df.r >= 1) & (df.C_snap > 0)]
            ax.plot(s.r, s.C_snap, "--", color=c, lw=1, alpha=0.7)
            if abs(eps + 1.7627) < 1e-3:
                rr = np.arange(1, L // 2 + 1)
                ax.plot(rr, d.C_glob.iloc[0] * rr ** -0.25, ":", color="k", lw=1, label=r"$r^{-1/4}$ (exact)")
        ax.set_yscale("log")
        if logy == "loglog":
            ax.set_xscale("log")
        ax.set_xlabel("r (lattice units)")
        ax.set_ylabel(r"$C(r)=\langle n_0 n_r\rangle-\rho^2$")
        ax.set_title(f"Active-active correlation, equilibrium, L={L}  (dashed: per-snapshot ρ)", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.plots, name), dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(len(curves) or 1, 1, figsize=(8, 2.4 * max(len(curves), 1)), squeeze=False)
    for ax, edir in zip(axes[:, 0], sorted(glob.glob(os.path.join(a.out, "eps*")))):
        for rd, c in zip(sorted(glob.glob(os.path.join(edir, "replica_*"))), colors):
            ts = read_ts(rd)
            ax.plot(ts.t, ts.m, lw=0.5, color=c)
        ax.set_title(os.path.basename(edir), fontsize=9)
        ax.set_ylabel("m")
    axes[-1, 0].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(os.path.join(a.plots, "m_timeseries.png"), dpi=150)
    plt.close(fig)
    print(f"\nWrote {a.plots}")


if __name__ == "__main__":
    main()
