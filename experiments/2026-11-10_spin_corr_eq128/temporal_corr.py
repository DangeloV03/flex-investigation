"""Temporal autocorrelation of the magnetization for one eps of the 128^2 equilibrium runs.

C_x(t) = <dx(0) dx(t)> / <dx^2> for x = m and x = |m|, taken from the production rows of each
replica's timeseries.csv (one sample per snapshot interval, 1000 time units by default).
  - m: mean set to 0 (exact by up/down symmetry at coexistence). This keeps the slow
    decorrelation by sign flips; subtracting each replica's own mean would hide it.
  - |m|: pooled mean over all replicas.
Numerator and denominator are pooled over replicas; the error band is the replica spread
of the per-replica normalized C(t) / sqrt(nrep).
Also reports, per observable:
  tau_int  integrated time 1/2 + sum C(t), with Sokal's self-consistent window (W >= 6 tau_int)
  tau_exp  exponential fit of C(t) where 0.1 < C < 0.8

Usage: python temporal_corr.py --eps-dir <OUT>/epsm1p7627 --plots <OUT>/plots
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def acf_raw(x):
    """Unnormalized autocovariance sum_i x_i x_{i+t} / (n - t), via FFT (x already centered)."""
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    c = np.fft.irfft(f * np.conj(f))[:n]
    return c / (n - np.arange(n))


def tau_int(rho, c=6.0):
    """Sokal window: smallest W with W >= c * tau_int(W). Returns (tau in samples, W)."""
    cs = 0.5 + np.cumsum(rho[1:])
    for W in range(1, len(cs)):
        if W >= c * cs[W - 1]:
            return cs[W - 1], W
    return cs[-1], len(cs)


def tau_exp(t, rho, lo=0.1, hi=0.8):
    m = (rho > lo) & (rho < hi) & (t > 0)
    # first contiguous stretch only (the tail is noisy)
    idx = np.where(m)[0]
    if len(idx) < 3:
        return np.nan, np.nan
    stop = np.where(np.diff(idx) > 1)[0]
    idx = idx[: stop[0] + 1] if len(stop) else idx
    if len(idx) < 3:
        return np.nan, np.nan
    sl, ic = np.polyfit(t[idx], np.log(rho[idx]), 1)
    return -1.0 / sl, np.exp(ic)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eps-dir", required=True)
    p.add_argument("--plots", required=True)
    a = p.parse_args()
    os.makedirs(a.plots, exist_ok=True)
    tag = os.path.basename(a.eps_dir.rstrip("/"))

    series = []
    f = None
    for f in sorted(glob.glob(os.path.join(a.eps_dir, "replica_*", "timeseries.csv"))):
        d = pd.read_csv(f).drop_duplicates(subset=["phase", "chunk"], keep="last")
        d = d[d.phase == "prod"].sort_values("chunk")
        series.append(d.m.to_numpy())
    n = min(len(s) for s in series)
    series = [s[:n] for s in series]
    t_all = pd.read_csv(f)
    t_all = t_all[t_all.phase == "prod"].t.to_numpy()
    dt = float(np.median(np.diff(t_all)))
    nrep = len(series)
    lag = np.arange(n) * dt
    absmean = np.mean([np.abs(s).mean() for s in series])

    out, rows = {}, []
    for name, fx, mean in (("m", lambda s: s, 0.0), ("|m|", np.abs, absmean)):
        raws = [acf_raw(fx(s) - mean) for s in series]
        pooled = np.mean(raws, axis=0)
        rho = pooled / pooled[0]
        per = np.array([r / r[0] for r in raws])
        err = per.std(axis=0, ddof=1) / np.sqrt(nrep)
        ti, W = tau_int(rho)
        # fit only where the replica error is small (m: the slow flip part, C from 0.95 down to 0.35)
        te, A = tau_exp(lag, rho, *((0.35, 0.95) if name == "m" else (0.1, 0.8)))
        t_e = lag[np.argmax(rho < np.exp(-1))] if (rho < np.exp(-1)).any() else np.nan
        conv = W < n / 2   # Sokal window fits inside half the run -> tau_int is trustworthy
        out[name] = dict(rho=rho, per=per, err=err, tau_int=ti * dt, W=W * dt, tau_exp=te, A=A, t_e=t_e, conv=conv)
        rows.append(dict(obs=name, tau_int=ti * dt, window=W * dt, tau_int_converged=conv, tau_exp=te, t_1_over_e=t_e,
                         T_prod=n * dt, n_eff=n / (2 * ti) * nrep,
                         **{f"tau_int_rep{i}": tau_int(per[i])[0] * dt for i in range(nrep)}))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.plots, f"temporal_tau_{tag}.csv"), index=False)
    nkeep = int(min(n, 8e6 / dt))
    pd.DataFrame({"t": lag[:nkeep], "C_m": out["m"]["rho"][:nkeep], "C_m_err": out["m"]["err"][:nkeep],
                  "C_absm": out["|m|"]["rho"][:nkeep], "C_absm_err": out["|m|"]["err"][:nkeep]}
                 ).to_csv(os.path.join(a.plots, f"temporal_corr_{tag}.csv"), index=False)
    print(summ.to_string(index=False))

    # plot: left = short times (|m| relaxation inside one well), right = long times (sign flips of m)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    tm_abs = out["|m|"]["tau_int"]
    tm_m = out["m"]["tau_int"]
    # lags beyond ~T/3 have too few independent pairs to mean anything, so stop there
    t_m = out["m"]["tau_exp"] if np.isfinite(out["m"]["tau_exp"]) else tm_m
    for ax, tmax, which in ((axes[0], 5 * tm_abs, ("|m|", "m")), (axes[1], min(3 * t_m, n * dt / 3), ("m",))):
        k = lag <= tmax
        for name in which:
            o = out[name]
            col = "black" if name == "m" else "#d62728"
            for i in range(nrep):
                ax.plot(lag[k], o["per"][i][k], color=col, lw=0.6, alpha=0.25)
            ax.fill_between(lag[k], (o["rho"] - o["err"])[k], (o["rho"] + o["err"])[k], color=col, alpha=0.2, lw=0)
            ax.plot(lag[k], o["rho"][k], color=col, lw=2,
                    label=f"$C_{{{name}}}(t)$:  $\\tau_{{exp}}$ = {o['tau_exp']:.2g},  1/e at {o['t_e']:.2g},  "
                          + (f"$\\tau_{{int}}$ = {o['tau_int']:.2g}" if o["conv"] else "$\\tau_{int}$ not converged"))
            if np.isfinite(o["tau_exp"]):
                ax.plot(lag[k], o["A"] * np.exp(-lag[k] / o["tau_exp"]), color=col, ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1.2)
        ax.set_xlim(0, tmax)
        ax.axhline(np.exp(-1), color="gray", lw=0.6, ls=":")
        ax.set_xlabel("lag t (kMC time units)")
        ax.set_ylabel("normalized autocorrelation C(t)")
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_title("short times: |m| relaxes inside one phase")
    axes[1].set_title("long times: m decorrelates only by sign flips")
    eps = tag.replace("eps", "").replace("m", "-").replace("p", ".")
    fig.suptitle(f"Temporal autocorrelation of the magnetization, βε = {eps}, L = 128, {nrep} replicas "
                 f"× {n * dt:.2g} t.u. (thin: replicas; band: ±1 s.e.; dashed: exp fit; dotted: 1/e)",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(a.plots, f"temporal_corr_{tag}.png")
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    main()
