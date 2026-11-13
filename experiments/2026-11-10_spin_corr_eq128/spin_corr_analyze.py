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


N_BLOCKS = 40   # time blocks per replica, recorded for batch-means errors (merged to 20 and 10 as a check)


def replica_sums(snaps, n_blocks=N_BLOCKS):
    """Sum over snapshots of the 2D <n_i n_{i+d}> map, of rho, and of rho^2 (per snapshot).
    Also returns, per consecutive time block, the radially averaged <n_0 n_r> and the block density."""
    acc, rho_sum, rho2_sum = None, 0.0, 0.0
    edges = np.linspace(0, len(snaps), min(n_blocks, len(snaps)) + 1).astype(int)
    blocks = []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        bacc, brho = None, 0.0
        for p in snaps[b0:b1]:
            n = (np.load(p) == BONDING).astype(float)
            N = n.size
            F = np.fft.fft2(n)
            c2 = np.fft.ifft2(F * np.conj(F)).real / N
            bacc = c2 if bacc is None else bacc + c2
            r = n.mean()
            brho += r
            rho2_sum += r * r
        acc = bacc if acc is None else acc + bacc
        rho_sum += brho
        blocks.append((radial(bacc / (b1 - b0))[1], brho / (b1 - b0)))
    return acc, rho_sum, rho2_sum, len(snaps), blocks


def block_err(vals, merge=1):
    """Standard error of the mean over blocks (rows of vals), after merging `merge` adjacent blocks."""
    v = np.asarray(vals)
    nb = (len(v) // merge) * merge
    if nb < 2 * merge:
        return np.full(v.shape[1], np.nan)
    v = v[:nb].reshape(nb // merge, merge, -1).mean(axis=1)
    return v.std(axis=0, ddof=1) / np.sqrt(len(v))


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
    all_blocks = []   # per replica: list of (radial <n0 nr>, rho) per time block
    taus = []
    for rd in reps:
        snaps = sorted(glob.glob(os.path.join(rd, "snap_*.npy")))
        if not snaps:
            continue
        acc, rs, r2s, n, blocks = replica_sums(snaps)
        per_rep.append((acc / n, rs / n, r2s / n))
        all_blocks.append(blocks)
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
    # replica spread (each replica vs the common rho, so the errors describe C_glob). Near eps_c each
    # replica's own density differs (rare flips), which shifts its whole curve: these errors are then huge.
    reps_C = np.array([radial(a - rho ** 2)[1] for a, _, _ in per_rep])
    nrep = len(per_rep)
    err = reps_C.std(axis=0, ddof=1) / np.sqrt(nrep) if nrep > 1 else np.full_like(C_glob, np.nan)
    # full spin correlation <s_0 s_r>, s = 2n - 1, with each replica's OWN density:
    # <s0 sr> = 4<n0 nr> - 4 rho_rep + 1. It is the same for the + and - phase, so no offset.
    reps_S = np.array([radial(4 * a - 4 * rs + 1)[1] for a, rs, _ in per_rep])
    S = reps_S.mean(axis=0)
    S_err_rep = reps_S.std(axis=0, ddof=1) / np.sqrt(nrep) if nrep > 1 else np.full_like(S, np.nan)

    # batch-means errors: merge adjacent time blocks WITHIN each replica, then pool all replicas'
    # blocks. merge=2 gives blocks of 1/20 of a run (1e6 time units for the default 2e7 run).
    def pooled(merge, fn):
        rows = []
        for blocks in all_blocks:
            nb = (len(blocks) // merge) * merge
            for k in range(0, nb, merge):
                nn_b = np.mean([b[0] for b in blocks[k:k + merge]], axis=0)
                rho_b = np.mean([b[1] for b in blocks[k:k + merge]])
                rows.append(fn(nn_b, rho_b))
        return np.array(rows)
    S_fn = lambda nn_b, rho_b: 4 * nn_b - 4 * rho_b + 1
    C_fn = lambda nn_b, rho_b: nn_b - rho ** 2        # global rho, same convention as C_glob
    err_by_merge = {m: (block_err(pooled(m, S_fn)), block_err(pooled(m, C_fn))) for m in (1, 2, 4)}
    S_err, C_err = err_by_merge[2]
    df = pd.DataFrame(dict(r=r, C_glob=C_glob, C_glob_err=C_err, C_glob_err_rep=err, C_snap=C_snap,
                           C_axis=C_axis, G_spin=4 * C_glob, S_full=S, S_full_err=S_err,
                           S_full_err_rep=S_err_rep,
                           **{f"S_full_err_blk{m}": err_by_merge[m][0] for m in (1, 2, 4)},
                           **{f"C_glob_err_blk{m}": err_by_merge[m][1] for m in (1, 2, 4)},
                           **{f"S_rep{i}": reps_S[i] for i in range(nrep)},
                           # connected C per replica with its own density (shape check, no offset)
                           **{f"C_rep{i}": radial(a - rs ** 2)[1] for i, (a, rs, _) in enumerate(per_rep)}))
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


def oz(r, A, xi):
    """Ornstein-Zernike form in 2D."""
    return A * np.exp(-r / xi) / np.sqrt(r)


def fit_xi(df, L, r_min=10, r_max=40):
    """OZ fit of C_glob. Starts at r = 10: shorter windows are pre-asymptotic and overestimate xi."""
    r_max = min(r_max, L / 2 - 1)
    m = (df.r >= r_min) & (df.r <= r_max) & (df.C_glob > 0)
    if m.sum() < 3:
        m = (df.r >= 2) & (df.r <= L / 4) & (df.C_glob > 0)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    try:
        p, cov = curve_fit(oz, df.r[m], df.C_glob[m], p0=[df.C_glob[m].iloc[0] * 3, 5.0], maxfev=20000)
        if not (0 < p[1] < L):   # no finite correlation length inside the box
            return np.nan, np.nan, np.nan
        return p[1], np.sqrt(cov[1, 1]), p[0]
    except Exception:
        return np.nan, np.nan, np.nan


def fit_eta(df, r_min=2, r_max=16):
    """Effective eta from <s0 sr> ~ r^-eta (log-log slope)."""
    m = (df.r >= r_min) & (df.r <= r_max) & (df.S_full > 0)
    if m.sum() < 3:
        return np.nan
    return -np.polyfit(np.log(df.r[m]), np.log(df.S_full[m]), 1)[0]


EPS_C = -2 * np.log(1 + np.sqrt(2))
REP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK2, GRID = "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_correlations(curves, summ, path):
    """One panel per eps: C(r) + OZ fit away from eps_c, <s0 sr> log-log at eps_c."""
    items = sorted(curves.items())
    fig, axes = plt.subplots(1, len(items), figsize=(6 * len(items), 4.4), squeeze=False)
    for ax, (eps, (df, L)) in zip(axes[0], items):
        row = summ[np.isclose(summ.eps, eps)].iloc[0]
        nrep = sum(c.startswith("S_rep") for c in df.columns)
        if abs(eps - EPS_C) < 1e-3:
            d = df[(df.r >= 1) & (df.S_full > 0)]
            for i in range(nrep):
                ax.plot(d.r, d[f"S_rep{i}"], color=REP_COLORS[i], lw=0.8, alpha=0.6, label=f"replica {i}")
            ax.errorbar(d.r, d.S_full, yerr=d.S_full_err, fmt="o", ms=3.5, color="#0b0b0b", ecolor="#0b0b0b",
                        elinewidth=0.8, capsize=0, label="mean ± s.e. (time blocks)", zorder=5)
            rr = np.arange(1, L // 2 + 1)
            A = float(d.S_full[d.r == 2].iloc[0]) * 2 ** 0.25
            ax.plot(rr, A * rr ** -0.25, ":", color="#0b0b0b", lw=1.2, label=r"$r^{-1/4}$ (exact $\eta=1/4$)")
            ax.axhline(row.absm ** 2, color=INK2, lw=1, ls="--", label=r"$\langle|m|\rangle^2$ (finite-size plateau)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylabel(r"$\langle s_0 s_r\rangle$   ($s=\pm1$: active / not)")
            ax.set_title(f"βε = {eps:g} (ε_c): power law, fitted η = {row.eta_fit:.3f} for r = 2–16", fontsize=10)
        else:
            d = df[(df.r >= 1) & (df.C_glob > 0)]
            for i in range(nrep):
                ax.plot(d.r, d[f"C_rep{i}"], color=REP_COLORS[i], lw=0.8, alpha=0.6, label=f"replica {i}")
            ax.errorbar(d.r, d.C_glob, yerr=d.C_glob_err, fmt="o", ms=3.5, color="#0b0b0b", elinewidth=0.8,
                        capsize=0, label="mean ± s.e. (time blocks)", zorder=5)
            rr = np.linspace(2, min(60, L / 2), 200)
            if np.isfinite(row.xi_fit):
                ax.plot(rr, oz(rr, row.oz_A, row.xi_fit), "-", color="#0b0b0b", lw=1.2,
                        label=f"OZ fit, r = 10–40: ξ = {row.xi_fit:.2f}")
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(d.C_glob.min() / 2, 1e-6))
            ax.set_ylabel(r"$C(r)=\langle n_0 n_r\rangle-\rho^2$")
            ax.set_title(f"βε = {eps:g}: exponential decay, ξ = {row.xi_fit:.2f} (exact Ising {row.xi_ising_exact:.2f})",
                         fontsize=10)
        ax.set_xlabel("r (lattice units)")
        _style(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"Active–active spatial correlation, equilibrium, L = {items[0][1][1]}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_m(out, path, max_points=4000):
    """Per eps (column): one m(t) strip per replica, then the pooled production histogram of m."""
    edirs = sorted(glob.glob(os.path.join(out, "eps*")))
    reps = [sorted(glob.glob(os.path.join(e, "replica_*"))) for e in edirs]
    nrep = max(len(r) for r in reps)
    fig = plt.figure(figsize=(6.5 * len(edirs), 1.3 * nrep + 2.8))
    gs = fig.add_gridspec(nrep + 2, len(edirs), height_ratios=[1] * nrep + [0.55, 2.2], hspace=0.15, wspace=0.18)
    for j, (edir, rds) in enumerate(zip(edirs, reps)):
        meta = json.load(open(os.path.join(rds[0], "meta.json")))
        pooled = []
        for i, rd in enumerate(rds):
            ax = fig.add_subplot(gs[i, j])
            ts = read_ts(rd)
            step = max(1, len(ts) // max_points)
            t6 = ts.t.values[::step] / 1e6
            ax.plot(t6, ts.m.values[::step], lw=0.5, color=REP_COLORS[i])
            t_eq = ts[ts.phase == "eq"].t.max() / 1e6 if (ts.phase == "eq").any() else None
            if t_eq:
                ax.axvspan(0, t_eq, color=GRID, alpha=0.8, lw=0)
            ax.axhline(0, color=INK2, lw=0.5)
            ax.set_ylim(-0.9, 0.9)
            ax.set_yticks([-0.5, 0, 0.5])
            ax.text(0.995, 0.92, f"replica {i}", transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color=INK2)
            if i == 0:
                ax.set_title(f"βε = {meta['eps']:g}" + ("  (ε_c)" if abs(meta["eps"] - EPS_C) < 1e-3 else "")
                             + "   — shaded: equilibration", fontsize=10)
            if j == 0:
                ax.set_ylabel("m", fontsize=9)
            if i < len(rds) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"t  ($10^6$ time units)", fontsize=9)
            _style(ax)
            pooled.append(ts[ts.phase == "prod"].m.values)
        ax = fig.add_subplot(gs[nrep + 1, j])   # row nrep is a spacer
        allm = np.concatenate(pooled)
        ax.hist(allm, bins=np.linspace(-0.9, 0.9, 91), density=True, color="#2a78d6", alpha=0.85,
                edgecolor="white", linewidth=0.3)
        L = meta["L"]
        if abs(meta["eps"] - EPS_C) < 1e-3:
            for s in (-1, 1):
                ax.axvline(s * L ** -0.125, color="#0b0b0b", ls=":", lw=1)
            ax.text(0.99, 0.95, rf"dotted: $\pm L^{{-1/8}}$ = ±{L ** -0.125:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color=INK2)
        ax.set_xlabel("m (production, all replicas)", fontsize=9)
        if j == 0:
            ax.set_ylabel("density", fontsize=9)
        _style(ax)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
        xi, xi_err, A = fit_xi(df, L)
        rows.append(dict(eps=eps, L=L, n_snap=int(tau.n_snap.sum()), rho=rho, m=2 * rho - 1,
                         absm=tau.absm_mean.mean(), tau_m=tau.tau_m.mean(), tau_absm=tau.tau_absm.mean(),
                         n_eff_total=tau.n_eff.sum(), xi_fit=xi, xi_fit_err=xi_err, oz_A=A,
                         xi_ising_exact=ising_xi(eps), eta_fit=fit_eta(df)))
        curves[eps] = (df, L)
        print(f"\n== {tag}  rho={rho:.4f}  snapshots={int(tau.n_snap.sum())}")
        chk = df[df.r.isin([1, 4, 16, 32])][["r", "S_full", "S_full_err_rep", "S_full_err_blk1",
                                            "S_full_err_blk2", "S_full_err_blk4"]]
        print("error on <s0 sr>: replica spread vs batch means (blk1/2/4 = 1/40, 1/20, 1/10 of a run):")
        print(chk.to_string(index=False, float_format=lambda v: f"{v:.2e}"))
        print(tau.round(4).to_string(index=False))

    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.plots, "summary.csv"), index=False)
    print("\n" + summ.round(4).to_string(index=False))

    plot_correlations(curves, summ, os.path.join(a.plots, "correlations.png"))
    plot_m(a.out, os.path.join(a.plots, "m_timeseries.png"))
    # the older single-panel C(r) plots are superseded by correlations.png
    for old in ("C_r.png", "C_r_loglog.png"):
        if os.path.isfile(os.path.join(a.plots, old)):
            os.remove(os.path.join(a.plots, old))
    print(f"\nWrote {a.plots}")


if __name__ == "__main__":
    main()
