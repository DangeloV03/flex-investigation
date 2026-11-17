#!/usr/bin/env python3
"""
fss_runner.py: finite-size-scaling (FSS) analysis of one susceptibility campaign, in ONE file.

This is the one file to interact with. Everything you might want to change is a command-line option,
and every step is commented below. Two pieces are imported from this repository (src/flexsim), so run
it from a clone of the repo (any branch with run/fss_runner.py):
    flexsim.susc.fss_fit    the Melchert collapse fit and its S+1 error bars (step 2 below)
    flexsim.susc.mag_jumps  count_jumps, the well-crossing counter J (step 1 below)
Python packages: numpy, scipy, pandas, matplotlib (the "lattice" conda env on Della has them all).

=====================================================================================================
QUICK START
=====================================================================================================

    # Full analysis of one campaign. Point it at the folder that holds the run directories:
    python fss_runner.py /scratch/gpfs/WJACOBS/vd7294/flex-investigation/SUSC_RUNS_POS_DMU1_FITTED_COEX/runs \
        --out fss_pos_dmu1

    # Reading the raw time series is the slow part (minutes for ~160 (eps, L) points). It is cached
    # in <out>/fss_table.csv. To change only the fit or the plots, reuse the table (takes seconds):
    python fss_runner.py --table fss_pos_dmu1/fss_table.csv --out fss_pos_dmu1 --fix-nu 1.0

    # Use several cores for the reading step:
    python fss_runner.py <runs folder> --out <dir> --workers 8

Outputs, all in --out:
    fss_table.csv           one row per (L, eps): replica-averaged chi, <|m|>, U4, <m>, J, and kept/dropped
    observables_vs_eps.png  raw chi, <|m|>, U4 and <m> vs eps for every L (dropped points shown in grey)
    fss_collapse.png        the three data collapses (chi, <|m|>, U4) with the fitted eps_c and exponents
    chi_peak_scaling.png    chi_max(L) vs L on log-log axes. The slope is gamma/nu. Also eps*(L), the peak position.
    fss_fit_results.json    all fitted numbers with their S+1 error bars

=====================================================================================================
WHAT IS MEASURED (the physics in one paragraph)
=====================================================================================================

The model is a lattice gas that maps onto an Ising model. Each site is "bonding" (active) or not. The
order parameter is m = 2*rho_bonding - 1, in [-1, 1]. A run at fixed (eps, L) writes m(t) once per
chunk of kMC time. Near the critical coupling eps_c, the 2D Ising universality class predicts how
observables grow with system size L:

    chi(eps, L)   = L^{ gamma/nu} * F_chi( (eps - eps_c) L^{1/nu} )      Ising: gamma/nu = 7/4, 1/nu = 1
    <|m|>(eps, L) = L^{-beta/nu}  * F_m  ( (eps - eps_c) L^{1/nu} )      Ising: beta/nu  = 1/8
    U4(eps, L)    =                 F_U  ( (eps - eps_c) L^{1/nu} )      U4 has no L prefactor

Here F_* are universal "master curves". If eps_c and the exponents are right, plotting
y * L^{-exponent} against x = (eps - eps_c) L^{1/nu} puts every L on ONE curve: the "data collapse".
We search for the eps_c and exponents that give the best collapse. Because U4 needs no prefactor, its
collapse (or the crossing of the U4(eps) curves) fixes eps_c and 1/nu on their own.

=====================================================================================================
HOW THE DATA ARE READ (specific to this campaign layout)
=====================================================================================================

    <runs folder>/_<L>_<L>_S..._DMU.._K../_<eps>/susceptibility_data.csv    one row per replica per round
                                                 m_timeseries_<id>.csv        m(t) for row <id>

The sweep tops up unconverged points in rounds. Each top-up CONTINUES a replica. It copies that
replica's whole m(t) history into a new m_timeseries_<new id>.csv and deletes the old file. So for each
replica_id we use the newest row (largest prod_time) whose time-series file still exists. That file
holds the full production history.

Per replica r (all time averages are over its production m(t) samples):
    chi_r = beta * N * ( <m^2> - <|m|>^2 )   N = L*L. Subtracting <|m|>^2 rather than <m>^2 gives the
                                             connected susceptibility that stays finite when a replica
                                             flips between the + and - wells (standard for FSS).
    absm_r = <|m|>
    U4_r  = 1 - <m^4> / (3 <m^2>^2)          Binder cumulant: 2/3 deep in the ordered phase, 0 in the
                                             disordered phase, about 0.61 at the 2D Ising critical point.
    m_r   = <m>                              should be ~0 at coexistence. A large |<m>| means mu is off.
    J_r   = number of well-to-well jumps of m(t) (crossing from above 75% of the trace max to below 75%
            of the trace min, or back). Few jumps = the replica has not sampled both wells = unconverged.
Each (L, eps) point is the mean over its replicas, with error = standard error (std / sqrt(n_replicas)).

Quality cuts on the (L, eps) points, both on by default and matching the campaign rules:
    --max-abs-m 0.05   drop if |<m>| > 0.05. The run sat off coexistence, so chi/U4 are biased.
    --min-jumps 10     drop if <J> < 10. Too few well crossings to trust the average (the campaign's
                       own convergence threshold). Pass 0 to keep everything.
Dropped points stay in fss_table.csv (kept = False) and are drawn in grey. They are not used in fits.

=====================================================================================================
HOW THE COLLAPSE IS FITTED
=====================================================================================================

This is the Melchert method (O. Melchert, arXiv:0910.5403, "autoScale.py"). The code is the Python 3 port
in src/flexsim/susc/fss_fit.py (fss_quality, optimise_fss, s_plus_one_fss_errors). What it does:
  1. Rescale every point with trial parameters (eps_c, a = 1/nu, b): x = (eps - eps_c) L^a, y * L^b.
  2. For each rescaled point, take the two neighbouring points (one left, one right in x) from EVERY
     OTHER system size. Fit a straight line through them. That line is the local "master curve" estimate Y +/- dY.
  3. Collapse quality S = mean over points of (y - Y)^2 / (dy^2 + dY^2). S ~ 1 means the points scatter
     about the master curve by their error bars. S >> 1 means a poor collapse (or underestimated errors).
  4. Minimize S over the free parameters (scipy Nelder-Mead).
  5. Error bars: for each parameter, move it (others fixed) until S rises to S_min + 1 (Melchert's
     "S+1" errors). These are asymmetric and are written as +hi/-lo.
With only four sizes (16-64) the three-parameter fit can wander. --fix-nu 1.0 (the Ising value) holds
1/nu fixed, which usually stabilizes eps_c. --xr restricts the fit to a window of the rescaled x.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display (e.g. on a cluster node)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The collapse fit and the jump counter come from this repository (src/flexsim). Putting src/ on the
# path means this works from a plain clone, without "pip install -e .".
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from flexsim.susc.fss_fit import optimise_fss, s_plus_one_fss_errors  # noqa: E402  Melchert collapse fit
from flexsim.susc.mag_jumps import count_jumps  # noqa: E402  well-to-well jump counter (J)

# 2D Ising reference values, used as starting guesses and drawn as guide lines
ISING = {"inv_nu": 1.0, "gamma_nu": 7 / 4, "beta_nu": 1 / 8, "U4_star": 0.6107}

# One fixed colour + marker per system size, so every figure uses the same legend
L_STYLE = {16: ("#2a78d6", "o"), 24: ("#6250d6", "v"), 32: ("#1baf7a", "s"), 40: ("#008300", "P"),
           48: ("#eda100", "D"), 64: ("#e34948", "^"), 96: ("#8a3ffc", "X"), 128: ("#0b0b0b", "*")}


def lstyle(L):
    return L_STYLE.get(int(L), ("gray", "o"))


# =================================================================================================
# STEP 1: read the raw m(t) of every replica and reduce it to one row per (L, eps)
# =================================================================================================

def replica_observables(m: np.ndarray, L: int, beta: float) -> dict:
    """chi, <|m|>, U4, <m>, J for one replica's production m(t)."""
    N = L * L
    m2, m4, am = np.mean(m ** 2), np.mean(m ** 4), np.mean(np.abs(m))
    return dict(chi=beta * N * (m2 - am ** 2),
                absm=am,
                U4=1.0 - m4 / (3.0 * m2 ** 2) if m2 > 0 else np.nan,
                m=np.mean(m),
                J=count_jumps(m))


def read_point(run_dir: str) -> dict | None:
    """Reduce one (L, eps) run folder to replica-averaged observables (one table row)."""
    meta = pd.read_csv(os.path.join(run_dir, "susceptibility_data.csv"))
    if meta.empty:
        return None
    # newest row per replica that still has its time-series file (see the module docstring)
    meta = meta.dropna(subset=["id"])
    meta["ts"] = [os.path.join(run_dir, f"m_timeseries_{int(i)}.csv") for i in meta["id"]]
    meta = meta[[os.path.isfile(p) for p in meta["ts"]]]
    if meta.empty:
        return None
    rep_col = "replica_id" if "replica_id" in meta else "id"          # older runs: one row per replica
    order = "prod_time" if "prod_time" in meta else "id"              # newer round = longer prod_time
    newest = meta.sort_values([order, "id"]).groupby(rep_col).tail(1)

    L, eps, beta = int(newest["L"].iloc[0]), float(newest["epsilon"].iloc[0]), float(newest["beta"].iloc[0])
    reps = []
    for _, row in newest.iterrows():
        m = pd.read_csv(row["ts"], usecols=["m"])["m"].to_numpy(float)
        if m.size > 1:
            reps.append(replica_observables(m, L, beta) | dict(n_samples=m.size))
    if not reps:
        return None
    df = pd.DataFrame(reps)
    n = len(df)
    se = lambda c: df[c].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    out = dict(L=L, epsilon=eps, mu=float(newest["mu"].iloc[0]), n_replicas=n,
               samples_per_replica=int(df.n_samples.median()), run_dir=run_dir)
    for c in ("chi", "absm", "U4", "m", "J"):
        out[c], out[c + "_err"] = df[c].mean(), se(c)
    return out


def build_table(runs_folder: str, workers: int) -> pd.DataFrame:
    """Find every folder with a susceptibility_data.csv under runs_folder and reduce each one."""
    dirs = sorted(os.path.dirname(p) for p in
                  glob.glob(os.path.join(runs_folder, "**", "susceptibility_data.csv"), recursive=True))
    if not dirs:
        raise SystemExit(f"no susceptibility_data.csv found under {runs_folder}")
    print(f"reading {len(dirs)} (L, eps) run folders with {workers} worker(s) ...", flush=True)
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(read_point, dirs, chunksize=4))
    else:
        rows = [read_point(d) for d in dirs]
    return pd.DataFrame([r for r in rows if r]).sort_values(["L", "epsilon"]).reset_index(drop=True)


def apply_cuts(t: pd.DataFrame, max_abs_m: float, min_jumps: float) -> pd.DataFrame:
    """Mark which points enter the fits (kept) and record why the others were dropped."""
    t = t.copy()
    t["drop_reason"] = ["; ".join(r for r, bad in ((f"|<m>|>{max_abs_m:g}", abs(m) > max_abs_m),
                                                    (f"<J><{min_jumps:g}", J < min_jumps)) if bad)
                        for m, J in zip(t.m, t.J)]
    t["kept"] = t.drop_reason == ""
    return t


# =================================================================================================
# STEP 2: the Melchert collapse fit
# =================================================================================================

def to_dataset(t: pd.DataFrame, y: str) -> dict[int, np.ndarray]:
    """{L: array of rows [eps, y, dy]} for the kept points (at least 3 per L, dy > 0)."""
    ds = {}
    for L, sub in t[t.kept].groupby("L"):
        sub = sub.sort_values("epsilon")
        arr = np.column_stack([sub.epsilon, sub[y], np.maximum(sub[y + "_err"].fillna(0), 1e-8)])
        arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr) >= 3:
            ds[int(L)] = arr
    return ds


def fit_collapse(ds, x0, fixed, x_range):
    """Best collapse + S+1 error bars, using the repo's Melchert implementation (flexsim.susc.fss_fit).

    ds     : {L: rows [eps, y, dy]} from to_dataset
    x0     : starting [eps_c, a, b]
    fixed  : parameters held constant, e.g. {"a": 1.0} or {"b": 0.0}
    Returns eps_c, a, b, S and errors[name] = (minus, plus).
    """
    rename = {"eps_c": "xc", "a": "a", "b": "b"}           # fss_fit calls eps_c "xc"
    fx = {rename[k]: v for k, v in fixed.items()}
    best = optimise_fss(ds, x0, x_range, fx)                 # Nelder-Mead minimum of the quality S
    err = s_plus_one_fss_errors(ds, best, x_range, fx)       # walk each parameter until S = S_min + 1
    return dict(eps_c=best["xc"], a=best["a"], b=best["b"], S=best["S"], success=bool(best["success"]),
                errors={n: (err[f"{k}_err_lo"], err[f"{k}_err_hi"]) for n, k in rename.items()})


def pm(v, e, p=4):
    lo, hi = e
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{v:.{p}f} (err n/a)"
    if lo == hi == 0:
        return f"{v:.{p}f} (fixed)"
    return f"{v:.{p}f} +{hi:.{p}f}/-{lo:.{p}f}"


# =================================================================================================
# STEP 3: plots
# =================================================================================================

def style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_raw(t, eps_c, out, title):
    """chi, <|m|>, U4, <m> vs eps for every L; points cut from the fit are grey crosses."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    panels = [("chi", r"$\chi$", "log"), ("absm", r"$\langle|m|\rangle$", "linear"),
              ("U4", r"Binder $U_4$", "linear"), ("m", r"$\langle m\rangle$  (should be ≈ 0)", "linear")]
    for ax, (c, lab, sc) in zip(axes.flat, panels):
        for L, sub in t.groupby("L"):
            col, mk = lstyle(L)
            k = sub[sub.kept]
            ax.errorbar(k.epsilon, k[c], yerr=k[c + "_err"], fmt=mk + "-", ms=4, color=col, lw=0.9,
                        mfc="white", capsize=0, label=f"L = {L}")
            d = sub[~sub.kept]
            ax.plot(d.epsilon, d[c], "x", color="#a0a0a0", ms=4)
        if np.isfinite(eps_c):
            ax.axvline(eps_c, color="#52514e", ls=":", lw=1)
        if c == "U4":
            ax.axhline(ISING["U4_star"], color="#52514e", ls="--", lw=0.8)
            ax.text(0.01, 0.02, "dashed: 2D Ising U4* = 0.611;  dotted: fitted eps_c",
                    transform=ax.transAxes, fontsize=7.5, color="#52514e")
        if c == "m":
            ax.axhspan(-0.05, 0.05, color="#e3efe3", zorder=0)
        ax.set_yscale(sc)
        ax.set_xlabel(r"$\beta\varepsilon$")
        ax.set_ylabel(lab)
        style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].text(0.99, 0.02, "grey ×: dropped by the cuts (not fitted)", transform=axes[0, 1].transAxes,
                    ha="right", fontsize=7.5, color="#52514e")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out, "observables_vs_eps.png"), dpi=160)
    plt.close(fig)


def plot_collapses(datasets, fits, out, title):
    """Rescaled data using each observable's own best-fit parameters."""
    spec = [("chi", r"$\chi\,L^{-\gamma/\nu}$", "log"), ("absm", r"$\langle|m|\rangle\,L^{\beta/\nu}$", "linear"),
            ("U4", r"$U_4$", "linear")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (c, ylab, sc) in zip(axes, spec):
        f, ds = fits.get(c), datasets.get(c)
        if not f or not ds:
            ax.set_visible(False)
            continue
        for L, d in sorted(ds.items()):
            col, mk = lstyle(L)
            ax.errorbar((d[:, 0] - f["eps_c"]) * L ** f["a"], d[:, 1] * L ** f["b"], yerr=d[:, 2] * L ** f["b"],
                        fmt=mk, ms=4, color=col, mfc="white", capsize=0, label=f"L = {L}")
        e = f["errors"]
        lines = [rf"$\varepsilon_c$ = {pm(f['eps_c'], e['eps_c'])}", rf"$1/\nu$ = {pm(f['a'], e['a'], 3)}"]
        if c == "chi":
            lines.append(rf"$\gamma/\nu$ = {pm(-f['b'], e['b'], 3)}  (Ising 1.75)")
        elif c == "absm":
            lines.append(rf"$\beta/\nu$ = {pm(f['b'], e['b'], 3)}  (Ising 0.125)")
        lines.append(f"S = {f['S']:.2f}")
        ax.set_title("\n".join(lines), fontsize=8.5, loc="left")
        ax.set_yscale(sc)
        ax.set_xlabel(r"$(\varepsilon-\varepsilon_c)\,L^{1/\nu}$")
        ax.set_ylabel(ylab)
        style(ax)
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(out, "fss_collapse.png"), dpi=160)
    plt.close(fig)


def chi_peaks(t):
    """Peak of chi(eps) per L from a parabola through the top 3 kept points (finer than the eps grid)."""
    rows = []
    for L, sub in t[t.kept].groupby("L"):
        sub = sub.sort_values("epsilon").reset_index(drop=True)
        if len(sub) < 3:
            continue
        i = int(np.clip(sub.chi.idxmax(), 1, len(sub) - 2))
        x, y = sub.epsilon[i - 1:i + 2].to_numpy(), sub.chi[i - 1:i + 2].to_numpy()
        A, B, C = np.polyfit(x, y, 2)
        e_star = -B / (2 * A) if A < 0 else x[1]
        e_star = float(np.clip(e_star, x[0], x[2]))
        rows.append(dict(L=L, eps_star=e_star, chi_max=float(np.polyval([A, B, C], e_star)),
                         chi_max_err=float(sub.chi_err[i]), at_edge=bool(sub.chi.idxmax() in (0, len(sub) - 1))))
    return pd.DataFrame(rows)


def plot_peaks(pk, out, title):
    """chi_max ~ L^{gamma/nu}: the slope on log-log axes is gamma/nu (independent of the collapse fit)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    res = {}
    if len(pk) >= 2:
        ax = axes[0]
        # A peak at the edge of the kept points is not a real maximum (the cuts removed the other side),
        # so it is drawn as a grey cross and left out of the slope fit.
        ok = pk[~pk.at_edge]
        if len(ok) >= 2:
            sl, ic = np.polyfit(np.log(ok.L), np.log(ok.chi_max), 1)
        else:
            sl, ic = np.nan, np.nan
        res["gamma_nu_from_peaks"] = float(sl)
        res["peak_sizes_used"] = [int(L) for L in ok.L]
        ax.errorbar(ok.L, ok.chi_max, yerr=ok.chi_max_err, fmt="o", color="#2a78d6", mfc="white")
        ax.plot(pk.L[pk.at_edge], pk.chi_max[pk.at_edge], "x", color="#a0a0a0", ms=7,
                label="peak at edge of kept points (not a true max, not fitted)")
        LL = np.linspace(pk.L.min() * 0.9, pk.L.max() * 1.1, 50)
        ax.plot(LL, np.exp(ic) * LL ** sl, "-", color="#2a78d6", label=rf"fit slope $\gamma/\nu$ = {sl:.3f}  (L = {', '.join(str(int(L)) for L in ok.L)})")
        ax.plot(LL, np.exp(ic) * LL.min() ** (sl - 1.75) * LL ** 1.75, ":", color="#52514e", label="Ising slope 7/4")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("L"); ax.set_ylabel(r"$\chi_{max}(L)$")
        ax.legend(frameon=False, fontsize=8)
        style(ax)
        ax = axes[1]
        ax.plot(pk.L, pk.eps_star, "o-", color="#e34948")
        ax.plot(pk.L[pk.at_edge], pk.eps_star[pk.at_edge], "x", color="#a0a0a0", ms=9, mew=2,
                label="at edge of kept points (unreliable)")
        if pk.at_edge.any():
            ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("L"); ax.set_ylabel(r"peak position $\varepsilon^*(L)$")
        ax.text(0.02, 0.02, r"$\varepsilon^*(L) - \varepsilon_c \propto L^{-1/\nu}$", transform=ax.transAxes, fontsize=8)
        style(ax)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "chi_peak_scaling.png"), dpi=160)
    plt.close(fig)
    return res


# =================================================================================================
# main: argument parsing and the three steps
# =================================================================================================

def main(argv=None):
    """Run the whole analysis. argv is a list of command-line words (None = use sys.argv), so other
    scripts can call it directly, e.g. main(["SUSC_RUNS_X/runs", "--out", "fss_x"])."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="?", help="campaign runs folder (contains _L_L_.../_eps/ run folders)")
    ap.add_argument("--table", help="reuse a cached fss_table.csv instead of re-reading the time series")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--title", default=None, help="figure title (default: the runs folder name)")
    ap.add_argument("--workers", type=int, default=1, help="processes for reading the time series")
    ap.add_argument("--max-abs-m", type=float, default=0.05, help="drop points with |<m>| above this")
    ap.add_argument("--min-jumps", type=float, default=10, help="drop points with <J> below this (0 = keep all)")
    ap.add_argument("--sizes", type=int, nargs="+", help="only use these L (default: all)")
    ap.add_argument("--eps-c", type=float, help="starting guess for eps_c (default: U4 curve crossing)")
    ap.add_argument("--fix-nu", type=float, help="hold 1/nu fixed at this value (e.g. 1.0 = Ising)")
    ap.add_argument("--fix-eps-c", action="store_true", help="hold eps_c fixed at --eps-c")
    ap.add_argument("--xr", type=float, nargs=2, metavar=("XMIN", "XMAX"),
                    help="only fit points with rescaled x = (eps-eps_c) L^{1/nu} in this window")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    # ---- step 1: table of replica-averaged observables -----------------------------------------
    if a.table:
        t = pd.read_csv(a.table)
        t = t.drop(columns=[c for c in ("kept", "drop_reason") if c in t])
    elif a.runs:
        t = build_table(os.path.abspath(a.runs), a.workers)
    else:
        ap.error("give a runs folder or --table")
    if a.sizes:
        t = t[t.L.isin(a.sizes)]
    t = apply_cuts(t, a.max_abs_m, a.min_jumps)
    t.to_csv(os.path.join(a.out, "fss_table.csv"), index=False)
    title = a.title or os.path.basename(os.path.normpath(os.path.dirname(os.path.abspath(a.runs))
                                                         if a.runs else os.path.abspath(a.out)))
    print(f"{len(t)} (L, eps) points, {int(t.kept.sum())} kept after cuts "
          f"(|<m>| <= {a.max_abs_m}, <J> >= {a.min_jumps})")
    for L, sub in t.groupby("L"):
        print(f"  L={L:3d}: {int(sub.kept.sum())}/{len(sub)} kept, eps {sub.epsilon.min():.4f}..{sub.epsilon.max():.4f}")

    # ---- step 2: collapse fits --------------------------------------------------------------------
    # starting eps_c: where U4 of the two largest sizes cross (else the chi peak of the largest L)
    eps0 = a.eps_c
    if eps0 is None:
        k = t[t.kept]
        Ls = sorted(k.L.unique())
        if len(Ls) >= 2:
            A_, B_ = (k[k.L == L].sort_values("epsilon") for L in Ls[-2:])
            grid = np.intersect1d(A_.epsilon.round(6), B_.epsilon.round(6))
            dU = np.interp(grid, B_.epsilon, B_.U4) - np.interp(grid, A_.epsilon, A_.U4)
            sgn = np.where(np.diff(np.sign(dU)) != 0)[0]
            if len(sgn):
                i = sgn[len(sgn) // 2]
                eps0 = float(grid[i] - dU[i] * (grid[i + 1] - grid[i]) / (dU[i + 1] - dU[i]))
        if eps0 is None and len(k):
            eps0 = float(k.loc[k[k.L == k.L.max()].chi.idxmax(), "epsilon"])
    if eps0 is None:
        raise SystemExit("no (L, eps) point survived the cuts, nothing to fit "
                         "(loosen --max-abs-m / --min-jumps, or check the runs folder)")
    print(f"starting eps_c guess: {eps0:.4f}")

    fixed_base = {}
    if a.fix_nu is not None:
        fixed_base["a"] = a.fix_nu
    if a.fix_eps_c:
        fixed_base["eps_c"] = eps0
    xr = tuple(a.xr) if a.xr else (-np.inf, np.inf)
    inv_nu0 = a.fix_nu if a.fix_nu is not None else ISING["inv_nu"]
    starts = {"chi": [eps0, inv_nu0, -ISING["gamma_nu"]],   # b = -gamma/nu
              "absm": [eps0, inv_nu0, ISING["beta_nu"]],     # b = +beta/nu
              "U4": [eps0, inv_nu0, 0.0]}                    # b = 0 (U4 is scale-free)
    datasets, fits = {}, {}
    for c, x0 in starts.items():
        ds = to_dataset(t, c)
        if len(ds) < 2:
            print(f"{c}: fewer than 2 sizes with >= 3 kept points, no fit")
            continue
        fixed = dict(fixed_base) | ({"b": 0.0} if c == "U4" else {})
        print(f"fitting {c} collapse ...", flush=True)
        datasets[c], fits[c] = ds, fit_collapse(ds, x0, fixed, xr)
        f = fits[c]
        print(f"  eps_c = {pm(f['eps_c'], f['errors']['eps_c'], 5)}   1/nu = {pm(f['a'], f['errors']['a'], 3)}"
              f"   b = {pm(f['b'], f['errors']['b'], 3)}   S = {f['S']:.2f}")

    # ---- step 3: plots and results file ---------------------------------------------------------
    eps_best = fits["U4"]["eps_c"] if "U4" in fits else fits.get("chi", {}).get("eps_c", np.nan)
    plot_raw(t, eps_best, a.out, title)
    plot_collapses(datasets, fits, a.out, title)
    pk = chi_peaks(t)
    peak_res = plot_peaks(pk, a.out, title)
    res = dict(runs=a.runs, table=a.table, cuts=dict(max_abs_m=a.max_abs_m, min_jumps=a.min_jumps),
               n_points=int(len(t)), n_kept=int(t.kept.sum()), fixed=fixed_base, x_range=a.xr,
               fits={c: dict(eps_c=f["eps_c"], eps_c_err=f["errors"]["eps_c"], inv_nu=f["a"],
                             inv_nu_err=f["errors"]["a"], b=f["b"], b_err=f["errors"]["b"], S=f["S"],
                             meaning_of_b={"chi": "-gamma/nu", "absm": "beta/nu", "U4": "0 (fixed)"}[c])
                     for c, f in fits.items()},
               chi_peaks=pk.to_dict("records"), **peak_res)
    with open(os.path.join(a.out, "fss_fit_results.json"), "w") as fh:
        json.dump(clean_json(res), fh, indent=2)
    print(f"wrote {a.out}/: fss_table.csv observables_vs_eps.png fss_collapse.png chi_peak_scaling.png "
          f"fss_fit_results.json")
    return res


def clean_json(x):
    """numpy numbers -> plain Python, NaN/inf -> null, so fss_fit_results.json is strict JSON."""
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


if __name__ == "__main__":
    main()
