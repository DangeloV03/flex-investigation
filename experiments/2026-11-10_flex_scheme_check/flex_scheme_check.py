"""Mentor item 4: is the FLEX curve in the flex-vs-sim plot the right one for each driving scheme?

For every COEX_RUNS_<EXP>/ly<N>/manage.csv under --runs-root, this script:
  1. Recomputes FLEX the way the pipeline does it (flexsim.coex.plot_flex_vs_sim.flex_curve,
     k_eff = min(1, e^{-+2eps - df - dmu}) i.e. k conditioned on n = 2 neighbours) and checks it
     equals the single-site n=2 model, and how far it is from the homogeneous-drive curve.
  2. Tabulates k_n for n = 0..4 per campaign (shows where the n = 2 rate is saturated at k).
  3. Compares measured mu_coex with four predictions: homo, n2 (= current FLEX), avg_p, avg_h
     (see single_site.py) and writes RMS residuals.

Outputs (--outdir):
  pipeline_check.csv     max |pipeline FLEX - n2| and max |n2 - homo| per experiment
  k_n_table.csv          k_n(n = 0..4) at the coldest/central/warmest measured eps
  points.csv             every measured point with all four predictions
  rms_summary.csv        RMS(sim - prediction) per experiment and Ly
  overview.png/.pdf      mu_coex vs eps, all variants + data
  residual_overview.png  sim - prediction for n2 and avg_h

Usage:  python flex_scheme_check.py --runs-root <dir with COEX_RUNS_*> --outdir <dir> [--stride 1]
"""
import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import single_site as ss  # noqa: E402

try:
    from flexsim.coex.plot_flex_vs_sim import flex_curve as pipeline_flex_curve
except ImportError:  # run outside the flexsim env: fall back to the repo src/
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    from flexsim.coex.plot_flex_vs_sim import flex_curve as pipeline_flex_curve

VARIANTS = ["homo", "n2", "avg_p", "avg_h"]
VSTYLE = {"homo": dict(color="#9a9893", ls="--", lw=1.2, label="FLEX, homogeneous k"),
          "n2": dict(color="#0b0b0b", ls="-", lw=1.5, label="FLEX, k at n=2 (current plot)"),
          "avg_p": dict(color="#8a5cd1", ls="-.", lw=1.3, label=r"single-site, $\langle P_B\rangle_n=1/2$"),
          "avg_h": dict(color="#d6409f", ls=":", lw=1.8, label=r"single-site, $\langle h\rangle_n=0$")}
LY_COLOR = {16: "#2a78d6", 20: "#eb6834", 40: "#1baf7a"}
ROWS = ["homo", "negative_drive", "positive_drive"]
COLS = [-1.0, 1.0, 2.0, 3.0, 4.0]


def load(runs_root, max_err):
    frames = []
    for path in sorted(glob.glob(os.path.join(runs_root, "COEX_RUNS_*", "ly*", "manage.csv"))):
        exp = re.search(r"COEX_RUNS_([^/]+)/", path).group(1)
        d = pd.read_csv(path)
        col = "mu_coex_FITTED" if "mu_coex_FITTED" in d else "mu_coex_SIM"
        d = d.rename(columns={col: "mu_sim", col + "_error": "mu_sim_err"})
        d = d.dropna(subset=["mu_sim"])
        d = d[d.mu_sim_err < max_err]
        d["experiment"] = exp
        frames.append(d[["experiment", "scheme", "delta_f", "delta_mu", "k", "Ly", "epsilon", "mu_sim", "mu_sim_err"]])
    if not frames:
        raise SystemExit(f"no COEX_RUNS_*/ly*/manage.csv under {runs_root}")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stride", type=int, default=1, help="use every n-th point (speed)")
    ap.add_argument("--max-err", type=float, default=0.01, help="drop fits with a larger mu error")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    meas = load(a.runs_root, a.max_err)
    meas = meas.groupby(["experiment", "Ly"], group_keys=False).apply(lambda g: g.iloc[:: a.stride])

    # 1. pipeline FLEX == single-site n2 ?  and how far from homo?
    checks, ktab, curves = [], [], []
    for exp, g in meas.groupby("experiment"):
        r = g.iloc[0]
        args = (r.delta_f, r.delta_mu, r.k, r.scheme)
        eps = np.round(np.arange(g.epsilon.min() - 0.05, g.epsilon.max() + 0.05, 0.005), 6)
        pipe = pipeline_flex_curve(r.scheme, r.delta_f, r.delta_mu, r.k, eps)
        var = {v: np.array([ss.mu_coex(e, *args, variant=v) for e in eps]) for v in VARIANTS}
        checks.append(dict(experiment=exp, scheme=r.scheme, dmu=r.delta_mu,
                           max_abs_pipeline_minus_n2=np.nanmax(np.abs(pipe - var["n2"])),
                           max_abs_n2_minus_homo=np.nanmax(np.abs(var["n2"] - var["homo"])),
                           frac_eps_k2_saturated=np.mean([ss.k_n(2, e, *args) >= r.k for e in eps])))
        curves.append(pd.DataFrame(dict(experiment=exp, epsilon=eps, pipeline=pipe, **var)))
        for e in [g.epsilon.min(), g.epsilon.median(), g.epsilon.max()]:
            ktab.append(dict(experiment=exp, scheme=r.scheme, dmu=r.delta_mu, epsilon=round(e, 4),
                             **{f"k_n{n}": ss.k_n(n, e, *args) for n in range(5)}))
    checks = pd.DataFrame(checks)
    curves = pd.concat(curves, ignore_index=True)
    checks.to_csv(os.path.join(a.outdir, "pipeline_check.csv"), index=False)
    pd.DataFrame(ktab).to_csv(os.path.join(a.outdir, "k_n_table.csv"), index=False)
    print("Pipeline FLEX vs single-site n=2 (should be ~0) and vs homogeneous k:")
    print(checks.round(5).to_string(index=False))

    # 3. predictions at every measured point
    for v in VARIANTS:
        meas[v] = [ss.mu_coex(r.epsilon, r.delta_f, r.delta_mu, r.k, r.scheme, variant=v)
                   for r in meas.itertuples()]
        meas["res_" + v] = meas.mu_sim - meas[v]
    meas.to_csv(os.path.join(a.outdir, "points.csv"), index=False)
    rms = meas.groupby(["experiment", "scheme", "delta_mu", "Ly"]).apply(
        lambda g: pd.Series({"n": len(g), **{f"rms_{v}": np.sqrt(np.nanmean(g["res_" + v] ** 2)) for v in VARIANTS}}))
    rms = rms.reset_index()
    rms.to_csv(os.path.join(a.outdir, "rms_summary.csv"), index=False)
    print("\nRMS(sim - prediction):")
    print(rms.round(4).to_string(index=False))

    # figures
    for residual in (False, True):
        fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(18, 10), squeeze=False)
        for i, sch in enumerate(ROWS):
            for j, dmu in enumerate(COLS):
                ax = axes[i, j]
                g = meas[(meas.scheme == sch) & np.isclose(meas.delta_mu, dmu)]
                if g.empty:
                    ax.axis("off")
                    continue
                exp = g.experiment.iloc[0]
                c = curves[curves.experiment == exp]
                if residual:
                    ax.axhline(0, color="#0b0b0b", lw=1)
                    for ly, gl in g.groupby("Ly"):
                        ax.plot(gl.epsilon, gl.res_n2, "o", ms=2.5, color=LY_COLOR.get(ly, "k"),
                                label=f"sim − n2, Ly={ly}")
                        ax.plot(gl.epsilon, gl.res_avg_h, "x", ms=3, color=LY_COLOR.get(ly, "k"), alpha=0.6,
                                label=f"sim − avg_h, Ly={ly}")
                else:
                    for v in VARIANTS:
                        ax.plot(c.epsilon, c[v], **VSTYLE[v])
                    for ly, gl in g.groupby("Ly"):
                        ax.plot(gl.epsilon, gl.mu_sim, "o", ms=2.5, color=LY_COLOR.get(ly, "k"),
                                mec="white", mew=0.3, label=f"sim Ly={ly}")
                ax.set_title(f"{exp}: {sch}, βΔμ={dmu:+g}", fontsize=9)
                ax.tick_params(labelsize=7)
                ax.grid(True, color="#e4e3df", lw=0.5)
        h, l = [], []
        for ax in axes.ravel():
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll not in l:
                    h.append(hh)
                    l.append(ll)
        fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.99, 0.97), fontsize=8, frameon=False)
        fig.suptitle("Coexistence μ: FLEX variants vs simulation" + (" — residuals (o: n2, x: avg_h)" if residual else "")
                     + "   [βΔf = 0, k = 1]", fontsize=12)
        fig.supxlabel(r"$\beta\varepsilon$")
        fig.supylabel(r"$\beta\mu_{\rm sim}-\beta\mu_{\rm pred}$" if residual else r"$\beta\mu_{\rm coex}$")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        base = os.path.join(a.outdir, "residual_overview" if residual else "overview")
        fig.savefig(base + ".png", dpi=170)
        if not residual:
            fig.savefig(base + ".pdf")
        plt.close(fig)
    print(f"\nWrote {a.outdir}")


if __name__ == "__main__":
    main()
