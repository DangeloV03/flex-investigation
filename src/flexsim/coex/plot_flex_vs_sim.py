"""FLEX-predicted vs simulated coexistence points in the (βε, βμ) plane.

Reads every ``COEX_RUNS_<EXP>/ly<N>/manage.csv`` under ``--runs-root`` (a Della
checkout or a local snapshot of it), and writes to ``--outdir``:

  measured_coex_points.csv     one row per fitted (experiment, Ly, ε) point
  flex_prediction_curves.csv   dense FLEX μ_coex(ε) curve per experiment
  flex_vs_sim_overview.{png,pdf}         all experiments, βμ vs βε
  flex_vs_sim_residual_overview.{png,pdf} all experiments, βμ_sim − βμ_FLEX
  per_experiment/<EXP>.{png,pdf,csv}     plane + residual panels for one experiment

Usage:
    python run/plots.py flex-vs-sim --runs-root . --outdir plots/flex_vs_sim
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flexsim.coex.flex_theory import coex_chemical_potential, coex_sflex

FLEX_SCHEME = {"homo": 1, "positive_drive": 2, "negative_drive": 3}
SCHEME_LABEL = {"homo": "homo", "positive_drive": "positive drive", "negative_drive": "negative drive"}
SCHEME_ROWS = ["homo", "negative_drive", "positive_drive"]
DMU_COLS = [-1.0, 1.0, 2.0, 3.0, 4.0]

# Categorical slots 1-3 of the reference palette (all-pairs CVD-validated).
LY_COLOR = {16: "#2a78d6", 20: "#eb6834", 40: "#1baf7a"}
LY_MARKER = {16: "o", 20: "s", 40: "^"}
FLEX_COLOR = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def load_measured(runs_root: str) -> pd.DataFrame:
    frames = []
    pattern = os.path.join(runs_root, "COEX_RUNS_*", "ly*", "manage.csv")
    for path in sorted(glob.glob(pattern)):
        exp = re.search(r"COEX_RUNS_([^/]+)", path).group(1)
        d = pd.read_csv(path)
        # Older analyzer schema used mu_coex_SIM; newer uses mu_coex_FITTED.
        if "mu_coex_FITTED" in d:
            mu, err = d["mu_coex_FITTED"], d["mu_coex_FITTED_error"]
        else:
            mu, err = d["mu_coex_SIM"], d["mu_coex_SIM_error"]
        out = pd.DataFrame({
            "experiment": exp,
            "scheme": d["scheme"],
            "flex_scheme": d["scheme"].map(FLEX_SCHEME),
            "beta_delta_f": d["delta_f"],
            "beta_delta_mu": d["delta_mu"],
            "k": d["k"],
            "Lx": d["Lx"],
            "Ly": d["Ly"],
            "beta_epsilon": d["epsilon"],
            "beta_mu_coex_FLEX": d["mu_coex_FLEX"],
            "beta_mu_coex_SIM": pd.to_numeric(mu, errors="coerce"),
            "beta_mu_coex_SIM_err": pd.to_numeric(err, errors="coerce"),
        })
        frames.append(out.dropna(subset=["beta_mu_coex_SIM"]))
    df = pd.concat(frames, ignore_index=True)
    df["residual_SIM_minus_FLEX"] = df["beta_mu_coex_SIM"] - df["beta_mu_coex_FLEX"]
    return df.sort_values(["experiment", "Ly", "beta_epsilon"]).reset_index(drop=True)


def flex_curve(scheme: str, df_: float, dmu: float, k: float, eps: np.ndarray) -> np.ndarray:
    """FLEX μ_coex on a dense ε grid; NaN where the root solve does not converge."""
    out = np.full_like(eps, np.nan)
    for i, e in enumerate(eps):
        mu = coex_chemical_potential(e, df_, dmu, k, DRIVEN=True, scheme=FLEX_SCHEME[scheme])
        if np.isfinite(mu) and abs(coex_sflex(mu, *_k_args(scheme, e, df_, dmu, k))) < 1e-8:
            out[i] = mu
    return out


def _k_args(scheme, eps, df_, dmu, k):
    # Mirrors the effective-k choice inside coex_chemical_potential.
    if scheme == "positive_drive":
        k = min(1.0, np.exp(-df_ - dmu + 2 * eps)) * k
    elif scheme == "negative_drive":
        k = min(1.0, np.exp(-df_ - dmu - 2 * eps)) * k
    return (k, df_, eps, dmu)


def build_curves(meas: pd.DataFrame, pad: float = 0.05, step: float = 0.0025) -> pd.DataFrame:
    rows = []
    for exp, g in meas.groupby("experiment"):
        r = g.iloc[0]
        eps = np.arange(g.beta_epsilon.min() - pad, g.beta_epsilon.max() + pad + step / 2, step)
        mu = flex_curve(r.scheme, r.beta_delta_f, r.beta_delta_mu, r.k, eps)
        rows.append(pd.DataFrame({
            "experiment": exp, "scheme": r.scheme, "flex_scheme": r.flex_scheme,
            "beta_delta_f": r.beta_delta_f, "beta_delta_mu": r.beta_delta_mu, "k": r.k,
            "beta_epsilon": np.round(eps, 6), "beta_mu_coex_FLEX": mu,
        }))
    return pd.concat(rows, ignore_index=True)


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)


def _title(g) -> str:
    r = g.iloc[0]
    return f"{r.experiment}: {SCHEME_LABEL[r.scheme]}, βΔμ = {r.beta_delta_mu:+g}"


def draw_plane(ax, g, c, ms=3.5):
    ax.plot(c.beta_epsilon, c.beta_mu_coex_FLEX, color=FLEX_COLOR, lw=1.5, label="FLEX", zorder=3)
    for ly, gl in g.groupby("Ly"):
        ax.errorbar(gl.beta_epsilon, gl.beta_mu_coex_SIM, yerr=gl.beta_mu_coex_SIM_err,
                    fmt=LY_MARKER[ly], ms=ms, mfc=LY_COLOR[ly], mec="white", mew=0.5,
                    ecolor=LY_COLOR[ly], elinewidth=0.8, lw=0, label=f"sim $L_y$={ly}", zorder=4)
    _style(ax)


def draw_residual(ax, g, ms=3.5):
    ax.axhline(0, color=FLEX_COLOR, lw=1.2, label="FLEX")
    for ly, gl in g.groupby("Ly"):
        ax.errorbar(gl.beta_epsilon, gl.residual_SIM_minus_FLEX, yerr=gl.beta_mu_coex_SIM_err,
                    fmt=LY_MARKER[ly], ms=ms, mfc=LY_COLOR[ly], mec="white", mew=0.5,
                    ecolor=LY_COLOR[ly], elinewidth=0.8, lw=0, label=f"sim $L_y$={ly}", zorder=4)
    # Scale to the point values (and zero), not the error bars: a few poorly
    # converged fits carry huge errors that would flatten the real trend.
    r = g.residual_SIM_minus_FLEX
    lo, hi = min(r.min(), 0.0), max(r.max(), 0.0)
    pad = 0.12 * (hi - lo or 1e-3)
    ax.set_ylim(lo - pad, hi + pad)
    _style(ax)


def _grid_figure(meas, curves, residual: bool):
    fig, axes = plt.subplots(len(SCHEME_ROWS), len(DMU_COLS), figsize=(17, 9.5), squeeze=False)
    for i, scheme in enumerate(SCHEME_ROWS):
        for j, dmu in enumerate(DMU_COLS):
            ax = axes[i, j]
            g = meas[(meas.scheme == scheme) & np.isclose(meas.beta_delta_mu, dmu)]
            if g.empty:
                ax.axis("off")
                continue
            if residual:
                draw_residual(ax, g, ms=3)
            else:
                draw_plane(ax, g, curves[curves.experiment == g.experiment.iloc[0]], ms=3)
            ax.set_title(_title(g), fontsize=9)
            if i == len(SCHEME_ROWS) - 1 or (scheme == "homo" and j > 1):
                ax.set_xlabel(r"$\beta\varepsilon$", fontsize=9)
            if j == 0:
                ax.set_ylabel(r"$\beta\mu_{\rm sim}-\beta\mu_{\rm FLEX}$" if residual
                              else r"$\beta\mu_{\rm coex}$", fontsize=9)
    handles = [plt.Line2D([], [], color=FLEX_COLOR, lw=1.5, label="FLEX prediction")] + [
        plt.Line2D([], [], marker=LY_MARKER[ly], ls="", mfc=LY_COLOR[ly], mec="white",
                   color=LY_COLOR[ly], ms=7, label=f"Simulation, $L_y$ = {ly}") for ly in LY_COLOR]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.93), ncol=1,
               frameon=False, fontsize=10)
    fig.suptitle("Coexistence chemical potential: FLEX prediction vs simulation"
                 + (" (residual)" if residual else "")
                 + r"   [$\beta\Delta f = 0$, $k = 1$]", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _save(fig, base):
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default=".", help="dir containing COEX_RUNS_* (default: repo root)")
    p.add_argument("--outdir", default="plots/flex_vs_sim")
    args = p.parse_args()

    meas = load_measured(args.runs_root)
    curves = build_curves(meas)
    per_dir = os.path.join(args.outdir, "per_experiment")
    os.makedirs(per_dir, exist_ok=True)
    meas.to_csv(os.path.join(args.outdir, "measured_coex_points.csv"), index=False)
    curves.to_csv(os.path.join(args.outdir, "flex_prediction_curves.csv"), index=False)

    _save(_grid_figure(meas, curves, residual=False), os.path.join(args.outdir, "flex_vs_sim_overview"))
    _save(_grid_figure(meas, curves, residual=True), os.path.join(args.outdir, "flex_vs_sim_residual_overview"))

    for exp, g in meas.groupby("experiment"):
        c = curves[curves.experiment == exp]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True,
                                     gridspec_kw={"height_ratios": [2.2, 1]})
        draw_plane(a1, g, c)
        draw_residual(a2, g)
        a1.set_ylabel(r"$\beta\mu_{\rm coex}$")
        a2.set_ylabel(r"$\beta\mu_{\rm sim}-\beta\mu_{\rm FLEX}$")
        a2.set_xlabel(r"$\beta\varepsilon$")
        a1.set_title(_title(g) + r"  [$\beta\Delta f=0$, $k=1$]", fontsize=10)
        a1.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        _save(fig, os.path.join(per_dir, exp))
        g.to_csv(os.path.join(per_dir, f"{exp}_measured.csv"), index=False)
        c.to_csv(os.path.join(per_dir, f"{exp}_flex_curve.csv"), index=False)

    summary = meas.groupby(["experiment", "Ly"]).agg(
        n_points=("beta_epsilon", "size"),
        eps_min=("beta_epsilon", "min"), eps_max=("beta_epsilon", "max"),
        mean_residual=("residual_SIM_minus_FLEX", "mean"),
        max_abs_residual=("residual_SIM_minus_FLEX", lambda s: s.abs().max()),
    ).reset_index()
    summary.to_csv(os.path.join(args.outdir, "residual_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
