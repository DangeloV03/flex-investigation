"""Tests for criticality/plot_eq_L_scaling.py (multi-L eq mentor plots)."""

from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd

import plot_eq_L_scaling as pls


def _write_manage(path: str, *, ly: int, eps_mu: list[tuple[float, float]]) -> None:
    fields = [
        "epsilon", "delta_f", "delta_mu", "k", "scheme", "Lx", "Ly",
        "mu_coex_FLEX", "isSubmitted", "isRan", "isAnalyzed",
        "mu_coex_FITTED", "mu_coex_FITTED_error",
        "RequestForAdditionalData", "combo_path",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for eps, mu in eps_mu:
            w.writerow({
                "epsilon": eps,
                "delta_f": -20.0,
                "delta_mu": 0.0,
                "k": 0.0,
                "scheme": "homo",
                "Lx": 10 * ly,
                "Ly": ly,
                "mu_coex_FLEX": mu,
                "isSubmitted": "1",
                "isRan": "1",
                "isAnalyzed": "1",
                "mu_coex_FITTED": mu,
                "mu_coex_FITTED_error": 0.001,
                "RequestForAdditionalData": "0",
                "combo_path": f"coex/coex_eq/ly{ly}/results",
            })


def _write_crit(path: str, *, ly: int, eps_c: float) -> None:
    fields = [
        "scheme", "delta_f", "delta_mu", "k", "L_short", "L_long", "x_axis",
        "criticality_estimate", "epsilon_c_estimate", "beta", "method",
        "fit_uncertainty", "recommended_uncertainty",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "scheme": "homo",
            "delta_f": -20.0,
            "delta_mu": 0.0,
            "k": 0.0,
            "L_short": ly,
            "L_long": 10 * ly,
            "x_axis": "beta_epsilon",
            "criticality_estimate": eps_c,
            "epsilon_c_estimate": eps_c,
            "beta": 1.0,
            "method": "sigmoid",
            "fit_uncertainty": 0.01,
            "recommended_uncertainty": 0.02,
        })


def test_collect_and_plot_eq_scaling(tmp_path):
    coex = tmp_path / "coex_eq"
    crit = tmp_path / "criticality"
    out = tmp_path / "out"
    # Synthetic: eps_c and mu_coex(eps_c) both increase with L (mentor: positive mu slope)
    specs = {
        16: (-1.72, {(-1.80, -3.55), (-1.72, -3.40), (-1.65, -3.25)}),
        20: (-1.70, {(-1.80, -3.52), (-1.70, -3.35), (-1.65, -3.22)}),
        40: (-1.68, {(-1.80, -3.48), (-1.68, -3.28), (-1.65, -3.18)}),
    }
    for ly, (eps_c, pts) in specs.items():
        mdir = coex / f"ly{ly}"
        cdir = crit / f"ly{ly}"
        mdir.mkdir(parents=True)
        cdir.mkdir(parents=True)
        _write_manage(str(mdir / "manage.csv"), ly=ly, eps_mu=sorted(pts))
        _write_crit(str(cdir / "criticality.csv"), ly=ly, eps_c=eps_c)

    df = pls.collect_scaling_table(
        [16, 20, 40], coex_root=str(coex), crit_root=str(crit), crit_prefix="ly",
    )
    assert list(df["L_short"]) == [16, 20, 40]
    assert np.all(np.diff(df["beta_mu_coex_at_eps_c"]) > 0)
    assert np.all(np.diff(df["beta_epsilon_c"]) > 0)

    fit = pls.fit_fss(df)
    assert np.isfinite(fit["beta_eps_c_infty"])
    # with increasing eps_c (less negative) as L grows, 1/L intercept should be finite
    assert fit["beta_eps_c_infty"] > df["beta_epsilon_c"].max() - 0.5

    png_mu = str(out / "beta_mu_coex_vs_L.png")
    png_eps = str(out / "beta_eps_c_vs_L.png")
    out.mkdir()
    pls.plot_vs_L(
        df, y_col="beta_mu_coex_at_eps_c", yerr_col="mu_coex_uncertainty",
        ylabel="y", title="t", out_png=png_mu,
    )
    pls.plot_vs_L(
        df, y_col="beta_epsilon_c", yerr_col="epsilon_c_uncertainty",
        ylabel="y", title="t", out_png=png_eps,
    )
    pls.plot_fss_invL(df, fit, str(out / "beta_eps_c_vs_invL.png"))
    assert os.path.isfile(png_mu)
    assert os.path.isfile(png_eps)
    assert os.path.isfile(str(out / "beta_eps_c_vs_invL.png"))

    loo = pls.leave_one_out_fss(df)
    assert list(loo["leave_out_L"]) == [16, 20, 40]
    drop16 = df[df["L_short"] != 16].reset_index(drop=True)
    held = df[df["L_short"] == 16].reset_index(drop=True)
    fit16 = pls.fit_fss(drop16)
    assert fit16["L_fit"] == "20,40"
    pls.plot_fss_invL(
        drop16, fit16, str(out / "beta_eps_c_vs_invL_loo_drop16.png"),
        df_held_out=held,
        title="LOO drop 16",
    )
    assert os.path.isfile(str(out / "beta_eps_c_vs_invL_loo_drop16.png"))
    # LOO drop-16 intercept matches the 20+40-only row
    row = loo.loc[loo["leave_out_L"] == 16].iloc[0]
    assert abs(row["beta_eps_c_infty"] - fit16["beta_eps_c_infty"]) < 1e-12
