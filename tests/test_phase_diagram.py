"""Per-delta_mu BC_max families, scheme comparison plot, epsilon-grid coverage."""

from __future__ import annotations

import os
import pytest

import numpy as np
import pandas as pd

from flexsim.coex.layout import combo_dir_name, mu_dir_name
from flexsim.criticality.phase_diagram import discover_delta_mus, inspect_coverage, phase_diagram
from bimodality_helpers import homogeneous, phase_separated, write_mu_dir


def test_phase_diagram_family_over_dmu(tmp_path):
    """A BC_max-vs-(beta*epsilon) curve per delta_mu, with each curve's crossover
    shifted by delta_mu (the thermal-phase-diagram family plot)."""
    base = str(tmp_path / "results")
    scheme, delta_f, k, Lx, Ly = "homo", 0.0, 1.0, 60, 10
    rng = np.random.default_rng(11)
    epsilons = [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4]
    delta_mus = [0.0, 1.0, 2.0]
    for dmu in delta_mus:
        eps_c = -1.9 + 0.15 * dmu  # crossover shifts with delta_mu
        for eps in epsilons:
            cp = {"scheme": scheme, "delta_f": delta_f, "delta_mu": dmu,
                  "k": k, "Lx": Lx, "Ly": Ly, "epsilon": eps}
            combo_dir = os.path.join(base, combo_dir_name(cp))
            for mu in (-0.1, 0.0, 0.1):
                if eps <= eps_c:
                    snaps = [phase_separated(Lx, Ly, rng) for _ in range(5)]
                else:
                    snaps = [homogeneous(Lx, Ly, rng) for _ in range(5)]
                write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(mu)),
                              mu, snaps, cp, eps)

    assert discover_delta_mus(base, scheme, delta_f, Lx, Ly) == delta_mus

    out_dir = str(tmp_path / "pd_out")
    results = phase_diagram(base, scheme=scheme, delta_f=delta_f, k=k,
                               Lx=Lx, Ly=Ly, out_dir=out_dir, make_plots=True)
    assert len(results) == len(delta_mus)
    # criticality (beta*eps)_c should increase with delta_mu (curves shift right)
    by_dmu = {r["delta_mu"]: r["criticality_estimate"] for r in results}
    assert by_dmu[0.0] < by_dmu[1.0] < by_dmu[2.0]
    # family CSV has all delta_mu; plot written
    df = pd.read_csv(os.path.join(out_dir, "bc_vs_beta_epsilon.csv"))
    assert set(df["delta_mu"].round(6)) == set(delta_mus)
    assert os.path.isfile(os.path.join(out_dir, "bc_max_phase_diagram_60x10.png"))
    assert os.path.isfile(os.path.join(out_dir, "criticality.csv"))


def test_plot_bc_family_no_transition_bracket(tmp_path):
    """Family plot with show_transition_bracket=False skips orange shading."""
    from flexsim.criticality.plot_bimodality import plot_bc_family

    rows = []
    for dmu in [0.0, 1.0]:
        for i, eps in enumerate([-2.0, -1.8, -1.6]):
            rows.append({
                "delta_mu": dmu, "L_long": 160, "L_short": 16,
                "beta_epsilon": eps, "BC": 0.9 - 0.1 * i,
                "BC_err": 0.02,
            })
    bc_csv = str(tmp_path / "bc.csv")
    pd.DataFrame(rows).to_csv(bc_csv, index=False)
    crit_csv = str(tmp_path / "crit.csv")
    pd.DataFrame([{
        "delta_mu": 0.0, "L_long": 160, "criticality_estimate": -1.7,
        "transition_x_high": -1.85, "transition_x_low": -1.55,
        "transition_bc_high": 0.85, "transition_bc_low": 0.65,
        "transition_half_width": 0.15,
    }]).to_csv(crit_csv, index=False)
    out_png = str(tmp_path / "family.png")
    plot_bc_family(bc_csv, out_png, crit_csv=crit_csv, show_transition_bracket=False)
    assert os.path.isfile(out_png)


def test_plot_bc_scheme_comparison(tmp_path):
    """Side-by-side scheme comparison figure includes BC_err vertical bars."""
    from flexsim.criticality.plot_bimodality import plot_bc_scheme_comparison

    rng = np.random.default_rng(99)
    rows = []
    for dmu in [0.0, 1.0]:
        for i, eps in enumerate([-2.0, -1.8, -1.6]):
            rows.append({
                "delta_mu": dmu, "L_long": 60, "L_short": 10,
                "beta_epsilon": eps, "BC": 0.9 - 0.1 * i - 0.05 * dmu,
                "BC_err": 0.01 + 0.005 * rng.random(),
            })
    csv = str(tmp_path / "bc.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    out_png = str(tmp_path / "compare.png")
    plot_bc_scheme_comparison(
        [(csv, "Scheme 1: Homogenous"), (csv, "Scheme 3: Negative")],
        out_png,
    )
    assert os.path.isfile(out_png)


def test_inspect_coverage_reports_step(tmp_path):
    """inspect_coverage reports the epsilon grid step and flags coarse grids."""
    base = str(tmp_path / "results")
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 60, "Ly": 10}
    rng = np.random.default_rng(41)
    # a coarse 0.1 grid
    for eps in (-2.0, -1.9, -1.8, -1.7):
        cp = {**combo, "epsilon": round(eps, 4)}
        cdir = os.path.join(base, combo_dir_name(cp))
        write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(0.0)),
                      0.0, [phase_separated(60, 10, rng)], cp, round(eps, 4))

    rep = inspect_coverage(base, scheme="homo", delta_f=0.0, k=1.0,
                              Lx=60, Ly=10, ref_step=0.005)
    assert len(rep) == 1
    r = rep[0]
    assert r["n_eps"] == 4
    assert r["step_min"] == pytest.approx(0.1, abs=1e-6)
    assert r["as_fine_as_ref"] is False        # 0.1 is coarser than 0.005
