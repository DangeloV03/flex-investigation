"""Locating epsilon_c: crossover bracket, threshold crossing, end-to-end find_criticality."""

from __future__ import annotations

import csv
import os
import pytest

import numpy as np

from flexsim.coex.layout import combo_dir_name, mu_dir_name
from flexsim.criticality.bc import BC_BIMODAL_CUTOFF, bc_bootstrap_error
from flexsim.criticality.bc_sweep import sweep_bc
from flexsim.criticality.epsilon_c import (
    find_criticality,
    grid_neighbor_uncertainty,
    locate_epsilon_c,
    transition_bracket,
)
from bimodality_helpers import homogeneous, phase_separated, write_mu_dir


def test_grid_neighbor_uncertainty_matches_spacing():
    x = np.round(np.arange(-1.80, -1.595, 0.005), 6)
    assert grid_neighbor_uncertainty(x, -1.76) == pytest.approx(0.005)
    # between -1.765 and -1.760: farther neighbor is 0.003 away
    assert grid_neighbor_uncertainty(x, -1.762) == pytest.approx(0.003)
    rng = np.random.default_rng(0)
    arr = np.concatenate([np.full((4, 50), -1.0), np.full((4, 50), 1.0)]) \
        + rng.normal(0, 0.05, size=(8, 50))
    err = bc_bootstrap_error(arr, n_boot=100)
    assert np.isfinite(err) and err >= 0.0
    assert np.isnan(bc_bootstrap_error(arr[:1]))  # <2 snapshots -> NaN


def test_transition_bracket_gradual_crossover():
    """Flat noisy plateau then gradual drop: half-width >> sigmoid unc."""
    x = np.linspace(-2.0, -1.4, 121)
    bc = np.where(x < -1.85, 0.92, np.where(x > -1.65, 0.55, 0.80 - 0.5 * (x + 1.75)))
    bracket = transition_bracket(x, bc)
    assert np.isfinite(bracket["transition_x_high"])
    assert np.isfinite(bracket["transition_x_low"])
    assert bracket["transition_x_low"] > bracket["transition_x_high"]
    assert bracket["transition_half_width"] > 0.03


def test_locate_epsilon_c_linear_intersect_5_9():
    """Synthetic BC = a + b x; locator must recover the 5/9 intersection."""
    # BC = -0.5 * x  with x in [-2,-1] => BC in [1.0, 0.5]; 5/9≈0.5556 => x= -2*(5/9)
    level = BC_BIMODAL_CUTOFF
    xs = [-2.0, -1.8, -1.6, -1.4, -1.2, -1.0]
    rows = []
    for x in xs:
        bc = -0.5 * x  # perfect line through origin-ish
        rows.append({
            "L_long": 60, "L_short": 10, "beta": 1.0,
            "beta_epsilon": x, "BC": bc, "BC_err": 0.01,
        })
    result = locate_epsilon_c(rows, 60, x_col="beta_epsilon")
    assert result["method"] == "linear_bc_vs_x_5_9"
    assert result["BC_target"] == pytest.approx(level)
    assert result["BC_at_criticality"] == pytest.approx(level)
    # polyfit of BC = -0.5 x => x_c = -2*(5/9)
    assert result["criticality_estimate"] == pytest.approx(-2.0 * level, abs=1e-6)
    assert np.isfinite(result["fit_uncertainty"])
    assert result["fit_uncertainty"] == pytest.approx(0.0, abs=1e-9)
    assert "recommended_uncertainty" in result


def test_locate_epsilon_c_includes_transition_fields(tmp_path):
    base = str(tmp_path / "results")
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 60, "Ly": 10}
    rng = np.random.default_rng(4)
    eps_true_c = -1.7
    epsilons = [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4]
    for eps in epsilons:
        cp = {**combo, "epsilon": eps}
        combo_dir = os.path.join(base, combo_dir_name(cp))
        for mu in (-0.3, 0.0, 0.3):
            if eps <= eps_true_c:
                snaps = [phase_separated(60, 10, rng) for _ in range(5)]
            else:
                snaps = [homogeneous(60, 10, rng) for _ in range(5)]
            write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(mu)),
                          mu, snaps, cp, epsilon=eps)

    rows = sweep_bc(base, combo, str(tmp_path / "bc.csv"), str(tmp_path / "cache"))
    result = locate_epsilon_c(rows, 60, x_col="beta_epsilon")
    assert result["method"] == "linear_bc_vs_x_5_9"
    assert result["BC_target"] == pytest.approx(BC_BIMODAL_CUTOFF)
    assert "recommended_uncertainty" in result
    assert np.isfinite(result["recommended_uncertainty"])
    assert np.isfinite(result["fit_uncertainty"])
    assert result["BC_at_criticality"] == pytest.approx(BC_BIMODAL_CUTOFF)
    # linear intersection need not land on a grid point
    assert np.isfinite(result["criticality_estimate"])
    xs = sorted(float(r["beta_epsilon"]) for r in rows)
    assert min(xs) - 0.5 <= result["criticality_estimate"] <= max(xs) + 0.5


def test_find_criticality_end_to_end(tmp_path):
    """Synthetic sweep: below eps_c -> phase separated (bimodal), above -> homo."""
    base = str(tmp_path / "results")
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 60, "Ly": 10}
    rng = np.random.default_rng(4)
    eps_true_c = -1.7
    epsilons = [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4]
    for eps in epsilons:
        cp = {**combo, "epsilon": eps}
        combo_dir = os.path.join(base, combo_dir_name(cp))
        # a small mu sweep; mu_coex ~ 0, phase behaviour keyed to epsilon
        for mu in (-0.3, 0.0, 0.3):
            if eps <= eps_true_c:
                snaps = [phase_separated(60, 10, rng) for _ in range(5)]
            else:
                snaps = [homogeneous(60, 10, rng) for _ in range(5)]
            write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(mu)),
                          mu, snaps, cp, epsilon=eps)

    out_dir = str(tmp_path / "criticality_out")
    result = find_criticality(base, combo, out_dir=out_dir, make_plots=False)

    # BC csv written, one row per epsilon, and BC falls as epsilon rises.
    bc_csv = os.path.join(out_dir, "bc_vs_epsilon.csv")
    assert os.path.isfile(bc_csv)
    rows = list(csv.DictReader(open(bc_csv)))
    assert len(rows) == len(epsilons)
    bc_by_eps = {float(r["epsilon"]): float(r["BC"]) for r in rows}
    assert bc_by_eps[-2.0] > bc_by_eps[-1.4]              # bimodal -> unimodal
    assert bc_by_eps[-2.0] > BC_BIMODAL_CUTOFF          # clearly bimodal below
    assert bc_by_eps[-1.4] < BC_BIMODAL_CUTOFF          # clearly unimodal above

    # epsilon_c lands within the sweep, near the true crossover.
    assert epsilons[0] <= result["epsilon_c_estimate"] <= epsilons[-1]
    assert os.path.isfile(os.path.join(out_dir, "epsilon_c.csv"))
