"""P(phi_col) histograms and the multi-size BC_max study."""

from __future__ import annotations

import os
import pytest

import numpy as np
import pandas as pd

from flexsim.coex.layout import combo_dir_name, mu_dir_name, param_tag
from flexsim.criticality.bc import BC_BIMODAL_CUTOFF, BONDING, EMPTY
from flexsim.criticality.figures import histogram_data, make_histogram_figure, run_fss
from bimodality_helpers import homogeneous, phase_separated, write_mu_dir


def test_histogram_data_picks_coexistence_mu(tmp_path):
    """Figure 1 data: at one epsilon with a bimodal mu and a unimodal mu,
    histogram_data must return the bimodal (max-BC, coexistence) mu's sample."""
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 60, "Ly": 10}
    base = str(tmp_path / "results")
    rng = np.random.default_rng(21)
    cp = {**combo, "epsilon": -2.0}
    cdir = os.path.join(base, combo_dir_name(cp))
    write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(-0.3)),
                  -0.3, [homogeneous(60, 10, rng) for _ in range(5)], cp, -2.0)
    write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(0.0)),
                  0.0, [phase_separated(60, 10, rng) for _ in range(5)], cp, -2.0)

    data = histogram_data(base, combo, [-2.0], str(tmp_path / "cache"),
                             mu_reduction="max")
    assert len(data) == 1
    d = data[0]
    assert d["mu"] == pytest.approx(0.0)               # picked the bimodal mu
    assert d["BC"] > BC_BIMODAL_CUTOFF
    assert d["pooled"].ndim == 1
    # bimodal: column phi clusters near +1 (liquid) and -1 (gas)
    assert d["pooled"].max() > 0.5 and d["pooled"].min() < -0.5


def test_histogram_data_balanced_prefers_even_phase_fractions(tmp_path):
    """balanced mu selection should avoid a skewed all-gas snapshot when a more
    even phase-separated mu exists with comparable BC."""
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 60, "Ly": 10}
    base = str(tmp_path / "results")
    rng = np.random.default_rng(22)
    cp = {**combo, "epsilon": -2.0}
    cdir = os.path.join(base, combo_dir_name(cp))

    skewed = np.full((60, 10), EMPTY, dtype=np.uint32)
    skewed[:4, :] = BONDING
    balanced = phase_separated(60, 10, rng)

    write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(-0.2)),
                  -0.2, [skewed for _ in range(5)], cp, -2.0)
    write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(0.0)),
                  0.0, [balanced for _ in range(5)], cp, -2.0)

    data = histogram_data(base, combo, [-2.0], str(tmp_path / "cache"),
                             mu_reduction="balanced")
    d = data[0]
    assert d["mu"] == pytest.approx(0.0)
    assert d["balance"] > 0.2


def test_make_histogram_figure_writes_png(tmp_path):
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 60, "Ly": 10}
    base = str(tmp_path / "results")
    rng = np.random.default_rng(23)
    for eps in (-2.0, -1.5):
        cp = {**combo, "epsilon": eps}
        cdir = os.path.join(base, combo_dir_name(cp))
        snaps = ([phase_separated(60, 10, rng) for _ in range(4)] if eps <= -1.7
                 else [homogeneous(60, 10, rng) for _ in range(4)])
        write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(0.0)), 0.0, snaps, cp, eps)
    out_png = str(tmp_path / "hist.png")
    make_histogram_figure(base, combo, [-2.0, -1.5], out_png,
                             cache_dir=str(tmp_path / "cache"))
    assert os.path.isfile(out_png)
    stem = os.path.splitext(out_png)[0]
    samples_csv = stem + "_samples.csv"
    hist_csv = stem + "_hist.csv"
    assert os.path.isfile(samples_csv)
    assert os.path.isfile(hist_csv)
    samples = pd.read_csv(samples_csv)
    hist = pd.read_csv(hist_csv)
    assert "phi_col" in samples.columns
    assert {"bin_center", "density", "count"}.issubset(hist.columns)
    assert set(samples["epsilon"].unique()) == {-2.0, -1.5}


def test_run_fss_multisize(tmp_path):
    """Figure 2: BC_max vs beta*eps for two sizes x two delta_mu; per-dmu plots."""
    base = str(tmp_path / "results")
    scheme, delta_f, k = "homo", 0.0, 1.0
    rng = np.random.default_rng(24)
    epsilons = [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4]
    sizes = [(80, 8), (160, 16)]
    eps_c = -1.7
    for (Lx, Ly) in sizes:
        for dmu in (0.0, 1.0):
            for eps in epsilons:
                cp = {"scheme": scheme, "delta_f": delta_f, "delta_mu": dmu,
                      "k": k, "Lx": Lx, "Ly": Ly, "epsilon": eps}
                cdir = os.path.join(base, combo_dir_name(cp))
                for mu in (0.0, 0.1):
                    snaps = ([phase_separated(Lx, Ly, rng) for _ in range(4)]
                             if eps <= eps_c else
                             [homogeneous(Lx, Ly, rng) for _ in range(4)])
                    write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(mu)),
                                  mu, snaps, cp, eps)

    out_dir = str(tmp_path / "fss_out")
    results = run_fss(base, scheme=scheme, delta_f=delta_f, k=k, sizes=sizes,
                         delta_mus=[0.0, 1.0], out_dir=out_dir, make_plots=True)
    assert len(results) == 4  # 2 sizes x 2 delta_mu, all fit
    for dmu in (0.0, 1.0):
        png = os.path.join(out_dir, f"bc_vs_beta_epsilon_dmu{param_tag(dmu)}.png")
        assert os.path.isfile(png)
    df = pd.read_csv(os.path.join(out_dir, "bc_vs_beta_epsilon.csv"))
    assert set(df["L_long"]) == {80, 160}
