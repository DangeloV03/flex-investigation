"""
tests/test_bimodality.py

Tests for the bimodality-based epsilon_c pipeline (criticality/bimodality.py).

Validates Sarle's BC against synthetic cases with known limits (spec deliverable
#3) before trusting it on real data, plus the column reduction, mu selection,
pooling, and the end-to-end sweep on a tiny synthetic coex run.

Run with:
    cd /path/to/flex-investigation
    python -m pytest tests/test_bimodality.py -v
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd
import pytest

import bimodality as bm
from combo_paths import mu_dir_name

BONDING, INERT, EMPTY = bm.BONDING, bm.INERT, bm.EMPTY


# --------------------------------------------------------------------------
# Step 3 - Sarle's BC known limits
# --------------------------------------------------------------------------

def test_bc_single_gaussian_near_one_third():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=200_000)
    bc = bm.sarle_bc(x)["BC"]
    assert bc == pytest.approx(1.0 / 3.0, abs=0.02)


def test_bc_two_separated_gaussians_near_one():
    rng = np.random.default_rng(1)
    half = 100_000
    # Two tight, symmetric, well-separated peaks -> BC -> 1.
    x = np.concatenate([
        rng.normal(-1.0, 0.02, size=half),
        rng.normal(+1.0, 0.02, size=half),
    ])
    bc = bm.sarle_bc(x)["BC"]
    assert bc > 0.9


def test_bc_affine_invariant():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(-1, 0.1, 50_000), rng.normal(1, 0.1, 50_000)])
    base = bm.sarle_bc(x)["BC"]
    scaled = bm.sarle_bc(3.7 * x - 42.0)["BC"]  # phi_col vs rho_bonding = affine map
    assert scaled == pytest.approx(base, rel=1e-9)


def test_bc_requires_min_samples():
    with pytest.raises(ValueError):
        bm.sarle_bc(np.array([1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------
# Step 1 - column order parameter
# --------------------------------------------------------------------------

def test_column_op_phi_convention():
    # (Lx, Ly) = (4, 10). Column 0 all BONDING -> +1; column 1 all EMPTY -> -1;
    # column 2 all INERT -> -1; column 3 half/half BONDING/EMPTY -> 0.
    Lx, Ly = 4, 10
    snap = np.empty((Lx, Ly), dtype=np.uint32)
    snap[0, :] = BONDING
    snap[1, :] = EMPTY
    snap[2, :] = INERT
    snap[3, : Ly // 2] = BONDING
    snap[3, Ly // 2 :] = EMPTY
    phi = bm.column_op(snap)
    assert phi.shape == (Lx,)
    np.testing.assert_allclose(phi, [1.0, -1.0, -1.0, 0.0])


def test_untag_roundtrip():
    from combo_paths import param_tag
    for v in (-2.0, -1.45, 0.5, 0.0, 2.5):
        assert bm.untag(param_tag(v)) == pytest.approx(v)


# --------------------------------------------------------------------------
# helpers to fabricate a tiny synthetic coex run on disk
# --------------------------------------------------------------------------

def _write_mu_dir(mu_dir, mu, snapshots, combo_params, epsilon, beta=1.0):
    os.makedirs(mu_dir, exist_ok=True)
    for i, snap in enumerate(snapshots):
        np.save(os.path.join(mu_dir, f"final_lattice_{i}.npy"), snap)
    with open(os.path.join(mu_dir, "output.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "epsilon", "delta_f", "delta_mu", "k",
                    "scheme", "Lx", "Ly", "mu", "beta", "rho_active"])
        for i in range(len(snapshots)):
            w.writerow([i, epsilon, combo_params["delta_f"], combo_params["delta_mu"],
                        combo_params["k"], combo_params["scheme"],
                        combo_params["Lx"], combo_params["Ly"], mu, beta, 0.5])


def _phase_separated(Lx, Ly, rng):
    """Slab: left half BONDING, right half EMPTY, with a jittered interface."""
    snap = np.full((Lx, Ly), EMPTY, dtype=np.uint32)
    edge = Lx // 2 + rng.integers(-1, 2)
    snap[:edge, :] = BONDING
    return snap


def _homogeneous(Lx, Ly, rng):
    """Single phase near half density, no persistent interface (unimodal columns)."""
    return rng.integers(0, 2, size=(Lx, Ly)).astype(np.uint32) * BONDING


def test_nearest_mu_dir_and_extract(tmp_path):
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 40, "Ly": 8}
    combo_dir = str(tmp_path / "combo")
    rng = np.random.default_rng(3)
    for mu in (-0.5, 0.0, 0.5):
        snaps = [_phase_separated(40, 8, rng) for _ in range(4)]
        _write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(mu)),
                      mu, snaps, combo, epsilon=-2.0)
    mu_dir, mu = bm.nearest_mu_dir(combo_dir, mu_coex=0.1)
    assert mu == pytest.approx(0.0)  # 0.0 is nearest signed mu to 0.1
    arr, meta = bm.extract_column_op(mu_dir)
    assert arr.shape == (4, 40)
    assert meta["Ly"] == 8


def test_pool_column_op_shapes():
    arr = np.arange(4 * 40, dtype=float).reshape(4, 40)
    pooled, n_pooled, L_long = bm.pool_column_op(arr)
    assert pooled.shape == (160,)
    assert n_pooled == 160
    assert L_long == 40


def test_sweep_bc_max_over_mu(tmp_path):
    """At one epsilon with two mu dirs (one phase-separated, one homogeneous),
    sweep_bc with mu_reduction=max keeps the MAX BC and reports which mu achieved it."""
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 60, "Ly": 10}
    cp = {**combo, "epsilon": -2.0}
    base = str(tmp_path / "results")
    combo_dir = os.path.join(base, bm.combo_dir_name(cp))
    rng = np.random.default_rng(7)
    # mu=-0.4: homogeneous (unimodal); mu=0.0: phase separated (bimodal, higher BC)
    _write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(-0.4)),
                  -0.4, [_homogeneous(60, 10, rng) for _ in range(5)], cp, -2.0, beta=2.0)
    _write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(0.0)),
                  0.0, [_phase_separated(60, 10, rng) for _ in range(5)], cp, -2.0, beta=2.0)

    out_csv = str(tmp_path / "bc.csv")
    cache = str(tmp_path / "cache")
    rows = bm.sweep_bc(base, combo, out_csv, cache, mu_reduction="max")
    assert len(rows) == 1
    r = rows[0]
    assert r["n_mu_scanned"] == 2
    assert r["mu_at_max"] == pytest.approx(0.0)         # bimodal mu wins
    assert r["BC"] > bm.BC_BIMODAL_CUTOFF
    assert r["beta"] == pytest.approx(2.0)
    assert r["beta_epsilon"] == pytest.approx(2.0 * -2.0)  # beta*epsilon


def test_sweep_bc_zero_mean_picks_near_zero_magnetization(tmp_path):
    """zero_mean selects the mu with ⟨φ⟩ closest to 0 even if another mu has higher BC."""
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 0.0, "k": 0.0,
             "Lx": 60, "Ly": 10}
    cp = {**combo, "epsilon": -1.76}
    base = str(tmp_path / "results")
    combo_dir = os.path.join(base, bm.combo_dir_name(cp))
    rng = np.random.default_rng(11)
    # skewed mostly-gas: high |mean|, can still have large BC
    gas_heavy = _phase_separated(60, 10, rng)
    gas_heavy[:, :50] = bm.EMPTY
    # balanced coexistence: ⟨φ⟩ closer to 0
    balanced = _phase_separated(60, 10, rng)
    _write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(-0.5)),
                  -0.5, [gas_heavy for _ in range(5)], cp, -1.76)
    _write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(0.0)),
                  0.0, [balanced for _ in range(5)], cp, -1.76)

    rows = bm.sweep_bc(base, combo, str(tmp_path / "bc.csv"), str(tmp_path / "cache"),
                       mu_reduction="zero_mean")
    assert len(rows) == 1
    assert rows[0]["mu_at_max"] == pytest.approx(0.0)
    assert abs(rows[0]["mean"]) < 0.35


def test_grid_neighbor_uncertainty_matches_spacing():
    x = np.round(np.arange(-1.80, -1.595, 0.005), 6)
    assert bm.grid_neighbor_uncertainty(x, -1.76) == pytest.approx(0.005)
    # between -1.765 and -1.760: farther neighbor is 0.003 away
    assert bm.grid_neighbor_uncertainty(x, -1.762) == pytest.approx(0.003)
    rng = np.random.default_rng(0)
    arr = np.concatenate([np.full((4, 50), -1.0), np.full((4, 50), 1.0)]) \
        + rng.normal(0, 0.05, size=(8, 50))
    err = bm.bc_bootstrap_error(arr, n_boot=100)
    assert np.isfinite(err) and err >= 0.0
    assert np.isnan(bm.bc_bootstrap_error(arr[:1]))  # <2 snapshots -> NaN


def test_sweep_bc_writes_error_bar(tmp_path):
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 60, "Ly": 10}
    cp = {**combo, "epsilon": -2.0}
    base = str(tmp_path / "results")
    combo_dir = os.path.join(base, bm.combo_dir_name(cp))
    rng = np.random.default_rng(9)
    _write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(0.0)),
                  0.0, [_phase_separated(60, 10, rng) for _ in range(8)], cp, -2.0)
    rows = bm.sweep_bc(base, combo, str(tmp_path / "bc.csv"), str(tmp_path / "cache"))
    assert "BC_err" in rows[0]
    assert np.isfinite(rows[0]["BC_err"]) and rows[0]["BC_err"] >= 0.0


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
            combo_dir = os.path.join(base, bm.combo_dir_name(cp))
            for mu in (-0.1, 0.0, 0.1):
                if eps <= eps_c:
                    snaps = [_phase_separated(Lx, Ly, rng) for _ in range(5)]
                else:
                    snaps = [_homogeneous(Lx, Ly, rng) for _ in range(5)]
                _write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(mu)),
                              mu, snaps, cp, eps)

    assert bm.discover_delta_mus(base, scheme, delta_f, Lx, Ly) == delta_mus

    out_dir = str(tmp_path / "pd_out")
    results = bm.phase_diagram(base, scheme=scheme, delta_f=delta_f, k=k,
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
    from plot_bimodality import plot_bc_family

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


def test_backfill_bc_err_from_cache(tmp_path):
    """Rows without BC_err get bootstrap errors from column_op cache."""
    rng = np.random.default_rng(7)
    arr = np.stack([_phase_separated(60, 10, rng) for _ in range(6)], axis=0)
    cache_dir = tmp_path / "cache" / "column_op"
    cache_dir.mkdir(parents=True)
    key = "60x10_homo_deltaF0p0_dmu1p0_epsilonm2p0__mu1234567"
    np.save(cache_dir / f"{key}.npy", arr)
    bc_csv = tmp_path / "bc.csv"
    pd.DataFrame([{
        "scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
        "epsilon": -2.0, "beta": 1.0, "beta_epsilon": -2.0,
        "L_short": 10, "L_long": 60, "BC": 0.95, "mu_at_max": 1.234567,
    }]).to_csv(bc_csv, index=False)
    filled, already, total = bm.backfill_bc_err(
        str(bc_csv), cache_dir=str(cache_dir),
    )
    assert total == 1
    assert filled == 1
    df = pd.read_csv(bc_csv)
    assert np.isfinite(df["BC_err"].iloc[0])
    assert df["BC_err"].iloc[0] > 0


def test_plot_bc_scheme_comparison(tmp_path):
    """Side-by-side scheme comparison figure includes BC_err vertical bars."""
    from plot_bimodality import plot_bc_scheme_comparison

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


def test_histogram_data_picks_coexistence_mu(tmp_path):
    """Figure 1 data: at one epsilon with a bimodal mu and a unimodal mu,
    histogram_data must return the bimodal (max-BC, coexistence) mu's sample."""
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 60, "Ly": 10}
    base = str(tmp_path / "results")
    rng = np.random.default_rng(21)
    cp = {**combo, "epsilon": -2.0}
    cdir = os.path.join(base, bm.combo_dir_name(cp))
    _write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(-0.3)),
                  -0.3, [_homogeneous(60, 10, rng) for _ in range(5)], cp, -2.0)
    _write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(0.0)),
                  0.0, [_phase_separated(60, 10, rng) for _ in range(5)], cp, -2.0)

    data = bm.histogram_data(base, combo, [-2.0], str(tmp_path / "cache"),
                             mu_reduction="max")
    assert len(data) == 1
    d = data[0]
    assert d["mu"] == pytest.approx(0.0)               # picked the bimodal mu
    assert d["BC"] > bm.BC_BIMODAL_CUTOFF
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
    cdir = os.path.join(base, bm.combo_dir_name(cp))

    skewed = np.full((60, 10), bm.EMPTY, dtype=np.uint32)
    skewed[:4, :] = bm.BONDING
    balanced = _phase_separated(60, 10, rng)

    _write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(-0.2)),
                  -0.2, [skewed for _ in range(5)], cp, -2.0)
    _write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(0.0)),
                  0.0, [balanced for _ in range(5)], cp, -2.0)

    data = bm.histogram_data(base, combo, [-2.0], str(tmp_path / "cache"),
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
        cdir = os.path.join(base, bm.combo_dir_name(cp))
        snaps = ([_phase_separated(60, 10, rng) for _ in range(4)] if eps <= -1.7
                 else [_homogeneous(60, 10, rng) for _ in range(4)])
        _write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(0.0)), 0.0, snaps, cp, eps)
    out_png = str(tmp_path / "hist.png")
    bm.make_histogram_figure(base, combo, [-2.0, -1.5], out_png,
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
                cdir = os.path.join(base, bm.combo_dir_name(cp))
                for mu in (0.0, 0.1):
                    snaps = ([_phase_separated(Lx, Ly, rng) for _ in range(4)]
                             if eps <= eps_c else
                             [_homogeneous(Lx, Ly, rng) for _ in range(4)])
                    _write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(mu)),
                                  mu, snaps, cp, eps)

    out_dir = str(tmp_path / "fss_out")
    results = bm.run_fss(base, scheme=scheme, delta_f=delta_f, k=k, sizes=sizes,
                         delta_mus=[0.0, 1.0], out_dir=out_dir, make_plots=True)
    assert len(results) == 4  # 2 sizes x 2 delta_mu, all fit
    for dmu in (0.0, 1.0):
        png = os.path.join(out_dir, f"bc_vs_beta_epsilon_dmu{bm.param_tag(dmu)}.png")
        assert os.path.isfile(png)
    df = pd.read_csv(os.path.join(out_dir, "bc_vs_beta_epsilon.csv"))
    assert set(df["L_long"]) == {80, 160}


def test_nested_scheme3_layout_discovery(tmp_path):
    """scheme3 layout: base/dmu<X>/results/<combo>. results_roots + discovery
    must find every delta_mu across the per-dmu results/ subfolders."""
    base = str(tmp_path / "scheme3")
    scheme, delta_f, k, Lx, Ly = "negative_drive", 0.0, 1.0, 60, 10
    rng = np.random.default_rng(31)
    epsilons = [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4]
    dmu_folders = {"dmu1p0": 1.0, "dmu2p0": 2.0}
    for folder, dmu in dmu_folders.items():
        eps_c = -1.9 + 0.1 * dmu
        for eps in epsilons:
            cp = {"scheme": scheme, "delta_f": delta_f, "delta_mu": dmu,
                  "k": k, "Lx": Lx, "Ly": Ly, "epsilon": eps}
            cdir = os.path.join(base, folder, "results", bm.combo_dir_name(cp))
            for mu in (-4.5, -4.4):
                snaps = ([_phase_separated(Lx, Ly, rng) for _ in range(4)]
                         if eps <= eps_c else
                         [_homogeneous(Lx, Ly, rng) for _ in range(4)])
                _write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(mu)),
                              mu, snaps, cp, eps)

    # discovery walks the nested results/ roots
    assert bm.discover_delta_mus(base, scheme, delta_f, Lx, Ly) == [1.0, 2.0]

    out_dir = str(tmp_path / "out")
    results = bm.phase_diagram(base, scheme=scheme, delta_f=delta_f, k=k,
                               Lx=Lx, Ly=Ly, out_dir=out_dir, make_plots=False)
    assert {r["delta_mu"] for r in results} == {1.0, 2.0}


def test_inspect_coverage_reports_step(tmp_path):
    """inspect_coverage reports the epsilon grid step and flags coarse grids."""
    base = str(tmp_path / "results")
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 60, "Ly": 10}
    rng = np.random.default_rng(41)
    # a coarse 0.1 grid
    for eps in (-2.0, -1.9, -1.8, -1.7):
        cp = {**combo, "epsilon": round(eps, 4)}
        cdir = os.path.join(base, bm.combo_dir_name(cp))
        _write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(0.0)),
                      0.0, [_phase_separated(60, 10, rng)], cp, round(eps, 4))

    rep = bm.inspect_coverage(base, scheme="homo", delta_f=0.0, k=1.0,
                              Lx=60, Ly=10, ref_step=0.005)
    assert len(rep) == 1
    r = rep[0]
    assert r["n_eps"] == 4
    assert r["step_min"] == pytest.approx(0.1, abs=1e-6)
    assert r["as_fine_as_ref"] is False        # 0.1 is coarser than 0.005


def test_transition_bracket_gradual_crossover():
    """Flat noisy plateau then gradual drop: half-width >> sigmoid unc."""
    x = np.linspace(-2.0, -1.4, 121)
    bc = np.where(x < -1.85, 0.92, np.where(x > -1.65, 0.55, 0.80 - 0.5 * (x + 1.75)))
    bracket = bm.transition_bracket(x, bc)
    assert np.isfinite(bracket["transition_x_high"])
    assert np.isfinite(bracket["transition_x_low"])
    assert bracket["transition_x_low"] > bracket["transition_x_high"]
    assert bracket["transition_half_width"] > 0.03


def test_locate_epsilon_c_includes_transition_fields(tmp_path):
    base = str(tmp_path / "results")
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 60, "Ly": 10}
    rng = np.random.default_rng(4)
    eps_true_c = -1.7
    epsilons = [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4]
    for eps in epsilons:
        cp = {**combo, "epsilon": eps}
        combo_dir = os.path.join(base, bm.combo_dir_name(cp))
        for mu in (-0.3, 0.0, 0.3):
            if eps <= eps_true_c:
                snaps = [_phase_separated(60, 10, rng) for _ in range(5)]
            else:
                snaps = [_homogeneous(60, 10, rng) for _ in range(5)]
            _write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(mu)),
                          mu, snaps, cp, epsilon=eps)

    rows = bm.sweep_bc(base, combo, str(tmp_path / "bc.csv"), str(tmp_path / "cache"))
    result = bm.locate_epsilon_c(rows, 60, x_col="beta_epsilon")
    assert result["method"].startswith("closest_bc_")
    assert result["BC_target"] == pytest.approx(bm.BC_CRIT_TARGET)
    assert "recommended_uncertainty" in result
    assert np.isfinite(result["recommended_uncertainty"])
    assert result["recommended_uncertainty"] == pytest.approx(0.1)
    assert abs(result["BC_at_criticality"] - bm.BC_CRIT_TARGET) <= abs(
        float(min(rows, key=lambda r: abs(float(r["BC"]) - bm.BC_CRIT_TARGET))["BC"])
        - bm.BC_CRIT_TARGET
    ) + 1e-12
    # discrete pick: criticality is one of the measured beta*epsilon grid points
    xs = {float(r["beta_epsilon"]) for r in rows}
    assert result["criticality_estimate"] in xs

    # alternate target 5/9 should be selectable and can differ
    alt = bm.locate_epsilon_c(
        rows, 60, x_col="beta_epsilon", bc_target=bm.BC_BIMODAL_CUTOFF,
    )
    assert alt["BC_target"] == pytest.approx(bm.BC_BIMODAL_CUTOFF)
    assert alt["method"] == f"closest_bc_{bm.BC_BIMODAL_CUTOFF:.4f}"
    assert alt["criticality_estimate"] in xs


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
        combo_dir = os.path.join(base, bm.combo_dir_name(cp))
        # a small mu sweep; mu_coex ~ 0, phase behaviour keyed to epsilon
        for mu in (-0.3, 0.0, 0.3):
            if eps <= eps_true_c:
                snaps = [_phase_separated(60, 10, rng) for _ in range(5)]
            else:
                snaps = [_homogeneous(60, 10, rng) for _ in range(5)]
            _write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(mu)),
                          mu, snaps, cp, epsilon=eps)

    out_dir = str(tmp_path / "criticality_out")
    result = bm.find_criticality(base, combo, out_dir=out_dir, make_plots=False)

    # BC csv written, one row per epsilon, and BC falls as epsilon rises.
    bc_csv = os.path.join(out_dir, "bc_vs_epsilon.csv")
    assert os.path.isfile(bc_csv)
    rows = list(csv.DictReader(open(bc_csv)))
    assert len(rows) == len(epsilons)
    bc_by_eps = {float(r["epsilon"]): float(r["BC"]) for r in rows}
    assert bc_by_eps[-2.0] > bc_by_eps[-1.4]              # bimodal -> unimodal
    assert bc_by_eps[-2.0] > bm.BC_BIMODAL_CUTOFF          # clearly bimodal below
    assert bc_by_eps[-1.4] < bm.BC_BIMODAL_CUTOFF          # clearly unimodal above

    # epsilon_c lands within the sweep, near the true crossover.
    assert epsilons[0] <= result["epsilon_c_estimate"] <= epsilons[-1]
    assert os.path.isfile(os.path.join(out_dir, "epsilon_c.csv"))
