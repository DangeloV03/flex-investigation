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
    sweep_bc keeps the MAX BC and reports which mu achieved it."""
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

    data = bm.histogram_data(base, combo, [-2.0], str(tmp_path / "cache"))
    assert len(data) == 1
    d = data[0]
    assert d["mu"] == pytest.approx(0.0)               # picked the bimodal mu
    assert d["BC"] > bm.BC_BIMODAL_CUTOFF
    assert d["pooled"].ndim == 1
    # bimodal: column phi clusters near +1 (liquid) and -1 (gas)
    assert d["pooled"].max() > 0.5 and d["pooled"].min() < -0.5


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
