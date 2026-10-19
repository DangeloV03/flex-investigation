"""BC across an epsilon sweep: mu selection, error bars, cache backfill, nested layouts."""

from __future__ import annotations

import os
import pytest

import numpy as np
import pandas as pd

from flexsim.coex.layout import combo_dir_name, mu_dir_name
from flexsim.criticality.bc import BC_BIMODAL_CUTOFF, EMPTY, extract_column_op
from flexsim.criticality.bc_sweep import backfill_bc_err, sweep_bc
from flexsim.criticality.mu_selection import nearest_mu_dir
from flexsim.criticality.phase_diagram import discover_delta_mus, phase_diagram
from bimodality_helpers import homogeneous, phase_separated, write_mu_dir


def test_nearest_mu_dir_and_extract(tmp_path):
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 40, "Ly": 8}
    combo_dir = str(tmp_path / "combo")
    rng = np.random.default_rng(3)
    for mu in (-0.5, 0.0, 0.5):
        snaps = [phase_separated(40, 8, rng) for _ in range(4)]
        write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(mu)),
                      mu, snaps, combo, epsilon=-2.0)
    mu_dir, mu = nearest_mu_dir(combo_dir, mu_coex=0.1)
    assert mu == pytest.approx(0.0)  # 0.0 is nearest signed mu to 0.1
    arr, meta = extract_column_op(mu_dir)
    assert arr.shape == (4, 40)
    assert meta["Ly"] == 8


def test_sweep_bc_max_over_mu(tmp_path):
    """At one epsilon with two mu dirs (one phase-separated, one homogeneous),
    sweep_bc with mu_reduction=max keeps the MAX BC and reports which mu achieved it."""
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 60, "Ly": 10}
    cp = {**combo, "epsilon": -2.0}
    base = str(tmp_path / "results")
    combo_dir = os.path.join(base, combo_dir_name(cp))
    rng = np.random.default_rng(7)
    # mu=-0.4: homogeneous (unimodal); mu=0.0: phase separated (bimodal, higher BC)
    write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(-0.4)),
                  -0.4, [homogeneous(60, 10, rng) for _ in range(5)], cp, -2.0, beta=2.0)
    write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(0.0)),
                  0.0, [phase_separated(60, 10, rng) for _ in range(5)], cp, -2.0, beta=2.0)

    out_csv = str(tmp_path / "bc.csv")
    cache = str(tmp_path / "cache")
    rows = sweep_bc(base, combo, out_csv, cache, mu_reduction="max")
    assert len(rows) == 1
    r = rows[0]
    assert r["n_mu_scanned"] == 2
    assert r["mu_at_max"] == pytest.approx(0.0)         # bimodal mu wins
    assert r["BC"] > BC_BIMODAL_CUTOFF
    assert r["beta"] == pytest.approx(2.0)
    assert r["beta_epsilon"] == pytest.approx(2.0 * -2.0)  # beta*epsilon


def test_sweep_bc_zero_mean_picks_near_zero_magnetization(tmp_path):
    """zero_mean selects the mu with ⟨φ⟩ closest to 0 even if another mu has higher BC."""
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 0.0, "k": 0.0,
             "Lx": 60, "Ly": 10}
    cp = {**combo, "epsilon": -1.76}
    base = str(tmp_path / "results")
    combo_dir = os.path.join(base, combo_dir_name(cp))
    rng = np.random.default_rng(11)
    # skewed mostly-gas: high |mean|, can still have large BC
    gas_heavy = phase_separated(60, 10, rng)
    gas_heavy[:, :50] = EMPTY
    # balanced coexistence: ⟨φ⟩ closer to 0
    balanced = phase_separated(60, 10, rng)
    write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(-0.5)),
                  -0.5, [gas_heavy for _ in range(5)], cp, -1.76)
    write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(0.0)),
                  0.0, [balanced for _ in range(5)], cp, -1.76)

    rows = sweep_bc(base, combo, str(tmp_path / "bc.csv"), str(tmp_path / "cache"),
                       mu_reduction="zero_mean")
    assert len(rows) == 1
    assert rows[0]["mu_at_max"] == pytest.approx(0.0)
    assert abs(rows[0]["mean"]) < 0.35


def test_sweep_bc_writes_error_bar(tmp_path):
    combo = {"scheme": "homo", "delta_f": 0.0, "delta_mu": 1.0, "k": 1.0,
             "Lx": 60, "Ly": 10}
    cp = {**combo, "epsilon": -2.0}
    base = str(tmp_path / "results")
    combo_dir = os.path.join(base, combo_dir_name(cp))
    rng = np.random.default_rng(9)
    write_mu_dir(os.path.join(combo_dir, "mu_sweeps", mu_dir_name(0.0)),
                  0.0, [phase_separated(60, 10, rng) for _ in range(8)], cp, -2.0)
    rows = sweep_bc(base, combo, str(tmp_path / "bc.csv"), str(tmp_path / "cache"))
    assert "BC_err" in rows[0]
    assert np.isfinite(rows[0]["BC_err"]) and rows[0]["BC_err"] >= 0.0


def test_backfill_bc_err_from_cache(tmp_path):
    """Rows without BC_err get bootstrap errors from column_op cache."""
    rng = np.random.default_rng(7)
    arr = np.stack([phase_separated(60, 10, rng) for _ in range(6)], axis=0)
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
    filled, already, total = backfill_bc_err(
        str(bc_csv), cache_dir=str(cache_dir),
    )
    assert total == 1
    assert filled == 1
    df = pd.read_csv(bc_csv)
    assert np.isfinite(df["BC_err"].iloc[0])
    assert df["BC_err"].iloc[0] > 0


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
            cdir = os.path.join(base, folder, "results", combo_dir_name(cp))
            for mu in (-4.5, -4.4):
                snaps = ([phase_separated(Lx, Ly, rng) for _ in range(4)]
                         if eps <= eps_c else
                         [homogeneous(Lx, Ly, rng) for _ in range(4)])
                write_mu_dir(os.path.join(cdir, "mu_sweeps", mu_dir_name(mu)),
                              mu, snaps, cp, eps)

    # discovery walks the nested results/ roots
    assert discover_delta_mus(base, scheme, delta_f, Lx, Ly) == [1.0, 2.0]

    out_dir = str(tmp_path / "out")
    results = phase_diagram(base, scheme=scheme, delta_f=delta_f, k=k,
                               Lx=Lx, Ly=Ly, out_dir=out_dir, make_plots=False)
    assert {r["delta_mu"] for r in results} == {1.0, 2.0}
