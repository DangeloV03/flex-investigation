"""Sarle's BC known limits, the column order parameter, and pooling."""

from __future__ import annotations

import pytest

import numpy as np

from flexsim.criticality.bc import BONDING, EMPTY, INERT, column_op, pool_column_op, sarle_bc, untag


def test_bc_single_gaussian_near_one_third():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=200_000)
    bc = sarle_bc(x)["BC"]
    assert bc == pytest.approx(1.0 / 3.0, abs=0.02)


def test_bc_two_separated_gaussians_near_one():
    rng = np.random.default_rng(1)
    half = 100_000
    # Two tight, symmetric, well-separated peaks -> BC -> 1.
    x = np.concatenate([
        rng.normal(-1.0, 0.02, size=half),
        rng.normal(+1.0, 0.02, size=half),
    ])
    bc = sarle_bc(x)["BC"]
    assert bc > 0.9


def test_bc_affine_invariant():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(-1, 0.1, 50_000), rng.normal(1, 0.1, 50_000)])
    base = sarle_bc(x)["BC"]
    scaled = sarle_bc(3.7 * x - 42.0)["BC"]  # phi_col vs rho_bonding = affine map
    assert scaled == pytest.approx(base, rel=1e-9)


def test_bc_requires_min_samples():
    with pytest.raises(ValueError):
        sarle_bc(np.array([1.0, 2.0, 3.0]))


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
    phi = column_op(snap)
    assert phi.shape == (Lx,)
    np.testing.assert_allclose(phi, [1.0, -1.0, -1.0, 0.0])


def test_untag_roundtrip():
    from flexsim.coex.layout import param_tag
    for v in (-2.0, -1.45, 0.5, 0.0, 2.5):
        assert untag(param_tag(v)) == pytest.approx(v)


def test_pool_column_op_shapes():
    arr = np.arange(4 * 40, dtype=float).reshape(4, 40)
    pooled, n_pooled, L_long = pool_column_op(arr)
    assert pooled.shape == (160,)
    assert n_pooled == 160
    assert L_long == 40
