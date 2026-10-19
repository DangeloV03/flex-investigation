"""Analyzer physics and decision rules: phi/psi, sign changes, refinement, mu_coex error."""

from __future__ import annotations

import pytest

import numpy as np

from flexsim.coex.criteria import (
    N_REFINEMENT_POINTS,
    PSI_COEX_MAX,
    compute_mu_coex_sim_error,
    extension_window,
    has_phi_sign_change,
    interior_psi_minimum,
    is_coex_resolved,
    is_psi_minimum_acceptable,
    refinement_mus,
    sign_change_bracket,
    unsampled_mus,
)
from flexsim.coex.phi_psi import build_curves, calculate_phi_psi
from coex_helpers import make_output_df


# ============================================================
# Analyzer — physics pure functions
# ============================================================

class TestCalculatePhiPsi:

    def test_phi_positive_when_active_dominant(self):
        df = make_output_df(rho_a=0.8, rho_i=0.1, rho_e=0.1)
        phi, _, psi, _ = calculate_phi_psi(df)
        assert phi > 0
        assert abs(psi - abs(phi)) < 1e-10

    def test_phi_negative_when_inert_and_empty_dominant(self):
        df = make_output_df(rho_a=0.1, rho_i=0.5, rho_e=0.4)
        phi, _, _, _ = calculate_phi_psi(df)
        assert phi < 0

    def test_psi_equals_abs_phi(self):
        df = make_output_df(rho_a=0.3, rho_i=0.4, rho_e=0.3)
        phi, _, psi, _ = calculate_phi_psi(df)
        assert abs(psi - abs(phi)) < 1e-10

    def test_error_nonnegative(self):
        df = make_output_df(rho_a=0.5, rho_i=0.2, rho_e=0.3)
        _, phi_err, _, psi_err = calculate_phi_psi(df)
        assert phi_err >= 0
        assert psi_err >= 0


class TestBuildCurves:

    def _points(self, mu_phi_pairs):
        """Build (mu, df) list with controlled phi values."""
        points = []
        for mu, rho_a in mu_phi_pairs:
            # rho_i=0, rho_e=0 → phi=rho_a
            df = make_output_df(rho_a=rho_a, rho_i=0.0, rho_e=0.0, mu=mu, n=50)
            points.append((mu, df))
        return points

    def test_build_curves_sorted(self):
        pts = self._points([(-3.0, 0.6), (-3.5, 0.8), (-2.5, 0.3)])
        mu_vals, phi_vals, *_ = build_curves(pts)
        assert list(mu_vals) == sorted(mu_vals)

    def test_build_curves_sign_tracks_rho_a(self):
        pts = self._points([(-3.0, 0.8), (-2.0, 0.1)])
        mu_vals, phi_vals, *_ = build_curves(pts)
        assert phi_vals[0] > phi_vals[1]


# ============================================================
# Analyzer — sign change & bracket
# ============================================================

class TestSignChange:

    def test_detects_sign_change(self):
        phi = np.array([0.3, 0.1, -0.1, -0.3])
        assert has_phi_sign_change(phi)

    def test_no_sign_change_all_positive(self):
        assert not has_phi_sign_change(np.array([0.1, 0.2, 0.3]))

    def test_no_sign_change_all_negative(self):
        assert not has_phi_sign_change(np.array([-0.3, -0.1, -0.05]))

    def test_bracket_correct_order(self):
        mu = np.array([-3.0, -2.5, -2.0, -1.5])
        phi = np.array([0.3, 0.1, -0.1, -0.3])
        lo, hi = sign_change_bracket(mu, phi)
        assert lo < hi
        assert lo == pytest.approx(-2.5, abs=1e-6)
        assert hi == pytest.approx(-2.0, abs=1e-6)

    def test_bracket_none_when_no_sign_change(self):
        mu = np.array([-3.0, -2.5, -2.0])
        phi = np.array([0.3, 0.2, 0.1])
        assert sign_change_bracket(mu, phi) is None


# ============================================================
# Analyzer — refinement_mus
# ============================================================

class TestRefinementMus:

    def test_returns_new_points_inside_bracket(self):
        mu_lo, mu_hi = -3.0, -2.0
        existing = np.array([-3.0, -2.0])
        new_mus = refinement_mus(mu_lo, mu_hi, existing)
        assert len(new_mus) > 0
        for m in new_mus:
            assert mu_lo <= m <= mu_hi
        assert all(not any(abs(m - e) < 1e-6 for e in existing) for m in new_mus)

    def test_returns_midpoints_when_dense_grid_fills_bracket(self):
        """When N linspace points are all already sampled, fall back to midpoints."""
        mu_lo, mu_hi = -3.0, -2.0
        # Fill the bracket with N_REFINEMENT_POINTS evenly-spaced points
        existing = np.linspace(mu_lo, mu_hi, N_REFINEMENT_POINTS)
        new_mus = refinement_mus(mu_lo, mu_hi, existing)
        # Should still find new midpoints between existing ones
        assert len(new_mus) > 0
        for m in new_mus:
            assert mu_lo <= m <= mu_hi

    def test_unsampled_mus_filters_existing(self):
        candidates = [-3.0, -2.5, -2.0]
        existing = np.array([-3.0, -2.0])
        result = unsampled_mus(candidates, existing)
        assert result == [-2.5]

    def test_unsampled_mus_tolerance(self):
        # 5e-7 < 1e-6 tolerance → treated as already sampled
        candidates = [-3.0000005]
        existing = np.array([-3.0])
        result = unsampled_mus(candidates, existing)
        assert result == []


# ============================================================
# Analyzer — psi / coexistence checks
# ============================================================

class TestPsiChecks:

    def test_psi_acceptable_below_threshold(self):
        psi = np.array([0.3, 0.01, 0.2])
        assert is_psi_minimum_acceptable(psi)

    def test_psi_not_acceptable_above_threshold(self):
        psi = np.array([0.3, 0.1, 0.2])
        assert not is_psi_minimum_acceptable(psi)

    def test_psi_boundary_exactly_at_threshold(self):
        psi = np.array([0.3, PSI_COEX_MAX, 0.2])
        assert is_psi_minimum_acceptable(psi)

    def test_interior_psi_minimum_true(self):
        psi = np.array([0.3, 0.01, 0.2])
        assert interior_psi_minimum(psi)

    def test_interior_psi_minimum_false_at_edge(self):
        psi = np.array([0.01, 0.1, 0.2])
        assert not interior_psi_minimum(psi)

    def test_interior_psi_minimum_too_short(self):
        assert not interior_psi_minimum(np.array([0.1, 0.0]))

    def test_is_coex_resolved_true(self):
        mu = np.array([-3.0, -2.5, -2.0, -1.5])
        phi = np.array([-0.03, -0.01, 0.01, 0.05])
        phi_err = np.array([0.1, 0.1, 0.1, 0.1])
        psi = np.abs(phi)
        assert is_coex_resolved(mu, phi, phi_err, psi)

    def test_is_coex_resolved_false_when_neighbors_not_close(self):
        mu = np.array([-3.0, -2.5, -2.0, -1.5])
        phi = np.array([-0.5, -0.1, 0.4, 0.8])
        phi_err = np.array([0.01, 0.01, 0.01, 0.01])
        psi = np.abs(phi)
        assert not is_coex_resolved(mu, phi, phi_err, psi)

    def test_is_coex_resolved_false_too_short(self):
        mu = np.array([-3.0, -2.0])
        phi = np.array([-0.1, 0.1])
        assert not is_coex_resolved(mu, phi, np.zeros(2), np.abs(phi))


# ============================================================
# Analyzer — extension_window
# ============================================================

class TestExtensionWindow:

    def test_extends_lower_when_all_phi_positive(self):
        mu = np.array([-3.0, -2.5, -2.0])
        phi = np.array([0.4, 0.2, 0.1])
        lo, hi = extension_window(mu, phi)
        assert hi == pytest.approx(-3.0, abs=1e-6)

    def test_extends_upper_when_all_phi_negative(self):
        mu = np.array([-3.0, -2.5, -2.0])
        phi = np.array([-0.4, -0.2, -0.1])
        lo, hi = extension_window(mu, phi)
        assert lo == pytest.approx(-2.0, abs=1e-6)

    def test_toward_edge_lower(self):
        mu = np.array([-3.0, -2.5, -2.0])
        phi = np.array([-0.2, 0.1, 0.3])
        lo, hi = extension_window(mu, phi, toward_edge=0)
        assert hi == pytest.approx(-3.0, abs=1e-6)

    def test_toward_edge_upper(self):
        mu = np.array([-3.0, -2.5, -2.0])
        phi = np.array([-0.2, 0.1, 0.3])
        lo, hi = extension_window(mu, phi, toward_edge=len(mu) - 1)
        assert lo == pytest.approx(-2.0, abs=1e-6)


# ============================================================
# Analyzer — compute_mu_coex_sim_error
# ============================================================

class TestMuCoexSimError:

    def test_uses_neighbor_errors_at_interior_min(self):
        mu = np.array([-3.0, -2.5, -2.0])
        psi = np.array([0.3, 0.01, 0.2])
        phi_errs = np.array([0.05, 0.02, 0.07])
        err = compute_mu_coex_sim_error(mu, phi_errs, psi)
        assert err == pytest.approx(max(0.05, 0.07), abs=1e-10)

    def test_uses_edge_error_at_boundary_min(self):
        mu = np.array([-3.0, -2.5, -2.0])
        psi = np.array([0.01, 0.1, 0.3])
        phi_errs = np.array([0.04, 0.02, 0.07])
        err = compute_mu_coex_sim_error(mu, phi_errs, psi)
        assert err == pytest.approx(0.04, abs=1e-10)
