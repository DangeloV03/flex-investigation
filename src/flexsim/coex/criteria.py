"""Decision rules for a coex combo: is mu_coex resolved, which mu to sample next, fit mu_coex."""

from __future__ import annotations

import time

import numpy as np

from flexsim.coex.layout import combo_dir
from flexsim.coex.manage_csv import update_manage_field
from flexsim.coex.phi_psi import plot_combo


MAX_ADDITIONAL_REQUESTS = 10
N_INITIAL_MU_POINTS = 10  # must match generate.N_MU_POINTS
N_REFINEMENT_POINTS = 10
PHI_NEIGHBOR_SIGMA_K = 2.0  # |phi| <= k * max(phi_err, PHI_ABS_TOL) counts as "close"
PHI_ABS_TOL = 0.05
PSI_COEX_MAX = 0.025  # min(psi) must be at or below this to accept mu_coex_FITTED
# ---------------------------------------------------------------------------
# Coexistence resolution check
# ---------------------------------------------------------------------------

def phi_is_close_to_zero(phi: float, phi_err: float) -> bool:
    """True if |phi| is within k sigma of zero (with an absolute floor)."""
    threshold = PHI_NEIGHBOR_SIGMA_K * max(phi_err, PHI_ABS_TOL)
    return abs(phi) <= threshold


def is_coex_resolved(
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
) -> bool:
    """True when min(psi) has neg/pos phi neighbors that are both near zero."""
    if len(mu_vals) < 3:
        return False

    min_idx = int(np.argmin(psi_vals))
    if min_idx == 0 or min_idx == len(mu_vals) - 1:
        return False

    phi_left = phi_vals[min_idx - 1]
    phi_right = phi_vals[min_idx + 1]
    if phi_left >= 0 or phi_right <= 0:
        return False

    return (
        phi_is_close_to_zero(phi_left, phi_errs[min_idx - 1])
        and phi_is_close_to_zero(phi_right, phi_errs[min_idx + 1])
    )


def psi_min_index(psi_vals: np.ndarray) -> int:
    return int(np.argmin(psi_vals))


def min_psi_value(psi_vals: np.ndarray) -> float:
    return float(np.min(psi_vals))


def is_psi_minimum_acceptable(psi_vals: np.ndarray) -> bool:
    """True when argmin(psi) is small enough to trust mu_coex_SIM."""
    return min_psi_value(psi_vals) <= PSI_COEX_MAX


def interior_psi_minimum(psi_vals: np.ndarray) -> bool:
    """True when argmin(psi) is not at the edge of the sampled mu range."""
    if len(psi_vals) < 3:
        return False
    min_idx = psi_min_index(psi_vals)
    return 0 < min_idx < len(psi_vals) - 1


def has_phi_sign_change(phi_vals: np.ndarray) -> bool:
    signs = np.sign(phi_vals)
    return not np.all(signs == signs[0])


def sign_change_bracket(
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
) -> tuple[float, float] | None:
    """Return (mu_lo, mu_hi) bracketing phi=0, or None if no sign change."""
    if not has_phi_sign_change(phi_vals):
        return None
    pos_mask = phi_vals > 0
    neg_mask = phi_vals < 0
    mu_pos = float(mu_vals[pos_mask][np.argmin(np.abs(phi_vals[pos_mask]))])
    mu_neg = float(mu_vals[neg_mask][np.argmin(np.abs(phi_vals[neg_mask]))])
    return min(mu_pos, mu_neg), max(mu_pos, mu_neg)


def unsampled_mus(candidate_mus: list[float], mu_vals: np.ndarray) -> list[float]:
    return [
        m for m in candidate_mus
        if not any(abs(m - float(existing)) < 1e-6 for existing in mu_vals)
    ]


def refinement_mus(mu_lo: float, mu_hi: float, mu_vals: np.ndarray) -> list[float]:
    """Return new mu points inside [mu_lo, mu_hi], subdividing when coarse grid fills bracket."""
    for n in (N_REFINEMENT_POINTS, 2 * N_REFINEMENT_POINTS, 4 * N_REFINEMENT_POINTS):
        candidates = [round(float(m), 6) for m in np.linspace(mu_lo, mu_hi, n)]
        new_mus = unsampled_mus(candidates, mu_vals)
        if new_mus:
            return new_mus

    in_bracket = sorted(
        float(m) for m in mu_vals if mu_lo - 1e-9 <= float(m) <= mu_hi + 1e-9
    )
    midpoints = [
        round(0.5 * (in_bracket[i] + in_bracket[i + 1]), 6)
        for i in range(len(in_bracket) - 1)
    ]
    return unsampled_mus(midpoints, mu_vals)


def count_in_bracket(mu_vals: np.ndarray, mu_lo: float, mu_hi: float) -> int:
    return int(np.sum((mu_vals >= mu_lo) & (mu_vals <= mu_hi)))


def extension_window(
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    *,
    toward_edge: int | None = None,
) -> tuple[float, float]:
    """Return (mu_lo, mu_hi) for a window extension.

    If toward_edge is 0, extend below the current range; if len(mu)-1, extend above.
    Otherwise infer direction from phi sign (all positive -> lower mu, else higher).
    """
    window = float(mu_vals[-1] - mu_vals[0])
    if toward_edge == 0:
        return float(mu_vals[0] - window), float(mu_vals[0])
    if toward_edge is not None and toward_edge == len(mu_vals) - 1:
        return float(mu_vals[-1]), float(mu_vals[-1] + window)
    if np.all(phi_vals > 0):
        return float(mu_vals[0] - window), float(mu_vals[0])
    return float(mu_vals[-1]), float(mu_vals[-1] + window)


def compute_mu_coex_sim_error(
    mu_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
) -> float:
    """Replica-scatter scale at mu neighbors bracketing argmin(psi).

    This is a phi uncertainty proxy used for resolution checks, not a Delta-mu
    error bar on mu_coex_SIM.
    """
    min_idx = psi_min_index(psi_vals)
    if min_idx == 0 or min_idx == len(mu_vals) - 1:
        return float(phi_errs[min_idx])
    return float(max(phi_errs[min_idx - 1], phi_errs[min_idx + 1]))


def fit_zero_crossing_with_error(
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
) -> tuple[float, float] | tuple[None, None]:
    """Weighted linear fit to phi(mu); return (mu_zero, mu_zero_err).

    Uses points inside the sign-change bracket plus one neighbour on each side.
    Weights are 1/phi_err^2. Error on the zero crossing is propagated from the
    fit covariance matrix. Returns (None, None) when no sign change exists or
    the fit is degenerate.
    """
    bracket = sign_change_bracket(mu_vals, phi_vals)
    if bracket is None:
        return None, None

    mu_lo, mu_hi = bracket
    sorted_mus = np.sort(mu_vals)
    lo_idx = int(np.searchsorted(sorted_mus, mu_lo))
    hi_idx = int(np.searchsorted(sorted_mus, mu_hi))
    expanded_lo = float(sorted_mus[max(0, lo_idx - 1)])
    expanded_hi = float(sorted_mus[min(len(sorted_mus) - 1, hi_idx + 1)])

    mask = (mu_vals >= expanded_lo - 1e-9) & (mu_vals <= expanded_hi + 1e-9)
    mu_fit = mu_vals[mask]
    phi_fit = phi_vals[mask]
    err_fit = phi_errs[mask]

    if len(mu_fit) < 2:
        return None, None

    w = np.where(err_fit > 0, 1.0 / (err_fit ** 2), 1.0)
    try:
        coeffs, cov = np.polyfit(mu_fit, phi_fit, deg=1, w=np.sqrt(w), cov=True)
    except (np.linalg.LinAlgError, ValueError):
        return None, None

    slope, intercept = coeffs
    if abs(slope) < 1e-12:
        return None, None

    mu_zero = float(-intercept / slope)

    # Error propagation: var(mu_zero) via delta method
    var_s, var_i = float(cov[0, 0]), float(cov[1, 1])
    cov_si = float(cov[0, 1])
    var_mu = (
        (intercept / slope ** 2) ** 2 * var_s
        + (1.0 / slope) ** 2 * var_i
        - 2.0 * (intercept / slope ** 3) * cov_si
    )
    mu_zero_err = float(np.sqrt(max(0.0, var_mu)))
    return mu_zero, mu_zero_err


def finalize_combo(
    combo_key: tuple,
    combo: dict,
    tag: str,
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
    psi_errs: np.ndarray,
    manage_path: str,
    results_dir: str,
    n_requests: int,
    reason: str = "",
):
    """Set mu_coex_FITTED via linear fit zero crossing, save plot, mark combo analyzed."""
    mu_coex_fitted, fitted_error = fit_zero_crossing_with_error(mu_vals, phi_vals, phi_errs)
    if mu_coex_fitted is None:
        # fallback to argmin(psi) if fit fails (e.g. only 1 bracket point)
        min_idx = int(np.argmin(psi_vals))
        mu_coex_fitted = float(mu_vals[min_idx])
        fitted_error = compute_mu_coex_sim_error(mu_vals, phi_errs, psi_vals)
    suffix = f" ({reason})" if reason else ""
    print(f"[analyzer] {tag}: mu_coex_FITTED = {mu_coex_fitted:.6f}, "
          f"error = {fitted_error:.6f}{suffix}")

    plot_combo(
        combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
        mu_coex_sim=mu_coex_fitted, results_dir=results_dir,
    )
    update_manage_field(manage_path, combo, {
        "mu_coex_FITTED": mu_coex_fitted,
        "mu_coex_FITTED_error": fitted_error,
        "isAnalyzed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "RequestForAdditionalData": n_requests,
        "combo_path": combo_dir(combo, results_dir),
    })


def finalize_unstable(
    combo_key: tuple,
    combo: dict,
    tag: str,
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
    psi_errs: np.ndarray,
    manage_path: str,
    results_dir: str,
    n_requests: int,
    reason: str = "",
):
    """Mark unstable combo analyzed with mu_coex_FITTED=NaN after max refinement requests."""
    suffix = f" ({reason})" if reason else ""
    print(f"[analyzer] {tag}: unstable, mu_coex_FITTED=NaN{suffix}")
    plot_combo(
        combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
        mu_coex_sim=None, results_dir=results_dir,
    )
    update_manage_field(manage_path, combo, {
        "mu_coex_FITTED": "NaN",
        "mu_coex_FITTED_error": "NaN",
        "isAnalyzed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "RequestForAdditionalData": n_requests,
        "combo_path": combo_dir(combo, results_dir),
    })
