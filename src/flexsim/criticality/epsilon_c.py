"""Locate epsilon_c where BC crosses from bimodal to unimodal, with uncertainty."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from flexsim.criticality.bc import BC_BIMODAL_CUTOFF, TRANSITION_BC_HIGH, TRANSITION_BC_LOW
from flexsim.criticality.bc_sweep import EPS_C_FIELDS, append_row, sweep_bc


# ---------------------------------------------------------------------------
# Step 5 - locate epsilon_c
# ---------------------------------------------------------------------------

def _sigmoid(x, a, b, x_c, w):
    # a = high-x (unimodal ~1/3) asymptote, b = low-x (bimodal) plateau;
    # x_c = inflection point (the criticality estimate), w = crossover width.
    z = np.clip((x - x_c) / w, -700.0, 700.0)  # avoid exp overflow warnings
    return a + (b - a) / (1.0 + np.exp(z))


def _threshold_crossing(x, bc, level=BC_BIMODAL_CUTOFF):
    """Linear-interpolated x where BC(x) crosses `level` (descending)."""
    for i in range(len(x) - 1):
        y0, y1 = bc[i] - level, bc[i + 1] - level
        if y0 == 0:
            return float(x[i])
        if y0 * y1 < 0:
            t = y0 / (y0 - y1)
            return float(x[i] + t * (x[i + 1] - x[i]))
    return float("nan")


def _crossing_on_descending_curve(
    x: np.ndarray, bc: np.ndarray, level: float,
) -> float:
    """First x (scanning left->right) where BC(x) crosses `level` downward."""
    return _threshold_crossing(x, bc, level)


def _crossing_envelope(
    x: np.ndarray,
    bc: np.ndarray,
    bc_err: np.ndarray,
    level: float,
) -> tuple[float, float]:
    """Min/max crossing x at `level` using BC +/- BC_err on segment endpoints."""
    crossings: list[float] = []
    for i in range(len(x) - 1):
        e0, e1 = float(x[i]), float(x[i + 1])
        b0n, b1n = float(bc[i]), float(bc[i + 1])
        s0 = float(bc_err[i]) if np.isfinite(bc_err[i]) else 0.0
        s1 = float(bc_err[i + 1]) if np.isfinite(bc_err[i + 1]) else 0.0
        for ds0, ds1 in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            b0, b1 = b0n + ds0 * s0, b1n + ds1 * s1
            y0, y1 = b0 - level, b1 - level
            if y0 == 0:
                crossings.append(e0)
            elif y0 * y1 < 0:
                t = y0 / (y0 - y1)
                crossings.append(e0 + t * (e1 - e0))
    if crossings:
        return min(crossings), max(crossings)
    c = _crossing_on_descending_curve(x, bc, level)
    return c, c


def transition_bracket(
    x: np.ndarray,
    bc: np.ndarray,
    bc_err: Optional[np.ndarray] = None,
    *,
    bc_high: float = TRANSITION_BC_HIGH,
    bc_low: float = TRANSITION_BC_LOW,
) -> dict:
    """Bracket a gradual BC crossover using two contour levels.

    On a flat, noisy curve the sigmoid inflection is poorly constrained. This
    reports the beta*epsilon (or epsilon) interval where BC falls from bc_high
    to bc_low, plus a wider envelope when BC_err is available.

    Returns NaNs when a contour is never crossed on the supplied sweep.
    """
    x = np.asarray(x, dtype=float)
    bc = np.asarray(bc, dtype=float)
    order = np.argsort(x)
    x, bc = x[order], bc[order]
    if bc_err is not None:
        bc_err = np.asarray(bc_err, dtype=float)[order]

    x_high = _crossing_on_descending_curve(x, bc, bc_high)
    x_low = _crossing_on_descending_curve(x, bc, bc_low)
    half = float("nan")
    if np.isfinite(x_high) and np.isfinite(x_low) and x_low > x_high:
        half = (x_low - x_high) / 2.0

    half_env = float("nan")
    if bc_err is not None and np.all(np.isfinite(bc_err) & (bc_err > 0)):
        hi_lo, hi_hi = _crossing_envelope(x, bc, bc_err, bc_high)
        lo_lo, lo_hi = _crossing_envelope(x, bc, bc_err, bc_low)
        if all(np.isfinite(v) for v in (hi_lo, lo_hi)) and lo_hi > hi_lo:
            half_env = (lo_hi - hi_lo) / 2.0

    # Bracket widths kept for diagnostics only; locate_epsilon_c overwrites
    # recommended_uncertainty with the ε-grid neighbor spacing.
    recommended = half_env if np.isfinite(half_env) else half
    return {
        "transition_bc_high": bc_high,
        "transition_bc_low": bc_low,
        "transition_x_high": x_high,
        "transition_x_low": x_low,
        "transition_half_width": half,
        "transition_half_width_envelope": half_env,
        "recommended_uncertainty": recommended,
    }


def grid_neighbor_uncertainty(x: np.ndarray, x_c: float) -> float:
    """Uncertainty = distance from x_c to its farthest nearest grid neighbor.

    On a uniform spacing Δ (e.g. 0.005) this is Δ when x_c sits on a grid point,
    and at most Δ when x_c lies between two neighbors.
    """
    grid = np.unique(np.asarray(x, dtype=float))
    if grid.size < 2 or not np.isfinite(x_c):
        return float("nan")
    if x_c <= grid[0]:
        return float(grid[1] - grid[0])
    if x_c >= grid[-1]:
        return float(grid[-1] - grid[-2])
    right = int(np.searchsorted(grid, x_c, side="right"))
    left = right - 1
    if abs(float(grid[left]) - x_c) <= 1e-12:
        dists: list[float] = []
        if left > 0:
            dists.append(abs(float(grid[left] - grid[left - 1])))
        if left < grid.size - 1:
            dists.append(abs(float(grid[left + 1] - grid[left])))
        return float(max(dists)) if dists else float("nan")
    return float(max(abs(x_c - float(grid[left])), abs(float(grid[right]) - x_c)))


def locate_epsilon_c(
    rows: list[dict],
    L_long: int,
    *,
    x_col: str = "beta_epsilon",
    bc_level: float = BC_BIMODAL_CUTOFF,
) -> dict:
    """Estimate criticality from a linear fit of BC vs x intersecting bc_level.

    Fits BC = a + b·x (x = beta*epsilon by default) by OLS, then solves
    a + b·x_c = 5/9 (or `bc_level`). Reports epsilon_c_estimate = x_c / mean(beta)
    when x is beta*epsilon.

    Errors:
      * BC vs βε points still use per-point bootstrap `BC_err` (std error).
      * `fit_uncertainty` ≈ residual_SE / |slope| (maps BC scatter to x).
      * `recommended_uncertainty` is the distance from x_c to its farthest
        nearest neighbor on the discrete x-grid (ε spacing, typically 0.005).
    """
    sel = [r for r in rows
           if int(float(r["L_long"])) == int(L_long) and np.isfinite(float(r["BC"]))]

    def _err(r):
        try:
            return float(r.get("BC_err"))
        except (TypeError, ValueError):
            return float("nan")

    pts = sorted([(float(r[x_col]), float(r["BC"]), _err(r)) for r in sel],
                 key=lambda t: t[0])
    if len(pts) < 3:
        raise ValueError(f"need >=3 finite BC points for L_long={L_long}, got {len(pts)}")
    x = np.array([p[0] for p in pts], dtype=float)
    bc = np.array([p[1] for p in pts], dtype=float)
    bc_err = np.array([p[2] for p in pts], dtype=float)
    L_short = int(float(sel[0]["L_short"]))
    beta_mean = float(np.mean([float(r.get("beta", 1.0)) for r in sel]))

    level = float(bc_level)
    slope, intercept = [float(v) for v in np.polyfit(x, bc, 1)]
    if abs(slope) < 1e-15:
        raise ValueError("BC vs x linear fit has ~zero slope; cannot intersect level")
    x_c = (level - intercept) / slope
    bc_at = level  # by construction of the intersection
    method = "linear_bc_vs_x_5_9" if abs(level - BC_BIMODAL_CUTOFF) < 1e-12 else "linear_bc_vs_x"

    yhat = slope * x + intercept
    dof = max(len(x) - 2, 1)
    resid_se = float(np.sqrt(np.sum((bc - yhat) ** 2) / dof))
    fit_unc = resid_se / abs(slope)

    eps_c = x_c / beta_mean if (x_col == "beta_epsilon" and beta_mean) else x_c
    bracket = transition_bracket(x, bc, bc_err)
    grid_unc = grid_neighbor_uncertainty(x, float(x_c))
    return {
        "L_short": L_short,
        "L_long": int(L_long),
        "x_axis": x_col,
        "criticality_estimate": float(x_c),
        "epsilon_c_estimate": float(eps_c),
        "beta": beta_mean,
        "method": method,
        "fit_uncertainty": float(fit_unc),
        "BC_at_criticality": float(bc_at),
        "BC_target": level,
        "linear_slope": slope,
        "linear_intercept": intercept,
        **bracket,
        "recommended_uncertainty": grid_unc,
    }


# ---------------------------------------------------------------------------
# Orchestrator - the requested "find criticality of a coex run" function
# ---------------------------------------------------------------------------

def find_criticality(
    base_dir: str,
    combo_params: dict,
    *,
    out_dir: str = "criticality",
    manage_csv: Optional[str] = None,
    make_plots: bool = True,
) -> dict:
    """Full path: coex run data -> epsilon_c for one (scheme, size).

    combo_params: {scheme, delta_f, delta_mu, k, Lx, Ly} (epsilon omitted).
    Writes <out_dir>/bc_vs_epsilon.csv and appends to <out_dir>/epsilon_c.csv,
    caches column ops under <out_dir>/cache/column_op/, and (optionally) writes
    plots. Returns the epsilon_c dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    bc_csv = os.path.join(out_dir, "bc_vs_epsilon.csv")
    eps_c_csv = os.path.join(out_dir, "epsilon_c.csv")
    cache_dir = os.path.join(out_dir, "cache", "column_op")

    rows = sweep_bc(base_dir, combo_params, bc_csv, cache_dir, manage_csv=manage_csv)
    if not rows:
        raise RuntimeError(f"no BC rows produced for {combo_params} under {base_dir}")

    L_long = int(combo_params["Lx"])
    result = locate_epsilon_c(rows, L_long, x_col="beta_epsilon")
    append_row(eps_c_csv, result, EPS_C_FIELDS)

    if make_plots:
        try:
            from flexsim.criticality.plot_bimodality import plot_bc_vs_epsilon

            size = f"{int(combo_params['Lx'])}x{int(combo_params['Ly'])}"
            plot_bc_vs_epsilon(
                bc_csv, os.path.join(out_dir, f"bc_max_vs_beta_epsilon_{size}.png"),
                x_col="beta_epsilon", crit=result["criticality_estimate"],
                crit_row=result,
            )
        except Exception as exc:  # plotting is non-critical
            print(f"[bimodality] plotting skipped: {exc}", flush=True)

    rec = result.get("recommended_uncertainty", result["fit_uncertainty"])
    print(
        f"[bimodality] criticality({L_long}x{combo_params['Ly']}) = "
        f"(beta*eps)_c={result['criticality_estimate']:.4f} "
        f"+/- {rec} (grid neighbor)  "
        f"BC={result.get('BC_at_criticality', float('nan')):.4f} "
        f"(linear fit ∩ 5/9={BC_BIMODAL_CUTOFF:.4f})  "
        f"[epsilon_c={result['epsilon_c_estimate']:.4f}] ({result['method']})",
        flush=True,
    )
    return result
