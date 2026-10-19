"""Lattice states, densities, energy, initial configurations, and chi with its error."""

from __future__ import annotations

import numpy as np


EMPTY, INERT, BONDING = 0, 1, 2

def compute_densities(state: np.ndarray) -> tuple[float, float, float]:
    """Return (rho_bonding, rho_inert, rho_empty) — fractions of each species."""
    flat = state.ravel()
    n = flat.size
    rho_bonding = float(np.count_nonzero(flat == BONDING)) / n
    rho_inert = float(np.count_nonzero(flat == INERT)) / n
    rho_empty = float(np.count_nonzero(flat == EMPTY)) / n
    return rho_bonding, rho_inert, rho_empty


def compute_energy(
    state: np.ndarray, beta: float, epsilon: float, mu: float, delta_f: float
) -> float:
    """Total dimensionless energy βE (equals E at β=1).

    E = (βε/2) Σᵢ_{bonding} n_bonding_neighbors(i) − βμ·n_bonding − β(μ+Δf)·n_inert

    Matches the Rust Energy::record() implementation exactly.
    """

    # 2d Grid of 0's and 1's where 1's are the bonding sites
    bonding = (state == BONDING).astype(np.float64)
    bonding_neighbors = (
        np.roll(bonding, 1, axis=0) +
        np.roll(bonding, -1, axis=0) +
        np.roll(bonding, 1, axis=1) +
        np.roll(bonding, -1, axis=1)
    )
    e_interact = 0.5 * beta * epsilon * float(np.dot(bonding.ravel(), bonding_neighbors.ravel()))
    flat = state.ravel()
    n_bonding = float(np.count_nonzero(flat == BONDING))
    n_inert = float(np.count_nonzero(flat == INERT))
    e_chem = -beta * mu * n_bonding - beta * (mu + delta_f) * n_inert
    return e_interact + e_chem


def build_initial_state(Lx: int, Ly: int, fraction: float, seed: int) -> np.ndarray:
    """Random fraction of sites set to BONDING, rest EMPTY."""
    rng = np.random.default_rng(seed)
    state = np.zeros((Lx, Ly), dtype=np.uint32)
    n_active = int(round(fraction * Lx * Ly))
    if n_active > 0:
        idx = rng.choice(Lx * Ly, n_active, replace=False)
        state.ravel()[idx] = BONDING
    return state


def replica_fraction(base_fraction: float, run_id: int) -> float:
    """Alternate the initial BONDING fill between two mirror-image sides so
    replicas seed both branches of the magnetization distribution.

    Even run_id -> high side (base_fraction, e.g. 0.8 = 80% BONDING).
    Odd  run_id -> low  side (1 - base_fraction, e.g. 0.2 = 20% BONDING).
    With base_fraction=0.5 both sides coincide (symmetric, original behaviour).
    """
    high = max(base_fraction, 1.0 - base_fraction)
    low = 1.0 - high
    return high if run_id % 2 == 0 else low


def sem(values: np.ndarray) -> float:
    """Standard error of the mean."""
    n = len(values)
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


def compute_chi_err(
    m_abs_mean: float,
    m_abs_mean_err: float,
    m2_mean_err: float,
    n_sites: int,
    beta: float,
) -> float:
    """Delta-method SEM for χ = N·β·(⟨m²⟩-⟨|m|⟩²)."""
    factor = n_sites * beta
    var = (factor * m2_mean_err) ** 2 + (2.0 * factor * m_abs_mean * m_abs_mean_err) ** 2
    return float(np.sqrt(var))


def compute_chi(m2_mean: float, m_abs_mean: float, n_sites: int, beta: float) -> float:
    """χ = (1/NT)(⟨M²⟩-⟨|M|⟩²) with M = N·m, |M| = N·|m|, T = 1/β.

    ⟨M²⟩ = N²⟨m²⟩ and ⟨|M|⟩² = N²⟨|m|⟩², so this equals N·β·(⟨m²⟩-⟨|m|⟩²).
    Using ⟨|m|⟩ (not ⟨m⟩) removes the spurious symmetry-degeneracy term that small,
    flipping lattices would otherwise pick up — the connected (FSS) susceptibility.
    """
    M2_mean = n_sites ** 2 * m2_mean
    absM_mean = n_sites * m_abs_mean
    return beta / n_sites * (M2_mean - absM_mean ** 2)
