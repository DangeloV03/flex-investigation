"""Column order parameter phi_col from lattice snapshots and Sarle's bimodality coefficient."""

from __future__ import annotations

import glob
import json
import os
import time

import numpy as np

from flexsim.coex.layout import mu_dir_name


# Lattice state encoding (from coex/runner.py).
EMPTY, INERT, BONDING = 0, 1, 2

# Sarle heuristic cutoff / criticality level: BC = 5/9.
BC_BIMODAL_CUTOFF = 5.0 / 9.0

# Transition-bracket defaults: where BC has fallen partway down the crossover.
# Used to quote an honest uncertainty when the sigmoid inflection sits on a
# flat, noisy plateau (see criticality/UNCERTAINTY.md).
TRANSITION_BC_HIGH = 0.85
TRANSITION_BC_LOW = 0.65

# ---------------------------------------------------------------------------
# tag helpers
# ---------------------------------------------------------------------------

def untag(tag: str) -> float:
    """Inverse of coex.layout.param_tag: 'm2p0' -> -2.0, '0p5' -> 0.5."""
    neg = tag.startswith("m")
    body = tag[1:] if neg else tag
    value = float(body.replace("p", "."))
    return -value if neg else value


# ---------------------------------------------------------------------------
# Step 1 - column-order-parameter extraction
# ---------------------------------------------------------------------------

def column_op(snapshot: np.ndarray) -> np.ndarray:
    """Per-column phi order parameter for one (Lx, Ly) snapshot.

    Maps BONDING(=2) -> +1 and {INERT(=1), EMPTY(=0)} -> -1, then averages over
    the short axis (axis=1). Returns a length-Lx array in [-1, 1]:
    phi_col[x] = rho_bonding(x) - rho_inert(x) - rho_empty(x).
    """
    spin = np.where(snapshot == BONDING, 1.0, -1.0)
    return spin.mean(axis=1)


def extract_column_op(mu_dir: str) -> tuple[np.ndarray, dict]:
    """Load every final_lattice_*.npy in a mu dir and stack their column ops.

    Returns (arr, meta) where arr has shape (n_snapshots, Lx) and meta records
    provenance: n_snapshots, Lx, Ly, and the source file paths.
    """
    paths = sorted(glob.glob(os.path.join(mu_dir, "final_lattice_*.npy")))
    if not paths:
        raise FileNotFoundError(f"no final_lattice_*.npy in {mu_dir}")
    cols = []
    Lx = Ly = None
    for p in paths:
        snap = np.load(p)
        if snap.ndim != 2:
            raise ValueError(f"{p}: expected 2D (Lx, Ly), got shape {snap.shape}")
        if Lx is None:
            Lx, Ly = int(snap.shape[0]), int(snap.shape[1])
        cols.append(column_op(snap))
    arr = np.asarray(cols, dtype=float)  # (n_snapshots, Lx)
    meta = {
        "n_snapshots": len(paths),
        "Lx": Lx,
        "Ly": Ly,
        "source_paths": paths,
    }
    return arr, meta


def cache_column_op(
    mu_dir: str,
    cache_dir: str,
    *,
    epsilon: float,
    mu_used: float,
    mu_coex: float,
    combo_name: str,
) -> tuple[np.ndarray, dict]:
    """extract_column_op with an .npy cache + .json provenance sidecar.

    Cache key: <combo_name>__mu<tag>. Reuses the cache when it is newer than
    every source snapshot; otherwise recomputes and rewrites.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{combo_name}__{mu_dir_name(mu_used)}"
    npy_path = os.path.join(cache_dir, key + ".npy")
    json_path = os.path.join(cache_dir, key + ".json")

    sources = sorted(glob.glob(os.path.join(mu_dir, "final_lattice_*.npy")))
    if os.path.isfile(npy_path) and os.path.isfile(json_path) and sources:
        cache_mtime = os.path.getmtime(npy_path)
        if all(os.path.getmtime(s) <= cache_mtime for s in sources):
            arr = np.load(npy_path)
            with open(json_path) as f:
                meta = json.load(f)
            return arr, meta

    arr, meta = extract_column_op(mu_dir)
    meta.update(
        {
            "epsilon": epsilon,
            "mu_used": mu_used,
            "mu_coex": mu_coex,
            "combo_name": combo_name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    np.save(npy_path, arr)
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    return arr, meta


# ---------------------------------------------------------------------------
# Step 2 - pooling (single epsilon only, never across epsilon)
# ---------------------------------------------------------------------------

def pool_column_op(arr: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Flatten a single-epsilon (n_snapshots, Lx) column-op array.

    Returns (pooled_1d, n_pooled, L_long). L_long (= Lx) is returned separately
    because n_pooled conflates the true independent sample size (n_snapshots)
    with the column count (columns within a snapshot are spatially correlated).
    """
    arr = np.asarray(arr, dtype=float)
    L_long = int(arr.shape[1])
    pooled = arr.reshape(-1)
    return pooled, int(pooled.size), L_long


# ---------------------------------------------------------------------------
# Step 3 - Sarle's bimodality coefficient
# ---------------------------------------------------------------------------

def sarle_bc(pooled: np.ndarray) -> dict:
    """Sarle's bimodality coefficient of a pooled 1D sample.

    BC = (skew^2 + 1) / (kurtosis_excess + corr(n)),  corr(n)=3(n-1)^2/[(n-2)(n-3)].
    Uses sample skewness and EXCESS kurtosis (Gaussian -> 0).

    Reference limits:
      * unimodal Gaussian            -> BC -> 1/3 as n -> inf
      * two well-separated masses    -> BC -> 1   as n -> inf
      * heuristic cutoff BC > 5/9    -> "likely bimodal" (a convention, not a test)
    At the sample sizes here corr(n) sits very close to its asymptote of 3, so it
    barely moves BC; it is included for correctness.
    """
    x = np.asarray(pooled, dtype=float).ravel()
    n = x.size
    if n < 4:
        raise ValueError(f"need n >= 4 for the finite-sample correction, got {n}")
    mean = float(x.mean())
    std = float(x.std())  # population std (ddof=0)
    if std == 0.0:
        # A degenerate point mass: skew/kurtosis undefined; report a unimodal-ish BC.
        return {
            "n_pooled": n, "mean": mean, "std": 0.0,
            "skew": 0.0, "kurtosis_excess": 0.0, "BC": float("nan"),
        }
    z = (x - mean) / std
    skew = float(np.mean(z ** 3))
    kurt_excess = float(np.mean(z ** 4) - 3.0)
    corr = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    bc = (skew ** 2 + 1.0) / (kurt_excess + corr)
    return {
        "n_pooled": n,
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis_excess": kurt_excess,
        "BC": float(bc),
    }


def bc_bootstrap_error(arr: np.ndarray, *, n_boot: int = 200, seed: int = 0) -> float:
    """1-sigma bootstrap error on Sarle's BC for a (n_snapshots, Lx) column-op array.

    Resamples whole SNAPSHOTS (rows) with replacement, not individual columns:
    the independent units are the replicas, while columns within a snapshot are
    spatially correlated (the interface is one connected object), so a naive
    per-value bootstrap would understate the error. Returns the std of BC over
    n_boot resamples; NaN if there are fewer than 2 snapshots.
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[0]
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    bcs = []
    for _ in range(n_boot):
        pooled = arr[rng.integers(0, n, size=n)].reshape(-1)
        try:
            b = sarle_bc(pooled)["BC"]
        except ValueError:
            continue
        if np.isfinite(b):
            bcs.append(b)
    return float(np.std(bcs, ddof=1)) if len(bcs) > 1 else float("nan")
