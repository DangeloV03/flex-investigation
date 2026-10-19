"""Pick the mu directory (or mu_coex value) to analyse for one epsilon of a coex run."""

from __future__ import annotations

import csv
import glob
import os
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# mu selection - the mu-sweep gives one ensemble point per mu; we reduce over it
# ---------------------------------------------------------------------------

def read_output_field(mu_dir: str, field: str) -> Optional[float]:
    """First-row value of `field` from a mu dir's output.csv (e.g. signed mu,
    beta). The dir name only encodes abs(mu), so the signed value must be read
    from the CSV."""
    path = os.path.join(mu_dir, "output.csv")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                return float(row[field])
    except (KeyError, ValueError, StopIteration):
        return None
    return None


def enumerate_mu_dirs(combo_dir: str) -> list[tuple[str, float]]:
    """All (mu_dir, signed_mu) under combo_dir/mu_sweeps with a readable mu."""
    out = []
    for d in sorted(glob.glob(os.path.join(combo_dir, "mu_sweeps", "mu*"))):
        if not os.path.isdir(d):
            continue
        mu = read_output_field(d, "mu")
        if mu is not None:
            out.append((d, mu))
    return out


def nearest_mu_dir(combo_dir: str, mu_coex: float) -> tuple[str, float]:
    """Return (mu_dir, mu) whose signed mu is closest to mu_coex."""
    candidates = enumerate_mu_dirs(combo_dir)
    if not candidates:
        raise FileNotFoundError(
            f"no mu dir with a readable output.csv under {combo_dir}/mu_sweeps"
        )
    return min(candidates, key=lambda dm: abs(dm[1] - mu_coex))


def _mu_coex_from_phi_psi(combo_dir: str) -> Optional[float]:
    """Interpolate the mu where phi crosses zero from phi_psi.csv."""
    path = os.path.join(combo_dir, "phi_psi.csv")
    if not os.path.isfile(path):
        return None
    mus, phis = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                mus.append(float(row["mu"]))
                phis.append(float(row["phi"]))
            except (KeyError, ValueError):
                continue
    if len(mus) < 2:
        return None
    order = np.argsort(mus)
    mus = np.asarray(mus)[order]
    phis = np.asarray(phis)[order]
    for i in range(len(mus) - 1):
        if phis[i] == 0.0:
            return float(mus[i])
        if phis[i] * phis[i + 1] < 0:  # sign change -> linear interp to phi=0
            t = phis[i] / (phis[i] - phis[i + 1])
            return float(mus[i] + t * (mus[i + 1] - mus[i]))
    return float(mus[int(np.argmin(np.abs(phis)))])  # else nearest to zero


def resolve_mu_coex(
    epsilon: float,
    combo_dir: str,
    *,
    manage_csv: Optional[str] = None,
    combo_params: Optional[dict] = None,
) -> float:
    """mu_coex for one epsilon: manage.csv mu_coex_FITTED, else phi_psi.csv, else 2*eps."""
    if manage_csv and os.path.isfile(manage_csv):
        from flexsim.susc.sweep import load_mu_map

        cp = combo_params or {}
        mu_map = load_mu_map(
            manage_csv,
            delta_f=cp.get("delta_f"),
            delta_mu=cp.get("delta_mu"),
            k=cp.get("k"),
            scheme=cp.get("scheme"),
        )
        mu = mu_map.get(round(float(epsilon), 6))
        if mu is not None:
            return float(mu)
    mu = _mu_coex_from_phi_psi(combo_dir)
    if mu is not None:
        return mu
    return 2.0 * float(epsilon)
