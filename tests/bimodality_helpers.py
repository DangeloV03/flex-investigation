"""Fabricate a tiny synthetic coex run on disk for the criticality tests."""

from __future__ import annotations

import csv
import os

import numpy as np

from flexsim.criticality.bc import BONDING, EMPTY


def write_mu_dir(mu_dir, mu, snapshots, combo_params, epsilon, beta=1.0):
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


def phase_separated(Lx, Ly, rng):
    """Slab: left half BONDING, right half EMPTY, with a jittered interface."""
    snap = np.full((Lx, Ly), EMPTY, dtype=np.uint32)
    edge = Lx // 2 + rng.integers(-1, 2)
    snap[:edge, :] = BONDING
    return snap


def homogeneous(Lx, Ly, rng):
    """Single phase near half density, no persistent interface (unimodal columns)."""
    return rng.integers(0, 2, size=(Lx, Ly)).astype(np.uint32) * BONDING
