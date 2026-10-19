"""Shared fixtures for the coex tests: synthetic output.csv frames and manage.csv rows."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from flexsim.coex.manage_csv import MANAGE_FIELDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_output_df(rho_a, rho_i, rho_e, mu=0.0, n=8):
    """Synthetic output.csv-like DataFrame for a single mu point."""
    return pd.DataFrame({
        "mu": [mu] * n,
        "rho_active": np.random.default_rng(42).normal(rho_a, 0.01, n),
        "rho_inert": np.random.default_rng(7).normal(rho_i, 0.01, n),
        "rho_empty": np.random.default_rng(13).normal(rho_e, 0.01, n),
        "epsilon": [-2.0] * n,
        "delta_f": [-20.0] * n,
        "delta_mu": [0.0] * n,
        "k": [0.0] * n,
        "scheme": ["homo"] * n,
        "Lx": [160] * n,
        "Ly": [16] * n,
    })


COMBO_DEFAULTS = {
    "epsilon": -2.0,
    "delta_f": -20.0,
    "delta_mu": 0.0,
    "k": 0.0,
    "scheme": "homo",
    "Lx": 160,
    "Ly": 16,
}

def write_raw_manage(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANAGE_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in MANAGE_FIELDS})


def default_manage_row(**overrides) -> dict:
    row = {
        "epsilon": -2.0, "delta_f": -20.0, "delta_mu": 0.0,
        "k": 0.0, "scheme": "homo", "Lx": 160, "Ly": 16,
        "mu_coex_FLEX": "", "isSubmitted": "", "isRan": "",
        "isAnalyzed": "", "mu_coex_FITTED": "", "mu_coex_FITTED_error": "",
        "RequestForAdditionalData": "0", "combo_path": "",
    }
    row.update(overrides)
    return row
