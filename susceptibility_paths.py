"""
Path helpers and constants for the susceptibility campaign.

Coexistence (slab): same geometry/IC as json_runner, under susceptibility_results/coex/.
Production (square L×L): susceptibility_runner, under susceptibility_results/susceptibility_{L}x{L}_.../
"""

from __future__ import annotations

import os

from combo_paths import combo_dir_name

COEX_SAMPLES_DIR = "susceptibility_samples/coex"
PROD_SAMPLES_DIR = "susceptibility_samples/prod"
COEX_RESULTS_DIR = "susceptibility_results/coex"
PROD_RESULTS_BASE = "susceptibility_results"
MANAGE_CSV = "susceptibility_manage.csv"
COEX_MANIFEST = "susceptibility_coex_queue.json"
PROD_MANIFEST = "susceptibility_prod_queue.json"
SUSCEPTIBILITY_DATA_CSV = "susceptibility_data.csv"

# Join prod jobs to coex manage rows (μ_coex is independent of square L).
COEX_LOOKUP_FIELDS = ["epsilon", "delta_f", "delta_mu", "k", "scheme"]

SQUARE_L_VALUES = [16, 32, 48, 64, 96, 128, 256]

# Ising-limit campaign defaults (β=1 => delta_f = βΔf).
ISING_DELTA_F = -20.0
ISING_K = 0.0
ISING_DELTA_MU = 0.0
ISING_SCHEME = "homo"


def eps_filename_tag(epsilon: float) -> str:
    return "eps" + str(abs(epsilon)).replace(".", "p")


def dmu_filename_tag(delta_mu: float) -> str:
    body = str(abs(delta_mu)).replace(".", "p")
    if delta_mu < 0:
        return f"dm-{body}"
    return f"dm{body}"


def susceptibility_prod_dir_name(params: dict) -> str:
    """Directory name for one square-L production run."""
    return f"susceptibility_{combo_dir_name(params)}"


def susceptibility_prod_dir(params: dict, base: str = PROD_RESULTS_BASE) -> str:
    return os.path.join(base, susceptibility_prod_dir_name(params))


def coex_combo_dir(params: dict) -> str:
    return os.path.join(COEX_RESULTS_DIR, combo_dir_name(params))


def coex_job_filename(scheme: str, epsilon: float, delta_mu: float, ly: int, mu_idx: int) -> str:
    outer_tag = f"{eps_filename_tag(epsilon)}_{dmu_filename_tag(delta_mu)}"
    return f"{scheme}_{outer_tag}_Ly{ly}_mu{mu_idx:02d}.json"


def prod_job_filename(scheme: str, epsilon: float, delta_mu: float, l: int) -> str:
    outer_tag = f"{eps_filename_tag(epsilon)}_{dmu_filename_tag(delta_mu)}"
    return f"susceptibility_{scheme}_{outer_tag}_L{l}.json"
