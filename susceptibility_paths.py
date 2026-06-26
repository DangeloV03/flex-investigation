"""
Path helpers and constants for the susceptibility campaign.

Coexistence (slab): same geometry/IC as json_runner, under susceptibility_results/coex/.
Production (square L×L): susceptibility_runner, under susceptibility_results/susceptibility_{L}x{L}_.../
"""

from __future__ import annotations

import csv
import os

from combo_paths import combo_dir_name

COEX_SAMPLES_DIR = "susceptibility_samples/coex"
PROD_SAMPLES_DIR = "susceptibility_samples/prod"
EXACT_SAMPLES_DIR = "susceptibility_samples/exact"
EXACT_RANDOM_SAMPLES_DIR = "susceptibility_samples/exact_random"
COEX_RESULTS_DIR = "susceptibility_results/coex"
PROD_RESULTS_BASE = "susceptibility_results"
EXACT_RESULTS_BASE = "susceptibility_results/exact"
EXACT_RANDOM_RESULTS_BASE = "susceptibility_results/exact_random"
MANAGE_CSV = "susceptibility_manage.csv"
COEX_MANIFEST = "susceptibility_coex_queue.json"
PROD_MANIFEST = "susceptibility_prod_queue.json"
EXACT_MANIFEST = "susceptibility_exact_queue.json"
EXACT_RANDOM_MANIFEST = "susceptibility_exact_random_queue.json"
SUSCEPTIBILITY_DATA_CSV = "susceptibility_data.csv"

# Pre-SEM schema (smoke tests / early prod); current adds *_err columns after each moment/chi.
SUSCEPTIBILITY_CSV_FIELDS_LEGACY = [
    "id",
    "replica_id",
    "epsilon",
    "delta_f",
    "delta_mu",
    "k",
    "scheme",
    "L",
    "Lx",
    "Ly",
    "mu",
    "mu_coex_FITTED",
    "m_mean",
    "m2_mean",
    "m4_mean",
    "chi",
    "beta",
    "eq_time",
    "prod_time",
    "prod_chunks",
    "initial_fraction",
    "seed",
    "time",
]

# v1: added per-column SEM fields (27 total)
SUSCEPTIBILITY_CSV_FIELDS_V1 = [
    *SUSCEPTIBILITY_CSV_FIELDS_LEGACY[:13],
    "m_mean_err",
    "m2_mean",
    "m2_mean_err",
    "m4_mean",
    "m4_mean_err",
    "chi",
    "chi_err",
    *SUSCEPTIBILITY_CSV_FIELDS_LEGACY[16:],
]

# v2: adds energy columns (33 total)
SUSCEPTIBILITY_CSV_FIELDS = [
    *SUSCEPTIBILITY_CSV_FIELDS_V1,
    "e_mean",
    "e_mean_err",
    "e2_mean",
    "e2_mean_err",
]

_FIELDNAMES_BY_WIDTH = {
    len(SUSCEPTIBILITY_CSV_FIELDS): SUSCEPTIBILITY_CSV_FIELDS,
    len(SUSCEPTIBILITY_CSV_FIELDS_V1): SUSCEPTIBILITY_CSV_FIELDS_V1,
    len(SUSCEPTIBILITY_CSV_FIELDS_LEGACY): SUSCEPTIBILITY_CSV_FIELDS_LEGACY,
}

# Join prod jobs to coex manage rows (μ_coex is independent of square L).
COEX_LOOKUP_FIELDS = ["epsilon", "delta_f", "delta_mu", "k", "scheme"]

SQUARE_L_VALUES = [16, 32, 48, 64, 96, 128, 256]

# Ising-limit campaign defaults (β=1 => delta_f = βΔf).
ISING_DELTA_F = -20.0
ISING_K = 0.0
ISING_DELTA_MU = 0.0
ISING_SCHEME = "homo"


def read_susceptibility_csv(path: str) -> list[dict]:
    """Load prod CSV rows, tolerating mixed legacy (23-col) and current (27-col) lines."""
    if not os.path.isfile(path):
        return []

    rows: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header (may not match data rows after schema upgrade)
        for line_no, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue
            width = len(row)
            fieldnames = _FIELDNAMES_BY_WIDTH.get(width)
            if fieldnames is None:
                print(
                    f"Warning: skip {path}:{line_no} ({width} fields, expected "
                    f"{len(SUSCEPTIBILITY_CSV_FIELDS)} or "
                    f"{len(SUSCEPTIBILITY_CSV_FIELDS_LEGACY)})",
                    flush=True,
                )
                continue
            record = dict(zip(fieldnames, row))
            rows.append({field: record.get(field, "") for field in SUSCEPTIBILITY_CSV_FIELDS})
    return rows


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


def patch_coex_job_json(json_path: str) -> bool:
    """Ensure coex job JSON writes under susceptibility_results/coex (not results/).

    Stale copies restored from samples/coex/done/ often lack these fields.
    Returns True if the file was updated.
    """
    import json

    with open(json_path, encoding="utf-8") as f:
        job = json.load(f)
    changed = False
    if job.get("results_base") != COEX_RESULTS_DIR:
        job["results_base"] = COEX_RESULTS_DIR
        changed = True
    if job.get("manage_csv") != MANAGE_CSV:
        job["manage_csv"] = MANAGE_CSV
        changed = True
    if not changed:
        return False
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)
        f.write("\n")
    return True


def patch_prod_job_json(json_path: str) -> bool:
    """Ensure prod susceptibility jobs use the campaign results base and manage.csv."""
    import json

    with open(json_path, encoding="utf-8") as f:
        job = json.load(f)
    changed = False
    if job.get("results_base") != PROD_RESULTS_BASE:
        job["results_base"] = PROD_RESULTS_BASE
        changed = True
    if job.get("manage_csv") != MANAGE_CSV:
        job["manage_csv"] = MANAGE_CSV
        changed = True
    if not changed:
        return False
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)
        f.write("\n")
    return True


def patch_exact_job_json(json_path: str) -> bool:
    """Ensure exact-mu jobs write under susceptibility_results/exact/."""
    import json

    with open(json_path, encoding="utf-8") as f:
        job = json.load(f)
    if job.get("results_base") == EXACT_RESULTS_BASE:
        return False
    job["results_base"] = EXACT_RESULTS_BASE
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)
        f.write("\n")
    return True


def patch_exact_random_job_json(json_path: str) -> bool:
    """Ensure exact-random-IC jobs write under susceptibility_results/exact_random/."""
    import json

    with open(json_path, encoding="utf-8") as f:
        job = json.load(f)
    if job.get("results_base") == EXACT_RANDOM_RESULTS_BASE:
        return False
    job["results_base"] = EXACT_RANDOM_RESULTS_BASE
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)
        f.write("\n")
    return True


def prod_job_filename(scheme: str, epsilon: float, delta_mu: float, l: int) -> str:
    outer_tag = f"{eps_filename_tag(epsilon)}_{dmu_filename_tag(delta_mu)}"
    return f"susceptibility_{scheme}_{outer_tag}_L{l}.json"
