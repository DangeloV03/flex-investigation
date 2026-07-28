"""
Path helpers and constants for the susceptibility campaign.

Coexistence (slab): same geometry/IC as json_runner, under susceptibility/results/coex/.

Production (square L×L): two layouts:
  Legacy: susceptibility/results/susceptibility_{L}x{L}_.../
  Smart runner (SUSC_RUNS): SUSC_RUNS/_{Lx}_{Ly}_S{n}_DF{df}_DMU{dmu}_K{k}/_{epsilon}/
"""

from __future__ import annotations

import csv
import os
import re

from combo_paths import combo_dir_name

COEX_SAMPLES_DIR = "susceptibility/samples/coex"
COEX_RESULTS_DIR = "susceptibility/results/coex"
MANAGE_CSV = "susceptibility/manage.csv"
COEX_MANIFEST = "susceptibility/coex_queue.json"
SUSCEPTIBILITY_DATA_CSV = "susceptibility_data.csv"

# Smart-runner output base (SUSC_RUNS layout).
SUSC_RUNS_BASE = "SUSC_RUNS"
PROD_RESULTS_BASE = SUSC_RUNS_BASE

# Scheme name → integer code used in SUSC_RUNS directory names.
SCHEME_CODES: dict[str, int] = {"homo": 1, "positive": 2, "negative": 3}

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
SUSCEPTIBILITY_CSV_FIELDS_V2 = [
    *SUSCEPTIBILITY_CSV_FIELDS_V1,
    "e_mean",
    "e_mean_err",
    "e2_mean",
    "e2_mean_err",
]

# v3: adds resume_id for top-up lineage tracking (34 total)
SUSCEPTIBILITY_CSV_FIELDS = [
    *SUSCEPTIBILITY_CSV_FIELDS_V2,
    "resume_id",  # empty for fresh runs; original replica's id for top-up rows
]

_FIELDNAMES_BY_WIDTH = {
    len(SUSCEPTIBILITY_CSV_FIELDS): SUSCEPTIBILITY_CSV_FIELDS,
    len(SUSCEPTIBILITY_CSV_FIELDS_V2): SUSCEPTIBILITY_CSV_FIELDS_V2,
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


def find_susceptibility_csvs(results_dir: str) -> list[str]:
    """All susceptibility_data.csv files for the phase rooted at results_dir.

    Run dirs are always {base}/susceptibility_{combo}/ for every phase, so glob
    exactly one level deep. This keeps phases isolated: pointing at the prod base
    (susceptibility/results/) no longer sweeps up coex/, exact/, exact_random/,
    exact_split/ nested underneath it.
    """
    import glob

    return sorted(
        glob.glob(os.path.join(results_dir, "susceptibility_*", SUSCEPTIBILITY_DATA_CSV))
    )


def eps_filename_tag(epsilon: float) -> str:
    return "eps" + str(abs(epsilon)).replace(".", "p")


def dmu_filename_tag(delta_mu: float) -> str:
    body = str(abs(delta_mu)).replace(".", "p")
    if delta_mu < 0:
        return f"dm-{body}"
    return f"dm{body}"


def susceptibility_prod_dir_name(params: dict) -> str:
    """Directory name for legacy square-L production run."""
    return f"susceptibility_{combo_dir_name(params)}"


# ---------------------------------------------------------------------------
# SUSC_RUNS path helpers (smart-runner layout)
# ---------------------------------------------------------------------------

def _float_tag(v: float) -> str:
    """Clean float string: removes trailing zeros but always keeps one decimal."""
    s = f"{v:g}"
    if "." not in s and "e" not in s.lower():
        s += ".0"
    return s


def susc_param_dir_name(params: dict) -> str:
    """_{Lx}_{Ly}_S{n}_DF{df}_DMU{dmu}_K{k} — param-level dir inside SUSC_RUNS."""
    scheme = str(params.get("scheme", "homo"))
    n = SCHEME_CODES.get(scheme, scheme)
    lx = int(params["Lx"])
    ly = int(params["Ly"])
    df = _float_tag(float(params["delta_f"]))
    dmu = _float_tag(float(params["delta_mu"]))
    k = _float_tag(float(params["k"]))
    return f"_{lx}_{ly}_S{n}_DF{df}_DMU{dmu}_K{k}"


def susc_eps_dir_name(epsilon: float) -> str:
    """_{epsilon} — epsilon-level dir inside a param dir."""
    return f"_{_float_tag(epsilon)}"


def susc_run_dir(params: dict, base: str = SUSC_RUNS_BASE) -> str:
    """Full path: base/_{Lx}_{Ly}_S{n}_.../_epsilon/ for a SUSC_RUNS campaign."""
    return os.path.join(
        base,
        susc_param_dir_name(params),
        susc_eps_dir_name(float(params["epsilon"])),
    )


def parse_susc_run_dir(dirpath: str) -> tuple[int | None, float | None]:
    """Parse (L, epsilon) from a SUSC_RUNS epsilon-level directory path.

    Expects the last two components to be _{Lx}_{Ly}_S{n}_...  and  _{epsilon}.
    Returns (None, None) if the path doesn't match.
    """
    parts = os.path.normpath(dirpath).split(os.sep)
    if len(parts) < 2:
        return None, None
    eps_dir = parts[-1]   # e.g. "_-1.76"
    param_dir = parts[-2] # e.g. "_48_48_S1_DF-20.0_DMU0.0_K0.0"
    try:
        eps = float(eps_dir.lstrip("_"))
    except ValueError:
        return None, None
    m = re.match(r"_(\d+)_", param_dir)
    L = int(m.group(1)) if m else None
    return L, eps


def find_susc_run_csvs(base: str) -> list[str]:
    """Find all susceptibility_data.csv files under a SUSC_RUNS base.

    Globs two levels deep: {base}/*/_*/{SUSCEPTIBILITY_DATA_CSV}.
    """
    import glob
    return sorted(
        glob.glob(os.path.join(base, "*", "_*", SUSCEPTIBILITY_DATA_CSV))
    )


def susceptibility_prod_dir(params: dict, base: str = PROD_RESULTS_BASE) -> str:
    """Output directory for one square-L production run (SUSC_RUNS layout)."""
    return susc_run_dir(params, base)


def coex_combo_dir(params: dict) -> str:
    return os.path.join(COEX_RESULTS_DIR, combo_dir_name(params))


def coex_job_filename(scheme: str, epsilon: float, delta_mu: float, ly: int, mu_idx: int) -> str:
    outer_tag = f"{eps_filename_tag(epsilon)}_{dmu_filename_tag(delta_mu)}"
    return f"{scheme}_{outer_tag}_Ly{ly}_mu{mu_idx:02d}.json"


def patch_coex_job_json(json_path: str) -> bool:
    """Ensure coex job JSON writes under susceptibility/results/coex (not results/).

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




def prod_job_filename(scheme: str, epsilon: float, delta_mu: float, l: int) -> str:
    outer_tag = f"{eps_filename_tag(epsilon)}_{dmu_filename_tag(delta_mu)}"
    return f"susceptibility_{scheme}_{outer_tag}_L{l}.json"
