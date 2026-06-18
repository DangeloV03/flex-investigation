"""
generate_samples.py

LHS campaign over (epsilon, delta_mu) for homo / Ly=32:
  1. Draw N_LHS Latin-hypercube samples, snap to grid, dedupe.
  2. Skip combos already in manage.csv, queue, samples/, or results/.
  3. Compute mu_coex_FLEX per combo; skip if mu_coex_FLEX > 0.
  4. Sweep mu over mu_coex_FLEX +/- 0.1 (10 points), write JSON files.
  5. Merge new rows into manage.csv and append jobs to run_all_queue.json.

Usage:
    python generate_samples.py
"""

import csv
import glob
import json
import os
import pathlib
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

from flex_coex_chemical_potential_prediction import coex_chemical_potential
from combo_paths import COMBO_KEY_FIELDS, combo_dir, combo_has_results, combo_key_from_dict
from queue_manifest import merge_pending, read_manifest

# ---------------------------------------------------------------------------
# Campaign constants
# ---------------------------------------------------------------------------

SCHEME = "homo"
FLEX_INDEX = 1
LY = 32
LX = 320
DELTA_F = 0.0
K = 1.0

EPS_BOUNDS = (-2.5, -1.4)
EPS_STEP = 0.1
DMU_BOUNDS = (-1.0, 6.0)
DMU_STEP = 0.5
N_LHS = 150
LHS_SEED = 42

MU_WINDOW = 0.1
N_MU_POINTS = 10

RUN_SETTINGS = {
    "beta": 1.0,
    "k": K,
    "initial_condition": "slab_half_active_half_empty",
    "num_parallel_runs": 4,
    "eq_time": 10000.0,
    "prod_time": 10000.0,
    "seed_base": 1000,
}

OUTPUT_DIR = "samples"
MANAGE_CSV = "manage.csv"
MANIFEST_PATH = "run_all_queue.json"
RESULTS_DIR = "results"

COMBO_KEY_FIELDS = COMBO_KEY_FIELDS  # re-export
MANAGE_FIELDS = COMBO_KEY_FIELDS + [
    "mu_coex_FLEX",
    "isSubmitted",
    "isRan",
    "isAnalyzed",
    "mu_coex_SIM",
    "mu_coex_SIM_error",
    "RequestForAdditionalData",
    "combo_path",
]


def snap_to_grid(value: float, lo: float, step: float) -> float:
    idx = round((value - lo) / step)
    return round(lo + idx * step, 6)


def lhs_outer_combos(n: int = N_LHS, seed: int = LHS_SEED) -> list[tuple[float, float]]:
    """Draw LHS samples in (epsilon, delta_mu), snap to grid, return unique pairs."""
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n=n)
    eps_lo, eps_hi = EPS_BOUNDS
    dmu_lo, dmu_hi = DMU_BOUNDS
    scaled = qmc.scale(
        unit,
        [eps_lo, dmu_lo],
        [eps_hi, dmu_hi],
    )
    seen: set[tuple[float, float]] = set()
    pairs: list[tuple[float, float]] = []
    for eps_raw, dmu_raw in scaled:
        eps = snap_to_grid(float(eps_raw), eps_lo, EPS_STEP)
        dmu = snap_to_grid(float(dmu_raw), dmu_lo, DMU_STEP)
        key = (eps, dmu)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def combo_dict(epsilon: float, delta_mu: float) -> dict:
    return {
        "epsilon": epsilon,
        "delta_f": DELTA_F,
        "delta_mu": delta_mu,
        "k": K,
        "scheme": SCHEME,
        "Lx": LX,
        "Ly": LY,
    }


def eps_filename_tag(epsilon: float) -> str:
    return "eps" + str(abs(epsilon)).replace(".", "p")


def dmu_filename_tag(delta_mu: float) -> str:
    body = str(abs(delta_mu)).replace(".", "p")
    if delta_mu < 0:
        return f"dm-{body}"
    return f"dm{body}"


def mu_sweep(mu_coex_flex: float) -> list[float]:
    values = np.linspace(mu_coex_flex - MU_WINDOW, mu_coex_flex + MU_WINDOW, N_MU_POINTS)
    return [round(float(v), 6) for v in values]


def read_manage(manage_path: str) -> list[dict]:
    if not os.path.isfile(manage_path):
        return []
    with open(manage_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.setdefault("mu_coex_SIM_error", "")
        row.setdefault("combo_path", "")
    return rows


def write_manage(manage_path: str, rows: list[dict]) -> None:
    with open(manage_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANAGE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANAGE_FIELDS})


def combo_from_json_path(json_path: str) -> tuple | None:
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path) as f:
            job = json.load(f)
        return combo_key_from_dict(job)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def collect_active_combo_keys(
    manage_path: str,
    manifest_path: str,
    samples_dir: str,
    results_dir: str,
) -> dict[tuple, str]:
    """Return combo_key -> reason for combos already submitted/running/done."""
    active: dict[tuple, str] = {}

    for row in read_manage(manage_path):
        if any(row.get(f, "") for f in ("isSubmitted", "isRan", "isAnalyzed")):
            key = combo_key_from_dict(row)
            active.setdefault(key, "manage")

    manifest = read_manifest(manifest_path)
    for json_path in manifest.get("pending", []) + list(manifest.get("in_flight", {}).values()):
        key = combo_from_json_path(json_path)
        if key is not None:
            active.setdefault(key, "queue")

    for json_path in glob.glob(os.path.join(samples_dir, "*.json")):
        key = combo_from_json_path(json_path)
        if key is not None:
            active.setdefault(key, "samples")

    results_path = pathlib.Path(results_dir)
    for csv_path in results_path.glob("*/*/output.csv"):
        try:
            df = pd.read_csv(csv_path, nrows=1)
            if not all(col in df.columns for col in COMBO_KEY_FIELDS):
                continue
            job = {
                f: df[f].iloc[0].item() if hasattr(df[f].iloc[0], "item")
                else df[f].iloc[0]
                for f in COMBO_KEY_FIELDS
            }
            key = combo_key_from_dict(job)
            active.setdefault(key, "results")
        except Exception:
            continue

    for row in read_manage(manage_path):
        combo = {f: row[f] for f in COMBO_KEY_FIELDS}
        if combo_has_results(combo, results_dir):
            key = combo_key_from_dict(combo)
            active.setdefault(key, "results")

    return active


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    outer_pairs = lhs_outer_combos()
    print(f"LHS: {N_LHS} draws -> {len(outer_pairs)} unique snapped (epsilon, delta_mu) pairs")

    active_combos = collect_active_combo_keys(
        MANAGE_CSV, MANIFEST_PATH, OUTPUT_DIR, RESULTS_DIR,
    )
    existing_rows = read_manage(MANAGE_CSV)
    existing_keys = {combo_key_from_dict(row) for row in existing_rows}

    pending_paths: list[str] = []
    new_manage_rows: list[dict] = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    n_files = 0
    skipped_flex = 0
    skipped_dedup = 0

    for epsilon, delta_mu in outer_pairs:
        combo = combo_dict(epsilon, delta_mu)
        key = combo_key_from_dict(combo)

        if key in active_combos:
            print(f"[skip dedup:{active_combos[key]}] eps={epsilon}, dmu={delta_mu}")
            skipped_dedup += 1
            continue

        try:
            mu_coex_flex = coex_chemical_potential(
                epsilon=epsilon,
                df=DELTA_F,
                dmu=delta_mu,
                chem_rec_baserate=K,
                DRIVEN=True,
                scheme=FLEX_INDEX,
            )
            mu_coex_flex = float(np.asarray(mu_coex_flex).ravel()[0])
        except Exception as exc:
            print(f"[skip flex] eps={epsilon}, dmu={delta_mu}: {exc}")
            skipped_flex += 1
            continue

        if mu_coex_flex > 0:
            print(f"[skip flex] eps={epsilon}, dmu={delta_mu}: mu_coex_FLEX={mu_coex_flex:.6f} > 0")
            skipped_flex += 1
            continue

        print(f"eps={epsilon}, dmu={delta_mu}: mu_coex_FLEX={mu_coex_flex:.6f}")
        mu_values = mu_sweep(mu_coex_flex)

        if key not in existing_keys:
            new_manage_rows.append({
                "epsilon": epsilon,
                "delta_f": DELTA_F,
                "delta_mu": delta_mu,
                "k": K,
                "scheme": SCHEME,
                "Lx": LX,
                "Ly": LY,
                "mu_coex_FLEX": mu_coex_flex,
                "isSubmitted": timestamp,
                "isRan": "",
                "isAnalyzed": "",
                "mu_coex_SIM": "",
                "mu_coex_SIM_error": "",
                "RequestForAdditionalData": 0,
                "combo_path": combo_dir(combo),
            })
            existing_keys.add(key)

        outer_tag = f"{eps_filename_tag(epsilon)}_{dmu_filename_tag(delta_mu)}"
        for idx, mu in enumerate(mu_values):
            job = {
                **combo,
                "mu": mu,
                "mu_coex_FLEX": mu_coex_flex,
                "run_settings": RUN_SETTINGS,
            }
            filename = f"{SCHEME}_{outer_tag}_Ly{LY}_mu{idx:02d}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "w") as f:
                json.dump(job, f, indent=2)
            pending_paths.append(filepath)
            n_files += 1

    merge_pending(pending_paths, path=MANIFEST_PATH)
    write_manage(MANAGE_CSV, existing_rows + new_manage_rows)

    print(f"\nWrote {n_files} JSON files to '{OUTPUT_DIR}/'")
    print(f"Merged {len(pending_paths)} jobs into '{MANIFEST_PATH}'")
    print(f"Added {len(new_manage_rows)} new rows to '{MANAGE_CSV}' "
          f"({len(existing_rows)} existing)")
    print(f"Skipped {skipped_dedup} dedup, {skipped_flex} FLEX filter")


if __name__ == "__main__":
    main()
