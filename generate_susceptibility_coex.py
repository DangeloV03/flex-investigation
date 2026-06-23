"""
generate_susceptibility_coex.py

Generate slab coexistence μ-sweep jobs for the Ising-limit susceptibility study.
Uses the same json_runner + analyzer workflow as the main campaign (slab IC,
Lx = 10 * Ly). μ_coex_SIM from this phase is reused for all square-L prod runs
at the same ε.

Usage:
    python generate_susceptibility_coex.py
    python generate_susceptibility_coex.py --eps-step 0.05 --ly 16
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from combo_paths import COMBO_KEY_FIELDS, combo_key_from_dict
from flex_coex_chemical_potential_prediction import coex_chemical_potential
from generate_samples import append_manage_rows, collect_active_combo_keys, frange, mu_sweep
from queue_manifest import merge_pending
from susceptibility_paths import (
    COEX_MANIFEST,
    COEX_RESULTS_DIR,
    COEX_SAMPLES_DIR,
    ISING_DELTA_F,
    ISING_DELTA_MU,
    ISING_K,
    ISING_SCHEME,
    MANAGE_CSV,
    coex_combo_dir,
    coex_job_filename,
)

FLEX_INDEX = 1
LX_MULTIPLIER = 10
DEFAULT_COEX_LY = 16

EPS_MIN = -2.0
EPS_MAX = -1.4
EPS_STEP = 0.05

MU_WINDOW = 0.1
N_MU_POINTS = 10

RUN_SETTINGS = {
    "beta": 1.0,
    "k": ISING_K,
    "initial_condition": "slab_half_active_half_empty",
    "num_parallel_runs": 4,
    "eq_time": 10000.0,
    "prod_time": 10000.0,
    "seed_base": 2000,
}

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


def combo_dict(epsilon: float, ly: int, lx: int) -> dict:
    return {
        "epsilon": epsilon,
        "delta_f": ISING_DELTA_F,
        "delta_mu": ISING_DELTA_MU,
        "k": ISING_K,
        "scheme": ISING_SCHEME,
        "Lx": lx,
        "Ly": ly,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate susceptibility coex μ-sweep JSONs")
    parser.add_argument("--eps-min", type=float, default=EPS_MIN)
    parser.add_argument("--eps-max", type=float, default=EPS_MAX)
    parser.add_argument("--eps-step", type=float, default=EPS_STEP)
    parser.add_argument("--ly", type=int, default=DEFAULT_COEX_LY, help="Slab Ly (Lx = 10*Ly)")
    parser.add_argument("--samples-dir", default=COEX_SAMPLES_DIR)
    parser.add_argument("--manage", default=MANAGE_CSV)
    parser.add_argument("--manifest", default=COEX_MANIFEST)
    parser.add_argument("--results-dir", default=COEX_RESULTS_DIR)
    args = parser.parse_args()

    ly = args.ly
    lx = LX_MULTIPLIER * ly
    eps_values = frange(args.eps_min, args.eps_max, args.eps_step)

    os.makedirs(args.samples_dir, exist_ok=True)

    active_combos = collect_active_combo_keys(
        args.manage, args.manifest, args.samples_dir, args.results_dir,
    )

    pending_paths: list[str] = []
    new_manage_rows: list[dict] = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    n_files = 0
    n_existing_json = 0
    skipped_flex = 0
    skipped_dedup = 0

    existing_keys = set()
    if os.path.isfile(args.manage):
        with open(args.manage, newline="") as f:
            for row in csv.DictReader(f):
                existing_keys.add(combo_key_from_dict(row))

    print(
        f"Susceptibility coex: epsilon [{args.eps_min}, {args.eps_max}] "
        f"step {args.eps_step} ({len(eps_values)} pts), Ly={ly}, Lx={lx}"
    )

    for epsilon in eps_values:
        combo = combo_dict(epsilon, ly, lx)
        key = combo_key_from_dict(combo)

        if key in active_combos:
            print(f"[skip dedup:{active_combos[key]}] eps={epsilon}")
            skipped_dedup += 1
            continue

        try:
            mu_coex_flex = coex_chemical_potential(
                epsilon=epsilon,
                df=ISING_DELTA_F,
                dmu=ISING_DELTA_MU,
                chem_rec_baserate=ISING_K,
                DRIVEN=True,
                scheme=FLEX_INDEX,
            )
            mu_coex_flex = float(np.asarray(mu_coex_flex).ravel()[0])
        except Exception as exc:
            print(f"[skip flex] eps={epsilon}: {exc}")
            skipped_flex += 1
            continue

        if mu_coex_flex > 0:
            print(f"[skip flex] eps={epsilon}: mu_coex_FLEX={mu_coex_flex:.6f} > 0")
            skipped_flex += 1
            continue

        print(f"eps={epsilon}: mu_coex_FLEX={mu_coex_flex:.6f}")
        mu_values = mu_sweep(mu_coex_flex)

        if key not in existing_keys:
            new_manage_rows.append({
                **combo,
                "mu_coex_FLEX": mu_coex_flex,
                "isSubmitted": timestamp,
                "isRan": "",
                "isAnalyzed": "",
                "mu_coex_SIM": "",
                "mu_coex_SIM_error": "",
                "RequestForAdditionalData": 0,
                "combo_path": coex_combo_dir(combo),
            })
            existing_keys.add(key)

        for idx, mu in enumerate(mu_values):
            job = {
                **combo,
                "mu": mu,
                "mu_coex_FLEX": mu_coex_flex,
                "run_settings": dict(RUN_SETTINGS),
                "results_base": args.results_dir,
                "manage_csv": args.manage,
            }
            filename = coex_job_filename(ISING_SCHEME, epsilon, ISING_DELTA_MU, ly, idx)
            filepath = os.path.join(args.samples_dir, filename)
            if os.path.isfile(filepath):
                n_existing_json += 1
            else:
                with open(filepath, "w") as f:
                    json.dump(job, f, indent=2)
                n_files += 1
            pending_paths.append(filepath)

    merge_pending(pending_paths, path=args.manifest)
    n_added = append_manage_rows(args.manage, new_manage_rows)

    print(f"\nWrote {n_files} new JSON files to '{args.samples_dir}/' "
          f"({n_existing_json} already existed)")
    print(f"Queued {len(pending_paths)} path(s) into '{args.manifest}'")
    print(f"Added {n_added} new rows to '{args.manage}'")
    print(f"Skipped {skipped_dedup} dedup, {skipped_flex} FLEX filter")


if __name__ == "__main__":
    main()
