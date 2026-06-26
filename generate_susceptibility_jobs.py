"""
generate_susceptibility_jobs.py

Create square-L production JSON jobs from susceptibility_manage.csv (slab coex rows).
Each job runs at mu_coex_SIM for one (epsilon, L) pair.

Usage:
    python generate_susceptibility_jobs.py
    python generate_susceptibility_jobs.py --L 16 32 64
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from generate_samples import frange
from queue_manifest import merge_pending
from susceptibility_paths import (
    COEX_LOOKUP_FIELDS,
    ISING_DELTA_F,
    ISING_DELTA_MU,
    ISING_K,
    ISING_SCHEME,
    MANAGE_CSV,
    PROD_MANIFEST,
    PROD_SAMPLES_DIR,
    SQUARE_L_VALUES,
    prod_job_filename,
)

EPS_MIN = -2.0
EPS_MAX = -1.4
EPS_STEP = 0.01

DEFAULT_RUN_SETTINGS = {
    "beta": 1.0,
    "initial_fraction": 0.8,
    "num_parallel_runs": 8,
    "eq_time": 100000.0,
    "prod_time": 100000.0,
    "prod_chunks": 1000,
    "seed_base": 5000,
}


def lookup_key(row: dict) -> tuple[str, ...]:
    return tuple(str(row[f]) for f in COEX_LOOKUP_FIELDS)


def read_coex_rows(manage_path: str) -> dict[tuple[str, ...], dict]:
    """Map (epsilon, delta_f, ...) -> manage row with mu_coex_SIM."""
    if not os.path.isfile(manage_path):
        return {}

    by_key: dict[tuple[str, ...], dict] = {}
    with open(manage_path, newline="") as f:
        for row in csv.DictReader(f):
            sim = str(row.get("mu_coex_FITTED", "")).strip()
            if not sim or sim.lower() == "nan":
                continue
            by_key[lookup_key(row)] = row
    return by_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate susceptibility production JSON jobs")
    parser.add_argument("--manage", default=MANAGE_CSV)
    parser.add_argument("--samples-dir", default=PROD_SAMPLES_DIR)
    parser.add_argument("--manifest", default=PROD_MANIFEST)
    parser.add_argument("--eps-min", type=float, default=EPS_MIN)
    parser.add_argument("--eps-max", type=float, default=EPS_MAX)
    parser.add_argument("--eps-step", type=float, default=EPS_STEP)
    parser.add_argument(
        "--L",
        type=int,
        nargs="+",
        default=SQUARE_L_VALUES,
        help="Square lattice sizes (Lx = Ly = L)",
    )
    parser.add_argument("--eq-time", type=float, default=None)
    parser.add_argument("--prod-time", type=float, default=None)
    parser.add_argument("--num-parallel-runs", type=int, default=None)
    args = parser.parse_args()

    coex_rows = read_coex_rows(args.manage)
    if not coex_rows:
        print(f"No rows with numeric mu_coex_SIM in '{args.manage}'. Run coex phase first.")
        return

    os.makedirs(args.samples_dir, exist_ok=True)
    eps_values = frange(args.eps_min, args.eps_max, args.eps_step)
    l_values = sorted(set(args.L))

    pending_paths: list[str] = []
    n_files = 0
    n_existing = 0
    n_skipped = 0

    for epsilon in eps_values:
        lookup = lookup_key({
            "epsilon": epsilon,
            "delta_f": ISING_DELTA_F,
            "delta_mu": ISING_DELTA_MU,
            "k": ISING_K,
            "scheme": ISING_SCHEME,
        })
        row = coex_rows.get(lookup)
        if row is None:
            print(f"[skip] no mu_coex_SIM for eps={epsilon}")
            n_skipped += 1
            continue

        mu_coex_sim = float(row["mu_coex_FITTED"])

        for l_val in l_values:
            run_settings = dict(DEFAULT_RUN_SETTINGS)
            if args.eq_time is not None:
                run_settings["eq_time"] = args.eq_time
            if args.prod_time is not None:
                run_settings["prod_time"] = args.prod_time
            if args.num_parallel_runs is not None:
                run_settings["num_parallel_runs"] = args.num_parallel_runs

            job = {
                "epsilon": epsilon,
                "delta_f": ISING_DELTA_F,
                "delta_mu": ISING_DELTA_MU,
                "k": ISING_K,
                "scheme": ISING_SCHEME,
                "Lx": l_val,
                "Ly": l_val,
                "mu": mu_coex_sim,
                "mu_coex_FITTED": mu_coex_sim,
                "run_settings": run_settings,
            }

            filename = prod_job_filename(ISING_SCHEME, epsilon, ISING_DELTA_MU, l_val)
            filepath = os.path.join(args.samples_dir, filename)
            if os.path.isfile(filepath):
                n_existing += 1
            else:
                with open(filepath, "w") as f:
                    json.dump(job, f, indent=2)
                n_files += 1
            pending_paths.append(filepath)

    merge_pending(pending_paths, path=args.manifest)

    print(f"Wrote {n_files} new JSON files to '{args.samples_dir}/' ({n_existing} existed)")
    print(f"Queued {len(pending_paths)} path(s) into '{args.manifest}'")
    print(f"Skipped {n_skipped} epsilon values without mu_coex_SIM")


if __name__ == "__main__":
    main()
