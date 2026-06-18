#!/usr/bin/env python3
"""
Re-enqueue missing initial mu jobs for manage.csv rows that never finished.

Targets rows without isAnalyzed where fewer than N_MU_POINTS result dirs exist.
Recreates JSON under samples/ and merge_pending into run_all_queue.json.

Usage (on Della, from project root):
    python scripts/requeue_incomplete.py --dry-run
    python scripts/requeue_incomplete.py
    python scripts/requeue_incomplete.py --reset-ran   # clear isRan when 0 results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_samples import (
    COMBO_KEY_FIELDS,
    MANAGE_FIELDS,
    MANIFEST_PATH,
    N_MU_POINTS,
    OUTPUT_DIR,
    RESULTS_DIR,
    RUN_SETTINGS,
    dmu_filename_tag,
    eps_filename_tag,
    mu_sweep,
    read_manage,
    write_manage,
)
from combo_paths import combo_dir, mu_dir, combo_has_results, legacy_combo_dir_names, mu_dir_name
from queue_manifest import merge_pending, read_manifest

MANAGE_CSV = "manage.csv"


def combo_from_row(row: dict) -> dict:
    return {f: row[f] for f in COMBO_KEY_FIELDS}


def count_completed_mus(row: dict) -> tuple[int, set[float]]:
    combo = combo_from_row(row)
    mu_coex_flex = float(row["mu_coex_FLEX"])
    completed: set[float] = set()
    for mu in mu_sweep(mu_coex_flex):
        params = {**combo, "mu": mu}
        csv_path = os.path.join(mu_dir(params), "output.csv")
        if os.path.isfile(csv_path):
            completed.add(mu)
            continue
        flat_csv = os.path.join(combo_dir(combo), mu_dir_name(mu), "output.csv")
        if os.path.isfile(flat_csv):
            completed.add(mu)
            continue
        for legacy in legacy_combo_dir_names(combo):
            legacy_csv = os.path.join(
                RESULTS_DIR, legacy, mu_dir_name(mu), "output.csv",
            )
            if os.path.isfile(legacy_csv):
                completed.add(mu)
                break
    return len(completed), completed


def queued_combo_mus(manifest_path: str) -> dict[tuple, set[float]]:
    """Map combo_key -> mu values already pending or in flight."""
    manifest = read_manifest(manifest_path)
    paths = manifest.get("pending", []) + list(manifest.get("in_flight", {}).values())
    by_combo: dict[tuple, set[float]] = {}
    for json_path in paths:
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path) as f:
                job = json.load(f)
            key = tuple(str(job[f]) for f in COMBO_KEY_FIELDS)
            by_combo.setdefault(key, set()).add(round(float(job["mu"]), 6))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return by_combo


def make_job_json(combo: dict, mu: float, mu_coex_flex: float, samples_dir: str) -> str:
    epsilon = float(combo["epsilon"])
    delta_mu = float(combo["delta_mu"])
    ly = int(combo["Ly"])
    scheme = str(combo["scheme"])
    outer_tag = f"{eps_filename_tag(epsilon)}_{dmu_filename_tag(delta_mu)}"
    mu_values = mu_sweep(mu_coex_flex)
    try:
        idx = mu_values.index(round(float(mu), 6))
    except ValueError:
        idx = min(
            range(len(mu_values)),
            key=lambda i: abs(mu_values[i] - float(mu)),
        )
    filename = f"{scheme}_{outer_tag}_Ly{ly}_mu{idx:02d}.json"
    filepath = os.path.join(samples_dir, filename)
    job = {
        **{f: combo[f] for f in COMBO_KEY_FIELDS},
        "mu": round(float(mu), 6),
        "mu_coex_FLEX": mu_coex_flex,
        "run_settings": RUN_SETTINGS,
    }
    os.makedirs(samples_dir, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(job, f, indent=2)
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Re-enqueue missing mu jobs for incomplete manage.csv rows",
    )
    parser.add_argument("--manage", default=MANAGE_CSV)
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--samples", default=OUTPUT_DIR)
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reset-ran",
        action="store_true",
        help="Clear isRan on rows with zero completed mu results",
    )
    args = parser.parse_args()

    rows = read_manage(args.manage)
    queued = queued_combo_mus(args.manifest)
    pending_paths: list[str] = []
    touched_rows = 0

    for row in rows:
        if str(row.get("isAnalyzed", "")).strip():
            continue

        combo = combo_from_row(row)
        key = tuple(str(combo[f]) for f in COMBO_KEY_FIELDS)
        n_done, done_mus = count_completed_mus(row)
        mu_coex_flex = float(row["mu_coex_FLEX"])
        expected = mu_sweep(mu_coex_flex)
        already_queued = queued.get(key, set())

        missing = [
            mu for mu in expected
            if mu not in done_mus and mu not in already_queued
        ]

        if n_done >= N_MU_POINTS and not missing:
            continue

        eps = combo["epsilon"]
        dmu = combo["delta_mu"]
        print(
            f"eps={eps} dmu={dmu}: {n_done}/{N_MU_POINTS} on disk, "
            f"{len(already_queued)} queued, {len(missing)} to enqueue"
        )
        touched_rows += 1

        if args.reset_ran and n_done == 0 and str(row.get("isRan", "")).strip():
            print(f"  reset isRan (was {row['isRan']!r}, but 0 results)")
            if not args.dry_run:
                row["isRan"] = ""
                row["combo_path"] = combo_dir(combo)

        for mu in missing:
            path = make_job_json(combo, mu, mu_coex_flex, args.samples)
            print(f"  + {path}  mu={mu:.6f}")
            pending_paths.append(path)

        if not args.dry_run and not row.get("combo_path"):
            row["combo_path"] = combo_dir(combo)

    if not pending_paths:
        print("Nothing to enqueue.")
        return

    print(f"\n{'Would enqueue' if args.dry_run else 'Enqueuing'} {len(pending_paths)} jobs "
          f"for {touched_rows} combo(s)")

    if args.dry_run:
        return

    merge_pending(pending_paths, path=args.manifest)
    if args.reset_ran or any(not r.get("combo_path") for r in rows):
        write_manage(args.manage, rows)
    print(f"Merged into {args.manifest}. Restart run_all if it exited.")


if __name__ == "__main__":
    main()
