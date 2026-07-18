#!/usr/bin/env python3
"""
Reset specific epsilon combos and re-enqueue their initial mu-sweep jobs.

For each target ε:
  1. Clear manage.csv analysis/run fields
  2. Delete results/ for that combo (canonical + legacy layouts)
  3. Remove sample JSONs (samples/, samples/done/, samples/staging/)
  4. Drop matching paths from the queue manifest
  5. Write fresh initial N_MU_POINTS job JSONs and prepend to pending

Stop the dispatcher/analyzer tmux session first, or run immediately before
restarting them so they do not re-mark rows while you reset.

Usage (homo Δμ=1 campaign on the workstation):
    python coex/rerun_combos.py --dry-run
    python coex/rerun_combos.py

    # or use the wrapper:
    ./coex/rerun_critical_homo_dmu1p0.sh --dry-run
    ./coex/rerun_critical_homo_dmu1p0.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combo_paths import (
    COMBO_KEY_FIELDS,
    combo_dir,
    combo_key_from_dict,
    legacy_combo_dir_names,
)
from generate_samples import N_MU_POINTS, mu_sweep
from generate_susceptibility_coex import RUN_SETTINGS
from queue_manifest import locked_manifest, prepend_pending
from susceptibility_paths import coex_job_filename

# Critical-region ε values that failed (NaN fit or bad μ_coex_FITTED).
DEFAULT_EPSILONS = [-1.775, -1.780, -1.785, -1.790, -1.800, -1.765]

PROJECT_DONE_DIR = "samples/done"
PROJECT_STAGING_DIR = "samples/staging"

MANAGE_FIELDS = COMBO_KEY_FIELDS + [
    "mu_coex_FLEX",
    "isSubmitted",
    "isRan",
    "isAnalyzed",
    "mu_coex_FITTED",
    "mu_coex_FITTED_error",
    "RequestForAdditionalData",
    "combo_path",
]


def read_manage(manage_path: str) -> list[dict]:
    if not os.path.isfile(manage_path):
        return []
    with open(manage_path, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    rows: list[dict] = []
    for line_no, row in enumerate(raw_rows, start=2):
        row = {str(k).lstrip("\ufeff"): v for k, v in row.items()}
        row.setdefault("mu_coex_FITTED_error", "")
        row.setdefault("combo_path", "")
        missing = [f for f in COMBO_KEY_FIELDS if f not in row or row[f] is None]
        if missing:
            print(f"WARNING: skipping malformed manage.csv line {line_no} (missing {missing})")
            continue
        rows.append(row)
    return rows


def write_manage(manage_path: str, rows: list[dict]) -> None:
    with open(manage_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANAGE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANAGE_FIELDS})


def combo_from_row(row: dict) -> dict:
    combo = {f: row[f] for f in COMBO_KEY_FIELDS}
    combo["epsilon"] = float(combo["epsilon"])
    combo["delta_f"] = float(combo["delta_f"])
    combo["delta_mu"] = float(combo["delta_mu"])
    combo["k"] = float(combo["k"])
    combo["Lx"] = int(float(combo["Lx"]))
    combo["Ly"] = int(float(combo["Ly"]))
    return combo


def find_row_by_epsilon(rows: list[dict], epsilon: float) -> dict | None:
    for row in rows:
        if abs(float(row["epsilon"]) - epsilon) < 1e-9:
            return row
    return None


def reset_manage_row(row: dict, results_dir: str, combo: dict) -> None:
    row["isRan"] = ""
    row["isAnalyzed"] = ""
    row["mu_coex_FITTED"] = ""
    row["mu_coex_FITTED_error"] = ""
    row["RequestForAdditionalData"] = "0"
    row["combo_path"] = combo_dir(combo, base=results_dir)


def combo_key_from_json_path(json_path: str) -> tuple[str, ...] | None:
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            job = json.load(f)
        if not all(f in job for f in COMBO_KEY_FIELDS):
            return None
        return combo_key_from_dict(job)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def iter_json_dirs(samples_dir: str) -> list[str]:
    dirs = [samples_dir, os.path.join(samples_dir, "done"), PROJECT_DONE_DIR, PROJECT_STAGING_DIR]
    seen: set[str] = set()
    out: list[str] = []
    for d in dirs:
        abspath = os.path.abspath(d)
        if abspath in seen:
            continue
        seen.add(abspath)
        if os.path.isdir(d):
            out.append(d)
    return out


def remove_combo_jsons(
    combo_key: tuple[str, ...],
    samples_dir: str,
    *,
    dry_run: bool,
) -> list[str]:
    removed: list[str] = []
    for directory in iter_json_dirs(samples_dir):
        for name in os.listdir(directory):
            if not name.endswith(".json"):
                continue
            path = os.path.join(directory, name)
            key = combo_key_from_json_path(path)
            if key != combo_key:
                continue
            removed.append(path)
            if dry_run:
                print(f"  would remove JSON: {path}")
            else:
                os.remove(path)
    return removed


def remove_combo_results(combo: dict, results_dir: str, *, dry_run: bool) -> list[str]:
    removed: list[str] = []
    candidates = [combo_dir(combo, base=results_dir)]
    candidates.extend(os.path.join(results_dir, legacy) for legacy in legacy_combo_dir_names(combo))
    seen: set[str] = set()
    for path in candidates:
        abspath = os.path.abspath(path)
        if abspath in seen:
            continue
        seen.add(abspath)
        if not os.path.isdir(path):
            continue
        removed.append(path)
        if dry_run:
            print(f"  would remove results: {path}")
        else:
            shutil.rmtree(path)
    return removed


def purge_manifest(combo_key: tuple[str, ...], manifest_path: str, *, dry_run: bool) -> tuple[int, int]:
    """Remove pending/in_flight entries for combo_key. Returns (pending_removed, in_flight_removed)."""
    if dry_run:
        manifest = json.load(open(manifest_path, encoding="utf-8"))
        pending_removed = sum(
            1 for p in manifest.get("pending", []) if combo_key_from_json_path(p) == combo_key
        )
        in_flight_removed = sum(
            1 for p in manifest.get("in_flight", {}).values() if combo_key_from_json_path(p) == combo_key
        )
        return pending_removed, in_flight_removed

    with locked_manifest(manifest_path) as manifest:
        pending_before = len(manifest["pending"])
        manifest["pending"] = [
            p for p in manifest["pending"]
            if combo_key_from_json_path(p) != combo_key
        ]
        pending_removed = pending_before - len(manifest["pending"])

        in_flight_before = len(manifest["in_flight"])
        manifest["in_flight"] = {
            job_id: path
            for job_id, path in manifest["in_flight"].items()
            if combo_key_from_json_path(path) != combo_key
        }
        in_flight_removed = in_flight_before - len(manifest["in_flight"])

    return pending_removed, in_flight_removed


def make_initial_jobs(
    combo: dict,
    row: dict,
    samples_dir: str,
    results_dir: str,
    manage_path: str,
    run_settings: dict,
    *,
    dry_run: bool,
) -> list[str]:
    mu_coex_flex = float(row["mu_coex_FLEX"])
    mu_values = mu_sweep(mu_coex_flex)
    if len(mu_values) != N_MU_POINTS:
        raise RuntimeError(f"expected {N_MU_POINTS} mu points, got {len(mu_values)}")

    paths: list[str] = []
    scheme = str(combo["scheme"])
    epsilon = float(combo["epsilon"])
    delta_mu = float(combo["delta_mu"])
    ly = int(combo["Ly"])

    settings = dict(run_settings)
    settings["k"] = float(combo["k"])

    for idx, mu in enumerate(mu_values):
        filename = coex_job_filename(scheme, epsilon, delta_mu, ly, idx)
        filepath = os.path.join(samples_dir, filename)
        job = {
            **{f: combo[f] for f in COMBO_KEY_FIELDS},
            "mu": mu,
            "mu_coex_FLEX": mu_coex_flex,
            "run_settings": dict(settings),
            "results_base": results_dir,
            "manage_csv": manage_path,
        }
        paths.append(filepath)
        if dry_run:
            print(f"  would write {filepath}  mu={mu:.6f}")
            continue
        os.makedirs(samples_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2)
            f.write("\n")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset and re-enqueue coex combos at specific epsilon values",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=DEFAULT_EPSILONS,
        help=f"Target ε values (default: {DEFAULT_EPSILONS})",
    )
    parser.add_argument("--manage", default="coex_manage_homo_dmu1p0.csv")
    parser.add_argument("--manifest", default="coex_homo_dmu1p0_queue.json")
    parser.add_argument("--samples", default="coex_samples/homo_dmu1p0")
    parser.add_argument("--results", default="susceptibility_results/coex_homo_dmu1p0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_manage(args.manage)
    if not rows:
        raise SystemExit(f"No rows in {args.manage}")

    eps_targets = [round(float(e), 6) for e in args.epsilon]
    all_job_paths: list[str] = []

    print(f"Target ε values: {eps_targets}")
    print(f"  manage={args.manage}")
    print(f"  manifest={args.manifest}")
    print(f"  samples={args.samples}")
    print(f"  results={args.results}")
    if args.dry_run:
        print("(dry run — no files changed)\n")

    for epsilon in eps_targets:
        row = find_row_by_epsilon(rows, epsilon)
        if row is None:
            print(f"\n[skip] ε={epsilon}: no row in {args.manage}")
            continue

        combo = combo_from_row(row)
        combo_key = combo_key_from_dict(combo)
        tag = combo_dir(combo, base=args.results).split(os.sep)[-1]

        print(f"\n=== ε={epsilon} ({tag}) ===")
        print(
            f"  was: isAnalyzed={row.get('isAnalyzed', '')!r}  "
            f"mu_coex_FITTED={row.get('mu_coex_FITTED', '')!r}  "
            f"requests={row.get('RequestForAdditionalData', '')!r}"
        )

        n_results = len(remove_combo_results(combo, args.results, dry_run=args.dry_run))
        print(f"  results dirs: {n_results}")

        json_removed = remove_combo_jsons(combo_key, args.samples, dry_run=args.dry_run)
        print(f"  sample JSONs removed: {len(json_removed)}")

        if os.path.isfile(args.manifest):
            pending_rm, inflight_rm = purge_manifest(combo_key, args.manifest, dry_run=args.dry_run)
            print(f"  queue: removed {pending_rm} pending, {inflight_rm} in_flight")
            if inflight_rm and not args.dry_run:
                print(
                    "  WARNING: cleared in_flight entries — restart run_all.py if jobs were running",
                )
        else:
            print(f"  queue: manifest {args.manifest} not found (will create on enqueue)")

        if not args.dry_run:
            reset_manage_row(row, args.results, combo)

        job_paths = make_initial_jobs(
            combo,
            row,
            args.samples,
            args.results,
            args.manage,
            RUN_SETTINGS,
            dry_run=args.dry_run,
        )
        print(f"  initial jobs: {len(job_paths)} x {N_MU_POINTS} mu points")
        all_job_paths.extend(job_paths)

    if args.dry_run:
        print(f"\nWould enqueue {len(all_job_paths)} job(s) total")
        return 0

    write_manage(args.manage, rows)

    if all_job_paths:
        n_added = prepend_pending(all_job_paths, path=args.manifest)
        print(f"\nPrepended {n_added} new job(s) to {args.manifest}")
    else:
        print("\nNo jobs enqueued.")

    print(
        "\nNext: ensure dispatcher + analyzer are running, e.g.\n"
        "  tmux attach -t coex-dmu1p0\n"
        "or restart with run_all.py --local --manifest ... and analyzer.py ...",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
