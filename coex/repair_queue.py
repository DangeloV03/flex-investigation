#!/usr/bin/env python3
"""
Repair run_all_queue.json after failures: restore missing JSON from samples/done/
and clear stale in_flight entries for jobs no longer in Slurm.

Usage (on Della login node, from project root):
    python coex/repair_queue.py --dry-run
    python coex/repair_queue.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from queue_manifest import (
    DONE_DIR,
    MANIFEST_PATH,
    archive_json,
    ensure_job_json,
    locked_manifest,
    read_manifest,
    restore_json_from_done,
)

SUCCESS_STATES = {"COMPLETED", "COMPLETING"}
FAILURE_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
}


def normalize_sacct_state(state: str | None) -> str | None:
    if not state:
        return None
    return state.split()[0].split("|")[0].rstrip("+")


def sacct_state(job_id: str) -> str | None:
    try:
        result = subprocess.run(
            ["sacct", "-n", "-X", "-j", job_id, "--format=State"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    return normalize_sacct_state(lines[0] if lines else None)


def active_slurm_ids() -> set[str]:
    try:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%i"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return set()
    if result.returncode != 0:
        return set()
    return {ln.strip() for ln in result.stdout.splitlines() if ln.strip()}


def restore_missing_queue_json(
    manifest_path: str,
    *,
    dry_run: bool,
) -> tuple[int, int]:
    manifest = read_manifest(manifest_path)
    paths = set(manifest.get("pending", [])) | set(manifest.get("in_flight", {}).values())
    restored = 0
    still_missing = 0
    for path in sorted(paths):
        if os.path.isfile(path):
            continue
        src = os.path.join(DONE_DIR, os.path.basename(path))
        if os.path.isfile(src):
            print(f"restore: {path} <- {src}")
            if not dry_run:
                restore_json_from_done(path)
            restored += 1
        else:
            print(f"missing (no done/ copy): {path}")
            still_missing += 1
    return restored, still_missing


def reconcile_stale_in_flight(
    manifest_path: str,
    *,
    dry_run: bool,
) -> tuple[int, int, int]:
    manifest = read_manifest(manifest_path)
    in_flight = dict(manifest.get("in_flight", {}))
    active = active_slurm_ids()
    cleared = 0
    requeued = 0
    waiting = 0

    for job_id, json_path in sorted(in_flight.items()):
        if job_id in active:
            waiting += 1
            continue

        state = sacct_state(job_id)
        if state is None:
            print(f"keep in_flight (sacct unknown): job {job_id} -> {json_path}")
            waiting += 1
            continue

        if state in SUCCESS_STATES:
            action = "archive path (job succeeded)"
        elif state in FAILURE_STATES:
            action = "re-queue path (job failed)"
            requeued += 1
        else:
            action = f"re-queue path (state={state!r})"
            requeued += 1

        print(f"clear in_flight: job {job_id} state={state} -> {json_path} ({action})")
        if not dry_run:
            with locked_manifest(manifest_path) as m:
                m["in_flight"].pop(str(job_id), None)
            if state in SUCCESS_STATES:
                archive_json(json_path)
            else:
                ensure_job_json(json_path)
                with locked_manifest(manifest_path) as m:
                    if json_path not in m["pending"]:
                        m["pending"].insert(0, json_path)
        cleared += 1

    return cleared, requeued, waiting


def main():
    parser = argparse.ArgumentParser(description="Repair queue manifest and sample JSON")
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"=== restore missing JSON ({args.manifest}) ===")
    restored, still_missing = restore_missing_queue_json(
        args.manifest, dry_run=args.dry_run,
    )
    print(f"restored={restored} still_missing={still_missing}")

    print("\n=== reconcile stale in_flight ===")
    cleared, requeued, waiting = reconcile_stale_in_flight(
        args.manifest, dry_run=args.dry_run,
    )
    print(f"cleared={cleared} requeued={requeued} still_in_flight={waiting}")

    if args.dry_run:
        print("\n(dry run — no files or manifest changed)")


if __name__ == "__main__":
    main()
