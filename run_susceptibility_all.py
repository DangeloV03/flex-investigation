"""
run_susceptibility_all.py

Slurm/local dispatcher for susceptibility jobs (coex or prod phase).

Coex phase uses json_runner.py (slab geometry); prod phase uses susceptibility_runner.py.

Usage:
    # Coexistence μ sweeps (slab)
    python run_susceptibility_all.py --phase coex --local --once

    # Square-L susceptibility production
    python run_susceptibility_all.py --phase prod --local --once
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import queue_manifest as qm
from queue_manifest import (
    ensure_job_json,
    mark_in_flight,
    pop_next_pending,
    read_manifest,
    remove_in_flight,
    requeue_front,
    stage_job_json,
)
from run_all import (
    FAILURE_STATES,
    POLL_INTERVAL,
    SLURM_CONFIG,
    SUCCESS_STATES,
    active_slurm_ids,
    build_slurm,
    finish_job,
    sacct_state,
    slurm_available,
    walltime_for_json,
)
from susceptibility_paths import (
    COEX_MANIFEST,
    COEX_SAMPLES_DIR,
    PROD_MANIFEST,
    PROD_SAMPLES_DIR,
    patch_coex_job_json,
    patch_prod_job_json,
)

MAX_CONCURRENT = 100

PHASE_CONFIG = {
    "coex": {
        "runner": "json_runner.py",
        "manifest": COEX_MANIFEST,
        "samples_root": COEX_SAMPLES_DIR,
    },
    "prod": {
        "runner": "susceptibility_runner.py",
        "manifest": PROD_MANIFEST,
        "samples_root": PROD_SAMPLES_DIR,
    },
}


def submit_slurm_job(
    json_path: str,
    python: str,
    runner: str,
    config_path: str = SLURM_CONFIG,
) -> str:
    walltime = walltime_for_json(json_path, config_path)
    slurm = build_slurm(config_path, time=walltime)
    abs_json = os.path.abspath(json_path)
    job_id = slurm.sbatch(f"{python} -u {runner} {abs_json}")
    return str(job_id)


def reconcile_in_flight(
    *,
    manifest: str,
    done_dir: str,
    slurm=None,
    local_jobs: dict[str, subprocess.Popen] | None = None,
) -> None:
    manifest_data = read_manifest(manifest)
    in_flight = dict(manifest_data.get("in_flight", {}))
    qm.DONE_DIR = done_dir

    if local_jobs is not None:
        for job_id, proc in list(local_jobs.items()):
            ret = proc.poll()
            if ret is None:
                continue
            json_path = in_flight.get(job_id)
            if json_path:
                finish_job(json_path, ret == 0)
                remove_in_flight(job_id, path=manifest)
            local_jobs.pop(job_id, None)
        return

    active_ids = active_slurm_ids(slurm)
    for job_id, json_path in list(in_flight.items()):
        if job_id in active_ids:
            continue
        state = sacct_state(job_id)
        if state is None:
            print(
                f"[run_susceptibility_all] sacct unknown for job {job_id}, "
                f"leaving in_flight: {json_path}"
            )
            continue
        success = state in SUCCESS_STATES
        if state not in SUCCESS_STATES and state not in FAILURE_STATES:
            success = False
        finish_job(json_path, success)
        remove_in_flight(job_id, path=manifest)


def submit_up_to_cap(
    *,
    manifest: str,
    done_dir: str,
    staging_dir: str,
    runner: str,
    python: str,
    phase: str,
    slurm=None,
    local_jobs: dict[str, subprocess.Popen] | None = None,
    max_concurrent: int,
    config_path: str = SLURM_CONFIG,
) -> int:
    submitted = 0
    qm.DONE_DIR = done_dir
    qm.STAGING_DIR = staging_dir

    while True:
        manifest_data = read_manifest(manifest)
        in_flight_count = len(manifest_data.get("in_flight", {}))
        if local_jobs is not None:
            in_flight_count = len(local_jobs)
        if in_flight_count >= max_concurrent:
            break

        json_path = pop_next_pending(manifest)
        if json_path is None:
            break

        if not ensure_job_json(json_path, done_dir=done_dir):
            print(f"[run_susceptibility_all] missing JSON, re-queuing: {json_path}")
            requeue_front(json_path, path=manifest)
            continue

        patch_job = patch_coex_job_json if phase == "coex" else patch_prod_job_json
        if patch_job(json_path):
            print(f"[run_susceptibility_all] patched paths in {json_path}")

        staged_path = stage_job_json(json_path, staging_dir=staging_dir)

        if local_jobs is not None:
            proc = subprocess.Popen([python, "-u", runner, staged_path])
            job_id = f"local-{proc.pid}"
            mark_in_flight(job_id, json_path, path=manifest)
            local_jobs[job_id] = proc
        else:
            job_id = submit_slurm_job(staged_path, python, runner, config_path)
            mark_in_flight(job_id, json_path, path=manifest)

        print(f"[run_susceptibility_all] submitted {job_id}: {json_path}")
        submitted += 1

    return submitted


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatcher for susceptibility campaign jobs")
    parser.add_argument(
        "--phase",
        choices=["coex", "prod"],
        required=True,
        help="coex: slab μ sweep via json_runner; prod: square L via susceptibility_runner",
    )
    parser.add_argument("--config", default=SLURM_CONFIG)
    parser.add_argument("--interval", type=float, default=POLL_INTERVAL)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--exit-when-empty",
        action="store_true",
        help="Exit once pending and in_flight are both zero (default: keep polling for analyzer refinements)",
    )
    args = parser.parse_args()

    cfg = PHASE_CONFIG[args.phase]
    runner = cfg["runner"]
    manifest = cfg["manifest"]
    samples_root = cfg["samples_root"]
    done_dir = os.path.join(samples_root, "done")
    staging_dir = os.path.join(samples_root, "staging")

    qm.MANIFEST_PATH = manifest
    qm.DONE_DIR = done_dir
    qm.STAGING_DIR = staging_dir

    use_local = args.local or not slurm_available()
    if use_local and not args.local:
        print("[run_susceptibility_all] sbatch not found — running in local mode")

    python = sys.executable
    slurm = None
    local_jobs: dict[str, subprocess.Popen] | None = {} if use_local else None

    if not use_local:
        slurm = build_slurm(args.config)

    print(
        f"[run_susceptibility_all] phase={args.phase} runner={runner} "
        f"manifest={manifest} samples={samples_root} cwd={os.getcwd()} "
        f"mode={'local' if use_local else 'slurm'} "
        f"(max {args.max_concurrent} concurrent)"
    )
    if use_local:
        print(
            "[run_susceptibility_all] WARNING: local mode — jobs run on this node, "
            "not via sbatch; squeue will stay empty.",
            file=sys.stderr,
        )

    while True:
        reconcile_in_flight(
            manifest=manifest,
            done_dir=done_dir,
            slurm=slurm,
            local_jobs=local_jobs,
        )
        n_submitted = submit_up_to_cap(
            manifest=manifest,
            done_dir=done_dir,
            staging_dir=staging_dir,
            runner=runner,
            python=python,
            phase=args.phase,
            slurm=slurm,
            local_jobs=local_jobs,
            max_concurrent=args.max_concurrent,
            config_path=args.config,
        )
        manifest_data = read_manifest(manifest)
        pending = len(manifest_data.get("pending", []))
        in_flight = len(manifest_data.get("in_flight", {}))
        print(
            f"[run_susceptibility_all] cycle: submitted={n_submitted}, "
            f"pending={pending}, in_flight={in_flight}"
        )
        if pending > 0 and n_submitted == 0 and in_flight == 0 and not use_local:
            print(
                "[run_susceptibility_all] NOTE: pending>0 but nothing submitted this cycle "
                f"(manifest={os.path.abspath(manifest)})",
                file=sys.stderr,
            )

        if args.once:
            break
        if args.exit_when_empty and pending == 0 and in_flight == 0:
            print("[run_susceptibility_all] queue empty — exiting")
            break
        if pending == 0 and in_flight == 0:
            print(
                f"[run_susceptibility_all] queue idle — waiting {args.interval}s "
                f"for analyzer refinements (Ctrl-C to stop)",
            )
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
