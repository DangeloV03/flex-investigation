#!/usr/bin/env python3
"""
Estimate flex_sim job duration and total campaign wall time on Della.

Uses sacct for empirical per-job times (when available) and the queue manifest
+ results/ for progress. Run from the project root on a login node.

Usage:
    python scripts/estimate_runtime.py
    python scripts/estimate_runtime.py --job-id 9832776
    python scripts/estimate_runtime.py --since 2026-06-17
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

MANIFEST_PATH = "run_all_queue.json"
RESULTS_DIR = "results"
SLURM_CONFIG = "slurm_config.yml"
DEFAULT_MAX_CONCURRENT = 100
JOB_NAME = "flex_sim"


def parse_elapsed(raw: str) -> float | None:
    """Parse sacct Elapsed field to seconds."""
    raw = raw.strip()
    if not raw or raw in {"Unknown", "None", "N/A"}:
        return None
    # DD-HH:MM:SS
    m = re.fullmatch(r"(\d+)-(\d+):(\d+):(\d+)", raw)
    if m:
        d, h, mi, s = map(int, m.groups())
        return d * 86400 + h * 3600 + mi * 60 + s
    # HH:MM:SS or MM:SS
    parts = raw.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, mi, s = nums
        return h * 3600 + mi * 60 + s
    if len(nums) == 2:
        mi, s = nums
        return mi * 60 + s
    return None


def fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "n/a"
    s = int(round(seconds))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    if d:
        return f"{d}d {h:02d}h {m:02d}m"
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


def load_max_concurrent(config_path: str) -> int:
    if yaml is None or not os.path.isfile(config_path):
        return DEFAULT_MAX_CONCURRENT
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    # run_all caps concurrency; slurm config has no field for this
    return DEFAULT_MAX_CONCURRENT


def load_slurm_time_limit(config_path: str) -> str | None:
    if yaml is None or not os.path.isfile(config_path):
        return None
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("time")


def sacct_jobs(
    *,
    user: str | None,
    job_name: str,
    since: str | None,
    job_id: str | None,
) -> list[dict]:
    if shutil.which("sacct") is None:
        return []

    cmd = [
        "sacct",
        "-n",
        "-X",
        "-P",
        "-o",
        "JobID,JobName,State,Elapsed,ExitCode",
    ]
    if job_id:
        cmd.extend(["-j", job_id])
    else:
        if user:
            cmd.extend(["-u", user])
        cmd.extend(["--name", job_name])
        if since:
            cmd.extend(["-S", since])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return []

    if proc.returncode != 0:
        print(f"sacct failed: {proc.stderr.strip()}", file=sys.stderr)
        return []

    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jid, name, state, elapsed, exit_code = parts[:5]
        secs = parse_elapsed(elapsed)
        rows.append({
            "job_id": jid.strip(),
            "name": name.strip(),
            "state": state.strip(),
            "elapsed_raw": elapsed.strip(),
            "elapsed_s": secs,
            "exit_code": exit_code.strip(),
        })
    return rows


def queue_counts(manifest_path: str) -> tuple[int, int]:
    if not os.path.isfile(manifest_path):
        return 0, 0
    with open(manifest_path) as f:
        manifest = json.load(f)
    pending = len(manifest.get("pending", []))
    in_flight = len(manifest.get("in_flight", {}))
    return pending, in_flight


def count_finished_results(results_dir: str) -> int:
    from combo_paths import iter_output_csvs
    return sum(1 for _ in iter_output_csvs(results_dir))


def summarize_elapsed(jobs: list[dict], state_filter: set[str] | None = None) -> list[float]:
    out = []
    for j in jobs:
        if state_filter and j["state"] not in state_filter:
            continue
        if j["elapsed_s"] is not None and j["elapsed_s"] > 0:
            out.append(j["elapsed_s"])
    return out


def print_stats(label: str, times: list[float]) -> None:
    if not times:
        print(f"  {label}: no data")
        return
    times_sorted = sorted(times)
    print(f"  {label} ({len(times)} jobs):")
    print(f"    min:    {fmt_duration(times_sorted[0])}")
    print(f"    median: {fmt_duration(statistics.median(times_sorted))}")
    print(f"    mean:   {fmt_duration(statistics.mean(times_sorted))}")
    if len(times_sorted) >= 2:
        print(f"    max:    {fmt_duration(times_sorted[-1])}")
    if len(times_sorted) >= 20:
        p90 = times_sorted[int(0.9 * (len(times_sorted) - 1))]
        print(f"    p90:    {fmt_duration(p90)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate flex_sim job and campaign runtime.")
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--config", default=SLURM_CONFIG)
    parser.add_argument("--user", default=os.environ.get("USER"))
    parser.add_argument("--job-name", default=JOB_NAME)
    parser.add_argument("--job-id", help="Report elapsed time for one Slurm job ID")
    parser.add_argument(
        "--since",
        default=None,
        help="sacct start date YYYY-MM-DD (default: today for campaign estimate)",
    )
    parser.add_argument(
        "--seconds-per-job",
        type=float,
        default=None,
        help="Override per-job estimate instead of sacct median",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help=f"Override dispatcher concurrency (default {DEFAULT_MAX_CONCURRENT})",
    )
    args = parser.parse_args()

    if args.job_id:
        jobs = sacct_jobs(user=args.user, job_name=args.job_name, since=None, job_id=args.job_id)
        if not jobs:
            print(f"No sacct record for job {args.job_id}")
            return 1
        j = jobs[0]
        print(f"Job {j['job_id']}  state={j['state']}  elapsed={j['elapsed_raw']}  exit={j['exit_code']}")
        if j["elapsed_s"] is not None:
            print(f"  ({j['elapsed_s']:.0f} seconds)")
        return 0

    since = args.since or datetime.now().strftime("%Y-%m-%d")
    jobs = sacct_jobs(user=args.user, job_name=args.job_name, since=since, job_id=None)

    completed_times = summarize_elapsed(jobs, {"COMPLETED", "COMPLETING"})
    running = [j for j in jobs if j["state"] == "RUNNING"]
    failed = [j for j in jobs if j["state"] in {"FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY"}]

    pending, in_flight = queue_counts(args.manifest)
    finished_csv = count_finished_results(args.results)
    max_conc = args.max_concurrent or load_max_concurrent(args.config)
    slurm_limit = load_slurm_time_limit(args.config)

    print("=== flex_sim runtime estimate ===\n")
    print(f"sacct since {since} (job name {args.job_name!r})")
    print_stats("Completed jobs", completed_times)

    if running:
        run_times = summarize_elapsed(running, None)
        print(f"  Running now: {len(running)} job(s)")
        if run_times:
            print(f"    elapsed so far (median): {fmt_duration(statistics.median(run_times))}")

    if failed:
        print(f"  Failed/cancelled/timeout: {len(failed)} job(s)")

    if slurm_limit:
        print(f"\nSlurm wall limit per job (slurm_config.yml): {slurm_limit}")

    per_job = args.seconds_per_job
    if per_job is None and completed_times:
        per_job = statistics.median(completed_times)
    elif per_job is None:
        print(
            "\nNo completed sacct jobs yet — cannot estimate per-job time.\n"
            "  Run after a few jobs finish, or pass --seconds-per-job <sec> manually.",
            file=sys.stderr,
        )
        per_job = None

    print(f"\nQueue / progress:")
    print(f"  pending (manifest):     {pending}")
    print(f"  in_flight (manifest):   {in_flight}")
    print(f"  output.csv (results):   {finished_csv}")
    print(f"  max concurrent:         {max_conc}")

    if per_job is not None:
        remaining = pending + in_flight
        # Rough wall clock: remaining jobs / parallelism * per-job time
        est_remaining_s = (remaining * per_job) / max(max_conc, 1)
        est_total_from_scratch = ((pending + in_flight + finished_csv) * per_job) / max(max_conc, 1)

        print(f"\nEstimate (using {fmt_duration(per_job)} per job):")
        print(f"  remaining wall time:  ~{fmt_duration(est_remaining_s)}")
        print(f"  full campaign (approx, if starting from 0): ~{fmt_duration(est_total_from_scratch)}")
        print(
            "\nNote: Ignores analyzer refinement jobs, failures/retries, and queue wait.\n"
            "      Refinement can add 0–5 extra μ-batches per combo (~×1–6 per combo worst case)."
        )
    else:
        print("\nManual one-job check after a job completes:")
        print("  sacct -j JOBID -X --format=JobID,State,Elapsed,ExitCode")
        print("  python scripts/estimate_runtime.py --job-id JOBID")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
