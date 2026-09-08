#!/usr/bin/env python3
"""Delete the check-round artifacts produced by the no-op top-up loop.

Rounds 2..N of a stalled campaign re-analysed the exact same samples: their
round PNGs, timing reports and slurm_jobs.csv rows describe work that never
happened.  This removes those artifacts and the header-only timing CSVs left
by top-up jobs that skipped every size, plus the corresponding Slurm logs.

Simulation data is never touched: susceptibility_data.csv, final_lattice_*.npy
and m_timeseries_*.csv are refused outright.

    # see what would go
    python scripts/prune_stalled_rounds.py --results-base SUSC_RUNS_S1B

    # actually delete, keeping round 1
    python scripts/prune_stalled_rounds.py --results-base SUSC_RUNS_S1B --apply
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import sys

PROTECTED = ("susceptibility_data.csv", "final_lattice_", "m_timeseries_")
JOBS_CSV = "slurm_jobs.csv"
ROUND_PATTERNS = ("jump_check_round_{n}.png", "timing_round_{n}.md",
                  "timing_round_{n}.csv", "timing_round_{n}.png")


def _guard(path: str) -> None:
    base = os.path.basename(path)
    if any(p in base for p in PROTECTED):
        raise SystemExit(f"REFUSING to delete simulation data: {path}")


def _rounds_present(results_base: str) -> list[int]:
    rounds = []
    for p in glob.glob(os.path.join(results_base, "jump_check_round_*.png")):
        m = re.search(r"jump_check_round_(\d+)\.png$", p)
        if m:
            rounds.append(int(m.group(1)))
    return sorted(rounds)


def _is_header_only(path: str) -> bool:
    try:
        with open(path, newline="") as f:
            rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    except OSError:
        return False
    return len(rows) <= 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-base", required=True,
                    help="Campaign root, e.g. SUSC_RUNS_S1B")
    ap.add_argument("--keep-through", type=int, default=1,
                    help="Highest round number to keep (default: 1)")
    ap.add_argument("--slurm-reports", default="slurm_reports",
                    help="Directory holding susc_top_*.out/err (default: slurm_reports)")
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete. Without this, only report.")
    args = ap.parse_args()

    base = args.results_base.rstrip("/")
    if not os.path.isdir(base):
        raise SystemExit(f"No such campaign directory: {base}")

    doomed: list[str] = []

    # 1. Round figures and timing reports above the keep line.
    stale_rounds = [n for n in _rounds_present(base) if n > args.keep_through]
    for n in stale_rounds:
        for pat in ROUND_PATTERNS:
            p = os.path.join(base, pat.format(n=n))
            if os.path.exists(p):
                doomed.append(p)

    # 2. slurm_jobs.csv rows for those rounds; collect their job ids.
    jobs_path = os.path.join(base, JOBS_CSV)
    stale_job_ids: set[str] = set()
    kept_rows: list[dict] = []
    jobs_header: list[str] = []
    if os.path.isfile(jobs_path):
        with open(jobs_path, newline="") as f:
            reader = csv.DictReader(f)
            jobs_header = list(reader.fieldnames or [])
            for row in reader:
                try:
                    rnd = int(row.get("round", ""))
                except ValueError:
                    kept_rows.append(row)
                    continue
                if rnd > args.keep_through:
                    stale_job_ids.add(str(row.get("job_id", "")).strip())
                else:
                    kept_rows.append(row)

    # 3. Per-job timing CSVs for stale jobs — only the header-only ones, so a
    #    job that did real work keeps its timing record.
    timing_doomed: list[str] = []
    timing_kept_with_data: list[str] = []
    for job_id in sorted(stale_job_ids):
        if not job_id or job_id.startswith("DRY"):
            continue
        p = os.path.join(base, "timing", f"{job_id}.csv")
        if not os.path.isfile(p):
            continue
        (timing_doomed if _is_header_only(p) else timing_kept_with_data).append(p)
    doomed.extend(timing_doomed)

    # 4. Slurm logs from top-up jobs that skipped every size for this campaign.
    log_doomed: list[str] = []
    for out in sorted(glob.glob(os.path.join(args.slurm_reports, "susc_top_*.out"))):
        try:
            with open(out, errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        if "does not exist, skipping" not in text and "no run found" not in text:
            continue
        if os.path.basename(base) not in text:
            continue
        if ">>> TOP-UP" in text:      # this job actually ran something
            continue
        log_doomed.append(out)
        err = out[:-4] + ".err"
        if os.path.exists(err):
            log_doomed.append(err)
    doomed.extend(log_doomed)

    # 5. Stale stall fingerprint, if the new check ever wrote one.
    fp = os.path.join(base, ".jump_progress.json")
    if os.path.exists(fp):
        doomed.append(fp)

    for p in doomed:
        _guard(p)

    print(f"campaign:        {base}")
    print(f"keep through:    round {args.keep_through}")
    print(f"stale rounds:    {len(stale_rounds)}"
          + (f" ({stale_rounds[0]}–{stale_rounds[-1]})" if stale_rounds else ""))
    n_round_files = sum(
        1 for p in doomed
        if os.path.basename(p).startswith(("jump_check_round_", "timing_round_"))
    )
    print(f"round artifacts: {n_round_files}")
    print(f"timing CSVs:     {len(timing_doomed)} header-only removed, "
          f"{len(timing_kept_with_data)} kept (contain data)")
    print(f"slurm logs:      {len(log_doomed)}")
    print(f"slurm_jobs.csv:  {len(stale_job_ids)} job rows dropped, {len(kept_rows)} kept")
    total_bytes = sum(os.path.getsize(p) for p in doomed if os.path.exists(p))
    print(f"total:           {len(doomed)} files, {total_bytes / 1e6:.1f} MB")

    if not args.apply:
        print("\nDRY RUN — nothing deleted. Re-run with --apply.")
        for p in doomed[:15]:
            print(f"  would delete {p}")
        if len(doomed) > 15:
            print(f"  … and {len(doomed) - 15} more")
        return 0

    for p in doomed:
        _guard(p)
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    if os.path.isfile(jobs_path) and stale_job_ids:
        shutil.copy2(jobs_path, jobs_path + ".bak")
        with open(jobs_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=jobs_header)
            w.writeheader()
            w.writerows(kept_rows)
        print(f"rewrote {jobs_path} (backup at {jobs_path}.bak)")

    print(f"\nDeleted {len(doomed)} files. Next check will run as "
          f"round {args.keep_through + 1}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
