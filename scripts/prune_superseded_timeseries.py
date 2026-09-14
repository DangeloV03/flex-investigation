#!/usr/bin/env python3
"""Delete m_timeseries files that a later top-up fully supersedes.

Each top-up resumes replica X and writes the complete history (X's chunks plus
the new ones) to a new file m_timeseries_{Y}.csv, recording resume_id=X on Y's
row.  X's file is therefore a byte-for-byte prefix of Y's: pure duplication
that grew quadratically with rounds (S1B: 726 of 748 GiB).

A file is deleted only when all of these hold:
  * a row in the same susceptibility_data.csv has resume_id == its id;
  * following resume_id links forward ends at exactly one newest file;
  * that newest file exists, is longer, and starts with this file's exact
    bytes (full streaming compare; --quick checks size + first/last MiB).
The newest generation is never touched, nor are susceptibility_data.csv rows
or final_lattice_*.npy.  Do not run while top-ups for the campaign are active.

    # see what would go (full verification; run in tmux, it reads every file)
    python scripts/prune_superseded_timeseries.py --results-base SUSC_RUNS_S1A

    # try a handful of run dirs first, then everything
    python scripts/prune_superseded_timeseries.py --results-base SUSC_RUNS_S1A --limit-dirs 3 --apply
    python scripts/prune_superseded_timeseries.py --results-base SUSC_RUNS_S1A --apply
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "susceptibility"), os.path.join(REPO, "coex")]

from susceptibility_paths import SUSCEPTIBILITY_DATA_CSV, read_susceptibility_csv  # noqa: E402

BLOCK = 1 << 20


def _is_prefix(old_path: str, new_path: str, quick: bool) -> bool:
    """True if new_path is longer than old_path and begins with its bytes."""
    old_size = os.path.getsize(old_path)
    if os.path.getsize(new_path) <= old_size:
        return False
    with open(old_path, "rb") as f_old, open(new_path, "rb") as f_new:
        if quick:
            head = f_old.read(BLOCK)
            if f_new.read(len(head)) != head:
                return False
            tail_start = max(0, old_size - BLOCK)
            f_old.seek(tail_start)
            f_new.seek(tail_start)
            tail = f_old.read()
            return f_new.read(len(tail)) == tail
        while True:
            block = f_old.read(BLOCK)
            if not block:
                return True
            if f_new.read(len(block)) != block:
                return False


def _newest_descendant(run_id: str, children: dict[str, list[str]]) -> str | None:
    """Follow resume links forward; None if a replica ever forked."""
    seen = {run_id}
    cur = run_id
    while cur in children:
        kids = children[cur]
        if len(kids) != 1 or kids[0] in seen:
            return None
        cur = kids[0]
        seen.add(cur)
    return cur


def prune_run_dir(run_dir: str, *, quick: bool, apply: bool) -> dict:
    stats = {"superseded": 0, "verified": 0, "bytes": 0, "mismatch": 0,
             "ambiguous": 0, "missing_successor": 0}
    rows = read_susceptibility_csv(os.path.join(run_dir, SUSCEPTIBILITY_DATA_CSV))
    children: dict[str, list[str]] = {}
    for row in rows:
        rid = str(row.get("id", "")).strip()
        parent = str(row.get("resume_id", "")).strip()
        if rid and parent:
            children.setdefault(parent, []).append(rid)

    for parent in sorted(children, key=lambda x: int(x) if x.isdigit() else -1):
        old_path = os.path.join(run_dir, f"m_timeseries_{parent}.csv")
        if not os.path.isfile(old_path):
            continue  # already pruned
        stats["superseded"] += 1
        newest = _newest_descendant(parent, children)
        if newest is None or newest == parent:
            stats["ambiguous"] += 1
            continue
        new_path = os.path.join(run_dir, f"m_timeseries_{newest}.csv")
        if not os.path.isfile(new_path):
            stats["missing_successor"] += 1
            continue
        try:
            ok = _is_prefix(old_path, new_path, quick)
        except OSError:
            ok = False
        if not ok:
            stats["mismatch"] += 1
            continue
        size = os.path.getsize(old_path)
        if apply:
            os.remove(old_path)
        stats["verified"] += 1
        stats["bytes"] += size
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-base", required=True,
                    help="Campaign root, e.g. SUSC_RUNS_S1A")
    ap.add_argument("--quick", action="store_true",
                    help="Verify size + first/last MiB instead of every byte")
    ap.add_argument("--limit-dirs", type=int, default=None,
                    help="Only process the first N run dirs (for a trial)")
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete. Without this, only report.")
    args = ap.parse_args()

    base = args.results_base.rstrip("/")
    if not os.path.isdir(base):
        raise SystemExit(f"No such campaign directory: {base}")

    run_dirs = sorted(os.path.dirname(p) for p in
                      glob.glob(os.path.join(base, "*", "_*", SUSCEPTIBILITY_DATA_CSV)))
    if args.limit_dirs is not None:
        run_dirs = run_dirs[: args.limit_dirs]

    totals = {"superseded": 0, "verified": 0, "bytes": 0, "mismatch": 0,
              "ambiguous": 0, "missing_successor": 0}
    print(f"campaign:  {base}  ({len(run_dirs)} run dirs, "
          f"{'quick' if args.quick else 'full'} verification, "
          f"{'APPLY' if args.apply else 'dry run'})", flush=True)
    for i, run_dir in enumerate(run_dirs, start=1):
        s = prune_run_dir(run_dir, quick=args.quick, apply=args.apply)
        for k in totals:
            totals[k] += s[k]
        if i == 1 or i % 25 == 0 or i == len(run_dirs):
            print(f"  {i}/{len(run_dirs)} dirs  {totals['verified']} files  "
                  f"{totals['bytes'] / 1024**3:.1f} GiB", flush=True)

    verb = "deleted" if args.apply else "would delete"
    print(f"\nsuperseded files found:   {totals['superseded']}")
    print(f"verified and {verb}: {totals['verified']} "
          f"({totals['bytes'] / 1024**3:.1f} GiB)")
    print(f"kept, prefix mismatch:    {totals['mismatch']}")
    print(f"kept, forked lineage:     {totals['ambiguous']}")
    print(f"kept, newest file absent: {totals['missing_successor']}")
    if not args.apply:
        print("\nDRY RUN — nothing deleted. Re-run with --apply.")
    return 1 if totals["mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
