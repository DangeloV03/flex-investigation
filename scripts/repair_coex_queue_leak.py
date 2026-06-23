#!/usr/bin/env python3
"""
Move coex job paths mistakenly queued in run_all_queue.json back to the coex manifest.

Background: analyzer prepend_pending used to write to run_all_queue.json instead of
susceptibility_coex_queue.json. Main run_all.py ignores coex samples; coex dispatcher
never saw those jobs.

Usage:
    python scripts/repair_coex_queue_leak.py --dry-run
    python scripts/repair_coex_queue_leak.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from queue_manifest import locked_manifest, read_manifest
from susceptibility_paths import COEX_MANIFEST, COEX_SAMPLES_DIR

MAIN_MANIFEST = "run_all_queue.json"


def is_coex_path(path: str) -> bool:
    return path.startswith(COEX_SAMPLES_DIR) or "/coex/" in path and path.endswith(".json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair coex jobs leaked into run_all_queue.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    main_data = read_manifest(MAIN_MANIFEST)
    coex_data = read_manifest(COEX_MANIFEST)

    leaked_pending = [p for p in main_data.get("pending", []) if is_coex_path(p)]
    leaked_flight = {
        jid: p for jid, p in main_data.get("in_flight", {}).items() if is_coex_path(p)
    }

    coex_known = set(coex_data.get("pending", [])) | set(coex_data.get("in_flight", {}).values())
    to_coex = [p for p in leaked_pending if p not in coex_known]

    print(f"Leaked in {MAIN_MANIFEST}: pending={len(leaked_pending)}  in_flight={len(leaked_flight)}")
    print(f"Would prepend to {COEX_MANIFEST}: {len(to_coex)} path(s) not already there")

    if args.dry_run:
        for p in to_coex[:10]:
            print(f"  would move: {p}")
        if len(to_coex) > 10:
            print(f"  ... and {len(to_coex) - 10} more")
        return

    if leaked_flight:
        print(
            "WARNING: coex paths still in_flight on main queue — cancel those Slurm jobs or "
            "wait for them to finish before repair.",
        )
        for jid, p in leaked_flight.items():
            print(f"  job {jid}: {p}")

    with locked_manifest(MAIN_MANIFEST) as main_m:
        main_m["pending"] = [p for p in main_m.get("pending", []) if not is_coex_path(p)]
        main_m["in_flight"] = {
            jid: p for jid, p in main_m.get("in_flight", {}).items() if not is_coex_path(p)
        }

    if to_coex:
        with locked_manifest(COEX_MANIFEST) as coex_m:
            existing = set(coex_m.get("pending", [])) | set(coex_m.get("in_flight", {}).values())
            new_paths = [p for p in to_coex if p not in existing]
            coex_m["pending"] = new_paths + coex_m.get("pending", [])

    after_main = read_manifest(MAIN_MANIFEST)
    after_coex = read_manifest(COEX_MANIFEST)
    print(
        f"Done. {MAIN_MANIFEST}: pending={len(after_main.get('pending', []))}  "
        f"{COEX_MANIFEST}: pending={len(after_coex.get('pending', []))}",
    )


if __name__ == "__main__":
    main()
