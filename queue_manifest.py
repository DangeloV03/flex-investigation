"""
queue_manifest.py

Shared read-modify-write helpers for run_all_queue.json.

Manifest schema:
    {
        "pending": ["samples/job1.json", ...],   # front = next to submit
        "in_flight": {"12345": "samples/job2.json"}
    }
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from typing import Iterator

MANIFEST_PATH = "run_all_queue.json"
DONE_DIR = "samples/done"


def _empty_manifest() -> dict:
    return {"pending": [], "in_flight": {}}


@contextmanager
def locked_manifest(path: str = MANIFEST_PATH) -> Iterator[dict]:
    """Atomically read-modify-write the manifest under an exclusive file lock."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read()
        if raw.strip():
            manifest = json.loads(raw)
        else:
            manifest = _empty_manifest()
        manifest.setdefault("pending", [])
        manifest.setdefault("in_flight", {})
        yield manifest
        f.seek(0)
        f.truncate()
        json.dump(manifest, f, indent=2)
        f.write("\n")
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_manifest(path: str = MANIFEST_PATH) -> dict:
    if not os.path.isfile(path):
        return _empty_manifest()
    with open(path) as f:
        manifest = json.load(f)
    manifest.setdefault("pending", [])
    manifest.setdefault("in_flight", {})
    return manifest


def seed_pending(paths: list[str], path: str = MANIFEST_PATH) -> None:
    """Replace pending queue with paths (used by generate_samples)."""
    with locked_manifest(path) as manifest:
        manifest["pending"] = list(paths)


def prepend_pending(paths: list[str], path: str = MANIFEST_PATH) -> None:
    """Prepend paths to the front of pending (analyzer priority / stack)."""
    if not paths:
        return
    with locked_manifest(path) as manifest:
        existing = set(manifest["pending"])
        in_flight = set(manifest["in_flight"].values())
        new_paths = [p for p in paths if p not in existing and p not in in_flight]
        manifest["pending"] = new_paths + manifest["pending"]


def merge_pending(paths: list[str], path: str = MANIFEST_PATH) -> None:
    """Append new paths to pending, skipping duplicates and in-flight jobs."""
    if not paths:
        return
    with locked_manifest(path) as manifest:
        existing = set(manifest["pending"])
        in_flight = set(manifest["in_flight"].values())
        new_paths = [
            p for p in paths
            if p not in existing and p not in in_flight
        ]
        manifest["pending"].extend(new_paths)


def pop_next_pending(path: str = MANIFEST_PATH) -> str | None:
    """Remove and return the next pending JSON path, or None if empty."""
    with locked_manifest(path) as manifest:
        if not manifest["pending"]:
            return None
        return manifest["pending"].pop(0)


def mark_in_flight(job_id: str, json_path: str, path: str = MANIFEST_PATH) -> None:
    with locked_manifest(path) as manifest:
        manifest["in_flight"][str(job_id)] = json_path


def remove_in_flight(job_id: str, path: str = MANIFEST_PATH) -> str | None:
    """Remove a job from in_flight and return its JSON path."""
    with locked_manifest(path) as manifest:
        return manifest["in_flight"].pop(str(job_id), None)


def requeue_front(json_path: str, path: str = MANIFEST_PATH) -> None:
    """Put a failed job back at the front of pending."""
    prepend_pending([json_path], path=path)


def archive_json(json_path: str, done_dir: str = DONE_DIR) -> None:
    """Move a completed job JSON out of samples/."""
    if not os.path.isfile(json_path):
        return
    os.makedirs(done_dir, exist_ok=True)
    dest = os.path.join(done_dir, os.path.basename(json_path))
    if os.path.exists(dest):
        os.remove(dest)
    os.rename(json_path, dest)
