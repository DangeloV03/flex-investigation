"""Files of a susceptibility run: m(t) time series, the replica table, and checkpoints."""

from __future__ import annotations

import csv
import os

import numpy as np

from flexsim.susc.layout import SUSCEPTIBILITY_CSV_FIELDS, read_susceptibility_csv


CSV_FIELDNAMES = SUSCEPTIBILITY_CSV_FIELDS

TIMESERIES_FIELDNAMES = ["chunk", "rho_bonding", "rho_inert", "rho_empty", "m", "energy"]

def save_timeseries_csv(path: str, chunks: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMESERIES_FIELDNAMES)
        writer.writeheader()
        writer.writerows(chunks)


def get_next_id(csv_path: str) -> int:
    existing_ids = [
        int(r["id"])
        for r in read_susceptibility_csv(csv_path)
        if str(r.get("id", "")).strip()
    ]
    return max(existing_ids) + 1 if existing_ids else 0


def append_to_csv(csv_path: str, rows: list[dict]) -> None:
    existing = read_susceptibility_csv(csv_path)
    normalized = [{field: row.get(field, "") for field in CSV_FIELDNAMES} for row in rows]
    all_rows = existing + normalized
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)


def drop_superseded_timeseries(old_path: str, new_path: str) -> bool:
    """Delete a resumed-from timeseries once its successor is safely recorded.

    A top-up writes the full history (prior + new chunks) to a new file, so the
    prior file is a byte-for-byte prefix of it.  Keeping every generation grew
    disk quadratically (S1B: 726 of 748 GiB were superseded copies).  Called
    only after the successor's CSV row is written, and deletes only when the
    prefix really matches, so no history can be lost.
    """
    if os.path.abspath(old_path) == os.path.abspath(new_path):
        return False
    try:
        if os.path.getsize(new_path) <= os.path.getsize(old_path):
            return False
        with open(old_path, "rb") as f_old, open(new_path, "rb") as f_new:
            while True:
                block = f_old.read(1 << 20)
                if not block:
                    break
                if f_new.read(len(block)) != block:
                    return False
        os.remove(old_path)
    except OSError:
        return False
    return True


def load_timeseries_csv(path: str) -> list[dict]:
    """Read m_timeseries CSV; return list of row dicts with numeric types."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["chunk"] = int(r["chunk"])
        for k in ("rho_bonding", "rho_inert", "rho_empty", "m", "energy"):
            r[k] = float(r[k])
    return rows


def ckpt_path(outdir: str, run_id: int) -> str:
    return os.path.join(outdir, f"ckpt_{run_id}.npz")


def write_checkpoint(
    path: str,
    state: np.ndarray,
    chunks: list[dict],
    cumulative_time: float,
    resume_id: int | None,
) -> None:
    """Atomically write lattice + chunk history so a kill leaves the old file intact."""
    cols = {c: np.asarray([r[c] for r in chunks], dtype=float) for c in TIMESERIES_FIELDNAMES}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(
            f,
            state=state,
            cumulative_time=float(cumulative_time),
            resume_id=-1 if resume_id is None else int(resume_id),
            **cols,
        )
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict | None:
    """Return {state, chunks, cumulative_time, resume_id} or None if unreadable."""
    try:
        with np.load(path) as z:
            n = len(z["chunk"])
            chunks = [
                {"chunk": int(z["chunk"][i]),
                 **{c: float(z[c][i]) for c in TIMESERIES_FIELDNAMES if c != "chunk"}}
                for i in range(n)
            ]
            rid = int(z["resume_id"])
            return {
                "state": z["state"],
                "chunks": chunks,
                "cumulative_time": float(z["cumulative_time"]),
                "resume_id": None if rid < 0 else rid,
            }
    except Exception as exc:  # corrupt/partial file: fall back to a fresh run
        print(f"[susc.runner] ignoring unreadable checkpoint {path}: {exc}", flush=True)
        return None


def orphan_checkpoint_ids(outdir: str, done_ids: set[int]) -> list[int]:
    """Run ids with a checkpoint on disk but no row in the CSV (killed mid-run)."""
    ids = []
    for name in os.listdir(outdir):
        if name.startswith("ckpt_") and name.endswith(".npz"):
            try:
                rid = int(name[len("ckpt_"):-len(".npz")])
            except ValueError:
                continue
            if rid not in done_ids:
                ids.append(rid)
    return sorted(ids)
