"""
Tests for the smart susceptibility runner's append/resume logic.

Exercises the file I/O, CSV schema, and accumulation bookkeeping without
needing the lattice_gas Rust extension.
"""

from __future__ import annotations

import csv
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Stub out the compiled Rust extension so susceptibility_runner is importable locally.
for _mod in (
    "lattice_gas",
    "lattice_gas.load",
    "lattice_gas.boundary_condition",
    "lattice_gas.ending_criterion",
    "lattice_gas.markov_chain",
    "lattice_gas.simulate",
):
    sys.modules.setdefault(_mod, MagicMock())

# ---------------------------------------------------------------------------
# Helpers that mirror what susceptibility_runner.py does, but without Rust.
# ---------------------------------------------------------------------------


def _write_fresh_rows(csv_path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_ts(ts_path: str, m_values: list[float]) -> None:
    with open(ts_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["chunk", "rho_bonding", "rho_inert", "rho_empty", "m", "energy"]
        )
        w.writeheader()
        for i, m in enumerate(m_values):
            w.writerow(
                dict(chunk=i, rho_bonding=0.5, rho_inert=0.1, rho_empty=0.4, m=m, energy=-1.0)
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_csv_schema_v2_and_v3_roundtrip(tmp_path):
    """v2 rows (no resume_id) and v3 rows both read back with all fields."""
    from susceptibility_paths import (
        SUSCEPTIBILITY_CSV_FIELDS,
        SUSCEPTIBILITY_CSV_FIELDS_V2,
        read_susceptibility_csv,
    )

    v2_csv = str(tmp_path / "v2.csv")
    v3_csv = str(tmp_path / "v3.csv")

    # Write a v2-schema row (no resume_id column)
    row_v2 = {f: "" for f in SUSCEPTIBILITY_CSV_FIELDS_V2}
    row_v2.update(id="5", epsilon="-1.76", prod_time="1000000.0", time="120.3")
    _write_fresh_rows(v2_csv, [row_v2], SUSCEPTIBILITY_CSV_FIELDS_V2)

    # Write a v3-schema row (has resume_id)
    row_v3 = {f: "" for f in SUSCEPTIBILITY_CSV_FIELDS}
    row_v3.update(id="6", epsilon="-1.76", prod_time="1100000.0", time="140.0", resume_id="5")
    _write_fresh_rows(v3_csv, [row_v3], SUSCEPTIBILITY_CSV_FIELDS)

    rows_v2 = read_susceptibility_csv(v2_csv)
    assert len(rows_v2) == 1
    assert rows_v2[0]["id"] == "5"
    assert rows_v2[0]["epsilon"] == "-1.76"
    assert rows_v2[0]["resume_id"] == ""  # filled in as empty for old rows

    rows_v3 = read_susceptibility_csv(v3_csv)
    assert len(rows_v3) == 1
    assert rows_v3[0]["id"] == "6"
    assert rows_v3[0]["resume_id"] == "5"


def test_get_next_id_and_append(tmp_path):
    """get_next_id reads existing max; append_to_csv preserves old rows."""
    from susceptibility_paths import SUSCEPTIBILITY_CSV_FIELDS, SUSCEPTIBILITY_DATA_CSV
    from susceptibility_runner import append_to_csv, get_next_id

    csv_path = str(tmp_path / SUSCEPTIBILITY_DATA_CSV)

    # Empty CSV → next id is 0
    assert get_next_id(csv_path) == 0

    # Write two rows with ids 0 and 7
    row_a = {f: "" for f in SUSCEPTIBILITY_CSV_FIELDS}
    row_a.update(id="0", epsilon="-1.76", prod_time="1000000.0", time="100.0")
    row_b = {f: "" for f in SUSCEPTIBILITY_CSV_FIELDS}
    row_b.update(id="7", epsilon="-1.80", prod_time="1000000.0", time="105.0")
    _write_fresh_rows(csv_path, [row_a, row_b], SUSCEPTIBILITY_CSV_FIELDS)

    assert get_next_id(csv_path) == 8

    # Append a new row
    new_row = {f: "" for f in SUSCEPTIBILITY_CSV_FIELDS}
    new_row.update(id="8", epsilon="-1.76", prod_time="1000000.0", time="99.0", resume_id="")
    append_to_csv(csv_path, [new_row])

    from susceptibility_paths import read_susceptibility_csv
    rows = read_susceptibility_csv(csv_path)
    assert len(rows) == 3
    assert {r["id"] for r in rows} == {"0", "7", "8"}


def test_load_timeseries_csv(tmp_path):
    """_load_timeseries_csv reads m values correctly."""
    from susceptibility_runner import _load_timeseries_csv

    ts_path = str(tmp_path / "m_timeseries_0.csv")
    m_vals = [0.9, -0.9, 0.8, -0.8]
    _write_ts(ts_path, m_vals)

    chunks = _load_timeseries_csv(ts_path)
    assert len(chunks) == 4
    assert [float(c["m"]) for c in chunks] == pytest.approx(m_vals)


def test_resume_prod_time_accumulation(tmp_path):
    """Simulate what run_replica does in resume mode: prod_time and timeseries accumulate."""
    from susceptibility_paths import (
        SUSCEPTIBILITY_CSV_FIELDS,
        SUSCEPTIBILITY_DATA_CSV,
        read_susceptibility_csv,
    )
    from susceptibility_runner import (
        _load_timeseries_csv,
        append_to_csv,
        get_next_id,
    )

    run_dir = str(tmp_path)
    csv_path = os.path.join(run_dir, SUSCEPTIBILITY_DATA_CSV)

    # --- original run: 10 chunks, prod_time=1_000_000 ---
    orig_m = list(np.sin(np.linspace(0, 2 * np.pi, 10)))
    orig_prod_time = 1_000_000.0
    orig_wall_time = 50.0

    orig_row = {f: "" for f in SUSCEPTIBILITY_CSV_FIELDS}
    orig_row.update(id="0", prod_time=str(orig_prod_time), time=str(orig_wall_time),
                    resume_id="")
    _write_fresh_rows(csv_path, [orig_row], SUSCEPTIBILITY_CSV_FIELDS)
    _write_ts(os.path.join(run_dir, "m_timeseries_0.csv"), orig_m)
    np.save(os.path.join(run_dir, "final_lattice_0.npy"), np.zeros((4, 4), dtype=np.uint32))

    assert get_next_id(csv_path) == 1

    # --- simulate resume logic (mirrors run_replica with resume) ---
    prior_rows = read_susceptibility_csv(csv_path)
    max_pt = max(float(r.get("prod_time", 0) or 0) for r in prior_rows)
    latest = [r for r in prior_rows
              if abs(float(r.get("prod_time", 0) or 0) - max_pt) < 1.0]
    assert len(latest) == 1
    resume_id = int(latest[0]["id"])
    prior_prod_time = float(latest[0]["prod_time"])
    prior_wall_time = float(latest[0]["time"])

    prior_chunks = _load_timeseries_csv(os.path.join(run_dir, f"m_timeseries_{resume_id}.csv"))
    assert len(prior_chunks) == 10

    # new top-up: 5 more chunks
    topup_m = list(np.sin(np.linspace(np.pi, 3 * np.pi, 5)))
    topup_prod_time = 100_000.0
    topup_wall_time = 8.0
    topup_chunks = [
        {"chunk": len(prior_chunks) + i, "rho_bonding": 0.5, "rho_inert": 0.1,
         "rho_empty": 0.4, "m": m, "energy": -1.0}
        for i, m in enumerate(topup_m)
    ]

    all_chunks = prior_chunks + topup_chunks
    assert len(all_chunks) == 15  # 10 prior + 5 new

    # check chunk indices are monotonically increasing
    indices = [int(c["chunk"]) for c in all_chunks]
    assert indices == list(range(15))

    new_run_id = get_next_id(csv_path)  # = 1
    total_prod_time = prior_prod_time + topup_prod_time
    total_wall_time = prior_wall_time + topup_wall_time

    topup_row = {f: "" for f in SUSCEPTIBILITY_CSV_FIELDS}
    topup_row.update(
        id=str(new_run_id),
        prod_time=str(total_prod_time),
        time=str(total_wall_time),
        prod_chunks=str(len(all_chunks)),
        resume_id=str(resume_id),
    )
    append_to_csv(csv_path, [topup_row])

    # --- verify final CSV state ---
    rows = read_susceptibility_csv(csv_path)
    assert len(rows) == 2

    topup_row_back = next(r for r in rows if r["id"] == "1")
    assert float(topup_row_back["prod_time"]) == pytest.approx(1_100_000.0)
    assert float(topup_row_back["time"]) == pytest.approx(58.0)
    assert int(topup_row_back["prod_chunks"]) == 15
    assert topup_row_back["resume_id"] == "0"


def test_latest_gen_detection_multi_round(tmp_path):
    """After two top-up rounds, auto-detection picks the highest-prod_time rows."""
    from susceptibility_paths import SUSCEPTIBILITY_CSV_FIELDS, SUSCEPTIBILITY_DATA_CSV
    from susceptibility_runner import get_next_id

    csv_path = str(tmp_path / SUSCEPTIBILITY_DATA_CSV)

    # Two original replicas (IDs 0 and 1), prod_time=1M
    rows = []
    for rid in (0, 1):
        r = {f: "" for f in SUSCEPTIBILITY_CSV_FIELDS}
        r.update(id=str(rid), prod_time="1000000.0", time="50.0", resume_id="")
        rows.append(r)
    # First top-up (IDs 2 and 3), prod_time=1.1M
    for rid, resume_of in ((2, 0), (3, 1)):
        r = {f: "" for f in SUSCEPTIBILITY_CSV_FIELDS}
        r.update(id=str(rid), prod_time="1100000.0", time="60.0", resume_id=str(resume_of))
        rows.append(r)

    _write_fresh_rows(csv_path, rows, SUSCEPTIBILITY_CSV_FIELDS)
    assert get_next_id(csv_path) == 4

    from susceptibility_paths import read_susceptibility_csv
    all_rows = read_susceptibility_csv(csv_path)
    max_pt = max(float(r.get("prod_time", 0) or 0) for r in all_rows)
    latest = [r for r in all_rows if abs(float(r.get("prod_time", 0) or 0) - max_pt) < 1.0]
    latest_ids = {int(r["id"]) for r in latest}

    # Should pick IDs 2 and 3 (the top-up rows with higher prod_time)
    assert latest_ids == {2, 3}
