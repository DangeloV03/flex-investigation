"""Tests for scripts/prune_superseded_timeseries.py."""

from __future__ import annotations

import csv
import importlib.util
import os

import pytest

from susceptibility_paths import SUSCEPTIBILITY_CSV_FIELDS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "prune_superseded_timeseries", os.path.join(REPO, "scripts", "prune_superseded_timeseries.py")
)
prune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune)


def _campaign(tmp_path):
    """Lineage 0→1→2 (true prefixes), fresh 3, and 4→5 whose history diverges."""
    run_dir = tmp_path / "_16_16_S1_DF0.0_DMU0.0_K1.0" / "_-1.7"
    run_dir.mkdir(parents=True)
    lineage = {"0": "", "1": "0", "2": "1", "3": "", "4": "", "5": "4"}
    with open(run_dir / "susceptibility_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUSCEPTIBILITY_CSV_FIELDS)
        w.writeheader()
        for rid, parent in lineage.items():
            w.writerow({k: "" for k in SUSCEPTIBILITY_CSV_FIELDS} | {"id": rid, "resume_id": parent})
    head = "chunk,m\n"
    files = {
        "0": head + "0,0.1\n",
        "1": head + "0,0.1\n1,0.2\n",
        "2": head + "0,0.1\n1,0.2\n2,0.3\n",
        "3": head + "0,0.5\n",
        "4": head + "0,0.7\n",
        "5": head + "0,0.8\n1,0.9\n",
    }
    for rid, text in files.items():
        (run_dir / f"m_timeseries_{rid}.csv").write_text(text)
    return run_dir


def _present(run_dir):
    return sorted(p.name.split("_")[-1][:-4] for p in run_dir.glob("m_timeseries_*.csv"))


@pytest.mark.parametrize("quick", [False, True])
def test_dry_run_deletes_nothing(tmp_path, quick):
    run_dir = _campaign(tmp_path)
    stats = prune.prune_run_dir(str(run_dir), quick=quick, apply=False)
    assert stats["verified"] == 2      # 0 and 1, both prefixes of newest (2)
    assert stats["mismatch"] == 1      # 4 is not a prefix of 5
    assert _present(run_dir) == ["0", "1", "2", "3", "4", "5"]


@pytest.mark.parametrize("quick", [False, True])
def test_apply_keeps_newest_fresh_and_mismatched(tmp_path, quick):
    run_dir = _campaign(tmp_path)
    prune.prune_run_dir(str(run_dir), quick=quick, apply=True)
    assert _present(run_dir) == ["2", "3", "4", "5"]

    again = prune.prune_run_dir(str(run_dir), quick=quick, apply=True)
    assert again["verified"] == 0
    assert _present(run_dir) == ["2", "3", "4", "5"]
