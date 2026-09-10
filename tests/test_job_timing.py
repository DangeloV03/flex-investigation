"""Tests for per-round wall-time / Slurm efficiency reports."""

from __future__ import annotations

import os
import subprocess

from job_timing import (
    _parse_sacct_output,
    _stats_from_sacct_rows,
    append_slurm_job,
    fmt_hms,
    load_l_timings,
    parse_hms,
    parse_seff,
    write_round_timing_report,
)


def test_parse_hms_formats():
    assert parse_hms("90") == 90.0
    assert parse_hms("01:02:03") == 3723.0
    assert parse_hms("1-00:00:00") == 86400.0
    assert parse_hms("") is None


def test_fmt_hms():
    assert fmt_hms(75) == "1m 15s"
    assert fmt_hms(3723) == "1h 02m 03s"


def test_parse_seff_extracts_efficiencies():
    text = """
Job ID: 12345
State: COMPLETED (exit code 0)
CPU Efficiency: 87.50% of 06:09:20 core-walltime
Job Wall-clock time: 00:23:05
Memory Efficiency: 15.38% of 8.00 GB
"""
    stats = parse_seff(text)
    assert stats["cpu_efficiency_pct"] == 87.5
    assert stats["mem_efficiency_pct"] == 15.38
    assert stats["elapsed_seconds"] == 23 * 60 + 5
    assert stats["state"] == "COMPLETED"


def test_sacct_stats_sum_srun_steps():
    """CPU time lives in the srun step rows; .batch alone reports ~0%."""
    out = "\n".join([
        "13632772|susc_top_L96|TIMEOUT|86400||16|1440||8192M|0:0",
        "13632772.batch|batch|CANCELLED|86400|00:05.200|16||12000K||0:15",
        "13632772.0|python|CANCELLED|86400|10-00:00:00|16||5263200K||0:15",
    ])
    stats = _stats_from_sacct_rows("13632772", _parse_sacct_output(out))

    assert stats["state"] == "TIMEOUT"
    assert stats["elapsed_seconds"] == 86400.0
    # 10 CPU-days (plus the batch shell's few seconds) over 16 cores x 24 h.
    assert abs(stats["cpu_efficiency_pct"] - 62.5) < 0.01
    assert stats["time_efficiency_pct"] == 100.0
    assert stats["mem_efficiency_pct"] is not None
    assert 60.0 < stats["mem_efficiency_pct"] < 65.0


def test_sacct_stats_without_steps_fall_back_to_alloc_row():
    out = "13632772|susc_top_L96|COMPLETED|3600|08:00:00|16|1440||8192M|0:0"
    stats = _stats_from_sacct_rows("13632772", _parse_sacct_output(out))
    assert stats["cpu_efficiency_pct"] == 50.0


def test_run_swallows_timeout(monkeypatch):
    import job_timing

    def boom(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="seff", timeout=8)

    monkeypatch.setattr(job_timing.subprocess, "run", boom)
    assert job_timing._run(["seff", "1"]) == ""


def test_write_round_timing_report(tmp_path, monkeypatch):
    import job_timing

    monkeypatch.setattr(job_timing, "query_sacct_batch", lambda _ids: {
        "999": {
            "job_id": "999",
            "state": "COMPLETED",
            "elapsed_seconds": 3600.0,
            "cpu_efficiency_pct": 91.0,
            "time_efficiency_pct": 4.2,
            "mem_efficiency_pct": 12.0,
        }
    })

    base = str(tmp_path / "SUSC_RUNS_S1A")
    append_slurm_job(
        base, round_num=1, phase="sweep", job_id="999", epsilon=-1.76, sizes=[16, 32]
    )
    timing_dir = tmp_path / "SUSC_RUNS_S1A" / "timing"
    timing_dir.mkdir(parents=True)
    (timing_dir / "999.csv").write_text(
        "phase,job_id,epsilon,L,wall_seconds,ncpus,finished_at\n"
        "sweep,999,-1.76,16,120,16,2026-09-02T12:00:00Z\n"
        "sweep,999,-1.76,32,400,16,2026-09-02T12:06:40Z\n",
        encoding="utf-8",
    )

    md = write_round_timing_report(base, 1)
    text = open(md, encoding="utf-8").read()
    assert "Round 1" in text
    assert "| 16 |" in text
    assert "91.0%" in text
    assert "4.2%" in text
    assert os.path.isfile(os.path.join(base, "timing_round_1.png"))
    assert os.path.isfile(os.path.join(base, "timing_round_1.csv"))

    loaded = load_l_timings(base)
    assert list(loaded["L"]) == [16, 32]
    assert list(loaded["wall_seconds"]) == [120, 400]
