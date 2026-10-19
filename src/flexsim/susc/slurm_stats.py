"""Query sacct / seff for a job's state, wall time and CPU / memory efficiency. Never raises."""

from __future__ import annotations

import math
import re
import subprocess


def parse_hms(value: str) -> float | None:
    """Parse Slurm [[D-]HH:]MM:SS (or a plain integer/float of seconds)."""
    raw = (value or "").strip()
    if not raw or raw in {".", "Unknown", "INVALID"}:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    days = 0
    if "-" in raw:
        day_part, raw = raw.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = raw.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, s = nums
    elif len(nums) == 2:
        h, m, s = 0.0, nums[0], nums[1]
    elif len(nums) == 1:
        h, m, s = 0.0, 0.0, nums[0]
    else:
        return None
    return days * 86400.0 + h * 3600.0 + m * 60.0 + s


def _run(cmd: list[str], timeout: int = 8) -> str:
    """Run a Slurm helper. Never raise: a hung sacct/seff must not kill the campaign."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def _parse_mem(value: str) -> float | None:
    """Parse a Slurm memory string ('5263200K', '8G', '512Mc') to bytes."""
    raw = (value or "").strip()
    if not raw:
        return None
    m = re.match(r"^([0-9.]+)\s*([KMGTP]?)", raw)
    if not m:
        return None
    mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3,
            "T": 1024 ** 4, "P": 1024 ** 5}[m.group(2)]
    return float(m.group(1)) * mult


_SACCT_FIELDS = [
    "JobID",
    "JobName",
    "State",
    "ElapsedRaw",
    "TotalCPU",
    "AllocCPUS",
    "TimelimitRaw",
    "MaxRSS",
    "ReqMem",
    "ExitCode",
]
_SACCT_FORMAT = ",".join(_SACCT_FIELDS)


def _parse_sacct_output(out: str) -> list[dict]:
    rows: list[dict] = []
    for line in (out or "").strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        rows.append(dict(zip(_SACCT_FIELDS, parts)))
    return rows


def _stats_from_sacct_rows(job_id: str, rows: list[dict]) -> dict:
    """Job stats from every sacct line for one job.

    The work runs under srun, so its CPU time is accounted to the .0/.1 step
    rows — the .batch row holds only the wrapper shell.  Reading .batch alone
    reported ~0% CPU efficiency for every job, which is why a round running at
    a fraction of its cores went unnoticed until it blew the wall.  Elapsed,
    AllocCPUS and Timelimit come from the allocation line (steps lack them);
    TotalCPU is summed and MaxRSS maxed across the steps.
    """
    if not rows:
        return {"job_id": str(job_id)}

    alloc_row = next(
        (r for r in rows if "." not in str(r.get("JobID", ""))), rows[0]
    )
    step_rows = [r for r in rows if "." in str(r.get("JobID", ""))]

    elapsed = parse_hms(alloc_row.get("ElapsedRaw", ""))

    step_cpu = [parse_hms(r.get("TotalCPU", "")) for r in step_rows]
    step_cpu = [c for c in step_cpu if c is not None]
    total_cpu = sum(step_cpu) if step_cpu else parse_hms(alloc_row.get("TotalCPU", ""))

    try:
        alloc = float(alloc_row.get("AllocCPUS") or "nan")
    except ValueError:
        alloc = float("nan")
    try:
        tlim_min = float(alloc_row.get("TimelimitRaw") or "nan")
        tlim = tlim_min * 60.0 if math.isfinite(tlim_min) else None
    except ValueError:
        tlim = None

    cpu_eff = None
    if (
        elapsed
        and total_cpu is not None
        and math.isfinite(alloc)
        and elapsed > 0
        and alloc > 0
    ):
        cpu_eff = 100.0 * total_cpu / (elapsed * alloc)
    time_eff = None
    if elapsed and tlim and tlim > 0:
        time_eff = 100.0 * elapsed / tlim

    rss_vals = [_parse_mem(r.get("MaxRSS", "")) for r in step_rows] or [
        _parse_mem(alloc_row.get("MaxRSS", ""))
    ]
    rss_vals = [v for v in rss_vals if v]
    max_rss = max(rss_vals) if rss_vals else None

    req_raw = (alloc_row.get("ReqMem") or "").strip()
    req_bytes = _parse_mem(req_raw)
    # A trailing 'c' means the request was per allocated CPU.
    if req_bytes and req_raw.endswith("c") and math.isfinite(alloc):
        req_bytes *= alloc
    mem_eff = (
        100.0 * max_rss / req_bytes if (max_rss and req_bytes and req_bytes > 0) else None
    )

    return {
        "job_id": str(job_id),
        "state": (alloc_row.get("State") or "").split()[0],
        "elapsed_seconds": elapsed,
        "total_cpu_seconds": total_cpu,
        "alloc_cpus": alloc if math.isfinite(alloc) else None,
        "timelimit_seconds": tlim,
        "cpu_efficiency_pct": cpu_eff,
        "time_efficiency_pct": time_eff,
        "mem_efficiency_pct": mem_eff,
        "max_rss": alloc_row.get("MaxRSS") or (step_rows[0].get("MaxRSS") if step_rows else ""),
        "req_mem": req_raw,
        "exit_code": alloc_row.get("ExitCode") or "",
    }


def query_sacct(job_id: str) -> dict:
    """Job-level elapsed / CPU / memory from sacct. Empty dict if unavailable."""
    out = _run(
        [
            "sacct",
            "-j",
            str(job_id),
            "--parsable2",
            "--noheader",
            "--noconvert",
            f"--format={_SACCT_FORMAT}",
        ]
    )
    if not out.strip():
        return {}
    return _stats_from_sacct_rows(str(job_id), _parse_sacct_output(out))


def query_sacct_batch(job_ids: list[str]) -> dict[str, dict]:
    """One sacct call for many jobs. Keys are root job IDs (no .batch suffix)."""
    ids = [str(j) for j in job_ids if j and not str(j).startswith("DRY")]
    if not ids:
        return {}
    out = _run(
        [
            "sacct",
            "-j",
            ",".join(ids),
            "--parsable2",
            "--noheader",
            "--noconvert",
            f"--format={_SACCT_FORMAT}",
        ],
        timeout=20,
    )
    rows_by_root: dict[str, list[dict]] = {j: [] for j in ids}
    for row in _parse_sacct_output(out):
        root = str(row.get("JobID", "")).split(".")[0]
        if root in rows_by_root:
            rows_by_root[root].append(row)
    return {jid: _stats_from_sacct_rows(jid, rows) for jid, rows in rows_by_root.items()}


def parse_seff(text: str) -> dict:
    """Extract CPU / memory efficiency from `seff` stdout."""
    stats: dict = {}

    def _pct(label: str) -> float | None:
        m = re.search(rf"{label}:\s*([0-9.]+)\s*%", text)
        return float(m.group(1)) if m else None

    cpu = _pct("CPU Efficiency")
    mem = _pct("Memory Efficiency")
    if cpu is not None:
        stats["cpu_efficiency_pct"] = cpu
    if mem is not None:
        stats["mem_efficiency_pct"] = mem
    m = re.search(r"Job Wall-clock time:\s*(\S+)", text)
    if m:
        wall = parse_hms(m.group(1))
        if wall is not None:
            stats["elapsed_seconds"] = wall
    m = re.search(r"State:\s*(\S+)", text)
    if m:
        stats["state"] = m.group(1)
    return stats


def query_seff(job_id: str) -> dict:
    out = _run(["seff", str(job_id)])
    if not out.strip():
        return {}
    stats = parse_seff(out)
    stats["job_id"] = str(job_id)
    return stats


def query_job_stats(job_id: str) -> dict:
    """Sacct only. Do not call seff here — it can hang on Della compute nodes."""
    stats = {"job_id": str(job_id)}
    sacct = query_sacct(job_id)
    stats.update({k: v for k, v in sacct.items() if v not in (None, "")})
    return stats
