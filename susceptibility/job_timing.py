"""Per-round wall-clock and Slurm efficiency reports for the smart campaign.

Sweep/top-up bash scripts append one row per (ε, L) to
``{results_base}/timing/{job_id}.csv``.  The check job then queries ``sacct``
/ ``seff`` for those job IDs and writes:

  timing_round_{N}.md   — tables
  timing_round_{N}.csv  — machine-readable per-L rows + job stats joined
  timing_round_{N}.png  — wall time vs L
"""

from __future__ import annotations

import csv
import glob
import math
import os
import re
import subprocess
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TIMING_SUBDIR = "timing"
JOBS_CSV = "slurm_jobs.csv"
TIMING_FIELDS = [
    "phase",
    "job_id",
    "epsilon",
    "L",
    "wall_seconds",
    "ncpus",
    "finished_at",
]
JOB_FIELDS = ["round", "phase", "job_id", "epsilon", "sizes", "submitted_at"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_hms(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "—"
    s = int(round(float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


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


def append_slurm_job(
    results_base: str,
    *,
    round_num: int,
    phase: str,
    job_id: str,
    epsilon: float,
    sizes: list[int],
) -> None:
    os.makedirs(results_base, exist_ok=True)
    path = os.path.join(results_base, JOBS_CSV)
    new_file = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOB_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(
            {
                "round": str(round_num),
                "phase": phase,
                "job_id": str(job_id),
                "epsilon": f"{epsilon:.6g}",
                "sizes": " ".join(str(s) for s in sizes),
                "submitted_at": _utc_now(),
            }
        )


def load_slurm_jobs(results_base: str, round_num: int | None = None) -> pd.DataFrame:
    path = os.path.join(results_base, JOBS_CSV)
    if not os.path.isfile(path):
        return pd.DataFrame(columns=JOB_FIELDS)
    df = pd.read_csv(path)
    if round_num is not None and not df.empty and "round" in df.columns:
        df = df[df["round"].astype(int) == int(round_num)]
    return df.reset_index(drop=True)


def load_l_timings(results_base: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(results_base, TIMING_SUBDIR, "*.csv")))
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except (pd.errors.EmptyDataError, OSError):
            continue
    if not frames:
        return pd.DataFrame(columns=TIMING_FIELDS)
    df = pd.concat(frames, ignore_index=True)
    if "L" in df.columns:
        df["L"] = pd.to_numeric(df["L"], errors="coerce")
    if "epsilon" in df.columns:
        df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    if "wall_seconds" in df.columns:
        df["wall_seconds"] = pd.to_numeric(df["wall_seconds"], errors="coerce")
    if "ncpus" in df.columns:
        df["ncpus"] = pd.to_numeric(df["ncpus"], errors="coerce")
    if "job_id" in df.columns:
        df["job_id"] = df["job_id"].astype(str)
    return df


def _run(cmd: list[str], timeout: int = 30) -> str:
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def _pick_sacct_row(rows: list[dict]) -> dict:
    """Prefer the .batch step (actual work); else the allocation line."""
    if not rows:
        return {}
    for row in rows:
        jid = str(row.get("JobID", ""))
        if jid.endswith(".batch"):
            return row
    for row in rows:
        jid = str(row.get("JobID", ""))
        if "." not in jid:
            return row
    return rows[0]


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
            "--format=JobID,JobName,State,ElapsedRaw,TotalCPU,AllocCPUS,TimelimitRaw,MaxRSS,ReqMem,ExitCode",
        ]
    )
    if not out.strip():
        return {}
    fields = [
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
    rows: list[dict] = []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        rows.append(dict(zip(fields, parts)))
    row = _pick_sacct_row(rows)
    if not row:
        return {}

    elapsed = parse_hms(row.get("ElapsedRaw", ""))
    total_cpu = parse_hms(row.get("TotalCPU", ""))
    try:
        alloc = float(row.get("AllocCPUS") or "nan")
    except ValueError:
        alloc = float("nan")
    try:
        tlim_min = float(row.get("TimelimitRaw") or "nan")
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

    return {
        "job_id": str(job_id),
        "state": (row.get("State") or "").split()[0],
        "elapsed_seconds": elapsed,
        "total_cpu_seconds": total_cpu,
        "alloc_cpus": alloc if math.isfinite(alloc) else None,
        "timelimit_seconds": tlim,
        "cpu_efficiency_pct": cpu_eff,
        "time_efficiency_pct": time_eff,
        "max_rss": row.get("MaxRSS") or "",
        "req_mem": row.get("ReqMem") or "",
        "exit_code": row.get("ExitCode") or "",
    }


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
    """Merge sacct + seff. seff wins on CPU/mem efficiency when present."""
    stats = {"job_id": str(job_id)}
    sacct = query_sacct(job_id)
    seff = query_seff(job_id)
    stats.update({k: v for k, v in sacct.items() if v not in (None, "")})
    stats.update({k: v for k, v in seff.items() if v not in (None, "")})
    return stats


def _l_summary(timing: pd.DataFrame) -> pd.DataFrame:
    if timing.empty:
        return pd.DataFrame()
    g = timing.groupby("L")["wall_seconds"]
    out = g.agg(
        n="count",
        mean_s="mean",
        median_s="median",
        min_s="min",
        max_s="max",
        total_s="sum",
    ).reset_index()
    return out.sort_values("L")


def _write_png(timing: pd.DataFrame, png_path: str, round_num: int) -> None:
    if timing.empty:
        return
    L_vals = sorted(timing["L"].dropna().unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    data = [timing.loc[timing["L"] == L, "wall_seconds"].to_numpy(float) / 60.0 for L in L_vals]
    labels = [str(int(L)) for L in L_vals]
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=True)
    means = [float(d.mean()) if len(d) else float("nan") for d in data]
    ax.plot(range(1, len(L_vals) + 1), means, "o", color="tab:red", label="mean")
    ax.set_xlabel("L")
    ax.set_ylabel("Wall time (minutes)")
    ax.set_title(f"Real wall time per L — round {round_num}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _pct_cell(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "—"


def write_round_timing_report(results_base: str, round_num: int) -> str:
    """Write timing_round_{N}.{md,csv,png}. Returns the markdown path."""
    os.makedirs(results_base, exist_ok=True)
    jobs = load_slurm_jobs(results_base, round_num=round_num)
    timing = load_l_timings(results_base)
    if not jobs.empty and not timing.empty:
        job_ids = set(jobs["job_id"].astype(str))
        timing = timing[timing["job_id"].astype(str).isin(job_ids)].copy()

    job_stats: list[dict] = []
    for job_id in jobs["job_id"].astype(str).tolist() if not jobs.empty else []:
        if job_id.startswith("DRY"):
            continue
        stats = query_job_stats(job_id)
        match = jobs.loc[jobs["job_id"].astype(str) == job_id]
        if not match.empty:
            stats["epsilon"] = match.iloc[0].get("epsilon", "")
            stats["phase"] = match.iloc[0].get("phase", "")
            stats["sizes"] = match.iloc[0].get("sizes", "")
        job_stats.append(stats)

    png_path = os.path.join(results_base, f"timing_round_{round_num}.png")
    md_path = os.path.join(results_base, f"timing_round_{round_num}.md")
    csv_path = os.path.join(results_base, f"timing_round_{round_num}.csv")
    _write_png(timing, png_path, round_num)

    if not timing.empty:
        timing.to_csv(csv_path, index=False)

    l_tbl = _l_summary(timing)
    phase = "mixed"
    if not jobs.empty:
        phases = sorted({str(p) for p in jobs["phase"].dropna().unique()})
        phase = phases[0] if len(phases) == 1 else "+".join(phases)

    cpu_vals = [s.get("cpu_efficiency_pct") for s in job_stats if s.get("cpu_efficiency_pct") is not None]
    time_vals = [s.get("time_efficiency_pct") for s in job_stats if s.get("time_efficiency_pct") is not None]
    mem_vals = [s.get("mem_efficiency_pct") for s in job_stats if s.get("mem_efficiency_pct") is not None]

    lines = [
        f"# Timing report — Round {round_num}",
        "",
        f"**Phase:** {phase}  ",
        f"**Jobs:** {len(jobs)}  ",
        f"**Dated:** {_utc_now()}  ",
        "",
        f"![Wall time vs L]({os.path.basename(png_path)})" if not timing.empty else "_No per-L wall-time rows yet._",
        "",
        "## Wall time by system size (real time)",
        "",
        "Each value is wall-clock time for one (ε, L) inside a Slurm job "
        "(16 replicas in parallel). Mean/median are over ε in this round.",
        "",
        "| L | n | mean | median | min | max | total |",
        "|---|---|------|--------|-----|-----|-------|",
    ]
    if l_tbl.empty:
        lines.append("| — | — | — | — | — | — | — |")
    else:
        for _, row in l_tbl.iterrows():
            lines.append(
                f"| {int(row['L'])} | {int(row['n'])} | {fmt_hms(row['mean_s'])} "
                f"| {fmt_hms(row['median_s'])} | {fmt_hms(row['min_s'])} "
                f"| {fmt_hms(row['max_s'])} | {fmt_hms(row['total_s'])} |"
            )

    lines += [
        "",
        "## Slurm job efficiency",
        "",
        "- **CPU efficiency** — percent of allocated core-seconds actually used "
        "(`seff` / `TotalCPU / (Elapsed × AllocCPUS)`). 100% means all 16 cores stayed busy.",
        "- **Time efficiency** — `Elapsed / requested walltime` (how much of the 24 h request was used).",
        "- **Memory efficiency** — peak RSS vs requested memory (`seff`).",
        "",
        f"Round averages: CPU {_pct_cell(sum(cpu_vals)/len(cpu_vals) if cpu_vals else None)} · "
        f"time {_pct_cell(sum(time_vals)/len(time_vals) if time_vals else None)} · "
        f"memory {_pct_cell(sum(mem_vals)/len(mem_vals) if mem_vals else None)}",
        "",
        "| job | ε | state | elapsed | CPU % | time % | mem % |",
        "|-----|---|-------|---------|-------|--------|-------|",
    ]
    if not job_stats:
        lines.append("| — | — | — | — | — | — | — |")
    else:
        for s in job_stats:
            lines.append(
                f"| {s.get('job_id', '')} | {s.get('epsilon', '')} "
                f"| {s.get('state', '—')} | {fmt_hms(s.get('elapsed_seconds'))} "
                f"| {_pct_cell(s.get('cpu_efficiency_pct'))} "
                f"| {_pct_cell(s.get('time_efficiency_pct'))} "
                f"| {_pct_cell(s.get('mem_efficiency_pct'))} |"
            )

    lines += ["", "---", "*Generated by job_timing.py during the Della check job.*", ""]
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[timing] Round {round_num}: wrote {md_path}", flush=True)
    return md_path
