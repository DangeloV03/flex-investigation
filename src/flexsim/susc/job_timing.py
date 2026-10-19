"""Per-round wall-clock and Slurm efficiency reports for the smart campaign.

Sweep/top-up bash scripts append one row per (ε, L) to
``{results_base}/timing/{job_id}.csv``.  The check job then queries ``sacct``
/ ``seff`` for those job IDs and writes:

  timing_round_{N}.md   — tables
  timing_round_{N}.csv  — machine-readable per-L rows + job stats joined
  timing_round_{N}.png  — wall time vs L

Slurm queries (sacct / seff) are in susc/slurm_stats.py.
"""

from __future__ import annotations

import csv
import glob
import math
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

from flexsim.susc.slurm_stats import query_sacct_batch


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

    raw_ids = (
        [j for j in jobs["job_id"].astype(str).tolist() if not j.startswith("DRY")]
        if not jobs.empty
        else []
    )
    batch_stats = query_sacct_batch(raw_ids)
    job_stats: list[dict] = []
    for job_id in raw_ids:
        stats = batch_stats.get(job_id) or {"job_id": job_id}
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
