"""
plot_jump_report.py

Redraw the smart-sweep jump figure for every check round of a campaign, without
running `smart_sweep.py check` (no top-ups submitted, round count and stall
tracking untouched).

Per L, three stacked panels vs ε:
    ⟨J⟩               mean jumps per replica (threshold line)
    # replicas J > 10  against the replica count (gray dashed)
    max J             largest jump count over the replicas

Reconstructing past rounds
--------------------------
Round N's cutoff is the mtime of jump_check_round_N.png (written by check N right
after it loaded the data; top-ups only start after that).  A generation (CSV row)
counts as available at round N if its final_lattice_{id}.npy — written when the
generation finished, and never removed by prune_superseded_timeseries.py — is
older than that cutoff.  As in the check, only the available rows at the newest
prod_time are used.

Each top-up writes the replica's full history, so a generation's m series is the
first prod_chunks entries of its own timeseries file or of any descendant's
(following resume_id links) — this still works after superseded files are pruned.

A final "current" figure uses everything on disk now (incl. top-ups that finished
after the last check).

Usage (from repo root):
    python susceptibility/plot_jump_report.py --results-base SUSC_RUNS_S1B_IMPROVED_MU
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "coex")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

from count_mag_jumps import analyze_jumps
from smart_sweep import plot_jump_figure
from susceptibility_paths import find_susc_run_csvs, parse_susc_run_dir, read_susceptibility_csv

CURRENT = "current"


def round_cutoffs(results_base: str) -> dict[int, float]:
    """{round N: mtime of jump_check_round_N.png}."""
    cutoffs: dict[int, float] = {}
    for path in glob.glob(os.path.join(results_base, "jump_check_round_*.png")):
        m = re.search(r"jump_check_round_(\d+)\.png$", path)
        if m:
            cutoffs[int(m.group(1))] = os.path.getmtime(path)
    return dict(sorted(cutoffs.items()))


def _finished_at(run_dir: str, run_id: str) -> float | None:
    for name in (f"final_lattice_{run_id}.npy", f"m_timeseries_{run_id}.csv"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return os.path.getmtime(path)
    return None


def _timeseries_source(run_dir: str, run_id: str, children: dict[str, list[str]]) -> str | None:
    """This id's timeseries file, else the first existing descendant's (a superset)."""
    stack, seen = [run_id], set()
    while stack:
        rid = stack.pop()
        if rid in seen:
            continue
        seen.add(rid)
        path = os.path.join(run_dir, f"m_timeseries_{rid}.csv")
        if os.path.isfile(path):
            return path
        stack.extend(children.get(rid, []))
    return None


def _latest(rows: list[dict]) -> list[dict]:
    """Rows at the newest prod_time — same rule as load_susc_runs_groups."""
    if not rows:
        return []
    max_pt = max(float(r.get("prod_time", 0) or 0) for r in rows)
    return [r for r in rows if abs(float(r.get("prod_time", 0) or 0) - max_pt) < 1.0]


def load_groups_by_round(
    results_base: str, cutoffs: dict[int, float]
) -> tuple[dict, int]:
    """{round or CURRENT: {(L, ε): {"replicas": [m arrays]}}} and #rows skipped.

    Arrays for earlier rounds are views into the one loaded file per chain.
    """
    labels = [*cutoffs, CURRENT]
    groups: dict = {lab: {} for lab in labels}
    cache: dict[str, np.ndarray] = {}
    n_skipped = 0

    csv_paths = find_susc_run_csvs(results_base)
    print(f"found {len(csv_paths)} run dirs; reading timeseries …", flush=True)
    for i, data_csv in enumerate(csv_paths, start=1):
        if i == 1 or i % 25 == 0 or i == len(csv_paths):
            print(f"  loading {i}/{len(csv_paths)} …", flush=True)
        run_dir = os.path.dirname(data_csv)
        L, eps = parse_susc_run_dir(run_dir)
        if L is None or eps is None:
            continue
        rows = [r for r in read_susceptibility_csv(data_csv) if r.get("id")]
        if not rows:
            continue

        children: dict[str, list[str]] = {}
        for r in rows:
            parent = (r.get("resume_id") or "").strip()
            if parent:
                children.setdefault(parent, []).append(str(r["id"]))
        # Unknown finish time → never counted for a past round.
        finished = {str(r["id"]): _finished_at(run_dir, str(r["id"])) or float("inf")
                    for r in rows}

        for lab in labels:
            if lab == CURRENT:
                avail = rows
            else:
                cut = cutoffs[lab]
                avail = [r for r in rows if finished[str(r["id"])] <= cut]
            reps: list[np.ndarray] = []
            for r in _latest(avail):
                rid = str(r["id"])
                src = _timeseries_source(run_dir, rid, children)
                if src is None:
                    n_skipped += 1
                    continue
                if src not in cache:
                    try:
                        cache[src] = pd.read_csv(src, usecols=["m"])["m"].to_numpy(dtype=float)
                    except (ValueError, KeyError, OSError):
                        cache[src] = np.empty(0)
                n_chunks = int(float(r.get("prod_chunks", 0) or 0))
                m_arr = cache[src][:n_chunks] if n_chunks > 0 else cache[src]
                if m_arr.size > 0:
                    reps.append(m_arr)
            if reps:
                groups[lab][(L, round(eps, 8))] = {"replicas": reps}
    return groups, n_skipped


def summarize(groups: dict, threshold: float) -> pd.DataFrame:
    summary, _, _, _ = analyze_jumps(groups, eps_crit=0.0, eps_window=0.0,
                                     epsilon=None, frac=0.75, all_eps=True)
    if summary.empty:
        return summary
    summary["passes"] = summary["J_mean"] >= threshold
    return summary.sort_values(["L", "epsilon"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-base", required=True, help="SUSC_RUNS_* campaign directory")
    p.add_argument("--threshold", type=float, default=10.0,
                   help="Pass threshold on ⟨J⟩ (default: 10)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: <results-base>/jump_panels)")
    args = p.parse_args()

    cutoffs = round_cutoffs(args.results_base)
    print(f"check rounds found: {list(cutoffs) or 'none'}")
    groups, n_skipped = load_groups_by_round(args.results_base, cutoffs)
    if n_skipped:
        print(f"warning: {n_skipped} (row, round) entries had no timeseries on disk; skipped",
              file=sys.stderr)

    outdir = args.outdir or os.path.join(args.results_base, "jump_panels")
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(os.path.normpath(args.results_base))
    tables = []
    for lab, g in groups.items():
        summary = summarize(g, args.threshold)
        if summary.empty:
            print(f"round {lab}: no data")
            continue
        tag = f"round_{lab}" if lab != CURRENT else CURRENT
        head = f"round {lab}" if lab != CURRENT else "current (all data on disk)"
        png = os.path.join(outdir, f"jump_panels_{tag}.png")
        plot_jump_figure(summary, png, threshold=args.threshold,
                         title=f"Jump analysis — {name}, {head}  "
                               f"(threshold ⟨J⟩ ≥ {args.threshold:.0f})")
        n_pass = int(summary["passes"].sum())
        print(f"{head}: {n_pass}/{len(summary)} pairs pass → {png}")
        tables.append(summary.assign(round=str(lab)))

    if not tables:
        print(f"No jump data under {args.results_base}", file=sys.stderr)
        return 1
    csv_path = os.path.join(outdir, "jump_summary_by_round.csv")
    pd.concat(tables, ignore_index=True).to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
