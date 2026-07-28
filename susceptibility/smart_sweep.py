"""
smart_sweep.py

Orchestrator for the smart susceptibility campaign.

Subcommands
-----------
sweep   Submit initial epsilon sweep (10^6 eq + 10^6 prod per replica).
check   Post-sweep jump analysis; submits top-up jobs if needed; loops until
        every (ε, L) pair meets the ≥ threshold avg-jump requirement.

Typical usage
-------------
# Launch the campaign (from repo root):
python susceptibility/smart_sweep.py sweep \\
    --eps-min -2.0 --eps-max -1.4 --eps-step 0.005 \\
    --results-base SUSC_RUNS

# Run check manually (normally submitted automatically as a dependency):
python susceptibility/smart_sweep.py check --results-base SUSC_RUNS
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import sys
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Epsilon range helper (mirrors sweep_susceptibility.py)
# ---------------------------------------------------------------------------

def _frange(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range; round each value to avoid fp drift."""
    vals: list[float] = []
    n = int(round((stop - start) / step))
    for i in range(n + 1):
        v = round(start + i * step, 10)
        if v <= stop + 1e-12:
            vals.append(v)
    return vals


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _round_num(results_base: str) -> int:
    """Count existing round PNGs to determine the current round number."""
    existing = glob.glob(os.path.join(results_base, "jump_check_round_*.png"))
    return len(existing) + 1


def _write_report(
    summary: pd.DataFrame,
    results_base: str,
    *,
    round_num: int,
    threshold: float = 10.0,
) -> str:
    """Write jump_report.md and a round PNG figure; returns the md path."""
    os.makedirs(results_base, exist_ok=True)
    png_path = os.path.join(results_base, f"jump_check_round_{round_num}.png")
    md_path = os.path.join(results_base, "jump_report.md")

    # ---- figure ----
    L_vals = sorted(summary["L"].unique()) if not summary.empty else []
    ncols = min(4, len(L_vals)) if L_vals else 1
    nrows = max(1, math.ceil(len(L_vals) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.suptitle(
        f"Jump analysis — round {round_num}  (threshold J ≥ {threshold:.0f})",
        fontsize=11, fontweight="bold",
    )

    for ax_idx, L in enumerate(L_vals):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        sub = summary[summary["L"] == L].sort_values("epsilon")
        eps_arr = sub["epsilon"].to_numpy(float)
        j_arr = sub["J_mean"].to_numpy(float)
        if "J_stderr" in sub.columns:
            err_arr = sub["J_stderr"].fillna(0).to_numpy(float)
        elif "J_std" in sub.columns:
            err_arr = sub["J_std"].fillna(0).to_numpy(float)
        else:
            err_arr = np.zeros_like(j_arr)
        colors = ["tab:red" if not bool(p) else "tab:blue" for p in sub["passes"].tolist()]
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
                   label=f"threshold={threshold:.0f}")
        ax.errorbar(eps_arr, j_arr, yerr=err_arr, fmt="none", ecolor="gray",
                    capsize=2, linewidth=0.8, alpha=0.7)
        ax.scatter(eps_arr, j_arr, c=colors, s=28, zorder=3)
        ax.set_title(f"L = {L}", fontsize=9)
        ax.set_xlabel(r"$\varepsilon$", fontsize=8)
        ax.set_ylabel(r"$\langle J \rangle$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    # Hide unused axes
    for ax_idx in range(len(L_vals), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- markdown ----
    n_total = len(summary) if not summary.empty else 0
    n_pass = int(summary["passes"].sum()) if not summary.empty else 0
    n_fail = n_total - n_pass

    lines = [
        f"# Jump Analysis Report — Round {round_num}",
        "",
        f"**Date:** {date.today().isoformat()}  ",
        f"**Results base:** `{results_base}`  ",
        f"**Threshold:** J ≥ {threshold:.0f}  ",
        f"**Status:** {n_pass}/{n_total} pairs passing — "
        + (f"**{n_fail} pairs queued for top-up**" if n_fail > 0 else "✓ All pairs pass"),
        "",
        f"![Round {round_num} figure]({os.path.basename(png_path)})",
        "",
        "## Results table",
        "",
        "| L | ε | ⟨J⟩ | stderr | n_replicas | status |",
        "|---|---|-----|--------|------------|--------|",
    ]

    if not summary.empty:
        for _, row in summary.sort_values(["L", "epsilon"]).iterrows():
            status = "✓" if bool(row["passes"]) else "**RERUN**"
            j_str = f"{row['J_mean']:.2f}"
            err_val = row.get("J_stderr", float("nan"))
            err_f = float(err_val) if err_val is not None else float("nan")
            err_str = f"{err_f:.2f}" if math.isfinite(err_f) else "—"
            lines.append(
                f"| {int(row['L'])} | {row['epsilon']:.4f} | {j_str} | {err_str} "
                f"| {int(row['n_replicas'])} | {status} |"
            )
    else:
        lines.append("| — | — | — | — | — | no data |")

    lines += ["", "---", f"*Generated by smart_sweep.py*"]

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[check] Round {round_num}: {n_pass}/{n_total} passing → {md_path}", flush=True)
    return md_path


# ---------------------------------------------------------------------------
# sweep subcommand
# ---------------------------------------------------------------------------

def do_sweep(args: argparse.Namespace) -> None:
    eps_values = _frange(args.eps_min, args.eps_max, args.eps_step)
    if not eps_values:
        raise SystemExit("No epsilon values generated; check --eps-min/max/step.")

    results_base = args.results_base
    os.makedirs(results_base, exist_ok=True)
    os.makedirs("slurm_reports", exist_ok=True)

    sweep_script = args.sweep_script
    check_script = args.check_script

    print(
        f"[sweep] Submitting {len(eps_values)} epsilon jobs → {results_base}",
        flush=True,
    )

    extra: list[str] = []
    if args.mu_source:
        # Pre-fetch mu map and pass mu values per epsilon if available.
        # (Falls through to runner default mu=2ε if manage CSV absent.)
        try:
            import csv as _csv
            mu_map: dict[float, float] = {}
            with open(args.mu_source, newline="") as f:
                for row in _csv.DictReader(f):
                    try:
                        mu_map[float(row["epsilon"])] = float(row["mu_coex_FITTED"])
                    except (KeyError, ValueError):
                        pass
        except FileNotFoundError:
            mu_map = {}
    else:
        mu_map = {}

    if args.delta_f:
        extra += [args.delta_f]       # $5
    if args.delta_mu:
        extra += [args.delta_mu]      # $6
    if args.k:
        extra += [args.k]             # $7
    if args.scheme:
        extra += [args.scheme]        # $8

    sweep_job_ids: list[str] = []
    for eps in eps_values:
        mu_arg = str(mu_map.get(eps, "")) if mu_map else ""
        cmd = [
            "sbatch", "--parsable",
            sweep_script,
            f"{eps:.6g}",       # $1 epsilon
            results_base,       # $2 results_base
            "1",                # $3 num_batches
            mu_arg,             # $4 mu (empty = runner default)
            *extra,
        ]
        if args.dry_run:
            print(f"  [DRY-RUN] {' '.join(cmd)}", flush=True)
            sweep_job_ids.append(f"DRY{eps}")
            continue
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split(";")[0]  # --parsable may add cluster name
        sweep_job_ids.append(job_id)
        print(f"  Submitted job {job_id} for ε={eps:.4f}", flush=True)

    if args.dry_run:
        print(f"[sweep] DRY-RUN: would submit check job after {len(sweep_job_ids)} sweep jobs")
        return

    # Chain check job as a dependency on all sweep jobs.
    dep = "--dependency=afterok:" + ":".join(sweep_job_ids)
    check_cmd = [
        "sbatch", dep,
        check_script,
        results_base,
        str(args.threshold),
        check_script,
    ]
    result = subprocess.run(check_cmd, capture_output=True, text=True, check=True)
    check_job_id = result.stdout.strip().split(";")[0]
    print(
        f"[sweep] Submitted check job {check_job_id} "
        f"(dependency: afterok:{':'.join(sweep_job_ids)})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# check subcommand
# ---------------------------------------------------------------------------

def do_check(args: argparse.Namespace) -> None:
    from count_mag_jumps import compute_jump_summary

    results_base = args.results_base
    threshold = args.threshold
    check_script = args.check_script
    topup_script = args.topup_script

    print(f"[check] Loading jump data from {results_base} …", flush=True)
    summary = compute_jump_summary(results_base, threshold=threshold)

    round_num = _round_num(results_base)
    _write_report(summary, results_base, round_num=round_num, threshold=threshold)

    if summary.empty:
        print("[check] No data found — nothing to evaluate.", flush=True)
        return

    failing = summary[~summary["passes"]]
    if failing.empty:
        print("[check] All (ε, L) pairs meet the threshold. Campaign complete!", flush=True)
        return

    # Group failing pairs by epsilon; pass the failing L sizes to the topup job.
    failing_by_eps: dict[float, list[int]] = {}
    for _, row in failing.iterrows():
        eps = float(row["epsilon"])
        L = int(row["L"])
        failing_by_eps.setdefault(eps, []).append(L)

    print(
        f"[check] {len(failing)} pairs failing; submitting top-up for "
        f"{len(failing_by_eps)} epsilon values …",
        flush=True,
    )
    os.makedirs("slurm_reports", exist_ok=True)

    topup_job_ids: list[str] = []
    for eps in sorted(failing_by_eps):
        sizes = sorted(failing_by_eps[eps])
        size_strs = [str(s) for s in sizes]
        cmd = [
            "sbatch", "--parsable",
            topup_script,
            f"{eps:.6g}",
            results_base,
            *size_strs,
        ]
        if args.dry_run:
            print(f"  [DRY-RUN] {' '.join(cmd)}", flush=True)
            topup_job_ids.append(f"DRY{eps}")
            continue
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split(";")[0]
        topup_job_ids.append(job_id)
        print(
            f"  Submitted top-up job {job_id} for ε={eps:.4f} "
            f"L={sizes}",
            flush=True,
        )

    if args.dry_run:
        print(f"[check] DRY-RUN: would submit next check after {len(topup_job_ids)} top-up jobs")
        return

    # Self-schedule next check after all top-up jobs complete.
    dep = "--dependency=afterok:" + ":".join(topup_job_ids)
    next_check_cmd = [
        "sbatch", dep,
        check_script,
        results_base,
        str(threshold),
        check_script,
    ]
    result = subprocess.run(next_check_cmd, capture_output=True, text=True, check=True)
    next_id = result.stdout.strip().split(";")[0]
    print(
        f"[check] Submitted next check job {next_id} "
        f"(dependency: afterok:{':'.join(topup_job_ids)})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Smart susceptibility campaign orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # -- sweep --
    sp = sub.add_parser("sweep", help="Submit initial epsilon sweep")
    sp.add_argument("--eps-min", type=float, required=True)
    sp.add_argument("--eps-max", type=float, required=True)
    sp.add_argument("--eps-step", type=float, required=True)
    sp.add_argument("--results-base", default="SUSC_RUNS",
                    help="Root output directory (default: SUSC_RUNS)")
    sp.add_argument("--sweep-script", default="susceptibility/run_susceptibility_smart.sh")
    sp.add_argument("--check-script", default="susceptibility/run_smart_check.sh")
    sp.add_argument("--threshold", type=float, default=10.0,
                    help="Minimum avg jumps per (ε, L) pair (default: 10)")
    sp.add_argument("--mu-source", default=None,
                    help="manage.csv path for fitted μ values (optional)")
    sp.add_argument("--delta-f", default="", help="δf override (empty = runner default)")
    sp.add_argument("--delta-mu", default="", help="δμ override (empty = runner default)")
    sp.add_argument("--k", default="", help="k override (empty = runner default)")
    sp.add_argument("--scheme", default="", help="Scheme override (empty = runner default)")
    sp.add_argument("--dry-run", action="store_true",
                    help="Print sbatch commands without submitting")

    # -- check --
    cp = sub.add_parser("check", help="Analyse jumps and chain top-up if needed")
    cp.add_argument("--results-base", required=True,
                    help="SUSC_RUNS directory to analyse")
    cp.add_argument("--threshold", type=float, default=10.0)
    cp.add_argument("--check-script", default="susceptibility/run_smart_check.sh")
    cp.add_argument("--topup-script", default="susceptibility/run_susceptibility_topup.sh")
    cp.add_argument("--dry-run", action="store_true",
                    help="Print sbatch commands without submitting")

    args = p.parse_args()

    if args.command == "sweep":
        do_sweep(args)
    elif args.command == "check":
        do_check(args)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
