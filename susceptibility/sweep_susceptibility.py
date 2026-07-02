"""
sweep_susceptibility.py

Single-file driver for the exact-mu (mu = 2*epsilon) susceptibility campaign.

Sweeps epsilon over [eps-min, eps-max] and submits ONE sbatch job per epsilon
(run_susceptibility.sh), each of which loops L = 16 .. 256 internally on
SLURM_CPUS_PER_TASK parallel replicas. Results land in a dated, self-contained
folder; re-running (or --num-batches > 1) appends more replicas per (L, eps).

Usage:
    python sweep_susceptibility.py                       # submit full sweep
    python sweep_susceptibility.py --dry-run             # print sbatch cmds only
    python sweep_susceptibility.py --local --eps-min -1.8 --eps-max -1.8
    python sweep_susceptibility.py --num-batches 2       # 2 batches per job
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys

from generate_samples import frange

EPS_MIN = -2.0
EPS_MAX = -1.4
EPS_STEP = 0.005

DEFAULT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_susceptibility.sh")
RESULTS_ROOT = "susceptibility_results"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep epsilon and submit one sbatch per epsilon (exact mu = 2*epsilon)."
    )
    parser.add_argument("--eps-min", type=float, default=EPS_MIN)
    parser.add_argument("--eps-max", type=float, default=EPS_MAX)
    parser.add_argument("--eps-step", type=float, default=EPS_STEP)
    parser.add_argument("--num-batches", type=int, default=1, help="Batches per job (each appends)")
    parser.add_argument("--script", default=DEFAULT_SCRIPT, help="Path to run_susceptibility.sh")
    parser.add_argument("--label", default="exact", help="Prefix for the dated results folder")
    parser.add_argument(
        "--results-base",
        default=None,
        help="Override results folder (default: {RESULTS_ROOT}/{label}_{YYYY-MM-DD})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run run_susceptibility.sh directly (bash) instead of sbatch, for off-SLURM testing",
    )
    args = parser.parse_args()

    eps_values = frange(args.eps_min, args.eps_max, args.eps_step)
    today = dt.date.today().isoformat()
    results_base = args.results_base or os.path.join(RESULTS_ROOT, f"{args.label}_{today}")

    if not args.dry_run:
        os.makedirs(results_base, exist_ok=True)

    launcher = "bash" if args.local else "sbatch"
    print(
        f"Exact-mu sweep: epsilon [{args.eps_min}, {args.eps_max}] step {args.eps_step} "
        f"({len(eps_values)} pts), L=16..256, mu=2*epsilon"
    )
    print(f"Launcher: {launcher}   results_base: {results_base}   num_batches: {args.num_batches}")

    n_submitted = 0
    for eps in eps_values:
        cmd = [launcher, args.script, f"{eps:.4f}", results_base, str(args.num_batches)]
        print("  " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        n_submitted += 1

    verb = "would submit" if args.dry_run else ("ran" if args.local else "submitted")
    print(f"\n{verb.capitalize()} {n_submitted} job(s).")
    print(f"Results: {results_base}/susceptibility_<L>x<L>_..._epsilon<tag>/susceptibility_data.csv")
    print("To plot: python plot_susceptibility.py --results " f"{results_base} --outdir plots/{args.label}_{today}")


if __name__ == "__main__":
    sys.exit(main())
