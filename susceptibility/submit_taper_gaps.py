#!/usr/bin/env python3
"""
Scan a tapered homo campaign for missing (ε, L) and submit finish/full jobs.

Usage (repo root on Della):
    python susceptibility/submit_taper_gaps.py --dry-run
    python susceptibility/submit_taper_gaps.py

Defaults match homo_dmu1p0_taper_2026-07-19. PARTIAL ε get finish jobs (missing L
only, 72 h). NOT STARTED ε get full L=16..128 jobs (72 h).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_samples import frange
from sweep_susceptibility import load_mu_map

DEFAULT_RESULTS = "susceptibility/results/homo_dmu1p0_taper_2026-07-19"
DEFAULT_MU_SOURCE = "coex_manage_homo_dmu1p0.csv"
FULL_SCRIPT = "susceptibility/run_susceptibility_homo_dmu1p0_tapered.sh"
FINISH_SCRIPT = "susceptibility/run_susceptibility_homo_dmu1p0_tapered_finish.sh"
ALL_SIZES = [16, 32, 48, 64, 96, 128]


def scan_results(results_base: str) -> dict[float, set[int]]:
    by_eps: dict[float, set[int]] = {}
    pattern = os.path.join(results_base, "susceptibility_*", "susceptibility_data.csv")
    for path in glob.glob(pattern):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        eps = round(float(rows[0]["epsilon"]), 4)
        L = int(float(rows[0]["L"]))
        by_eps.setdefault(eps, set()).add(L)
    return by_eps


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit tapered homo gap-fill jobs.")
    parser.add_argument("--results-base", default=DEFAULT_RESULTS)
    parser.add_argument("--mu-source", default=DEFAULT_MU_SOURCE)
    parser.add_argument("--eps-min", type=float, default=-2.0)
    parser.add_argument("--eps-max", type=float, default=-1.4)
    parser.add_argument("--eps-step", type=float, default=0.005)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--delta-f", default="0.0")
    parser.add_argument("--delta-mu", default="1.0")
    parser.add_argument("--k", default="1.0")
    parser.add_argument("--scheme", default="homo")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.results_base):
        print(f"ERROR: results not found: {args.results_base}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.mu_source):
        print(f"ERROR: mu source not found: {args.mu_source}", file=sys.stderr)
        return 1

    mu_map = load_mu_map(
        args.mu_source,
        delta_f=float(args.delta_f),
        delta_mu=float(args.delta_mu),
        k=float(args.k),
        scheme=args.scheme,
    )
    by_eps = scan_results(args.results_base)
    eps_values = list(frange(args.eps_min, args.eps_max, args.eps_step))

    if not args.dry_run:
        os.makedirs("slurm_reports", exist_ok=True)

    n_full = n_finish = n_skip = n_no_mu = 0
    for eps in eps_values:
        eps = round(eps, 4)
        have = by_eps.get(eps, set())
        missing = [L for L in ALL_SIZES if L not in have]
        if not missing:
            n_skip += 1
            continue
        mu = mu_map.get(eps)
        if mu is None:
            print(f"  [skip] eps={eps:.4f}: no fitted mu in {args.mu_source}")
            n_no_mu += 1
            continue

        mu_arg = f"{mu:.6f}"
        job_args = [
            f"{eps:.4f}",
            args.results_base,
            str(args.num_batches),
            mu_arg,
            args.delta_f,
            args.delta_mu,
            args.k,
            args.scheme,
        ]

        if not have:
            script = FULL_SCRIPT
            export: list[str] = []
            label = "FULL"
            n_full += 1
        else:
            script = FINISH_SCRIPT
            sizes = " ".join(str(L) for L in missing)
            export = [f"--export=ALL,TAPER_SIZES={sizes}"]
            label = f"FINISH L={missing}"
            n_finish += 1

        cmd = ["sbatch", *export, script, *job_args]
        print(f"  [{label}] eps={eps:.4f}  " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    print(
        f"\n{'Would submit' if args.dry_run else 'Submitted'} "
        f"{n_full} full + {n_finish} finish job(s); "
        f"skipped {n_skip} complete, {n_no_mu} without mu."
    )
    print(f"Monitor: squeue -u $USER | grep -E 'susc_homo_taper|susc_taper_fin'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
