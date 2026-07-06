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
import csv
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


def _matches(row: dict, field: str, val) -> bool:
    """True if row[field] equals val (numeric-tolerant), or val is None (no filter)."""
    if val is None:
        return True
    raw = row.get(field, "")
    try:
        return abs(float(raw) - float(val)) < 1e-9
    except (TypeError, ValueError):
        return str(raw) == str(val)


def load_mu_map(
    path: str, *, delta_f=None, delta_mu=None, k=None, scheme=None
) -> dict[float, float]:
    """Map epsilon -> mu_coex_FITTED from a coex manage CSV.

    Rows with blank/NaN mu_coex_FITTED are skipped. When scheme params are given,
    only rows matching them are used (lets one CSV hold multiple schemes safely).
    """
    mu_map: dict[float, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            mu = (row.get("mu_coex_FITTED") or "").strip()
            if mu.lower() in ("", "nan"):
                continue
            if not (
                _matches(row, "delta_f", delta_f)
                and _matches(row, "delta_mu", delta_mu)
                and _matches(row, "k", k)
                and _matches(row, "scheme", scheme)
            ):
                continue
            try:
                eps_key = round(float(row["epsilon"]), 6)
            except (KeyError, TypeError, ValueError):
                continue
            mu_map[eps_key] = float(mu)
    return mu_map


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
        "--mu-source",
        default=None,
        help=(
            "Coex manage CSV with a mu_coex_FITTED column. When set, each epsilon "
            "runs at its fitted mu_coex instead of the exact mu = 2*epsilon. "
            "Epsilons with no (or NaN) fitted value are skipped."
        ),
    )
    parser.add_argument("--delta-f", type=float, default=None, help="Δf passed to the runner (and mu-source filter)")
    parser.add_argument("--delta-mu", type=float, default=None, help="Δμ passed to the runner (and mu-source filter)")
    parser.add_argument("--k", type=float, default=None, help="k passed to the runner (and mu-source filter)")
    parser.add_argument("--scheme", default=None, help="HeteroChain scheme passed to the runner (and mu-source filter)")
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

    mu_map: dict[float, float] | None = None
    if args.mu_source:
        mu_map = load_mu_map(
            args.mu_source,
            delta_f=args.delta_f,
            delta_mu=args.delta_mu,
            k=args.k,
            scheme=args.scheme,
        )
        if not mu_map:
            print(f"ERROR: no usable mu_coex_FITTED rows in {args.mu_source} "
                  f"(after scheme filter). Nothing to submit.")
            return

    if not args.dry_run:
        os.makedirs(results_base, exist_ok=True)
        # SLURM writes job logs to ~/slurm_reports (see run_susceptibility.sh) but
        # won't create the dir; make sure it exists so logs aren't dropped.
        if not args.local:
            os.makedirs(os.path.expanduser("~/slurm_reports"), exist_ok=True)

    launcher = "bash" if args.local else "sbatch"
    mu_desc = f"fitted mu from {args.mu_source}" if mu_map is not None else "mu = 2*epsilon"
    print(
        f"Susceptibility sweep: epsilon [{args.eps_min}, {args.eps_max}] step {args.eps_step} "
        f"({len(eps_values)} pts), L=16..256, {mu_desc}"
    )
    scheme_desc = (
        f"scheme={args.scheme} delta_f={args.delta_f} delta_mu={args.delta_mu} k={args.k}"
        if any(v is not None for v in (args.scheme, args.delta_f, args.delta_mu, args.k))
        else "scheme=runner defaults (Ising limit)"
    )
    print(f"Launcher: {launcher}   results_base: {results_base}   num_batches: {args.num_batches}")
    print(f"Params: {scheme_desc}")

    # Extra positional args consumed by run_susceptibility.sh ($4..$8). Empty
    # string => "use the runner default" (exact mu / Ising params).
    def _s(v) -> str:
        return "" if v is None else str(v)

    n_submitted = 0
    n_skipped = 0
    for eps in eps_values:
        if mu_map is not None:
            mu = mu_map.get(round(eps, 6))
            if mu is None:
                print(f"  [skip] eps={eps:.4f}: no fitted mu_coex in mu-source")
                n_skipped += 1
                continue
            mu_arg = f"{mu:.6f}"
        else:
            mu_arg = ""  # runner falls back to mu = 2*epsilon

        cmd = [
            launcher, args.script, f"{eps:.4f}", results_base, str(args.num_batches),
            mu_arg, _s(args.delta_f), _s(args.delta_mu), _s(args.k), _s(args.scheme),
        ]
        print("  " + " ".join(repr(c) if c == "" else c for c in cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        n_submitted += 1

    verb = "would submit" if args.dry_run else ("ran" if args.local else "submitted")
    print(f"\n{verb.capitalize()} {n_submitted} job(s)"
          + (f", skipped {n_skipped} (no fitted mu)." if n_skipped else "."))
    print(f"Results: {results_base}/susceptibility_<L>x<L>_..._epsilon<tag>/susceptibility_data.csv")
    print("To plot: python plot_susceptibility.py --results " f"{results_base} --outdir plots/{args.label}_{today}")


if __name__ == "__main__":
    sys.exit(main())
