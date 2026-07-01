"""
generate_susceptibility_exact_split.py

Generate square-L susceptibility jobs at the exact analytic coexistence
chemical potential mu = 2 * epsilon, with the replicas split 50/50 between
the two ordered initial conditions:

  - even run_ids: initial_fraction = 0.8  (dense basin,  m ≈ +0.6 at t=0)
  - odd  run_ids: initial_fraction = 0.2  (dilute basin, m ≈ -0.6 at t=0)

Rationale: at large L the well-flip time exceeds prod_time, so pooled P(m)
reflects basin selection, not Boltzmann weights. Seeding both basins by
construction (the Hamiltonian is particle-hole symmetric at mu = 2*epsilon)
gives a fair two-well sample without relying on spontaneous tunneling.

Everything is isolated from the other phases: own samples dir, own manifest,
own results base (susceptibility_results/exact_split/), own seed base.

Unlike older generators, existing job JSONs are ALWAYS rewritten so a
parameter fix here can never be silently shadowed by stale files on disk.

Usage:
    python generate_susceptibility_exact_split.py
    python run_susceptibility_all.py --phase exact_split
    python plot_susceptibility.py --results susceptibility_results/exact_split --outdir plots/exact_split
"""

from __future__ import annotations

import argparse
import json
import os

from generate_samples import frange
from queue_manifest import merge_pending
from susceptibility_paths import (
    EXACT_SPLIT_MANIFEST,
    EXACT_SPLIT_RESULTS_BASE,
    EXACT_SPLIT_SAMPLES_DIR,
    ISING_DELTA_F,
    ISING_DELTA_MU,
    ISING_K,
    ISING_SCHEME,
    SQUARE_L_VALUES,
)

EPS_MIN = -2.0
EPS_MAX = -1.4
EPS_STEP = 0.01

INITIAL_FRACTIONS = [0.8, 0.2]

DEFAULT_RUN_SETTINGS = {
    "beta": 1.0,
    "initial_fractions": INITIAL_FRACTIONS,
    "num_parallel_runs": 8,
    "eq_time": 100000.0,
    "prod_time": 100000.0,
    "prod_chunks": 1000,
    "seed_base": 15000,
}


def exact_split_job_filename(scheme: str, epsilon: float, l: int) -> str:
    eps_tag = f"eps{epsilon:.4f}".replace("-", "m").replace(".", "p")
    return f"exact_split_{scheme}_{eps_tag}_L{l}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exact-mu susceptibility jobs with 80/20 + 20/80 split ICs"
    )
    parser.add_argument("--eps-min", type=float, default=EPS_MIN)
    parser.add_argument("--eps-max", type=float, default=EPS_MAX)
    parser.add_argument("--eps-step", type=float, default=EPS_STEP)
    parser.add_argument("--L", type=int, nargs="+", default=SQUARE_L_VALUES)
    parser.add_argument("--samples-dir", default=EXACT_SPLIT_SAMPLES_DIR)
    parser.add_argument("--manifest", default=EXACT_SPLIT_MANIFEST)
    parser.add_argument("--results-base", default=EXACT_SPLIT_RESULTS_BASE)
    args = parser.parse_args()

    os.makedirs(args.samples_dir, exist_ok=True)
    eps_values = frange(args.eps_min, args.eps_max, args.eps_step)
    l_values = sorted(set(args.L))

    pending_paths: list[str] = []
    n_written = 0
    n_unchanged = 0

    print(
        f"Exact-mu split-IC jobs: epsilon [{args.eps_min}, {args.eps_max}] "
        f"step {args.eps_step} ({len(eps_values)} pts), "
        f"L={l_values}, mu=2*epsilon, initial_fractions={INITIAL_FRACTIONS}"
    )

    for epsilon in eps_values:
        mu_exact = 2.0 * epsilon

        for l_val in l_values:
            job = {
                "epsilon": epsilon,
                "delta_f": ISING_DELTA_F,
                "delta_mu": ISING_DELTA_MU,
                "k": ISING_K,
                "scheme": ISING_SCHEME,
                "Lx": l_val,
                "Ly": l_val,
                "mu": mu_exact,
                "mu_exact": mu_exact,
                "run_settings": dict(DEFAULT_RUN_SETTINGS),
                "results_base": args.results_base,
            }
            filename = exact_split_job_filename(ISING_SCHEME, epsilon, l_val)
            filepath = os.path.join(args.samples_dir, filename)
            payload = json.dumps(job, indent=2)
            if os.path.isfile(filepath):
                with open(filepath) as f:
                    if f.read().rstrip("\n") == payload:
                        n_unchanged += 1
                        pending_paths.append(filepath)
                        continue
            with open(filepath, "w") as f:
                f.write(payload)
            n_written += 1
            pending_paths.append(filepath)

    merge_pending(pending_paths, path=args.manifest)

    print(f"Wrote {n_written} JSON files ({n_unchanged} unchanged)")
    print(f"Queued {len(pending_paths)} jobs into '{args.manifest}'")
    print(f"Results will go to '{args.results_base}/'")
    print(f"\nTo run:  python run_susceptibility_all.py --phase exact_split")
    print(
        f"To plot: python plot_susceptibility.py "
        f"--results {args.results_base} --outdir plots/exact_split"
    )


if __name__ == "__main__":
    main()
