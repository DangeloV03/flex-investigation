"""
generate_susceptibility_exact_random.py

Generate square-L susceptibility jobs using the exact analytic coexistence
chemical potential mu = 2 * epsilon, with each replica starting from a
*random* (disordered) initial condition: initial_fraction=0.5, giving m≈0
at t=0, matching the Kumar & Dasgupta (2020) per-realization protocol.

Compare against generate_susceptibility_exact.py (ordered IC, fraction=0.8)
to check for hysteresis or ergodicity failure at large L.

Usage:
    python generate_susceptibility_exact_random.py
"""

from __future__ import annotations

import argparse
import json
import os

from generate_samples import frange
from queue_manifest import merge_pending
from susceptibility_paths import (
    EXACT_RANDOM_MANIFEST,
    EXACT_RANDOM_RESULTS_BASE,
    EXACT_RANDOM_SAMPLES_DIR,
    ISING_DELTA_F,
    ISING_DELTA_MU,
    ISING_K,
    ISING_SCHEME,
    SQUARE_L_VALUES,
)

EPS_MIN = -2.0
EPS_MAX = -1.4
EPS_STEP = 0.01

DEFAULT_RUN_SETTINGS = {
    "beta": 1.0,
    "initial_fraction": 0.5,
    "num_parallel_runs": 8,
    "eq_time": 100000.0,
    "prod_time": 100000.0,
    "prod_chunks": 1000,
    "seed_base": 9000,
}


def exact_random_job_filename(scheme: str, epsilon: float, l: int) -> str:
    eps_tag = f"eps{epsilon:.4f}".replace("-", "m").replace(".", "p")
    return f"exact_random_{scheme}_{eps_tag}_L{l}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exact-mu susceptibility jobs with random (disordered) IC"
    )
    parser.add_argument("--eps-min", type=float, default=EPS_MIN)
    parser.add_argument("--eps-max", type=float, default=EPS_MAX)
    parser.add_argument("--eps-step", type=float, default=EPS_STEP)
    parser.add_argument("--L", type=int, nargs="+", default=SQUARE_L_VALUES)
    parser.add_argument("--samples-dir", default=EXACT_RANDOM_SAMPLES_DIR)
    parser.add_argument("--manifest", default=EXACT_RANDOM_MANIFEST)
    parser.add_argument("--results-base", default=EXACT_RANDOM_RESULTS_BASE)
    args = parser.parse_args()

    os.makedirs(args.samples_dir, exist_ok=True)
    eps_values = frange(args.eps_min, args.eps_max, args.eps_step)
    l_values = sorted(set(args.L))

    pending_paths: list[str] = []
    n_files = 0
    n_existing = 0

    print(
        f"Exact-mu random-IC jobs: epsilon [{args.eps_min}, {args.eps_max}] "
        f"step {args.eps_step} ({len(eps_values)} pts), "
        f"L={l_values}, mu=2*epsilon, initial_fraction=0.5"
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
            filename = exact_random_job_filename(ISING_SCHEME, epsilon, l_val)
            filepath = os.path.join(args.samples_dir, filename)
            if os.path.isfile(filepath):
                n_existing += 1
            else:
                with open(filepath, "w") as f:
                    json.dump(job, f, indent=2)
                n_files += 1
            pending_paths.append(filepath)

    merge_pending(pending_paths, path=args.manifest)

    print(f"Wrote {n_files} new JSON files ({n_existing} already existed)")
    print(f"Queued {len(pending_paths)} jobs into '{args.manifest}'")
    print(f"Results will go to '{args.results_base}/'")
    print(f"\nTo run:  python run_susceptibility_all.py --phase exact_random")
    print(f"To plot: python plot_susceptibility.py --results {args.results_base} --outdir plots/exact_random")


if __name__ == "__main__":
    main()
