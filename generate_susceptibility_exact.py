"""
generate_susceptibility_exact.py

Generate square-L susceptibility jobs using the exact analytic coexistence
chemical potential mu = 2 * epsilon, bypassing the coex slab phase entirely.

This is a control experiment: if the susceptibility plots look correct here
but not with mu_coex_FITTED, the issue is in the coex measurement, not the
susceptibility runner itself.

Usage:
    python generate_susceptibility_exact.py
"""

from __future__ import annotations

import argparse
import json
import os

from generate_samples import frange
from queue_manifest import merge_pending
from susceptibility_paths import (
    EXACT_MANIFEST,
    EXACT_RESULTS_BASE,
    EXACT_SAMPLES_DIR,
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
    "initial_fraction": 0.8,
    "num_parallel_runs": 8,
    "eq_time": 100000.0,
    "prod_time": 100000.0,
    "prod_chunks": 20,
    "seed_base": 7000,
}


def exact_job_filename(scheme: str, epsilon: float, l: int) -> str:
    eps_tag = f"eps{epsilon:.4f}".replace("-", "m").replace(".", "p")
    return f"exact_{scheme}_{eps_tag}_L{l}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exact-mu susceptibility jobs (mu = 2*epsilon)"
    )
    parser.add_argument("--eps-min", type=float, default=EPS_MIN)
    parser.add_argument("--eps-max", type=float, default=EPS_MAX)
    parser.add_argument("--eps-step", type=float, default=EPS_STEP)
    parser.add_argument("--L", type=int, nargs="+", default=SQUARE_L_VALUES)
    parser.add_argument("--samples-dir", default=EXACT_SAMPLES_DIR)
    parser.add_argument("--manifest", default=EXACT_MANIFEST)
    parser.add_argument("--results-base", default=EXACT_RESULTS_BASE)
    args = parser.parse_args()

    os.makedirs(args.samples_dir, exist_ok=True)
    eps_values = frange(args.eps_min, args.eps_max, args.eps_step)
    l_values = sorted(set(args.L))

    pending_paths: list[str] = []
    n_files = 0
    n_existing = 0

    print(
        f"Exact-mu jobs: epsilon [{args.eps_min}, {args.eps_max}] "
        f"step {args.eps_step} ({len(eps_values)} pts), "
        f"L={l_values}, mu=2*epsilon"
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
            filename = exact_job_filename(ISING_SCHEME, epsilon, l_val)
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
    print(f"\nTo run:  python run_susceptibility_all.py --phase exact")
    print(f"To plot: python plot_susceptibility.py --results {args.results_base} --outdir plots/exact")


if __name__ == "__main__":
    main()
