"""
generate_test_samples.py

Validation runs for three known parameter sets. The FLEX estimate is used to
center the mu sweep — the exact mu_coex values are intentionally not used so
the simulation runs blind and results can be compared against ground truth afterward.

Parameter sets:
  - epsilon=-2.5,  delta_f=-10.0, k=0.0, delta_mu=0.0
  - epsilon=-1.76, delta_f=-10.0, k=0.0, delta_mu=0.0
  - epsilon=-1.4,  delta_f=-10.0, k=0.0, delta_mu=0.0

Usage:
    python generate_test_samples.py
"""

import csv
import json
import os
import time
import numpy as np

from flex_coex_chemical_potential_prediction import coex_chemical_potential


# ---------------------------------------------------------------------------
# Validation parameter sets
# ---------------------------------------------------------------------------

TEST_PARAMS = [
    {"epsilon": -2.5,  "delta_f": -10.0, "delta_mu": 0.0, "k": 0.0},
    {"epsilon": -1.76, "delta_f": -10.0, "delta_mu": 0.0, "k": 0.0},
    {"epsilon": -1.4,  "delta_f": -10.0, "delta_mu": 0.0, "k": 0.0},
]

# scheme name -> FLEX scheme index (1=homo, 2=positive_drive, 3=negative_drive)
SCHEME_TO_FLEX_INDEX = {
    "homo": 1,
}

LY_VALUES = [8, 16, 32]
LX_MULTIPLIER = 10  # Lx = LX_MULTIPLIER * Ly

# ---------------------------------------------------------------------------
# Inner mu sweep settings
# ---------------------------------------------------------------------------

MU_WINDOW = 0.1
N_MU_POINTS = 10

# ---------------------------------------------------------------------------
# Shared run settings
# ---------------------------------------------------------------------------

RUN_SETTINGS = {
    "beta": 1.0,
    "initial_condition": "slab_half_active_half_empty",
    "num_parallel_runs": 4,
    "eq_time": 10000.0,
    "prod_time": 1000.0,
    "seed_base": 1000,
}

OUTPUT_DIR = "test_samples"
MANAGE_CSV = "test_manage.csv"

COMBO_KEY_FIELDS = ["epsilon", "delta_f", "delta_mu", "k", "scheme", "Lx", "Ly"]
MANAGE_FIELDS = COMBO_KEY_FIELDS + [
    "mu_coex_flex",
    "isSubmitted",
    "isRan",
    "isAnalyzed",
    "mu_coex_SIM",
    "RequestForAdditionalData",
]


def mu_sweep(mu_coex_flex: float) -> list[float]:
    values = np.linspace(mu_coex_flex - MU_WINDOW, mu_coex_flex + MU_WINDOW, N_MU_POINTS)
    return [round(float(v), 6) for v in values]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_files = 0
    skipped = []
    manage_rows = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    for params in TEST_PARAMS:
        epsilon = params["epsilon"]
        delta_f = params["delta_f"]
        delta_mu = params["delta_mu"]
        k = params["k"]

        for scheme, flex_index in SCHEME_TO_FLEX_INDEX.items():
            mu_coex_flex = coex_chemical_potential(
                epsilon=epsilon,
                df=delta_f,
                dmu=delta_mu,
                chem_rec_baserate=k,
                DRIVEN=True,
                scheme=flex_index,
            )
            mu_coex_flex = float(np.asarray(mu_coex_flex).ravel()[0])

            if mu_coex_flex > 0:
                label = f"epsilon={epsilon} scheme={scheme}"
                print(f"[skip] {label}: mu_coex_FLEX={mu_coex_flex:.6f} > 0")
                skipped.append(label)
                continue

            print(f"epsilon={epsilon} scheme={scheme}: mu_coex_FLEX={mu_coex_flex:.6f}")
            mu_values = mu_sweep(mu_coex_flex)

            for Ly in LY_VALUES:
                Lx = LX_MULTIPLIER * Ly

                run_settings = dict(RUN_SETTINGS)
                run_settings["k"] = k

                manage_rows.append({
                    "epsilon": epsilon,
                    "delta_f": delta_f,
                    "delta_mu": delta_mu,
                    "k": k,
                    "scheme": scheme,
                    "Lx": Lx,
                    "Ly": Ly,
                    "mu_coex_flex": mu_coex_flex,
                    "isSubmitted": timestamp,
                    "isRan": "",
                    "isAnalyzed": "",
                    "mu_coex_SIM": "",
                    "RequestForAdditionalData": 0,
                })

                for idx, mu in enumerate(mu_values):
                    job = {
                        "epsilon": epsilon,
                        "delta_f": delta_f,
                        "delta_mu": delta_mu,
                        "k": k,
                        "scheme": scheme,
                        "Lx": Lx,
                        "Ly": Ly,
                        "mu": mu,
                        "mu_coex_flex": mu_coex_flex,
                        "run_settings": run_settings,
                    }

                    # name encodes all params to avoid collisions across test sets
                    eps_tag = str(abs(epsilon)).replace(".", "")
                    filename = f"test_eps{eps_tag}_{scheme}_Ly{Ly}_mu{idx:02d}.json"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    with open(filepath, "w") as f:
                        json.dump(job, f, indent=2)
                    n_files += 1

    with open(MANAGE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANAGE_FIELDS)
        writer.writeheader()
        for row in manage_rows:
            writer.writerow(row)

    print(f"\nWrote {n_files} JSON files to '{OUTPUT_DIR}/'")
    print(f"Wrote {len(manage_rows)} rows to '{MANAGE_CSV}'")
    if skipped:
        print(f"Skipped (mu_coex_FLEX > 0): {skipped}")


if __name__ == "__main__":
    main()