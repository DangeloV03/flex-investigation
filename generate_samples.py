"""
generate_samples.py

Workflow:
  1. Fixed: epsilon = -2.0, delta_f = 0, delta_mu = 0, k = 1.
  2. For each scheme in {homo, positive_drive, negative_drive}:
       - Compute mu_coex_FLEX (DRIVEN=True) using flex_coex_chemical_potential_prediction.
       - If mu_coex_FLEX > 0, skip this scheme.
       - Else, for each Ly in [8, 16, 32] (Lx = 10 * Ly):
           - Sweep mu over mu_coex_FLEX +/- 0.1, 10 points.
           - Write one self-contained JSON file per (scheme, Ly, mu).

Each JSON file is intended to be passed directly to json_runner.py, e.g.:
    python json_runner.py samples/homo_Ly8_mu4012345.json
"""

import csv
import json
import os
import time
import numpy as np

from flex_coex_chemical_potential_prediction import coex_chemical_potential


# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

EPSILON = -2.0
DELTA_F = 0.0
DELTA_MU = 0.0
K = 1.0  # chem_rec_baserate / inert_to_bonding_rate

# scheme name -> FLEX scheme index (1=homo, 2=positive_drive, 3=negative_drive)
SCHEME_TO_FLEX_INDEX = {
    "homo": 1,
    "positive_drive": 2,
    "negative_drive": 3,
}

LY_VALUES = [8, 16, 32]
LX_MULTIPLIER = 10  # Lx = LX_MULTIPLIER * Ly

# ---------------------------------------------------------------------------
# Inner mu sweep settings
# ---------------------------------------------------------------------------

MU_WINDOW = 0.1  # +/- around mu_coex_FLEX
N_MU_POINTS = 10

# ---------------------------------------------------------------------------
# Shared run settings (passed through to json_runner.py)
# ---------------------------------------------------------------------------

RUN_SETTINGS = {
    "beta": 1.0,
    "k": K,
    "initial_condition": "slab_half_active_half_empty",  # x < Lx/2 -> active, x >= Lx/2 -> empty
    "num_parallel_runs": 4,
    "eq_time": 50.0,
    "prod_time": 50.0,
    "seed_base": 1000,
}

OUTPUT_DIR = "samples"
MANAGE_CSV = "manage.csv"

# Columns identifying a unique outer combo (used as the match key by json_runner.py)
COMBO_KEY_FIELDS = ["epsilon", "delta_f", "delta_mu", "k", "scheme", "Lx", "Ly"]
MANAGE_FIELDS = COMBO_KEY_FIELDS + [
    "mu_coex_flex",
    "isSubmitted",
    "isRan",
    "isAnalyzed",
    "mu_coex_SIM",
]


def mu_sweep(mu_coex_flex: float) -> list[float]:
    values = np.linspace(mu_coex_flex - MU_WINDOW, mu_coex_flex + MU_WINDOW, N_MU_POINTS)
    return [round(float(v), 6) for v in values]


def mu_filename_tag(mu: float) -> str:
    """Encode |mu| in a filename-safe tag (6 decimal places)."""
    return f"mu{round(abs(mu) * 1_000_000):06d}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_files = 0
    skipped_schemes = []
    manage_rows = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    for scheme, flex_index in SCHEME_TO_FLEX_INDEX.items():
        mu_coex_flex = coex_chemical_potential(
            epsilon=EPSILON,
            df=DELTA_F,
            dmu=DELTA_MU,
            chem_rec_baserate=K,
            DRIVEN=True,
            scheme=flex_index,
        )
        # fsolve may return an array-like; coerce to float
        mu_coex_flex = float(np.asarray(mu_coex_flex).ravel()[0])

        if mu_coex_flex > 0:
            print(f"[skip] scheme={scheme}: mu_coex_FLEX={mu_coex_flex:.6f} > 0")
            skipped_schemes.append(scheme)
            continue

        print(f"scheme={scheme}: mu_coex_FLEX={mu_coex_flex:.6f}")
        mu_values = mu_sweep(mu_coex_flex)

        for Ly in LY_VALUES:
            Lx = LX_MULTIPLIER * Ly

            manage_rows.append(
                {
                    "epsilon": EPSILON,
                    "delta_f": DELTA_F,
                    "delta_mu": DELTA_MU,
                    "k": K,
                    "scheme": scheme,
                    "Lx": Lx,
                    "Ly": Ly,
                    "mu_coex_flex": mu_coex_flex,
                    "isSubmitted": timestamp,
                    "isRan": "",
                    "isAnalyzed": "",
                    "mu_coex_SIM": "",
                }
            )

            for mu in mu_values:
                job = {
                    "epsilon": EPSILON,
                    "delta_f": DELTA_F,
                    "delta_mu": DELTA_MU,
                    "k": K,
                    "scheme": scheme,
                    "Lx": Lx,
                    "Ly": Ly,
                    "mu": mu,
                    "mu_coex_flex": mu_coex_flex,
                    "run_settings": RUN_SETTINGS,
                }

                filename = f"{scheme}_Ly{Ly}_{mu_filename_tag(mu)}.json"
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
    if skipped_schemes:
        print(f"Skipped schemes (mu_coex_FLEX > 0): {skipped_schemes}")


if __name__ == "__main__":
    main()