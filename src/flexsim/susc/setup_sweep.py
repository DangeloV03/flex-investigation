"""Build a sweep CSV (lattice_size,epsilon,mu) from a mu_coex polynomial fit.

Usage:
  python setup_sweep.py FIT.npy --e_min -1.9 --e_max -1.5 --e_step 0.05 \
      --sizes 16 32 48 [-o mu_database.csv]

mu = np.polyval(coef, eps) for eps in np.arange(e_min, e_max, e_step)
(e_max is exclusive, as in np.arange). One row per (size, eps).
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("fit", help=".npy of polynomial coefficients (highest power first)")
ap.add_argument("--e_min", type=float, required=True)
ap.add_argument("--e_max", type=float, required=True)
ap.add_argument("--e_step", type=float, required=True)
ap.add_argument("--sizes", type=int, nargs="+", required=True)
ap.add_argument("-o", "--out", default="mu_database.csv")
a = ap.parse_args()

coef = np.load(a.fit)
# round to 5 decimals (also kills arange float noise, e.g. -1.7999999999 -> -1.8,
# which matters since average.py rebuilds directory names from float(eps))
eps = np.round(np.arange(a.e_min, a.e_max, a.e_step), 5)
mu = np.round(np.polyval(coef, eps), 5)

with open(a.out, "w") as f:
    f.write("lattice_size,epsilon,mu\n")
    for L in a.sizes:
        for e, m in zip(eps.tolist(), mu.tolist()):
            f.write(f"{L},{e!r},{m!r}\n")
print(f"wrote {len(a.sizes) * len(eps)} rows to {a.out}")
