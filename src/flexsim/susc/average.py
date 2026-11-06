"""Average <m> and chi over the M replicas of every row in a sweep CSV.

    python src/flexsim/susc/average.py mu_database.csv [outroot]

Reads <outroot>/L{L}_eps{epsilon}_mu{mu}/susceptibility_data.csv for each row (written
by sweep.py; outroot defaults to "output") and writes mu_database_iteration1.csv next to the input CSV: a copy
of it with two extra columns, m_mean and chi_mean.
"""

import csv
import os
import sys

import numpy as np

OUTROOT = "output"  # default; override with the optional second argument


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(f"usage: python {sys.argv[0]} mu_database.csv [outroot]")
    src = sys.argv[1]
    outroot = sys.argv[2] if len(sys.argv) == 3 else OUTROOT
    with open(src) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        L, eps, mu = int(r["lattice_size"]), float(r["epsilon"]), float(r["mu"])
        path = os.path.join(outroot, f"L{L}_eps{eps}_mu{mu}", "susceptibility_data.csv")
        with open(path) as f:
            reps = list(csv.DictReader(f))
        r["m_mean"] = np.mean([float(x["m_mean"]) for x in reps])
        r["chi_mean"] = np.mean([float(x["chi"]) for x in reps])

    dst = os.path.join(os.path.dirname(src), "mu_database_iteration1.csv")
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
