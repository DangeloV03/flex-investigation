"""Sweep over (L, epsilon, mu) from a CSV, M replicas per point, on N CPUs.

    python src/flexsim/susc/sweep.py mu_database.csv [outroot]

Every CSV row becomes an in-memory params dict (same keys as params.json) with the
fixed parameters below. All (row, replica) tasks go into one pool of N workers, so
when M < N several rows run at the same time. Each row's outdir gets params.json,
m_timeseries_<i>.csv, final_lattice_<i>.npy and susceptibility_data.csv (written
as soon as all M replicas of that row are done).
"""

import csv
import json
import multiprocessing as mp
import os
import sys

from replica import run_replica

# ---- fixed parameters (L, epsilon, mu come from the CSV) ----
N_REPLICAS = 16  # M: independent trajectories per (L, epsilon, mu)
FIXED = {
    "delta_mu": 2.0,
    "delta_f": 0.0,
    "k": 1.0,
    "scheme": "negative_drive",
    "beta": 1.0,
    "eq_time": 3e6,
    "prod_time": 5e5,
    "prod_chunks": 500,
    "initial_m": 0.8,
    "seed": 7000,
}
OUTROOT = "output"  # default; override with the optional second argument
# --------------------------------------------------------------


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(f"usage: python {sys.argv[0]} mu_database.csv [outroot]")
    outroot = sys.argv[2] if len(sys.argv) == 3 else OUTROOT
    with open(sys.argv[1]) as f:
        points = [{"L": int(r["lattice_size"]), "epsilon": float(r["epsilon"]), "mu": float(r["mu"])}
                  for r in csv.DictReader(f)]

    jobs = []
    for pt in points:
        p = {**FIXED, **pt, "n_replicas": N_REPLICAS,
             "outdir": os.path.join(outroot, f"L{pt['L']}_eps{pt['epsilon']}_mu{pt['mu']}")}
        os.makedirs(p["outdir"], exist_ok=True)
        with open(os.path.join(p["outdir"], "params.json"), "w") as f:
            json.dump(p, f, indent=2)
        jobs.append(p)

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    tasks = [(j, i, p) for j, p in enumerate(jobs) for i in range(N_REPLICAS)]
    print(f"{len(jobs)} points x {N_REPLICAS} replicas on {n_cpus} CPUs", flush=True)

    results = {j: [] for j in range(len(jobs))}
    with mp.Pool(n_cpus) as pool:
        for j, r in pool.imap_unordered(run_task, tasks):
            results[j].append(r)
            if len(results[j]) == N_REPLICAS:
                rows = sorted(results[j], key=lambda r: r["replica"])
                with open(os.path.join(jobs[j]["outdir"], "susceptibility_data.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                print(f"done {jobs[j]['outdir']}", flush=True)


def run_task(args):
    j, i, p = args
    return j, run_replica((i, p))


if __name__ == "__main__":
    main()
