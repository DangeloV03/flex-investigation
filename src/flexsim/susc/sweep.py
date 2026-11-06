"""Sweep over (L, epsilon, mu) from a CSV, M replicas per point, on N CPUs.

    python src/flexsim/susc/sweep.py sweep.json [--RANDOM_INITIAL]

The JSON holds "mu_database" (CSV with lattice_size,epsilon,mu), "outroot",
"n_replicas" (M) and every fixed physics parameter of replica.py (all keys except
L, epsilon, mu, outdir, which come from the CSV row). Every CSV row becomes a params
dict, all (row, replica) tasks go into one pool of N workers, and each row's outdir
gets params.json, m_timeseries_<i>.csv, final_lattice_<i>.npy and
susceptibility_data.csv (written as soon as all M replicas of that row are done).
With "continuation": true every replica of every row must already have its
final_lattice_<i>.npy and m_timeseries_<i>.csv; this is checked for all rows before
anything runs, and each row's parameters go to params_continuation.json instead.
"random_initial": true (or --RANDOM_INITIAL) gives every replica an initial m drawn
uniformly from [-1, 1] instead of +-initial_m (see replica.py).
"""

import csv
import json
import multiprocessing as mp
import os
import sys
import time

from replica import DEFAULTS, REQUIRED, check_continuation, load_params, run_replica

SWEEP_KEYS = ["mu_database", "outroot"]
PER_ROW = ["L", "epsilon", "mu", "outdir"]


def main():
    path, cfg = load_params(sys.argv, DEFAULTS)
    missing = [k for k in SWEEP_KEYS + REQUIRED if k not in PER_ROW and k not in cfg]
    if missing:
        sys.exit(f"missing parameters in {path}: {missing}")
    fixed = {k: v for k, v in cfg.items() if k not in SWEEP_KEYS}
    n_rep = int(fixed["n_replicas"])

    with open(cfg["mu_database"]) as f:
        points = [{"L": int(r["lattice_size"]), "epsilon": float(r["epsilon"]), "mu": float(r["mu"])}
                  for r in csv.DictReader(f)]

    if fixed["continuation"]:
        fixed["eq_time"] = 0
    jobs = [{**fixed, **pt,
             "outdir": os.path.join(cfg["outroot"], f"L{pt['L']}_eps{pt['epsilon']}_mu{pt['mu']}")}
            for pt in points]
    if fixed["continuation"]:  # fail before running anything if any replica can't be continued
        for p in jobs:
            check_continuation(p)
    for p in jobs:
        os.makedirs(p["outdir"], exist_ok=True)
        name = "params_continuation.json" if p["continuation"] else "params.json"
        with open(os.path.join(p["outdir"], name), "w") as f:
            json.dump(p, f, indent=2)

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    tasks = [(j, i, p) for j, p in enumerate(jobs) for i in range(n_rep)]
    mode = " (continuation)" if fixed["continuation"] else ""
    print(f"{len(jobs)} points x {n_rep} replicas on {n_cpus} CPUs{mode}", flush=True)

    results = {j: [] for j in range(len(jobs))}
    spans = {j: [] for j in range(len(jobs))}  # (start, end) of each replica
    with mp.Pool(n_cpus) as pool:
        for j, r, t0, t1 in pool.imap_unordered(run_task, tasks):
            results[j].append(r)
            spans[j].append((t0, t1))
            if len(results[j]) == n_rep:
                rows = sorted(results[j], key=lambda r: r["replica"])
                with open(os.path.join(jobs[j]["outdir"], "susceptibility_data.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                wall = int(max(t1 for _, t1 in spans[j]) - min(t0 for t0, _ in spans[j]))
                print(f"[{time.strftime('%H:%M')}] done {jobs[j]['outdir']}\n"
                      f"    wall time {wall // 3600}h {wall % 3600 // 60:02d}m {wall % 60:02d}s", flush=True)


def run_task(args):
    j, i, p = args
    t0 = time.time()
    r = run_replica((i, p))
    return j, r, t0, time.time()


if __name__ == "__main__":
    main()
