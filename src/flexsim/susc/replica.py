"""Susceptibility run on an L x L lattice: equilibrate, then sample m(t) in chunks.

    python src/flexsim/susc/replica.py params.json

Runs `n_replicas` independent replicas in parallel (one process each) and writes to
`outdir`:
    m_timeseries_<i>.csv     per chunk: rho_bonding, rho_inert, rho_empty, m, energy
    final_lattice_<i>.npy    final lattice of replica i
    susceptibility_data.csv  one row per replica: <m>, <m^2>, <m^4>, chi, <E>, <E^2>

m = rho_bonding - rho_inert - rho_empty, chi = N * beta * (<m^2> - <|m|>^2).

With "continuation": true, replica i restarts from final_lattice_<i>.npy in `outdir`
(no equilibration), appends its new chunks to m_timeseries_<i>.csv, and the
statistics in susceptibility_data.csv are computed over the whole (old + new) series.
"""

import csv
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile

import numpy as np
from lattice_gas import load
from lattice_gas.boundary_condition import Periodic
from lattice_gas.ending_criterion import Time
from lattice_gas.markov_chain import HeteroChain
from lattice_gas.simulate import simulate

EMPTY, INERT, BONDING = 0, 1, 2

DEFAULTS = {
    "beta": 1.0,
    "n_replicas": 1,
    "eq_time": 1e6,
    "prod_time": 2e6,
    "prod_chunks": 20000,
    "initial_m": 0.8,
    "seed": 7000,
    "continuation": False,
}
REQUIRED = ["epsilon", "mu", "delta_mu", "delta_f", "k", "scheme", "L", "outdir"]


def densities(state):
    n = state.size
    return (np.count_nonzero(state == BONDING) / n,
            np.count_nonzero(state == INERT) / n,
            np.count_nonzero(state == EMPTY) / n)


def energy(state, beta, epsilon, mu, delta_f):
    """beta*E = (beta*eps/2) sum_bonding n_bonding_neighbours - beta*mu*n_B - beta*(mu+df)*n_I."""
    b = (state == BONDING).astype(float)
    neighbours = np.roll(b, 1, 0) + np.roll(b, -1, 0) + np.roll(b, 1, 1) + np.roll(b, -1, 1)
    n_b = b.sum()
    n_i = np.count_nonzero(state == INERT)
    return 0.5 * beta * epsilon * float((b * neighbours).sum()) - beta * mu * n_b - beta * (mu + delta_f) * n_i


def initial_state(L, m, rng):
    """Random BONDING/EMPTY lattice with order parameter m (rho_bonding = (1 + m) / 2)."""
    state = np.zeros((L, L), dtype=np.uint32)
    state.ravel()[rng.choice(L * L, int(round((1 + m) / 2 * L * L)), replace=False)] = BONDING
    return state


def replica_files(outdir, i):
    return (os.path.join(outdir, f"final_lattice_{i}.npy"),
            os.path.join(outdir, f"m_timeseries_{i}.csv"))


def check_continuation(p):
    """Raise if any replica of p lacks the final lattice or time series to continue from."""
    missing = [f for i in range(int(p["n_replicas"])) for f in replica_files(p["outdir"], i)
               if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(f"continuation: missing {missing}")


def sem(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def run_replica(args):
    i, p = args
    if int(p["n_replicas"]) % 2:
        raise ValueError(f"n_replicas must be even, got {p['n_replicas']}")
    beta, L = p["beta"], int(p["L"])
    lattice_file, ts_file = replica_files(p["outdir"], i)
    old_rows = []
    if p["continuation"]:
        with open(ts_file) as f:
            old_rows = list(csv.DictReader(f))
    # Independent random stream per replica (reproducible from the base seed): it sets the
    # starting geometry and the engine seed of every simulate call. A continuation also
    # keys on the number of chunks already done, so it never reuses earlier seeds.
    rng = np.random.default_rng([p["seed"], i, len(old_rows)] if old_rows else [p["seed"], i])
    seed, *chunk_seeds = (int(x) for x in rng.integers(2**63, size=p["prod_chunks"] + 1))
    # Half the replicas (even i) start at m = +|initial_m|, the other half at -|initial_m|.
    m0 = abs(p["initial_m"]) if i % 2 == 0 else -abs(p["initial_m"])

    chain = HeteroChain(beta, p["epsilon"], p["delta_mu"],
                        np.exp(beta * (p["mu"] + p["delta_f"])),  # inert fugacity
                        np.exp(beta * p["mu"]),                    # bonding fugacity
                        p["k"], p["scheme"])
    boundary = Periodic()
    scratch = tempfile.mkdtemp(dir="/dev/shm" if os.path.isdir("/dev/shm") else None)  # RAM-backed on Linux

    if p["continuation"]:
        state = np.load(lattice_file).astype(np.uint32)  # no equilibration
        if state.shape != (L, L):
            raise ValueError(f"{lattice_file} has shape {state.shape}, expected {(L, L)}")
    else:
        state = initial_state(L, m0, rng)
        simulate(state, boundary, chain, [], [Time(p["eq_time"])], seed, scratch)
        state = load.final_state(scratch)
    first_chunk = int(old_rows[-1]["chunk"]) + 1 if old_rows else 0

    chunk_time = p["prod_time"] / p["prod_chunks"]
    rows = []
    for c in range(p["prod_chunks"]):
        simulate(state, boundary, chain, [], [Time(chunk_time)], chunk_seeds[c], scratch)
        state = load.final_state(scratch)
        rho_b, rho_i, rho_e = densities(state)
        rows.append({"chunk": first_chunk + c, "rho_bonding": rho_b, "rho_inert": rho_i, "rho_empty": rho_e,
                     "m": rho_b - rho_i - rho_e,
                     "energy": energy(state, beta, p["epsilon"], p["mu"], p["delta_f"])})
    shutil.rmtree(scratch, ignore_errors=True)

    np.save(lattice_file, state)
    with open(ts_file, "a" if p["continuation"] else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not p["continuation"]:
            w.writeheader()
        w.writerows(rows)

    all_rows = old_rows + rows
    m = np.array([float(r["m"]) for r in all_rows])
    e = np.array([float(r["energy"]) for r in all_rows])
    abs_m, m2 = np.abs(m).mean(), (m ** 2).mean()
    return {
        "replica": i, "seed": seed, "initial_m": m0,
        "m_mean": m.mean(), "m_mean_err": sem(m),
        "abs_m_mean": abs_m, "abs_m_mean_err": sem(np.abs(m)),
        "m2_mean": m2, "m2_mean_err": sem(m ** 2),
        "m4_mean": (m ** 4).mean(), "m4_mean_err": sem(m ** 4),
        "chi": beta * L * L * (m2 - abs_m ** 2),
        "e_mean": e.mean(), "e_mean_err": sem(e),
        "e2_mean": (e ** 2).mean(), "e2_mean_err": sem(e ** 2),
    }


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: python {sys.argv[0]} params.json")
    with open(sys.argv[1]) as f:
        p = {**DEFAULTS, **json.load(f)}
    missing = [key for key in REQUIRED if key not in p]
    if missing:
        sys.exit(f"missing parameters in {sys.argv[1]}: {missing}")

    os.makedirs(p["outdir"], exist_ok=True)
    if p["continuation"]:
        check_continuation(p)
        p["eq_time"] = 0
    name = "params_continuation.json" if p["continuation"] else "params.json"
    with open(os.path.join(p["outdir"], name), "w") as f:
        json.dump(p, f, indent=2)

    n = int(p["n_replicas"])
    print(f"L={p['L']} eps={p['epsilon']} mu={p['mu']} replicas={n} -> {p['outdir']}", flush=True)
    with mp.Pool(n) as pool:
        results = pool.map(run_replica, [(i, p) for i in range(n)])

    with open(os.path.join(p["outdir"], "susceptibility_data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    for r in results:
        print(f"replica {r['replica']}: m={r['m_mean']:.4f} chi={r['chi']:.4f}", flush=True)


if __name__ == "__main__":
    main()
