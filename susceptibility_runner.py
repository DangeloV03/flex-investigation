"""
susceptibility_runner.py

Long production runs on square L×L lattices at μ_coex for susceptibility measurement.
Initial condition: random 80% active (BONDING), 20% empty.

For each replica:
  - Equilibrate (eq_time), discard.
  - Production in chunks; record m = φ = ρ_active - ρ_inert - ρ_empty each chunk.
  - Compute time-averaged <m>, <m²>, <m⁴> and χ = (N/T)(<m²> - <m>²) with N = L², T = 1/β.

Outputs (per job directory):
  - susceptibility_data.csv
  - final_lattice_{id}.npy (optional snapshots)

Usage:
    python susceptibility_runner.py susceptibility_samples/prod/susceptibility_homo_eps1p76_dm0p0_L64.json
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil

import numpy as np

from lattice_gas import load
from lattice_gas.boundary_condition import Periodic
from lattice_gas.ending_criterion import Time
from lattice_gas.markov_chain import HeteroChain
from lattice_gas.simulate import simulate

from susceptibility_paths import PROD_RESULTS_BASE, SUSCEPTIBILITY_DATA_CSV, susceptibility_prod_dir

EMPTY, INERT, BONDING = 0, 1, 2

CSV_FIELDNAMES = [
    "id",
    "replica_id",
    "epsilon",
    "delta_f",
    "delta_mu",
    "k",
    "scheme",
    "L",
    "Lx",
    "Ly",
    "mu",
    "mu_coex_SIM",
    "m_mean",
    "m2_mean",
    "m4_mean",
    "chi",
    "beta",
    "eq_time",
    "prod_time",
    "prod_chunks",
    "initial_fraction",
    "seed",
    "time",
]


def compute_densities(state: np.ndarray) -> tuple[float, float, float]:
    total = state.size
    rho_active = float(np.count_nonzero(state == BONDING)) / total
    rho_inert = float(np.count_nonzero(state == INERT)) / total
    rho_empty = float(np.count_nonzero(state == EMPTY)) / total
    return rho_active, rho_inert, rho_empty


def compute_m(state: np.ndarray) -> float:
    rho_a, rho_i, rho_e = compute_densities(state)
    return rho_a - rho_i - rho_e


def build_initial_state(Lx: int, Ly: int, fraction: float, seed: int) -> np.ndarray:
    """Random fraction of sites set to BONDING, rest EMPTY."""
    rng = np.random.default_rng(seed)
    state = np.zeros((Lx, Ly), dtype=np.uint32)
    n_active = int(round(fraction * Lx * Ly))
    if n_active > 0:
        idx = rng.choice(Lx * Ly, n_active, replace=False)
        state.ravel()[idx] = BONDING
    return state


def compute_chi(m2_mean: float, m_mean: float, n_sites: int, beta: float) -> float:
    """χ = (N/T)(<m²> - <m>²) with T = 1/β."""
    temperature = 1.0 / beta
    return (n_sites / temperature) * (m2_mean - m_mean**2)


def get_next_id(csv_path: str) -> int:
    if not os.path.isfile(csv_path):
        return 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        existing_ids = [int(r["id"]) for r in reader if r.get("id", "").strip()]
    return max(existing_ids) + 1 if existing_ids else 0


def append_to_csv(csv_path: str, rows: list[dict]) -> None:
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})


def run_replica(args: tuple) -> dict:
    (
        replica_id,
        run_id,
        seed,
        params,
        run_settings,
        outdir,
    ) = args

    beta = run_settings["beta"]
    epsilon = params["epsilon"]
    delta_mu = params["delta_mu"]
    delta_f = params["delta_f"]
    k = params["k"]
    scheme = params["scheme"]
    mu = params["mu"]
    Lx = int(params["Lx"])
    Ly = int(params["Ly"])
    n_sites = Lx * Ly

    eq_time = run_settings["eq_time"]
    prod_time = run_settings["prod_time"]
    n_chunks = run_settings.get("prod_chunks", 20)
    initial_fraction = run_settings.get("initial_fraction", 0.8)

    inert_fugacity = np.exp(beta * (mu + delta_f))
    bonding_fugacity = np.exp(beta * mu)

    chain = HeteroChain(
        beta,
        epsilon,
        delta_mu,
        inert_fugacity,
        bonding_fugacity,
        k,
        scheme,
    )

    boundary = Periodic()
    state = build_initial_state(Lx, Ly, initial_fraction, seed)

    scratch_dir = os.path.join(outdir, f"_scratch_{replica_id}")

    simulate(state, boundary, chain, [], [Time(eq_time)], seed, scratch_dir)
    state = load.final_state(scratch_dir)
    print(f"[susceptibility_runner] replica={replica_id} equilibration done", flush=True)

    chunk_time = prod_time / n_chunks
    m_samples: list[float] = []
    cumulative_time = 0.0

    for chunk_idx in range(n_chunks):
        chunk_seed = seed + 1 + chunk_idx
        simulate(state, boundary, chain, [], [Time(chunk_time)], chunk_seed, scratch_dir)
        state = load.final_state(scratch_dir)
        cumulative_time += load.final_time(scratch_dir)

        m_t = compute_m(state)
        m_samples.append(m_t)
        print(
            f"[susceptibility_runner] replica={replica_id} chunk {chunk_idx + 1}/{n_chunks} "
            f"m={m_t:.4f} t={cumulative_time:.1f}",
            flush=True,
        )

    m_arr = np.asarray(m_samples, dtype=float)
    m_mean = float(np.mean(m_arr))
    m2_mean = float(np.mean(m_arr**2))
    m4_mean = float(np.mean(m_arr**4))
    chi = compute_chi(m2_mean, m_mean, n_sites, beta)

    np.save(os.path.join(outdir, f"final_lattice_{run_id}.npy"), state)
    shutil.rmtree(scratch_dir, ignore_errors=True)

    return {
        "id": run_id,
        "replica_id": replica_id,
        "epsilon": epsilon,
        "delta_f": delta_f,
        "delta_mu": delta_mu,
        "k": k,
        "scheme": scheme,
        "L": Lx,
        "Lx": Lx,
        "Ly": Ly,
        "mu": mu,
        "mu_coex_SIM": params.get("mu_coex_SIM", mu),
        "m_mean": m_mean,
        "m2_mean": m2_mean,
        "m4_mean": m4_mean,
        "chi": chi,
        "beta": beta,
        "eq_time": eq_time,
        "prod_time": prod_time,
        "prod_chunks": n_chunks,
        "initial_fraction": initial_fraction,
        "seed": seed,
        "time": cumulative_time,
    }


def summarize_replicas(results: list[dict]) -> None:
    if not results:
        return
    chi_vals = [r["chi"] for r in results]
    chi_mean = float(np.mean(chi_vals))
    chi_stderr = float(np.std(chi_vals, ddof=1) / np.sqrt(len(chi_vals))) if len(chi_vals) > 1 else 0.0
    print(
        f"[susceptibility_runner] chi across {len(results)} replicas: "
        f"{chi_mean:.6f} ± {chi_stderr:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Square-lattice susceptibility production runner")
    parser.add_argument("json_path", help="Path to susceptibility production job JSON")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: susceptibility_results/susceptibility_{L}x{L}_.../)",
    )
    args = parser.parse_args()

    with open(args.json_path) as f:
        params = json.load(f)

    run_settings = params["run_settings"]
    num_parallel_runs = run_settings["num_parallel_runs"]
    seed_base = run_settings["seed_base"]

    results_base = params.get("results_base", PROD_RESULTS_BASE)
    outdir = args.outdir or susceptibility_prod_dir(params, base=results_base)
    os.makedirs(outdir, exist_ok=True)

    eps = params["epsilon"]
    l_val = params["Lx"]
    mu_val = params["mu"]
    print(
        f"[susceptibility_runner] START {args.json_path} "
        f"eps={eps} L={l_val} mu={mu_val} replicas={num_parallel_runs} outdir={outdir}",
        flush=True,
    )

    csv_path = os.path.join(outdir, SUSCEPTIBILITY_DATA_CSV)
    next_id = get_next_id(csv_path)

    tasks = []
    for replica_id in range(num_parallel_runs):
        run_id = next_id + replica_id
        seed = seed_base + run_id * 2
        tasks.append((replica_id, run_id, seed, params, run_settings, outdir))

    with mp.Pool(processes=num_parallel_runs) as pool:
        results = pool.map(run_replica, tasks)

    results.sort(key=lambda r: r["id"])
    append_to_csv(csv_path, results)
    summarize_replicas(results)

    print(f"Wrote {len(results)} rows to {csv_path}", flush=True)
    for r in results:
        print(
            f"  replica {r['replica_id']}: m={r['m_mean']:.4f} "
            f"m2={r['m2_mean']:.4f} chi={r['chi']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
