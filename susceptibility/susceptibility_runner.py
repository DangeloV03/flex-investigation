"""
susceptibility_runner.py

Long production runs on a square L×L lattice at μ_coex for susceptibility
measurement. Driven entirely by command-line flags (one (ε, L) per invocation).

For each replica:
  - Equilibrate (eq_time), discard.
  - Production in chunks; record per-species densities (rho_bonding, rho_inert, rho_empty)
    at the end of each chunk.
  - Derive m = rho_bonding - rho_inert - rho_empty and M = N*m from raw densities.
  - Compute time-averaged ⟨m⟩, ⟨m²⟩, ⟨m⁴⟩ (order parameter / Binder) and
    χ = (N/T)(⟨m²⟩ - ⟨m⟩²) with T = 1/β.

Outputs (per job directory):
  - susceptibility_data.csv       — one row per replica with aggregate statistics
  - m_timeseries_{id}.csv         — per-chunk (chunk, rho_bonding, rho_inert, rho_empty, m)
  - m_timeseries_{id}.png         — m vs chunk plot per replica
  - final_lattice_{id}.npy        — final lattice snapshot

Re-running the same (ε, L) appends new replicas to susceptibility_data.csv
(run IDs continue from the max existing ID); it never overwrites.

Usage:
    python susceptibility_runner.py --epsilon -1.76 --L 64 --cpus 16 \\
        --results-base susceptibility_results/exact_2026-07-02
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lattice_gas import load
from lattice_gas.boundary_condition import Periodic
from lattice_gas.ending_criterion import Time
from lattice_gas.markov_chain import HeteroChain
from lattice_gas.simulate import simulate

from susceptibility_paths import (
    ISING_DELTA_F,
    ISING_DELTA_MU,
    ISING_K,
    ISING_SCHEME,
    PROD_RESULTS_BASE,
    SUSCEPTIBILITY_CSV_FIELDS,
    SUSCEPTIBILITY_DATA_CSV,
    read_susceptibility_csv,
    susceptibility_prod_dir,
)

EMPTY, INERT, BONDING = 0, 1, 2

CSV_FIELDNAMES = SUSCEPTIBILITY_CSV_FIELDS

TIMESERIES_FIELDNAMES = ["chunk", "rho_bonding", "rho_inert", "rho_empty", "m", "energy"]


def compute_densities(state: np.ndarray) -> tuple[float, float, float]:
    """Return (rho_bonding, rho_inert, rho_empty) — fractions of each species."""
    flat = state.ravel()
    n = flat.size
    rho_bonding = float(np.count_nonzero(flat == BONDING)) / n
    rho_inert = float(np.count_nonzero(flat == INERT)) / n
    rho_empty = float(np.count_nonzero(flat == EMPTY)) / n
    return rho_bonding, rho_inert, rho_empty


def compute_energy(
    state: np.ndarray, beta: float, epsilon: float, mu: float, delta_f: float
) -> float:
    """Total dimensionless energy βE (equals E at β=1).

    E = (βε/2) Σᵢ_{bonding} n_bonding_neighbors(i) − βμ·n_bonding − β(μ+Δf)·n_inert

    Matches the Rust Energy::record() implementation exactly.
    """

    # 2d Grid of 0's and 1's where 1's are the bonding sites
    bonding = (state == BONDING).astype(np.float64)
    bonding_neighbors = (
        np.roll(bonding, 1, axis=0) +
        np.roll(bonding, -1, axis=0) +
        np.roll(bonding, 1, axis=1) +
        np.roll(bonding, -1, axis=1)
    )
    e_interact = 0.5 * beta * epsilon * float(np.dot(bonding.ravel(), bonding_neighbors.ravel()))
    flat = state.ravel()
    n_bonding = float(np.count_nonzero(flat == BONDING))
    n_inert = float(np.count_nonzero(flat == INERT))
    e_chem = -beta * mu * n_bonding - beta * (mu + delta_f) * n_inert
    return e_interact + e_chem


def build_initial_state(Lx: int, Ly: int, fraction: float, seed: int) -> np.ndarray:
    """Random fraction of sites set to BONDING, rest EMPTY."""
    rng = np.random.default_rng(seed)
    state = np.zeros((Lx, Ly), dtype=np.uint32)
    n_active = int(round(fraction * Lx * Ly))
    if n_active > 0:
        idx = rng.choice(Lx * Ly, n_active, replace=False)
        state.ravel()[idx] = BONDING
    return state


def sem(values: np.ndarray) -> float:
    """Standard error of the mean."""
    n = len(values)
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


def compute_chi_err(
    m_abs_mean: float,
    m_abs_mean_err: float,
    m2_mean_err: float,
    n_sites: int,
    beta: float,
) -> float:
    """Delta-method SEM for χ = N·β·(⟨m²⟩-⟨|m|⟩²)."""
    factor = n_sites * beta
    var = (factor * m2_mean_err) ** 2 + (2.0 * factor * m_abs_mean * m_abs_mean_err) ** 2
    return float(np.sqrt(var))


def compute_chi(m2_mean: float, m_abs_mean: float, n_sites: int, beta: float) -> float:
    """χ = (1/NT)(⟨M²⟩-⟨|M|⟩²) with M = N·m, |M| = N·|m|, T = 1/β.

    ⟨M²⟩ = N²⟨m²⟩ and ⟨|M|⟩² = N²⟨|m|⟩², so this equals N·β·(⟨m²⟩-⟨|m|⟩²).
    Using ⟨|m|⟩ (not ⟨m⟩) removes the spurious symmetry-degeneracy term that small,
    flipping lattices would otherwise pick up — the connected (FSS) susceptibility.
    """
    M2_mean = n_sites ** 2 * m2_mean
    absM_mean = n_sites * m_abs_mean
    return beta / n_sites * (M2_mean - absM_mean ** 2)


def save_timeseries_csv(path: str, chunks: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMESERIES_FIELDNAMES)
        writer.writeheader()
        writer.writerows(chunks)


def save_timeseries_plot(path: str, chunks: list[dict], run_id: int, epsilon: float, L: int) -> None:
    chunk_indices = [c["chunk"] for c in chunks]
    m_values = [c["m"] for c in chunks]
    e_values = [c["energy"] for c in chunks]
    fig, (ax_m, ax_e) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax_m.plot(chunk_indices, m_values, "o-", markersize=4)
    ax_m.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    ax_m.set_ylabel("m")
    ax_m.grid(True, alpha=0.3)
    ax_e.plot(chunk_indices, e_values, "o-", markersize=4, color="tab:orange")
    ax_e.set_xlabel("Chunk")
    ax_e.set_ylabel("E")
    ax_e.grid(True, alpha=0.3)
    fig.suptitle(f"replica {run_id}  ε={epsilon}  L={L}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def get_next_id(csv_path: str) -> int:
    existing_ids = [
        int(r["id"])
        for r in read_susceptibility_csv(csv_path)
        if str(r.get("id", "")).strip()
    ]
    return max(existing_ids) + 1 if existing_ids else 0


def append_to_csv(csv_path: str, rows: list[dict]) -> None:
    existing = read_susceptibility_csv(csv_path)
    normalized = [{field: row.get(field, "") for field in CSV_FIELDNAMES} for row in rows]
    all_rows = existing + normalized
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)


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
    n_chunks = run_settings.get("prod_chunks", 1000)
    initial_fraction = run_settings.get("initial_fraction", 0.5)

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

    print(
        f"[susceptibility_runner] replica={replica_id} run_id={run_id} "
        f"initial_fraction={initial_fraction}",
        flush=True,
    )
    simulate(state, boundary, chain, [], [Time(eq_time)], seed, scratch_dir)
    state = load.final_state(scratch_dir)
    print(f"[susceptibility_runner] replica={replica_id} equilibration done", flush=True)

    chunk_time = prod_time / n_chunks
    chunk_records: list[dict] = []
    rho_B_samples: list[float] = []
    rho_I_samples: list[float] = []
    rho_E_samples: list[float] = []
    m_samples: list[float] = []
    e_samples: list[float] = []
    cumulative_time = 0.0

    for chunk_idx in range(n_chunks):
        chunk_seed = seed + 1 + chunk_idx
        simulate(state, boundary, chain, [], [Time(chunk_time)], chunk_seed, scratch_dir)
        state = load.final_state(scratch_dir)
        cumulative_time += load.final_time(scratch_dir)

        rho_B, rho_I, rho_E = compute_densities(state)
        m_t = rho_B - rho_I - rho_E
        e_t = compute_energy(state, beta, epsilon, mu, delta_f)

        rho_B_samples.append(rho_B)
        rho_I_samples.append(rho_I)
        rho_E_samples.append(rho_E)
        m_samples.append(m_t)
        e_samples.append(e_t)
        chunk_records.append({
            "chunk": chunk_idx,
            "rho_bonding": rho_B,
            "rho_inert": rho_I,
            "rho_empty": rho_E,
            "m": m_t,
            "energy": e_t,
        })
        print(
            f"[susceptibility_runner] replica={replica_id} chunk {chunk_idx + 1}/{n_chunks} "
            f"rho_B={rho_B:.4f} rho_I={rho_I:.6f} rho_E={rho_E:.4f} m={m_t:.4f} e={e_t:.2f} t={cumulative_time:.1f}",
            flush=True,
        )

    m_arr = np.asarray(m_samples, dtype=float)
    abs_m_arr = np.abs(m_arr)
    m2_arr = m_arr ** 2
    m4_arr = m_arr ** 4
    m_mean = float(np.mean(m_arr))
    abs_m_mean = float(np.mean(abs_m_arr))
    m2_mean = float(np.mean(m2_arr))
    m4_mean = float(np.mean(m4_arr))
    m_mean_err = sem(m_arr)
    abs_m_mean_err = sem(abs_m_arr)
    m2_mean_err = sem(m2_arr)
    m4_mean_err = sem(m4_arr)
    chi = compute_chi(m2_mean, abs_m_mean, n_sites, beta)
    chi_err = compute_chi_err(abs_m_mean, abs_m_mean_err, m2_mean_err, n_sites, beta)

    e_arr = np.asarray(e_samples, dtype=float)
    e2_arr = e_arr ** 2
    e_mean = float(np.mean(e_arr))
    e2_mean = float(np.mean(e2_arr))
    e_mean_err = sem(e_arr)
    e2_mean_err = sem(e2_arr)

    np.save(os.path.join(outdir, f"final_lattice_{run_id}.npy"), state)
    shutil.rmtree(scratch_dir, ignore_errors=True)

    ts_csv = os.path.join(outdir, f"m_timeseries_{run_id}.csv")
    save_timeseries_csv(ts_csv, chunk_records)

    ts_png = os.path.join(outdir, f"m_timeseries_{run_id}.png")
    save_timeseries_plot(ts_png, chunk_records, run_id, epsilon, Lx)

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
        "mu_coex_FITTED": params.get("mu_coex_FITTED", mu),
        "m_mean": m_mean,
        "m_mean_err": m_mean_err,
        "m2_mean": m2_mean,
        "m2_mean_err": m2_mean_err,
        "m4_mean": m4_mean,
        "m4_mean_err": m4_mean_err,
        "chi": chi,
        "chi_err": chi_err,
        "e_mean": e_mean,
        "e_mean_err": e_mean_err,
        "e2_mean": e2_mean,
        "e2_mean_err": e2_mean_err,
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
    parser.add_argument("--epsilon", type=float, required=True, help="Interaction strength ε")
    parser.add_argument("--L", type=int, required=True, help="Square lattice side (Lx = Ly = L)")
    parser.add_argument(
        "--mu",
        type=float,
        default=None,
        help="Chemical potential; default is μ_coex_EXACT = 2*ε",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="Parallel replicas per batch (= SLURM_CPUS_PER_TASK)",
    )
    parser.add_argument("--num-batches", type=int, default=1, help="Sequential batches to append")
    parser.add_argument("--eq-time", type=float, default=100000.0)
    parser.add_argument("--prod-time", type=float, default=200000.0)
    parser.add_argument("--prod-chunks", type=int, default=2000)
    parser.add_argument("--seed-base", type=int, default=7000)
    parser.add_argument("--initial-fraction", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--scheme", default=ISING_SCHEME)
    parser.add_argument("--delta-f", type=float, default=ISING_DELTA_F)
    parser.add_argument("--delta-mu", type=float, default=ISING_DELTA_MU)
    parser.add_argument("--k", type=float, default=ISING_K)
    parser.add_argument("--results-base", default=PROD_RESULTS_BASE)
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: {results_base}/susceptibility_{L}x{L}_.../)",
    )
    args = parser.parse_args()

    mu = args.mu if args.mu is not None else 2.0 * args.epsilon

    params = {
        "epsilon": args.epsilon,
        "delta_f": args.delta_f,
        "delta_mu": args.delta_mu,
        "k": args.k,
        "scheme": args.scheme,
        "Lx": args.L,
        "Ly": args.L,
        "mu": mu,
        "results_base": args.results_base,
    }
    run_settings = {
        "beta": args.beta,
        "num_parallel_runs": args.cpus,
        "num_batches": args.num_batches,
        "seed_base": args.seed_base,
        "eq_time": args.eq_time,
        "prod_time": args.prod_time,
        "prod_chunks": args.prod_chunks,
        "initial_fraction": args.initial_fraction,
    }

    num_parallel_runs = run_settings["num_parallel_runs"]
    num_batches = run_settings["num_batches"]
    seed_base = run_settings["seed_base"]

    outdir = args.outdir or susceptibility_prod_dir(params, base=args.results_base)
    os.makedirs(outdir, exist_ok=True)

    print(
        f"[susceptibility_runner] START "
        f"eps={args.epsilon} L={args.L} mu={mu} "
        f"replicas={num_parallel_runs} batches={num_batches} "
        f"total={num_parallel_runs * num_batches} outdir={outdir}",
        flush=True,
    )

    csv_path = os.path.join(outdir, SUSCEPTIBILITY_DATA_CSV)

    for batch_idx in range(num_batches):
        next_id = get_next_id(csv_path)
        print(
            f"[susceptibility_runner] batch {batch_idx + 1}/{num_batches}: "
            f"run_ids {next_id}–{next_id + num_parallel_runs - 1}",
            flush=True,
        )

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

        print(
            f"Wrote {len(results)} rows to {csv_path} "
            f"(batch {batch_idx + 1}/{num_batches})",
            flush=True,
        )
        for r in results:
            print(
                f"  replica {r['replica_id']}: m={r['m_mean']:.4f}±{r['m_mean_err']:.4f} "
                f"chi={r['chi']:.4f}±{r['chi_err']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
