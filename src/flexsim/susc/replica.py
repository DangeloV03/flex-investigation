"""Run one replica: equilibrate, then produce in chunks, recording m(t) and E(t)."""

from __future__ import annotations

import os
import shutil

import numpy as np
from lattice_gas import load
from lattice_gas.boundary_condition import Periodic
from lattice_gas.ending_criterion import Time
from lattice_gas.markov_chain import HeteroChain
from lattice_gas.simulate import simulate

from flexsim.susc.lattice import (
    build_initial_state,
    compute_chi,
    compute_chi_err,
    compute_densities,
    compute_energy,
    replica_fraction,
    sem,
)
from flexsim.susc.run_files import (
    ckpt_path,
    load_checkpoint,
    load_timeseries_csv,
    write_checkpoint,
    save_timeseries_csv,
)


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
    base_fraction = run_settings.get("initial_fraction", 0.5)
    initial_fraction = replica_fraction(base_fraction, run_id)

    # Resume-mode: per-task resume info injected by main() into run_settings.
    resume_dir: str | None = run_settings.get("resume_dir")
    resume_id: int | None = run_settings.get("resume_id")
    prior_prod_time: float = float(run_settings.get("prior_prod_time", 0.0))
    prior_wall_time: float = float(run_settings.get("prior_wall_time", 0.0))
    ckpt_every = int(run_settings.get("checkpoint_every", 0) or 0)
    ckpt_file = ckpt_path(outdir, run_id)
    ckpt = load_checkpoint(ckpt_file) if ckpt_every > 0 and os.path.isfile(ckpt_file) else None
    if ckpt is not None and (
        ckpt["resume_id"] != resume_id or ckpt["state"].shape != (Lx, Ly)
    ):
        print(
            f"[susc.runner] replica={replica_id} checkpoint {ckpt_file} does not "
            f"match this run (resume_id/shape); starting fresh",
            flush=True,
        )
        ckpt = None

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
    scratch_dir = os.path.join(outdir, f"_scratch_{replica_id}")

    if resume_dir is not None and resume_id is not None:
        # Resume mode: load prior lattice, skip equilibration.
        lattice_path = os.path.join(resume_dir, f"final_lattice_{resume_id}.npy")
        prior_ts_path = os.path.join(resume_dir, f"m_timeseries_{resume_id}.csv")
        state = np.load(lattice_path)
        prior_chunks = load_timeseries_csv(prior_ts_path)
        print(
            f"[susc.runner] replica={replica_id} run_id={run_id} "
            f"RESUME from id={resume_id} ({len(prior_chunks)} prior chunks)",
            flush=True,
        )
        eq_time = 0.0
    elif ckpt is not None:
        # Killed mid-run earlier: lattice + chunk history come from the checkpoint.
        state = ckpt["state"]
        prior_chunks = []
        print(
            f"[susc.runner] replica={replica_id} run_id={run_id} "
            f"CHECKPOINT resume at chunk {len(ckpt['chunks'])}/{n_chunks}",
            flush=True,
        )
    else:
        state = build_initial_state(Lx, Ly, initial_fraction, seed)
        prior_chunks = []
        print(
            f"[susc.runner] replica={replica_id} run_id={run_id} "
            f"initial_fraction={initial_fraction}",
            flush=True,
        )
        simulate(state, boundary, chain, [], [Time(eq_time)], seed, scratch_dir)
        state = load.final_state(scratch_dir)
        print(f"[susc.runner] replica={replica_id} equilibration done", flush=True)
        if ckpt_every > 0:
            write_checkpoint(ckpt_file, state, [], 0.0, resume_id)

    chunk_time = prod_time / n_chunks
    new_chunk_records: list[dict] = []
    cumulative_time = 0.0
    chunk_offset = len(prior_chunks)  # so new chunk indices continue from prior
    if ckpt is not None:
        state = ckpt["state"]
        new_chunk_records = ckpt["chunks"]
        cumulative_time = ckpt["cumulative_time"]

    for chunk_idx in range(len(new_chunk_records), n_chunks):
        chunk_seed = seed + 1 + chunk_idx
        simulate(state, boundary, chain, [], [Time(chunk_time)], chunk_seed, scratch_dir)
        state = load.final_state(scratch_dir)
        cumulative_time += load.final_time(scratch_dir)

        rho_B, rho_I, rho_E = compute_densities(state)
        m_t = rho_B - rho_I - rho_E
        e_t = compute_energy(state, beta, epsilon, mu, delta_f)

        new_chunk_records.append({
            "chunk": chunk_offset + chunk_idx,
            "rho_bonding": rho_B,
            "rho_inert": rho_I,
            "rho_empty": rho_E,
            "m": m_t,
            "energy": e_t,
        })
        print(
            f"[susc.runner] replica={replica_id} chunk {chunk_idx + 1}/{n_chunks} "
            f"rho_B={rho_B:.4f} rho_I={rho_I:.6f} rho_E={rho_E:.4f} m={m_t:.4f} e={e_t:.2f} t={cumulative_time:.1f}",
            flush=True,
        )
        if ckpt_every > 0 and (chunk_idx + 1) % ckpt_every == 0 and chunk_idx + 1 < n_chunks:
            write_checkpoint(ckpt_file, state, new_chunk_records, cumulative_time, resume_id)

    # Combine prior + new for statistics and saved timeseries.
    all_chunks = prior_chunks + new_chunk_records
    m_arr = np.asarray([float(c["m"]) for c in all_chunks], dtype=float)
    e_arr = np.asarray([float(c["energy"]) for c in all_chunks], dtype=float)

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

    e2_arr = e_arr ** 2
    e_mean = float(np.mean(e_arr))
    e2_mean = float(np.mean(e2_arr))
    e_mean_err = sem(e_arr)
    e2_mean_err = sem(e2_arr)

    np.save(os.path.join(outdir, f"final_lattice_{run_id}.npy"), state)
    shutil.rmtree(scratch_dir, ignore_errors=True)

    # Save combined timeseries (full history: prior + new chunks).
    ts_csv = os.path.join(outdir, f"m_timeseries_{run_id}.csv")
    save_timeseries_csv(ts_csv, all_chunks)

    total_prod_time = prior_prod_time + prod_time
    total_wall_time = prior_wall_time + cumulative_time

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
        "prod_time": total_prod_time,
        "prod_chunks": len(all_chunks),
        "initial_fraction": initial_fraction,
        "seed": seed,
        "time": total_wall_time,
        "resume_id": "" if resume_id is None else str(resume_id),
    }
