"""
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
  - final_lattice_{id}.npy        — final lattice snapshot

Optional checkpointing (--checkpoint-every N, off by default): every N chunks each
replica atomically writes ckpt_{id}.npz (lattice + chunk history so far).  If the
job is killed at the Slurm wall, re-running the same command picks up those
orphaned checkpoints (ids with no row in susceptibility_data.csv) and continues
from them; a checkpoint is deleted once its replica's row has been written.

Re-running the same (ε, L) appends new replicas to susceptibility_data.csv
(run IDs continue from the max existing ID); it never overwrites.

Usage:
    python run/susc.py run-one --epsilon -1.76 --L 64 --cpus 16 \\
        --results-base susceptibility/results/exact_2026-07-02

Modules: susc/lattice.py (energy, initial state, chi), susc/run_files.py
(time series, replica table, checkpoints), susc/replica.py (one replica's run).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import numpy as np

from flexsim.susc.layout import (
    ISING_DELTA_F,
    ISING_DELTA_MU,
    ISING_K,
    ISING_SCHEME,
    PROD_RESULTS_BASE,
    SUSCEPTIBILITY_DATA_CSV,
    read_susceptibility_csv,
    susceptibility_prod_dir,
)
from flexsim.susc.run_files import (
    ckpt_path,
    drop_superseded_timeseries,
    orphan_checkpoint_ids,
    append_to_csv,
    get_next_id,
)
from flexsim.susc.replica import run_replica


def summarize_replicas(results: list[dict]) -> None:
    if not results:
        return
    chi_vals = [r["chi"] for r in results]
    chi_mean = float(np.mean(chi_vals))
    chi_stderr = float(np.std(chi_vals, ddof=1) / np.sqrt(len(chi_vals))) if len(chi_vals) > 1 else 0.0
    print(
        f"[susc.runner] chi across {len(results)} replicas: "
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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "Every N production chunks, atomically save ckpt_<id>.npz (lattice + chunk "
            "history) so a job killed at the Slurm wall can continue where it stopped "
            "when the same command is re-run. 0 (default) disables checkpointing."
        ),
    )
    parser.add_argument(
        "--initial-fraction",
        type=float,
        default=0.8,
        help=(
            "High-side BONDING fill fraction. Even run_ids seed this fraction "
            "(e.g. 0.8 = 80/20), odd run_ids seed its mirror (0.2 = 20/80), so "
            "replicas sample both branches of the magnetization distribution."
        ),
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--scheme", default=ISING_SCHEME)
    parser.add_argument("--delta-f", type=float, default=ISING_DELTA_F)
    parser.add_argument("--delta-mu", type=float, default=ISING_DELTA_MU)
    parser.add_argument("--k", type=float, default=ISING_K)
    parser.add_argument("--results-base", default=PROD_RESULTS_BASE)
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: SUSC_RUNS/_{Lx}_{Ly}_S{n}_.../_eps/)",
    )
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Directory with final_lattice_*.npy and m_timeseries_*.csv to resume from",
    )
    parser.add_argument(
        "--resume-ids",
        type=int,
        nargs="+",
        default=None,
        help="Replica IDs to resume from (one per CPU slot; auto-detected from CSV if omitted)",
    )
    args = parser.parse_args()

    if args.mu is not None:
        mu = args.mu
    elif args.delta_mu == 0.0 and args.delta_f <= -15.0:
        # Ising limit: exact coexistence -ln(e^{2|ε|} - e^{βΔf}) is 2ε to <1e-6.
        mu = 2.0 * args.epsilon
    else:
        # μ = 2ε is NOT coexistence here (S1B ran 53 rounds at βΔf=0 this way,
        # pinned in one phase at every L ≥ 32).  Make the caller choose.
        raise SystemExit(
            f"--mu is required for delta_f={args.delta_f}, delta_mu={args.delta_mu}: "
            f"μ=2ε is coexistence only in the Ising limit (βΔf << 0, Δμ=0). "
            f"Launch with `run/susc.py sweep --mu-exact` or --mu-source."
        )

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
        "checkpoint_every": args.checkpoint_every,
    }

    num_parallel_runs = run_settings["num_parallel_runs"]
    num_batches = run_settings["num_batches"]
    seed_base = run_settings["seed_base"]

    outdir = args.outdir or susceptibility_prod_dir(params, base=args.results_base)
    os.makedirs(outdir, exist_ok=True)

    # Resolve resume IDs: auto-detect the latest-generation replicas from the CSV.
    resume_ids: list[int] | None = None
    prior_stats: dict[int, dict] = {}  # resume_id → {prior_prod_time, prior_wall_time}
    if args.resume_dir is not None:
        resume_dir_resolved = args.resume_dir
        prior_csv = os.path.join(resume_dir_resolved, SUSCEPTIBILITY_DATA_CSV)
        prior_rows = read_susceptibility_csv(prior_csv)
        if not prior_rows:
            raise SystemExit(f"No data found in {prior_csv} for resume")
        if args.resume_ids is not None:
            resume_ids = args.resume_ids
        else:
            # Take the replicas with the highest prod_time (latest generation).
            max_pt = max(float(r.get("prod_time", 0) or 0) for r in prior_rows)
            latest = [r for r in prior_rows
                      if abs(float(r.get("prod_time", 0) or 0) - max_pt) < 1.0]
            latest.sort(key=lambda r: int(r.get("id", 0) or 0))
            resume_ids = [int(r["id"]) for r in latest]
        assert resume_ids is not None  # set by either branch above
        for r in prior_rows:
            rid = int(r.get("id", -1) or -1)
            if rid in resume_ids:
                prior_stats[rid] = {
                    "prior_prod_time": float(r.get("prod_time", 0) or 0),
                    "prior_wall_time": float(r.get("time", 0) or 0),
                }
        # When resuming, ignore --cpus and use the count of resume_ids.
        num_parallel_runs = len(resume_ids)
        run_settings = dict(run_settings)
        run_settings["num_parallel_runs"] = num_parallel_runs
        print(
            f"[susc.runner] RESUME mode: {num_parallel_runs} replicas from "
            f"{resume_dir_resolved}, ids={resume_ids}",
            flush=True,
        )

    print(
        f"[susc.runner] START "
        f"eps={args.epsilon} L={args.L} mu={mu} "
        f"replicas={num_parallel_runs} batches={num_batches} "
        f"total={num_parallel_runs * num_batches} outdir={outdir}",
        flush=True,
    )

    csv_path = os.path.join(outdir, SUSCEPTIBILITY_DATA_CSV)

    for batch_idx in range(num_batches):
        next_id = get_next_id(csv_path)
        # Replicas killed mid-run left ckpt_<id>.npz without a CSV row: re-adopt those
        # ids first (same id -> same seed and checkpoint) so their work is not lost.
        orphan_ids: list[int] = []
        if args.checkpoint_every > 0 and resume_ids is None:
            done_ids = {
                int(r["id"]) for r in read_susceptibility_csv(csv_path)
                if str(r.get("id", "")).strip()
            }
            orphan_ids = orphan_checkpoint_ids(outdir, done_ids)[:num_parallel_runs]
            if orphan_ids:
                next_id = max(next_id, max(orphan_ids) + 1)
                print(
                    f"[susc.runner] resuming {len(orphan_ids)} checkpointed "
                    f"replicas: ids={orphan_ids}",
                    flush=True,
                )
        slot_run_ids = orphan_ids + list(range(next_id, next_id + num_parallel_runs - len(orphan_ids)))
        print(
            f"[susc.runner] batch {batch_idx + 1}/{num_batches}: "
            f"run_ids {slot_run_ids}",
            flush=True,
        )

        tasks = []
        for slot_idx in range(num_parallel_runs):
            run_id = slot_run_ids[slot_idx]
            seed = seed_base + run_id * 2
            task_settings = dict(run_settings)
            if resume_ids is not None:
                rid = resume_ids[slot_idx]
                stats = prior_stats.get(rid, {})
                task_settings["resume_dir"] = args.resume_dir
                task_settings["resume_id"] = rid
                task_settings["prior_prod_time"] = stats.get("prior_prod_time", 0.0)
                task_settings["prior_wall_time"] = stats.get("prior_wall_time", 0.0)
                task_settings["eq_time"] = 0.0  # skip equilibration on resume
            tasks.append((slot_idx, run_id, seed, params, task_settings, outdir))

        # Append each replica's row as it lands, not once the whole batch is
        # done: a job killed at the wall used to discard every finished replica
        # with it (up to a full day of compute per job).
        results = []
        with mp.Pool(processes=num_parallel_runs) as pool:
            for result in pool.imap_unordered(run_replica, tasks):
                append_to_csv(csv_path, [result])
                results.append(result)
                try:
                    os.remove(ckpt_path(outdir, result["id"]))
                except OSError:
                    pass
                print(
                    f"[susc.runner] replica {result['replica_id']} "
                    f"(id={result['id']}) written to {csv_path} "
                    f"[{len(results)}/{num_parallel_runs}]",
                    flush=True,
                )
                # The new row now points at the full history; the file it
                # resumed from is a redundant prefix of it.
                if args.resume_dir and str(result.get("resume_id", "")).strip():
                    drop_superseded_timeseries(
                        os.path.join(args.resume_dir, f"m_timeseries_{result['resume_id']}.csv"),
                        os.path.join(outdir, f"m_timeseries_{result['id']}.csv"),
                    )

        results.sort(key=lambda r: r["id"])
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
