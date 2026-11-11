"""One replica of the equilibrium 128x128 spin-spin correlation run (resumable).

Physics (same as the summer correlation runs): homo, beta*df = 0, k = 1, beta*dmu = 0,
exact coexistence mu = -ln(e^{2|eps|} - e^{df}). Spin s = +1 on BONDING (active),
-1 otherwise, so the active-active correlation is C(r) = G(r)/4.

Start: random lattice, half active (<m> ~ 0), the rest empty. Equilibrate for EQ_TIME,
then save a snapshot every CHUNK_TIME until PROD_TIME. Everything is checkpointed, so a
killed or timed-out job resumes where it stopped when it is resubmitted.

Output, per replica:  <out>/eps<tag>/replica_<r>/
  snap_00000.npy ...   uint8 lattice (0 empty, 1 inert, 2 bonding)
  timeseries.csv       chunk, phase, t, rho_B, m  (one row per chunk, eq and prod)
  state.npy            latest lattice (resume point)
  progress.json        phase, chunk, t, done

Usage:
  python spin_corr_run.py --eps -1.7627 --replica 0 --out <dir>
  python spin_corr_run.py --eps -1.7627 --replica 0 --out <dir> --smoke   # L=16, seconds
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
from lattice_gas import load
from lattice_gas.boundary_condition import Periodic
from lattice_gas.ending_criterion import Time
from lattice_gas.markov_chain import HeteroChain
from lattice_gas.simulate import simulate

EMPTY, INERT, BONDING = 0, 1, 2


def exact_mu(eps, df):
    return -np.log(np.exp(2 * abs(eps)) - np.exp(df))


def eps_tag(eps):
    return "eps" + f"{eps:.4f}".replace("-", "m").replace(".", "p")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--replica", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--eq-time", type=float, default=1_000_000)
    p.add_argument("--prod-time", type=float, default=20_000_000)
    p.add_argument("--chunk-time", type=float, default=1_000)
    p.add_argument("--eq-chunk", type=float, default=10_000, help="checkpoint interval during eq")
    p.add_argument("--delta-f", type=float, default=0.0)
    p.add_argument("--k", type=float, default=1.0)
    p.add_argument("--seed-base", type=int, default=20261110)
    p.add_argument("--smoke", action="store_true", help="L=16, short times: checks the pipeline")
    a = p.parse_args()
    if a.smoke:
        a.L, a.eq_time, a.prod_time, a.chunk_time, a.eq_chunk = 16, 200, 500, 50, 100

    mu = exact_mu(a.eps, a.delta_f)
    chain = HeteroChain(1.0, a.eps, 0.0, np.exp(mu + a.delta_f), np.exp(mu), a.k, "homo")
    boundary = Periodic()

    rdir = os.path.join(a.out, eps_tag(a.eps), f"replica_{a.replica}")
    os.makedirs(rdir, exist_ok=True)
    prog_path = os.path.join(rdir, "progress.json")
    ts_path = os.path.join(rdir, "timeseries.csv")
    state_path = os.path.join(rdir, "state.npy")
    scratch = os.path.join(rdir, "_scratch")

    n_eq = int(round(a.eq_time / a.eq_chunk))
    n_prod = int(round(a.prod_time / a.chunk_time))
    seed0 = a.seed_base + 1_000_000 * a.replica + int(round(1e4 * abs(a.eps)))

    if os.path.isfile(prog_path):
        prog = json.load(open(prog_path))
        if prog.get("done"):
            print(f"[spin_corr] {rdir} already done", flush=True)
            return
        state = np.load(state_path).astype(np.uint32)
        print(f"[spin_corr] resume {prog}", flush=True)
    else:
        rng = np.random.default_rng(seed0)
        state = np.zeros((a.L, a.L), dtype=np.uint32)
        state[rng.random((a.L, a.L)) < 0.5] = BONDING
        prog = {"phase": "eq", "chunk": 0, "t": 0.0, "done": False}
        meta = dict(vars(a), mu=mu, scheme="homo", delta_mu=0.0, n_eq_chunks=n_eq,
                    n_prod_chunks=n_prod, seed0=seed0)
        json.dump(meta, open(os.path.join(rdir, "meta.json"), "w"), indent=2)
        with open(ts_path, "w") as f:
            f.write("chunk,phase,t,rho_B,m\n")
    print(f"[spin_corr] eps={a.eps} L={a.L} mu={mu:.6f} replica={a.replica} "
          f"eq={a.eq_time} prod={a.prod_time} every {a.chunk_time}", flush=True)

    t_wall = time.time()
    while not prog["done"]:
        phase, i = prog["phase"], prog["chunk"]
        dt = a.eq_chunk if phase == "eq" else a.chunk_time
        seed = seed0 + (0 if phase == "eq" else 10_000_000) + i + 1
        simulate(state, boundary, chain, [], [Time(dt)], seed, scratch)
        state = load.final_state(scratch).astype(np.uint32)
        prog["t"] += float(load.final_time(scratch))

        rho = float(np.mean(state == BONDING))
        if phase == "prod":
            np.save(os.path.join(rdir, f"snap_{i:05d}.npy"), state.astype(np.uint8))
        with open(ts_path, "a") as f:
            f.write(f"{i},{phase},{prog['t']:.3f},{rho:.6f},{2 * rho - 1:.6f}\n")

        # advance the counter, write the resume point atomically
        prog["chunk"] = i + 1
        if phase == "eq" and prog["chunk"] >= n_eq:
            prog["phase"], prog["chunk"] = "prod", 0
        elif phase == "prod" and prog["chunk"] >= n_prod:
            prog["done"] = True
        np.save(state_path + ".tmp.npy", state.astype(np.uint8))
        os.replace(state_path + ".tmp.npy", state_path)
        json.dump(prog, open(prog_path + ".tmp", "w"))
        os.replace(prog_path + ".tmp", prog_path)

        if prog["chunk"] % 50 == 0 or prog["done"]:
            print(f"[spin_corr] {prog['phase']} chunk {prog['chunk']} t={prog['t']:.0f} "
                  f"m={2 * rho - 1:+.3f} wall={time.time() - t_wall:.0f}s", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"[spin_corr] DONE {rdir}", flush=True)


if __name__ == "__main__":
    main()
