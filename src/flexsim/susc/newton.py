"""Newton search for the coexistence mu of each (eps, L) susceptibility point.

Mentor's proposal (susceptibility notes, section 3.1): start at the coex-fit mu, measure
m0 = <m> after every segment, and while |m0| > 0.05 correct mu with a damped Newton step

    beta*mu_new = beta*mu_old - tau * m0 / slope,      slope = d<m>/d(beta*mu)

then run on at the final mu until the replicas average J >= 10 jumps. There is no
minimum jump count before m0 is judged and no standard-error test (agreed with the
mentor): a point that only looked fine by chance is checked again at J >= 10. That run's chi
is the one used for FSS. Every (eps, L) has its own chain, because the finite-size
shift of coexistence differs per L.

Slope. The chi stored in susceptibility_data.csv uses <|m|> (the connected FSS chi), and
that is not d<m>/d(beta*mu). In equilibrium, P ~ exp(beta*mu * N_occupied + ...), so
    d<m>/d(beta*mu) = Cov(m, N_occupied) = N * Cov(m, rho_bonding + rho_inert),
which is (N/2) Var(m) when there are no inert particles. It is computed from signed m,
pooled over all replicas, which gives ~N m_s^2 / 2 near coexistence. From the second
iteration on, the secant between the last two iterations is used when it is positive,
clamped to [0.5, 5] x the fluctuation slope (two noisy <m> that barely differ give a tiny
secant and would otherwise make a huge step). Every step is damped by tau and capped at dmu_max.

Folder layout (one "pair dir" per (eps, L), named like a SUSC_RUNS run dir):

    <base>/_<L>_<L>_S<n>_DF.._DMU.._K../_<eps>/
        newton_state.json     iteration history, config, next action, job ids
        newton_log.csv        one row per decision
        iter_0/               runner output at mu_0 (susceptibility_data.csv, m_timeseries_*, ...)
        iter_1/               runner output at mu_1, started from iter_0's final lattices
        ...

Data from an older mu never enters a newer iteration (runner --fresh-history).
`combine` links each converged pair's final iteration into a SUSC_RUNS-shaped folder
that the existing plot and FSS commands read.

Each segment is one Slurm job, followed by a small `decide` job (afterany, so a job that
hit the wall still gets its chain continued). The decide job measures the iteration,
picks the next action and submits the next pair of jobs. Touch <base>/STOP to stop
every chain at its next decision.

Subcommands: start, segment (inside a job), decide (inside a job), status, kick, combine.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

from flexsim.grid import frange
from flexsim.paths import SUSC_SLURM_DIR
from flexsim.susc.layout import (
    SUSCEPTIBILITY_DATA_CSV,
    latest_row_per_replica,
    read_susceptibility_csv,
    susc_run_dir,
)
from flexsim.susc.mag_jumps import count_jumps

STATE_FILE = "newton_state.json"
LOG_FILE = "newton_log.csv"
STOP_FILE = "STOP"
NEWTON_SCRIPT = str(SUSC_SLURM_DIR / "newton.sh")
CHUNK_TIME = 100.0          # simulated time per recorded sample, as in every other susc run
SEED_STRIDE = 100_000       # seed_base offset per iteration, so iterations never share seeds

DEFAULT_CONFIG = {
    "tau": 0.2,             # Newton damping
    "dmu_max": 1e-3,        # cap on |delta(beta*mu)| per step (coex fit is good to ~2e-4)
    "secant_lo": 0.5,       # measured secant slope is clamped to [secant_lo, secant_hi] x
    "secant_hi": 5.0,       #   the fluctuation slope, so a noisy secant can't make a huge step
    "m_tol": 0.05,          # |<m>| tolerance (same cut as the plots' drop rule)
    "j_target": 10.0,       # mean jumps for the final chi
    "max_segments": 25,     # segments per iteration before giving up
    "max_iter": 20,         # mu values before giving up
    "max_retries": 2,       # reruns of a segment that produced no data
    "eq0": 1e6,             # first iteration: equilibration from random 0.8/0.2 fills
    "prod0": 2e6,           # first iteration: first production segment
    "eq_iter": 1e5,         # later iterations: re-equilibration at the new mu
    "seg_prod": 1e6,        # every later segment
    "seed_base": 7000,
    "initial_fraction": 0.8,
    "cpus": 16,
    "account": "",          # optional sbatch --account
}


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------

def load_state(pair_dir: str) -> dict:
    with open(os.path.join(pair_dir, STATE_FILE)) as fh:
        return json.load(fh)


def save_state(pair_dir: str, state: dict) -> None:
    path = os.path.join(pair_dir, STATE_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, path)


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log_decision(pair_dir: str, row: dict) -> None:
    path = os.path.join(pair_dir, LOG_FILE)
    fields = ["time", "iteration", "segments", "mu", "m0", "se", "J", "n_replicas",
              "slope", "slope_kind", "action", "mu_next", "note"]
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Measuring one iteration
# ---------------------------------------------------------------------------

def replica_rows(iter_dir: str) -> list[dict]:
    """Each replica's newest row in an iteration folder, ordered by replica slot."""
    rows = latest_row_per_replica(read_susceptibility_csv(os.path.join(iter_dir, SUSCEPTIBILITY_DATA_CSV)))
    return sorted(rows, key=lambda r: int(r.get("replica_id", 0) or 0))


def load_series(iter_dir: str, rows: list[dict]) -> list[dict]:
    """[{m, occ}] per replica: m(t) and rho_bonding + rho_inert (t) of its newest row."""
    import pandas as pd

    out = []
    for r in rows:
        path = os.path.join(iter_dir, f"m_timeseries_{r['id']}.csv")
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path, usecols=["m", "rho_bonding", "rho_inert"])
        if len(df):
            out.append({"m": df["m"].to_numpy(float),
                        "occ": (df["rho_bonding"] + df["rho_inert"]).to_numpy(float)})
    return out


def measure_series(series: list[dict], n_sites: int) -> dict:
    """m0, its SE over replicas, the fluctuation slope d<m>/d(beta mu), and mean jumps."""
    means = np.array([s["m"].mean() for s in series])
    n = len(means)
    m_all = np.concatenate([s["m"] for s in series])
    occ_all = np.concatenate([s["occ"] for s in series])
    # Pooled over replicas and time, around the pooled means (not per replica):
    # replicas sitting in opposite phases are exactly what makes <m> respond to mu.
    cov = float(np.mean((m_all - m_all.mean()) * (occ_all - occ_all.mean())))
    return {
        "m0": float(means.mean()),
        "se": float(means.std(ddof=1) / math.sqrt(n)) if n > 1 else float("inf"),
        "slope_var": n_sites * cov,
        "J": float(np.mean([count_jumps(s["m"]) for s in series])),
        "abs_m": float(np.mean(np.abs(m_all))),
        "n_replicas": n,
        "n_samples": int(min(len(s["m"]) for s in series)),
    }


# ---------------------------------------------------------------------------
# Decisions (pure functions; unit-tested)
# ---------------------------------------------------------------------------

def newton_step(iterations: list[dict], meas: dict, cfg: dict, beta: float = 1.0) -> dict:
    """Next mu from the current iteration's measurement.

    iterations[-1] is the current iteration (mu = its mu); earlier entries carry their
    final measurement in "measurements"[-1].
    """
    mu = float(iterations[-1]["mu"])
    m0 = meas["m0"]
    slope, kind = meas["slope_var"], "variance"
    if len(iterations) >= 2 and iterations[-2].get("measurements"):
        prev = iterations[-2]["measurements"][-1]
        dmu_b = beta * (mu - float(iterations[-2]["mu"]))
        dm = m0 - prev["m0"]
        if dmu_b != 0.0 and dm / dmu_b > 0.0:
            slope, kind = dm / dmu_b, "secant"
            ref = meas["slope_var"]
            if math.isfinite(ref) and ref > 0.0:
                lo, hi = cfg.get("secant_lo", 0.5) * ref, cfg.get("secant_hi", 5.0) * ref
                if not lo <= slope <= hi:
                    slope, kind = min(max(slope, lo), hi), "secant-clamped"
    if not (math.isfinite(slope) and slope > 0.0):
        step, kind = -math.copysign(cfg["dmu_max"], m0), "fallback"
    else:
        step = -cfg["tau"] * m0 / slope
    capped = max(-cfg["dmu_max"], min(cfg["dmu_max"], step))
    return {"mu_next": mu + capped / beta, "slope": slope, "slope_kind": kind,
            "step_raw": step, "step": capped}


def choose_action(state: dict, meas: dict) -> tuple[str, str]:
    """('continue' | 'new_mu' | 'converged' | 'failed', reason) for the current iteration."""
    cfg = state["config"]
    it = state["iterations"][-1]
    nseg = len(it["measurements"])
    m0, se, J = meas["m0"], meas["se"], meas["J"]

    # Judged after every segment: no minimum jump count and no SE test. A point that
    # passes by chance is checked again when it reaches J >= j_target.
    if abs(m0) > cfg["m_tol"]:
        if len(state["iterations"]) >= cfg["max_iter"]:
            return "failed", f"<m>={m0:+.3f} still off after {cfg['max_iter']} mu values"
        return "new_mu", f"<m>={m0:+.3f} > m_tol={cfg['m_tol']} (SE={se:.3f})"
    if abs(m0) <= cfg["m_tol"] and J >= cfg["j_target"]:
        return "converged", f"<m>={m0:+.3f}, J={J:.1f}"
    if nseg >= cfg["max_segments"]:
        return "failed", f"{nseg} segments at this mu without converging (J={J:.1f}, <m>={m0:+.3f})"
    return "continue", f"<m> ok, J={J:.1f} < {cfg['j_target']}"


# ---------------------------------------------------------------------------
# Slurm plumbing
# ---------------------------------------------------------------------------

def _sbatch(cmd: list[str], dry_run: bool) -> str:
    if dry_run:
        print("  [DRY-RUN] " + " ".join(cmd), flush=True)
        return "DRY"
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out.stdout.strip().split(";")[0]


def submit_pair_jobs(pair_dir: str, state: dict, dry_run: bool = False) -> tuple[str, str]:
    """Submit the next segment and its decide job. Returns (segment id, decide id)."""
    cfg = state["config"]
    name = f"nwt_L{state['L']}_{state['epsilon']:+.4f}"
    acct = [f"--account={cfg['account']}"] if cfg.get("account") else []
    seg = _sbatch(["sbatch", "--parsable", f"--job-name={name}", f"--cpus-per-task={cfg['cpus']}",
                   *acct, NEWTON_SCRIPT, "segment", pair_dir], dry_run)
    dep = [] if dry_run else [f"--dependency=afterany:{seg}"]
    dec = _sbatch(["sbatch", "--parsable", f"--job-name={name}_d", *dep,
                   "--cpus-per-task=1", "--mem-per-cpu=4G", "--time=00:30:00",
                   *acct, NEWTON_SCRIPT, "decide", pair_dir], dry_run)
    if not dry_run:
        state.setdefault("jobs", []).append({"time": _now(), "segment": seg, "decide": dec,
                                             "iteration": len(state["iterations"]) - 1,
                                             "action": state["next"]["kind"]})
    return seg, dec


def runner_args(pair_dir: str, state: dict, cpus: int) -> list[str]:
    """Runner command line for the action in state['next']."""
    cfg, nxt = state["config"], state["next"]
    k = len(state["iterations"]) - 1
    it = state["iterations"][k]
    outdir = os.path.join(pair_dir, it["dir"])
    ph = state["physics"]
    args = [
        sys.executable, "-u", "-m", "flexsim.susc.runner",
        "--epsilon", repr(state["epsilon"]), "--L", str(state["L"]),
        "--mu", repr(float(it["mu"])), "--beta", repr(state["beta"]),
        "--delta-f", repr(ph["delta_f"]), "--delta-mu", repr(ph["delta_mu"]),
        "--k", repr(ph["k"]), "--scheme", ph["scheme"],
        "--cpus", str(cpus), "--outdir", outdir, "--results-base", os.path.dirname(os.path.dirname(pair_dir)),
        "--seed-base", str(int(cfg["seed_base"]) + SEED_STRIDE * k),
    ]
    kind = nxt["kind"]
    if kind == "start":
        prod = cfg["prod0"]
        args += ["--eq-time", repr(cfg["eq0"]), "--initial-fraction", repr(cfg["initial_fraction"])]
    elif kind == "continue":
        prod = cfg["seg_prod"]
        args += ["--eq-time", "0", "--resume-dir", outdir,
                 "--resume-ids", *[str(i) for i in nxt["resume_ids"]]]
    elif kind == "new_mu":
        prod = cfg["seg_prod"]
        prev = os.path.join(pair_dir, state["iterations"][k - 1]["dir"])
        args += ["--eq-time", repr(cfg["eq_iter"]), "--resume-dir", prev, "--fresh-history",
                 "--resume-ids", *[str(i) for i in nxt["resume_ids"]]]
    else:
        raise SystemExit(f"nothing to run: next action is {kind!r}")
    args += ["--prod-time", repr(float(prod)), "--prod-chunks", str(max(1, round(prod / CHUNK_TIME)))]
    return args


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _pair_dirs(base: str) -> list[str]:
    import glob
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(base, "*", "_*", STATE_FILE)))


def do_start(args) -> None:
    from flexsim.susc.sweep import load_mu_map

    base = os.path.abspath(args.base)
    eps_values = frange(args.eps_min, args.eps_max, args.eps_step)
    mu_map = load_mu_map(args.mu_source, delta_f=args.delta_f, delta_mu=args.delta_mu,
                         k=args.k, scheme=args.scheme)
    missing = [e for e in eps_values if round(e, 6) not in mu_map]
    if missing:
        raise SystemExit(f"{args.mu_source} has no mu for {len(missing)} eps (first: {missing[:5]})")

    cfg = dict(DEFAULT_CONFIG)
    for key in DEFAULT_CONFIG:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = type(DEFAULT_CONFIG[key])(val)

    physics = {"delta_f": args.delta_f, "delta_mu": args.delta_mu, "k": args.k, "scheme": args.scheme}
    os.makedirs(base, exist_ok=True)
    n_new = n_skip = 0
    print(f"[newton] {len(eps_values)} eps x {len(args.sizes)} L -> {base}", flush=True)
    for eps in eps_values:
        mu0 = mu_map[round(eps, 6)]
        for L in args.sizes:
            params = {**physics, "epsilon": eps, "Lx": L, "Ly": L}
            pair = os.path.abspath(susc_run_dir(params, base=base))
            if os.path.isfile(os.path.join(pair, STATE_FILE)):
                n_skip += 1
                continue
            state = {
                "version": 1, "epsilon": eps, "L": L, "beta": args.beta, "physics": physics,
                "mu_start": mu0, "mu_source": os.path.abspath(args.mu_source),
                "config": cfg, "status": "running", "created": _now(),
                "iterations": [{"dir": "iter_0", "mu": mu0, "measurements": [], "retries": 0}],
                "next": {"kind": "start"}, "jobs": [],
            }
            print(f"  L={L:4d} eps={eps:+.6f} mu0={mu0:+.8f}", flush=True)
            if args.dry_run:
                submit_pair_jobs(pair, state, dry_run=True)
                continue
            os.makedirs(pair, exist_ok=True)
            save_state(pair, state)
            submit_pair_jobs(pair, state)
            save_state(pair, state)
            n_new += 1
    print(f"[newton] started {n_new} chains, skipped {n_skip} that already exist"
          + (" (dry run, nothing written)" if args.dry_run else ""), flush=True)


def do_segment(args) -> None:
    pair = os.path.abspath(args.pair_dir)
    state = load_state(pair)
    if state["status"] != "running":
        print(f"[newton] {pair}: status {state['status']}, nothing to run", flush=True)
        return
    cmd = runner_args(pair, state, args.cpus or state["config"]["cpus"])
    print("[newton] " + " ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd).returncode)


def decide(pair: str, submit=submit_pair_jobs) -> str:
    """Measure the current iteration, record the decision, and submit what comes next."""
    state = load_state(pair)
    if state["status"] != "running":
        return state["status"]
    cfg = state["config"]
    k = len(state["iterations"]) - 1
    it = state["iterations"][k]
    iter_dir = os.path.join(pair, it["dir"])
    base = os.path.dirname(os.path.dirname(pair))
    row = {"time": _now(), "iteration": k, "mu": f"{it['mu']:.10f}"}

    if os.path.exists(os.path.join(base, STOP_FILE)):
        state["status"] = "stopped"
        save_state(pair, state)
        log_decision(pair, {**row, "action": "stopped", "note": f"{STOP_FILE} file in base"})
        return "stopped"

    rows = replica_rows(iter_dir)
    series = load_series(iter_dir, rows) if rows else []
    expected = len(it.get("resume_ids") or []) or cfg["cpus"]
    if not series:
        # The segment produced nothing (killed, node failure). Rerun it; a first
        # segment that left partial output is moved aside so replica slots don't clash.
        it["retries"] = it.get("retries", 0) + 1
        if it["retries"] > cfg["max_retries"]:
            state["status"] = "failed"
            note = f"no data after {it['retries']} tries"
        else:
            if os.path.isdir(iter_dir) and not it["measurements"]:
                os.rename(iter_dir, f"{iter_dir}_failed{it['retries']}")
            note = f"no data; retry {it['retries']}"
            submit(pair, state)
        save_state(pair, state)
        log_decision(pair, {**row, "action": "retry" if state["status"] == "running" else "failed",
                            "note": note})
        return state["status"]

    meas = measure_series(series, state["L"] ** 2)
    meas["prod_time"] = max(float(r.get("prod_time", 0) or 0) for r in rows)
    meas["time"] = row["time"]
    it["measurements"].append(meas)
    row.update(segments=len(it["measurements"]), m0=f"{meas['m0']:+.5f}", se=f"{meas['se']:.5f}",
               J=f"{meas['J']:.2f}", n_replicas=meas["n_replicas"])
    note = ""
    if meas["n_replicas"] < expected:
        note = f"only {meas['n_replicas']}/{expected} replicas have data; "

    action, why = choose_action(state, meas)
    note += why
    resume_ids = [int(r["id"]) for r in rows
                  if os.path.isfile(os.path.join(iter_dir, f"final_lattice_{r['id']}.npy"))]
    if action == "continue":
        state["next"] = {"kind": "continue", "resume_ids": resume_ids}
        it["resume_ids"] = resume_ids
        submit(pair, state)
    elif action == "new_mu":
        step = newton_step(state["iterations"], meas, cfg, state["beta"])
        row.update(slope=f"{step['slope']:.4g}", slope_kind=step["slope_kind"],
                   mu_next=f"{step['mu_next']:.10f}")
        if step["step"] != step["step_raw"]:
            note += f"; step capped ({step['step_raw']:+.2e} -> {step['step']:+.2e})"
        it["step"] = step
        state["iterations"].append({"dir": f"iter_{k + 1}", "mu": step["mu_next"],
                                    "measurements": [], "retries": 0, "resume_ids": resume_ids})
        state["next"] = {"kind": "new_mu", "resume_ids": resume_ids}
        submit(pair, state)
    else:
        state["status"] = action
        state["finished"] = row["time"]
    save_state(pair, state)
    log_decision(pair, {**row, "action": action, "note": note})
    print(f"[newton] L={state['L']} eps={state['epsilon']:+.6f} iter {k}: {action} ({note})", flush=True)
    return state["status"]


def do_decide(args) -> None:
    decide(os.path.abspath(args.pair_dir))


def _summary_row(pair: str, state: dict) -> dict:
    it = state["iterations"][-1]
    meas = it["measurements"][-1] if it["measurements"] else (
        state["iterations"][-2]["measurements"][-1] if len(state["iterations"]) > 1 else {})
    return {
        "L": state["L"], "epsilon": state["epsilon"], "status": state["status"],
        "iteration": len(state["iterations"]) - 1, "segments": len(it["measurements"]),
        "mu_start": state["mu_start"], "mu": it["mu"], "dmu": it["mu"] - state["mu_start"],
        "m0": meas.get("m0", float("nan")), "se": meas.get("se", float("nan")),
        "J": meas.get("J", float("nan")), "pair_dir": pair,
    }


def do_status(args) -> None:
    pairs = _pair_dirs(args.base)
    if not pairs:
        raise SystemExit(f"no Newton chains under {os.path.abspath(args.base)}")
    rows = [_summary_row(p, load_state(p)) for p in pairs]
    rows.sort(key=lambda r: (r["L"], r["epsilon"]))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + f"   (total {len(rows)})")
    print(f"{'L':>4} {'eps':>10} {'status':>10} {'it':>3} {'seg':>4} {'mu - mu0':>11} "
          f"{'<m>':>8} {'SE':>6} {'J':>6}")
    for r in rows:
        if args.only and r["status"] != args.only:
            continue
        print(f"{r['L']:4d} {r['epsilon']:+10.6f} {r['status']:>10} {r['iteration']:3d} {r['segments']:4d} "
              f"{r['dmu']:+11.2e} {r['m0']:+8.4f} {r['se']:6.4f} {r['J']:6.1f}")
    out = os.path.join(args.base, "newton_status.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}")


def _live_jobs(job_ids: list[str]) -> set[str]:
    ids = [j for j in job_ids if j and j != "DRY"]
    if not ids:
        return set()
    out = subprocess.run(["squeue", "-h", "-o", "%i", "-j", ",".join(ids)],
                         capture_output=True, text=True)
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def do_kick(args) -> None:
    """Restart chains whose jobs are all gone while their status is still 'running'.

    With --resume-stopped, chains stopped by a STOP file are set running again first
    (remove the STOP file before, or they stop again at once).
    """
    if args.resume_stopped and os.path.exists(os.path.join(args.base, STOP_FILE)):
        raise SystemExit(f"remove {os.path.join(args.base, STOP_FILE)} first")
    for pair in _pair_dirs(args.base):
        state = load_state(pair)
        if state["status"] == "stopped" and args.resume_stopped and not args.dry_run:
            state["status"] = "running"
            save_state(pair, state)
        if state["status"] != "running":
            continue
        last = state.get("jobs", [])[-1:] or [{}]
        if _live_jobs([last[0].get("segment", ""), last[0].get("decide", "")]):
            continue
        print(f"[kick] L={state['L']} eps={state['epsilon']:+.6f}: no live job, running decide now",
              flush=True)
        if not args.dry_run:
            decide(pair)


def do_combine(args) -> None:
    """Link each converged pair's final iteration into a SUSC_RUNS-shaped folder."""
    out = os.path.abspath(args.out)
    if os.path.isdir(out):
        for name in os.listdir(out):
            path = os.path.join(out, name)
            if name.endswith(".csv") or name.endswith(".png"):
                continue
            if not os.path.isdir(path) or os.path.islink(path) or not all(
                    os.path.islink(os.path.join(path, x)) for x in os.listdir(path)):
                raise SystemExit(f"{out} holds more than links ({path}); refusing to rebuild it")
        shutil.rmtree(out)
    os.makedirs(out)
    statuses = set(args.status)
    summary = []
    for base in args.base:
        for pair in _pair_dirs(base):
            state = load_state(pair)
            row = _summary_row(pair, state)
            summary.append(row)
            if state["status"] not in statuses:
                continue
            it_dir = os.path.join(pair, state["iterations"][-1]["dir"])
            size_dir = os.path.join(out, os.path.basename(os.path.dirname(pair)))
            os.makedirs(size_dir, exist_ok=True)
            os.symlink(it_dir, os.path.join(size_dir, os.path.basename(pair)))
    summary.sort(key=lambda r: (r["L"], r["epsilon"]))
    with open(os.path.join(out, "newton_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    linked = sum(r["status"] in statuses for r in summary)
    print(f"[combine] linked {linked}/{len(summary)} pairs with status in {sorted(statuses)} -> {out}")
    _plot_dmu(summary, os.path.join(out, "mu_shift_vs_eps.png"))


def _plot_dmu(summary: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for L in sorted({r["L"] for r in summary}):
        rs = sorted((r for r in summary if r["L"] == L), key=lambda r: r["epsilon"])
        ax.plot([r["epsilon"] for r in rs], [r["dmu"] for r in rs], "o-", ms=3, label=f"L={L}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$\beta\epsilon$")
    ax.set_ylabel(r"$\beta\mu_{\rm Newton} - \beta\mu_{\rm fit}$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[combine] {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("start", help="create a Newton chain per (eps, L) and submit its first segment")
    p.add_argument("--base", required=True, help="output folder for the chains")
    p.add_argument("--mu-source", required=True, help="CSV with epsilon, mu_coex_FITTED (e.g. refit fit output)")
    p.add_argument("--eps-min", type=float, required=True)
    p.add_argument("--eps-max", type=float, required=True)
    p.add_argument("--eps-step", type=float, required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128])
    p.add_argument("--delta-f", type=float, required=True)
    p.add_argument("--delta-mu", type=float, required=True)
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--scheme", required=True)
    p.add_argument("--beta", type=float, default=1.0)
    for key, val in DEFAULT_CONFIG.items():
        p.add_argument("--" + key.replace("_", "-"), dest=key, type=type(val), default=None,
                       help=f"default {val}")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=do_start)

    p = sub.add_parser("segment", help="(inside a job) run the next segment of one chain")
    p.add_argument("--pair-dir", required=True)
    p.add_argument("--cpus", type=int, default=None)
    p.set_defaults(func=do_segment)

    p = sub.add_parser("decide", help="(inside a job) measure, decide, submit the next segment")
    p.add_argument("--pair-dir", required=True)
    p.set_defaults(func=do_decide)

    p = sub.add_parser("status", help="table of every chain under a base")
    p.add_argument("--base", required=True)
    p.add_argument("--only", default=None, help="show only this status (running, converged, failed, ...)")
    p.set_defaults(func=do_status)

    p = sub.add_parser("kick", help="restart running chains that have no live Slurm job")
    p.add_argument("--base", required=True)
    p.add_argument("--resume-stopped", action="store_true",
                   help="also restart chains stopped by a STOP file")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=do_kick)

    p = sub.add_parser("combine", help="link converged final iterations into one SUSC_RUNS-shaped folder")
    p.add_argument("--base", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--status", nargs="+", default=["converged"])
    p.set_defaults(func=do_combine)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
