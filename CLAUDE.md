# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Susceptibility measurements of a driven three-state lattice gas (EMPTY=0, INERT=1, BONDING=2) on an L × L periodic lattice. The simulation itself is done by the external Rust engine [`lattice-gas`](https://github.com/moleary253/lattice-gas) (Python module `lattice_gas`); this repo only drives it and post-processes. Keep it minimal: a JSON runner, a CSV sweep, an averager, and their Slurm scripts — don't add features, config layers, or abstractions beyond what's asked.

## Environment and commands

Needs conda env `lattice` with `numpy` and a built `lattice_gas`. Jobs run on Princeton's Della cluster (Slurm; `module load anaconda3/2024.10`). There is no build step, test suite, or linter.

```bash
python src/flexsim/susc/replica.py params.json                 # single point, n_replicas in parallel
python src/flexsim/susc/sweep.py sweep.json                    # all rows of json["mu_database"]
python src/flexsim/susc/average.py mu_database.csv [outroot]   # -> mu_database_iteration1.csv
sbatch submit.sh params.json      # cluster: replica.py
sbatch sweep.sh                   # cluster: writes JSON from its bash vars, sweep.py then average.py
```

Scripts must be run as files (not `python -m`): `sweep.py` does `from replica import run_replica`, relying on the script's directory being on `sys.path`.

## Architecture

- `replica.py` holds all physics. `run_replica((i, p))` builds a `HeteroChain` (inert fugacity `exp(β(μ+Δf))`, bonding fugacity `exp(βμ)`), equilibrates for `eq_time`, then runs `prod_chunks` consecutive `simulate` calls of `prod_time/prod_chunks` each, reading the state back via `load.final_state(scratch)` after every chunk. Engine scratch dirs go in `/dev/shm` when present. Replicas start from random BONDING/EMPTY lattices, even ones at m = +`initial_m` and odd ones at m = −`initial_m` (`n_replicas` must be even), so both phases get sampled; replica `i` draws its starting geometry and every engine seed (equilibration and each chunk) from `default_rng([seed, i])`, so replicas have independent streams.
- With `"continuation": true` (`CONTINUATION=true` in `sweep.sh`) every replica restarts from its `final_lattice_<i>.npy` with no equilibration, appends to `m_timeseries_<i>.csv` (chunk numbers continue), and `susceptibility_data.csv` is recomputed over the full series. Missing files for any replica abort the sweep before anything runs; the run's params go to `params_continuation.json`, leaving `params.json` untouched.
- Order parameter is `m = ρ_bonding − ρ_inert − ρ_empty`; `chi = β·L²·(<m²> − <|m|>²)`. The energy in `energy()` is computed in Python from the final state of each chunk, not taken from the engine.
- `sweep.py` reads a JSON with `mu_database`, `outroot`, `n_replicas` and the fixed physics keys, turns each CSV row (`lattice_size,epsilon,mu`) into a params dict (the same keys as `params.json`), and calls `run_replica` directly. All (point, replica) tasks share one `mp.Pool` sized by `SLURM_CPUS_PER_TASK`. A point's `susceptibility_data.csv` is written once all of its replicas finish.
- `average.py` locates each point's output by rebuilding the directory name `L{L}_eps{float(eps)}_mu{float(mu)}`. That format must stay identical to the one in `sweep.py`, because the name depends on Python's float repr (e.g. `-2.0`).
- Changing sweep physics means editing the parameter block in `sweep.sh` (it writes `$OUTROOT/sweep_params.json`). Single runs take everything from the JSON, with defaults coming from `DEFAULTS`/`REQUIRED` in `replica.py`.

## Outputs (per outdir)

`params.json`, `m_timeseries_<i>.csv` (one row per chunk), `final_lattice_<i>.npy`, and `susceptibility_data.csv` (one row per replica: moments of m and E with SEMs, plus chi). Sweep outputs go under `output/` by default (`OUTROOT` in `sweep.sh`).
