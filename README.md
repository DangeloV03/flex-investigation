# flex-investigation

Compare **FLEX theory** predictions for the coexistence chemical potential (μ_coex) against **Monte Carlo lattice-gas** simulations on driven heterogeneous chains.

**New here? → [QUICKSTART.md](QUICKSTART.md)**

---

## What this repo does

1. **Coexistence campaign** — generates (ε, Δμ) parameter sweeps, submits Slurm jobs on Princeton Della, and compares μ_coex from FLEX theory vs. simulation.
2. **Susceptibility campaign** — runs finite-size scaling of χ, Binder cumulant U₄, and ⟨|m|⟩ across a grid of (ε, L) values at exact coexistence (μ = 2ε), used to locate the critical point.

---

## Prerequisites

- Access to [Princeton Della](https://researchcomputing.princeton.edu/systems/della)
- Access to the private [`lattice-gas`](https://github.com/moleary253/lattice-gas) GitHub repo (ask your PI)
- A GitHub account added as a collaborator to this repo

See [QUICKSTART.md](QUICKSTART.md) for full setup from scratch.

---

## Scripts reference

### Core pipeline (coexistence campaign)

| Script | Purpose |
|--------|---------|
| `generate_samples.py` | LHS parameter sweep → job JSONs, `manage.csv`, queue seed |
| `json_runner.py` | Run one job: parallel replicas, write densities + lattice snapshots |
| `run_all.py` | Slurm dispatcher (or `--local` for laptop) |
| `analyzer.py` | Results watcher: plots φ(μ)/ψ(μ), adaptive μ refinement |
| `queue_manifest.py` | Locked read/write helpers for `run_all_queue.json` |
| `flex_coex_chemical_potential_prediction.py` | FLEX μ_coex solver |

### Susceptibility campaign (Ising limit, exact μ)

| Script | Purpose |
|--------|---------|
| `generate_susceptibility_exact.py` | Grid of (ε, L) → job JSONs with μ = 2ε |
| `susceptibility_runner.py` | L×L square lattice, measures χ, U₄, ⟨|m|⟩ per chunk |
| `run_susceptibility_all.py` | Slurm dispatcher for susceptibility jobs |
| `plot_susceptibility.py` | Plots χ(ε), ⟨|m|⟩(ε), U₄(ε), peak χ vs L |
| `plot_fss.py` | Finite-size scaling collapse (χ and ⟨|m|⟩ vs rescaled ε) |

### Helper scripts (`scripts/`)

| Script | When to use |
|--------|-------------|
| `env.sh` | Source on login: exports, conda activate, import check |
| `start_sus_exact_daemons.sh` | Start exact susceptibility dispatcher in tmux (`sus-exact`) |
| `start_sus_coex_daemons.sh` | Start coex susceptibility dispatcher + analyzer in tmux |
| `start_sus_prod_daemons.sh` | Start prod susceptibility dispatcher in tmux |
| `start_daemons.sh` / `stop_daemons.sh` | Start/stop coexistence campaign tmux session |
| `repair_queue.py` | Restore missing JSONs, clear stale `in_flight` entries |
| `requeue_incomplete.py` | Re-enqueue jobs that never finished |
| `retry_nan_combos.py` | Re-run analyzer on rows marked NaN |
| `estimate_runtime.py` | Estimate remaining campaign wall time |

---

## Configuration

**`slurm_config.yml`** — edit before submitting jobs:
- `partition` — usually `cpu` on Della
- `account` — uncomment and set if required
- `report_dir` / `output` / `error` — Slurm log paths (must exist before jobs run)
- `setup_cmds` — match your Conda module name and env name

---

## Requirements

- Python 3.11+, Rust/Cargo, maturin
- Python packages: `numpy scipy matplotlib pandas pyyaml simple-slurm` (see `requirements.txt`)
- `lattice_gas` — built from ZIP in `~/software/lattice-gas` via `./build-rust-lib.sh`

---

## Project layout

```
flex-investigation/
├── README.md / QUICKSTART.md
├── slurm_config.yml
├── requirements.txt
│
├── # Coexistence campaign
├── generate_samples.py
├── json_runner.py
├── run_all.py
├── analyzer.py
├── queue_manifest.py
├── combo_paths.py
├── flex_coex_chemical_potential_prediction.py
│
├── # Susceptibility campaign
├── generate_susceptibility_exact.py   # exact μ = 2ε (current active)
├── generate_susceptibility_coex.py    # coex-phase μ sweep
├── generate_susceptibility_jobs.py    # prod-phase job gen
├── susceptibility_runner.py
├── susceptibility_paths.py
├── run_susceptibility_all.py
├── plot_susceptibility.py
├── plot_fss.py
│
├── scripts/
│   ├── env.sh                         # login setup (source from ~/.bashrc)
│   ├── start_sus_exact_daemons.sh     # start exact susceptibility campaign
│   ├── start_sus_coex_daemons.sh      # start coex susceptibility phase
│   ├── start_sus_prod_daemons.sh      # start prod susceptibility phase
│   ├── start_daemons.sh               # start coexistence campaign
│   ├── stop_daemons.sh
│   ├── repair_queue.py
│   ├── requeue_incomplete.py
│   ├── retry_nan_combos.py
│   └── estimate_runtime.py
│
├── tests/
│   └── test_pipeline.py
│
├── susceptibility_results/      # gitignored — simulation output
├── susceptibility_samples/      # gitignored — generated job JSONs
├── results/                     # gitignored — coexistence output
├── samples/                     # gitignored — coexistence job JSONs
└── plots/                       # gitignored — generated figures
```
