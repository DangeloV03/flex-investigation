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
| `start_daemons.sh` / `stop_daemons.sh` | Start/stop campaign tmux session on Della |
| `repair_queue.py` | Restore missing JSONs, clear stale `in_flight` entries |

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
├── generate_samples.py          # coexistence campaign setup
├── json_runner.py               # single-job worker
├── run_all.py                   # Slurm dispatcher
├── analyzer.py                  # results watcher + refinement
├── generate_susceptibility_exact.py
├── susceptibility_runner.py
├── run_susceptibility_all.py
├── plot_susceptibility.py
├── plot_fss.py                  # FSS collapse plots
├── slurm_config.yml
├── requirements.txt
├── scripts/
│   ├── env.sh
│   ├── start_daemons.sh
│   └── stop_daemons.sh
├── susceptibility_results/      # gitignored — simulation output
└── plots/                       # gitignored — generated figures
```
