# flex-investigation

Pipeline for studying coexistence chemical potential in driven heterogeneous lattice-gas simulations. Compares FLEX theory predictions against Monte Carlo simulations across schemes, system sizes, and chemical-potential sweeps.

## Overview

| Script | Purpose |
|--------|---------|
| `generate_samples.py` | Compute `mu_coex_FLEX`, sweep μ, write job JSON files and `manage.csv` |
| `Json_runner.py` | Run one job: parallel replicas, time-averaged densities, lattice snapshots |
| `run_all.py` | Sequentially dispatch every job in `samples/` (local stand-in for Slurm) |
| `flex_coex_chemical_potential_prediction.py` | FLEX coexistence chemical potential solver |
| `analyzer.py` | Watch `results/`, compute `mu_coex_SIM`, update `manage.csv`, save plots |

## Requirements

- Python 3.11+ (3.13 tested via Conda)
- [numpy](https://numpy.org/), [scipy](https://scipy.org/)
- `lattice_gas` — external Monte Carlo package providing `HeteroChain`, `simulate`, etc. (install from its source repo)

Example Conda setup:

```bash
conda create -n lattice python=3.13 -y
conda activate lattice
pip install numpy scipy
# install lattice_gas from its repo
```

## Workflow

### 1. Generate sample jobs

```bash
python generate_samples.py
```

This writes:

- `samples/<scheme>_Ly<L>_<mu_tag>.json` — one self-contained job per (scheme, Ly, μ)
- `manage.csv` — tracks outer combos and run status. Two coexistence columns:
  - `mu_coex_FLEX` — theory prediction from FLEX (written by `generate_samples.py`)
  - `mu_coex_SIM` — simulation estimate from min ψ (written by `analyzer.py`)

Sample filenames encode **|μ|** at 6 decimal places, e.g. `mu3948181` → |μ| = 3.948181.

Fixed parameters (edit in `generate_samples.py`):

- ε = −2.0, Δf = 0, Δμ = 0, k = 1
- Schemes: `homo`, `positive_drive`, `negative_drive`
- Ly ∈ {8, 16, 32}, Lx = 10 × Ly
- μ sweep: `mu_coex_FLEX` ± 0.1, 10 points

### 2. Run a single job

```bash
python Json_runner.py samples/homo_Ly16_mu3948181.json
```

Outputs go to `results/<job_basename>/`:

- `output.csv` — columns: `id`, `rho_active`, `rho_inert`, `rho_empty`, `time`, `mu`
- `final_lattice_<id>.npy` — final lattice for each replica
- `<job_basename>.json` — job file moved here after a successful run

Re-runs append new rows to `output.csv` with sequential IDs.

Optional custom output directory:

```bash
python Json_runner.py samples/homo_Ly16_mu3948181.json --outdir results/custom_run
```

### 3. Run all pending jobs

```bash
python run_all.py
```

Runs every `samples/*.json` file in order. Intended as a laptop-friendly substitute for a future Slurm array job.

### 4. Analyze results

```bash
python analyzer.py
```

Polls `results/`, builds φ(μ) and ψ(μ) curves per combo, refines/extends μ sweeps as needed, and writes `mu_coex_SIM` to `manage.csv`. Plots mark both `mu_coex_FLEX` (blue) and `mu_coex_SIM` (red).

## Initial condition

Slab geometry: x < Lx/2 is active (bonding), x ≥ Lx/2 is empty. Periodic boundary conditions (via `lattice_gas`).

## Project layout

```
flex-investigation/
├── generate_samples.py
├── Json_runner.py
├── run_all.py
├── flex_coex_chemical_potential_prediction.py
├── notes.md                  # design notes and TODOs
├── samples/                  # generated job JSON (gitignored)
├── results/                  # simulation outputs (gitignored)
└── manage.csv                # run tracking (gitignored, regenerated)
```

## Notes

See `notes.md` for the full functional decomposition, analyzer design, and open questions (e.g. mixed periodic boundaries).
