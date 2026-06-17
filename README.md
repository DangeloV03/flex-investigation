# flex-investigation

Pipeline for studying coexistence chemical potential in driven heterogeneous lattice-gas simulations. Compares FLEX theory predictions against Monte Carlo simulations across schemes, system sizes, and chemical-potential sweeps.

## Overview

| Script | Purpose |
|--------|---------|
| `generate_samples.py` | Compute `mu_coex_FLEX`, sweep μ, write job JSON files, `manage.csv`, and seed `run_all_queue.json` |
| `json_runner.py` | Run one job: parallel replicas, time-averaged densities, lattice snapshots |
| `run_all.py` | Long-running Slurm dispatcher (via `simple_slurm`); sole submission point for compute jobs |
| `queue_manifest.py` | Shared queue manifest helpers for `run_all.py` and `analyzer.py` |
| `flex_coex_chemical_potential_prediction.py` | FLEX coexistence chemical potential solver |
| `analyzer.py` | Watch `results/`, compute `mu_coex_SIM`, enqueue refinement jobs, save plots |

## Requirements

- Python 3.11+ (3.13 tested via Conda)
- See [`requirements.txt`](requirements.txt): numpy, scipy, matplotlib, pandas, pyyaml, simple-slurm
- `lattice_gas` — external Monte Carlo package providing `HeteroChain`, `simulate`, etc. (install from its source repo)

Example Conda setup:

```bash
conda create -n lattice python=3.13 -y
conda activate lattice
pip install -r requirements.txt
# install lattice_gas from its repo
```

## Workflow

### 1. Generate sample jobs

```bash
python generate_samples.py
```

This writes:

- `samples/<scheme>_Ly<L>_<mu_tag>.json` — one self-contained job per (scheme, Ly, μ)
- `run_all_queue.json` — pending job queue consumed by `run_all.py`
- `manage.csv` — tracks outer combos and run status:
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
python json_runner.py samples/homo_Ly16_mu3948181.json
```

Outputs go to `results/<scheme>_eps<eps>_Ly<Ly>/<mu_tag>/`:

- `output.csv` — columns: `id`, `rho_active`, `rho_inert`, `rho_empty`, `time`, `mu`, …
- `final_lattice_<id>.npy` — final lattice for each replica

Re-runs append new rows to `output.csv` with sequential IDs.

### 3. Dispatch jobs (local or Della)

**Local testing** (no Slurm — runs jobs via subprocess):

```bash
python run_all.py --local
```

**Princeton Della** — edit [`slurm_config.yml`](slurm_config.yml) for your partition, account, and module/conda setup, then start the dispatcher on a login node:

```bash
ssh <NetID>@della9.princeton.edu
cd /scratch/gpfs/WJACOBS/vd7294/flex-investigation   # your project path
git pull origin dangelo/run-on-della

# Start run_all + analyzer in tmux (detached; survives SSH logout)
chmod +x scripts/start_daemons.sh scripts/stop_daemons.sh
./scripts/start_daemons.sh
tmux attach -t flex-investigation    # watch output; Ctrl-b d to detach

# Stop later
./scripts/stop_daemons.sh
```

The tmux session has two windows: `run_all` (Slurm dispatcher) and `analyzer` (results watcher).

`run_all.py`:

- Reads `run_all_queue.json` (`pending` + `in_flight`)
- Submits up to **100 concurrent** Slurm jobs via [simple_slurm](https://pypi.org/project/simple-slurm/)
- Polls `squeue` every 30s and submits more as slots free up
- Archives completed JSON files to `samples/done/`
- Re-queues failed jobs at the front of `pending`

`analyzer.py` never calls Slurm directly. When it needs more μ points, it writes JSON to `samples/` and **prepends** those paths to `run_all_queue.json` so refinement jobs run first.

### 4. Analyze results

```bash
python analyzer.py
```

Polls `results/`, builds φ(μ) and ψ(μ) curves per combo, refines/extends μ sweeps as needed, enqueues new jobs via the manifest, and writes `mu_coex_SIM` to `manage.csv`.

## Queue manifest

[`run_all_queue.json`](run_all_queue.json) is shared between `run_all.py` and `analyzer.py`:

```json
{
  "pending": ["samples/homo_Ly16_mu00.json"],
  "in_flight": {"12345678": "samples/homo_Ly8_mu01.json"}
}
```

- `pending` — submission order; front = next job; analyzer prepends here (stack priority)
- `in_flight` — Slurm job ID → JSON path while running

## Initial condition

Slab geometry: x < Lx/2 is active (bonding), x ≥ Lx/2 is empty. Periodic boundary conditions (via `lattice_gas`).

## Project layout

```
flex-investigation/
├── generate_samples.py
├── json_runner.py
├── run_all.py
├── queue_manifest.py
├── slurm_config.yml
├── requirements.txt
├── flex_coex_chemical_potential_prediction.py
├── analyzer.py
├── notes.md
├── run_all_queue.json        # job queue (gitignored)
├── samples/                  # pending job JSON (gitignored)
├── samples/done/             # archived JSON after successful runs
├── slurm_reports/            # on Della: /home/vd7294/slurm_reports (Slurm stdout/stderr)
├── results/                  # simulation outputs (gitignored)
└── manage.csv                # run tracking (gitignored)
```

## Notes

See `notes.md` for the full functional decomposition, analyzer design, and open questions (e.g. mixed periodic boundaries).
