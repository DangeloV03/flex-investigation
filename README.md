# flex-investigation

Monte Carlo lattice-gas simulations of driven chains, compared against FLEX theory.

- **Coexistence (coex):** on an Lx × Ly slab, sweep μ at each (ε, Δμ) and fit the
  coexistence chemical potential μ_coex. Compare it to the FLEX prediction.
- **Susceptibility (susc):** on L × L lattices at coexistence, measure χ, ⟨|m|⟩, the
  heat capacity c and the Binder cumulant U₄ against ε, for finite-size scaling.
- **Criticality:** locate ε_c from the coex snapshots, using the bimodality of the
  column order parameter, and from the susc magnetization distributions.

Simulations use the [`lattice-gas`](https://github.com/moleary253/lattice-gas) Rust
engine and run on Princeton Della through Slurm.

## Layout

```
run/            what you run: four entry points, plus examples/
  coex.py         coexistence campaigns
  susc.py         susceptibility campaigns
  plots.py        figures from finished runs
  criticality.py  epsilon_c analysis
  examples/       small runs to try on Della (start here)
src/flexsim/    implementation (installed as the `flexsim` package)
  coex/  susc/  criticality/
tests/          pytest suite
docs/           method notes (bimodality analysis, error bars)
tools/          Della → workstation data sync
output/         all data, plots, queues and Slurm logs (gitignored)
```

Only `run/` is meant to be used directly. Each `run/*.py` lists its commands:

```bash
python run/susc.py                 # list commands
python run/susc.py sweep --help    # options for one command
```

## Setup

You need Python ≥ 3.10 and a built `lattice-gas`. For first-time setup on Della
(clone, conda env, building `lattice-gas`, Slurm account), see
[docs/della_setup.md](docs/della_setup.md). Short version:

```bash
conda activate lattice          # env with lattice_gas built into it
pip install -e .                # installs flexsim (once per checkout)
source env.sh                   # each login: conda env, library path, import check
```

## Commands

| Entry point | Command | What it does |
|---|---|---|
| `run/coex.py` | `generate` | create the (ε, Δμ) job grid and queue it |
| | `generate-susc` | create slab μ-sweep jobs for the susceptibility study |
| | `start` / `stop` | start or stop the dispatcher and analyzer in tmux |
| | `dispatch`, `analyze` | run either one in the foreground |
| | `run-job` | run one job JSON (debugging) |
| `run/susc.py` | `sweep` | submit an ε × L sweep. A check job runs top-ups until every point has enough magnetization jumps |
| | `check` | run that check by hand |
| | `fixed-sweep` | older fixed-length sweep (no top-ups) |
| | `run-one` | run one (ε, L) point (debugging) |
| `run/plots.py` | `susc` | χ, ⟨\|m\|⟩, c, U₄ vs ε; peak χ vs L |
| | `fss` | finite-size scaling collapse of χ and ⟨\|m\|⟩ |
| | `chi-scaling` | fit χ_max ∝ L^(γ/ν), with leave-one-out and bootstrap errors |
| | `jumps` | magnetization jump counts per (ε, L) |
| | `flex-vs-sim` | FLEX vs simulated μ_coex |
| `run/criticality.py` | `phase-diagram` | BC_max vs βε for one slab size, one curve per Δμ |
| | `bc-fss` | the same for several slab sizes |
| | `histograms` | P(φ_col) at several ε |
| | `inspect` | ε grid coverage per Δμ |
| | `replot-family`, `scheme-comparison` | replot / compare saved BC curves |
| | `L-scaling` | ε_c and BC vs L across multi-L coex runs |
| | `distributions` | pooled P(m) from susc runs: first order vs continuous |

## Examples

`run/examples/` has one small, complete run of each kind. Each script's header
explains the physics, cost and outputs.

```bash
bash run/examples/susc.sh launch          # 3 ε × L = 16, 32; ~30 min per round on Della
bash run/examples/susc.sh status
bash run/examples/susc.sh plots

bash run/examples/coex.sh launch          # 5 ε at βΔμ = 1 on a 160 × 16 slab
bash run/examples/coex.sh status
bash run/examples/coex.sh plots           # FLEX vs simulation
bash run/examples/coex.sh stop

bash run/examples/criticality.sh coex     # BC vs ε from the coex example
bash run/examples/criticality.sh susc     # P(m) from the susc example
```

## Where outputs go

Everything is written under `output/` (gitignored). Set `FLEX_OUTPUT_ROOT` to use
another directory. The `run/` scripts change into that directory first, so a
relative path such as `--results-base SUSC_RUNS_S1A` means `output/SUSC_RUNS_S1A`.
Slurm logs go to `output/slurm_reports/`.

Slurm resources are set in `src/flexsim/coex/slurm_config.yml` (coex jobs) and in
the `#SBATCH` headers of `src/flexsim/susc/slurm/*.sh` (susc jobs).

## Tests

```bash
pip install -e '.[test]'
pytest
```

The tests do not need `lattice-gas`.
