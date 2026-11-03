# Ly = 40 coex runners

Every `run/coex_ly40_*.sh` script is a thin wrapper around one shared runner,
`run/coex_ly40_extend.sh`. A wrapper only sets the settings below and then
calls the runner. This page lists every setting, what it does, and its default.

## Quick start (Della)

```bash
cd <your clone of flex-investigation>          # on branch cleanup, after git pull
module load anaconda3/2024.10
conda activate lattice

bash run/coex_ly40_d3f.sh all       # create the jobs and start
bash run/coex_ly40_d3f.sh status    # progress
bash run/coex_ly40_d3f.sh stop      # stop the tmux session (Slurm jobs already queued keep running)
```

Data always goes to `/scratch/gpfs/WJACOBS/vd7294/flex-investigation`, whichever
clone or account you run from.

## Actions (first argument)

| Action | What it does |
|---|---|
| `generate` | Backs up `manage.csv` and `queue.json`, then **appends** the new ε × μ jobs to them. Existing rows are never removed, and ε already in the ledger are skipped. Writes a marker file so the same grid isn't added twice. |
| `start` | Starts tmux session `coex-<EXP>-ly40` with two windows: `dispatcher` (submits Slurm jobs) and `analyzer` (fits μ_coex and queues extra μ where needed). It keeps going until every ε in the range is analyzed. |
| `all` | `generate`, then `start`. |
| `status` | Counts the analyzed and fitted ε in the range, the pending and in-flight jobs in the queue, whether the tmux session is running on this login node, and your Slurm job count. |
| `stop` | Kills this campaign's tmux session only. |

tmux sessions belong to one login node. If `status` says "not running", check the other login node before you start it again.

## Settings

Set them in a wrapper with `export`, or on the command line before `bash`.

| Setting | Default | Meaning |
|---|---|---|
| `EXP` | **required** | Campaign name. The data goes in `COEX_RUNS_<EXP>/ly40/`, and the tmux session is `coex-<EXP>-ly40`. |
| `DELTA_MU` | **required** | βΔμ, the chemical drive. |
| `WRAPPER` | **required** | Absolute path of the script being run. The tmux windows call it again, so the settings carry over. Wrappers set it automatically. |
| `SCHEME` | `positive_drive` | Simulation rate scheme: `homo`, `positive_drive` or `negative_drive`. |
| `FLEX_SCHEME` | `2` | Scheme for the FLEX prediction that picks the starting μ. **It must match `SCHEME`** (see the table below). |
| `DELTA_F` | `0.0` | βΔf. |
| `K` | `1.0` | Base recycling rate k. |
| `EPS_MIN` | `-2.1` | Coldest ε (βε) in the grid. |
| `EPS_MAX` | `-1.6` | Warmest ε in the grid. |
| `EPS_STEP` | `0.01` | ε spacing. |
| `MU_WINDOW` | `0.05` | The first μ sweep at each ε spans FLEX μ_coex ± this. |
| `N_MU_POINTS` | `10` | Number of μ values in the first sweep. The analyzer adds more later if they don't bracket coexistence. |
| `MAX_CONCURRENT` | `30` | Most Slurm jobs this campaign keeps queued or running at once. |
| `NEW_BASE` | `0` | `1` = create `COEX_RUNS_<EXP>/ly40` if it doesn't exist. Without it, a missing folder is an error, which catches a mistyped `EXP`. |
| `NEW_CAMPAIGN` | `0` | `1` (together with `NEW_BASE=1`) = also create `COEX_RUNS_<EXP>` itself, for a brand-new campaign. New folders get ACLs so both accounts (vd7294 and mo9089) can write in them. |
| `FORCE` | `0` | `1` = run `generate` even if this exact grid was generated before. Duplicate ε are still skipped. |
| `DATA_ROOT` | `/scratch/gpfs/WJACOBS/vd7294/flex-investigation` | Where the data lives. Leave it as is. |

Fixed, not settable: Ly = 40 and Lx = 400. Each job gets 1 GB of memory (`src/flexsim/coex/slurm_config.yml`) and a 24 h wall time (`LY_WALLTIME` in `src/flexsim/coex/dispatcher.py`).

### Scheme values

| Run type | `SCHEME` | `FLEX_SCHEME` | Effective rate k |
|---|---|---|---|
| Homogeneous | `homo` | `1` | k |
| Positive drive | `positive_drive` | `2` | k · min(1, exp(−Δf − Δμ + 2ε)) |
| Negative drive | `negative_drive` | `3` | k · min(1, exp(−Δf − Δμ − 2ε)) |

## Existing runners

All of them use βΔf = 0 and k = 1.

| Script | EXP | Scheme | βΔμ | ε range | Step |
|---|---|---|---|---|---|
| `coex_ly40_c2b.sh` | C2B | homo | −1 | −2.1005 … −1.3805 | 0.02 |
| `coex_ly40_cold_c3b.sh` | C3B | negative_drive | −1 | −2.0813 … −1.7813 | 0.02 |
| `coex_ly40_cold_c3c.sh` | C3C | negative_drive | +2 | −2.088902 … −1.908902 | 0.02 |
| `coex_ly40_cold_d3a.sh` | D3A | positive_drive | +1 | −2.08321 … −1.68321 | 0.02 |
| `coex_ly40_cold_d3b.sh` | D3B | positive_drive | −1 | −2.091185 … −1.811185 | 0.02 |
| `coex_ly40_extend_d3c.sh` | D3C | positive_drive (default) | +2 | −2.1 … −1.6 (default) | 0.01 |
| `coex_ly40_extend_d3d.sh` | D3D | positive_drive (default) | +3 | −2.1 … −1.6 (default) | 0.01 |
| `coex_ly40_extend_d3e.sh` | D3E | positive_drive (default) | +4 | −2.1 … −1.6 (default) | 0.01 |
| `coex_ly40_d3f.sh` | D3F | positive_drive | −2 | −2.2 … −1.5 | 0.02 |

The D3C/D3D/D3E wrappers set only `EXP` and `DELTA_MU`. Everything else in those runs came from the defaults above.

## Starting a new campaign

Copy `run/coex_ly40_d3f.sh`, then change `EXP`, `SCHEME`, `FLEX_SCHEME`, `DELTA_MU` and the ε range. Or skip the file and set everything inline. For example, homogeneous βΔμ = +2:

```bash
EXP=C2C SCHEME=homo FLEX_SCHEME=1 DELTA_MU=2.0 NEW_BASE=1 NEW_CAMPAIGN=1 \
EPS_MIN=-2.2 EPS_MAX=-1.5 EPS_STEP=0.02 \
WRAPPER=$PWD/run/coex_ly40_extend.sh \
bash run/coex_ly40_extend.sh all
```

The inline form needs the same settings again for `status` and `stop`, so a wrapper file is easier for anything you'll check on more than once.

## Rules when two accounts share the data

- **Run each campaign from one account at a time.** If the other account already has its tmux session running, don't start one too.
- To move a campaign from one account to the other, run `stop` on the first account, then `start` on the second. Jobs already in Slurm finish normally, and the new dispatcher won't resubmit them.
- Don't delete anything in `COEX_RUNS_*`. `generate` only ever appends, and it makes backups (`*.bak_extend_<date>`) first.

## What each campaign folder holds

```
COEX_RUNS_<EXP>/ly40/
  manage.csv      one row per ε: FLEX μ_coex, fitted μ_coex (mu_coex_FITTED) and its error, and the status of each step
  queue.json      jobs that are pending or in flight
  samples/        job inputs (samples/done/ once a job finishes)
  results/        simulation outputs
  .extend_<min>_<max>_<step>   marker: this grid has been generated
```

## Lower-level command

The runner calls `python run/coex.py generate-susc` with these flags:
`--scheme`, `--flex-scheme`, `--delta-f`, `--delta-mu`, `--k`, `--ly`, `--eps-min`, `--eps-max`, `--eps-step`, `--mu-window`, `--n-mu-points`, `--samples-dir`, `--results-dir`, `--manage`, `--manifest`.

It then runs `python run/coex.py dispatch` and `python run/coex.py analyze`.

To see all of their options:
```bash
python run/coex.py generate-susc --help
python run/coex.py dispatch --help
python run/coex.py analyze --help
```
