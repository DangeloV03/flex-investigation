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

---

# Susceptibility (susc) runners

A susc campaign runs L × L lattices at the coexistence μ from a Ly = 40 coex campaign, and measures
χ, U₄ and ⟨|m|⟩ at each (ε, L). Every `run/susc_fitted_coex_*.sh` script is a thin wrapper around one
shared runner, `run/susc_fitted_coex.sh`. A wrapper only sets the settings below and then calls the runner.

These scripts are on branches `cleanup` and `newton-mu` (not yet on `main`).

## Quick start (Della)

```bash
cd <your clone of flex-investigation>          # on branch cleanup or newton-mu, after git pull
module load anaconda3/2024.10
conda activate lattice

bash run/susc_fitted_coex_neg_dmu2.sh fit             # cubic fit of coex μ(ε) -> fitted_mu.csv
DRY_RUN=1 bash run/susc_fitted_coex_neg_dmu2.sh launch  # print the sbatch commands, submit nothing
bash run/susc_fitted_coex_neg_dmu2.sh launch          # submit; top-up rounds chain on their own
bash run/susc_fitted_coex_neg_dmu2.sh status          # queue + latest jump report
bash run/susc_fitted_coex_neg_dmu2.sh analyze         # χ/U₄ plots + FSS
```

No tmux needed: once `launch` returns, everything runs inside Slurm. Data always goes to
`/scratch/gpfs/WJACOBS/vd7294/flex-investigation`, whichever clone or account you run from.

## How a campaign runs

1. **fit**: fits a polynomial (cubic by default) to `mu_coex_FITTED(ε)` from the coex `manage.csv`,
   using only the coex points with ε in [`FIT_MIN`, `FIT_MAX`]. Outliers are rejected. The fitted μ on
   the susc ε grid goes to `<WORK>/fitted_mu.csv`, and `fit.png` / `fit.json` show the fit. The fit
   smooths over individual coex points that came out wrong.
2. **launch**: submits one Slurm job per (ε, L) (job names `susc_L16`, `susc_L32`, …), plus one
   `susc_check` job that waits for all of them (`afterany`, so a job that times out doesn't block it).
   Each job runs 16 replicas, with 1×10⁶ equilibration and 2×10⁶ production KMC time.
3. **check / top-ups**: the check job counts the magnetisation jumps J (sign flips of m) for every
   (ε, L). Points with mean J below `J_THRESHOLD` get a top-up job (`susc_top_L<L>`, +1×10⁶ production,
   continuing from the last lattice), and another check job is chained after them. This repeats until
   every point passes. If 3 rounds in a row add no data, the chain stops (STALLED in the check log).
   Each round writes `<WORK>/runs/jump_report.md`.
4. **analyze**: links every point with |⟨m⟩| ≤ `M_CUT` into `<WORK>/combined/` and plots from there.
   Points with ⟨m⟩ far from 0 sat off coexistence, so their χ is biased. They stay on disk and are
   listed in `combined/dropped_points.csv`.

## Actions (first argument)

| Action | What it does |
|---|---|
| `fit` | Cubic fit of the coex μ(ε) → `<WORK>/fitted_mu.csv`, `fit.png`, `fit.json`. |
| `launch` | Submits the sweep and the first check job, and appends the output to `<WORK>/launch.log`. It refuses if `<WORK>/runs` already has data (`FORCE=1` overrides). `DRY_RUN=1` only prints the commands. |
| `status` | Counts your `susc_*` jobs by name and state, and shows the latest jump report, including the pairs that haven't passed yet. |
| `analyze` | `combine` (drops |⟨m⟩| > `M_CUT`), then the χ/U₄ plots (`<WORK>/plots`) and FSS (`<WORK>/plots/fss`). It can run at any time, including while top-ups are still going. |

## Settings

Set them in a wrapper with `export`, or on the command line before `bash`.

| Setting | Default | Meaning |
|---|---|---|
| `WORK` | **required** | Campaign folder, e.g. `SUSC_RUNS_NEG_DMU2_FITTED_COEX`. Runs go in `<WORK>/runs`. |
| `PHYS_CAMPAIGN` | **required** | An existing `SUSC_RUNS_*` folder with the same physics. `fit` only reads the physics from it. |
| `COEX` | **required** | Coex ledger with the fitted μ, e.g. `COEX_RUNS_C3C/ly40/manage.csv`. |
| `FIT_MIN`, `FIT_MAX` | **required** | ε range of the coex points used in the fit. Keep it a bit wider than the susc grid, and only where the coex fits are good. |
| `EPS_MIN`, `EPS_MAX` | **required** | Susc ε grid (endpoints included). |
| `EPS_STEP` | `0.005` | Susc ε spacing. |
| `DELTA_MU` | **required** | βΔμ. |
| `SCHEME` | **required** | `homo`, `positive_drive` or `negative_drive`. |
| `DELTA_F` | `0.0` | βΔf. |
| `K` | `1.0` | Base recycling rate k. |
| `DEGREE` | `3` | Polynomial degree of the coex fit. |
| `SIZES` | `16 32 48 64` | L values. L = 96 and 128 are much slower and are left out of the current campaigns. |
| `J_THRESHOLD` | `10` | Mean jumps needed before a point counts as done. |
| `M_CUT` | `0.05` | `analyze` drops points with |⟨m⟩| above this. |
| `DRY_RUN` | `0` | `1` = `launch` prints the sbatch commands only. |
| `FORCE` | `0` | `1` = `launch` even if `<WORK>/runs` already has data, e.g. to add sizes later: `SIZES="96 128" FORCE=1 bash <wrapper> launch`. |
| `DATA_ROOT` | `/scratch/gpfs/WJACOBS/vd7294/flex-investigation` | Where the data lives. Leave it as is. |

Fixed in the Slurm scripts (`src/flexsim/susc/slurm/`): 16 CPUs and 512 MB per CPU per job, 24 h wall
time, and the equilibration/production times above.

## Existing susc runners

| Script | WORK | Scheme | βΔμ | Coex fit | ε grid | L |
|---|---|---|---|---|---|---|
| `susc_fitted_coex_neg_dmu2.sh` | `SUSC_RUNS_NEG_DMU2_FITTED_COEX` | negative_drive | +2 | C3C ly40, ε ∈ [−1.95, −1.60] | −1.878902 … −1.678902, step 0.005 | 16–64 |
| `susc_fitted_coex_neg_dmu1.sh` | `SUSC_RUNS_NEG_DMU1_FITTED_COEX` | negative_drive | +1 | C3A ly40 (`manage_recovered.csv`*), ε ∈ [−1.89, −1.55] | −1.823718 … −1.623718, step 0.005 | 16–64 |
| `susc_fitted_coex_neg_dmu3.sh` | `SUSC_RUNS_NEG_DMU3_FITTED_COEX` | negative_drive | +3 | C3D ly40, ε ∈ [−2.13, −1.78] | −2.05772 … −1.85772, step 0.005 | 16–64 |
| `susc_fitted_coex_neg_dmu_m1.sh` | `SUSC_RUNS_NEG_DMU-1_FITTED_COEX` | negative_drive | −1 | C3B ly40, ε ∈ [−1.95, −1.55] | −1.8232 … −1.6232, step 0.005 | 16–64 |
| `susc_fitted_coex_neg_dmu_m2.sh` | `SUSC_RUNS_NEG_DMU-2_FITTED_COEX` | negative_drive | −2 | C3F ly40, ε ∈ [−1.95, −1.60] | −1.8232 … −1.6232, step 0.005 (ε_c assumed = C3B's) | 16–64 |
| `susc_pos_dmu2_shift.sh` | `SUSC_RUNS_POS_DMU2_shift` | positive_drive | +2 | D3C ly40, ε ∈ [−1.85, −1.55] | −1.800042 … −1.635042, step 0.005 | 16–128 |

\* A generator bug had wiped the C3A Ly=40 fits for ε ≥ −1.748718. `manage_recovered.csv` is `manage.csv` with
those 121 fits put back from the analyzer log (`logs/c3_auto_20260923T130922Z.log`). They match the μ the old
`SUSC_RUNS_NEG_DMU1` ran at to 5×10⁻⁷. `manage.csv` itself is unchanged.

`susc_pos_dmu2_shift.sh` predates the shared runner. It has the same four actions, but its settings
are hard-coded in the file, and its `analyze` also merges the older `SUSC_RUNS_POS_DMU2` points.

`susc_newton.sh` / `susc_newton_homo_1.0.sh` (Newton correction of μ for each (ε, L)) are paused and
not covered here.

## Starting a new susc campaign

Copy `run/susc_fitted_coex_neg_dmu2.sh`, then change `WORK`, `PHYS_CAMPAIGN`, `COEX`, the fit range,
the ε grid and the physics. Run `fit` first and look at `fit.png` before launching. For example,
negative drive βΔμ = 3 from the C3D coex:

```bash
cp run/susc_fitted_coex_neg_dmu2.sh run/susc_fitted_coex_neg_dmu3.sh
# edit: WORK=SUSC_RUNS_NEG_DMU3_FITTED_COEX PHYS_CAMPAIGN=SUSC_RUNS_NEG_DMU3
#       COEX=COEX_RUNS_C3D/ly40/manage.csv DELTA_MU=3.0, plus FIT_MIN/MAX and EPS_MIN/MAX
bash run/susc_fitted_coex_neg_dmu3.sh fit
DRY_RUN=1 bash run/susc_fitted_coex_neg_dmu3.sh launch
bash run/susc_fitted_coex_neg_dmu3.sh launch
```

A good ε grid is ε_c ± 0.1, where ε_c is the Ly = 40 Binder-cumulant crossing of that coex campaign.

## Stopping a susc campaign

Every campaign uses the same job names (`susc_L*`, `susc_top_L*`, `susc_check`). So **don't cancel
by name** (`scancel -n …`): it would also kill the other campaigns' jobs. Cancel by job ID instead.
The IDs of each submission are in `<WORK>/runs/slurm_jobs.csv` (and `<WORK>/launch.log`). Cancel the
pending `susc_check` job too, or it will submit another round of top-ups.

## What each susc folder holds

```
<WORK>/
  fitted_mu.csv, fit.png, fit.json   coex fit and the μ used for every ε
  launch.log                         output of launch (job IDs)
  runs/_<L>_<L>_.../_<eps>/          one folder per (ε, L): susceptibility_data.csv, m_timeseries_*.csv, lattices
  runs/jump_report.md                latest check round: J per (ε, L)
  combined/                          links to the points kept by analyze (+ dropped_points.csv)
  plots/, plots/fss/                 χ, U₄, ⟨|m|⟩ vs ε, and FSS
```

## Lower-level commands

The runner calls `python run/susc.py refit fit`, `python run/susc.py sweep` and
`python run/susc.py refit combine`. To see all their options:

```bash
python run/susc.py sweep --help
python run/susc.py check --help        # rerun a check / top-up round by hand
python run/susc.py refit fit --help
python run/susc.py refit combine --help
```
