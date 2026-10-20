# First-time setup on Della

You'll need a Princeton NetID with Della access, and collaborator access to this
repo and to the private [`lattice-gas`](https://github.com/moleary253/lattice-gas)
repo (ask your PI).

## 1. Clone onto scratch

Run outputs can be large, and `/home` has a small quota, so the checkout goes on
`/scratch`:

```bash
ssh <netid>@della.princeton.edu
mkdir -p /scratch/gpfs/WJACOBS/$USER && cd /scratch/gpfs/WJACOBS/$USER
git clone https://github.com/DangeloV03/flex-investigation.git
cd flex-investigation
```

When GitHub asks for a password, use a personal access token with `repo` scope.

## 2. Conda environment

```bash
module load anaconda3/2024.10
conda create -n lattice python=3.11 -y
conda activate lattice
pip install maturin
pip install -e .            # flexsim and its Python dependencies
```

## 3. Build `lattice-gas`

Download the repo as a ZIP on your laptop (GitHub → Code → Download ZIP), then:

```bash
# laptop
scp ~/Downloads/lattice-gas-main.zip <netid>@della.princeton.edu:~/software/

# Della, with the lattice env active
cd ~/software && unzip lattice-gas-main.zip && mv lattice-gas-main lattice-gas
cd lattice-gas
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
./build-rust-lib.sh
```

Rebuild only when the `lattice-gas` source changes.

## 4. Login environment

```bash
echo 'source /scratch/gpfs/WJACOBS/$USER/flex-investigation/env.sh' >> ~/.bashrc
source ~/.bashrc
```

`env.sh` loads Anaconda, activates `lattice`, sets the library path for the Rust
extension, changes into the repo and runs an import check. Look for
`import check: OK`. If it says `FAILED`, step 3 didn't finish, or step 2's
`pip install -e .` was run in a different env.

## 5. Slurm account

All jobs write their logs to `output/slurm_reports/`, which is created
automatically. If `sbatch` complains about your account, add it:

- coex jobs: uncomment `account:` in `src/flexsim/coex/slurm_config.yml`
- susc jobs: add `#SBATCH --account=<account>` to `src/flexsim/susc/slurm/*.sh`

`sacctmgr show user $USER` lists your accounts.

## 6. Try it

```bash
bash run/examples/susc.sh launch
squeue -u $USER
```

Then see the Examples section of the [README](../README.md).

## Day to day

```bash
ssh <netid>@della.princeton.edu      # env.sh runs from ~/.bashrc
git pull                             # no reinstall needed: pip install -e is live
tmux attach -t flex-investigation    # coex dispatcher/analyzer; Ctrl-b d to detach
```

Use `tmux` (or the `start` commands) for anything that must outlive your SSH
session. Susc sweeps run entirely inside Slurm and need no tmux.

## Troubleshooting

- **`No module named 'lattice_gas'`**: the `lattice` env isn't active, or
  `LD_LIBRARY_PATH` is missing `$CONDA_PREFIX/lib`. Re-source `env.sh`.
- **`No module named 'flexsim'`**: run `pip install -e .` in the repo with the
  `lattice` env active.
- **Jobs die at once with `Disk quota exceeded`**: something is writing to
  `/home`. Check that the checkout, or `FLEX_OUTPUT_ROOT`, is on `/scratch`.
- **A Slurm job failed**: read `output/slurm_reports/<name>_<jobid>.err`
  (susc jobs) or `output/slurm_reports/<jobid>.err` (coex jobs).
