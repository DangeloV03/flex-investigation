# Quickstart — from zero to a running susceptibility campaign on Della

This guide assumes you have:
- A Princeton NetID with access to [Della](https://researchcomputing.princeton.edu/systems/della)
- Been added as a collaborator on the private [`lattice-gas`](https://github.com/moleary253/lattice-gas) GitHub repo (ask your PI if not)
- Been added as a collaborator on this (`flex-investigation`) repo

That's it. Everything else is explained below.

---

## Part 1 — First time setup (do this once)

### Step 1 — SSH into Della

Open your terminal and connect:

```bash
ssh <your-netid>@della.princeton.edu
```

You'll be on a login node. This is where you run everything that isn't a simulation (git, conda, file setup). Heavy computation goes to compute nodes via Slurm (handled automatically).

---

### Step 2 — Clone this repo onto Della's scratch filesystem

Simulation output files can be large. We put the project on **scratch** (`/scratch/gpfs/`), not home (`~`), because home has a small quota.

```bash
mkdir -p /scratch/gpfs/WJACOBS/$USER
cd /scratch/gpfs/WJACOBS/$USER
git clone https://github.com/DangeloV03/flex-investigation.git
cd flex-investigation
```

If this is your first time using GitHub on Della, you may need to authenticate. The easiest approach is a **personal access token**:
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)", give it `repo` scope, copy it
3. When `git clone` asks for a password, paste the token

You only need to do this once — git will cache it.

---

### Step 3 — Get the `lattice-gas` library

`lattice-gas` is a private Rust+Python library. You **cannot** `git clone` it on Della (the build requires Cargo which isn't easily available there). Instead, download a ZIP from GitHub on your **laptop**, then copy it to Della.

**On your laptop:**
1. Go to https://github.com/moleary253/lattice-gas in your browser
2. Click the green **Code** button → **Download ZIP**
3. Save the file (it will be named something like `lattice-gas-main.zip`)
4. Copy it to Della:

```bash
# Run this on your laptop (not on Della):
scp ~/Downloads/lattice-gas-main.zip <your-netid>@della.princeton.edu:~/software/
```

**Back on Della:**

```bash
mkdir -p ~/software
cd ~/software
unzip lattice-gas-main.zip
mv lattice-gas-main lattice-gas    # normalize the folder name
```

Now you should have the source at `~/software/lattice-gas/`. Confirm:

```bash
ls ~/software/lattice-gas/
# You should see: Cargo.toml  build-rust-lib.sh  src/  python/  ...
```

---

### Step 4 — Set up the Conda environment

Della uses modules to manage software. Load Anaconda first, then create a dedicated environment:

```bash
module load anaconda3/2024.10
conda create -n lattice python=3.11 -y
conda activate lattice
```

Install the Python dependencies for `flex-investigation`:

```bash
cd /scratch/gpfs/WJACOBS/$USER/flex-investigation
pip install maturin
pip install -r requirements.txt
```

`maturin` is the build tool for the Rust extension. The `requirements.txt` installs `numpy`, `scipy`, `matplotlib`, `pandas`, `pyyaml`, and `simple-slurm`.

**Add the environment script to your `~/.bashrc`** so everything is set up automatically every time you log in:

```bash
echo 'source /scratch/gpfs/WJACOBS/$USER/flex-investigation/scripts/env.sh' >> ~/.bashrc
source ~/.bashrc
```

`scripts/env.sh` does several things at once: loads the Anaconda module, activates the `lattice` conda environment, sets the `LD_LIBRARY_PATH` needed by the Rust extension, exports `PROJECT_ROOT`, and prints a quick status summary so you can see if imports are working. You should see output like:

```
flex-investigation environment
  host:            della9
  PROJECT_ROOT:    /scratch/gpfs/WJACOBS/<your-netid>/flex-investigation
  CONDA_PREFIX:    /home/<your-netid>/.conda/envs/lattice
  import check:    OK
```

If the import check says `FAILED` after Step 5, come back and re-source this script.

---

### Step 5 — Build the `lattice-gas` Rust extension

`lattice-gas` is written in Rust and compiled into a Python module. You need to build it once (and again whenever the `lattice-gas` source changes).

```bash
cd ~/software/lattice-gas
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
./build-rust-lib.sh
```

This will take a minute or two. You should see Cargo compiling, then `maturin` packaging the result. The `LD_LIBRARY_PATH` export above is needed for the build — `env.sh` handles setting this automatically for all future sessions.

---

### Step 6 — Verify the full stack

`env.sh` already runs an import check when sourced. Re-source it now to confirm everything is wired up:

```bash
source /scratch/gpfs/WJACOBS/$USER/flex-investigation/scripts/env.sh
```

Look for `import check: OK` in the output. If you see `FAILED`, the most likely cause is that the Rust extension didn't build correctly — go back to Step 5 and make sure `./build-rust-lib.sh` completed without errors while the `lattice` conda env was active.

---

### Step 7 — Configure Slurm for your account

Before submitting jobs, open `slurm_config.yml` and check a few things:

```bash
nano slurm_config.yml
```

The file looks like this:

```yaml
job_name: flex_sim
cpus_per_task: 4
mem: 8G
time: "24:00:00"
partition: cpu
# account: your_account   # uncomment if required
report_dir: ~/slurm_reports
output: "~/slurm_reports/%j.out"
error: "~/slurm_reports/%j.err"

setup_cmds:
  - "module load anaconda3/2024.10"
  - "source \"$(conda info --base)/etc/profile.d/conda.sh\""
  - "conda activate lattice"
  - "export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}\""
  - "export PYTHONUNBUFFERED=1"
```

Things to check:
- **`partition`** — leave as `cpu` unless your group has a dedicated partition
- **`account`** — if Della gives you an error about accounts when you run `sbatch`, uncomment this line and fill in your group's account name (ask your PI or run `sacctmgr show user $USER`)
- **`report_dir`** — this directory must exist before jobs run. Create it:

```bash
mkdir -p ~/slurm_reports
```

- **`setup_cmds`** — these run at the start of every Slurm job. The defaults should work if you followed Steps 4–5 exactly. If your Conda module version is different, update the `module load` line.

Save and close (`Ctrl+X`, then `Y`, then `Enter` in nano).

---

## Part 2 — Running the susceptibility campaign

This is the main workflow currently active. It measures χ, ⟨|m|⟩, and Binder cumulant U₄ across a grid of system sizes (L = 16, 32, 48, 64, 96, 128, 256) and interaction strengths (ε from −2.0 to −1.4), at exact coexistence μ = 2ε.

---

### Step 8 — Seed all the jobs (run once)

This creates one job file per (ε, L) combination and populates the job queue:

```bash
cd /scratch/gpfs/WJACOBS/$USER/flex-investigation
python generate_susceptibility_exact.py
```

You should see it printing job file names as they're created. When it finishes:

```bash
ls susceptibility_samples/exact/ | head -5    # should show many .json files
ls susceptibility_samples/exact/ | wc -l       # should be ~430 (61 ε values × 7 L sizes)
```

You only need to run this once. If some jobs fail and you re-run it, it skips jobs that already have results.

---

### Step 9 — Start the dispatcher in a tmux session

We use `tmux` so the dispatcher keeps running after you disconnect from SSH. If you close your terminal or your SSH connection drops, the jobs and the dispatcher keep going.

```bash
tmux new-session -d -s susceptibility
tmux send-keys -t susceptibility \
  "source /scratch/gpfs/WJACOBS/$USER/flex-investigation/scripts/env.sh && python run_susceptibility_all.py --phase exact" \
  Enter
```

This starts `run_susceptibility_all.py` in a detached tmux session named `susceptibility`. The `source env.sh` at the front ensures conda and `LD_LIBRARY_PATH` are set even though the tmux window starts with a fresh shell. The dispatcher will submit jobs to Slurm, wait for them to finish, and keep going until the queue is empty.

To watch what it's doing:

```bash
tmux attach -t susceptibility
# Press Ctrl+b then d to detach without stopping it
```

---

### Step 10 — Check progress

While jobs are running, you can check how many have finished:

```bash
# How many result directories exist so far
find susceptibility_results -name "susceptibility_data.csv" | wc -l
# Full run = 61 ε × 7 L = 427 directories

# See your jobs in the Slurm queue
squeue -u $USER

# Watch Slurm log output (replace XXXXXX with a job ID from squeue)
cat ~/slurm_reports/XXXXXX.out
```

A typical job takes 5–30 minutes depending on L. L=256 jobs take the longest.

---

### Step 11 — Generate plots once enough data is in

You can plot at any time, even while jobs are still running. The script skips missing data gracefully.

```bash
python plot_susceptibility.py \
    --results susceptibility_results \
    --outdir plots/susceptibility
```

This creates four plots in `plots/susceptibility/`:
- `chi_vs_epsilon.png` — susceptibility χ(ε) for each L; peak location estimates ε_c
- `abs_m_vs_epsilon.png` — order parameter ⟨|m|⟩(ε)
- `binder_vs_epsilon.png` — Binder cumulant U₄(ε); where curves for different L cross = ε_c
- `peak_chi_vs_L.png` — how the peak χ scales with L (log-log)

---

### Step 12 — Finite-size scaling collapse (after all data is in)

Once all jobs are done, run the FSS analysis. This finds the critical point ε_c and the critical exponents by collapsing data from all L onto a single master curve:

```bash
python plot_fss.py \
    --results susceptibility_results \
    --outdir plots/fss \
    --xc -1.75 \
    --xr -5 5
```

`--xc -1.75` is the initial guess for ε_c (the optimizer will refine it).  
`--xr -5 5` restricts the quality function to the region near the peak on the rescaled axis.

This produces:
- `plots/fss/fss_chi_collapse.png` — χ·L^(−γ/ν) vs (ε − ε_c)·L^(1/ν)
- `plots/fss/fss_m_collapse.png` — ⟨|m|⟩·L^(β/ν) vs (ε − ε_c)·L^(1/ν)

And prints the optimized exponents to the terminal.

---

### Step 13 — Copy plots back to your laptop

```bash
# Run this on your laptop:
scp -r <your-netid>@della.princeton.edu:/scratch/gpfs/WJACOBS/<your-netid>/flex-investigation/plots ./
```

Or use `rsync` to sync only new files:

```bash
rsync -avz <your-netid>@della.princeton.edu:/scratch/gpfs/WJACOBS/<your-netid>/flex-investigation/plots/ ./plots/
```

---

## Part 3 — Day-to-day workflow

Once set up, your typical session looks like this:

```bash
# 1. SSH in
ssh <your-netid>@della.princeton.edu
# env.sh runs automatically from ~/.bashrc — you should see the status summary printed
# If you don't see it, run:  source /scratch/gpfs/WJACOBS/$USER/flex-investigation/scripts/env.sh

# 2. Pull any code updates (env.sh already cd'd you into the project)
git pull

# 3. Check if the dispatcher is still running
tmux attach -t susceptibility   # Ctrl-b then d to detach without stopping it

# 4. Check how many jobs are done
find susceptibility_results -name "susceptibility_data.csv" | wc -l

# 5. Plot the current state
python plot_susceptibility.py --results susceptibility_results --outdir plots/susceptibility
```

---

## Troubleshooting

**`ImportError: No module named 'lattice_gas'`**
```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# If that fixes it, add the export to ~/.bashrc
```

**`conda: command not found`**
```bash
module load anaconda3/2024.10
conda activate lattice
```

**`FileNotFoundError: No susceptibility_data.csv`** when plotting
The `--results` path must be the directory that *contains* the per-run subdirectories. Check with:
```bash
find susceptibility_results -name "susceptibility_data.csv" | head -3
# The path to pass to --results is the part before the first subdirectory name
```

**Slurm jobs fail immediately**
```bash
cat ~/slurm_reports/<jobid>.err
# Usually an environment issue — check that slurm_config.yml setup_cmds matches your conda setup
```

**`tmux: command not found`**
Tmux is always available on Della login nodes. If you somehow don't see it, just run the dispatcher directly:
```bash
python run_susceptibility_all.py --phase exact
# (this will stop when your SSH session ends, so only use this for short tests)
```

**Dispatcher stopped mid-campaign**
Just restart it — it picks up where it left off:
```bash
tmux new-session -d -s susceptibility
tmux send-keys -t susceptibility \
  "source /scratch/gpfs/WJACOBS/$USER/flex-investigation/scripts/env.sh && python run_susceptibility_all.py --phase exact" \
  Enter
```

**Want to update the code after a change is pushed**
```bash
cd /scratch/gpfs/WJACOBS/$USER/flex-investigation
git pull
```
No need to rebuild `lattice-gas` unless the Rust source changed.
