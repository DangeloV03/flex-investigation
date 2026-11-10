# Equilibrium spin-spin correlation, 128×128 (mentor item 3, 2026-11-10)

Active-active spatial correlation C(r) = ⟨n₀ n_r⟩ − ρ² for the equilibrium system at
βε = −1.7627 (ε_c) and βε = −1.65.

**Physics.** These settings match the summer `correlation-length-investigation` runs:
- homo, βΔf = 0, k = 1, βΔμ = 0
- exact coexistence μ = −ln(e^{2|ε|} − 1), periodic L = 128
- n = 1 on BONDING sites, 0 otherwise

**Protocol.**
- 4 independent replicas per ε, each started from a random lattice with half the sites active (⟨m⟩ ≈ 0).
- Equilibration of 2×10⁵ time units, then 2×10⁶ of production with a snapshot every 1000 time units: 2000 snapshots per replica, 8000 per ε.
- Each replica is its own 1-CPU Slurm task (8 tasks in total).

## Run on Della (from the repo root, after `git pull` on the `experiments` branch)

```bash
bash experiments/2026-11-10_spin_corr_eq128/submit.sh smoke     # ~1 min, L=16, checks lattice_gas + resume + analysis
bash experiments/2026-11-10_spin_corr_eq128/submit.sh launch    # 8 tasks + 3 chained resume rounds
bash experiments/2026-11-10_spin_corr_eq128/submit.sh status
bash experiments/2026-11-10_spin_corr_eq128/submit.sh analyze   # anytime; uses whatever snapshots exist
```

Output: `/scratch/gpfs/WJACOBS/vd7294/flex-investigation/SPIN_CORR_EQ128/`
(`eps<tag>/replica_<r>/snap_*.npy`, `timeseries.csv`, `plots/`). Override with `OUT=...`.
Other knobs (env vars): `EPS_LIST`, `NREP`, `EQ_TIME`, `PROD_TIME`, `CHUNK_TIME`, `WALL`, `RESUME_ROUNDS`.

**Resumable.** Each task checkpoints after every chunk. A timed-out task is picked up by the next
chained round (`afterany`), and rounds exit at once for replicas that have finished. If 3 rounds
are not enough, rerun `launch` with `RESUME_ROUNDS=…`. It resumes and never restarts from scratch.

**Speed estimate.** Summer L=256 runs did 1.5×10⁵ time units in under 5 h on 2 CPUs, so 128²
should manage roughly 10⁵ time units per CPU-hour. The 2.2×10⁶ total then takes about 1 day per replica.
Check `status` (or the `wall=` field in `logs/*.out`) after an hour to confirm.

## What `analyze` writes (`plots/`)

- `corr_<eps>.csv` with these columns:
  - `C_glob`: subtracts the global ρ². This is the main estimator.
  - `C_glob_err`: standard error across replicas.
  - `C_snap`: subtracts each snapshot's own ρ², which was the summer convention. Near ε_c it cuts off the long-range tail.
  - `C_axis`: values along the lattice axes only.
  - `G_spin = 4 C`.
- `tau_<eps>.csv`: integrated autocorrelation time of m and |m| per replica, and n_eff = T_prod/(2τ).
  **Check this.** At ε_c and L = 128, τ can exceed the 1000-unit snapshot spacing, in which case the
  8000 snapshots are not all independent.
- `summary.csv`: ρ, ⟨|m|⟩, τ, n_eff, and a fitted ξ (Ornstein-Zernike form e^{−r/ξ}/√r), compared against
  the exact Ising ξ (1/ξ = ln coth K − 2K, K = |ε|/4, which gives ξ ≈ 8.7 at −1.65).
- `C_r.png`, `C_r_loglog.png`: C(r), with the exact r^{−1/4} reference at ε_c.
- `m_timeseries.png`: m(t) for every replica, to check that equilibration has finished.

Tested locally against a mock `lattice_gas` (the resume path, deduplication of a repeated
chunk, and the analysis). It has not yet been run with the real simulator; `smoke` does that.
