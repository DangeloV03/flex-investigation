# flex-investigation

Susceptibility runs of a driven lattice gas on an L × L lattice, using the
[`lattice-gas`](https://github.com/moleary253/lattice-gas) Rust engine.

Requires a conda env `lattice` with `numpy` and a built `lattice_gas`.

## Single run

```bash
sbatch submit.sh params.json                     # on the cluster
python src/flexsim/susc/replica.py params.json   # directly
```

| Key | Meaning |
|---|---|
| `epsilon`, `mu`, `delta_mu`, `delta_f`, `k`, `beta` | physics (βε, βμ, βΔμ, βΔf, rate k, β) |
| `scheme` | `homo`, `positive_drive` or `negative_drive` |
| `L` | lattice side |
| `n_replicas` | independent replicas, run in parallel (≤ CPUs of the job) |
| `eq_time`, `prod_time`, `prod_chunks` | equilibration time, production time, number of samples |
| `initial_fraction` | start fill: even replicas this fraction BONDING, odd ones 1 − it |
| `seed` | base seed (replica i uses seed + 2i) |
| `outdir` | output folder |

## Sweep

```bash
sbatch sweep.sh mu_database.csv                               # on the cluster
python src/flexsim/susc/sweep.py mu_database.csv [outroot]    # directly
python src/flexsim/susc/average.py mu_database.csv [outroot]  # directly
```

Each CSV row (`lattice_size,epsilon,mu`) is one point. All other parameters and
the replicas per point (M) are set at the top of `sweep.py`. N CPUs
(`--cpus-per-task` in `sweep.sh`) run all point × replica tasks from one pool,
so several points run at once when M < N. Output goes to
`<outroot>/L<L>_eps<epsilon>_mu<mu>/`, where `outroot` defaults to `output`
(set `OUTROOT` in `sweep.sh`).

After the sweep, `sweep.sh` runs `average.py`, which averages `<m>` and `chi`
over the M replicas of each point and writes `mu_database_iteration1.csv`
(next to the input CSV): a copy of it with extra columns `m_mean` and `chi_mean`.

## Outputs

Per `outdir`: `params.json`, `m_timeseries_<i>.csv`, `final_lattice_<i>.npy`,
`susceptibility_data.csv` (one row per replica).
