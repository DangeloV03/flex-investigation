# flex-investigation

Susceptibility runs of a driven lattice gas on an L × L lattice, using the
[`lattice-gas`](https://github.com/moleary253/lattice-gas) Rust engine.

```bash
sbatch submit.sh params.json                     # on the cluster
python src/flexsim/susc/replica.py params.json   # directly
```

`params.json` holds every parameter of a run:

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

Outputs in `outdir`: `m_timeseries_<i>.csv`, `final_lattice_<i>.npy`,
`susceptibility_data.csv` (one row per replica) and a copy of the parameters.

Requires a conda env `lattice` with `numpy` and a built `lattice_gas`.
