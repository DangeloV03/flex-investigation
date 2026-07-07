# Multiplier Test — does peak χ grow with production run length?

Stress-test whether the measured peak susceptibility χ^max(L) is converged in
production time. In the critical window **|ε| ∈ [1.7, 1.8]** (ε ∈ [−1.8, −1.7]) we
re-run the exact-μ (μ = 2ε) production measurement at four production-time
multipliers and plot **peak χ vs L** for each. If the curves rise with the multiplier,
production is under-converged; if they overlie each other, χ^max is time-converged.

Baseline is `PROD_TIME = 200000` in `susceptibility/run_susceptibility.sh`. Each
variant scales `PROD_CHUNKS` with `PROD_TIME` so the per-sample interval
`chunk_time = PROD_TIME / PROD_CHUNKS = 100` stays constant — "longer" means *more
samples at the same measurement cadence*.

| folder | script                        | PROD_TIME | PROD_CHUNKS |
|--------|-------------------------------|-----------|-------------|
| `2x/`  | `scripts/run_susceptibility_2x.sh` | 400000    | 4000        |
| `3x/`  | `scripts/run_susceptibility_3x.sh` | 600000    | 6000        |
| `4x/`  | `scripts/run_susceptibility_4x.sh` | 800000    | 8000        |
| `5x/`  | `scripts/run_susceptibility_5x.sh` | 1000000   | 10000       |

Each multiplier sweeps **21 ε points** in [−1.8, −1.7] (standard `--eps-step 0.005`),
and each ε job loops **all 7 L** = 16, 32, 48, 64, 96, 128, 256 internally.

## Layout

```
multiplier_test/
  scripts/run_susceptibility_{2,3,4,5}x.sh   # copies of the base script, scaled PROD_TIME/PROD_CHUNKS
  2x/ 3x/ 4x/ 5x/                            # results-base per multiplier (susceptibility_* dirs land here)
  plots/{2,3,4,5}x/                          # per-multiplier peak_chi_vs_L(_pooled).png + .csv
  plots/peak_chi_vs_L_comparison.png         # overlay of all four (plot_multiplier_comparison.py)
  plot_multiplier_comparison.py
```

## Run (on Della, from the repo root)

Repo lives at `/scratch/gpfs/WJACOBS/vd7294/flex-investigation`. Submit each
multiplier (add `--dry-run` first to preview the 21 sbatch commands):

```bash
python susceptibility/sweep_susceptibility.py --script multiplier_test/scripts/run_susceptibility_2x.sh --eps-min -1.8 --eps-max -1.7 --eps-step 0.005 --results-base multiplier_test/2x --label mult2x
python susceptibility/sweep_susceptibility.py --script multiplier_test/scripts/run_susceptibility_3x.sh --eps-min -1.8 --eps-max -1.7 --eps-step 0.005 --results-base multiplier_test/3x --label mult3x
python susceptibility/sweep_susceptibility.py --script multiplier_test/scripts/run_susceptibility_4x.sh --eps-min -1.8 --eps-max -1.7 --eps-step 0.005 --results-base multiplier_test/4x --label mult4x
python susceptibility/sweep_susceptibility.py --script multiplier_test/scripts/run_susceptibility_5x.sh --eps-min -1.8 --eps-max -1.7 --eps-step 0.005 --results-base multiplier_test/5x --label mult5x
```

Each submits 21 sbatch jobs (84 total). μ defaults to 2ε (exact coexistence).

**Wall-time note:** the run scripts request a 3-day wall (`#SBATCH --time=3-00:00:00`)
to give 4×/5× at L=256 room to finish. If a job still times out, re-run the same sweep
(or add `--num-batches N`) to append replicas.

## Plot & analyze

Per-multiplier peak χ vs L (repeat for 3x/4x/5x), then the overlay:

```bash
python susceptibility/plot_susceptibility.py --results multiplier_test/2x --outdir multiplier_test/plots/2x --pooled
python susceptibility/plot_susceptibility.py --results multiplier_test/3x --outdir multiplier_test/plots/3x --pooled
python susceptibility/plot_susceptibility.py --results multiplier_test/4x --outdir multiplier_test/plots/4x --pooled
python susceptibility/plot_susceptibility.py --results multiplier_test/5x --outdir multiplier_test/plots/5x --pooled

python multiplier_test/plot_multiplier_comparison.py --plots-root multiplier_test/plots --out multiplier_test/plots/peak_chi_vs_L_comparison.png
```

The comparison script overlays the four χ^max(L) curves (log-log, error bars) and
prints the fitted γ/ν and χ^max per multiplier. Optionally add `--tail-fraction 0.5`
to `plot_susceptibility.py` to check equilibration within each run.
```
