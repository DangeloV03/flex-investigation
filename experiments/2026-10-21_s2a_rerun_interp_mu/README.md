# S2A rerun at linearly interpolated μ (2026-10-21/22)

Question: if the 4 bad-fit ε in S2A are rerun at μ linearly interpolated from the nearest good
S2A neighbours, is |⟨m⟩| < 0.05 at every L?

- Data on Della: `/scratch/gpfs/WJACOBS/vd7294/flex-experiments/analysis_s2a_m/rerun_interp_mu/`
  (`SUSC_RUNS_S2A_INTERP/`). `SUSC_RUNS_S2A` was not touched.
- Same protocol as S2A (homo, βΔf=0, k=1, βΔμ=1, L 16–128, top-ups to J ≥ 10). Private copies of
  the job scripts (`scripts/`) differ only in log paths and seed base 500000. The original used 7000,
  so seeds never overlap. Launched 2026-10-21; all 30 pairs passed at round 5 (2026-10-22).
- `mu_source.csv`: new and original μ. `-1.735` is a control at its original μ with new seeds.
- `plot_rerun_compare.py` → `rerun_compare.png` / `.csv`.

## Result
| ε | μ change | original ⟨m⟩, L=128 | rerun ⟨m⟩, L=128 | rerun, all L within ±0.05? |
|---|---|---|---|---|
| −1.665 | −1.9e-3 | +0.139 | +0.007 | yes |
| −1.685 | +8.9e-4 | −0.076 | +0.016 | yes |
| −1.740 | −1.5e-3 | +0.518 | +0.045 | no (L=64 +0.051±0.027, L=96 +0.062±0.022) |
| −1.745 | +5.3e-4 | −0.332 | +0.138 | no (overshoots; positive at L ≥ 32) |
| −1.735 control | 0 | −0.007 | +0.013 | yes (reproduces original) |

- The two hotter ε are fixed outright. −1.74 drops ~10× and is borderline.
- −1.745 overshoots because its lower neighbour −1.755 is itself high: its μ is ~2.3e-4 above the
  smooth curve, and its own ⟨m⟩ at L=128 is +0.35. The midpoint interpolation inherits about half
  that (~1e-4). At ~0.07 ⟨m⟩ per 1e-4 near ε ≈ −1.74 and L=128, that predicts +0.08 to +0.14,
  as observed. −1.74 takes only a quarter weight from −1.755, which predicts +0.04 to +0.06,
  also as observed.
- The control matches the original within error. Run-to-run variation is small, so the changes
  above come from μ.
