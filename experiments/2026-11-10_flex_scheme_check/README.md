# FLEX vs driving scheme (mentor item 4) + ε_c shift napkin model (item 2), 2026-11-10

## Run on Della (from the repo root, `experiments` branch)

```bash
bash experiments/2026-11-10_flex_scheme_check/run.sh          # item 4 on current COEX_RUNS_* -> FLEX_SCHEME_CHECK/
bash experiments/2026-11-10_flex_scheme_check/run.sh eps_c    # item 2 prediction table
```

`preview_2026-10-16_snapshot/` is the same analysis run locally on the 2026-10-16 coex snapshot
(`output/data/coex_snapshot_2026-10-16`).

## Model (`single_site.py`)

One site, with its n bonding neighbours held fixed. The rates are copied exactly from lattice-gas
`hetero_open_drive_chain.rs`:
- I→B: k_n
- B→I: k_n e^{Δf + nε + Δμ}, so **Δμ > 0 drives B→I** (deactivation)
- k_n = k·min(1, e^{∓nε − Δf − Δμ}) for negative/positive drive

The quantity used throughout is h(n) = ln P_B/(P_E+P_I).

## Item 4: was the right FLEX curve used?

**Yes for both drives.** The plotted FLEX curve already uses the scheme-specific k conditioned on two
neighbours, k_eff = min(1, e^{∓2ε−Δf−Δμ}). It equals the single-site h(2)=0 condition to machine
precision for all 12 campaigns (`pipeline_check.csv`).

**Why the negative-drive curve looks like the homogeneous one.** At Δf = 0, k = 1,
k(n=2) = min(1, e^{2|ε|−Δμ}) = 1 whenever Δμ < 2|ε| ≈ 3.4. So for C3A, C3B and C3C, and for 97% of the
C3D/C3E ε range, the n=2-conditioned FLEX is *identical* to the homogeneous-drive curve. For positive
drive it is not (k(n=2) ≪ 1, and the homogeneous curve is off by 0.2–7.5 in μ).

**The weakness is conditioning on exactly n = 2.** Sites with 0, 1 or 3 neighbours feel the drive
even when the n=2 rate is saturated. Averaging the single-site model over n ~ Binomial(4, ½) cuts the
residuals substantially (RMS of μ_sim − μ_pred, all Ly pooled, from `rms_summary.csv`):

| campaign | homo k | FLEX n=2 (current) | ⟨P_B⟩ₙ = ½ | ⟨h⟩ₙ = 0 |
|---|---|---|---|---|
| C3A neg +1 | 0.006 | 0.006 | **0.002** | 0.005 |
| C3C neg +2 | 0.07 | 0.07 | **0.02** | 0.02 |
| C3D neg +3 | 0.29 | 0.29 | 0.07 | **0.05** |
| C3E neg +4 | 0.57 | 0.57 | 0.13 | **0.11** |
| D3B pos −1 | 0.16 | 0.027 | 0.008 | **0.0015** |
| D3A pos +1 | 0.46 | 0.029 | 0.011 | **0.005** |
| D3C pos +2 | 1.4 | 0.048 | 0.018 | **0.009** |
| D3D pos +3 | 3.2 | 0.057 | 0.022 | **0.010** |
| D3E pos +4 | – | 0.061 | 0.024 | **0.011** |

The homogeneous-drive campaigns (C2A/C2B) and C3B (≡ homogeneous) are unaffected, at about 5e-4.

## Item 2: which scheme moves ε_c to the right?

Effective coupling: ε_eff = −(h(3) − h(1))/2 at coexistence. The equilibrium system gives
ε_eff = ε, and the predicted ε_c is where ε_eff reaches −1.7627 (`eps_c_shift.py`, `eps_c_predictions.csv`).

- **Homogeneous drive:** ε_eff = ε, so no shift at any Δμ.
- **Negative drive, Δμ > 0:** the coupling gets weaker, so ε_c moves *left*. Predictions: +2 → −1.79, +3 → −1.97, +4 → −2.22.
- **Positive drive, Δμ > 0:** the coupling gets stronger, so ε_c moves *right*. At k = 1: +1 → −1.715, +2 → −1.693, +3 → −1.683, +4 → −1.680, flattening out by +4.
- **Rule, in the simulator's sign convention:** the drive must deactivate in dilute regions, i.e. k_dilute > k_dense with Δμ > 0. The mentor's pairing has the opposite Δμ sign, which is presumably a difference in sign convention.
- **Big lever, the base rate k:** positive drive at k = 10 gives ε_c ≈ −1.57 / −1.39 / −1.26 / −1.19 for Δμ = +1…+4. k = 100 gives about the same; k = 0.1 gives almost nothing.

**Check against local BC data** (`bc_crossings_local.csv`: BC = 5/9 sigmoid crossing; shifts relative
to Δμ = +1 at the same Ly, which cancels most of the slab finite-size offset):

| | measured (Ly 16 / 20 / 40) | predicted |
|---|---|---|
| pos −1 | −0.133 / −0.121 / −0.129 | −0.125 |
| pos +2 | +0.033 / +0.033 / +0.040 | +0.023 |
| pos +3 | +0.053 / +0.056 / +0.056 | +0.032 |
| pos +4 | +0.057 / +0.058 / +0.049 | +0.036 |
| neg +2 | −0.062 (Ly16) | −0.026 |
| neg +3 | −0.255 (Ly16) | −0.202 |

The sign is right everywhere; the magnitude is under-predicted by about 1.5×. At Ly = 16, positive drive with
Δμ ≥ +2 already sits about 0.05 to the right of the equilibrium run (−1.579) and the homogeneous Δμ=+1 run (−1.570).
The local Ly=40 windows mostly don't bracket the crossing, so those values are extrapolated.

**Suggested test:** a positive-drive, k = 10 Ly=16 coex scout with BC at Δμ = +2 and +4 (predicted
ε_c ≈ −1.39, −1.19).
