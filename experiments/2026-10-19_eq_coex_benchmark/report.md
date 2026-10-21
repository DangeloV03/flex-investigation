# Equilibrium coex benchmark (2026-10-19)

## Why
The mentor flagged ⟨m⟩ ≠ 0 in the new susc runs. `scan_m_bias.py` (latest generation, z = ⟨m⟩ / SE over
replicas; with ⟨m⟩ = 0 only ~0.3% of points should have |z| > 3):

| campaign | μ source | \|z\|>3 |
|---|---|---|
| S1A | exact 2ε (Ising) | 2% |
| S1B_IMPROVED_MU | exact, `--mu-exact` | 3% |
| S2A | fitted slab coex | 46% |
| POS_DMU1 / 2 / 3 | fitted D3x ly40 | 59 / 54 / 55% |
| NEG_DMU1 / 2 / 3 | fitted C3x ly40 | 40 / 55 / 79% |

- The exact-μ campaigns are clean. Every campaign that uses fitted coex μ is off coexistence.
- ⟨m⟩ follows the ε-to-ε jitter of the fitted μ around a smooth curve: r ≈ 0.6–0.9 at L = 64. That
  jitter is only 2–5×10⁻⁴, but χ grows with L and amplifies it.
- NEG_DMU2/3 also have a systematic offset at small L (NEG_DMU3 L=16: ⟨m⟩ ≈ +0.17).

## Test
Run the same slab coex protocol where the answer is exact. That means equilibrium: homo, βΔf = 0,
k = 1, βΔμ = 0, where μ_coex = −ln(e^{2|ε|} − 1).
- Ly ∈ {16, 20, 40}, Lx = 10·Ly, ε ∈ [−1.9, −1.5] step 0.02, 10 μ points ±0.05 around exact.
- Della: `/scratch/gpfs/WJACOBS/vd7294/flex-experiments/COEX_RUNS_EQ_DMU0/`, launcher
  `run_eq_coex.sh`, tmux `coex-EQ` on della9. Launched 2026-10-19 23:51 EDT.
- Compare: `python compare_eq_coex.py --root <COEX_RUNS_EQ_DMU0> --out <dir>`. It compares each
  Ly and an FSS extrapolation (1/Ly or 1/Ly², whichever has the lower χ²) against exact.

## Results
_pending: jobs held by the Della maintenance window, 2026-10-20 05:30–18:00._

## Existing data: July Ising-limit coex (`coex/coex_eq`, βΔf = −20, k = 0, Δμ = 0, exact μ = 2ε)
41 ε in [−1.8, −1.6] at Ly 16/20/40. 7 of 123 fits failed (error > 5e-3) and are dropped.

| set | mean bias | RMS error | median σ_fit | within 2σ |
|---|---|---|---|---|
| Ly = 16 | −6.6e-05 | 5.5e-04 | 2.8e-04 | 76% |
| Ly = 20 | +3.3e-05 | 4.0e-04 | 2.9e-04 | 72% |
| Ly = 40 | −1.7e-05 | 2.4e-04 | 1.4e-04 | 74% |
| FSS 1/Ly | −2.0e-04 | 8.7e-04 | 3.2e-04 | 78% |
| FSS 1/Ly² | −1.2e-04 | 4.9e-04 | 2.0e-04 | 78% |

- No bias at any Ly. That's expected: the Ising limit is symmetric, so slab width can't shift μ.
- Scatter shrinks with Ly; Ly = 40 is best. The FSS extrapolation is worse than Ly = 40 alone,
  because with no bias to remove it only amplifies noise.
- Reported fit errors are ~1.5× too small (≈75% within 2σ, not 95%).
- Even Ly = 40 scatters by ~2.4e-4, ~20× the ~1e-5 tolerance from the susc runs.
- This case can't show an asymmetry-driven bias. The βΔf = 0 run can: Ly = 16 there is biased by
  −3e-4 (16/19 ε below exact).
