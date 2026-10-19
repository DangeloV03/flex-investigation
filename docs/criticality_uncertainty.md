# Bimodality criticality: what the errors mean

This note explains the two uncertainty columns in `criticality.csv` and what
the shaded region on `bc_max_phase_diagram_*.png` shows.

## The curve

For each ε we plot **max Sarle BC** over the μ-sweep (how bimodal the column
order-parameter distribution is). Below ε_c the slab is two-phase (BC high);
above ε_c it homogenizes (BC lower).

On a **sharp** transition the curve looks like a steep step. On your data it is
a **long, flat, noisy plateau** that drifts down slowly — so ε_c is not pinned
to one grid point.

## Two kinds of error

### 1. `BC_err` (vertical bars on the plot)

Bootstrap over **4 replicas** at fixed ε, after picking the winning μ dir.
This is “how much does BC wiggle if I resample my snapshots?” It does **not**
capture flatness of the whole crossover.

### 2. `fit_uncertainty` (old default)

From a **global sigmoid fit** to the entire curve (inflection point covariance).
Works when the curve is a clean S-shape. On a flat plateau it is often **too
small** (e.g. ±0.008 when the visible transition spans ±0.07).

## Transition bracket (recommended)

We mark the crossover using two BC levels:

| Level | Meaning |
|-------|---------|
| **BC = 0.85** | Still clearly two-phase (high plateau) |
| **BC = 0.65** | Transition well underway |

Scanning left → right on the plot (βε increasing):

1. Find βε where BC **crosses 0.85** → `transition_x_high` (left edge)
2. Find βε where BC **crosses 0.65** → `transition_x_low` (right edge)

The **orange vertical band** on the plot is the interval
[`transition_x_high`, `transition_x_low`].

**Half-width:**

```
transition_half_width = (transition_x_low - transition_x_high) / 2
```

Quote: **ε_c ≈ −1.70 ± 0.07** (using the sigmoid point estimate and this
half-width).

### `recommended_uncertainty`

Uses **`transition_half_width`** (the ± shown on the plot). Falls back to the
envelope or sigmoid fit only if the bracket cannot be computed.

### `transition_half_width_envelope` (optional, conservative)

Same crossings with **BC ± BC_err** on each segment; can be much wider on a
flat curve. Check this if you want a worst-case bound; usually quote
`recommended_uncertainty` instead.

## How to read the plot

- **Blue points + vertical bars:** measured max BC ± replica bootstrap
- **Orange dashed horizontals:** BC = 0.85 and 0.65 contour levels
- **Yellow horizontal band:** BC values between those contours
- **Orange vertical band:** βε range where BC falls from 0.85 → 0.65
- **Red vertical line:** sigmoid inflection (point estimate)
- **± annotation:** `transition_half_width`

If the red line sits inside the orange band but the band is wide, the message
is: “we have a best guess, but the data only localizes ε_c to this whole
region.”

## Re-running

```bash
python -u criticality/bimodality.py phase-diagram \
  --base-dir susceptibility_results/coex_homo_dmu1p0 \
  --scheme homo --delta-f 0.0 --k 1.0 \
  --Lx 160 --Ly 16 --delta-mus 1.0 \
  --manage-csv coex_manage_homo_dmu1p0.csv \
  --out-dir criticality/homo_dmu1p0
```

Outputs: updated `criticality.csv`, `bc_vs_beta_epsilon.csv`, and
`bc_max_phase_diagram_160x16.png` with the bracket drawn.
