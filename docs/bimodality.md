# Task: Build a bimodality-coefficient pipeline to locate ε_c from existing μ_coex snapshot data

## Context / where this fits

We already have a working μ_coex calculation pipeline in this repo
(`generate_test_samples.py`, `run_all.py`, `json_runner.py`, `analyzer.py`)
that produces and stores snapshot `.npy` files for a sweep of ε values, at
three system sizes: (L_short, L_long) = (8, 80), (16, 160), (32, 320). Each
snapshot is a 2D occupation array of shape `(L_short, L_long)` — L_short is
the "slab width" direction, L_long is the "slab length" direction along which
a liquid-gas interface can form.

This task adds a **new, separate analysis stage** that consumes those
already-saved snapshot `.npy` files (do not modify the simulation/production
code) and locates the critical epsilon, ε_c, at each system size, using a
statistical measure of how bimodal the column-density distribution is.

Before writing any code: **inspect the existing snapshot storage layout**
(directory structure, file naming, chunk size, and — importantly — whether
occupation values are stored as {0,1} or as ±1) and the existing CSV
tracking system used elsewhere in the pipeline, and follow those same
conventions for this new stage rather than inventing new ones. If the
occupation convention isn't obvious from the code/data, ask before assuming.

---

## The physics: why column densities go bimodal → unimodal

This is a lattice gas (or lattice-spin) system on a slab geometry, simulated
in the grand canonical (μ-V-T) ensemble, with a heterogeneous nonequilibrium
drive parameterized by ε. Below the critical drive strength, the system
phase-separates into a coexisting liquid region and a gas region within the
slab, with a single interface between them running roughly along the short
(L_short) direction. As ε increases toward its critical value ε_c, the
interface becomes rougher and eventually delocalizes; above ε_c, the system
is in a single homogeneous phase with no persistent interface.

We want to *detect this transition directly from configuration snapshots*,
independent of the μ_coex calculation, by looking at the shape of the
distribution of local density along the slab's long axis.

**Column density.** For a snapshot of shape `(L_short, L_long)`, define the
density of column x (x = 0 … L_long-1) as the average occupation over the
short axis:

```
rho_col[x] = mean over y of snapshot[y, x]
```

This gives one number per column, i.e. an array of length L_long per
snapshot.

**Why this is bimodal below ε_c.** When the system is phase-separated, most
columns sit entirely within the liquid region (rho_col near the liquid
density) or entirely within the gas region (rho_col near the gas density);
only the handful of columns actually at the interface have intermediate
values. So the *distribution* of rho_col values, pooled over many columns and
many snapshots, has two peaks — one near the liquid density, one near the gas
density — with a sparse in-between region. As ε → ε_c, the interface widens
and roughens, more columns take on intermediate values, and the two peaks
broaden and move toward each other. Above ε_c there's no persistent phase
separation at all, and the distribution of rho_col collapses into a single
peak (unimodal, roughly Gaussian, centered on the single bulk density).

So: **bimodal P(rho_col) ⟺ below ε_c (phase separated). Unimodal P(rho_col) ⟺
at/above ε_c (single phase).** Tracking a bimodality measure across the ε
sweep and finding where it crosses over from "bimodal" to "unimodal" gives an
independent estimate of ε_c, which can be compared against the μ_coex-based
estimate.

---

## Step 1 — Column density extraction

For each ε in the existing sweep, and for a chunk of stored snapshots of
shape `(n_snapshots, L_short, L_long)`, compute the column-density array:

```
rho_col[t, x] = mean over y of snapshot[t, y, x]
```

producing an array of shape `(n_snapshots, L_long)` per (ε, L) chunk.

Cache these to disk (mirroring the existing `.npy` caching convention used
elsewhere in the pipeline) so this reduction doesn't have to be repeated on
every analysis run. Store alongside them: ε, L_short, L_long, n_snapshots in
the chunk, and the source snapshot file path for provenance.

**Important — confirm the value convention before implementing this.** If
occupation is stored as ±1 rather than {0,1}, `mean over y` still works
mechanically but the resulting "column density" is really a column
magnetization ranging over [-1, 1] rather than [0, 1] — the math below is
identical either way (it's convention-agnostic), but name the quantity and
its axes consistently with whichever convention the codebase already uses so
plots and downstream code aren't confusing.

---

## Step 2 — Pooling (same ε only, never across ε)

For a given ε, pool the column-density array across **both** snapshots and
columns into one flat 1D array of length `n_snapshots × L_long`. This flat
array is the empirical sample of P(rho_col) at that ε.

This is legitimate because every snapshot at fixed ε is a valid equilibrium
sample at the same ensemble point, and by translation invariance along the
long axis, every column shares the same marginal single-column distribution
(they're correlated with each other, not independent, but that doesn't bias
the pooled histogram's shape — see the note on the dip test below for where
the correlation *does* matter).

**Do not pool across different ε values.** This mirrors a pooling artifact
already identified elsewhere in this project (pooling snapshots across
different μ produces a composite histogram that isn't a real thermodynamic
distribution) — the same failure mode applies here if columns/snapshots from
different ε get mixed into one histogram.

Output per ε: one 1D array `pooled_rho_col`, plus its length `n_pooled` and
`L_long` (needed later since `n_pooled` conflates true independent sample
size with the column count).

---

## Step 3 — Sarle's bimodality coefficient (primary metric)

For each ε's pooled array, compute:

- `mean`, `std`
- sample skewness γ:  `γ = (1/n) Σ [(x_i - mean)/std]^3`
- sample excess kurtosis κ:  `κ = (1/n) Σ [(x_i - mean)/std]^4  - 3`
- finite-sample bias correction:  `corr(n) = 3(n-1)² / [(n-2)(n-3)]`
- **bimodality coefficient:**

  ```
  BC = (γ² + 1) / (κ + corr(n))
  ```

**Interpretation for reference (include this understanding in
docstrings/comments, it matters for correctness review):**
- A perfect unimodal Gaussian gives BC → 1/3 as n → ∞.
- A maximally bimodal symmetric distribution (two well-separated point masses)
  gives BC → 1 as n → ∞.
- The commonly cited heuristic cutoff BC > 5/9 ≈ 0.556 ("likely bimodal") is a
  convention, not a hypothesis test — treat it as a rough guide, not a hard
  threshold, when writing the crossover-detection logic in Step 5.
- Use **excess kurtosis** (Gaussian → 0) consistently with the correction term
  as written above; if you use a library function that returns raw kurtosis
  (Gaussian → 3) instead, either subtract 3 or adjust the formula — be
  explicit in code about which convention is in use, since mixing them
  silently gives wrong BC values.
- Given the sample sizes here (thousands of snapshots × tens-to-hundreds of
  columns → n in the many thousands to hundreds of thousands), the finite-
  sample correction term will sit very close to its asymptotic value of 3 and
  will have negligible effect — implement it anyway for correctness, but
  don't expect it to visibly change results at these system sizes.

Output: for every (ε, L), one row with:
`{epsilon, L_short, L_long, n_pooled, mean, std, skew, kurtosis_excess, BC}`

Save this as a CSV/tracking-file entry per the existing tracking system
convention in the repo (append-as-you-go, not held in memory for the whole
sweep) — this table is the main deliverable of this task and should persist
independent of everything else.

---

## Step 4 — Cross-checks near the candidate crossover only

These are more expensive and correlation-sensitive, so only run them for the
window of ε values where Step 3's BC(ε) curve is actively transitioning (not
for the full sweep).

**4a. Two-component Gaussian mixture fit.** Fit both a 1-component and a
2-component Gaussian mixture (EM) to the pooled array. Record component
weights, means, sigmas, and BIC for each fit, plus ΔBIC = BIC(1-component) −
BIC(2-component). A large positive ΔBIC favors the 2-component (bimodal)
description; as ε → ε_c this should shrink toward zero or go negative.

**4b. Hartigan's dip test.** This one needs a decorrelated subsample, not the
raw pooled array, because adjacent columns are strongly correlated (the
interface moves as a single connected object across the whole slab) — feeding
the raw pooled array in would make the dip test's p-value falsely confident.
Subsample by taking every k-th column per snapshot; until a proper
correlation-length estimate exists elsewhere in the pipeline, use a
conservative placeholder of k = L_short and flag this explicitly as
provisional in the output (e.g. a `stride_provisional: true` field), so it's
easy to find and update later once a real correlation length is available.

Save both cross-check results to their own tracking entries, tagged with
which ε values they were actually run for (not the full sweep).

---

## Step 5 — Locate ε_c per system size

Using the BC(ε) table from Step 3 for a single L, find where BC transitions
from its bimodal plateau down toward the unimodal asymptote (~1/3). Prefer
fitting a smooth monotonic crossover function (e.g. a sigmoid) to BC(ε) and
taking its inflection point as ε_c(L); fall back to a simple threshold
crossing (e.g. where BC crosses 5/9) if a sigmoid fit is impractical given
the sweep's ε resolution.

Output one row per L: `{L_short, L_long, epsilon_c_estimate, method,
fit_uncertainty}`, saved to its own persistent table — this feeds the
downstream finite-size-scaling work (χ vs ε, max χ vs L) already in progress
elsewhere in the project, so it should be a stable, easily-reloadable
artifact.

---

## Deliverables checklist

1. A function/module that goes from a stored snapshot chunk (ε, L) → cached
   column-density array (Step 1).
2. A pooling function (Step 2) that pools only within a single ε.
3. A Sarle's-BC function (Step 3) operating on a pooled 1D array, returning
   the full stats dict above — written and tested against at least one
   synthetic case by hand (e.g. a symmetric two-Gaussian mixture with known
   separation, and a single Gaussian) to confirm it reproduces the expected
   BC → 1/3 and BC → 1 limits before trusting it on real data.
4. A sweep driver that runs Steps 1–3 across the existing ε sweep for all
   three system sizes and writes the BC-vs-ε table.
5. Cross-check functions for Step 4 (GMM ΔBIC, dip test with documented
   stride), runnable on a specified subset of ε values near a candidate
   crossover.
6. A crossover-detection function (Step 5) that takes a single L's BC-vs-ε
   table and returns an ε_c estimate with uncertainty.
7. Basic plotting: BC vs ε per L (to sanity check by eye), and P(rho_col)
   histograms for a couple of representative ε values (well below, near, and
   well above the candidate ε_c) so the bimodal → unimodal shape change can
   be visually confirmed, not just inferred from the scalar BC.

Follow the existing repo conventions for CSV tracking, caching with
provenance metadata, and separation of concerns (extraction / pooling /
metric computation / sweep driving / crossover detection should be separate,
testable functions, not one monolithic script) already established elsewhere
in this codebase.