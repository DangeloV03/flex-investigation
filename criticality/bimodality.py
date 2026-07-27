"""
bimodality.py

Locate the critical drive strength epsilon_c of a coex run *independently* of the
mu_coex calculation, by measuring how bimodal the column-order-parameter
distribution of the stored configuration snapshots is.

Physics (see criticality/bimodality.md):
  Below epsilon_c the slab phase-separates into a liquid region and a gas region
  with a single interface, so most columns sit at the liquid or the gas value and
  P(phi_col) is BIMODAL. As epsilon -> epsilon_c the interface roughens and
  delocalizes; above epsilon_c there is a single homogeneous phase and P(phi_col)
  is UNIMODAL. Tracking a bimodality measure across the epsilon sweep and finding
  where it crosses over from bimodal to unimodal gives epsilon_c.

Data reality (differs from the generic spec; see json_runner.py / combo_paths.py):
  * Snapshots are the *final* lattice per replica: results/<combo>/mu_sweeps/
    mu<tag>/final_lattice_<id>.npy  (~4-8 per mu dir, uint32).
  * Shape is (Lx, Ly) = (long, short). The slab interface runs along Lx (axis 0),
    so L_long = Lx and L_short = Ly. The column reduction is a mean over the SHORT
    axis (axis=1), giving one value per long-axis column x = 0 .. Lx-1.
  * Three states {EMPTY=0, INERT=1, BONDING=2}. We reduce each column to the phi
    order parameter phi_col = rho_bonding - rho_inert - rho_empty, i.e. the spin
    map BONDING->+1, {INERT,EMPTY}->-1 averaged over the short axis (range [-1,1]),
    matching analyzer.py's coexistence order parameter phi = 2*rho_bonding - 1.
  * Each epsilon has a whole mu-sweep; phase separation only exists at
    coexistence, so per epsilon we pool the snapshots from the mu dir nearest
    mu_coex.

Separation of concerns (each function is independently testable):
  extraction   : column_op, extract_column_op, cache_column_op
  mu selection : resolve_mu_coex, nearest_mu_dir
  pooling      : pool_column_op          (single epsilon only, never across epsilon)
  metric       : sarle_bc
  sweep driver : sweep_bc
  crossover    : locate_epsilon_c
  orchestrator : find_criticality
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
import time
from typing import Optional

import numpy as np

# --- make sibling source folders importable (mirrors conftest.py / env.sh) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _sub in ("coex", "susceptibility"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from combo_paths import combo_dir_name, mu_dir_name, param_tag, size_tag  # noqa: E402

# Lattice state encoding (from coex/json_runner.py).
EMPTY, INERT, BONDING = 0, 1, 2

# Sarle heuristic cutoff: BC > 5/9 ~ "likely bimodal" (a convention, not a test).
BC_BIMODAL_CUTOFF = 5.0 / 9.0

# Transition-bracket defaults: where BC has fallen partway down the crossover.
# Used to quote an honest uncertainty when the sigmoid inflection sits on a
# flat, noisy plateau (see criticality/UNCERTAINTY.md).
TRANSITION_BC_HIGH = 0.85
TRANSITION_BC_LOW = 0.65

# ---------------------------------------------------------------------------
# tag helpers
# ---------------------------------------------------------------------------

def untag(tag: str) -> float:
    """Inverse of combo_paths.param_tag: 'm2p0' -> -2.0, '0p5' -> 0.5."""
    neg = tag.startswith("m")
    body = tag[1:] if neg else tag
    value = float(body.replace("p", "."))
    return -value if neg else value


# ---------------------------------------------------------------------------
# Step 1 - column-order-parameter extraction
# ---------------------------------------------------------------------------

def column_op(snapshot: np.ndarray) -> np.ndarray:
    """Per-column phi order parameter for one (Lx, Ly) snapshot.

    Maps BONDING(=2) -> +1 and {INERT(=1), EMPTY(=0)} -> -1, then averages over
    the short axis (axis=1). Returns a length-Lx array in [-1, 1]:
    phi_col[x] = rho_bonding(x) - rho_inert(x) - rho_empty(x).
    """
    spin = np.where(snapshot == BONDING, 1.0, -1.0)
    return spin.mean(axis=1)


def extract_column_op(mu_dir: str) -> tuple[np.ndarray, dict]:
    """Load every final_lattice_*.npy in a mu dir and stack their column ops.

    Returns (arr, meta) where arr has shape (n_snapshots, Lx) and meta records
    provenance: n_snapshots, Lx, Ly, and the source file paths.
    """
    paths = sorted(glob.glob(os.path.join(mu_dir, "final_lattice_*.npy")))
    if not paths:
        raise FileNotFoundError(f"no final_lattice_*.npy in {mu_dir}")
    cols = []
    Lx = Ly = None
    for p in paths:
        snap = np.load(p)
        if snap.ndim != 2:
            raise ValueError(f"{p}: expected 2D (Lx, Ly), got shape {snap.shape}")
        if Lx is None:
            Lx, Ly = int(snap.shape[0]), int(snap.shape[1])
        cols.append(column_op(snap))
    arr = np.asarray(cols, dtype=float)  # (n_snapshots, Lx)
    meta = {
        "n_snapshots": len(paths),
        "Lx": Lx,
        "Ly": Ly,
        "source_paths": paths,
    }
    return arr, meta


def cache_column_op(
    mu_dir: str,
    cache_dir: str,
    *,
    epsilon: float,
    mu_used: float,
    mu_coex: float,
    combo_name: str,
) -> tuple[np.ndarray, dict]:
    """extract_column_op with an .npy cache + .json provenance sidecar.

    Cache key: <combo_name>__mu<tag>. Reuses the cache when it is newer than
    every source snapshot; otherwise recomputes and rewrites.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{combo_name}__{mu_dir_name(mu_used)}"
    npy_path = os.path.join(cache_dir, key + ".npy")
    json_path = os.path.join(cache_dir, key + ".json")

    sources = sorted(glob.glob(os.path.join(mu_dir, "final_lattice_*.npy")))
    if os.path.isfile(npy_path) and os.path.isfile(json_path) and sources:
        cache_mtime = os.path.getmtime(npy_path)
        if all(os.path.getmtime(s) <= cache_mtime for s in sources):
            arr = np.load(npy_path)
            with open(json_path) as f:
                meta = json.load(f)
            return arr, meta

    arr, meta = extract_column_op(mu_dir)
    meta.update(
        {
            "epsilon": epsilon,
            "mu_used": mu_used,
            "mu_coex": mu_coex,
            "combo_name": combo_name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    np.save(npy_path, arr)
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    return arr, meta


# ---------------------------------------------------------------------------
# mu selection - the mu-sweep gives one ensemble point per mu; we reduce over it
# ---------------------------------------------------------------------------

def _read_output_field(mu_dir: str, field: str) -> Optional[float]:
    """First-row value of `field` from a mu dir's output.csv (e.g. signed mu,
    beta). The dir name only encodes abs(mu), so the signed value must be read
    from the CSV."""
    path = os.path.join(mu_dir, "output.csv")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                return float(row[field])
    except (KeyError, ValueError, StopIteration):
        return None
    return None


def enumerate_mu_dirs(combo_dir: str) -> list[tuple[str, float]]:
    """All (mu_dir, signed_mu) under combo_dir/mu_sweeps with a readable mu."""
    out = []
    for d in sorted(glob.glob(os.path.join(combo_dir, "mu_sweeps", "mu*"))):
        if not os.path.isdir(d):
            continue
        mu = _read_output_field(d, "mu")
        if mu is not None:
            out.append((d, mu))
    return out


def nearest_mu_dir(combo_dir: str, mu_coex: float) -> tuple[str, float]:
    """Return (mu_dir, mu) whose signed mu is closest to mu_coex."""
    candidates = enumerate_mu_dirs(combo_dir)
    if not candidates:
        raise FileNotFoundError(
            f"no mu dir with a readable output.csv under {combo_dir}/mu_sweeps"
        )
    return min(candidates, key=lambda dm: abs(dm[1] - mu_coex))


def _mu_coex_from_phi_psi(combo_dir: str) -> Optional[float]:
    """Interpolate the mu where phi crosses zero from phi_psi.csv."""
    path = os.path.join(combo_dir, "phi_psi.csv")
    if not os.path.isfile(path):
        return None
    mus, phis = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                mus.append(float(row["mu"]))
                phis.append(float(row["phi"]))
            except (KeyError, ValueError):
                continue
    if len(mus) < 2:
        return None
    order = np.argsort(mus)
    mus = np.asarray(mus)[order]
    phis = np.asarray(phis)[order]
    for i in range(len(mus) - 1):
        if phis[i] == 0.0:
            return float(mus[i])
        if phis[i] * phis[i + 1] < 0:  # sign change -> linear interp to phi=0
            t = phis[i] / (phis[i] - phis[i + 1])
            return float(mus[i] + t * (mus[i + 1] - mus[i]))
    return float(mus[int(np.argmin(np.abs(phis)))])  # else nearest to zero


def resolve_mu_coex(
    epsilon: float,
    combo_dir: str,
    *,
    manage_csv: Optional[str] = None,
    combo_params: Optional[dict] = None,
) -> float:
    """mu_coex for one epsilon: manage.csv mu_coex_FITTED, else phi_psi.csv, else 2*eps."""
    if manage_csv and os.path.isfile(manage_csv):
        from sweep_susceptibility import load_mu_map

        cp = combo_params or {}
        mu_map = load_mu_map(
            manage_csv,
            delta_f=cp.get("delta_f"),
            delta_mu=cp.get("delta_mu"),
            k=cp.get("k"),
            scheme=cp.get("scheme"),
        )
        mu = mu_map.get(round(float(epsilon), 6))
        if mu is not None:
            return float(mu)
    mu = _mu_coex_from_phi_psi(combo_dir)
    if mu is not None:
        return mu
    return 2.0 * float(epsilon)


# ---------------------------------------------------------------------------
# Step 2 - pooling (single epsilon only, never across epsilon)
# ---------------------------------------------------------------------------

def pool_column_op(arr: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Flatten a single-epsilon (n_snapshots, Lx) column-op array.

    Returns (pooled_1d, n_pooled, L_long). L_long (= Lx) is returned separately
    because n_pooled conflates the true independent sample size (n_snapshots)
    with the column count (columns within a snapshot are spatially correlated).
    """
    arr = np.asarray(arr, dtype=float)
    L_long = int(arr.shape[1])
    pooled = arr.reshape(-1)
    return pooled, int(pooled.size), L_long


# ---------------------------------------------------------------------------
# Step 3 - Sarle's bimodality coefficient
# ---------------------------------------------------------------------------

def sarle_bc(pooled: np.ndarray) -> dict:
    """Sarle's bimodality coefficient of a pooled 1D sample.

    BC = (skew^2 + 1) / (kurtosis_excess + corr(n)),  corr(n)=3(n-1)^2/[(n-2)(n-3)].
    Uses sample skewness and EXCESS kurtosis (Gaussian -> 0).

    Reference limits:
      * unimodal Gaussian            -> BC -> 1/3 as n -> inf
      * two well-separated masses    -> BC -> 1   as n -> inf
      * heuristic cutoff BC > 5/9    -> "likely bimodal" (a convention, not a test)
    At the sample sizes here corr(n) sits very close to its asymptote of 3, so it
    barely moves BC; it is included for correctness.
    """
    x = np.asarray(pooled, dtype=float).ravel()
    n = x.size
    if n < 4:
        raise ValueError(f"need n >= 4 for the finite-sample correction, got {n}")
    mean = float(x.mean())
    std = float(x.std())  # population std (ddof=0)
    if std == 0.0:
        # A degenerate point mass: skew/kurtosis undefined; report a unimodal-ish BC.
        return {
            "n_pooled": n, "mean": mean, "std": 0.0,
            "skew": 0.0, "kurtosis_excess": 0.0, "BC": float("nan"),
        }
    z = (x - mean) / std
    skew = float(np.mean(z ** 3))
    kurt_excess = float(np.mean(z ** 4) - 3.0)
    corr = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    bc = (skew ** 2 + 1.0) / (kurt_excess + corr)
    return {
        "n_pooled": n,
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis_excess": kurt_excess,
        "BC": float(bc),
    }


def bc_bootstrap_error(arr: np.ndarray, *, n_boot: int = 200, seed: int = 0) -> float:
    """1-sigma bootstrap error on Sarle's BC for a (n_snapshots, Lx) column-op array.

    Resamples whole SNAPSHOTS (rows) with replacement, not individual columns:
    the independent units are the replicas, while columns within a snapshot are
    spatially correlated (the interface is one connected object), so a naive
    per-value bootstrap would understate the error. Returns the std of BC over
    n_boot resamples; NaN if there are fewer than 2 snapshots.
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[0]
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    bcs = []
    for _ in range(n_boot):
        pooled = arr[rng.integers(0, n, size=n)].reshape(-1)
        try:
            b = sarle_bc(pooled)["BC"]
        except ValueError:
            continue
        if np.isfinite(b):
            bcs.append(b)
    return float(np.std(bcs, ddof=1)) if len(bcs) > 1 else float("nan")


# ---------------------------------------------------------------------------
# CSV helper (append-as-you-go, header on first write)
# ---------------------------------------------------------------------------

def _append_row(csv_path: str, row: dict, fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


BC_FIELDS = [
    "scheme", "delta_f", "delta_mu", "k",
    "epsilon", "beta", "beta_epsilon", "L_short", "L_long", "n_pooled",
    "mean", "std", "skew", "kurtosis_excess", "BC", "BC_err",
    "mu_at_max", "mu_coex", "n_snapshots", "n_mu_scanned",
    "mu_reduction", "source_dir",
]


def _resolve_data_path(path: str, *, base_dir: str | None = None) -> str:
    """Resolve a CSV path (absolute, cwd-relative, or under base_dir/PROJECT_ROOT)."""
    if os.path.isabs(path):
        return path
    for root in (os.getcwd(), base_dir, os.environ.get("PROJECT_ROOT")):
        if not root:
            continue
        candidate = os.path.normpath(os.path.join(root, path))
        if os.path.exists(candidate):
            return candidate
    return os.path.normpath(os.path.join(os.getcwd(), path))


def _column_op_cache_key(row: dict) -> str:
    cp = {
        "scheme": row["scheme"],
        "delta_f": float(row["delta_f"]),
        "delta_mu": float(row["delta_mu"]),
        "k": float(row["k"]) if row.get("k") not in (None, "") else 1.0,
        "Lx": int(row["L_long"]),
        "Ly": int(row["L_short"]),
        "epsilon": float(row["epsilon"]),
    }
    combo_name = combo_dir_name({**cp, "epsilon": cp["epsilon"]})
    return f"{combo_name}__{mu_dir_name(float(row['mu_at_max']))}"


def _load_column_op_for_row(
    row: dict,
    cache_dir: str,
    *,
    base_dir: str | None = None,
) -> Optional[np.ndarray]:
    """Load the winning-mu column-op array for one BC CSV row."""
    key = _column_op_cache_key(row)
    npy_path = os.path.join(cache_dir, key + ".npy")
    if os.path.isfile(npy_path):
        return np.load(npy_path)

    source = row.get("source_dir")
    if source:
        mu_dir = _resolve_data_path(source, base_dir=base_dir)
        if os.path.isdir(mu_dir):
            try:
                arr, _ = extract_column_op(mu_dir)
                return arr
            except FileNotFoundError:
                pass

    if base_dir:
        cp = {
            "scheme": row["scheme"],
            "delta_f": float(row["delta_f"]),
            "delta_mu": float(row["delta_mu"]),
            "k": float(row["k"]) if row.get("k") not in (None, "") else 1.0,
            "Lx": int(row["L_long"]),
            "Ly": int(row["L_short"]),
            "epsilon": float(row["epsilon"]),
        }
        combo_dir = os.path.join(
            _resolve_data_path(base_dir),
            combo_dir_name({**cp, "epsilon": cp["epsilon"]}),
        )
        mu_dir = os.path.join(combo_dir, "mu_sweeps", mu_dir_name(float(row["mu_at_max"])))
        if os.path.isdir(mu_dir):
            try:
                arr, _ = extract_column_op(mu_dir)
                return arr
            except FileNotFoundError:
                pass
    return None


def _bc_err_is_finite(row: dict) -> bool:
    err = row.get("BC_err")
    if err in (None, ""):
        return False
    try:
        val = float(err)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(val) and val > 0)


def backfill_bc_err(
    bc_csv: str,
    *,
    cache_dir: str | None = None,
    base_dir: str | None = None,
) -> tuple[int, int, int]:
    """Fill missing BC_err from column-op cache or source snapshots.

    Returns (newly_filled, already_present, total_rows).
    Rewrites bc_csv in place when any row is updated.
    """
    with open(bc_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "BC_err" not in fieldnames:
        fieldnames.append("BC_err")
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(bc_csv)), "cache", "column_op")

    filled = already = 0
    for row in rows:
        if _bc_err_is_finite(row):
            already += 1
            continue
        arr = _load_column_op_for_row(row, cache_dir, base_dir=base_dir)
        if arr is None:
            continue
        err = bc_bootstrap_error(arr)
        if np.isfinite(err) and err > 0:
            row["BC_err"] = err
            filled += 1

    if filled:
        with open(bc_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return filled, already, len(rows)


def _bc_err_stats(rows: list[dict]) -> tuple[int, float, float]:
    vals = []
    for row in rows:
        if _bc_err_is_finite(row):
            vals.append(float(row["BC_err"]))
    if not vals:
        return 0, float("nan"), float("nan")
    return len(vals), float(min(vals)), float(max(vals))

EPS_C_FIELDS = [
    "L_short", "L_long", "x_axis", "criticality_estimate",
    "epsilon_c_estimate", "beta", "method", "fit_uncertainty",
    "transition_bc_high", "transition_bc_low",
    "transition_x_high", "transition_x_low", "transition_half_width",
    "transition_half_width_envelope", "recommended_uncertainty",
]

# Per-(Delta mu) criticality rows for the phase-diagram family.
CRIT_FIELDS = [
    "scheme", "delta_f", "delta_mu", "k", "L_short", "L_long", "x_axis",
    "criticality_estimate", "epsilon_c_estimate", "beta", "method", "fit_uncertainty",
    "transition_bc_high", "transition_bc_low",
    "transition_x_high", "transition_x_low", "transition_half_width",
    "transition_half_width_envelope", "recommended_uncertainty",
]


# ---------------------------------------------------------------------------
# Step 3 driver - sweep BC across the epsilon sweep for one (scheme, size)
# ---------------------------------------------------------------------------

def results_roots(base_dir: str) -> list[str]:
    """Directories under base_dir that directly hold combo folders.

    Handles two on-disk layouts transparently:
      * flat campaign:  base_dir holds {size}_{scheme}_..._epsilon* combos
        (the `results/` layout) -> [base_dir].
      * nested campaign: base_dir holds per-delta_mu subdirs each with their own
        `results/` (the `scheme3/dmu*/results/` layout) -> [dmu1p0/results, ...].
    So --base-dir can point at either `results` or `scheme3` and Just Work.
    """
    roots: list[str] = []
    if glob.glob(os.path.join(base_dir, "*_epsilon*")):  # combos live here directly
        roots.append(base_dir)
    if os.path.isdir(os.path.join(base_dir, "results")):  # single nested results/
        roots.append(os.path.join(base_dir, "results"))
    for r in sorted(glob.glob(os.path.join(base_dir, "*", "results"))):  # dmu*/results
        if os.path.isdir(r):
            roots.append(r)
    return list(dict.fromkeys(roots)) or [base_dir]


def discover_epsilons(base_dir: str, combo_params: dict) -> list[tuple[float, str]]:
    """Find (epsilon, combo_dir) for every epsilon present for this scheme+size.

    combo_params must contain scheme, delta_f, delta_mu, Lx, Ly (epsilon omitted).
    Searches every results root (flat or nested layout, see results_roots).
    """
    prefix = combo_dir_name({**combo_params, "epsilon": 0.0}).rsplit("epsilon", 1)[0] + "epsilon"
    found = []
    for root in results_roots(base_dir):
        for path in sorted(glob.glob(os.path.join(root, prefix + "*"))):
            if not os.path.isdir(path):
                continue
            tag = os.path.basename(path).rsplit("epsilon", 1)[1]
            try:
                eps = untag(tag)
            except ValueError:
                continue
            found.append((eps, path))
    found.sort(key=lambda t: t[0])
    return found


def _best_mu_over_sweep(
    combo_dir: str,
    cache_dir: str,
    *,
    epsilon: float,
    mu_coex: float,
    combo_name: str,
    mu_dirs: Optional[list[tuple[str, float]]] = None,
    keep_pooled: bool = False,
    selection: str = "max",
    n_boot: int = 200,
) -> Optional[dict]:
    """Scan the mu-sweep and return the stats dict for one selected mu.

    Shared by sweep_bc (BC_max curve) and pooled_at_coexistence (histograms) so
    both can pick the same mu when selection="max". selection="balanced" prefers
    the mu with the most even liquid/gas column fractions among bimodal points
    (for illustration histograms). Returns None if no mu dir yields a finite BC.
    The winner also carries a bootstrap error 'BC_err' (resampling snapshots);
    with keep_pooled=True the winning pooled array is included.
    """
    if mu_dirs is None:
        mu_dirs = enumerate_mu_dirs(combo_dir)
    candidates = []
    for mu_dir, mu in mu_dirs:
        try:
            arr, meta = cache_column_op(
                mu_dir, cache_dir,
                epsilon=epsilon, mu_used=mu, mu_coex=mu_coex, combo_name=combo_name,
            )
        except FileNotFoundError:
            continue  # this mu dir has no snapshots; skip it, keep scanning
        pooled, n_pooled, L_long = pool_column_op(arr)
        stats = sarle_bc(pooled)
        if not np.isfinite(stats["BC"]):
            continue
        frac_liq = float((pooled > 0.75).mean())
        frac_gas = float((pooled < -0.75).mean())
        cand = {
            **stats, "mu": mu, "mu_dir": mu_dir, "arr": arr,
            "n_pooled": n_pooled, "L_long": L_long, "Ly": meta["Ly"],
            "n_snapshots": meta["n_snapshots"],
            "frac_liq": frac_liq, "frac_gas": frac_gas,
            "balance": min(frac_liq, frac_gas),
        }
        if keep_pooled:
            cand["pooled"] = pooled
        candidates.append(cand)

    if not candidates:
        return None
    if selection == "balanced":
        strong = [c for c in candidates if c["BC"] >= BC_BIMODAL_CUTOFF]
        pool = strong if strong else candidates
        winner = max(pool, key=lambda c: (c["balance"], c["BC"]))
    else:
        winner = max(candidates, key=lambda c: c["BC"])
    # bootstrap error only for the winning mu (cheap: one point per epsilon)
    winner["BC_err"] = bc_bootstrap_error(winner.pop("arr"), n_boot=n_boot)
    return winner


def sweep_bc(
    base_dir: str,
    combo_params: dict,
    out_csv: str,
    cache_dir: str,
    *,
    manage_csv: Optional[str] = None,
    mu_reduction: str = "max",
) -> list[dict]:
    """Steps 1-3 across the epsilon sweep for one (scheme, size); append BC rows.

    For each epsilon the mu-sweep provides several ensemble points. We reduce the
    sweep to one bimodality number per epsilon:
      * mu_reduction="max"          -> the MAXIMUM BC over the mu-sweep (default):
        the most phase-separated ensemble point, analogous to peak-chi. This is
        the quantity for the BC_max-vs-(beta*epsilon) crossover.
      * mu_reduction="nearest_coex" -> BC at the single mu dir nearest mu_coex.

    x-axis is beta*epsilon (beta read from the winning mu dir's output.csv,
    default 1.0). Returns the per-epsilon rows (also appended to out_csv).
    """
    rows = []
    for eps, combo_dir in discover_epsilons(base_dir, combo_params):
        combo_name = os.path.basename(combo_dir)
        cp = {**combo_params, "epsilon": eps}
        mu_coex = resolve_mu_coex(eps, combo_dir, manage_csv=manage_csv, combo_params=cp)

        mu_dirs = enumerate_mu_dirs(combo_dir)
        if not mu_dirs:
            print(f"[bimodality] skip eps={eps}: no mu dirs under {combo_dir}", flush=True)
            continue
        if mu_reduction == "nearest_coex":
            mu_dirs = [min(mu_dirs, key=lambda dm: abs(dm[1] - mu_coex))]

        best = _best_mu_over_sweep(
            combo_dir, cache_dir,
            epsilon=eps, mu_coex=mu_coex, combo_name=combo_name, mu_dirs=mu_dirs,
            selection=mu_reduction if mu_reduction != "nearest_coex" else "max",
        )
        if best is None:
            print(f"[bimodality] skip eps={eps}: no finite BC over mu-sweep", flush=True)
            continue

        beta = _read_output_field(best["mu_dir"], "beta") or 1.0
        row = {
            "scheme": combo_params["scheme"],
            "delta_f": combo_params["delta_f"],
            "delta_mu": combo_params["delta_mu"],
            "k": combo_params.get("k"),
            "epsilon": eps,
            "beta": beta,
            "beta_epsilon": beta * eps,
            "L_short": int(combo_params["Ly"]),
            "L_long": best["L_long"],
            "n_pooled": best["n_pooled"],
            "mean": best["mean"],
            "std": best["std"],
            "skew": best["skew"],
            "kurtosis_excess": best["kurtosis_excess"],
            "BC": best["BC"],
            "BC_err": best.get("BC_err"),
            "mu_at_max": best["mu"],
            "mu_coex": mu_coex,
            "n_snapshots": best["n_snapshots"],
            "n_mu_scanned": len(mu_dirs),
            "mu_reduction": mu_reduction,
            "source_dir": best["mu_dir"],
        }
        _append_row(out_csv, row, BC_FIELDS)
        rows.append(row)
        print(
            f"[bimodality] eps={eps:+.4f} beta*eps={beta * eps:+.4f} "
            f"L={best['L_long']}x{combo_params['Ly']} n_mu={len(mu_dirs)} "
            f"BC_max={best['BC']:.4f} @mu={best['mu']:+.4f}",
            flush=True,
        )
    return rows


# ---------------------------------------------------------------------------
# Step 5 - locate epsilon_c
# ---------------------------------------------------------------------------

def _sigmoid(x, a, b, x_c, w):
    # a = high-x (unimodal ~1/3) asymptote, b = low-x (bimodal) plateau;
    # x_c = inflection point (the criticality estimate), w = crossover width.
    z = np.clip((x - x_c) / w, -700.0, 700.0)  # avoid exp overflow warnings
    return a + (b - a) / (1.0 + np.exp(z))


def _threshold_crossing(x, bc, level=BC_BIMODAL_CUTOFF):
    """Linear-interpolated x where BC(x) crosses `level` (descending)."""
    for i in range(len(x) - 1):
        y0, y1 = bc[i] - level, bc[i + 1] - level
        if y0 == 0:
            return float(x[i])
        if y0 * y1 < 0:
            t = y0 / (y0 - y1)
            return float(x[i] + t * (x[i + 1] - x[i]))
    return float("nan")


def _crossing_on_descending_curve(
    x: np.ndarray, bc: np.ndarray, level: float,
) -> float:
    """First x (scanning left->right) where BC(x) crosses `level` downward."""
    return _threshold_crossing(x, bc, level)


def _crossing_envelope(
    x: np.ndarray,
    bc: np.ndarray,
    bc_err: np.ndarray,
    level: float,
) -> tuple[float, float]:
    """Min/max crossing x at `level` using BC +/- BC_err on segment endpoints."""
    crossings: list[float] = []
    for i in range(len(x) - 1):
        e0, e1 = float(x[i]), float(x[i + 1])
        b0n, b1n = float(bc[i]), float(bc[i + 1])
        s0 = float(bc_err[i]) if np.isfinite(bc_err[i]) else 0.0
        s1 = float(bc_err[i + 1]) if np.isfinite(bc_err[i + 1]) else 0.0
        for ds0, ds1 in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            b0, b1 = b0n + ds0 * s0, b1n + ds1 * s1
            y0, y1 = b0 - level, b1 - level
            if y0 == 0:
                crossings.append(e0)
            elif y0 * y1 < 0:
                t = y0 / (y0 - y1)
                crossings.append(e0 + t * (e1 - e0))
    if crossings:
        return min(crossings), max(crossings)
    c = _crossing_on_descending_curve(x, bc, level)
    return c, c


def transition_bracket(
    x: np.ndarray,
    bc: np.ndarray,
    bc_err: Optional[np.ndarray] = None,
    *,
    bc_high: float = TRANSITION_BC_HIGH,
    bc_low: float = TRANSITION_BC_LOW,
) -> dict:
    """Bracket a gradual BC crossover using two contour levels.

    On a flat, noisy curve the sigmoid inflection is poorly constrained. This
    reports the beta*epsilon (or epsilon) interval where BC falls from bc_high
    to bc_low, plus a wider envelope when BC_err is available.

    Returns NaNs when a contour is never crossed on the supplied sweep.
    """
    x = np.asarray(x, dtype=float)
    bc = np.asarray(bc, dtype=float)
    order = np.argsort(x)
    x, bc = x[order], bc[order]
    if bc_err is not None:
        bc_err = np.asarray(bc_err, dtype=float)[order]

    x_high = _crossing_on_descending_curve(x, bc, bc_high)
    x_low = _crossing_on_descending_curve(x, bc, bc_low)
    half = float("nan")
    if np.isfinite(x_high) and np.isfinite(x_low) and x_low > x_high:
        half = (x_low - x_high) / 2.0

    half_env = float("nan")
    if bc_err is not None and np.all(np.isfinite(bc_err) & (bc_err > 0)):
        hi_lo, hi_hi = _crossing_envelope(x, bc, bc_err, bc_high)
        lo_lo, lo_hi = _crossing_envelope(x, bc, bc_err, bc_low)
        if all(np.isfinite(v) for v in (hi_lo, lo_hi)) and lo_hi > hi_lo:
            half_env = (lo_hi - hi_lo) / 2.0

    recommended = half_env if np.isfinite(half_env) else half
    return {
        "transition_bc_high": bc_high,
        "transition_bc_low": bc_low,
        "transition_x_high": x_high,
        "transition_x_low": x_low,
        "transition_half_width": half,
        "transition_half_width_envelope": half_env,
        "recommended_uncertainty": recommended,
    }


def locate_epsilon_c(rows: list[dict], L_long: int, *, x_col: str = "beta_epsilon") -> dict:
    """Estimate the criticality for one size from its BC_max(x) crossover.

    `x_col` is the sweep axis (default "beta_epsilon"; use "epsilon" for BC vs eps).
    Prefers a sigmoid fit -> criticality = inflection point (uncertainty from the
    fit covariance); falls back to the BC=5/9 threshold crossing if the fit fails.
    Also reports epsilon_c_estimate = criticality / mean(beta) so downstream FSS
    work has epsilon_c even when the x-axis is beta*epsilon.
    `rows` is a list of BC rows (from sweep_bc or read from the BC CSV).
    """
    sel = [r for r in rows
           if int(float(r["L_long"])) == int(L_long) and np.isfinite(float(r["BC"]))]

    def _err(r):
        try:
            return float(r.get("BC_err"))
        except (TypeError, ValueError):
            return float("nan")

    pts = sorted([(float(r[x_col]), float(r["BC"]), _err(r)) for r in sel],
                 key=lambda t: t[0])
    if len(pts) < 3:
        raise ValueError(f"need >=3 finite BC points for L_long={L_long}, got {len(pts)}")
    x = np.array([p[0] for p in pts])
    bc = np.array([p[1] for p in pts])
    bc_err = np.array([p[2] for p in pts])
    L_short = int(float(sel[0]["L_short"]))
    beta_mean = float(np.mean([float(r.get("beta", 1.0)) for r in sel]))
    # weight the fit by the bootstrap errors when all are present and positive
    sigma = bc_err if np.all(np.isfinite(bc_err) & (bc_err > 0)) else None

    method = "sigmoid"
    x_c = float("nan")
    unc = float("nan")
    try:
        from scipy.optimize import curve_fit

        p0 = [1.0 / 3.0, float(bc.max()), float(np.median(x)),
              max((x.max() - x.min()) / 10.0, 1e-3)]
        popt, pcov = curve_fit(_sigmoid, x, bc, p0=p0, sigma=sigma,
                               absolute_sigma=sigma is not None, maxfev=20000)
        x_c = float(popt[2])
        unc = float(np.sqrt(pcov[2, 2])) if np.all(np.isfinite(pcov)) else float("nan")
        if not (x.min() <= x_c <= x.max()):  # fit ran away -> fall back
            raise RuntimeError("sigmoid inflection outside sweep range")
    except Exception as exc:
        method = "threshold_5_9"
        x_c = _threshold_crossing(x, bc)
        unc = float("nan")
        print(f"[bimodality] sigmoid fit fell back to threshold: {exc}", flush=True)

    eps_c = x_c / beta_mean if (x_col == "beta_epsilon" and beta_mean) else x_c
    bracket = transition_bracket(x, bc, bc_err)
    recommended = bracket["transition_half_width"]
    if not np.isfinite(recommended):
        recommended = bracket["transition_half_width_envelope"]
    if not np.isfinite(recommended):
        recommended = unc
    return {
        "L_short": L_short,
        "L_long": int(L_long),
        "x_axis": x_col,
        "criticality_estimate": x_c,
        "epsilon_c_estimate": eps_c,
        "beta": beta_mean,
        "method": method,
        "fit_uncertainty": unc,
        **bracket,
        "recommended_uncertainty": recommended,
    }


# ---------------------------------------------------------------------------
# Orchestrator - the requested "find criticality of a coex run" function
# ---------------------------------------------------------------------------

def find_criticality(
    base_dir: str,
    combo_params: dict,
    *,
    out_dir: str = "criticality",
    manage_csv: Optional[str] = None,
    make_plots: bool = True,
) -> dict:
    """Full path: coex run data -> epsilon_c for one (scheme, size).

    combo_params: {scheme, delta_f, delta_mu, k, Lx, Ly} (epsilon omitted).
    Writes <out_dir>/bc_vs_epsilon.csv and appends to <out_dir>/epsilon_c.csv,
    caches column ops under <out_dir>/cache/column_op/, and (optionally) writes
    plots. Returns the epsilon_c dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    bc_csv = os.path.join(out_dir, "bc_vs_epsilon.csv")
    eps_c_csv = os.path.join(out_dir, "epsilon_c.csv")
    cache_dir = os.path.join(out_dir, "cache", "column_op")

    rows = sweep_bc(base_dir, combo_params, bc_csv, cache_dir, manage_csv=manage_csv)
    if not rows:
        raise RuntimeError(f"no BC rows produced for {combo_params} under {base_dir}")

    L_long = int(combo_params["Lx"])
    result = locate_epsilon_c(rows, L_long, x_col="beta_epsilon")
    _append_row(eps_c_csv, result, EPS_C_FIELDS)

    if make_plots:
        try:
            from plot_bimodality import plot_bc_vs_epsilon

            size = f"{int(combo_params['Lx'])}x{int(combo_params['Ly'])}"
            plot_bc_vs_epsilon(
                bc_csv, os.path.join(out_dir, f"bc_max_vs_beta_epsilon_{size}.png"),
                x_col="beta_epsilon", crit=result["criticality_estimate"],
                crit_row=result,
            )
        except Exception as exc:  # plotting is non-critical
            print(f"[bimodality] plotting skipped: {exc}", flush=True)

    rec = result.get("recommended_uncertainty", result["fit_uncertainty"])
    print(
        f"[bimodality] criticality({L_long}x{combo_params['Ly']}) = "
        f"(beta*eps)_c={result['criticality_estimate']:.4f} "
        f"+/- {rec} (transition bracket; sigmoid fit +/- {result['fit_uncertainty']})  "
        f"[epsilon_c={result['epsilon_c_estimate']:.4f}] ({result['method']})",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Phase diagram - a BC_max-vs-(beta*epsilon) curve per delta_mu (one scheme+size)
# ---------------------------------------------------------------------------

def discover_delta_mus(base_dir: str, scheme: str, delta_f: float,
                       Lx: int, Ly: int) -> list[float]:
    """All delta_mu values present on disk for this scheme+delta_f+size.

    Combo dir names are {size}_{scheme}_deltaF{df}_dmu{dmu}_epsilon{eps} (k is
    not in the path), so we glob the fixed prefix and parse the dmu tag.
    """
    prefix = f"{size_tag(Lx, Ly)}_{scheme}_deltaF{param_tag(delta_f)}_dmu"
    dmus: set[float] = set()
    for root in results_roots(base_dir):
        for path in glob.glob(os.path.join(root, prefix + "*")):
            if not os.path.isdir(path):
                continue
            tail = os.path.basename(path)[len(prefix):]  # "{dmu_tag}_epsilon{eps_tag}"
            if "_epsilon" not in tail:
                continue
            try:
                dmus.add(round(untag(tail.split("_epsilon", 1)[0]), 6))
            except ValueError:
                continue
    return sorted(dmus)


# ---------------------------------------------------------------------------
# Data-quality check - is the epsilon grid fine enough near the crossover?
# ---------------------------------------------------------------------------

def inspect_coverage(
    base_dir: str,
    *,
    scheme: str,
    delta_f: float,
    k: float,
    Lx: int,
    Ly: int,
    delta_mus: Optional[list[float]] = None,
    ref_step: float = 0.005,
) -> list[dict]:
    """Report the epsilon grid per delta_mu (count, range, step min/median/max).

    `ref_step` is the reference spacing to compare against (default 0.005, the
    susceptibility sweep's step). Flags any delta_mu whose finest step is coarser
    than ref_step so you can see where the coex grid under-resolves the crossover.
    """
    if delta_mus is None:
        delta_mus = discover_delta_mus(base_dir, scheme, delta_f, Lx, Ly)
    if not delta_mus:
        print(f"[inspect] no {Lx}x{Ly} {scheme} deltaF={delta_f} combos under {base_dir}",
              flush=True)
        return []

    report = []
    print(f"[inspect] reference step = {ref_step} (susceptibility sweep)", flush=True)
    for dmu in delta_mus:
        cp = {"scheme": scheme, "delta_f": delta_f, "delta_mu": dmu,
              "k": k, "Lx": Lx, "Ly": Ly}
        eps = sorted(e for e, _ in discover_epsilons(base_dir, cp))
        if not eps:
            print(f"[inspect] dmu={dmu:+.2f}: no epsilons found", flush=True)
            continue
        d = np.diff(eps) if len(eps) > 1 else np.array([float("nan")])
        step_min = float(np.min(d))
        info = {
            "delta_mu": dmu, "n_eps": len(eps),
            "eps_min": eps[0], "eps_max": eps[-1],
            "step_min": step_min, "step_median": float(np.median(d)),
            "step_max": float(np.max(d)),
            "as_fine_as_ref": bool(step_min <= ref_step + 1e-9),
        }
        report.append(info)
        flag = "OK " if info["as_fine_as_ref"] else "COARSE"
        print(f"[inspect] dmu={dmu:+.2f}  n_eps={len(eps):3d}  "
              f"range=[{eps[0]:+.3f},{eps[-1]:+.3f}]  "
              f"step min/med/max={step_min:.4f}/{np.median(d):.4f}/{np.max(d):.4f}"
              f"  [{flag}]", flush=True)
    return report


def phase_diagram(
    base_dir: str,
    *,
    scheme: str,
    delta_f: float,
    k: float,
    Lx: int,
    Ly: int,
    delta_mus: Optional[list[float]] = None,
    out_dir: str = "criticality",
    manage_csv: Optional[str] = None,
    mu_reduction: str = "max",
    make_plots: bool = True,
    show_transition_bracket: bool = True,
) -> list[dict]:
    """BC_max-vs-(beta*epsilon) curve for each delta_mu at a fixed scheme+size.

    Reproduces the "thermal phase diagram" family plot: one sigmoid per delta_mu,
    y = max Sarle BC over the mu-sweep, x = beta*epsilon, and the sigmoid
    inflection is that delta_mu's criticality. If delta_mus is None, every
    delta_mu present on disk is used.

    Writes <out_dir>/bc_vs_beta_epsilon.csv (all delta_mu, tagged) and
    <out_dir>/criticality.csv (one row per delta_mu), and the family plot.
    Returns the list of per-delta_mu criticality dicts.
    """
    os.makedirs(out_dir, exist_ok=True)
    bc_csv = os.path.join(out_dir, "bc_vs_beta_epsilon.csv")
    crit_csv = os.path.join(out_dir, "criticality.csv")
    cache_dir = os.path.join(out_dir, "cache", "column_op")

    if delta_mus is None:
        delta_mus = discover_delta_mus(base_dir, scheme, delta_f, Lx, Ly)
        if not delta_mus:
            raise RuntimeError(
                f"no {Lx}x{Ly} {scheme} deltaF={delta_f} combos under {base_dir}"
            )
        print(f"[bimodality] discovered delta_mu: {delta_mus}", flush=True)

    results = []
    for dmu in delta_mus:
        combo = {"scheme": scheme, "delta_f": delta_f, "delta_mu": dmu,
                 "k": k, "Lx": Lx, "Ly": Ly}
        rows = sweep_bc(base_dir, combo, bc_csv, cache_dir,
                        manage_csv=manage_csv, mu_reduction=mu_reduction)
        if len(rows) < 3:
            print(f"[bimodality] dmu={dmu}: only {len(rows)} epsilon points, "
                  "skipping criticality fit", flush=True)
            continue
        res = locate_epsilon_c(rows, int(Lx), x_col="beta_epsilon")
        res.update({"scheme": scheme, "delta_f": delta_f, "delta_mu": dmu, "k": k})
        _append_row(crit_csv, res, CRIT_FIELDS)
        results.append(res)
        rec = res.get("recommended_uncertainty", res["fit_uncertainty"])
        print(
            f"[bimodality] dmu={dmu:+.3f}: (beta*eps)_c="
            f"{res['criticality_estimate']:.4f} +/- {rec} "
            f"(transition; sigmoid +/- {res['fit_uncertainty']}) "
            f"({res['method']})",
            flush=True,
        )

    if make_plots and results:
        try:
            from plot_bimodality import plot_bc_family

            size = f"{Lx}x{Ly}"
            plot_bc_family(
                bc_csv, os.path.join(out_dir, f"bc_max_phase_diagram_{size}.png"),
                crit_csv=crit_csv,
                show_transition_bracket=show_transition_bracket,
            )
        except Exception as exc:  # plotting is non-critical
            print(f"[bimodality] family plot skipped: {exc}", flush=True)

    return results


def scheme_comparison(
    panels: list[dict],
    *,
    out_dir: str = "criticality/scheme_comparison",
    mu_reduction: str = "max",
) -> str:
    """Side-by-side Scheme 1 vs Scheme 3 BC_max family plots with error bars.

    Each entry in ``panels`` is a dict with keys:
      base_dir, scheme, delta_f, k, Lx, Ly, delta_mus, title
    and optional manage_csv. Runs ``phase_diagram`` per panel (no per-panel
    family plot), then writes one combined PNG via ``plot_bc_scheme_comparison``.
    """
    os.makedirs(out_dir, exist_ok=True)
    plot_panels: list[tuple[str, str]] = []
    for i, panel in enumerate(panels):
        sub_out = os.path.join(out_dir, f"panel{i}")
        os.makedirs(sub_out, exist_ok=True)
        phase_diagram(
            panel["base_dir"],
            scheme=panel["scheme"],
            delta_f=panel["delta_f"],
            k=panel["k"],
            Lx=panel["Lx"],
            Ly=panel["Ly"],
            delta_mus=panel["delta_mus"],
            out_dir=sub_out,
            manage_csv=panel.get("manage_csv"),
            mu_reduction=mu_reduction,
            make_plots=False,
        )
        bc_csv = os.path.join(sub_out, "bc_vs_beta_epsilon.csv")
        plot_panels.append((bc_csv, panel["title"]))

    from plot_bimodality import plot_bc_scheme_comparison

    size = f"{panels[0]['Lx']}x{panels[0]['Ly']}"
    out_png = os.path.join(out_dir, f"bc_max_scheme_comparison_{size}.png")
    plot_bc_scheme_comparison(plot_panels, out_png)
    print(f"[bimodality] wrote comparison figure {out_png}", flush=True)
    return out_png


# Default panel configs for the standard Scheme 1 vs Scheme 3 comparison figure.
DEFAULT_SCHEME_COMPARISON_PANELS = [
    {
        "title": "Scheme 1: Homogenous",
        "base_dir": "results",
        "scheme": "homo",
        "delta_f": 0.0,
        "k": 1.0,
        "Lx": 160,
        "Ly": 16,
        "delta_mus": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
    },
    {
        "title": "Scheme 3: Negative",
        "base_dir": "scheme3",
        "scheme": "negative_drive",
        "delta_f": 0.0,
        "k": 0.1,
        "Lx": 160,
        "Ly": 16,
        "delta_mus": [1.0, 2.0, 3.0, 4.0, 4.5],
    },
]


# ---------------------------------------------------------------------------
# Figure 1 - P(phi_col) histograms at coexistence, several epsilon, one size
# ---------------------------------------------------------------------------

def pooled_at_coexistence(
    base_dir: str,
    combo_params: dict,
    cache_dir: str,
    *,
    manage_csv: Optional[str] = None,
    mu_reduction: str = "max",
) -> dict:
    """Pooled column-op sample (+stats) at the coexistence mu for one combo.

    combo_params includes epsilon. Uses the exact same max-BC-over-mu selection
    as sweep_bc, so the returned distribution is the one whose BC the curve
    reports. Returns a dict with 'pooled', 'BC', 'mu', 'n_pooled', ...
    """
    combo_dir = os.path.join(base_dir, combo_dir_name(combo_params))
    eps = combo_params["epsilon"]
    mu_coex = resolve_mu_coex(eps, combo_dir, manage_csv=manage_csv, combo_params=combo_params)
    mu_dirs = enumerate_mu_dirs(combo_dir)
    if not mu_dirs:
        raise FileNotFoundError(f"no mu dirs under {combo_dir}")
    if mu_reduction == "nearest_coex":
        mu_dirs = [min(mu_dirs, key=lambda dm: abs(dm[1] - mu_coex))]
    best = _best_mu_over_sweep(
        combo_dir, cache_dir,
        epsilon=eps, mu_coex=mu_coex, combo_name=os.path.basename(combo_dir),
        mu_dirs=mu_dirs, keep_pooled=True,
        selection=mu_reduction if mu_reduction != "nearest_coex" else "max",
    )
    if best is None:
        raise ValueError(f"no finite BC at eps={eps} for {combo_dir}")
    return best


def histogram_data(
    base_dir: str,
    combo_params: dict,
    epsilons: list[float],
    cache_dir: str,
    *,
    manage_csv: Optional[str] = None,
    mu_reduction: str = "balanced",
) -> list[dict]:
    """P(phi_col) samples at coexistence for each requested epsilon (nearest
    available), for a fixed (scheme, size, delta_mu). combo_params omits epsilon.
    """
    avail = [eps for eps, _ in discover_epsilons(base_dir, combo_params)]
    if not avail:
        raise FileNotFoundError(f"no epsilon combos for {combo_params} under {base_dir}")
    data = []
    for target in epsilons:
        eps = min(avail, key=lambda e: abs(e - target))
        best = pooled_at_coexistence(
            base_dir, {**combo_params, "epsilon": eps}, cache_dir,
            manage_csv=manage_csv, mu_reduction=mu_reduction,
        )
        data.append({
            "epsilon": eps, "pooled": best["pooled"], "BC": best["BC"],
            "mu": best["mu"], "n_pooled": best["n_pooled"], "Ly": best.get("Ly"),
            "frac_liq": best.get("frac_liq"), "frac_gas": best.get("frac_gas"),
            "balance": best.get("balance"),
        })
        print(f"[bimodality] hist eps={eps:+.3f} BC={best['BC']:.3f} "
              f"n_pooled={best['n_pooled']} @mu={best['mu']:+.3f} "
              f"liq={best.get('frac_liq', 0):.0%} gas={best.get('frac_gas', 0):.0%}",
              flush=True)
    return data


def make_histogram_figure(
    base_dir: str,
    combo_params: dict,
    epsilons: list[float],
    out_png: str,
    *,
    cache_dir: Optional[str] = None,
    manage_csv: Optional[str] = None,
    mu_reduction: str = "balanced",
    write_csv: bool = True,
) -> str:
    """Figure 1: side-by-side P(phi_col) histograms at several epsilon for one size.

    Also writes `<png_stem>_samples.csv` and `<png_stem>_hist.csv` when
    write_csv is True (same coexistence samples / bins as the figure).
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(out_png) or ".", "cache", "column_op")
    data = histogram_data(
        base_dir, combo_params, epsilons, cache_dir,
        manage_csv=manage_csv, mu_reduction=mu_reduction,
    )

    from plot_bimodality import plot_pooled_histograms

    size = f"{int(combo_params['Lx'])}x{int(combo_params['Ly'])}"
    title = (f"$P(\\phi_{{col}})$  {size}  $\\Delta\\mu$={combo_params['delta_mu']}"
             "  (bimodal $\\to$ unimodal)")
    plot_pooled_histograms(data, out_png, title=title, write_csv=write_csv)
    return out_png


# ---------------------------------------------------------------------------
# Figure 2 - BC_max vs beta*epsilon for several sizes (finite-size study)
# ---------------------------------------------------------------------------

def run_fss(
    base_dir: str,
    *,
    scheme: str,
    delta_f: float,
    k: float,
    sizes: list[tuple[int, int]],
    delta_mus: list[float],
    out_dir: str = "criticality",
    manage_csv: Optional[str] = None,
    mu_reduction: str = "max",
    make_plots: bool = True,
) -> list[dict]:
    """BC_max-vs-(beta*epsilon) for several slab sizes, one figure per delta_mu.

    sizes: list of (Lx, Ly). For each (size, delta_mu) present on disk it runs
    the BC_max sweep into one shared CSV (tagged by L and delta_mu) and fits the
    criticality. Then, per delta_mu, it plots one curve per size (Figure 2) so
    the crossover sharpening with L is visible. Missing (size, delta_mu) combos
    are skipped. Returns the per-(size, delta_mu) criticality dicts.
    """
    os.makedirs(out_dir, exist_ok=True)
    bc_csv = os.path.join(out_dir, "bc_vs_beta_epsilon.csv")
    crit_csv = os.path.join(out_dir, "criticality.csv")
    cache_dir = os.path.join(out_dir, "cache", "column_op")
    for stale in (bc_csv, crit_csv):  # fresh: avoid duplicate rows on re-run
        if os.path.isfile(stale):
            os.remove(stale)

    all_rows: list[dict] = []
    results: list[dict] = []
    for (Lx, Ly) in sizes:
        for dmu in delta_mus:
            combo = {"scheme": scheme, "delta_f": delta_f, "delta_mu": dmu,
                     "k": k, "Lx": Lx, "Ly": Ly}
            rows = sweep_bc(base_dir, combo, bc_csv, cache_dir,
                            manage_csv=manage_csv, mu_reduction=mu_reduction)
            if not rows:
                print(f"[bimodality] no data for {Lx}x{Ly} dmu={dmu}; skipping",
                      flush=True)
                continue
            all_rows.extend(rows)
            if len(rows) >= 3:
                res = locate_epsilon_c(rows, int(Lx), x_col="beta_epsilon")
                res.update({"scheme": scheme, "delta_f": delta_f,
                            "delta_mu": dmu, "k": k})
                _append_row(crit_csv, res, CRIT_FIELDS)
                results.append(res)
                print(f"[bimodality] {Lx}x{Ly} dmu={dmu:+.2f}: (beta*eps)_c="
                      f"{res['criticality_estimate']:.4f} +/- "
                      f"{res['fit_uncertainty']} ({res['method']})", flush=True)

    if make_plots and all_rows:
        from plot_bimodality import plot_bc_vs_epsilon

        for dmu in sorted({r["delta_mu"] for r in all_rows}):
            out_png = os.path.join(out_dir, f"bc_vs_beta_epsilon_dmu{param_tag(dmu)}.png")
            plot_bc_vs_epsilon(
                bc_csv, out_png, delta_mu=dmu,
                title=f"Max BC vs $\\beta\\epsilon$  $\\Delta\\mu$={dmu}",
            )
            print(f"[bimodality] wrote {out_png}", flush=True)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_size(token: str) -> tuple[int, int]:
    """'80x8' -> (80, 8) i.e. (Lx, Ly), matching the combo dir naming."""
    lx, ly = token.lower().split("x")
    return int(lx), int(ly)


def _add_common(sp) -> None:
    sp.add_argument("--base-dir", required=True,
                    help="Dir holding the {size}_{scheme}_deltaF..._dmu..._epsilon* combos.")
    sp.add_argument("--scheme", default="homo")
    sp.add_argument("--delta-f", type=float, required=True)
    sp.add_argument("--k", type=float, default=1.0,
                    help="Chemical rate (provenance only in max mode; not in dir name).")
    sp.add_argument("--out-dir", default="criticality")
    sp.add_argument("--manage-csv", default=None,
                    help="Optional coex manage CSV for mu_coex_FITTED (not needed in max mode).")
    sp.add_argument("--mu-reduction", choices=["max", "nearest_coex", "balanced"],
                    default="max",
                    help="How to pick one mu per epsilon: max BC (curves), "
                         "nearest mu_coex, or balanced liquid/gas fractions (histograms).")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Bimodality-based criticality from coex snapshots: max Sarle BC "
                    "over the mu-sweep vs beta*epsilon; criticality = sigmoid inflection.",
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # phase-diagram: one size, a BC_max curve per delta_mu (family plot).
    pd_ = sub.add_parser("phase-diagram", help="one size, a curve per delta_mu")
    _add_common(pd_)
    pd_.add_argument("--Lx", type=int, required=True, help="Long axis (e.g. 160).")
    pd_.add_argument("--Ly", type=int, required=True, help="Short axis (e.g. 16).")
    pd_.add_argument("--delta-mus", type=float, nargs="*", default=None,
                     help="Explicit delta_mu list; default discovers every one on disk.")
    pd_.add_argument("--no-plots", action="store_true")
    pd_.add_argument("--no-transition-bracket", action="store_true",
                     help="Family plot: BC_err bars only (no orange transition shading).")

    # replot-family: regenerate PNG from an existing bc_vs_beta_epsilon.csv.
    rp = sub.add_parser(
        "replot-family",
        help="replot bc_vs_beta_epsilon.csv with BC_err bars (no recomputation)",
    )
    rp.add_argument("--bc-csv", required=True,
                    help="Existing bc_vs_beta_epsilon.csv from phase-diagram.")
    rp.add_argument("--out", default=None,
                    help="Output PNG path (default: bc_max_phase_diagram_{L}.png beside CSV).")
    rp.add_argument("--crit-csv", default=None,
                    help="Optional criticality.csv for transition brackets.")
    rp.add_argument("--no-transition-bracket", action="store_true",
                    help="BC_err bars only; skip orange transition shading and crit markers.")
    rp.add_argument("--cache-dir", default=None,
                    help="column_op cache dir (default: <csv-dir>/cache/column_op).")
    rp.add_argument("--base-dir", default=None,
                    help="Resolve source_dir / combo paths (e.g. results).")
    rp.add_argument("--skip-backfill", action="store_true",
                    help="Do not recompute missing BC_err from cache/snapshots.")

    # histograms (Figure 1): P(phi_col) at several epsilon, one size + delta_mu.
    hg = sub.add_parser("histograms", help="Figure 1: P(phi_col) at several epsilon")
    _add_common(hg)
    hg.set_defaults(mu_reduction="balanced")
    hg.add_argument("--Lx", type=int, required=True)
    hg.add_argument("--Ly", type=int, required=True)
    hg.add_argument("--delta-mu", type=float, required=True)
    hg.add_argument("--epsilons", type=float, nargs="+", required=True,
                    help="epsilon values to show (nearest available is used).")

    # fss (Figure 2): BC_max vs beta*epsilon for several sizes, per delta_mu.
    fs = sub.add_parser("fss", help="Figure 2: BC_max vs beta*epsilon, several sizes")
    _add_common(fs)
    fs.add_argument("--sizes", nargs="+", required=True,
                    help="LxxLy tokens, e.g. 80x8 160x16 320x32.")
    fs.add_argument("--delta-mus", type=float, nargs="+", required=True,
                    help="delta_mu values (missing size/delta_mu combos are skipped).")

    # inspect: report the epsilon grid fineness per delta_mu (data-quality check).
    ins = sub.add_parser("inspect", help="report epsilon grid coverage/step per delta_mu")
    _add_common(ins)
    ins.add_argument("--Lx", type=int, required=True)
    ins.add_argument("--Ly", type=int, required=True)
    ins.add_argument("--delta-mus", type=float, nargs="*", default=None,
                     help="Explicit delta_mu list; default discovers every one on disk.")
    ins.add_argument("--ref-step", type=float, default=0.005,
                     help="Reference epsilon step to compare against (default 0.005).")

    # scheme-comparison: side-by-side Scheme 1 vs Scheme 3 family plots + error bars.
    sc = sub.add_parser(
        "scheme-comparison",
        help="Scheme 1 vs Scheme 3 side-by-side BC_max family plot (with BC_err bars)",
    )
    sc.add_argument("--out-dir", default="criticality/scheme_comparison")
    sc.add_argument("--mu-reduction", choices=["max", "nearest_coex", "balanced"],
                    default="max")
    sc.add_argument(
        "--plot-only",
        nargs="+",
        metavar="CSV:TITLE",
        help="Skip recomputation; plot existing CSVs as PANEL (e.g. "
             "criticality/s1/bc_vs_beta_epsilon.csv:Scheme\\ 1:\\ Homogenous).",
    )

    args = p.parse_args()

    if args.mode == "phase-diagram":
        results = phase_diagram(
            args.base_dir, scheme=args.scheme, delta_f=args.delta_f, k=args.k,
            Lx=args.Lx, Ly=args.Ly, delta_mus=args.delta_mus, out_dir=args.out_dir,
            manage_csv=args.manage_csv, mu_reduction=args.mu_reduction,
            make_plots=not args.no_plots,
            show_transition_bracket=not args.no_transition_bracket,
        )
        print(f"\n[bimodality] wrote {len(results)} delta_mu curve(s) to {args.out_dir}/")

    elif args.mode == "replot-family":
        from plot_bimodality import plot_bc_family

        bc_csv = args.bc_csv
        bc_csv = os.path.abspath(bc_csv)
        if not args.skip_backfill:
            filled, already, total = backfill_bc_err(
                bc_csv,
                cache_dir=args.cache_dir,
                base_dir=args.base_dir,
            )
            print(
                f"[bimodality] BC_err backfill: filled {filled}, "
                f"already present {already}, total {total}",
                flush=True,
            )
        with open(bc_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        n_err, err_min, err_max = _bc_err_stats(rows)
        if n_err == 0:
            print(
                "[bimodality] WARNING: no finite BC_err in CSV — plot will have no "
                "error bars. Ensure cache/column_op/*.npy exists beside the CSV, or "
                "pass --base-dir results so source snapshots can be read.",
                flush=True,
            )
        else:
            print(
                f"[bimodality] BC_err: {n_err}/{len(rows)} rows, "
                f"range [{err_min:.4g}, {err_max:.4g}]",
                flush=True,
            )
        with open(bc_csv, newline="") as f:
            row = next(csv.DictReader(f))
        L_long = int(row["L_long"])
        L_short = int(row["L_short"])
        size = f"{L_long}x{L_short}"
        out_png = args.out or os.path.join(
            os.path.dirname(os.path.abspath(bc_csv)),
            f"bc_max_phase_diagram_{size}.png",
        )
        crit_csv = args.crit_csv
        if crit_csv is None:
            default_crit = os.path.join(os.path.dirname(os.path.abspath(bc_csv)), "criticality.csv")
            if os.path.isfile(default_crit):
                crit_csv = default_crit
        plot_bc_family(
            bc_csv,
            out_png,
            crit_csv=crit_csv,
            show_transition_bracket=not args.no_transition_bracket,
        )
        print(f"\n[bimodality] wrote family plot {out_png}")

    elif args.mode == "histograms":
        combo = {"scheme": args.scheme, "delta_f": args.delta_f, "delta_mu": args.delta_mu,
                 "k": args.k, "Lx": args.Lx, "Ly": args.Ly}
        size = f"{args.Lx}x{args.Ly}"
        out_png = os.path.join(
            args.out_dir, f"pdf_phi_col_{size}_dmu{param_tag(args.delta_mu)}.png")
        make_histogram_figure(
            args.base_dir, combo, args.epsilons, out_png,
            cache_dir=os.path.join(args.out_dir, "cache", "column_op"),
            manage_csv=args.manage_csv,
            mu_reduction=args.mu_reduction,
        )
        print(f"\n[bimodality] wrote histogram figure {out_png}")

    elif args.mode == "fss":
        sizes = [_parse_size(s) for s in args.sizes]
        results = run_fss(
            args.base_dir, scheme=args.scheme, delta_f=args.delta_f, k=args.k,
            sizes=sizes, delta_mus=args.delta_mus, out_dir=args.out_dir,
            manage_csv=args.manage_csv, mu_reduction=args.mu_reduction,
        )
        print(f"\n[bimodality] wrote {len(results)} (size,delta_mu) criticality "
              f"row(s) to {args.out_dir}/")

    elif args.mode == "inspect":
        inspect_coverage(
            args.base_dir, scheme=args.scheme, delta_f=args.delta_f, k=args.k,
            Lx=args.Lx, Ly=args.Ly, delta_mus=args.delta_mus, ref_step=args.ref_step,
        )

    elif args.mode == "scheme-comparison":
        if args.plot_only:
            from plot_bimodality import plot_bc_scheme_comparison

            panels = []
            for spec in args.plot_only:
                panel_csv, title = spec.split(":", 1)
                panels.append((panel_csv, title.replace("\\ ", " ")))
            size = "160x16"
            out_png = os.path.join(args.out_dir, f"bc_max_scheme_comparison_{size}.png")
            plot_bc_scheme_comparison(panels, out_png)
            print(f"\n[bimodality] wrote comparison figure {out_png}")
        else:
            out_png = scheme_comparison(
                DEFAULT_SCHEME_COMPARISON_PANELS,
                out_dir=args.out_dir,
                mu_reduction=args.mu_reduction,
            )
            print(f"\n[bimodality] wrote comparison figure {out_png}")

    else:
        raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
