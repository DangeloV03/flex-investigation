"""Bimodality coefficient across the epsilon sweep of one (scheme, size), written as CSV rows."""

from __future__ import annotations

import csv
import glob
import os
from typing import Optional

import numpy as np

from flexsim.coex.layout import combo_dir_name, mu_dir_name
from flexsim.criticality.bc import BC_BIMODAL_CUTOFF, bc_bootstrap_error, cache_column_op, extract_column_op, pool_column_op, sarle_bc, untag
from flexsim.criticality.mu_selection import read_output_field, enumerate_mu_dirs, resolve_mu_coex


# ---------------------------------------------------------------------------
# CSV helper (append-as-you-go, header on first write)
# ---------------------------------------------------------------------------

def append_row(csv_path: str, row: dict, fieldnames: list[str]) -> None:
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


def bc_err_stats(rows: list[dict]) -> tuple[int, float, float]:
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
    "BC_at_criticality", "BC_target",
    "transition_bc_high", "transition_bc_low",
    "transition_x_high", "transition_x_low", "transition_half_width",
    "transition_half_width_envelope", "recommended_uncertainty",
]

# Per-(Delta mu) criticality rows for the phase-diagram family.
CRIT_FIELDS = [
    "scheme", "delta_f", "delta_mu", "k", "L_short", "L_long", "x_axis",
    "criticality_estimate", "epsilon_c_estimate", "beta", "method", "fit_uncertainty",
    "BC_at_criticality", "BC_target",
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


def _select_mu_winner(candidates: list[dict], selection: str) -> dict:
    """Pick one mu-dir stats dict according to selection policy."""
    if selection == "balanced":
        strong = [c for c in candidates if c["BC"] >= BC_BIMODAL_CUTOFF]
        pool = strong if strong else candidates
        return max(pool, key=lambda c: (c["balance"], c["BC"]))
    if selection == "zero_mean":
        # Coexistence: ⟨φ_col⟩ closest to 0; ties -> max BC among those.
        abs_mean = [abs(float(c["mean"])) for c in candidates]
        best_abs = min(abs_mean)
        near_zero = [
            c for c, a in zip(candidates, abs_mean)
            if abs(a - best_abs) <= 1e-12
        ]
        return max(near_zero, key=lambda c: c["BC"])
    # default: max BC over the mu-sweep
    return max(candidates, key=lambda c: c["BC"])


def best_mu_over_sweep(
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

    Shared by sweep_bc and pooled_at_coexistence. selection=
      * "max"        -> maximum Sarle BC over the mu-sweep
      * "balanced"   -> most even liquid/gas fractions among bimodal points
      * "zero_mean"  -> ⟨φ_col⟩ closest to 0 (coexistence), then BC at that mu
    Returns None if no mu dir yields a finite BC. Winner carries bootstrap
    'BC_err'; with keep_pooled=True the winning pooled array is included.
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
    winner = _select_mu_winner(candidates, selection)
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
    mu_reduction: str = "zero_mean",
) -> list[dict]:
    """Steps 1-3 across the epsilon sweep for one (scheme, size); append BC rows.

    For each epsilon the mu-sweep provides several ensemble points. We reduce the
    sweep to one bimodality number per epsilon:
      * mu_reduction="zero_mean"    -> μ with ⟨φ_col⟩ closest to 0, then BC there
        (default for coexistence criticality).
      * mu_reduction="max"          -> MAXIMUM BC over the mu-sweep.
      * mu_reduction="nearest_coex" -> BC at the single mu dir nearest mu_coex.
      * mu_reduction="balanced"     -> most even liquid/gas column fractions.

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

        selection = (
            "max" if mu_reduction == "nearest_coex" else mu_reduction
        )
        best = best_mu_over_sweep(
            combo_dir, cache_dir,
            epsilon=eps, mu_coex=mu_coex, combo_name=combo_name, mu_dirs=mu_dirs,
            selection=selection,
        )
        if best is None:
            print(f"[bimodality] skip eps={eps}: no finite BC over mu-sweep", flush=True)
            continue

        beta = read_output_field(best["mu_dir"], "beta") or 1.0
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
        append_row(out_csv, row, BC_FIELDS)
        rows.append(row)
        print(
            f"[bimodality] eps={eps:+.4f} beta*eps={beta * eps:+.4f} "
            f"L={best['L_long']}x{combo_params['Ly']} n_mu={len(mu_dirs)} "
            f"BC={best['BC']:.4f} ⟨φ⟩={best['mean']:+.4f} @mu={best['mu']:+.4f} "
            f"({mu_reduction})",
            flush=True,
        )
    return rows
