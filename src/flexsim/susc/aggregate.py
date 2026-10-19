"""Per-(L, epsilon) observables from m(t) time series: chi, c, U4, <|m|> (per trajectory and pooled)."""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from flexsim.susc.layout import find_susceptibility_csvs, read_susceptibility_csv


def _tail_slice(n: int, tail_fraction: float) -> slice:
    """Index slice keeping the last `tail_fraction` of an n-length series (>=1 sample)."""
    if tail_fraction >= 1.0:
        return slice(None)
    keep = max(1, int(round(n * tail_fraction)))
    return slice(n - keep, None)


def _compute_traj_stats(ts_path: str, meta: dict, tail_fraction: float = 1.0) -> dict | None:
    """Compute per-trajectory observables from a single m_timeseries CSV.

    tail_fraction < 1 keeps only the last fraction of each series (equilibrated tail).
    """
    if not os.path.isfile(ts_path):
        return None

    ts = pd.read_csv(ts_path)
    if ts.empty:
        return None
    ts = ts.iloc[_tail_slice(len(ts), tail_fraction)]

    beta = float(meta["beta"])
    mu = float(meta["mu"])
    delta_f = float(meta["delta_f"])
    L = int(float(meta["L"]))
    N = L * L

    m_arr = ts["m"].values.astype(float)
    m_mean = float(np.mean(m_arr))
    abs_m_mean = float(np.mean(np.abs(m_arr)))
    m2_mean = float(np.mean(m_arr ** 2))
    m4_mean = float(np.mean(m_arr ** 4))

    # χ = (1/NT)(⟨M²⟩-⟨|M|⟩²) with M = N·m, |M| = N·|m|, T = 1/β.
    # Subtracting ⟨|M|⟩² (not ⟨M⟩²) removes the between-well m₀² term that flipping
    # small-L runs would otherwise pick up — the connected (FSS) susceptibility.
    M_arr = N * m_arr
    chi = beta / N * (float(np.mean(M_arr ** 2)) - float(np.mean(np.abs(M_arr))) ** 2)
    u4 = 1.0 - m4_mean / (3.0 * m2_mean ** 2) if m2_mean != 0 else float("nan")

    result: dict = {
        "L": L,
        "epsilon": float(meta["epsilon"]),
        "beta": beta,
        "m_mean": m_mean,
        "abs_m_mean": abs_m_mean,
        "m2_mean": m2_mean,
        "m4_mean": m4_mean,
        "chi": chi,
        "u4": u4,
    }

    # c(T,L) = (1/NT²)(⟨E²⟩ - ⟨E⟩²).  At μ=2ε the Nm terms in e_interact and
    # e_chem cancel exactly so E_total is spin-invariant; use the stored energy directly.
    if "energy" in ts.columns:
        e_total = ts["energy"].values.astype(float)
        e2_mean = float(np.mean(e_total ** 2))
        e_mean  = float(np.mean(e_total))
        T = 1.0 / beta
        epsilon = float(meta["epsilon"])
        result["c"] = (e2_mean - e_mean ** 2) / (N * T ** 2 * abs(epsilon))

    return result


def aggregate(
    results_dir: str,
    tail_fraction: float = 1.0,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan for susceptibility_data.csv files under results_dir, load each replica's
    m_timeseries CSV, compute per-trajectory observables, then average over replicas
    grouped by (L, ε). tail_fraction < 1 uses only each series' last fraction.
    """
    paths = find_susceptibility_csvs(results_dir)
    if not paths:
        raise FileNotFoundError(
            f"No susceptibility_data.csv under {results_dir!r} "
            f"(resolved: {os.path.abspath(results_dir)}). "
            f"Pass --results relative to repo root, e.g. susceptibility/results/exact_2026-07-02"
        )

    traj_records: list[dict] = []
    n_paths = len(paths)
    if verbose:
        print(f'  Found {n_paths} run dirs; loading replica timeseries...', flush=True)
    n_loaded = 0
    for i_path, csv_path in enumerate(paths):
        dirpath = os.path.dirname(csv_path)
        meta_rows = read_susceptibility_csv(csv_path)
        for meta in meta_rows:
            run_id = str(meta.get("id", "")).strip()
            if not run_id:
                continue
            ts_path = os.path.join(dirpath, f"m_timeseries_{run_id}.csv")
            stats = _compute_traj_stats(ts_path, meta, tail_fraction=tail_fraction)
            if stats:
                traj_records.append(stats)
                n_loaded += 1
        if verbose and ((i_path + 1) % 10 == 0 or i_path + 1 == n_paths):
            print(
                f'  ... {i_path + 1}/{n_paths} run dirs, {n_loaded} replicas loaded',
                flush=True,
            )

    if not traj_records:
        raise FileNotFoundError(
            "No timeseries files found — check that runs have completed."
        )

    if verbose:
        print(f'  Aggregating {n_loaded} replicas into (L, ε) means...', flush=True)

    def _stderr(s: pd.Series) -> float:
        return float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0

    df = pd.DataFrame(traj_records)
    rows_agg = []
    for (l_val, eps), sub in df.groupby(["L", "epsilon"]):
        row: dict = {
            "L": int(l_val),
            "epsilon": float(eps),
            "beta": float(sub["beta"].iloc[0]),
            "chi_mean": float(sub["chi"].mean()),
            "chi_stderr": _stderr(sub["chi"]),
            "m_mean": float(sub["m_mean"].mean()),
            "m_mean_stderr": _stderr(sub["m_mean"]),
            "abs_m_mean": float(sub["abs_m_mean"].mean()),
            "abs_m_mean_stderr": _stderr(sub["abs_m_mean"]),
            "m2_mean": float(sub["m2_mean"].mean()),
            "m4_mean": float(sub["m4_mean"].mean()),
            "u4": float(sub["u4"].mean()),
            "u4_err": _stderr(sub["u4"]),
            "n_replicas": int(len(sub)),
        }
        if "c" in sub.columns and sub["c"].notna().any():
            row["c_mean"] = float(sub["c"].mean())
            row["c_stderr"] = _stderr(sub["c"])
        else:
            row["c_mean"] = float("nan")
            row["c_stderr"] = float("nan")
        rows_agg.append(row)

    return pd.DataFrame(rows_agg).sort_values(["L", "epsilon"]), pd.DataFrame(traj_records)


def _load_traj_arrays(ts_path: str, meta: dict, tail_fraction: float = 1.0) -> dict | None:
    """Load one trajectory's raw per-chunk m (and recovered E_int) arrays.

    tail_fraction < 1 keeps only the last fraction of each series (equilibrated tail).
    """
    if not os.path.isfile(ts_path):
        return None
    ts = pd.read_csv(ts_path)
    if ts.empty or "m" not in ts.columns:
        return None
    ts = ts.iloc[_tail_slice(len(ts), tail_fraction)]

    beta = float(meta["beta"])
    mu = float(meta["mu"])
    delta_f = float(meta["delta_f"])
    L = int(float(meta["L"]))
    N = L * L

    m = ts["m"].to_numpy(float)
    e_int = None
    if {"energy", "rho_bonding", "rho_inert"}.issubset(ts.columns):
        rho_B = ts["rho_bonding"].to_numpy(float)
        rho_I = ts["rho_inert"].to_numpy(float)
        e_total = ts["energy"].to_numpy(float)
        e_chem = -beta * mu * N * rho_B - beta * (mu + delta_f) * N * rho_I
        e_int = e_total - e_chem

    return {"L": L, "epsilon": float(meta["epsilon"]), "beta": beta, "N": N, "m": m, "e_int": e_int}


def _jackknife(arrays: list[np.ndarray], stat_fn) -> tuple[float, float]:
    """Leave-one-replica-out jackknife of a pooled statistic.

    stat_fn maps a pooled sample array -> scalar. Returns (full_estimate, stderr),
    where the error reflects between-replica variation — the right scale here.
    """
    n = len(arrays)
    full = float(stat_fn(np.concatenate(arrays)))
    if n < 2:
        return full, 0.0
    partials = np.array([
        float(stat_fn(np.concatenate([arrays[j] for j in range(n) if j != i])))
        for i in range(n)
    ])
    mean = partials.mean()
    err = float(np.sqrt((n - 1) / n * np.sum((partials - mean) ** 2)))
    return full, err


def aggregate_pooled(
    results_dir: str,
    tail_fraction: float = 1.0,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool every replica's samples per (L, ε), then compute χ, c, U4 once.

    Workaround for non-ergodic short runs: combining replicas reconstructs the full
    P(m) the per-trajectory estimator misses. Errors are leave-one-replica-out jackknife.
    tail_fraction < 1 pools only each series' last fraction (equilibrated tail).
    """
    paths = find_susceptibility_csvs(results_dir)
    if not paths:
        raise FileNotFoundError(
            f"No susceptibility_data.csv under {results_dir!r} "
            f"(resolved: {os.path.abspath(results_dir)}). "
            f"Pass --results relative to repo root, e.g. susceptibility/results/exact_2026-07-02"
        )

    groups: dict[tuple[int, float], list[dict]] = defaultdict(list)
    n_paths = len(paths)
    if verbose:
        print(f'  Found {n_paths} run dirs; loading replica timeseries (pooled)...', flush=True)
    n_loaded = 0
    for i_path, csv_path in enumerate(paths):
        dirpath = os.path.dirname(csv_path)
        for meta in read_susceptibility_csv(csv_path):
            run_id = str(meta.get("id", "")).strip()
            if not run_id:
                continue
            ts_path = os.path.join(dirpath, f"m_timeseries_{run_id}.csv")
            rec = _load_traj_arrays(ts_path, meta, tail_fraction=tail_fraction)
            if rec:
                groups[(rec["L"], rec["epsilon"])].append(rec)
                n_loaded += 1
        if verbose and ((i_path + 1) % 10 == 0 or i_path + 1 == n_paths):
            print(
                f'  ... {i_path + 1}/{n_paths} run dirs, {n_loaded} replicas loaded',
                flush=True,
            )

    if not groups:
        raise FileNotFoundError("No timeseries files found — check that runs have completed.")

    if verbose:
        print(
            f'  Pooling {n_loaded} replicas across {len(groups)} (L, ε) groups...',
            flush=True,
        )

    rows = []
    traj_rows: list[dict] = []
    for (l_val, eps), recs in groups.items():
        beta = recs[0]["beta"]
        N = recs[0]["N"]
        m_arrays = [r["m"] for r in recs]
        pooled_m = np.concatenate(m_arrays)
        for m_arr in m_arrays:
            traj_rows.append({
                "L": int(l_val),
                "epsilon": float(eps),
                "m_mean": float(np.mean(m_arr)),
                "m2_mean": float(np.mean(m_arr ** 2)),
                "m4_mean": float(np.mean(m_arr ** 4)),
            })

        chi, chi_err = _jackknife(
            m_arrays, lambda a, N=N, beta=beta: N * beta * (np.mean(a ** 2) - np.mean(np.abs(a)) ** 2)
        )
        m_mean, m_mean_err = _jackknife(m_arrays, lambda a: float(np.mean(a)))
        abs_m_mean, abs_m_mean_err = _jackknife(
            m_arrays, lambda a: float(np.mean(np.abs(a)))
        )
        u4, u4_err = _jackknife(
            m_arrays,
            lambda a: 1.0 - np.mean(a ** 4) / (3.0 * np.mean(a ** 2) ** 2)
            if np.mean(a ** 2) != 0 else float("nan"),
        )

        row: dict = {
            "L": int(l_val),
            "epsilon": float(eps),
            "beta": beta,
            "chi_mean": chi,
            "chi_stderr": chi_err,
            "m_mean": m_mean,
            "m_mean_stderr": m_mean_err,
            "abs_m_mean": abs_m_mean,
            "abs_m_mean_stderr": abs_m_mean_err,
            "m2_mean": float(np.mean(pooled_m ** 2)),
            "m4_mean": float(np.mean(pooled_m ** 4)),
            "u4": u4,
            "u4_err": u4_err,
            "n_replicas": len(recs),
            "n_samples": int(pooled_m.size),
            "chunks_per_replica": float(np.mean([a.size for a in m_arrays])),
        }

        e_arrays = [r["e_int"] for r in recs if r["e_int"] is not None]
        if e_arrays:
            # Match the per-trajectory convention: c = (⟨E²⟩-⟨E⟩²)/(N·T²·|ε|), T = 1/β.
            T = 1.0 / beta
            denom = N * T ** 2 * abs(float(eps))
            c, c_err = _jackknife(
                e_arrays, lambda a, denom=denom: (np.mean(a ** 2) - np.mean(a) ** 2) / denom
            )
            row["c_mean"] = c
            row["c_stderr"] = c_err
        else:
            row["c_mean"] = float("nan")
            row["c_stderr"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["L", "epsilon"]), pd.DataFrame(traj_rows)
