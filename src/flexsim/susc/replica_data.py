"""Load per-replica m(t) series of a susceptibility results tree, grouped by (L, epsilon)."""

from __future__ import annotations

import os

import pandas as pd

from flexsim.susc.layout import find_susceptibility_csvs, read_susceptibility_csv


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_run_dir(dirname: str) -> tuple[int | None, float | None]:
    """
    Parse L and epsilon from a run directory name without reading any file.
    e.g. susceptibility_128x128_homo_deltaFm20p0_dmu0p0_epsilonm1p935
      -> L=128, eps=-1.935
    """
    import re
    m_L = re.search(r"susceptibility_(\d+)x\d+", dirname)
    L = int(m_L.group(1)) if m_L else None
    m_e = re.search(r"epsilon(m?)(\d+)(?:p(\d+))?", dirname)
    if m_e:
        sign = -1 if m_e.group(1) == "m" else 1
        dec = m_e.group(3) or "0"
        eps = sign * float(f"{m_e.group(2)}.{dec}")
    else:
        eps = None
    return L, eps


def fast_inventory(results_dir: str) -> list[dict]:
    """
    Instant --list scan: reads NO CSV content.
    For each run dir, parses L/eps from the name, counts CSV rows via
    line-count (not CSV parsing), and counts timeseries files via glob.
    """
    import glob as _glob

    SUSCEPTIBILITY_DATA_CSV = "susceptibility_data.csv"
    records = []
    abs_dir = os.path.abspath(results_dir)
    try:
        entries = [e for e in os.scandir(abs_dir)
                   if e.is_dir() and e.name.startswith("susceptibility_")]
    except FileNotFoundError:
        return records

    for entry in entries:
        csv_path = os.path.join(entry.path, SUSCEPTIBILITY_DATA_CSV)
        if not os.path.isfile(csv_path):
            continue
        L, eps = _parse_run_dir(entry.name)
        if L is None or eps is None:
            continue
        # Count data rows (lines minus header) without parsing CSV content
        with open(csv_path, "rb") as f:
            n_lines = sum(1 for _ in f)
        n_replicas = max(0, n_lines - 1)
        # Count timeseries files present
        n_ts = len(_glob.glob(os.path.join(entry.path, "m_timeseries_*.csv")))
        records.append({"L": L, "eps": eps, "n_replicas": n_replicas, "n_ts": n_ts})

    return records


def load_replica_groups(results_dir: str) -> dict[tuple[int, float], dict]:
    """
    Returns { (L, eps): {"beta": float, "N": int, "replicas": [m_array, ...]} }
    One m_array (shape [n_chunks]) per replica row.
    """
    groups: dict[tuple[int, float], dict] = {}
    csvs = find_susceptibility_csvs(results_dir)
    if not csvs:
        raise FileNotFoundError(f"No susceptibility_data.csv found under {results_dir}")

    for csv_path in csvs:
        dirpath = os.path.dirname(csv_path)
        for meta in read_susceptibility_csv(csv_path):
            run_id = str(meta.get("id", "")).strip()
            if not run_id:
                continue
            ts_path = os.path.join(dirpath, f"m_timeseries_{run_id}.csv")
            if not os.path.isfile(ts_path):
                continue
            ts = pd.read_csv(ts_path)
            if ts.empty or "m" not in ts.columns:
                continue
            L = int(float(meta["L"]))
            eps = round(float(meta["epsilon"]), 6)
            beta = float(meta["beta"])
            key = (L, eps)
            if key not in groups:
                groups[key] = {"beta": beta, "N": L * L, "replicas": []}
            groups[key]["replicas"].append(ts["m"].to_numpy(float))

    return groups
