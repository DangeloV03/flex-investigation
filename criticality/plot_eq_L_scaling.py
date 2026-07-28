#!/usr/bin/env python3
"""Finite-size scaling plots for the multi-L equilibrium coex campaign.

Builds the two mentor plots from coex_eq/ly*/ + criticality/eq_ly*/:

  1. beta * mu_coex(epsilon_c) vs L
  2. beta * epsilon_c vs L

Usage (repo root, after coex is analyzed and criticality CSVs exist):

    python -u criticality/plot_eq_L_scaling.py
    python -u criticality/plot_eq_L_scaling.py --lys 16 20 40 \\
        --out-dir criticality/eq_multi_L

If criticality/eq_ly*/criticality.csv is missing, run first:

    ./coex/run_eq_multi_L_criticality.sh
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_LYS = (16, 20, 40)
DEFAULT_OUT = "criticality/eq_multi_L"
DEFAULT_COEX_ROOT = "coex_eq"


def _finite(x) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _mu_coex_value(row: dict) -> float | None:
    """Prefer analyzer fitted mu; fall back to legacy mu_coex_SIM."""
    for key in ("mu_coex_FITTED", "mu_coex_SIM"):
        raw = str(row.get(key, "")).strip()
        if not raw or raw.lower() == "nan":
            continue
        if _finite(raw):
            return float(raw)
    return None


def _mu_err(row: dict) -> float | None:
    for key in ("mu_coex_FITTED_error", "mu_coex_SIM_error"):
        raw = str(row.get(key, "")).strip()
        if raw and raw.lower() != "nan" and _finite(raw):
            return float(raw)
    return None


def load_manage_mu_curve(manage_csv: str) -> pd.DataFrame:
    """Return epsilon, mu_coex, mu_err, beta for analyzed rows."""
    rows = []
    with open(manage_csv, newline="") as f:
        for row in csv.DictReader(f):
            mu = _mu_coex_value(row)
            if mu is None:
                continue
            if not _finite(row.get("epsilon")):
                continue
            beta = float(row["beta"]) if _finite(row.get("beta")) else 1.0
            rows.append({
                "epsilon": float(row["epsilon"]),
                "mu_coex": mu,
                "mu_err": _mu_err(row),
                "beta": beta,
                "Lx": int(float(row["Lx"])) if _finite(row.get("Lx")) else None,
                "Ly": int(float(row["Ly"])) if _finite(row.get("Ly")) else None,
            })
    if not rows:
        return pd.DataFrame(columns=["epsilon", "mu_coex", "mu_err", "beta", "Lx", "Ly"])
    df = pd.DataFrame(rows).sort_values("epsilon").drop_duplicates("epsilon", keep="last")
    return df.reset_index(drop=True)


def load_criticality(crit_csv: str) -> dict:
    with open(crit_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty criticality CSV: {crit_csv}")
    return rows[0]


def interpolate_mu_at_eps(curve: pd.DataFrame, epsilon_c: float) -> tuple[float, float | None]:
    """Linear interpolate mu_coex(epsilon_c); clamp to nearest endpoint if outside."""
    if curve.empty:
        raise ValueError("no mu_coex points to interpolate")
    eps = curve["epsilon"].to_numpy(float)
    mu = curve["mu_coex"].to_numpy(float)
    if epsilon_c <= eps.min():
        i = int(np.argmin(eps))
        return float(mu[i]), _optional_err(curve, i)
    if epsilon_c >= eps.max():
        i = int(np.argmax(eps))
        return float(mu[i]), _optional_err(curve, i)
    mu_c = float(np.interp(epsilon_c, eps, mu))
    # error: average of neighboring analyzed errors if present
    i_hi = int(np.searchsorted(eps, epsilon_c, side="left"))
    i_lo = max(i_hi - 1, 0)
    i_hi = min(i_hi, len(eps) - 1)
    errs = [e for e in (_optional_err(curve, i_lo), _optional_err(curve, i_hi)) if e is not None]
    err = float(np.mean(errs)) if errs else None
    return mu_c, err


def _optional_err(curve: pd.DataFrame, i: int) -> float | None:
    raw = curve.iloc[i].get("mu_err")
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    if _finite(raw):
        return float(raw)
    return None


def collect_scaling_table(
    lys: list[int],
    *,
    coex_root: str = DEFAULT_COEX_ROOT,
    crit_root: str = "criticality",
) -> pd.DataFrame:
    """One row per L: epsilon_c, mu_coex(at eps_c), beta-scaled columns."""
    out_rows = []
    for ly in lys:
        manage = os.path.join(coex_root, f"ly{ly}", "manage.csv")
        crit_csv = os.path.join(crit_root, f"eq_ly{ly}", "criticality.csv")
        if not os.path.isfile(manage):
            print(f"[skip] missing {manage}", flush=True)
            continue
        if not os.path.isfile(crit_csv):
            print(
                f"[skip] missing {crit_csv} — run ./coex/run_eq_multi_L_criticality.sh {ly}",
                flush=True,
            )
            continue

        curve = load_manage_mu_curve(manage)
        if curve.empty:
            print(f"[skip] Ly={ly}: no analyzed mu_coex in {manage}", flush=True)
            continue

        crit = load_criticality(crit_csv)
        eps_c = float(crit["epsilon_c_estimate"])
        beta = float(crit["beta"]) if _finite(crit.get("beta")) else float(curve["beta"].iloc[0])
        L_short = int(float(crit.get("L_short", ly)))
        L_long = int(float(crit.get("L_long", 10 * ly)))

        mu_c, mu_err = interpolate_mu_at_eps(curve, eps_c)
        fit_unc = float(crit["fit_uncertainty"]) if _finite(crit.get("fit_uncertainty")) else None
        rec_unc = (
            float(crit["recommended_uncertainty"])
            if _finite(crit.get("recommended_uncertainty"))
            else fit_unc
        )

        out_rows.append({
            "L_short": L_short,
            "L_long": L_long,
            "beta": beta,
            "epsilon_c": eps_c,
            "beta_epsilon_c": beta * eps_c,
            "epsilon_c_uncertainty": rec_unc,
            "mu_coex_at_eps_c": mu_c,
            "beta_mu_coex_at_eps_c": beta * mu_c,
            "mu_coex_uncertainty": mu_err,
            "n_mu_eps_points": int(len(curve)),
            "manage_csv": manage,
            "criticality_csv": crit_csv,
        })
        print(
            f"[eq-L] Ly={L_short}: eps_c={eps_c:.5f}  "
            f"mu_coex(eps_c)={mu_c:.5f}  n_eps={len(curve)}",
            flush=True,
        )

    if not out_rows:
        raise SystemExit(
            "No L rows collected. Need analyzed manage.csv + criticality/eq_ly*/criticality.csv."
        )
    return pd.DataFrame(out_rows).sort_values("L_short").reset_index(drop=True)


def plot_vs_L(
    df: pd.DataFrame,
    *,
    y_col: str,
    yerr_col: str | None,
    ylabel: str,
    title: str,
    out_png: str,
) -> str:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    x = df["L_short"].to_numpy(float)
    y = df[y_col].to_numpy(float)
    yerr = None
    if yerr_col and yerr_col in df.columns:
        err = pd.to_numeric(df[yerr_col], errors="coerce")
        if err.notna().any():
            yerr = err.to_numpy(float)

    ax.errorbar(x, y, yerr=yerr, fmt="o-", color="#2F4A7A", markersize=8,
                capsize=3, linewidth=1.5)
    ax.set_xlabel(r"$L$ (short axis)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eq-L] wrote {out_png}", flush=True)
    return out_png


def main() -> None:
    p = argparse.ArgumentParser(description="Plot βμ_coex(ε_c) and βε_c vs L")
    p.add_argument("--lys", type=int, nargs="+", default=list(DEFAULT_LYS))
    p.add_argument("--coex-root", default=DEFAULT_COEX_ROOT)
    p.add_argument("--crit-root", default="criticality")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_scaling_table(
        args.lys, coex_root=args.coex_root, crit_root=args.crit_root,
    )
    csv_path = out_dir / "eq_scaling_vs_L.csv"
    df.to_csv(csv_path, index=False)
    print(f"[eq-L] wrote {csv_path}", flush=True)

    plot_vs_L(
        df,
        y_col="beta_mu_coex_at_eps_c",
        yerr_col="mu_coex_uncertainty",
        ylabel=r"$\beta\mu_{\mathrm{coex}}(\varepsilon_c)$",
        title=r"Equilibrium coexistence: $\beta\mu_{\mathrm{coex}}$ at $\varepsilon_c$ vs $L$",
        out_png=str(out_dir / "beta_mu_coex_vs_L.png"),
    )
    plot_vs_L(
        df,
        y_col="beta_epsilon_c",
        yerr_col="epsilon_c_uncertainty",
        ylabel=r"$\beta\varepsilon_c$",
        title=r"Equilibrium coexistence: $\beta\varepsilon_c$ vs $L$",
        out_png=str(out_dir / "beta_eps_c_vs_L.png"),
    )


if __name__ == "__main__":
    main()
