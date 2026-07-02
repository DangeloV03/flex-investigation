"""
analyze_chi_max_scaling.py

Two analyses for the FSS scaling chi_max(L) ~ A * L^(gamma/nu).

Reference: Kumar & Dasgupta (2020) PRE 102, 052111 — Table I, E_0 = -2:
    gamma/nu = 1.73 +/- 0.01,  A = 0.095

1. REPLICA SUBSAMPLING
   For n in REP_COUNTS, draw n replicas at each (L, eps), pool their m-samples,
   compute chi = N*beta*(mean(m^2) - mean(|m|)^2), take chi_max over eps, and
   fit the power law. Repeated N_BOOT times to get bootstrap distributions.
   Produces:
     chi_max_vs_L_by_nrep.png  — log-log curves for each n, with K&D reference
     gamma_nu_vs_nrep.png      — fitted gamma/nu +/- bootstrap std vs n_rep

2. LEAVE-ONE-L-OUT (LOO) JACKKNIFE
   Using all replicas, compute chi_max(L) at each L, fit gamma/nu, then
   refit dropping each L in turn. Reports gamma/nu +/- jackknife error and
   compares to K&D within combined uncertainty.
   Produces:
     chi_max_vs_L_full.png     — log-log with fit, K&D line, LOO shaded band
     loo_gamma_nu_per_L.png    — gamma/nu(L_dropped) vs L
     loo_jackknife_results.txt — numerical summary

Usage:
    python analyze_chi_max_scaling.py
    python analyze_chi_max_scaling.py --results susceptibility_results/exact \\
        --outdir plots/chi_max_scaling --n-boot 500
    python analyze_chi_max_scaling.py --loo-only   # skip slow bootstrap
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from susceptibility_paths import find_susceptibility_csvs, read_susceptibility_csv

# Kumar & Dasgupta (2020), Table I, E_0 = -2
KD_GAMMA_NU = 1.73
KD_GAMMA_NU_ERR = 0.01
KD_A = 0.095

DEFAULT_REP_COUNTS = [8, 16, 32, 64, 96]
N_BOOT = 300

NREP_PALETTE = {
    8:  "#d73027",
    16: "#fc8d59",
    32: "#f4a71e",
    64: "#91bfdb",
    96: "#4575b4",
}

L_COLOR = {
    16: "black", 32: "red", 48: "#2ca02c", 64: "blue",
    96: "cyan", 128: "saddlebrown", 256: "orange",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def chi_from_pooled(m_arrays: list[np.ndarray], N: int, beta: float) -> float:
    """Connected susceptibility from pooled replica samples."""
    pooled = np.concatenate(m_arrays)
    return N * beta * (float(np.mean(pooled ** 2)) - float(np.mean(np.abs(pooled))) ** 2)


def chi_max_for_L(
    groups: dict,
    L: int,
    eps_list: list[float],
    replica_indices: list[int] | None = None,
) -> float:
    """
    chi_max(L): max over all eps of chi computed from the selected replica indices.
    If replica_indices is None, use all replicas.
    """
    chi_vals = []
    for eps in eps_list:
        key = (L, eps)
        if key not in groups:
            continue
        data = groups[key]
        reps = data["replicas"]
        selected = [reps[i] for i in replica_indices] if replica_indices is not None else reps
        if not selected:
            continue
        chi_vals.append(chi_from_pooled(selected, data["N"], data["beta"]))
    return float(np.max(chi_vals)) if chi_vals else float("nan")


def fit_power_law(L_arr: np.ndarray, chi_arr: np.ndarray) -> tuple[float, float]:
    """OLS: log(chi) = log(A) + (gamma/nu)*log(L). Returns (gamma_nu, A)."""
    slope, intercept = np.polyfit(np.log(L_arr), np.log(chi_arr), 1)
    return float(slope), float(np.exp(intercept))


# ---------------------------------------------------------------------------
# Analysis 1: replica subsampling
# ---------------------------------------------------------------------------

def subsampling_analysis(
    groups: dict,
    Ls: list[int],
    eps_per_L: dict[int, list[float]],
    rep_counts: list[int],
    n_boot: int,
    outdir: str,
) -> None:
    rng = np.random.default_rng(42)

    # How many replicas are available at each (L, eps)?
    n_avail = {
        L: min(len(groups[(L, eps)]["replicas"]) for eps in eps_per_L[L] if (L, eps) in groups)
        for L in Ls
    }
    rep_counts = [n for n in rep_counts if n <= min(n_avail.values())]
    if not rep_counts:
        print("  Not enough replicas for subsampling — skipping.")
        return

    # Bootstrap: for each (n_rep, draw) -> chi_max per L -> gamma/nu
    # Shape: chi_boot[n_rep][L] = array of n_boot chi_max values
    chi_boot: dict[int, dict[int, list[float]]] = {n: {L: [] for L in Ls} for n in rep_counts}
    gnu_boot: dict[int, list[float]] = {n: [] for n in rep_counts}

    for n_rep in rep_counts:
        print(f"  n_rep={n_rep}: {n_boot} bootstrap draws ...", flush=True)
        for _ in range(n_boot):
            chi_max_draw: dict[int, float] = {}
            for L in Ls:
                n = n_avail[L]
                idx = rng.choice(n, n_rep, replace=False).tolist()
                v = chi_max_for_L(groups, L, eps_per_L[L], idx)
                chi_max_draw[L] = v

            valid = [(L, chi_max_draw[L]) for L in Ls
                     if np.isfinite(chi_max_draw[L]) and chi_max_draw[L] > 0]
            for L in Ls:
                chi_boot[n_rep][L].append(chi_max_draw[L])
            if len(valid) >= 3:
                L_arr = np.array([v[0] for v in valid], float)
                c_arr = np.array([v[1] for v in valid], float)
                gnu, _ = fit_power_law(L_arr, c_arr)
                gnu_boot[n_rep].append(gnu)

    # --- Plot 1a: chi_max vs L for each n_rep ---
    fig, ax = plt.subplots(figsize=(7, 5))
    L_fine = np.geomspace(min(Ls), max(Ls), 200)

    for n_rep in rep_counts:
        color = NREP_PALETTE.get(n_rep, "gray")
        L_arr = np.array(Ls, float)
        means = np.array([np.nanmean(chi_boot[n_rep][L]) for L in Ls])
        stds  = np.array([np.nanstd(chi_boot[n_rep][L]) for L in Ls])
        ax.errorbar(L_arr, means, yerr=stds, fmt="o-", color=color,
                    label=f"n={n_rep}", capsize=3, markersize=5)

    ax.loglog(L_fine, KD_A * L_fine ** KD_GAMMA_NU, "k--", linewidth=1.5,
              label=rf"K&D: $A={KD_A}$, $\gamma/\nu={KD_GAMMA_NU}$")
    ax.set_xlabel("$L$", fontsize=12)
    ax.set_ylabel(r"$\chi^{\rm max}(L)$", fontsize=12)
    ax.set_title(r"$\chi^{\rm max}$ vs $L$ — replica subsampling")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(outdir, "chi_max_vs_L_by_nrep.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Wrote {path}")

    # --- Plot 1b: fitted gamma/nu vs n_rep ---
    gnu_means = [np.nanmean(gnu_boot[n]) for n in rep_counts if gnu_boot[n]]
    gnu_stds  = [np.nanstd(gnu_boot[n])  for n in rep_counts if gnu_boot[n]]
    valid_n   = [n for n in rep_counts if gnu_boot[n]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(valid_n, gnu_means, yerr=gnu_stds, fmt="o-", color="navy",
                capsize=4, markersize=6, label="bootstrap mean ± std")
    ax.axhline(KD_GAMMA_NU, color="red", linestyle="--", linewidth=1.5,
               label=rf"K&D: $\gamma/\nu = {KD_GAMMA_NU} \pm {KD_GAMMA_NU_ERR}$")
    ax.fill_between(
        [0, max(rep_counts) + 8],
        KD_GAMMA_NU - KD_GAMMA_NU_ERR,
        KD_GAMMA_NU + KD_GAMMA_NU_ERR,
        color="red", alpha=0.15,
    )
    ax.set_xlabel("Number of replicas", fontsize=11)
    ax.set_ylabel(r"Fitted $\gamma/\nu$", fontsize=11)
    ax.set_title(r"Convergence of $\gamma/\nu$ with replica count")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(rep_counts) + 8)
    path = os.path.join(outdir, "gamma_nu_vs_nrep.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Wrote {path}")

    print("\n=== Subsampling summary (gamma/nu) ===")
    print(f"{'n_rep':>6}  {'mean':>8}  {'std':>8}  {'n_draws':>8}")
    for n_rep, mean, std in zip(valid_n, gnu_means, gnu_stds):
        print(f"{n_rep:>6}  {mean:>8.4f}  {std:>8.4f}  {len(gnu_boot[n_rep]):>8}")
    print(f"{'K&D':>6}  {KD_GAMMA_NU:>8.2f}  {KD_GAMMA_NU_ERR:>8.2f}")


# ---------------------------------------------------------------------------
# Analysis 2: leave-one-L-out jackknife
# ---------------------------------------------------------------------------

def loo_jackknife_analysis(
    groups: dict,
    Ls: list[int],
    eps_per_L: dict[int, list[float]],
    outdir: str,
) -> None:

    # chi_max per L using ALL replicas, plus replica-jackknife error bars
    chi_max_full: dict[int, float] = {}
    chi_max_err: dict[int, float] = {}

    for L in Ls:
        reps_per_eps = {eps: groups[(L, eps)]["replicas"]
                        for eps in eps_per_L[L] if (L, eps) in groups}
        if not reps_per_eps:
            continue

        # Full chi_max
        N = groups[(L, eps_per_L[L][0])]["N"]
        beta = groups[(L, eps_per_L[L][0])]["beta"]
        n_reps = min(len(reps_per_eps[eps]) for eps in reps_per_eps)

        def _chi_max_subset(idx: list[int]) -> float:
            vals = []
            for reps in reps_per_eps.values():
                selected = [reps[i] for i in idx]
                vals.append(chi_from_pooled(selected, N, beta))
            return float(np.max(vals))

        full_idx = list(range(n_reps))
        chi_max_full[L] = _chi_max_subset(full_idx)

        # Replica LOO for error on chi_max
        loo_vals = [_chi_max_subset([j for j in full_idx if j != i]) for i in full_idx]
        loo_arr = np.array(loo_vals)
        chi_max_err[L] = float(
            np.sqrt((n_reps - 1) / n_reps * np.sum((loo_arr - loo_arr.mean()) ** 2))
        )

    Ls_fit = sorted(chi_max_full)
    L_arr  = np.array(Ls_fit, float)
    chi_arr = np.array([chi_max_full[L] for L in Ls_fit])
    err_arr = np.array([chi_max_err[L]  for L in Ls_fit])

    # Full power-law fit
    gnu_full, A_full = fit_power_law(L_arr, chi_arr)

    # LOO over L values
    n = len(Ls_fit)
    gnu_loo = np.zeros(n)
    A_loo   = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        gnu_loo[i], A_loo[i] = fit_power_law(L_arr[mask], chi_arr[mask])

    gnu_loo_mean = gnu_loo.mean()
    gnu_loo_err  = float(
        np.sqrt((n - 1) / n * np.sum((gnu_loo - gnu_loo_mean) ** 2))
    )

    sigma_diff = abs(gnu_full - KD_GAMMA_NU) / np.sqrt(gnu_loo_err**2 + KD_GAMMA_NU_ERR**2)

    # --- Print & save text results ---
    lines = [
        "=== LOO Jackknife: gamma/nu for chi_max ~ A * L^(gamma/nu) ===",
        f"",
        f"Full fit:    gamma/nu = {gnu_full:.4f},  A = {A_full:.4f}",
        f"LOO error:   gamma/nu = {gnu_full:.4f} +/- {gnu_loo_err:.4f}",
        f"K&D (2020):  gamma/nu = {KD_GAMMA_NU} +/- {KD_GAMMA_NU_ERR}",
        f"",
        f"Difference:  {gnu_full - KD_GAMMA_NU:+.4f}  ({sigma_diff:.2f} sigma combined)",
        f"",
        f"Per-L LOO fits (gamma/nu when that L is excluded):",
        f"{'L_dropped':>10}  {'gamma/nu_LOO':>13}  {'A_LOO':>10}",
        *[f"{int(Ls_fit[i]):>10}  {gnu_loo[i]:>13.4f}  {A_loo[i]:>10.4f}"
          for i in range(n)],
    ]
    for line in lines:
        print(line)

    txt_path = os.path.join(outdir, "loo_jackknife_results.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {txt_path}")

    # --- Plot 2a: chi_max vs L with fits and K&D reference ---
    L_fine = np.geomspace(L_arr.min(), L_arr.max(), 300)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(L_arr, chi_arr, yerr=err_arr, fmt="o", color="black",
                markersize=7, capsize=4, zorder=5, label="simulation (96 replicas)")
    ax.loglog(L_fine, A_full * L_fine ** gnu_full, "-", color="blue", linewidth=2.0,
              label=(rf"fit: $A={A_full:.3f}$, "
                     rf"$\gamma/\nu={gnu_full:.3f}\pm{gnu_loo_err:.3f}$"))
    ax.loglog(L_fine, KD_A * L_fine ** KD_GAMMA_NU, "--", color="red", linewidth=1.8,
              label=rf"K&D: $A={KD_A}$, $\gamma/\nu={KD_GAMMA_NU}\pm{KD_GAMMA_NU_ERR}$")

    # Shade LOO range (min to max gamma/nu LOO, anchored to full A)
    ax.fill_between(
        L_fine,
        A_full * L_fine ** gnu_loo.min(),
        A_full * L_fine ** gnu_loo.max(),
        color="blue", alpha=0.12, label="LOO range",
    )

    ax.set_xlabel("$L$", fontsize=12)
    ax.set_ylabel(r"$\chi^{\rm max}(L)$", fontsize=12)
    ax.set_title(r"Peak susceptibility vs $L$ — LOO jackknife")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(outdir, "chi_max_vs_L_full.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Wrote {path}")

    # --- Plot 2b: gamma/nu per dropped L ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Ls_fit, gnu_loo, "o-", color="navy", markersize=7, zorder=4,
            label=r"$\gamma/\nu$ with $L$ dropped")
    ax.axhline(gnu_full, color="blue", linewidth=1.5, linestyle="-",
               label=rf"Full fit: {gnu_full:.4f}")
    ax.fill_between(
        [Ls_fit[0] * 0.7, Ls_fit[-1] * 1.4],
        gnu_full - gnu_loo_err, gnu_full + gnu_loo_err,
        color="blue", alpha=0.15,
    )
    ax.axhline(KD_GAMMA_NU, color="red", linewidth=1.5, linestyle="--",
               label=rf"K&D: {KD_GAMMA_NU} $\pm$ {KD_GAMMA_NU_ERR}")
    ax.fill_between(
        [Ls_fit[0] * 0.7, Ls_fit[-1] * 1.4],
        KD_GAMMA_NU - KD_GAMMA_NU_ERR,
        KD_GAMMA_NU + KD_GAMMA_NU_ERR,
        color="red", alpha=0.15,
    )
    ax.set_xscale("log")
    ax.set_xlabel("$L$ dropped", fontsize=11)
    ax.set_ylabel(r"$\gamma/\nu$ (LOO fit)", fontsize=11)
    ax.set_title(r"LOO jackknife: $\gamma/\nu$ sensitivity to each $L$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(Ls_fit[0] * 0.7, Ls_fit[-1] * 1.4)
    path = os.path.join(outdir, "loo_gamma_nu_per_L.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="chi_max FSS: replica subsampling + LOO jackknife"
    )
    parser.add_argument("--results", default="susceptibility_results/exact",
                        help="Results directory (the exact phase)")
    parser.add_argument("--outdir", default="plots/chi_max_scaling")
    parser.add_argument("--rep-counts", type=int, nargs="+", default=DEFAULT_REP_COUNTS,
                        help="Replica counts to test (default: 8 16 32 64 96)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help=f"Bootstrap draws per (n_rep, L) (default: {N_BOOT})")
    parser.add_argument("--loo-only", action="store_true",
                        help="Skip subsampling; only run LOO jackknife (fast)")
    parser.add_argument("--subsample-only", action="store_true",
                        help="Skip LOO; only run subsampling")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading replicas from '{args.results}' ...", flush=True)
    groups = load_replica_groups(args.results)

    if not groups:
        raise SystemExit("No data loaded — check --results path.")

    Ls = sorted({L for (L, _) in groups})
    eps_per_L = {
        L: sorted({eps for (l, eps) in groups if l == L})
        for L in Ls
    }
    n_avail_min = min(
        min(len(groups[(L, eps)]["replicas"]) for eps in eps_per_L[L] if (L, eps) in groups)
        for L in Ls
    )

    print(f"L values: {Ls}")
    print(f"Eps points per L: {[len(eps_per_L[L]) for L in Ls]}")
    print(f"Replicas available (min over all L, eps): {n_avail_min}")
    print(f"K&D reference: gamma/nu = {KD_GAMMA_NU} +/- {KD_GAMMA_NU_ERR},  A = {KD_A}")

    if not args.loo_only:
        rep_counts = [n for n in args.rep_counts if n <= n_avail_min]
        print(f"\n--- Replica subsampling (n_boot={args.n_boot}, "
              f"rep_counts={rep_counts}) ---", flush=True)
        subsampling_analysis(groups, Ls, eps_per_L, rep_counts, args.n_boot, args.outdir)

    if not args.subsample_only:
        print("\n--- LOO jackknife (all replicas) ---", flush=True)
        loo_jackknife_analysis(groups, Ls, eps_per_L, args.outdir)


if __name__ == "__main__":
    main()
