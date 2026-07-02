"""
diagnose_distributions.py

Distinguish a first-order coexistence line from a continuous critical point, and
expose how much fluctuation the per-trajectory χ/c estimator throws away.

At a fixed ε (default the χ peak, ε≈-1.73), for each L this script:
  1. Pools every m and E_int sample across all replicas AND all chunks.
  2. Histograms P(m) and P(E_int) per L  → bimodal gap that stays open as L grows
     ⇒ first order (latent heat); single peak narrowing ⇒ continuous.
  3. Prints, per L, the POOLED χ and c (one variance over the full pooled sample)
     beside the PER-TRAJECTORY-AVERAGED χ and c (what the current pipeline reports).
     A large pooled/per-traj gap means the estimator is discarding the between-phase
     variance that should diverge with L.

Energy recovery matches plot_susceptibility.py exactly:
  e_int = e_total − e_chem,  e_chem = −β·μ·N·ρ_B − β·(μ+Δf)·N·ρ_I

Usage:
    python diagnose_distributions.py
    python diagnose_distributions.py --results susceptibility_results --epsilon -1.73
    python diagnose_distributions.py --list           # show available ε and L
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

L_COLOR: dict[int, str] = {
    16: "black", 32: "red", 48: "#2ca02c", 64: "blue",
    96: "cyan", 128: "saddlebrown", 256: "orange",
}


def sarle_bimodality(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient (finite-sample). BC > 5/9 ≈ 0.555 ⇒ bimodal."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return float("nan")
    s = x.std()
    if s == 0:
        return float("nan")
    z = (x - x.mean()) / s
    g1 = float(np.mean(z ** 3))          # skewness
    g2 = float(np.mean(z ** 4) - 3.0)    # excess kurtosis
    den = g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (g1 ** 2 + 1.0) / den if den != 0 else float("nan")


def largest_gap_split(e: np.ndarray) -> dict:
    """Crude 2-phase split at the widest gap in sorted samples → latent-heat estimate."""
    e = np.sort(np.asarray(e, dtype=float))
    if e.size < 2:
        return {"delta": 0.0, "frac_low": float("nan"), "mean_low": float("nan"), "mean_high": float("nan")}
    gaps = np.diff(e)
    k = int(np.argmax(gaps))
    low, high = e[: k + 1], e[k + 1:]
    return {
        "delta": float(high.mean() - low.mean()),
        "frac_low": float(low.size / e.size),
        "mean_low": float(low.mean()),
        "mean_high": float(high.mean()),
    }


def integrated_autocorr_time(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time τ (in sample units), automatic windowing.

    τ = 1 + 2 Σ_k ρ(k), window W chosen as smallest W ≥ c·τ (Sokal). Effective
    independent samples = n / τ. A nearly-frozen series (no fluctuation) → nan.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        return float("nan")
    x = x - x.mean()
    c0 = float(np.dot(x, x) / n)
    if c0 <= 0:
        return float("nan")  # constant series → frozen, τ undefined
    tau = 1.0
    for k in range(1, n):
        rho = float(np.dot(x[: n - k], x[k:]) / n) / c0
        tau += 2.0 * rho
        if k >= c * tau:
            break
    return max(tau, 1.0)


def _recover_eint(ts: pd.DataFrame, beta: float, mu: float, delta_f: float, N: int) -> np.ndarray | None:
    if not {"energy", "rho_bonding", "rho_inert"}.issubset(ts.columns):
        return None
    rho_B = ts["rho_bonding"].to_numpy(float)
    rho_I = ts["rho_inert"].to_numpy(float)
    e_total = ts["energy"].to_numpy(float)
    e_chem = -beta * mu * N * rho_B - beta * (mu + delta_f) * N * rho_I
    return e_total - e_chem


def collect(results_dir: str) -> pd.DataFrame:
    """One row per replica: L, epsilon, beta, N, and the per-chunk m / E_int arrays."""
    paths = find_susceptibility_csvs(results_dir)
    if not paths:
        raise FileNotFoundError(f"No susceptibility_data.csv under {results_dir}")

    records: list[dict] = []
    for csv_path in paths:
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
            beta = float(meta["beta"])
            mu = float(meta["mu"])
            delta_f = float(meta["delta_f"])
            L = int(float(meta["L"]))
            N = L * L
            try:
                pt = float(meta.get("prod_time", "") or "nan")
                pc = float(meta.get("prod_chunks", "") or "nan")
                steps_per_sample = pt / pc if pc else float("nan")
            except (TypeError, ValueError):
                steps_per_sample = float("nan")
            e_int = _recover_eint(ts, beta, mu, delta_f, N)
            records.append({
                "L": L,
                "epsilon": float(meta["epsilon"]),
                "beta": beta,
                "N": N,
                "steps_per_sample": steps_per_sample,
                "m": ts["m"].to_numpy(float),
                "e_int": e_int,
            })
    if not records:
        raise FileNotFoundError("No timeseries files found — check that runs have completed.")
    return pd.DataFrame(records)


def pick_epsilon(df: pd.DataFrame, target: float) -> float:
    eps_values = np.sort(df["epsilon"].unique())
    return float(eps_values[int(np.argmin(np.abs(eps_values - target)))])


def analyze(df: pd.DataFrame, eps: float, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    sub = df[np.isclose(df["epsilon"], eps)]
    Ls = sorted(sub["L"].unique())
    if not Ls:
        raise ValueError(f"No data at ε={eps}")

    print(f"\n=== Distribution diagnostic at ε = {eps:.4f} ===")
    header = (
        f"{'L':>5} {'n_samp':>7} {'⟨m⟩':>8} {'BC(m)':>7} "
        f"{'χ_pool':>10} {'χ_pertraj':>10} {'ratio':>6} "
        f"{'BC(E)':>7} {'c_pool':>9} {'c_pertraj':>9} {'ΔE':>9} {'ΔE/N':>7}"
    )
    print(header)
    print("-" * len(header))

    fig, axes = plt.subplots(len(Ls), 2, figsize=(11, 2.1 * len(Ls)), squeeze=False)
    summary_rows: list[dict] = []

    for i, L in enumerate(Ls):
        rows = sub[sub["L"] == L]
        beta = float(rows["beta"].iloc[0])
        N = int(rows["N"].iloc[0])

        m_pool = np.concatenate(list(rows["m"]))
        chi_pool = N * beta * float(m_pool.var())            # one variance over pooled m
        chi_pertraj = float(np.mean([N * beta * mm.var() for mm in rows["m"]]))
        chi_ratio = chi_pool / chi_pertraj if chi_pertraj else float("nan")
        bc_m = sarle_bimodality(m_pool)

        e_list = [e for e in rows["e_int"] if e is not None]
        if e_list:
            e_pool = np.concatenate(e_list)
            c_pool = float(e_pool.var()) / N
            c_pertraj = float(np.mean([ee.var() / N for ee in e_list]))
            bc_e = sarle_bimodality(e_pool)
            gap = largest_gap_split(e_pool)
            delta, delta_n = gap["delta"], gap["delta"] / N
        else:
            e_pool = None
            c_pool = c_pertraj = bc_e = delta = delta_n = float("nan")

        print(
            f"{L:>5} {m_pool.size:>7} {m_pool.mean():>8.4f} {bc_m:>7.3f} "
            f"{chi_pool:>10.2f} {chi_pertraj:>10.2f} {chi_ratio:>6.1f} "
            f"{bc_e:>7.3f} {c_pool:>9.2f} {c_pertraj:>9.2f} {delta:>9.2f} {delta_n:>7.3f}"
        )
        summary_rows.append({
            "L": L, "epsilon": eps, "n_samples": int(m_pool.size),
            "m_mean": float(m_pool.mean()), "bc_m": bc_m,
            "chi_pooled": chi_pool, "chi_pertraj": chi_pertraj,
            "bc_e": bc_e, "c_pooled": c_pool, "c_pertraj": c_pertraj,
            "latent_dE": delta, "latent_dE_per_N": delta_n,
        })

        color = L_COLOR.get(L, "gray")
        ax_m, ax_e = axes[i][0], axes[i][1]
        m_mean_pool = float(m_pool.mean())
        ax_m.hist(m_pool, bins=40, color=color, alpha=0.75, density=True)
        ax_m.axvline(m_mean_pool, color="k", linewidth=1.2, linestyle="--")
        ax_m.set_ylabel(f"L={L}", fontsize=9)
        ax_m.set_title(f"P(m)  BC={bc_m:.2f}  ⟨m⟩={m_mean_pool:.3f}", fontsize=8)
        if e_pool is not None:
            ax_e.hist(e_pool, bins=40, color=color, alpha=0.75, density=True)
            ax_e.set_title(f"P(E_int)  BC={bc_e:.2f}", fontsize=8)
        for ax in (ax_m, ax_e):
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("m")
    axes[-1][1].set_xlabel("E_int")
    fig.suptitle(f"Pooled distributions at ε={eps:.3f}   (BC>0.56 ⇒ bimodal)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    hist_path = os.path.join(outdir, f"dist_eps{abs(eps):.3f}".replace(".", "p") + ".png")
    fig.savefig(hist_path, dpi=140)
    plt.close(fig)
    print(f"\nWrote {hist_path}")

    smry = pd.DataFrame(summary_rows)
    csv_path = os.path.join(outdir, f"dist_eps{abs(eps):.3f}".replace(".", "p") + ".csv")
    smry.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    _scaling_plot(smry, eps, outdir)
    _autocorr_report(sub, eps, outdir)
    _heat_capacity_report(sub, eps, outdir)

    print(
        "\nRead it like this:\n"
        "  • BC(E) > 0.56 and ΔE/N roughly constant across L  ⇒ first order (latent heat).\n"
        "  • χ_pool ≫ χ_pertraj (ratio ≫ 1)                   ⇒ per-traj estimator is\n"
        "    discarding the between-phase variance that should scale as L^1.75.\n"
        "  • Single-peaked P(m) narrowing with L, BC small     ⇒ continuous; trust pooled χ/c.\n"
        "  • Autocorr: if Neff_tot collapses toward ~1 as L grows, large-L is under-sampled\n"
        "    (critical slowing down) — the run length, not the snapshot count, is the limit.\n"
        "  • Heat cap: if |corr(E,m)|→1 and c_resid ≪ c_raw, c is contaminated by the\n"
        "    magnetization fluctuation (energy–order-parameter mixing); c_resid is the fix."
    )


def _heat_capacity_report(sub: pd.DataFrame, eps: float, outdir: str) -> None:
    """Test whether c is inflated by E_int–m mixing (energy tracking the order parameter).

    c_raw    = Var(E_int)/N                              (current definition)
    c_resid  = [Var(E_int) − Cov(E_int,m)²/Var(m)] / N   (m-linear part projected out)
    If E_int ≈ E_sym + a·N·m, c_resid recovers Var(E_sym)/N ≈ the true specific heat.
    Done per trajectory then averaged, matching the production estimator.
    """
    Ls = sorted(sub["L"].unique())
    print(f"\n=== Heat capacity: energy–order-parameter mixing check at ε = {eps:.4f} ===")
    header = f"{'L':>5} {'corr(E,m)':>10} {'c_raw':>10} {'c_resid':>10} {'drop':>7}"
    print(header)
    print("-" * len(header))
    for L in Ls:
        recs = sub[sub["L"] == L]
        N = int(recs["N"].iloc[0])
        c_raw_l, c_resid_l, corr_l = [], [], []
        for _, r in recs.iterrows():
            e, m = r["e_int"], r["m"]
            if e is None:
                continue
            em = e - e.mean()
            mm = m - m.mean()
            var_e = float(np.mean(em ** 2))
            var_m = float(np.mean(mm ** 2))
            cov = float(np.mean(em * mm))
            if var_e <= 0 or var_m <= 0:
                continue
            c_raw_l.append(var_e / N)
            c_resid_l.append((var_e - cov ** 2 / var_m) / N)
            corr_l.append(cov / np.sqrt(var_e * var_m))
        if not c_raw_l:
            continue
        c_raw, c_resid, corr = np.mean(c_raw_l), np.mean(c_resid_l), np.mean(corr_l)
        drop = c_resid / c_raw if c_raw else float("nan")
        print(f"{L:>5} {corr:>10.3f} {c_raw:>10.2f} {c_resid:>10.3f} {drop:>7.3f}")


def _autocorr_report(sub: pd.DataFrame, eps: float, outdir: str) -> None:
    """Per-L integrated autocorrelation time of m and effective independent sample size."""
    Ls = sorted(sub["L"].unique())
    print(f"\n=== Autocorrelation / effective sample size at ε = {eps:.4f} ===")
    header = (
        f"{'L':>5} {'n/rep':>7} {'reps':>5} {'τ(samp)':>9} {'τ(steps)':>11} "
        f"{'Neff/rep':>9} {'Neff_tot':>9} {'frozen':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for L in Ls:
        recs = sub[sub["L"] == L]
        taus, n_per, sps, n_frozen = [], None, float("nan"), 0
        for _, r in recs.iterrows():
            m = r["m"]
            n_per = m.size
            sps = r.get("steps_per_sample", float("nan"))
            t = integrated_autocorr_time(m)
            if np.isnan(t):
                n_frozen += 1
            else:
                taus.append(t)
        if taus:
            tau = float(np.mean(taus))
            neff_rep = n_per / tau
            neff_tot = neff_rep * len(taus)
        else:
            tau = neff_rep = float("nan")
            neff_tot = 0.0
        tau_steps = tau * sps if sps == sps else float("nan")
        print(
            f"{L:>5} {n_per:>7} {len(recs):>5} {tau:>9.1f} {tau_steps:>11.0f} "
            f"{neff_rep:>9.1f} {neff_tot:>9.1f} {f'{n_frozen}/{len(recs)}':>8}"
        )
        rows.append({"L": L, "tau_samples": tau, "tau_steps": tau_steps,
                     "neff_per_rep": neff_rep, "neff_total": neff_tot})

    s = pd.DataFrame(rows).sort_values("L")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.loglog(s["L"], s["tau_samples"], "o-", color="tab:purple")
    ax1.set_xlabel("L"); ax1.set_ylabel(r"$\tau_{int}$ (samples)")
    ax1.set_title("Autocorrelation time"); ax1.grid(True, which="both", alpha=0.3)
    ax2.loglog(s["L"], s["neff_total"], "o-", color="tab:green")
    ax2.axhline(1.0, color="r", ls="--", lw=0.8, label="1 independent config")
    ax2.set_xlabel("L"); ax2.set_ylabel(r"$N_{eff}$ (total)")
    ax2.set_title("Effective independent samples"); ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Critical slowing down check at ε={eps:.3f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(outdir, f"autocorr_eps{abs(eps):.3f}".replace(".", "p") + ".png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Wrote {path}")


def _scaling_plot(smry: pd.DataFrame, eps: float, outdir: str) -> None:
    s = smry.sort_values("L")
    L = s["L"].to_numpy(float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.loglog(L, s["chi_pooled"], "o-", color="tab:blue", label="pooled")
    ax1.loglog(L, s["chi_pertraj"], "s--", color="tab:gray", label="per-traj (current)")
    ax1.loglog(L, s["chi_pooled"].iloc[0] * (L / L[0]) ** 1.75, ":", color="k", label=r"$L^{1.75}$ (2D Ising)")
    ax1.set_xlabel("L"); ax1.set_ylabel(r"$\chi_{\max}$"); ax1.set_title("Susceptibility scaling")
    ax1.legend(fontsize=8); ax1.grid(True, which="both", alpha=0.3)

    if s["c_pooled"].notna().any():
        ax2.loglog(L, s["c_pooled"], "o-", color="tab:red", label="pooled")
        ax2.loglog(L, s["c_pertraj"], "s--", color="tab:gray", label="per-traj (current)")
        ax2.loglog(L, s["c_pooled"].iloc[0] * (L / L[0]) ** 2.0, ":", color="k", label=r"$L^{2}$ (first order)")
    ax2.set_xlabel("L"); ax2.set_ylabel(r"$c_{\max}$"); ax2.set_title("Heat capacity scaling")
    ax2.legend(fontsize=8); ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Finite-size scaling at ε={eps:.3f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(outdir, f"scaling_eps{abs(eps):.3f}".replace(".", "p") + ".png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Pooled P(m)/P(E_int) diagnostic for first-order vs continuous")
    p.add_argument("--results", default="susceptibility_results")
    p.add_argument("--epsilon", type=float, default=-1.73, help="target ε (snapped to nearest available)")
    p.add_argument("--outdir", default="plots/diagnostics")
    p.add_argument("--list", action="store_true", help="list available ε and L, then exit")
    args = p.parse_args()

    df = collect(args.results)
    if args.list:
        print("Available L:", sorted(df["L"].unique()))
        eps_values = np.sort(df["epsilon"].unique())
        print(f"Available ε ({eps_values.size}): {eps_values.min():.3f} … {eps_values.max():.3f}")
        print("ε values:", ", ".join(f"{e:.3f}" for e in eps_values))
        return

    eps = pick_epsilon(df, args.epsilon)
    if not np.isclose(eps, args.epsilon):
        print(f"Note: ε={args.epsilon} not found; using nearest available ε={eps:.4f}")
    analyze(df, eps, args.outdir)


if __name__ == "__main__":
    main()
