"""Compare slab-fitted coex mu (Ly 16/20/40 + FSS) with the exact equilibrium mu.

Equilibrium (delta_mu = 0): mu_coex = -ln(exp(2|eps|) - exp(delta_f)).

Usage (Della, any env with numpy/pandas/matplotlib):
    python compare_eq_coex.py --root <COEX_RUNS_EQ_DMU0> --out <dir>
Writes eq_coex_comparison.csv, eq_coex_summary.md and eq_coex_vs_exact.png.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LYS = [16, 20, 40]
COLORS = {16: "#2a78d6", 20: "#eb6834", 40: "#1baf7a", "FSS": "#0b0b0b"}
MARKERS = {16: "o", 20: "s", 40: "^", "FSS": "D"}
INK2 = "#52514e"
JITTER = 3e-4  # mu jitter level that gave <m> != 0 in the susc runs
MAX_FIT_ERR = 5e-3  # analyzer fits with a larger error are treated as failed


def exact_mu(eps, delta_f=0.0):
    return -np.log(np.exp(2 * np.abs(eps)) - np.exp(delta_f))


def load(root, delta_f=0.0):
    frames = []
    for ly in LYS:
        path = os.path.join(root, f"ly{ly}", "manage.csv")
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        # Older ledgers use the legacy mu_coex_SIM names.
        mu_col = "mu_coex_FITTED" if "mu_coex_FITTED" in df else "mu_coex_SIM"
        df = df.rename(columns={mu_col: "mu_fit", f"{mu_col}_error": "mu_err"})
        df["Ly"] = ly
        frames.append(df[["epsilon", "Ly", "mu_fit", "mu_err"]])
    out = pd.concat(frames).dropna(subset=["mu_fit"])
    out["epsilon"] = out["epsilon"].round(6)
    out["mu_exact"] = exact_mu(out["epsilon"], delta_f)
    out["dmu"] = out["mu_fit"] - out["mu_exact"]
    bad = out["mu_err"].isna() | (out["mu_err"] > MAX_FIT_ERR)
    for ly, e, err in out.loc[bad, ["Ly", "epsilon", "mu_err"]].itertuples(index=False):
        print(f"[failed fit, dropped] Ly={ly} eps={e} err={err:.2e}")
    return out[~bad]


def fss_fit(sub, power):
    """Weighted fit mu(Ly) = mu_inf + a / Ly**power. Returns mu_inf, err, chi2, a."""
    x = 1.0 / sub["Ly"].to_numpy(float) ** power
    y = sub["mu_fit"].to_numpy(float)
    s = np.maximum(sub["mu_err"].to_numpy(float), 1e-6)
    A = np.vstack([np.ones_like(x), x]).T / s[:, None]
    coef, *_ = np.linalg.lstsq(A, y / s, rcond=None)
    cov = np.linalg.inv(A.T @ A)
    chi2 = float(np.sum(((A @ coef) - y / s) ** 2))
    return coef[0], float(np.sqrt(cov[0, 0])), chi2, coef[1]


def fss_table(data, delta_f=0.0):
    rows = []
    for eps, sub in data.groupby("epsilon"):
        if sub["Ly"].nunique() < 2:
            continue
        r = {"epsilon": eps, "n_ly": sub["Ly"].nunique(), "mu_exact": exact_mu(eps, delta_f)}
        for p, tag in [(1, "inv_ly"), (2, "inv_ly2")]:
            mu_inf, err, chi2, a = fss_fit(sub, p)
            r.update({f"mu_inf_{tag}": mu_inf, f"err_{tag}": err, f"chi2_{tag}": chi2,
                      f"a_{tag}": a, f"dmu_{tag}": mu_inf - r["mu_exact"]})
        rows.append(r)
    return pd.DataFrame(rows)


def stats(dmu, err):
    dmu, err = np.asarray(dmu), np.asarray(err)
    within = np.mean(np.abs(dmu) <= 2 * err) if len(dmu) else np.nan
    return len(dmu), np.mean(dmu), np.sqrt(np.mean(dmu**2)), np.median(err), within


def plot(data, fss, best, path, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    for ax in (ax1, ax2):
        ax.axhspan(-JITTER, JITTER, color="#e6e5e0", zorder=0, lw=0)
        ax.axhline(0, color=INK2, lw=1, zorder=1)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ly in LYS:
        d = data[data["Ly"] == ly].sort_values("epsilon")
        ax1.errorbar(d["epsilon"], d["dmu"], yerr=d["mu_err"], color=COLORS[ly],
                     marker=MARKERS[ly], ms=6, lw=2, capsize=2, mec="white", mew=1,
                     label=f"Ly = {ly}")
    if len(fss):
        ax1.errorbar(fss["epsilon"], fss[f"dmu_{best}"], yerr=fss[f"err_{best}"],
                     color=COLORS["FSS"], marker="D", ms=6, lw=2, ls="--", capsize=2,
                     mec="white", mew=1,
                     label=f"FSS, Ly → ∞ ({'1/Ly' if best == 'inv_ly' else '1/Ly²'})")
    ax1.set_xlabel("βε")
    ax1.set_ylabel("μ_fit − μ_exact")
    ax1.set_title("(a) Slab coex μ minus exact equilibrium μ", loc="left", fontsize=11)
    ax1.text(0.01, 0.02, f"grey band: ±{JITTER:.0e}, the μ jitter that gave ⟨m⟩ ≠ 0",
             transform=ax1.transAxes, fontsize=8, color=INK2)
    ax1.legend(frameon=False, fontsize=9, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, -0.14))

    # (b) dmu vs 1/Ly at a few eps spread over the grid.
    eps_all = sorted(fss["epsilon"]) if len(fss) else []
    picks = [eps_all[i] for i in np.linspace(0, len(eps_all) - 1, 3).astype(int)] if eps_all else []
    shades = ["#0d366b", "#256abf", "#6da7ec"]
    power = 1 if best == "inv_ly" else 2
    for eps, c in zip(picks, shades):
        d = data[data["epsilon"] == eps].sort_values("Ly")
        x = 1.0 / d["Ly"] ** power
        ax2.errorbar(x, d["dmu"], yerr=d["mu_err"], color=c, marker="o", ms=6, lw=0,
                     elinewidth=1.5, capsize=2, mec="white", mew=1, label=f"βε = {eps:g}")
        r = fss[fss["epsilon"] == eps].iloc[0]
        xs = np.linspace(0, x.max() * 1.05, 20)
        ax2.plot(xs, r[f"dmu_{best}"] + r[f"a_{best}"] * xs, color=c, lw=1.5, ls="--")
    ax2.set_xlim(left=-0.002 if power == 1 else -0.0001)
    ax2.set_xlabel("1 / Ly" if power == 1 else "1 / Ly²")
    ax2.locator_params(axis="x", nbins=5)
    ax2.set_ylabel("μ_fit − μ_exact")
    ax2.set_title("(b) Extrapolation to Ly → ∞", loc="left", fontsize=11)
    ax2.legend(frameon=False, fontsize=9, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.14))

    fig.suptitle(title, x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta-f", type=float, default=0.0)
    ap.add_argument("--title", default="Equilibrium benchmark: homo, βΔf = 0, k = 1, βΔμ = 0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = load(args.root, args.delta_f)
    fss = fss_table(data, args.delta_f)
    data.to_csv(os.path.join(args.out, "eq_coex_comparison.csv"), index=False)
    fss.to_csv(os.path.join(args.out, "eq_coex_fss.csv"), index=False)

    # Pick the FSS form with lower total chi2.
    best = "inv_ly"
    if len(fss) and fss["chi2_inv_ly2"].sum() < fss["chi2_inv_ly"].sum():
        best = "inv_ly2"

    lines = ["| set | n ε | mean bias | RMS error | median σ_fit | within 2σ |",
             "|---|---|---|---|---|---|"]
    for ly in LYS:
        d = data[data["Ly"] == ly]
        n, mb, rms, med, w = stats(d["dmu"], d["mu_err"])
        lines.append(f"| Ly = {ly} | {n} | {mb:+.2e} | {rms:.2e} | {med:.1e} | {w:.0%} |")
    for tag, name in [("inv_ly", "FSS 1/Ly"), ("inv_ly2", "FSS 1/Ly²")]:
        if len(fss):
            n, mb, rms, med, w = stats(fss[f"dmu_{tag}"], fss[f"err_{tag}"])
            lines.append(f"| {name} (Σχ² = {fss[f'chi2_{tag}'].sum():.1f}) | {n} | {mb:+.2e} "
                         f"| {rms:.2e} | {med:.1e} | {w:.0%} |")
    summary = "\n".join(lines)
    open(os.path.join(args.out, "eq_coex_summary.md"), "w").write(summary + "\n")
    print(summary)

    plot(data, fss, best, os.path.join(args.out, "eq_coex_vs_exact.png"), args.title)
    print(f"wrote {args.out}/eq_coex_vs_exact.png (FSS form shown: {best})")


if __name__ == "__main__":
    main()
