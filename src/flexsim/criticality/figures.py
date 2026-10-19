"""P(phi_col) histograms at coexistence and BC_max for several sizes."""

from __future__ import annotations

import os
from typing import Optional

from flexsim.coex.layout import combo_dir_name, param_tag
from flexsim.criticality.mu_selection import enumerate_mu_dirs, resolve_mu_coex
from flexsim.criticality.bc_sweep import (
    CRIT_FIELDS,
    append_row,
    best_mu_over_sweep,
    discover_epsilons,
    sweep_bc,
)
from flexsim.criticality.epsilon_c import locate_epsilon_c


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
    best = best_mu_over_sweep(
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

    from flexsim.criticality.plot_bimodality import plot_pooled_histograms

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
                append_row(crit_csv, res, CRIT_FIELDS)
                results.append(res)
                print(f"[bimodality] {Lx}x{Ly} dmu={dmu:+.2f}: (beta*eps)_c="
                      f"{res['criticality_estimate']:.4f} +/- "
                      f"{res['fit_uncertainty']} ({res['method']})", flush=True)

    if make_plots and all_rows:
        from flexsim.criticality.plot_bimodality import plot_bc_vs_epsilon

        for dmu in sorted({r["delta_mu"] for r in all_rows}):
            out_png = os.path.join(out_dir, f"bc_vs_beta_epsilon_dmu{param_tag(dmu)}.png")
            plot_bc_vs_epsilon(
                bc_csv, out_png, delta_mu=dmu,
                title=f"Max BC vs $\\beta\\epsilon$  $\\Delta\\mu$={dmu}",
            )
            print(f"[bimodality] wrote {out_png}", flush=True)

    return results
