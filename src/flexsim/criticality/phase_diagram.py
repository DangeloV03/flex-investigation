"""BC_max vs beta*epsilon families (one curve per delta_mu), scheme comparison, grid coverage."""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np

from flexsim.coex.layout import param_tag, size_tag
from flexsim.criticality.bc import untag
from flexsim.criticality.bc_sweep import CRIT_FIELDS, append_row, discover_epsilons, results_roots, sweep_bc
from flexsim.criticality.epsilon_c import locate_epsilon_c


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
    mu_reduction: str = "zero_mean",
    make_plots: bool = True,
    show_transition_bracket: bool = True,
) -> list[dict]:
    """BC-vs-(beta*epsilon) curve for each delta_mu at a fixed scheme+size.

    Reproduces the "thermal phase diagram" family plot: one curve per delta_mu,
    y = Sarle BC at the selected mu (default: ⟨φ_col⟩ closest to 0), x = beta*epsilon.
    Criticality is the linear-fit BC(x) ∩ BC=5/9. If delta_mus is None,
    every delta_mu present on disk is used.

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
        append_row(crit_csv, res, CRIT_FIELDS)
        results.append(res)
        rec = res.get("recommended_uncertainty", res["fit_uncertainty"])
        print(
            f"[bimodality] dmu={dmu:+.3f}: (beta*eps)_c="
            f"{res['criticality_estimate']:.4f} +/- {rec} "
            f"(grid neighbor; linear BC∩5/9, "
            f"BC_at={res.get('BC_at_criticality', float('nan')):.4f}) "
            f"({res['method']})",
            flush=True,
        )

    if make_plots and results:
        try:
            from flexsim.criticality.plot_bimodality import plot_bc_family

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

    from flexsim.criticality.plot_bimodality import plot_bc_scheme_comparison

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
