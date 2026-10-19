"""
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

Data reality (differs from the generic spec; see coex/runner.py / coex/layout.py):
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
    coexistence, so per epsilon we take the mu with ⟨φ_col⟩ closest to 0
    (zero_mean) and report Sarle BC there (max / balanced / nearest_coex
    remain available).

Modules (each step is independently testable):
  bc.py            column_op, extract_column_op, pool_column_op, sarle_bc
  mu_selection.py  resolve_mu_coex, nearest_mu_dir
  bc_sweep.py      sweep_bc: BC at every epsilon of one (scheme, size)
  epsilon_c.py     locate_epsilon_c (crossover fit), find_criticality (orchestrator)
  phase_diagram.py BC_max families per delta_mu, scheme comparison, grid coverage
  figures.py       P(phi_col) histograms and the multi-size BC_max study
  bimodality.py    this file: the command line (run/criticality.py)
"""

from __future__ import annotations

import csv
import os

from flexsim.coex.layout import param_tag
from flexsim.criticality.bc import BC_BIMODAL_CUTOFF
from flexsim.criticality.bc_sweep import bc_err_stats, backfill_bc_err
from flexsim.criticality.phase_diagram import DEFAULT_SCHEME_COMPARISON_PANELS, inspect_coverage, phase_diagram, scheme_comparison
from flexsim.criticality.figures import make_histogram_figure, run_fss


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
    sp.add_argument("--mu-reduction",
                    choices=["zero_mean", "max", "nearest_coex", "balanced"],
                    default="zero_mean",
                    help="How to pick one mu per epsilon: ⟨φ⟩ closest to 0 "
                         "(default), max BC, nearest mu_coex, or balanced "
                         "liquid/gas fractions (histograms).")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Bimodality-based criticality from coex snapshots: Sarle BC "
                    "at ⟨φ⟩≈0 vs beta*epsilon; criticality = linear BC(x) "
                    f"intersecting 5/9≈{BC_BIMODAL_CUTOFF:.4f}.",
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
    fs = sub.add_parser("bc-fss", help="Figure 2: BC_max vs beta*epsilon, several sizes")
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
    sc.add_argument("--mu-reduction",
                    choices=["zero_mean", "max", "nearest_coex", "balanced"],
                    default="zero_mean")
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
        from flexsim.criticality.plot_bimodality import plot_bc_family

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
        n_err, err_min, err_max = bc_err_stats(rows)
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

    elif args.mode == "bc-fss":
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
            from flexsim.criticality.plot_bimodality import plot_bc_scheme_comparison

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
