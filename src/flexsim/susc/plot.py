"""
Plot susceptibility χ vs ε, order parameter |m| vs ε, heat capacity c vs ε, and Binder U4 vs ε.

Reads m_timeseries_{id}.csv files (raw time series per replica) and computes all observables
from scratch, following the paper's ordering:
  1. time averages ⟨m⟩, ⟨|m|⟩, ⟨m²⟩, ⟨m⁴⟩, ⟨E_int⟩, ⟨E_int²⟩ per single trajectory
  2. per-trajectory observables: χ, c, U4
  3. average over replicas per (L, ε)

E_interact is recovered from the stored total energy and densities:
  e_interact = e_total − e_chem
  e_chem     = −β·μ·N·ρ_B − β·(μ+Δf)·N·ρ_I

Usage:
    python run/plots.py susc
    python run/plots.py susc --results susceptibility/results/exact --outdir plots/exact

The observables live in susc/aggregate.py and the figures in susc/figures.py;
this file is the command line (run/plots.py susc).
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from flexsim.susc.aggregate import aggregate, aggregate_pooled
from flexsim.susc.figures import draw_peak_chi_figure, plot_binder_vs_epsilon, plot_chi_vs_epsilon, plot_fig7_panels, plot_heat_capacity_vs_epsilon, plot_m_histograms_at_peak, plot_m_vs_epsilon, plot_peak_chi_vs_L, print_moments_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot susceptibility campaign results")
    parser.add_argument("--results", default="susceptibility/results")
    parser.add_argument("--outdir", default="plots/susceptibility")
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="Pool all replica samples per (L, ε) before computing χ/c/U4 "
        "(vs per-trajectory then averaged). Writes *_pooled.png alongside the originals.",
    )
    parser.add_argument(
        "--peak-only",
        action="store_true",
        help="Skip aggregation and replot peak_chi_vs_L from the existing CSV in --outdir.",
    )
    parser.add_argument(
        "--peak-chi-only",
        action="store_true",
        help="Only aggregate χ(ε,L), take max over ε per L, write peak_chi_vs_L csv/png "
             "(respects --pooled). Skips Fig 7 and other plots to save memory.",
    )
    parser.add_argument(
        "--tail-fraction", type=float, default=1.0,
        help="Use only the last fraction of each replica's series (e.g. 0.2 = last 20%%, "
             "the equilibrated tail). Default 1.0 = full series. Point --outdir somewhere "
             "separate to keep tail plots apart from full-data plots.",
    )
    parser.add_argument(
        "--fig7-only",
        action="store_true",
        help="Only reproduce the Fig. 7 two-panel figure (c vs βε and χ vs βε) and exit. "
             "Uses per-trajectory c and pooled χ. Skips other plots.",
    )
    parser.add_argument(
        "--abs-m-only",
        action="store_true",
        help="Only plot ⟨|m|⟩ vs βε and write abs_m_vs_beta_epsilon.csv/png; then exit.",
    )
    parser.add_argument(
        "--fit-min-L", type=float, default=0.0,
        help="Fit the χ^max vs L slope (γ/ν) using only L >= this value, excluding "
             "small-L corrections-to-scaling that bias the slope low. Default 0 = all L.",
    )
    args = parser.parse_args()
    if not 0.0 < args.tail_fraction <= 1.0:
        parser.error("--tail-fraction must be in (0, 1].")

    args.results = os.path.abspath(args.results)
    args.outdir = os.path.abspath(args.outdir)
    print(f"Results: {args.results}")
    print(f"Outdir:  {args.outdir}")

    if args.peak_only:
        ftag = "_pooled" if args.pooled else ""
        csv_path = os.path.join(args.outdir, f"peak_chi_vs_L{ftag}.csv")
        peaks = pd.read_csv(csv_path)
        draw_peak_chi_figure(peaks, args.outdir, pooled=args.pooled, fit_min_L=args.fit_min_L)
        return

    if args.peak_chi_only:
        if args.pooled:
            print("Peak χ vs L: pooled aggregation")
            agg, _ = aggregate_pooled(args.results, tail_fraction=args.tail_fraction, verbose=True)
        else:
            print("Peak χ vs L: per-trajectory then averaged")
            agg, _ = aggregate(args.results, tail_fraction=args.tail_fraction, verbose=True)
        print_moments_summary(agg)
        plot_peak_chi_vs_L(agg, args.outdir, pooled=args.pooled, fit_min_L=args.fit_min_L)
        return

    if args.tail_fraction < 1.0:
        print(f"Using last {args.tail_fraction:.0%} of each replica's series (equilibrated tail)")

    if args.fig7_only:
        print("Fig 7: per-trajectory c + pooled χ vs βε")
        agg_c, _ = aggregate(args.results, tail_fraction=args.tail_fraction)
        agg_chi, _ = aggregate_pooled(args.results, tail_fraction=args.tail_fraction)
        plot_fig7_panels(agg_c, agg_chi, args.outdir)
        return

    if args.abs_m_only:
        if args.pooled:
            print("⟨|m|⟩: pooled aggregation")
            agg, _ = aggregate_pooled(args.results, tail_fraction=args.tail_fraction)
        else:
            print("⟨|m|⟩: per-trajectory then averaged")
            agg, _ = aggregate(args.results, tail_fraction=args.tail_fraction)
        plot_m_vs_epsilon(agg, args.outdir, pooled=args.pooled)
        return

    if args.pooled:
        print("Aggregation: POOLED (all replica samples combined before χ/c/U4)")
        agg, traj_df = aggregate_pooled(args.results, tail_fraction=args.tail_fraction)
    else:
        print("Aggregation: per-trajectory then averaged")
        agg, traj_df = aggregate(args.results, tail_fraction=args.tail_fraction)

    print_moments_summary(agg)
    agg_c, _ = aggregate(args.results, tail_fraction=args.tail_fraction)
    agg_chi, _ = aggregate_pooled(args.results, tail_fraction=args.tail_fraction)
    plot_fig7_panels(agg_c, agg_chi, args.outdir)
    plot_chi_vs_epsilon(agg, args.outdir, pooled=args.pooled)
    plot_m_vs_epsilon(agg, args.outdir, pooled=args.pooled)
    plot_binder_vs_epsilon(agg, args.outdir, pooled=args.pooled)
    plot_heat_capacity_vs_epsilon(agg, args.outdir, pooled=args.pooled)
    plot_peak_chi_vs_L(agg, args.outdir, pooled=args.pooled, fit_min_L=args.fit_min_L)
    plot_m_histograms_at_peak(traj_df, agg, args.outdir, pooled=args.pooled)


if __name__ == "__main__":
    main()
