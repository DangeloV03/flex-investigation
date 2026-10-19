"""
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
    python run/plots.py chi-scaling
    python run/plots.py chi-scaling --results susceptibility/results/exact \\
        --outdir plots/chi_max_scaling --n-boot 500
    python run/plots.py chi-scaling --loo-only   # skip slow bootstrap

Data loading is in susc/replica_data.py and the analyses in susc/chi_max_fits.py.
"""

from __future__ import annotations

import argparse
import os

from flexsim.susc.layout import find_susceptibility_csvs
from flexsim.susc.replica_data import fast_inventory, load_replica_groups
from flexsim.susc.chi_max_fits import (
    DEFAULT_REP_COUNTS,
    KD_A,
    KD_GAMMA_NU,
    KD_GAMMA_NU_ERR,
    N_BOOT,
    loo_jackknife_analysis,
    subsampling_analysis,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="chi_max FSS: replica subsampling + LOO jackknife"
    )
    parser.add_argument("--results", default="susceptibility/results/exact",
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
    parser.add_argument("--list", action="store_true",
                        help="Print full inventory of what is found in --results, then exit")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    abs_results = os.path.abspath(args.results)
    print(f"Results dir (absolute): {abs_results}", flush=True)

    if args.list:
        # Instant path: parse dir names + line-count CSVs, never read CSV content
        records = fast_inventory(args.results)
        if not records:
            raise SystemExit("No run dirs with susceptibility_data.csv found — check --results path.")

        from collections import defaultdict
        by_L_eps: dict[tuple[int, float], dict] = defaultdict(
            lambda: {"n_replicas": 0, "n_ts": 0}
        )
        for r in records:
            key = (r["L"], round(r["eps"], 4))
            by_L_eps[key]["n_replicas"] += r["n_replicas"]
            by_L_eps[key]["n_ts"]       += r["n_ts"]

        Ls_list  = sorted({L for (L, _) in by_L_eps})
        eps_by_L = {L: sorted({eps for (l, eps) in by_L_eps if l == L}) for L in Ls_list}
        total_reps = sum(v["n_replicas"] for v in by_L_eps.values())
        total_ts   = sum(v["n_ts"]       for v in by_L_eps.values())

        print(f"Run dirs with data:     {len(records)}")
        print(f"(L, eps) pairs:         {len(by_L_eps)}")
        print(f"Total replica rows:     {total_reps}")
        print(f"Timeseries files found: {total_ts}")
        print(f"L values:               {Ls_list}")
        for L in Ls_list:
            e = eps_by_L[L]
            reps = [by_L_eps[(L, eps)]["n_replicas"] for eps in e]
            print(f"  L={L:>4}:  eps [{min(e):.4f}, {max(e):.4f}]  "
                  f"({len(e)} pts)  replicas/pt: min={min(reps)} max={max(reps)}")

        rep_vals   = [v["n_replicas"] for v in by_L_eps.values() if v["n_replicas"] > 0]
        unique_rep = set(rep_vals)
        print()
        if len(unique_rep) == 1:
            print(f"Replicas per (L, eps):  {rep_vals[0]}  (uniform ✓)")
        else:
            print(f"WARNING: uneven replica counts — min={min(rep_vals)} max={max(rep_vals)}")
            print(f"\n{'L':>5}  {'eps':>8}  {'n_rep':>6}  {'n_ts':>6}")
            for (L, eps) in sorted(by_L_eps):
                v = by_L_eps[(L, eps)]
                flag = "  <-- MISSING TS" if v["n_ts"] < v["n_replicas"] else ""
                print(f"{L:>5}  {eps:>8.4f}  {v['n_replicas']:>6}  {v['n_ts']:>6}{flag}")

        print(f"\nK&D reference: gamma/nu = {KD_GAMMA_NU} +/- {KD_GAMMA_NU_ERR},  A = {KD_A}")
        return

    # Full load (reads all timeseries) — only reached when not --list
    csvs = find_susceptibility_csvs(args.results)
    print(f"Run dirs found:         {len(csvs)}")
    print("Loading timeseries (this may take a few minutes) ...", flush=True)
    groups = load_replica_groups(args.results)

    if not groups:
        raise SystemExit("No data loaded — check --results path.")

    Ls = sorted({L for (L, _) in groups})
    eps_per_L = {
        L: sorted({eps for (l, eps) in groups if l == L})
        for L in Ls
    }
    rep_counts_all = {k: len(groups[k]["replicas"]) for k in groups}
    unique_counts = set(rep_counts_all.values())
    total_replicas = sum(rep_counts_all.values())

    print(f"(L, eps) pairs loaded:  {len(groups)}")
    print(f"Total replica rows:     {total_replicas}")
    print(f"L values:               {Ls}")
    print(f"Eps range per L:        "
          f"{[(L, round(min(eps_per_L[L]),4), round(max(eps_per_L[L]),4), len(eps_per_L[L])) for L in Ls]}")
    if len(unique_counts) > 1:
        print(f"WARNING: uneven replica counts — min={min(unique_counts)} max={max(unique_counts)}")

    n_avail_min = min(rep_counts_all.values())
    # Max replicas available at ANY eps for each L — what the subsampling can reach
    n_avail_max_per_L = {
        L: max(len(groups[(L, eps)]["replicas"]) for eps in eps_per_L[L])
        for L in Ls
    }
    n_avail_max_min = min(n_avail_max_per_L.values())
    print(f"Min replicas (global floor):    {n_avail_min}  (some off-critical eps)")
    print(f"Max replicas per L (subsampling cap): "
          f"{n_avail_max_min}  (at critical eps; all L values)")
    print(f"K&D reference: gamma/nu = {KD_GAMMA_NU} +/- {KD_GAMMA_NU_ERR},  A = {KD_A}")

    if not args.loo_only:
        rep_counts = [n for n in args.rep_counts if n <= n_avail_max_min]
        print(f"\n--- Replica subsampling (n_boot={args.n_boot}, "
              f"rep_counts={rep_counts}) ---", flush=True)
        subsampling_analysis(groups, Ls, eps_per_L, rep_counts, args.n_boot, args.outdir)

    if not args.subsample_only:
        print("\n--- LOO jackknife (all replicas) ---", flush=True)
        loo_jackknife_analysis(groups, Ls, eps_per_L, args.outdir)


if __name__ == "__main__":
    main()
