"""
Orchestrator for the smart susceptibility campaign.

Subcommands
-----------
sweep   Submit initial epsilon sweep (10^6 eq + 2x10^6 prod per replica), one job per (ε, L).
check   Post-sweep jump analysis; writes jump_report.md plus a timing report
        (wall time per L, Slurm CPU/time/memory efficiency); submits top-up
        jobs if needed; loops until every (ε, L) pair meets the ≥ threshold
        avg-jump requirement.

Typical usage
-------------
# Launch the campaign (from repo root):
python run/susc.py sweep \\
    --eps-min -1.8 --eps-max -1.6 --eps-step 0.005 \\
    --results-base SUSC_RUNS

# Run check manually (normally submitted automatically as a dependency):
python run/susc.py check --results-base SUSC_RUNS

The per-round report and stall detection live in susc/smart_report.py.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

from flexsim.susc.job_timing import append_slurm_job, write_round_timing_report
from flexsim.grid import frange
from flexsim.paths import SUSC_SLURM_DIR
from flexsim.susc.smart_report import (
    check_stalled,
    latest_round,
    progress_fingerprint,
    write_report,
)


SMART_L_VALUES = [16, 32, 48, 64, 96, 128]


def _split_size_groups(sizes: list[int]) -> list[tuple[str, list[int]]]:
    """One Slurm job per L.

    The top-up script loops over its sizes sequentially, so grouping 16–96 into
    one job made that job cost the *sum* of its sizes.  Once the replicas
    ordered into the dense phase (~29x more events per unit simulated time) the
    summed cost blew past the 24 h wall and the largest L in each group was lost
    every round.  Per-L jobs keep a slow L from dragging the others down.
    """
    return [(f"L{s}", [s]) for s in sorted(sizes)]

def _exact_mu_map(eps_values: list[float], delta_f: float, delta_mu: float) -> dict[float, float]:
    """Exact equilibrium coexistence μ per ε: μ = -ln(e^{2|ε|} - e^{βΔf}).

    Same closed form as coex_chemical_potential(..., DRIVEN=False).  It reduces
    to μ = 2ε only in the Ising limit (βΔf ≪ 0); at βΔf = 0 it sits 0.03-0.04
    above 2ε for ε in [-1.8, -1.6] — enough to pin L ≥ 32 in one phase.
    """
    if delta_mu != 0.0:
        raise SystemExit(
            f"--mu-exact is the equilibrium result and needs delta_mu=0 "
            f"(got {delta_mu}); use --mu-source with measured μ_coex instead."
        )
    mu_map: dict[float, float] = {}
    for eps in eps_values:
        arg = math.exp(2.0 * abs(eps)) - math.exp(delta_f)
        if arg <= 0.0:
            raise SystemExit(f"No coexistence for ε={eps}, βΔf={delta_f}: e^(2|ε|) <= e^(βΔf).")
        mu_map[eps] = -math.log(arg)
    return mu_map


# ---------------------------------------------------------------------------
# sweep subcommand
# ---------------------------------------------------------------------------

def do_sweep(args: argparse.Namespace) -> None:
    eps_values = frange(args.eps_min, args.eps_max, args.eps_step)
    if not eps_values:
        raise SystemExit("No epsilon values generated; check --eps-min/max/step.")

    if args.skip_eps:
        skip = {round(e, 6) for e in args.skip_eps}
        unused = sorted(skip - {round(e, 6) for e in eps_values})
        if unused:
            raise SystemExit(f"--skip-eps values not on the ε grid: {unused}")
        eps_values = [e for e in eps_values if round(e, 6) not in skip]
        print(f"[sweep] Skipping {len(skip)} ε: {sorted(skip)}", flush=True)
        if not eps_values:
            raise SystemExit("--skip-eps removed every ε.")

    results_base = args.results_base
    os.makedirs(results_base, exist_ok=True)
    os.makedirs("slurm_reports", exist_ok=True)

    sweep_script = args.sweep_script
    check_script = args.check_script

    size_groups = _split_size_groups(args.sizes)
    n_groups = len(size_groups)
    print(
        f"[sweep] Submitting {len(eps_values)} ε × {n_groups} size-groups "
        f"({len(eps_values) * n_groups} jobs) → {results_base}",
        flush=True,
    )

    if args.mu_exact and args.mu_source:
        raise SystemExit("Pass either --mu-exact or --mu-source, not both.")
    if args.mu_exact:
        mu_map = _exact_mu_map(
            eps_values,
            # Runner default when --delta-f is omitted (Ising limit, βΔf = -20).
            delta_f=float(args.delta_f) if args.delta_f != "" else -20.0,
            delta_mu=float(args.delta_mu) if args.delta_mu != "" else 0.0,
        )
        for eps in eps_values:
            print(f"[sweep] ε={eps:+.4f}  μ_coex(exact)={mu_map[eps]:+.6f}  "
                  f"(2ε={2 * eps:+.4f})", flush=True)
    elif args.mu_source:
        # Only rows matching this campaign's physics, and every ε must be
        # covered: silently falling back to the runner's μ = 2ε is how S1B ran
        # 53 rounds off coexistence.
        from flexsim.susc.sweep import load_mu_map

        by_eps = load_mu_map(
            args.mu_source,
            delta_f=args.delta_f or None,
            delta_mu=args.delta_mu or None,
            k=args.k or None,
            scheme=args.scheme or None,
        )
        missing = [eps for eps in eps_values if round(eps, 6) not in by_eps]
        if missing:
            raise SystemExit(
                f"{args.mu_source} has no mu_coex_FITTED matching this physics for "
                f"{len(missing)} of {len(eps_values)} ε (first: {missing[:5]}). "
                f"Refusing to fall back to μ = 2ε."
            )
        mu_map = {eps: by_eps[round(eps, 6)] for eps in eps_values}
    else:
        mu_map = {}

    # Always pass $5–$8 so a later override cannot shift into an earlier slot.
    # Empty string => slurm/smart_sweep.sh leaves that flag unset
    # (runner defaults: Δf=-20, Δμ=0, k=0, scheme=homo, μ=2ε).
    extra = [
        args.delta_f or "",
        args.delta_mu or "",
        args.k or "",
        args.scheme or "",
    ]

    sweep_job_ids: list[str] = []
    for eps in eps_values:
        mu_arg = f"{mu_map[eps]:.10g}" if eps in mu_map else ""
        for tag, sizes in size_groups:
            cmd = [
                "sbatch", "--parsable",
                f"--job-name=susc_{tag}",
                sweep_script,
                f"{eps:.6g}",       # $1 epsilon
                results_base,       # $2 results_base
                "1",                # $3 num_batches
                mu_arg,             # $4 mu (empty = runner default)
                *extra,             # $5 delta_f, $6 delta_mu, $7 k, $8 scheme
                *[str(s) for s in sizes],
            ]
            if args.dry_run:
                print(f"  [DRY-RUN] {' '.join(cmd)}", flush=True)
                sweep_job_ids.append(f"DRY{eps}{tag}")
                continue
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(";")[0]
            sweep_job_ids.append(job_id)
            append_slurm_job(
                results_base,
                round_num=1,
                phase="sweep",
                job_id=job_id,
                epsilon=eps,
                sizes=sizes,
            )
            print(
                f"  Submitted job {job_id} for ε={eps:.4f} L={sizes} ({tag})",
                flush=True,
            )

    if args.dry_run:
        print(f"[sweep] DRY-RUN: would submit check job after {len(sweep_job_ids)} sweep jobs")
        return

    # Chain check job as a dependency on all sweep jobs.  afterany, not
    # afterok: a job that hits the wall exits TIMEOUT, which would leave the
    # check permanently unsatisfiable and silently kill the campaign.
    dep = "--dependency=afterany:" + ":".join(sweep_job_ids)
    check_cmd = [
        "sbatch", "--parsable", dep,
        check_script,
        results_base,
        str(args.threshold),
        check_script,
    ]
    result = subprocess.run(check_cmd, capture_output=True, text=True, check=True)
    check_job_id = result.stdout.strip().split(";")[0]
    print(
        f"[sweep] Submitted check job {check_job_id} "
        f"(dependency: afterany:{':'.join(sweep_job_ids)})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# check subcommand
# ---------------------------------------------------------------------------

def do_check(args: argparse.Namespace) -> None:
    from flexsim.susc.mag_jumps import compute_jump_summary

    results_base = args.results_base
    threshold = args.threshold
    check_script = args.check_script
    topup_script = args.topup_script

    print(f"[check] Loading jump data from {results_base} …", flush=True)
    summary = compute_jump_summary(results_base, threshold=threshold)

    round_num = latest_round(results_base)
    write_report(summary, results_base, round_num=round_num, threshold=threshold)

    if summary.empty:
        _safe_timing_report(results_base, round_num)
        print("[check] No data found — nothing to evaluate.", flush=True)
        return

    stalled_rounds = check_stalled(results_base, progress_fingerprint(summary))

    failing = summary[~summary["passes"]]
    if failing.empty:
        _safe_timing_report(results_base, round_num)
        print("[check] All (ε, L) pairs meet the threshold. Campaign complete!", flush=True)
        return

    # Group failing pairs by epsilon; pass the failing L sizes to the topup job.
    failing_by_eps: dict[float, list[int]] = {}
    for _, row in failing.iterrows():
        eps = float(row["epsilon"])
        L = int(row["L"])
        failing_by_eps.setdefault(eps, []).append(L)

    if stalled_rounds >= args.max_stalled_rounds:
        _safe_timing_report(results_base, round_num)
        raise SystemExit(
            f"[check] STALLED: jump data unchanged for {stalled_rounds} consecutive "
            f"rounds ({len(failing)} pairs still failing). Top-up jobs are producing "
            f"no new samples — check the top-up Slurm logs in slurm_reports/ for "
            f"'no run found' warnings (campaign physics mismatch) or resume failures. "
            f"Not chaining another round."
        )

    print(
        f"[check] {len(failing)} pairs failing; submitting top-up for "
        f"{len(failing_by_eps)} epsilon values …",
        flush=True,
    )
    os.makedirs("slurm_reports", exist_ok=True)

    topup_job_ids: list[str] = []
    for eps in sorted(failing_by_eps):
        for tag, sizes in _split_size_groups(sorted(failing_by_eps[eps])):
            size_strs = [str(s) for s in sizes]
            cmd = [
                "sbatch", "--parsable",
                f"--job-name=susc_top_{tag}",
                topup_script,
                f"{eps:.6g}",
                results_base,
                *size_strs,
            ]
            if args.dry_run:
                print(f"  [DRY-RUN] {' '.join(cmd)}", flush=True)
                topup_job_ids.append(f"DRY{eps}{tag}")
                continue
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(";")[0]
            topup_job_ids.append(job_id)
            append_slurm_job(
                results_base,
                round_num=round_num + 1,
                phase="topup",
                job_id=job_id,
                epsilon=eps,
                sizes=sizes,
            )
            print(
                f"  Submitted top-up job {job_id} for ε={eps:.4f} "
                f"L={sizes} ({tag})",
                flush=True,
            )

    if args.dry_run:
        print(f"[check] DRY-RUN: would submit next check after {len(topup_job_ids)} top-up jobs")
        _safe_timing_report(results_base, round_num)
        return

    # Self-schedule next check after all top-up jobs finish (afterany: a
    # TIMEOUT must not strand the chain — see the sweep comment above).
    dep = "--dependency=afterany:" + ":".join(topup_job_ids)
    next_check_cmd = [
        "sbatch", "--parsable", dep,
        check_script,
        results_base,
        str(threshold),
        check_script,
    ]
    result = subprocess.run(next_check_cmd, capture_output=True, text=True, check=True)
    next_id = result.stdout.strip().split(";")[0]
    print(
        f"[check] Submitted next check job {next_id} "
        f"(dependency: afterany:{':'.join(topup_job_ids)})",
        flush=True,
    )
    _safe_timing_report(results_base, round_num)


def _safe_timing_report(results_base: str, round_num: int) -> None:
    """Timing is optional. Never let sacct/seff abort top-up submission."""
    try:
        write_round_timing_report(results_base, round_num)
    except Exception as exc:
        print(f"[check] WARNING: timing report failed ({exc!r}); campaign continues", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Smart susceptibility campaign orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # -- sweep --
    sp = sub.add_parser("sweep", help="Submit initial epsilon sweep")
    sp.add_argument("--eps-min", type=float, required=True)
    sp.add_argument("--eps-max", type=float, required=True)
    sp.add_argument("--eps-step", type=float, required=True)
    sp.add_argument("--results-base", default="SUSC_RUNS",
                    help="Root output directory (default: SUSC_RUNS)")
    sp.add_argument("--sweep-script", default=str(SUSC_SLURM_DIR / "smart_sweep.sh"))
    sp.add_argument("--check-script", default=str(SUSC_SLURM_DIR / "smart_check.sh"))
    sp.add_argument("--threshold", type=float, default=10.0,
                    help="Minimum avg jumps per (ε, L) pair (default: 10)")
    sp.add_argument("--mu-source", default=None,
                    help="manage.csv path for fitted μ values (optional)")
    sp.add_argument("--mu-exact", action="store_true",
                    help="Run each ε at the exact equilibrium coexistence "
                         "μ = -ln(e^{2|ε|} - e^{βΔf}) (requires delta_mu=0). "
                         "Without this or --mu-source the runner uses μ = 2ε, "
                         "which is coexistence only in the Ising limit.")
    sp.add_argument("--sizes", type=int, nargs="+", default=SMART_L_VALUES,
                    help="Square lattice sizes L (default: 16 32 48 64 96 128)")
    sp.add_argument("--skip-eps", type=float, nargs="+", default=[],
                    help="ε values to leave out of the grid (e.g. ones whose "
                         "--mu-source fit is missing or bad)")
    sp.add_argument("--delta-f", default="", help="δf override (empty = runner default)")
    sp.add_argument("--delta-mu", default="", help="δμ override (empty = runner default)")
    sp.add_argument("--k", default="", help="k override (empty = runner default)")
    sp.add_argument("--scheme", default="", help="Scheme override (empty = runner default)")
    sp.add_argument("--dry-run", action="store_true",
                    help="Print sbatch commands without submitting")

    # -- check --
    cp = sub.add_parser("check", help="Analyse jumps and chain top-up if needed")
    cp.add_argument("--results-base", required=True,
                    help="SUSC_RUNS directory to analyse")
    cp.add_argument("--threshold", type=float, default=10.0)
    cp.add_argument("--max-stalled-rounds", type=int, default=3,
                    help="Abort the chain after this many rounds with no new data "
                         "(default: 3)")
    cp.add_argument("--check-script", default=str(SUSC_SLURM_DIR / "smart_check.sh"))
    cp.add_argument("--topup-script", default=str(SUSC_SLURM_DIR / "topup.sh"))
    cp.add_argument("--dry-run", action="store_true",
                    help="Print sbatch commands without submitting")

    args = p.parse_args()

    if args.command == "sweep":
        do_sweep(args)
    elif args.command == "check":
        do_check(args)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
