#!/usr/bin/env python3
"""
Re-run analyzer on manage.csv rows marked mu_coex_SIM=NaN.

NaN usually means the analyzer hit RequestForAdditionalData >= 5 without
passing is_coex_resolved — not that there is too little data. Many NaN combos
(like a clear sigmoid with a sharp psi minimum) can be recovered by resetting
the request budget and re-analyzing; argmin(psi) is then written via
finalize_combo.

Usage (on Della):
    # Diagnose only — no file changes
    python scripts/retry_nan_combos.py --diagnose-only

    # Reset NaN rows and re-run analyzer once (may enqueue Slurm jobs)
    python scripts/retry_nan_combos.py

    # Preview reset + analysis
    python scripts/retry_nan_combos.py --dry-run

    # If retry still NaN but curve looks good: force argmin(psi) result
    python scripts/retry_nan_combos.py --force-min-psi

After running, ensure run_all.py is active if jobs were enqueued:
    ./scripts/start_daemons.sh
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import numpy as np

from analyzer import (
    COMBO_KEY_FIELDS,
    MAX_ADDITIONAL_REQUESTS,
    N_INITIAL_MU_POINTS,
    N_REFINEMENT_POINTS,
    PHI_ABS_TOL,
    PHI_NEIGHBOR_SIGMA_K,
    analyze_combo,
    build_curves,
    combo_dir_name,
    discover_combo_results,
    finalize_combo,
    find_manage_row,
    is_coex_resolved,
    phi_is_close_to_zero,
    read_manage,
    write_manage,
)
from combo_paths import RESULTS_DIR, combo_key_from_dict
from generate_samples import MANAGE_CSV, SAMPLES_DIR
from queue_manifest import read_manifest


def is_nan_row(row: dict) -> bool:
    if not str(row.get("isAnalyzed", "")).strip():
        return False
    return str(row.get("mu_coex_SIM", "")).strip().lower() == "nan"


def reset_row(row: dict) -> None:
    row["isAnalyzed"] = ""
    row["mu_coex_SIM"] = ""
    row["mu_coex_SIM_error"] = ""
    row["RequestForAdditionalData"] = "0"


def diagnose_nan_combo(row: dict, data: dict) -> list[str]:
    """Return human-readable lines explaining why this combo got NaN."""
    job = data["job"]
    tag = combo_dir_name(job)
    mu_vals, phi_vals, phi_errs, psi_vals, psi_errs = build_curves(data["points"])
    n_points = len(mu_vals)
    try:
        n_requests = int(row.get("RequestForAdditionalData") or 0)
    except ValueError:
        n_requests = 0

    lines = [
        f"\n{'=' * 72}",
        f"{tag}",
        f"  epsilon={row['epsilon']}  delta_mu={row['delta_mu']}  "
        f"mu_coex_FLEX={row.get('mu_coex_FLEX', '')}",
        f"  mu points on disk: {n_points}  (need {N_INITIAL_MU_POINTS} for initial analysis)",
        f"  RequestForAdditionalData: {n_requests}  (NaN at >= {MAX_ADDITIONAL_REQUESTS})",
    ]

    if n_points == 0:
        lines.append("  NO RESULT DATA — need simulation jobs, not re-analysis")
        return lines

    min_idx = int(np.argmin(psi_vals))
    mu_at_min = float(mu_vals[min_idx])
    lines.append(
        f"  argmin(psi): mu={mu_at_min:.6f}  psi={psi_vals[min_idx]:.4f}  "
        f"phi={phi_vals[min_idx]:.4f}"
    )

    signs = np.sign(phi_vals)
    has_sign_change = not np.all(signs == signs[0])
    lines.append(f"  phi sign change across mu sweep: {has_sign_change}")

    resolved = is_coex_resolved(mu_vals, phi_vals, phi_errs, psi_vals)
    lines.append(f"  is_coex_resolved (strict neighbor test): {resolved}")

    if n_points >= 3:
        if min_idx == 0 or min_idx == len(mu_vals) - 1:
            lines.append(
                f"    min(psi) at mu sweep edge (index {min_idx}/{n_points - 1}) — "
                "coex may lie outside sampled window"
            )
        elif not has_sign_change:
            lines.append("    no sign change — mu window likely too narrow")
        else:
            phi_left = float(phi_vals[min_idx - 1])
            phi_right = float(phi_vals[min_idx + 1])
            err_left = float(phi_errs[min_idx - 1])
            err_right = float(phi_errs[min_idx + 1])
            thr_left = PHI_NEIGHBOR_SIGMA_K * max(err_left, PHI_ABS_TOL)
            thr_right = PHI_NEIGHBOR_SIGMA_K * max(err_right, PHI_ABS_TOL)
            lines.append(
                f"    phi neighbors of min(psi): left={phi_left:+.4f}  right={phi_right:+.4f}"
            )
            lines.append(
                f"    'close to zero' thresholds: left |phi|<={thr_left:.4f}  "
                f"right |phi|<={thr_right:.4f}  (k={PHI_NEIGHBOR_SIGMA_K}, "
                f"floor={PHI_ABS_TOL})"
            )
            if phi_left >= 0:
                lines.append("    fail: left neighbor phi >= 0 (expected negative)")
            elif phi_right <= 0:
                lines.append("    fail: right neighbor phi <= 0 (expected positive)")
            if not phi_is_close_to_zero(phi_left, err_left):
                lines.append(
                    f"    fail: left neighbor |phi|={abs(phi_left):.4f} > threshold {thr_left:.4f}"
                )
            if not phi_is_close_to_zero(phi_right, err_right):
                lines.append(
                    f"    fail: right neighbor |phi|={abs(phi_right):.4f} > threshold {thr_right:.4f}"
                )

    if has_sign_change:
        pos_mask = phi_vals > 0
        neg_mask = phi_vals < 0
        mu_pos = float(mu_vals[pos_mask][np.argmin(np.abs(phi_vals[pos_mask]))])
        mu_neg = float(mu_vals[neg_mask][np.argmin(np.abs(phi_vals[neg_mask]))])
        mu_lo, mu_hi = min(mu_pos, mu_neg), max(mu_pos, mu_neg)
        in_bracket = int(np.sum((mu_vals >= mu_lo) & (mu_vals <= mu_hi)))
        lines.append(
            f"  sign-change bracket [{mu_lo:.4f}, {mu_hi:.4f}]: {in_bracket} mu points "
            f"(refinement target {N_REFINEMENT_POINTS})"
        )

    # What happens after reset (RequestForAdditionalData -> 0)?
    if n_points < N_INITIAL_MU_POINTS:
        lines.append(
            f"  after reset: analyzer waits for initial batch ({n_points}/{N_INITIAL_MU_POINTS})"
        )
    elif has_sign_change:
        if resolved:
            lines.append("  after reset: should finalize_combo immediately (neighbors resolved)")
        elif has_sign_change and n_points >= N_INITIAL_MU_POINTS:
            pos_mask = phi_vals > 0
            neg_mask = phi_vals < 0
            mu_pos = float(mu_vals[pos_mask][np.argmin(np.abs(phi_vals[pos_mask]))])
            mu_neg = float(mu_vals[neg_mask][np.argmin(np.abs(phi_vals[neg_mask]))])
            mu_lo, mu_hi = min(mu_pos, mu_neg), max(mu_pos, mu_neg)
            in_bracket = int(np.sum((mu_vals >= mu_lo) & (mu_vals <= mu_hi)))
            if in_bracket >= N_REFINEMENT_POINTS:
                lines.append(
                    f"  after reset: should finalize_combo with mu={mu_at_min:.6f} "
                    "(bracket already dense — no new Slurm jobs)"
                )
            else:
                lines.append(
                    "  after reset: analyzer will enqueue refinement mu jobs via run_all.py"
                )
    elif n_requests >= MAX_ADDITIONAL_REQUESTS:
        lines.append(
            "  after reset: no sign change — may enqueue window-extension jobs "
            "unless curve already spans transition"
        )

    if n_requests >= MAX_ADDITIONAL_REQUESTS and not resolved:
        lines.append(
            f"  WHY NaN NOW: hit max refinement requests ({n_requests}) while "
            "is_coex_resolved was still False. Resetting request count to 0 fixes "
            "most cases with dense brackets."
        )

    return lines


def force_min_psi_finalize(
    combo_key: tuple,
    data: dict,
    manage_path: str,
    results_dir: str,
    *,
    dry_run: bool,
) -> None:
    job = data["job"]
    combo = {f: job[f] for f in COMBO_KEY_FIELDS}
    tag = combo_dir_name(job)
    mu_vals, phi_vals, phi_errs, psi_vals, psi_errs = build_curves(data["points"])
    min_idx = int(np.argmin(psi_vals))
    if min_idx == 0 or min_idx == len(mu_vals) - 1:
        print(f"  SKIP force: min(psi) at edge for {tag}")
        return
    mu_at_min = float(mu_vals[min_idx])
    if dry_run:
        print(f"  would force finalize_combo mu={mu_at_min:.6f} for {tag}")
        return
    finalize_combo(
        combo_key, combo, tag, mu_vals, phi_vals, phi_errs,
        psi_vals, psi_errs, manage_path, results_dir, n_requests=0,
        reason="forced argmin(psi) retry",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose and re-analyze manage.csv rows with mu_coex_SIM=NaN",
    )
    parser.add_argument("--manage", default=MANAGE_CSV)
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--samples", default=SAMPLES_DIR)
    parser.add_argument("--manifest", default="run_all_queue.json")
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Print diagnostics only; do not reset or re-analyze",
    )
    parser.add_argument(
        "--force-min-psi",
        action="store_true",
        help="After retry, force argmin(psi) for combos still NaN with interior minimum",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_manage(args.manage)
    targets = [row for row in rows if is_nan_row(row)]
    if not targets:
        print("No analyzed NaN rows in manage.csv")
        return 0

    grouped = discover_combo_results(args.results)
    print(f"Found {len(targets)} NaN combo(s) in {args.manage}")

    to_reset: list[tuple[tuple, dict, dict]] = []

    for row in targets:
        combo = {f: row[f] for f in COMBO_KEY_FIELDS}
        combo_key = combo_key_from_dict(combo)
        data = grouped.get(combo_key)

        if data is None:
            print(f"\n{combo_dir_name(combo)}: NO results/ data — run requeue_incomplete.py")
            continue

        for line in diagnose_nan_combo(row, data):
            print(line)

        if not args.diagnose_only:
            to_reset.append((combo_key, data, row))

    if args.diagnose_only or not to_reset:
        if args.diagnose_only:
            print(f"\n(diagnose-only — {len(targets)} NaN row(s) inspected)")
        return 0

    print(f"\n{'Would reset' if args.dry_run else 'Resetting'} {len(to_reset)} row(s) "
          f"and re-running analyzer logic")

    for _combo_key, _data, row in to_reset:
        if args.dry_run:
            print(f"  reset: eps={row['epsilon']} dmu={row['delta_mu']}")
        else:
            reset_row(row)

    if not args.dry_run:
        write_manage(args.manage, rows)

    pending_points: dict[tuple, int] = {}
    n_finalized = 0
    n_still_nan = 0
    n_enqueued = 0

    manifest_before = read_manifest(args.manifest)
    pending_before = len(manifest_before.get("pending", []))

    for combo_key, data, row in to_reset:
        tag = combo_dir_name(data["job"])
        if args.dry_run:
            print(f"  would analyze: {tag}")
            continue

        analyze_combo(
            combo_key, data, args.manage, args.results, args.samples, args.manifest,
            pending_points,
        )

        updated = read_manage(args.manage)
        idx = find_manage_row(updated, {f: data["job"][f] for f in COMBO_KEY_FIELDS})
        if idx is None:
            continue
        result = str(updated[idx].get("mu_coex_SIM", "")).strip()
        if result.lower() == "nan":
            n_still_nan += 1
            print(f"\n  STILL NaN after retry: {tag}")
            for line in diagnose_nan_combo(updated[idx], data):
                if line.startswith("  ") or line.startswith("    "):
                    print(line)
            if args.force_min_psi:
                force_min_psi_finalize(
                    combo_key, data, args.manage, args.results, dry_run=False,
                )
                updated = read_manage(args.manage)
                result = str(updated[idx].get("mu_coex_SIM", "")).strip()
                if result.lower() != "nan" and result:
                    n_still_nan -= 1
                    n_finalized += 1
                    print(f"  FORCED OK: mu_coex_SIM={result}")
        elif result:
            n_finalized += 1
            print(f"\n  OK: {tag}  mu_coex_SIM={result}")
        else:
            print(f"\n  PENDING: {tag} — analyzer waiting for more mu jobs or initial batch")

    manifest_after = read_manifest(args.manifest)
    n_enqueued = len(manifest_after.get("pending", [])) - pending_before

    print(f"\n{'=' * 72}")
    print(f"Summary: {len(to_reset)} retried, {n_finalized} got numeric mu_coex_SIM, "
          f"{n_still_nan} still NaN")
    if n_enqueued > 0:
        print(f"  Enqueued ~{n_enqueued} new job(s) in {args.manifest}")
        print("  Ensure run_all.py is running: ./scripts/start_daemons.sh")
    elif not args.dry_run and n_finalized == len(to_reset):
        print("  No new Slurm jobs needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
