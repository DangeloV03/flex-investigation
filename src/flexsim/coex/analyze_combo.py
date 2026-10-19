"""Per-combo decision tree: refine around a phi sign change, extend the window, or finalize."""

from __future__ import annotations

import sys

import numpy as np

from flexsim.coex.layout import COMBO_KEY_FIELDS, combo_dir_name
from flexsim.coex.queue_manifest import read_manifest
from flexsim.coex.manage_csv import find_manage_row, read_manage
from flexsim.coex.phi_psi import build_curves
from flexsim.coex.criteria import (
    MAX_ADDITIONAL_REQUESTS,
    N_INITIAL_MU_POINTS,
    N_REFINEMENT_POINTS,
    PSI_COEX_MAX,
    count_in_bracket,
    extension_window,
    finalize_unstable,
    has_phi_sign_change,
    interior_psi_minimum,
    is_coex_resolved,
    psi_min_index,
    refinement_mus,
    sign_change_bracket,
)
from flexsim.coex.refine_steps import finalize_if_psi_acceptable, request_more_data


def _analyze_sign_change(
    *,
    combo_key: tuple,
    combo: dict,
    tag: str,
    job: dict,
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
    psi_errs: np.ndarray,
    manage_path: str,
    results_dir: str,
    samples_dir: str,
    manifest_path: str,
    pending_points: dict[tuple, int],
    rows: list[dict],
    idx: int,
    n_requests: int,
    n_points: int,
) -> None:
    """Phi changes sign: refine bracket, then finalize with argmin(psi)."""
    bracket = sign_change_bracket(mu_vals, phi_vals)
    if bracket is None:
        return
    mu_lo, mu_hi = bracket

    finalize_kwargs = dict(
        combo_key=combo_key,
        combo=combo,
        tag=tag,
        job=job,
        mu_vals=mu_vals,
        phi_vals=phi_vals,
        phi_errs=phi_errs,
        psi_vals=psi_vals,
        psi_errs=psi_errs,
        manage_path=manage_path,
        results_dir=results_dir,
        samples_dir=samples_dir,
        manifest_path=manifest_path,
        pending_points=pending_points,
        rows=rows,
        idx=idx,
        n_requests=n_requests,
        n_points=n_points,
    )

    if is_coex_resolved(mu_vals, phi_vals, phi_errs, psi_vals):
        finalize_if_psi_acceptable(
            **finalize_kwargs,
            finalize_reason="neighbors resolved",
            unstable_reason=f"min(psi) > {PSI_COEX_MAX} after max requests",
        )
        return

    in_bracket = count_in_bracket(mu_vals, mu_lo, mu_hi)
    bracket_dense = in_bracket >= N_REFINEMENT_POINTS
    interior_min = interior_psi_minimum(psi_vals)

    if bracket_dense and interior_min:
        finalize_if_psi_acceptable(
            **finalize_kwargs,
            finalize_reason="dense bracket",
            unstable_reason=f"min(psi) > {PSI_COEX_MAX} after max requests",
        )
        return

    if not interior_min:
        if n_requests >= MAX_ADDITIONAL_REQUESTS:
            finalize_unstable(
                combo_key, combo, tag, mu_vals, phi_vals, phi_errs,
                psi_vals, psi_errs, manage_path, results_dir, n_requests,
                reason="min(psi) at edge of mu window",
            )
            return

        min_idx = psi_min_index(psi_vals)
        new_lo, new_hi = extension_window(mu_vals, phi_vals, toward_edge=min_idx)
        new_mus = refinement_mus(new_lo, new_hi, mu_vals)
        if not new_mus:
            finalize_if_psi_acceptable(
                **finalize_kwargs,
                finalize_reason="argmin(psi)",
                unstable_reason="min(psi) at edge of mu window",
            )
            return

        direction = "lower" if min_idx == 0 else "higher"
        print(f"[analyzer] {tag}: min(psi) at {direction} edge, extending "
              f"[{new_lo:.4f}, {new_hi:.4f}]")
        request_more_data(
            tag=tag,
            combo_key=combo_key,
            combo=combo,
            new_mus=new_mus,
            job=job,
            manage_path=manage_path,
            rows=rows,
            idx=idx,
            n_requests=n_requests,
            n_points=n_points,
            samples_dir=samples_dir,
            manifest_path=manifest_path,
            results_dir=results_dir,
            pending_points=pending_points,
            action="extending toward edge",
        )
        return

    # Interior minimum: try refinement while bracket is sparse and budget remains.
    if not bracket_dense and n_requests < MAX_ADDITIONAL_REQUESTS:
        new_mus = refinement_mus(mu_lo, mu_hi, mu_vals)
        if new_mus:
            print(f"[analyzer] {tag}: sign change, refining "
                  f"[{mu_lo:.4f}, {mu_hi:.4f}] with {len(new_mus)} points "
                  f"({in_bracket}/{N_REFINEMENT_POINTS} in bracket)")
            queued = request_more_data(
                tag=tag,
                combo_key=combo_key,
                combo=combo,
                new_mus=new_mus,
                job=job,
                manage_path=manage_path,
                rows=rows,
                idx=idx,
                n_requests=n_requests,
                n_points=n_points,
                samples_dir=samples_dir,
                manifest_path=manifest_path,
                results_dir=results_dir,
                pending_points=pending_points,
                action="refining",
            )
            if queued:
                return
            if not bracket_dense:
                print(
                    f"[analyzer] {tag}: refinement already queued; "
                    f"waiting for new data before finalizing",
                )
                return

    finalize_if_psi_acceptable(
        **finalize_kwargs,
        finalize_reason="argmin(psi)",
        unstable_reason=f"min(psi) > {PSI_COEX_MAX} after max requests",
    )


def _analyze_no_sign_change(
    *,
    combo_key: tuple,
    combo: dict,
    tag: str,
    job: dict,
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
    psi_errs: np.ndarray,
    manage_path: str,
    results_dir: str,
    samples_dir: str,
    manifest_path: str,
    pending_points: dict[tuple, int],
    rows: list[dict],
    idx: int,
    n_requests: int,
    n_points: int,
) -> None:
    """Phi same sign everywhere: extend mu window until sign change or budget exhausted."""
    if n_requests >= MAX_ADDITIONAL_REQUESTS:
        finalize_unstable(
            combo_key, combo, tag, mu_vals, phi_vals, phi_errs,
            psi_vals, psi_errs, manage_path, results_dir, n_requests,
            reason="max requests, no sign change",
        )
        return

    new_lo, new_hi = extension_window(mu_vals, phi_vals)
    if np.all(phi_vals > 0):
        print(f"[analyzer] {tag}: all phi>0, extending window lower "
              f"[{new_lo:.4f}, {new_hi:.4f}]")
    else:
        print(f"[analyzer] {tag}: all phi<0, extending window higher "
              f"[{new_lo:.4f}, {new_hi:.4f}]")

    new_mus = refinement_mus(new_lo, new_hi, mu_vals)
    if not new_mus:
        finalize_unstable(
            combo_key, combo, tag, mu_vals, phi_vals, phi_errs,
            psi_vals, psi_errs, manage_path, results_dir, n_requests,
            reason="no unsampled mu in extension window",
        )
        return

    request_more_data(
        tag=tag,
        combo_key=combo_key,
        combo=combo,
        new_mus=new_mus,
        job=job,
        manage_path=manage_path,
        rows=rows,
        idx=idx,
        n_requests=n_requests,
        n_points=n_points,
        samples_dir=samples_dir,
        manifest_path=manifest_path,
        results_dir=results_dir,
        pending_points=pending_points,
        action="extending window",
    )


def _enrich_job_template(job: dict, results_dir: str, manage_path: str) -> dict:
    """Ensure refinement JSONs carry the same paths as initial susceptibility jobs."""
    enriched = dict(job)
    enriched["results_base"] = results_dir
    enriched["manage_csv"] = manage_path
    return enriched


def _manifest_has_activity(manifest_path: str) -> bool:
    manifest = read_manifest(manifest_path)
    return bool(manifest.get("pending")) or bool(manifest.get("in_flight"))


def analyze_combo(combo_key: tuple, data: dict, manage_path: str,
                  results_dir: str, samples_dir: str, manifest_path: str,
                  pending_points: dict[tuple, int]):
    job = _enrich_job_template(data["job"], results_dir, manage_path)
    points = data["points"]

    combo = {f: job[f] for f in COMBO_KEY_FIELDS}
    tag = combo_dir_name(job)

    mu_vals, phi_vals, phi_errs, psi_vals, psi_errs = build_curves(points)
    n_points = len(mu_vals)

    rows = read_manage(manage_path)
    idx = find_manage_row(rows, combo)
    if idx is None:
        print(
            f"[analyzer] WARNING: {tag}: no manage.csv row for combo "
            f"{ {f: combo[f] for f in COMBO_KEY_FIELDS} }",
            file=sys.stderr,
        )
        return
    n_requests = int(rows[idx].get("RequestForAdditionalData", 0))

    if rows[idx].get("isAnalyzed", ""):
        return

    if n_requests == 0 and n_points < N_INITIAL_MU_POINTS:
        print(f"[analyzer] {tag}: waiting for initial batch "
              f"({n_points}/{N_INITIAL_MU_POINTS})")
        return

    if n_requests > 0:
        points_at_request = pending_points.get(combo_key)
        if points_at_request is not None and n_points <= points_at_request:
            if not _manifest_has_activity(manifest_path):
                print(
                    f"[analyzer] {tag}: refinement stalled (queue empty, "
                    f"still {n_points} mu points); re-analyzing",
                )
                pending_points.pop(combo_key, None)
            elif (
                n_points >= N_INITIAL_MU_POINTS
                and has_phi_sign_change(phi_vals)
                and interior_psi_minimum(psi_vals)
            ):
                finalize_if_psi_acceptable(
                    combo_key=combo_key,
                    combo=combo,
                    tag=tag,
                    job=job,
                    mu_vals=mu_vals,
                    phi_vals=phi_vals,
                    phi_errs=phi_errs,
                    psi_vals=psi_vals,
                    psi_errs=psi_errs,
                    manage_path=manage_path,
                    results_dir=results_dir,
                    samples_dir=samples_dir,
                    manifest_path=manifest_path,
                    pending_points=pending_points,
                    rows=rows,
                    idx=idx,
                    n_requests=n_requests,
                    n_points=n_points,
                    finalize_reason="initial batch complete (no new refinement data)",
                    unstable_reason=f"min(psi) > {PSI_COEX_MAX} after max requests",
                )
                pending_points.pop(combo_key, None)
                return
            else:
                return

    common = dict(
        combo_key=combo_key,
        combo=combo,
        tag=tag,
        job=job,
        mu_vals=mu_vals,
        phi_vals=phi_vals,
        phi_errs=phi_errs,
        psi_vals=psi_vals,
        psi_errs=psi_errs,
        manage_path=manage_path,
        results_dir=results_dir,
        samples_dir=samples_dir,
        manifest_path=manifest_path,
        pending_points=pending_points,
        rows=rows,
        idx=idx,
        n_requests=n_requests,
        n_points=n_points,
    )

    if has_phi_sign_change(phi_vals):
        _analyze_sign_change(**common)
    else:
        _analyze_no_sign_change(**common)


def select_depth_first_combo(
    grouped: dict,
    manage_path: str,
) -> tuple[str, ...] | None:
    """Pick one unfinished combo: finish in-flight refinements first, else lowest epsilon."""
    rows = read_manage(manage_path)
    open_combos: list[tuple[float, tuple[str, ...], bool]] = []

    for combo_key, data in grouped.items():
        combo = {f: data["job"][f] for f in COMBO_KEY_FIELDS}
        idx = find_manage_row(rows, combo)
        if idx is None or rows[idx].get("isAnalyzed", ""):
            continue
        n_requests = int(rows[idx].get("RequestForAdditionalData", 0) or 0)
        open_combos.append((float(combo["epsilon"]), combo_key, n_requests > 0))

    if not open_combos:
        return None

    in_refinement = [(eps, key) for eps, key, active in open_combos if active]
    if in_refinement:
        return min(in_refinement, key=lambda item: item[0])[1]

    return min(open_combos, key=lambda item: item[0])[1]
