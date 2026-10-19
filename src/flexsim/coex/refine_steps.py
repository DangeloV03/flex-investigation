"""Analyzer actions: request more mu points, or finalize a combo when psi is acceptable."""

from __future__ import annotations

import numpy as np

from flexsim.coex.manage_csv import write_manage
from flexsim.coex.refine_jobs import jobs_already_in_manifest, enqueue_jobs
from flexsim.coex.criteria import (
    MAX_ADDITIONAL_REQUESTS,
    PSI_COEX_MAX,
    extension_window,
    finalize_combo,
    finalize_unstable,
    interior_psi_minimum,
    is_psi_minimum_acceptable,
    min_psi_value,
    psi_min_index,
    refinement_mus,
    sign_change_bracket,
)


# ---------------------------------------------------------------------------
# Core analysis logic per combo
# ---------------------------------------------------------------------------

def request_more_data(
    *,
    tag: str,
    combo_key: tuple,
    combo: dict,
    new_mus: list[float],
    job: dict,
    manage_path: str,
    rows: list[dict],
    idx: int,
    n_requests: int,
    n_points: int,
    samples_dir: str,
    manifest_path: str,
    results_dir: str,
    pending_points: dict[tuple, int],
    action: str,
) -> bool:
    """Enqueue refinement/extension jobs if needed. Returns True if jobs were queued."""
    if not new_mus:
        return False

    n_added, paths = enqueue_jobs(
        new_mus,
        job,
        samples_dir,
        manifest_path,
        results_dir=results_dir,
        manage_path=manage_path,
    )
    if n_added == 0:
        if jobs_already_in_manifest(paths, manifest_path):
            if n_requests == 0:
                rows[idx]["RequestForAdditionalData"] = 1
                write_manage(manage_path, rows)
                pending_points[combo_key] = n_points
                print(
                    f"[analyzer] {tag}: {action} jobs already in queue; "
                    f"synced RequestForAdditionalData=1",
                )
            else:
                pending_points.setdefault(combo_key, n_points)
                print(f"[analyzer] {tag}: {action} jobs already queued, waiting")
            return True
        print(f"[analyzer] {tag}: failed to queue {action} jobs")
        return False

    n_requests += 1
    rows[idx]["RequestForAdditionalData"] = n_requests
    write_manage(manage_path, rows)
    pending_points[combo_key] = n_points
    print(f"[analyzer] {tag}: {action}, queued {n_added} jobs "
          f"(request {n_requests}/{MAX_ADDITIONAL_REQUESTS})")
    return True


def _request_psi_improvement(
    *,
    tag: str,
    combo_key: tuple,
    combo: dict,
    job: dict,
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    psi_vals: np.ndarray,
    manage_path: str,
    rows: list[dict],
    idx: int,
    n_requests: int,
    n_points: int,
    samples_dir: str,
    manifest_path: str,
    results_dir: str,
    pending_points: dict[tuple, int],
) -> bool:
    """Queue refinement/extension when min(psi) is still above PSI_COEX_MAX."""
    if n_requests >= MAX_ADDITIONAL_REQUESTS:
        return False

    bracket = sign_change_bracket(mu_vals, phi_vals)
    if bracket is not None and interior_psi_minimum(psi_vals):
        mu_lo, mu_hi = bracket
        new_mus = refinement_mus(mu_lo, mu_hi, mu_vals)
        if new_mus:
            return request_more_data(
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
                action="refining (min psi above threshold)",
            )

    min_idx = psi_min_index(psi_vals)
    toward_edge = min_idx if not interior_psi_minimum(psi_vals) else None
    new_lo, new_hi = extension_window(mu_vals, phi_vals, toward_edge=toward_edge)
    new_mus = refinement_mus(new_lo, new_hi, mu_vals)
    if new_mus:
        return request_more_data(
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
            action="extending (min psi above threshold)",
        )
    return False


def finalize_if_psi_acceptable(
    *,
    finalize_reason: str,
    unstable_reason: str,
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
) -> bool:
    """Finalize when min(psi) is acceptable; otherwise refine or mark unstable."""
    if is_psi_minimum_acceptable(psi_vals):
        finalize_combo(
            combo_key, combo, tag, mu_vals, phi_vals, phi_errs,
            psi_vals, psi_errs, manage_path, results_dir, n_requests,
            reason=finalize_reason,
        )
        return True

    psi_min = min_psi_value(psi_vals)
    print(
        f"[analyzer] {tag}: min(psi)={psi_min:.4f} > {PSI_COEX_MAX}, "
        f"rejecting {finalize_reason!r}",
    )
    if _request_psi_improvement(
        tag=tag,
        combo_key=combo_key,
        combo=combo,
        job=job,
        mu_vals=mu_vals,
        phi_vals=phi_vals,
        psi_vals=psi_vals,
        manage_path=manage_path,
        rows=rows,
        idx=idx,
        n_requests=n_requests,
        n_points=n_points,
        samples_dir=samples_dir,
        manifest_path=manifest_path,
        results_dir=results_dir,
        pending_points=pending_points,
    ):
        return True

    reason = unstable_reason or f"min(psi)={psi_min:.4f} > {PSI_COEX_MAX}"
    finalize_unstable(
        combo_key, combo, tag, mu_vals, phi_vals, phi_errs,
        psi_vals, psi_errs, manage_path, results_dir, n_requests,
        reason=reason,
    )
    return True
