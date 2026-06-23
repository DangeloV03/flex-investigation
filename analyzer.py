"""
analyzer.py

Long-running watcher that:
  1. Polls results/ for completed output.csv files.
  2. Groups results by outer combo (scheme, Lx, Ly, epsilon, delta_f, delta_mu, k).
  3. For each combo with enough data, computes phi(mu) and psi(mu).
  4. Waits for all N_INITIAL_MU_POINTS (10) initial mu jobs before analyzing.
  5. Checks for sign change in phi:
       - Sign change found  -> refine mu in the bracket until dense or budget exhausted.
       - Neighbors of min(psi) near zero -> finalize (resolved) if min(psi) <= PSI_COEX_MAX.
       - Interior min(psi) with sign change -> finalize with argmin(psi) once bracket
         is dense or no new mu values remain, and min(psi) <= PSI_COEX_MAX.
       - No sign change     -> extend mu window; NaN only if budget exhausted.
     Max MAX_ADDITIONAL_REQUESTS additional data requests per combo.
  6. Finds min(psi) -> mu_coex_SIM.
  7. Saves phi/psi plot and CSV inside each combo folder under results/.
  8. Updates manage.csv with mu_coex_SIM, isAnalyzed, combo_path, RequestForAdditionalData.

Usage:
    python analyzer.py [--results results] [--manage manage.csv] [--interval 10]
"""

import argparse
import csv
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combo_paths import (
    COMBO_KEY_FIELDS,
    RESULTS_DIR,
    combo_dir,
    combo_dir_name,
    discover_combo_results,
    phi_psi_png_path,
    write_phi_psi_csv,
)
from queue_manifest import prepend_pending

MANAGE_CSV = "manage.csv"
SAMPLES_DIR = "samples"
POLL_INTERVAL = 10.0  # seconds
MAX_ADDITIONAL_REQUESTS = 5
N_INITIAL_MU_POINTS = 10  # must match generate_samples.N_MU_POINTS
N_REFINEMENT_POINTS = 10
PHI_NEIGHBOR_SIGMA_K = 2.0  # |phi| <= k * max(phi_err, PHI_ABS_TOL) counts as "close"
PHI_ABS_TOL = 0.05
PSI_COEX_MAX = 0.05  # min(psi) must be at or below this to accept mu_coex_SIM
MANAGE_FIELDS = COMBO_KEY_FIELDS + [
    "mu_coex_FLEX",
    "isSubmitted",
    "isRan",
    "isAnalyzed",
    "mu_coex_SIM",
    "mu_coex_SIM_error",
    "RequestForAdditionalData",
    "combo_path",
]


# ---------------------------------------------------------------------------
# manage.csv helpers
# ---------------------------------------------------------------------------

def read_manage(manage_path: str) -> list[dict]:
    if not os.path.isfile(manage_path):
        return []
    with open(manage_path, "r", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    rows: list[dict] = []
    for line_no, row in enumerate(raw_rows, start=2):
        row.setdefault("mu_coex_SIM_error", "")
        row.setdefault("combo_path", "")
        missing = [f for f in COMBO_KEY_FIELDS if f not in row or row[f] is None]
        if missing:
            print(
                f"[analyzer] WARNING: skipping malformed manage.csv line {line_no} "
                f"(missing {missing})",
                file=sys.stderr,
            )
            continue
        rows.append(row)
    return rows


def write_manage(manage_path: str, rows: list[dict]):
    with open(manage_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANAGE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANAGE_FIELDS})


def _combo_field_match(row_val, combo_val) -> bool:
    """Match manage.csv strings to job dict values (exact or numeric)."""
    if row_val == combo_val:
        return True
    rs = str(row_val).strip()
    cs = str(combo_val).strip()
    if rs == cs:
        return True
    try:
        return float(rs) == float(cs)
    except (TypeError, ValueError):
        return False


def find_manage_row(rows: list[dict], combo: dict) -> int | None:
    """Return index of the manage row matching this combo, or None."""
    for i, row in enumerate(rows):
        if all(_combo_field_match(row.get(f, ""), combo.get(f, "")) for f in COMBO_KEY_FIELDS):
            return i
    return None


def update_manage_field(manage_path: str, combo: dict, updates: dict):
    """Update specific fields on the matching combo row, only if currently empty
    (for timestamp fields) or always (for numeric fields like RequestForAdditionalData)."""
    rows = read_manage(manage_path)
    idx = find_manage_row(rows, combo)
    if idx is None:
        print(f"[analyzer] WARNING: no manage row found for {combo}", file=sys.stderr)
        return
    for field, value in updates.items():
        # Timestamp fields: only write if empty (first writer wins)
        if field in ("isRan", "isAnalyzed") and rows[idx].get(field, ""):
            continue
        rows[idx][field] = value
    write_manage(manage_path, rows)


# ---------------------------------------------------------------------------
# Physics calculations
# ---------------------------------------------------------------------------

def calculate_phi_psi(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Compute phi = <rho_active - rho_inert - rho_empty> and psi = |phi|
    with error propagation."""
    rho_a = df["rho_active"].mean()
    rho_i = df["rho_inert"].mean()
    rho_e = df["rho_empty"].mean()

    std_a = df["rho_active"].std()
    std_i = df["rho_inert"].std()
    std_e = df["rho_empty"].std()

    phi = rho_a - rho_i - rho_e
    psi = abs(phi)

    # Error propagation for psi = |phi|
    # d(psi)/d(rho_a) = sign(phi), etc.
    if psi > 0:
        prefactor = -phi / psi  # = -sign(phi)
        psi_error = np.sqrt(
            (std_a * phi / psi) ** 2
            + (std_i * prefactor) ** 2
            + (std_e * prefactor) ** 2
        )
        phi_error = np.sqrt(std_a**2 + std_i**2 + std_e**2)
    else:
        psi_error = 0.0
        phi_error = 0.0

    return phi, phi_error, psi, psi_error


def build_curves(points: list[tuple]) -> tuple:
    """Given [(mu, df), ...], return sorted arrays of mu, phi, phi_err, psi, psi_err."""
    points_sorted = sorted(points, key=lambda x: x[0])
    mu_vals, phi_vals, phi_errs, psi_vals, psi_errs = [], [], [], [], []
    for mu, df in points_sorted:
        phi, phi_err, psi, psi_err = calculate_phi_psi(df)
        mu_vals.append(mu)
        phi_vals.append(phi)
        phi_errs.append(phi_err)
        psi_vals.append(psi)
        psi_errs.append(psi_err)
    return (
        np.array(mu_vals),
        np.array(phi_vals),
        np.array(phi_errs),
        np.array(psi_vals),
        np.array(psi_errs),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_mu_coex(mu_coex_sim) -> bool:
    if mu_coex_sim is None:
        return False
    if isinstance(mu_coex_sim, str) and mu_coex_sim.lower() == "nan":
        return False
    if isinstance(mu_coex_sim, float) and np.isnan(mu_coex_sim):
        return False
    return True


def mu_coex_for_plot(
    mu_vals: np.ndarray,
    psi_vals: np.ndarray,
    manage_row: dict | None = None,
) -> float | None:
    """Pick mu_coex to annotate plots: manage.csv value, else argmin(psi).

    Returns None when manage.csv explicitly records mu_coex_SIM=NaN (unstable).
    """
    if manage_row is not None:
        raw = str(manage_row.get("mu_coex_SIM", "")).strip()
        if raw.lower() == "nan":
            return None
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    if len(mu_vals) == 0:
        return None
    return float(mu_vals[int(np.argmin(psi_vals))])


def plot_combo(combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
               mu_coex_sim=None, results_dir=RESULTS_DIR):
    job = dict(zip(COMBO_KEY_FIELDS, combo_key))
    tag = combo_dir_name(job)
    plot_path = phi_psi_png_path(job, results_dir)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    write_phi_psi_csv(
        job, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs, base=results_dir,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # phi plot
    ax1.errorbar(mu_vals, phi_vals, yerr=phi_errs, fmt="o-", capsize=4,
                 linewidth=1.2, markersize=5, label=r"$\phi$")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    if _plot_mu_coex(mu_coex_sim):
        ax1.axvline(mu_coex_sim, color="red", linestyle="--", linewidth=1,
                    label=f"$\\mu_{{coex}}^{{SIM}}={float(mu_coex_sim):.4f}$")
    ax1.set_xlabel(r"$\mu$", fontsize=13)
    ax1.set_ylabel(r"$\phi$", fontsize=13)
    ax1.set_title(f"$\\phi$ vs $\\mu$ — {tag}", fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    # psi plot
    ax2.errorbar(mu_vals, psi_vals, yerr=psi_errs, fmt="o-", capsize=4,
                 linewidth=1.2, markersize=5, color="orange", label=r"$\psi$")
    if _plot_mu_coex(mu_coex_sim):
        ax2.axvline(mu_coex_sim, color="red", linestyle="--", linewidth=1,
                    label=f"$\\mu_{{coex}}^{{SIM}}={float(mu_coex_sim):.4f}$")
    ax2.set_xlabel(r"$\mu$", fontsize=13)
    ax2.set_ylabel(r"$\psi$", fontsize=13)
    ax2.set_title(f"$\\psi$ vs $\\mu$ — {tag}", fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[analyzer] Plot saved: {plot_path}")


# ---------------------------------------------------------------------------
# Job dispatching
# ---------------------------------------------------------------------------

def make_job_json(job_template: dict, mu: float, samples_dir: str) -> str:
    """Write a new JSON job file for a given mu and return its path."""
    scheme = str(job_template["scheme"])
    Ly = int(job_template["Ly"])
    mu_tag = f"mu{round(abs(mu) * 1_000_000):07d}"
    filename = f"{scheme}_Ly{Ly}_{mu_tag}.json"
    os.makedirs(samples_dir, exist_ok=True)
    filepath = os.path.join(samples_dir, filename)

    # Coerce all values to native Python types for JSON serialization
    def to_native(v):
        if hasattr(v, "item"):  # catches all numpy scalars
            return v.item()
        return v

    job = {k: to_native(v) for k, v in job_template.items()}
    job["mu"] = round(float(mu), 6)

    with open(filepath, "w") as f:
        json.dump(job, f, indent=2)
    return filepath


def enqueue_jobs(
    mu_values: list[float],
    job_template: dict,
    samples_dir: str,
    manifest_path: str = "run_all_queue.json",
) -> int:
    """Write JSON files and prepend them to the run_all queue (priority stack).

    Returns the number of jobs actually added to the queue (duplicates skipped).
    """
    paths = []
    for mu in mu_values:
        json_path = make_job_json(job_template, mu, samples_dir)
        paths.append(json_path)
        print(f"[analyzer] Enqueued {json_path} (mu={mu:.6f})")
    import queue_manifest as qm

    prev = qm.MANIFEST_PATH
    qm.MANIFEST_PATH = manifest_path
    try:
        return prepend_pending(paths)
    finally:
        qm.MANIFEST_PATH = prev


# ---------------------------------------------------------------------------
# Coexistence resolution check
# ---------------------------------------------------------------------------

def phi_is_close_to_zero(phi: float, phi_err: float) -> bool:
    """True if |phi| is within k sigma of zero (with an absolute floor)."""
    threshold = PHI_NEIGHBOR_SIGMA_K * max(phi_err, PHI_ABS_TOL)
    return abs(phi) <= threshold


def is_coex_resolved(
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
) -> bool:
    """True when min(psi) has neg/pos phi neighbors that are both near zero."""
    if len(mu_vals) < 3:
        return False

    min_idx = int(np.argmin(psi_vals))
    if min_idx == 0 or min_idx == len(mu_vals) - 1:
        return False

    phi_left = phi_vals[min_idx - 1]
    phi_right = phi_vals[min_idx + 1]
    if phi_left >= 0 or phi_right <= 0:
        return False

    return (
        phi_is_close_to_zero(phi_left, phi_errs[min_idx - 1])
        and phi_is_close_to_zero(phi_right, phi_errs[min_idx + 1])
    )


def psi_min_index(psi_vals: np.ndarray) -> int:
    return int(np.argmin(psi_vals))


def min_psi_value(psi_vals: np.ndarray) -> float:
    return float(np.min(psi_vals))


def is_psi_minimum_acceptable(psi_vals: np.ndarray) -> bool:
    """True when argmin(psi) is small enough to trust mu_coex_SIM."""
    return min_psi_value(psi_vals) <= PSI_COEX_MAX


def interior_psi_minimum(psi_vals: np.ndarray) -> bool:
    """True when argmin(psi) is not at the edge of the sampled mu range."""
    if len(psi_vals) < 3:
        return False
    min_idx = psi_min_index(psi_vals)
    return 0 < min_idx < len(psi_vals) - 1


def has_phi_sign_change(phi_vals: np.ndarray) -> bool:
    signs = np.sign(phi_vals)
    return not np.all(signs == signs[0])


def sign_change_bracket(
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
) -> tuple[float, float] | None:
    """Return (mu_lo, mu_hi) bracketing phi=0, or None if no sign change."""
    if not has_phi_sign_change(phi_vals):
        return None
    pos_mask = phi_vals > 0
    neg_mask = phi_vals < 0
    mu_pos = float(mu_vals[pos_mask][np.argmin(np.abs(phi_vals[pos_mask]))])
    mu_neg = float(mu_vals[neg_mask][np.argmin(np.abs(phi_vals[neg_mask]))])
    return min(mu_pos, mu_neg), max(mu_pos, mu_neg)


def unsampled_mus(candidate_mus: list[float], mu_vals: np.ndarray) -> list[float]:
    return [
        m for m in candidate_mus
        if not any(abs(m - float(existing)) < 1e-6 for existing in mu_vals)
    ]


def count_in_bracket(mu_vals: np.ndarray, mu_lo: float, mu_hi: float) -> int:
    return int(np.sum((mu_vals >= mu_lo) & (mu_vals <= mu_hi)))


def extension_window(
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    *,
    toward_edge: int | None = None,
) -> tuple[float, float]:
    """Return (mu_lo, mu_hi) for a window extension.

    If toward_edge is 0, extend below the current range; if len(mu)-1, extend above.
    Otherwise infer direction from phi sign (all positive -> lower mu, else higher).
    """
    window = float(mu_vals[-1] - mu_vals[0])
    if toward_edge == 0:
        return float(mu_vals[0] - window), float(mu_vals[0])
    if toward_edge is not None and toward_edge == len(mu_vals) - 1:
        return float(mu_vals[-1]), float(mu_vals[-1] + window)
    if np.all(phi_vals > 0):
        return float(mu_vals[0] - window), float(mu_vals[0])
    return float(mu_vals[-1]), float(mu_vals[-1] + window)


def compute_mu_coex_sim_error(
    mu_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
) -> float:
    """Replica-scatter scale at mu neighbors bracketing argmin(psi).

    This is a phi uncertainty proxy used for resolution checks, not a Delta-mu
    error bar on mu_coex_SIM.
    """
    min_idx = psi_min_index(psi_vals)
    if min_idx == 0 or min_idx == len(mu_vals) - 1:
        return float(phi_errs[min_idx])
    return float(max(phi_errs[min_idx - 1], phi_errs[min_idx + 1]))


def finalize_combo(
    combo_key: tuple,
    combo: dict,
    tag: str,
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
    psi_errs: np.ndarray,
    manage_path: str,
    results_dir: str,
    n_requests: int,
    reason: str = "",
):
    """Set mu_coex_SIM, save plot/csv, and mark combo analyzed."""
    min_idx = int(np.argmin(psi_vals))
    mu_coex_sim = float(mu_vals[min_idx])
    sim_error = compute_mu_coex_sim_error(mu_vals, phi_errs, psi_vals)
    suffix = f" ({reason})" if reason else ""
    print(f"[analyzer] {tag}: mu_coex_SIM = {mu_coex_sim:.6f}, "
          f"error = {sim_error:.6f}{suffix}")

    plot_combo(
        combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
        mu_coex_sim=mu_coex_sim, results_dir=results_dir,
    )
    update_manage_field(manage_path, combo, {
        "mu_coex_SIM": mu_coex_sim,
        "mu_coex_SIM_error": sim_error,
        "isAnalyzed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "RequestForAdditionalData": n_requests,
        "combo_path": combo_dir(combo, results_dir),
    })


def finalize_unstable(
    combo_key: tuple,
    combo: dict,
    tag: str,
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
    psi_errs: np.ndarray,
    manage_path: str,
    results_dir: str,
    n_requests: int,
    reason: str = "",
):
    """Mark unstable combo analyzed with mu_coex_SIM=NaN after max refinement requests."""
    suffix = f" ({reason})" if reason else ""
    print(f"[analyzer] {tag}: unstable, mu_coex_SIM=NaN{suffix}")
    plot_combo(
        combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
        mu_coex_sim=None, results_dir=results_dir,
    )
    update_manage_field(manage_path, combo, {
        "mu_coex_SIM": "NaN",
        "mu_coex_SIM_error": "NaN",
        "isAnalyzed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "RequestForAdditionalData": n_requests,
        "combo_path": combo_dir(combo, results_dir),
    })


# ---------------------------------------------------------------------------
# Core analysis logic per combo
# ---------------------------------------------------------------------------

def _request_more_data(
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
    pending_points: dict[tuple, int],
    action: str,
) -> bool:
    """Enqueue refinement/extension jobs if needed. Returns True if jobs were queued."""
    if not new_mus:
        return False

    n_added = enqueue_jobs(new_mus, job, samples_dir, manifest_path)
    if n_added == 0:
        print(f"[analyzer] {tag}: {action} jobs already queued, waiting")
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
    pending_points: dict[tuple, int],
) -> bool:
    """Queue refinement/extension when min(psi) is still above PSI_COEX_MAX."""
    if n_requests >= MAX_ADDITIONAL_REQUESTS:
        return False

    bracket = sign_change_bracket(mu_vals, phi_vals)
    if bracket is not None and interior_psi_minimum(psi_vals):
        mu_lo, mu_hi = bracket
        new_mus = unsampled_mus(
            list(np.linspace(mu_lo, mu_hi, N_REFINEMENT_POINTS)),
            mu_vals,
        )
        if new_mus:
            return _request_more_data(
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
                pending_points=pending_points,
                action="refining (min psi above threshold)",
            )

    min_idx = psi_min_index(psi_vals)
    toward_edge = min_idx if not interior_psi_minimum(psi_vals) else None
    new_lo, new_hi = extension_window(mu_vals, phi_vals, toward_edge=toward_edge)
    new_mus = unsampled_mus(
        list(np.linspace(new_lo, new_hi, N_REFINEMENT_POINTS)),
        mu_vals,
    )
    if new_mus:
        return _request_more_data(
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
            pending_points=pending_points,
            action="extending (min psi above threshold)",
        )
    return False


def _finalize_if_psi_acceptable(
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
        _finalize_if_psi_acceptable(
            **finalize_kwargs,
            finalize_reason="neighbors resolved",
            unstable_reason=f"min(psi) > {PSI_COEX_MAX} after max requests",
        )
        return

    in_bracket = count_in_bracket(mu_vals, mu_lo, mu_hi)
    bracket_dense = in_bracket >= N_REFINEMENT_POINTS
    interior_min = interior_psi_minimum(psi_vals)

    if bracket_dense and interior_min:
        _finalize_if_psi_acceptable(
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
        new_mus = unsampled_mus(
            list(np.linspace(new_lo, new_hi, N_REFINEMENT_POINTS)),
            mu_vals,
        )
        if not new_mus:
            _finalize_if_psi_acceptable(
                **finalize_kwargs,
                finalize_reason="argmin(psi)",
                unstable_reason="min(psi) at edge of mu window",
            )
            return

        direction = "lower" if min_idx == 0 else "higher"
        print(f"[analyzer] {tag}: min(psi) at {direction} edge, extending "
              f"[{new_lo:.4f}, {new_hi:.4f}]")
        _request_more_data(
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
            pending_points=pending_points,
            action="extending toward edge",
        )
        return

    # Interior minimum: try refinement while bracket is sparse and budget remains.
    if not bracket_dense and n_requests < MAX_ADDITIONAL_REQUESTS:
        new_mus = unsampled_mus(
            list(np.linspace(mu_lo, mu_hi, N_REFINEMENT_POINTS)),
            mu_vals,
        )
        if new_mus:
            print(f"[analyzer] {tag}: sign change, refining "
                  f"[{mu_lo:.4f}, {mu_hi:.4f}] with {len(new_mus)} points "
                  f"({in_bracket}/{N_REFINEMENT_POINTS} in bracket)")
            queued = _request_more_data(
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

    _finalize_if_psi_acceptable(
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

    new_mus = unsampled_mus(list(np.linspace(new_lo, new_hi, N_REFINEMENT_POINTS)), mu_vals)
    if not new_mus:
        finalize_unstable(
            combo_key, combo, tag, mu_vals, phi_vals, phi_errs,
            psi_vals, psi_errs, manage_path, results_dir, n_requests,
            reason="no unsampled mu in extension window",
        )
        return

    _request_more_data(
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
        pending_points=pending_points,
        action="extending window",
    )


def analyze_combo(combo_key: tuple, data: dict, manage_path: str,
                  results_dir: str, samples_dir: str, manifest_path: str,
                  pending_points: dict[tuple, int]):
    job = data["job"]
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
            if (
                n_points >= N_INITIAL_MU_POINTS
                and has_phi_sign_change(phi_vals)
                and interior_psi_minimum(psi_vals)
            ):
                _finalize_if_psi_acceptable(
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


# ---------------------------------------------------------------------------
# Main watch loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Watch results/ and compute mu_coex_SIM per combo."
    )
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--manage", default=MANAGE_CSV)
    parser.add_argument("--samples", default=SAMPLES_DIR)
    parser.add_argument("--interval", type=float, default=POLL_INTERVAL)
    parser.add_argument("--manifest", default="run_all_queue.json",
                        help="Queue manifest for run_all.py")
    args = parser.parse_args()

    print(f"[analyzer] Watching '{args.results}' every {args.interval}s "
          f"(Ctrl-C to stop)")

    processed_combos = set()  # combo_keys that are fully analyzed
    pending_points: dict[tuple, int] = {}  # point count when last follow-up was queued

    while True:
        grouped = discover_combo_results(args.results)

        for combo_key, data in grouped.items():
            if combo_key in processed_combos:
                continue

            # Check if this combo is already marked analyzed in manage.csv
            rows = read_manage(args.manage)
            combo = {f: data["job"][f] for f in COMBO_KEY_FIELDS}
            idx = find_manage_row(rows, combo)
            if idx is not None and rows[idx].get("isAnalyzed", ""):
                processed_combos.add(combo_key)
                pending_points.pop(combo_key, None)
                continue

            analyze_combo(
                combo_key, data, args.manage, args.results, args.samples, args.manifest,
                pending_points,
            )

            # Re-check if now analyzed
            rows = read_manage(args.manage)
            if idx is not None and rows[idx].get("isAnalyzed", ""):
                processed_combos.add(combo_key)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()