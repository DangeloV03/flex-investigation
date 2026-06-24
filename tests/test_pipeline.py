"""
tests/test_pipeline.py

Unit and integration tests for the coex susceptibility pipeline.

Run with:
    cd /path/to/flex-investigation
    python -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure project root is on sys.path regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).parent.parent))

import queue_manifest as qm
from analyzer import (
    MAX_ADDITIONAL_REQUESTS,
    N_INITIAL_MU_POINTS,
    N_REFINEMENT_POINTS,
    PSI_COEX_MAX,
    build_curves,
    calculate_phi_psi,
    compute_mu_coex_sim_error,
    extension_window,
    find_manage_row,
    has_phi_sign_change,
    interior_psi_minimum,
    is_coex_resolved,
    is_psi_minimum_acceptable,
    make_job_json,
    read_manage,
    refinement_mus,
    sign_change_bracket,
    unsampled_mus,
    write_manage,
)
from combo_paths import (
    COMBO_KEY_FIELDS,
    combo_dir_name,
    mu_dir_name,
    param_tag,
)
from susceptibility_paths import (
    COEX_MANIFEST,
    COEX_RESULTS_DIR,
    MANAGE_CSV,
    coex_combo_dir,
    patch_coex_job_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rho_a, rho_i, rho_e, mu=0.0, n=8):
    """Synthetic output.csv-like DataFrame for a single mu point."""
    return pd.DataFrame({
        "mu": [mu] * n,
        "rho_active": np.random.default_rng(42).normal(rho_a, 0.01, n),
        "rho_inert": np.random.default_rng(7).normal(rho_i, 0.01, n),
        "rho_empty": np.random.default_rng(13).normal(rho_e, 0.01, n),
        "epsilon": [-2.0] * n,
        "delta_f": [-20.0] * n,
        "delta_mu": [0.0] * n,
        "k": [0.0] * n,
        "scheme": ["homo"] * n,
        "Lx": [160] * n,
        "Ly": [16] * n,
    })


COMBO_DEFAULTS = {
    "epsilon": -2.0,
    "delta_f": -20.0,
    "delta_mu": 0.0,
    "k": 0.0,
    "scheme": "homo",
    "Lx": 160,
    "Ly": 16,
}

MANAGE_FIELDS = COMBO_KEY_FIELDS + [
    "mu_coex_FLEX", "isSubmitted", "isRan", "isAnalyzed",
    "mu_coex_SIM", "mu_coex_SIM_error", "RequestForAdditionalData", "combo_path",
]


def _write_manage(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANAGE_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in MANAGE_FIELDS})


def _default_manage_row(**overrides) -> dict:
    row = {
        "epsilon": -2.0, "delta_f": -20.0, "delta_mu": 0.0,
        "k": 0.0, "scheme": "homo", "Lx": 160, "Ly": 16,
        "mu_coex_FLEX": "", "isSubmitted": "", "isRan": "",
        "isAnalyzed": "", "mu_coex_SIM": "", "mu_coex_SIM_error": "",
        "RequestForAdditionalData": "0", "combo_path": "",
    }
    row.update(overrides)
    return row


# ============================================================
# 1. queue_manifest tests
# ============================================================

class TestQueueManifest:

    def test_empty_file_reads_as_empty_manifest(self, tmp_path):
        p = str(tmp_path / "q.json")
        m = qm.read_manifest(p)
        assert m == {"pending": [], "in_flight": {}}

    def test_seed_pending_replaces(self, tmp_path):
        p = str(tmp_path / "q.json")
        qm.seed_pending(["a.json", "b.json"], path=p)
        qm.seed_pending(["c.json"], path=p)
        assert qm.read_manifest(p)["pending"] == ["c.json"]

    def test_prepend_pending_path_kwarg(self, tmp_path):
        """prepend_pending must honour the path= kwarg, not the module default."""
        default_manifest = str(tmp_path / "run_all_queue.json")
        coex_manifest = str(tmp_path / "coex_queue.json")

        # Module default points elsewhere
        old = qm.MANIFEST_PATH
        qm.MANIFEST_PATH = default_manifest
        try:
            qm.prepend_pending(["job.json"], path=coex_manifest)
        finally:
            qm.MANIFEST_PATH = old

        # Must appear in coex_manifest, not in default
        assert qm.read_manifest(coex_manifest)["pending"] == ["job.json"]
        assert qm.read_manifest(default_manifest)["pending"] == []

    def test_prepend_deduplicates_pending(self, tmp_path):
        p = str(tmp_path / "q.json")
        qm.seed_pending(["a.json"], path=p)
        added = qm.prepend_pending(["a.json", "b.json"], path=p)
        assert added == 1
        assert qm.read_manifest(p)["pending"] == ["b.json", "a.json"]

    def test_prepend_skips_in_flight(self, tmp_path):
        p = str(tmp_path / "q.json")
        qm.mark_in_flight("99", "inflight.json", path=p)
        added = qm.prepend_pending(["inflight.json"], path=p)
        assert added == 0
        assert qm.read_manifest(p)["pending"] == []

    def test_prepend_adds_to_front(self, tmp_path):
        p = str(tmp_path / "q.json")
        qm.seed_pending(["old.json"], path=p)
        qm.prepend_pending(["new.json"], path=p)
        assert qm.read_manifest(p)["pending"][0] == "new.json"

    def test_pop_next_pending_fifo(self, tmp_path):
        p = str(tmp_path / "q.json")
        qm.seed_pending(["first.json", "second.json"], path=p)
        assert qm.pop_next_pending(p) == "first.json"
        assert qm.pop_next_pending(p) == "second.json"
        assert qm.pop_next_pending(p) is None

    def test_pop_returns_none_when_empty(self, tmp_path):
        p = str(tmp_path / "q.json")
        assert qm.pop_next_pending(p) is None

    def test_merge_pending_appends_no_duplicates(self, tmp_path):
        p = str(tmp_path / "q.json")
        qm.seed_pending(["a.json"], path=p)
        qm.merge_pending(["a.json", "b.json"], path=p)
        assert qm.read_manifest(p)["pending"] == ["a.json", "b.json"]

    def test_mark_and_remove_in_flight(self, tmp_path):
        p = str(tmp_path / "q.json")
        qm.mark_in_flight("123", "job.json", path=p)
        assert qm.read_manifest(p)["in_flight"] == {"123": "job.json"}
        removed = qm.remove_in_flight("123", path=p)
        assert removed == "job.json"
        assert qm.read_manifest(p)["in_flight"] == {}

    def test_requeue_front(self, tmp_path):
        p = str(tmp_path / "q.json")
        qm.seed_pending(["b.json"], path=p)
        qm.requeue_front("a.json", path=p)
        assert qm.read_manifest(p)["pending"][0] == "a.json"

    def test_locked_manifest_creates_file(self, tmp_path):
        p = str(tmp_path / "new.json")
        with qm.locked_manifest(p) as m:
            m["pending"].append("x.json")
        assert qm.read_manifest(p)["pending"] == ["x.json"]

    def test_locked_manifest_round_trips_unicode(self, tmp_path):
        p = str(tmp_path / "q.json")
        with qm.locked_manifest(p) as m:
            m["pending"].append("samples/µ_sweep.json")
        assert qm.read_manifest(p)["pending"][0] == "samples/µ_sweep.json"


# ============================================================
# 2. combo_paths helpers
# ============================================================

class TestComboPaths:

    def test_param_tag_negative(self):
        assert param_tag(-2.0) == "m2p0"

    def test_param_tag_positive(self):
        assert param_tag(0.5) == "0p5"

    def test_param_tag_zero(self):
        assert param_tag(0.0) == "0p0"

    def test_param_tag_negative_decimal(self):
        assert param_tag(-1.85) == "m1p85"

    def test_mu_dir_name(self):
        # mu=-3.5 -> abs * 1e6 = 3500000 -> "mu3500000"
        assert mu_dir_name(-3.5) == "mu3500000"

    def test_mu_dir_name_leading_zeros(self):
        assert mu_dir_name(-0.5) == "mu0500000"

    def test_combo_dir_name_format(self):
        name = combo_dir_name(COMBO_DEFAULTS)
        assert "160x16" in name
        assert "homo" in name
        assert "deltaFm20p0" in name
        assert "dmu0p0" in name
        assert "epsilonm2p0" in name


# ============================================================
# 3. susceptibility_paths helpers
# ============================================================

class TestSusceptibilityPaths:

    def test_coex_combo_dir(self):
        d = coex_combo_dir(COMBO_DEFAULTS)
        assert d.startswith(COEX_RESULTS_DIR)

    def test_patch_coex_job_json_sets_results_base(self, tmp_path):
        job = {**COMBO_DEFAULTS, "mu": -3.5, "results_base": "results/"}
        p = str(tmp_path / "job.json")
        with open(p, "w") as f:
            json.dump(job, f)
        changed = patch_coex_job_json(p)
        assert changed
        with open(p) as f:
            patched = json.load(f)
        assert patched["results_base"] == COEX_RESULTS_DIR

    def test_patch_coex_job_json_sets_manage_csv(self, tmp_path):
        job = {**COMBO_DEFAULTS, "mu": -3.5, "manage_csv": "manage.csv"}
        p = str(tmp_path / "job.json")
        with open(p, "w") as f:
            json.dump(job, f)
        patch_coex_job_json(p)
        with open(p) as f:
            patched = json.load(f)
        assert patched["manage_csv"] == MANAGE_CSV

    def test_patch_coex_job_json_no_change_when_correct(self, tmp_path):
        job = {**COMBO_DEFAULTS, "mu": -3.5,
               "results_base": COEX_RESULTS_DIR, "manage_csv": MANAGE_CSV}
        p = str(tmp_path / "job.json")
        with open(p, "w") as f:
            json.dump(job, f)
        changed = patch_coex_job_json(p)
        assert not changed


# ============================================================
# 4. Analyzer — physics pure functions
# ============================================================

class TestCalculatePhiPsi:

    def test_phi_positive_when_active_dominant(self):
        df = _make_df(rho_a=0.8, rho_i=0.1, rho_e=0.1)
        phi, _, psi, _ = calculate_phi_psi(df)
        assert phi > 0
        assert abs(psi - abs(phi)) < 1e-10

    def test_phi_negative_when_inert_and_empty_dominant(self):
        df = _make_df(rho_a=0.1, rho_i=0.5, rho_e=0.4)
        phi, _, _, _ = calculate_phi_psi(df)
        assert phi < 0

    def test_psi_equals_abs_phi(self):
        df = _make_df(rho_a=0.3, rho_i=0.4, rho_e=0.3)
        phi, _, psi, _ = calculate_phi_psi(df)
        assert abs(psi - abs(phi)) < 1e-10

    def test_error_nonnegative(self):
        df = _make_df(rho_a=0.5, rho_i=0.2, rho_e=0.3)
        _, phi_err, _, psi_err = calculate_phi_psi(df)
        assert phi_err >= 0
        assert psi_err >= 0


class TestBuildCurves:

    def _points(self, mu_phi_pairs):
        """Build (mu, df) list with controlled phi values."""
        points = []
        for mu, rho_a in mu_phi_pairs:
            # rho_i=0, rho_e=0 → phi=rho_a
            df = _make_df(rho_a=rho_a, rho_i=0.0, rho_e=0.0, mu=mu, n=50)
            points.append((mu, df))
        return points

    def test_build_curves_sorted(self):
        pts = self._points([(-3.0, 0.6), (-3.5, 0.8), (-2.5, 0.3)])
        mu_vals, phi_vals, *_ = build_curves(pts)
        assert list(mu_vals) == sorted(mu_vals)

    def test_build_curves_sign_tracks_rho_a(self):
        pts = self._points([(-3.0, 0.8), (-2.0, 0.1)])
        mu_vals, phi_vals, *_ = build_curves(pts)
        assert phi_vals[0] > phi_vals[1]


# ============================================================
# 5. Analyzer — sign change & bracket
# ============================================================

class TestSignChange:

    def test_detects_sign_change(self):
        phi = np.array([0.3, 0.1, -0.1, -0.3])
        assert has_phi_sign_change(phi)

    def test_no_sign_change_all_positive(self):
        assert not has_phi_sign_change(np.array([0.1, 0.2, 0.3]))

    def test_no_sign_change_all_negative(self):
        assert not has_phi_sign_change(np.array([-0.3, -0.1, -0.05]))

    def test_bracket_correct_order(self):
        mu = np.array([-3.0, -2.5, -2.0, -1.5])
        phi = np.array([0.3, 0.1, -0.1, -0.3])
        lo, hi = sign_change_bracket(mu, phi)
        assert lo < hi
        assert lo == pytest.approx(-2.5, abs=1e-6)
        assert hi == pytest.approx(-2.0, abs=1e-6)

    def test_bracket_none_when_no_sign_change(self):
        mu = np.array([-3.0, -2.5, -2.0])
        phi = np.array([0.3, 0.2, 0.1])
        assert sign_change_bracket(mu, phi) is None


# ============================================================
# 6. Analyzer — refinement_mus
# ============================================================

class TestRefinementMus:

    def test_returns_new_points_inside_bracket(self):
        mu_lo, mu_hi = -3.0, -2.0
        existing = np.array([-3.0, -2.0])
        new_mus = refinement_mus(mu_lo, mu_hi, existing)
        assert len(new_mus) > 0
        for m in new_mus:
            assert mu_lo <= m <= mu_hi
        assert all(not any(abs(m - e) < 1e-6 for e in existing) for m in new_mus)

    def test_returns_midpoints_when_dense_grid_fills_bracket(self):
        """When N linspace points are all already sampled, fall back to midpoints."""
        mu_lo, mu_hi = -3.0, -2.0
        # Fill the bracket with N_REFINEMENT_POINTS evenly-spaced points
        existing = np.linspace(mu_lo, mu_hi, N_REFINEMENT_POINTS)
        new_mus = refinement_mus(mu_lo, mu_hi, existing)
        # Should still find new midpoints between existing ones
        assert len(new_mus) > 0
        for m in new_mus:
            assert mu_lo <= m <= mu_hi

    def test_unsampled_mus_filters_existing(self):
        candidates = [-3.0, -2.5, -2.0]
        existing = np.array([-3.0, -2.0])
        result = unsampled_mus(candidates, existing)
        assert result == [-2.5]

    def test_unsampled_mus_tolerance(self):
        # 5e-7 < 1e-6 tolerance → treated as already sampled
        candidates = [-3.0000005]
        existing = np.array([-3.0])
        result = unsampled_mus(candidates, existing)
        assert result == []


# ============================================================
# 7. Analyzer — psi / coexistence checks
# ============================================================

class TestPsiChecks:

    def test_psi_acceptable_below_threshold(self):
        psi = np.array([0.3, 0.01, 0.2])
        assert is_psi_minimum_acceptable(psi)

    def test_psi_not_acceptable_above_threshold(self):
        psi = np.array([0.3, 0.1, 0.2])
        assert not is_psi_minimum_acceptable(psi)

    def test_psi_boundary_exactly_at_threshold(self):
        psi = np.array([0.3, PSI_COEX_MAX, 0.2])
        assert is_psi_minimum_acceptable(psi)

    def test_interior_psi_minimum_true(self):
        psi = np.array([0.3, 0.01, 0.2])
        assert interior_psi_minimum(psi)

    def test_interior_psi_minimum_false_at_edge(self):
        psi = np.array([0.01, 0.1, 0.2])
        assert not interior_psi_minimum(psi)

    def test_interior_psi_minimum_too_short(self):
        assert not interior_psi_minimum(np.array([0.1, 0.0]))

    def test_is_coex_resolved_true(self):
        mu = np.array([-3.0, -2.5, -2.0, -1.5])
        phi = np.array([-0.03, -0.01, 0.01, 0.05])
        phi_err = np.array([0.1, 0.1, 0.1, 0.1])
        psi = np.abs(phi)
        assert is_coex_resolved(mu, phi, phi_err, psi)

    def test_is_coex_resolved_false_when_neighbors_not_close(self):
        mu = np.array([-3.0, -2.5, -2.0, -1.5])
        phi = np.array([-0.5, -0.1, 0.4, 0.8])
        phi_err = np.array([0.01, 0.01, 0.01, 0.01])
        psi = np.abs(phi)
        assert not is_coex_resolved(mu, phi, phi_err, psi)

    def test_is_coex_resolved_false_too_short(self):
        mu = np.array([-3.0, -2.0])
        phi = np.array([-0.1, 0.1])
        assert not is_coex_resolved(mu, phi, np.zeros(2), np.abs(phi))


# ============================================================
# 8. Analyzer — extension_window
# ============================================================

class TestExtensionWindow:

    def test_extends_lower_when_all_phi_positive(self):
        mu = np.array([-3.0, -2.5, -2.0])
        phi = np.array([0.4, 0.2, 0.1])
        lo, hi = extension_window(mu, phi)
        assert hi == pytest.approx(-3.0, abs=1e-6)

    def test_extends_upper_when_all_phi_negative(self):
        mu = np.array([-3.0, -2.5, -2.0])
        phi = np.array([-0.4, -0.2, -0.1])
        lo, hi = extension_window(mu, phi)
        assert lo == pytest.approx(-2.0, abs=1e-6)

    def test_toward_edge_lower(self):
        mu = np.array([-3.0, -2.5, -2.0])
        phi = np.array([-0.2, 0.1, 0.3])
        lo, hi = extension_window(mu, phi, toward_edge=0)
        assert hi == pytest.approx(-3.0, abs=1e-6)

    def test_toward_edge_upper(self):
        mu = np.array([-3.0, -2.5, -2.0])
        phi = np.array([-0.2, 0.1, 0.3])
        lo, hi = extension_window(mu, phi, toward_edge=len(mu) - 1)
        assert lo == pytest.approx(-2.0, abs=1e-6)


# ============================================================
# 9. Analyzer — compute_mu_coex_sim_error
# ============================================================

class TestMuCoexSimError:

    def test_uses_neighbor_errors_at_interior_min(self):
        mu = np.array([-3.0, -2.5, -2.0])
        psi = np.array([0.3, 0.01, 0.2])
        phi_errs = np.array([0.05, 0.02, 0.07])
        err = compute_mu_coex_sim_error(mu, phi_errs, psi)
        assert err == pytest.approx(max(0.05, 0.07), abs=1e-10)

    def test_uses_edge_error_at_boundary_min(self):
        mu = np.array([-3.0, -2.5, -2.0])
        psi = np.array([0.01, 0.1, 0.3])
        phi_errs = np.array([0.04, 0.02, 0.07])
        err = compute_mu_coex_sim_error(mu, phi_errs, psi)
        assert err == pytest.approx(0.04, abs=1e-10)


# ============================================================
# 10. Analyzer — manage.csv helpers
# ============================================================

class TestManageCsv:

    def test_read_write_roundtrip(self, tmp_path):
        path = str(tmp_path / "manage.csv")
        row = _default_manage_row()
        _write_manage(path, [row])
        rows = read_manage(path)
        assert len(rows) == 1
        assert rows[0]["epsilon"] == "-2.0"

    def test_read_missing_file_returns_empty(self, tmp_path):
        assert read_manage(str(tmp_path / "nope.csv")) == []

    def test_find_manage_row_found(self, tmp_path):
        path = str(tmp_path / "manage.csv")
        r1 = _default_manage_row(epsilon=-2.0)
        r2 = _default_manage_row(epsilon=-1.9)
        _write_manage(path, [r1, r2])
        rows = read_manage(path)
        combo = {**COMBO_DEFAULTS, "epsilon": -1.9}
        idx = find_manage_row(rows, combo)
        assert idx == 1

    def test_find_manage_row_not_found(self, tmp_path):
        path = str(tmp_path / "manage.csv")
        _write_manage(path, [_default_manage_row()])
        rows = read_manage(path)
        combo = {**COMBO_DEFAULTS, "epsilon": -1.5}
        assert find_manage_row(rows, combo) is None

    def test_find_manage_row_numeric_match(self, tmp_path):
        """Row stored as string '-2.0' should match float -2.0."""
        path = str(tmp_path / "manage.csv")
        _write_manage(path, [_default_manage_row(epsilon="-2.0")])
        rows = read_manage(path)
        combo = {**COMBO_DEFAULTS, "epsilon": -2.0}
        assert find_manage_row(rows, combo) == 0

    def test_write_manage_all_fields(self, tmp_path):
        path = str(tmp_path / "manage.csv")
        rows = [_default_manage_row(mu_coex_SIM="-3.5", isAnalyzed="2025-01-01")]
        write_manage(path, rows)
        rows2 = read_manage(path)
        assert rows2[0]["mu_coex_SIM"] == "-3.5"
        assert rows2[0]["isAnalyzed"] == "2025-01-01"


# ============================================================
# 11. Analyzer — make_job_json
# ============================================================

class TestMakeJobJson:

    def _template(self):
        return {
            **COMBO_DEFAULTS,
            "mu": -3.5,
            "run_settings": {"beta": 1.0, "num_parallel_runs": 4},
        }

    def test_make_job_json_creates_file(self, tmp_path):
        path = make_job_json(self._template(), mu=-3.5, samples_dir=str(tmp_path))
        assert os.path.isfile(path)

    def test_make_job_json_sets_results_base(self, tmp_path):
        path = make_job_json(
            self._template(), mu=-3.5, samples_dir=str(tmp_path),
            results_base="susceptibility_results/coex",
        )
        with open(path) as f:
            job = json.load(f)
        assert job["results_base"] == "susceptibility_results/coex"

    def test_make_job_json_sets_manage_csv(self, tmp_path):
        path = make_job_json(
            self._template(), mu=-3.5, samples_dir=str(tmp_path),
            manage_csv="susceptibility_manage.csv",
        )
        with open(path) as f:
            job = json.load(f)
        assert job["manage_csv"] == "susceptibility_manage.csv"

    def test_make_job_json_mu_value(self, tmp_path):
        path = make_job_json(self._template(), mu=-3.123456, samples_dir=str(tmp_path))
        with open(path) as f:
            job = json.load(f)
        assert abs(job["mu"] - (-3.123456)) < 1e-6

    def test_make_job_json_numpy_scalar_serializable(self, tmp_path):
        template = self._template()
        template["epsilon"] = np.float64(-2.0)
        path = make_job_json(template, mu=-3.5, samples_dir=str(tmp_path))
        with open(path) as f:
            job = json.load(f)
        assert abs(job["epsilon"] - (-2.0)) < 1e-10


# ============================================================
# 12. Integration — enqueue_jobs writes to the RIGHT manifest
# ============================================================

class TestEnqueueJobsManifestRouting:
    """The key regression: enqueue_jobs must write to manifest_path, not MANIFEST_PATH."""

    def _template(self):
        return {**COMBO_DEFAULTS, "mu": -3.5,
                "run_settings": {"beta": 1.0, "num_parallel_runs": 4}}

    def test_enqueue_writes_to_coex_manifest(self, tmp_path):
        from analyzer import enqueue_jobs

        run_all = str(tmp_path / "run_all_queue.json")
        coex = str(tmp_path / "susceptibility_coex_queue.json")
        samples = str(tmp_path / "samples")

        old = qm.MANIFEST_PATH
        qm.MANIFEST_PATH = run_all  # module default → wrong manifest
        try:
            n_added, paths = enqueue_jobs(
                [-3.5, -3.4],
                self._template(),
                samples,
                manifest_path=coex,  # <-- must honour this
                results_dir="susceptibility_results/coex",
                manage_path="susceptibility_manage.csv",
            )
        finally:
            qm.MANIFEST_PATH = old

        # Jobs must be in coex manifest
        coex_pending = qm.read_manifest(coex)["pending"]
        run_all_pending = qm.read_manifest(run_all)["pending"]

        assert len(coex_pending) == 2
        assert len(run_all_pending) == 0, (
            "BUG: jobs leaked to run_all_queue.json instead of coex manifest"
        )

    def test_enqueue_returns_count_added(self, tmp_path):
        from analyzer import enqueue_jobs

        coex = str(tmp_path / "coex.json")
        samples = str(tmp_path / "samples")

        n_added, paths = enqueue_jobs(
            [-3.5, -3.4, -3.3],
            self._template(),
            samples,
            manifest_path=coex,
        )
        assert n_added == 3
        assert len(paths) == 3

    def test_enqueue_deduplicates_across_calls(self, tmp_path):
        from analyzer import enqueue_jobs

        coex = str(tmp_path / "coex.json")
        samples = str(tmp_path / "samples")

        enqueue_jobs([-3.5], self._template(), samples, manifest_path=coex)
        n_added, _ = enqueue_jobs([-3.5, -3.4], self._template(), samples, manifest_path=coex)
        assert n_added == 1  # -3.5 already in queue

    def test_job_json_has_correct_results_base(self, tmp_path):
        from analyzer import enqueue_jobs

        coex = str(tmp_path / "coex.json")
        samples = str(tmp_path / "samples")

        _, paths = enqueue_jobs(
            [-3.5],
            self._template(),
            samples,
            manifest_path=coex,
            results_dir="susceptibility_results/coex",
        )
        with open(paths[0]) as f:
            job = json.load(f)
        assert job["results_base"] == "susceptibility_results/coex"
