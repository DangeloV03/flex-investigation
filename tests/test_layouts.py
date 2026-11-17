"""Directory layout helpers for coex combos and susceptibility runs."""

from __future__ import annotations

import json

from flexsim.coex.layout import combo_dir_name, mu_dir_name, param_tag
from flexsim.susc.layout import COEX_RESULTS_DIR, MANAGE_CSV, coex_combo_dir, patch_coex_job_json
from coex_helpers import COMBO_DEFAULTS


# ============================================================
# coex.layout helpers
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
# susc.layout helpers
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


def test_latest_row_per_replica_keeps_replicas_behind_a_partial_round():
    """A top-up round that finished for only half the replicas must not hide the rest."""
    from flexsim.susc.layout import latest_row_per_replica

    rows = []
    for gen, pt in enumerate((1e6, 2e6)):
        for rep in range(4):
            rows.append({"id": str(4 * gen + rep), "replica_id": str(rep),
                         "prod_time": str(pt), "m_mean": str(0.7 if rep % 2 == 0 else -0.7)})
    rows.append({"id": "8", "replica_id": "0", "prod_time": "3e6", "m_mean": "0.7"})   # partial round
    rows.append({"id": "9", "replica_id": "2", "prod_time": "3e6", "m_mean": "0.7"})
    latest = latest_row_per_replica(rows)
    assert [r["id"] for r in latest] == ["5", "7", "8", "9"]          # one row per replica
    assert sum(float(r["m_mean"]) for r in latest) == 0.0              # not the +0.7 of the partial round


def test_latest_row_per_replica_falls_back_without_replica_ids():
    from flexsim.susc.layout import latest_row_per_replica

    rows = [{"id": "0", "replica_id": "", "prod_time": "1e6"},
            {"id": "1", "replica_id": "", "prod_time": "2e6"}]
    assert [r["id"] for r in latest_row_per_replica(rows)] == ["1"]
