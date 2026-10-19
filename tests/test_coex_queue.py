"""Dispatcher queue (run_all_queue.json), job JSONs, and enqueueing to the right manifest."""

from __future__ import annotations

import json
import os

import numpy as np

import flexsim.coex.queue_manifest as qm
from flexsim.coex.refine_jobs import make_job_json
from coex_helpers import COMBO_DEFAULTS


# ============================================================
# queue_manifest tests
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
# Analyzer — make_job_json
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
            results_base="susceptibility/results/coex",
        )
        with open(path) as f:
            job = json.load(f)
        assert job["results_base"] == "susceptibility/results/coex"

    def test_make_job_json_sets_manage_csv(self, tmp_path):
        path = make_job_json(
            self._template(), mu=-3.5, samples_dir=str(tmp_path),
            manage_csv="susceptibility/manage.csv",
        )
        with open(path) as f:
            job = json.load(f)
        assert job["manage_csv"] == "susceptibility/manage.csv"

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
# Integration — enqueue_jobs writes to the RIGHT manifest
# ============================================================

class TestEnqueueJobsManifestRouting:
    """The key regression: enqueue_jobs must write to manifest_path, not MANIFEST_PATH."""

    def _template(self):
        return {**COMBO_DEFAULTS, "mu": -3.5,
                "run_settings": {"beta": 1.0, "num_parallel_runs": 4}}

    def test_enqueue_writes_to_coex_manifest(self, tmp_path):
        from flexsim.coex.refine_jobs import enqueue_jobs

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
                results_dir="susceptibility/results/coex",
                manage_path="susceptibility/manage.csv",
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
        from flexsim.coex.refine_jobs import enqueue_jobs

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
        from flexsim.coex.refine_jobs import enqueue_jobs

        coex = str(tmp_path / "coex.json")
        samples = str(tmp_path / "samples")

        enqueue_jobs([-3.5], self._template(), samples, manifest_path=coex)
        n_added, _ = enqueue_jobs([-3.5, -3.4], self._template(), samples, manifest_path=coex)
        assert n_added == 1  # -3.5 already in queue

    def test_job_json_has_correct_results_base(self, tmp_path):
        from flexsim.coex.refine_jobs import enqueue_jobs

        coex = str(tmp_path / "coex.json")
        samples = str(tmp_path / "samples")

        _, paths = enqueue_jobs(
            [-3.5],
            self._template(),
            samples,
            manifest_path=coex,
            results_dir="susceptibility/results/coex",
        )
        with open(paths[0]) as f:
            job = json.load(f)
        assert job["results_base"] == "susceptibility/results/coex"
