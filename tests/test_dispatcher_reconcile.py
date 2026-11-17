"""reconcile_in_flight: which in-flight jobs are archived, re-queued, or left alone."""

import flexsim.coex.dispatcher as d


def _run(monkeypatch, states, mine=()):
    in_flight = {jid: f"samples/{jid}.json" for jid in states}
    finished, removed = [], []
    monkeypatch.setattr(d, "read_manifest", lambda: {"pending": [], "in_flight": in_flight})
    monkeypatch.setattr(d, "active_slurm_ids", lambda slurm: set(mine))
    monkeypatch.setattr(d, "sacct_state", lambda jid: states[jid])
    monkeypatch.setattr(d, "finish_job", lambda path, success: finished.append((path, success)))
    monkeypatch.setattr(d, "remove_in_flight", lambda jid: removed.append(jid))
    d.reconcile_in_flight(slurm=None)
    return finished, removed


def test_other_accounts_live_jobs_are_left_alone(monkeypatch):
    # Not in `squeue --me` (another account submitted them) but still alive in sacct.
    finished, removed = _run(monkeypatch, {"1": "RUNNING", "2": "PENDING"})
    assert finished == [] and removed == []


def test_finished_jobs_are_archived_or_requeued(monkeypatch):
    finished, removed = _run(monkeypatch, {"1": "COMPLETED", "2": "FAILED", "3": "RUNNING"}, mine={"3"})
    assert finished == [("samples/1.json", True), ("samples/2.json", False)]
    assert removed == ["1", "2"]


def test_unknown_sacct_state_is_kept_for_retry(monkeypatch):
    finished, removed = _run(monkeypatch, {"1": None})
    assert finished == [] and removed == []
