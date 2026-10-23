"""Tests for flexsim.susc.coex_refit (fit, select, promote) on a tiny fake campaign."""

import argparse
import csv
import os

import numpy as np
import pytest

from flexsim.susc import coex_refit
from flexsim.susc.layout import SUSCEPTIBILITY_CSV_FIELDS

PARAM_DIR = "_{L}_{L}_S1_DF0.0_DMU1.0_K1.0"


def make_run(base, L, eps, m_values, mu, prod_time=2e6):
    run_dir = os.path.join(base, PARAM_DIR.format(L=L), f"_{eps:g}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "susceptibility_data.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUSCEPTIBILITY_CSV_FIELDS)
        w.writeheader()
        for i, m in enumerate(m_values):
            row: dict[str, object] = {f: "" for f in SUSCEPTIBILITY_CSV_FIELDS}
            row.update(id=i, replica_id=i, epsilon=eps, delta_f=0.0, delta_mu=1.0, k=1.0,
                       scheme="homo", L=L, Lx=L, Ly=L, mu=mu, m_mean=m, prod_time=prod_time,
                       beta=1.0)
            w.writerow(row)
    return run_dir


@pytest.fixture
def campaign(tmp_path):
    base = str(tmp_path / "SUSC_RUNS_X")
    for L in (16, 32):
        make_run(base, L, -1.80, [0.30, 0.32], mu=-3.10)    # cold and off: skipped by min_eps
        make_run(base, L, -1.74, [0.20, 0.22], mu=-2.96)    # off: rerun
        make_run(base, L, -1.70, [0.01, -0.01], mu=-2.87)   # fine
    return base


def test_robust_polyfit_recovers_line_and_drops_bad_points():
    eps = np.linspace(-1.8, -1.6, 21)
    mu = 2.1 * eps + 0.7 + np.random.default_rng(0).normal(0, 1e-5, eps.size)
    err = np.full(eps.size, 1e-4)
    mu[5] += 2e-3            # one jumped fit
    err[10] = 8e-3           # one fit the analyzer flagged
    coef, x0, resid, keep = coex_refit.robust_polyfit(eps, mu, err, degree=1, max_err=1e-3)
    assert not keep[5] and not keep[10]
    assert abs(coef[0] - 2.1) < 1e-3
    assert abs(np.polyval(coef, -1.7 - x0) - (2.1 * -1.7 + 0.7)) < 1e-4


def test_select_flags_off_eps_and_skips_cold(campaign, tmp_path):
    out = str(tmp_path / "work")
    coex_refit.do_select(argparse.Namespace(campaign=campaign, threshold=0.05, min_eps=-1.775,
                                            out=out))
    rerun = [float(r["epsilon"]) for r in csv.DictReader(open(os.path.join(out, "rerun_eps.csv")))]
    cold = [float(r["epsilon"]) for r in csv.DictReader(open(os.path.join(out, "skipped_cold_eps.csv")))]
    assert rerun == [-1.74]
    assert cold == [-1.80]


def test_promote_dry_run_then_swap(campaign, tmp_path):
    work = tmp_path / "work"
    staging = str(work / "runs")
    archive = str(work / "archived_runs")
    for L in (16, 32):
        make_run(staging, L, -1.74, [0.01, 0.0], mu=-2.9605)
    eps_file = work / "rerun_eps.csv"
    eps_file.write_text("epsilon,worst_L,worst_m,failing_L\n-1.74,32,+0.22,16 32\n")
    args = argparse.Namespace(campaign=campaign, staging=staging, archive=archive,
                              eps_file=str(eps_file), threshold=10.0, yes=False, force=True)

    coex_refit.do_promote(args)                      # dry run: nothing moves
    assert not os.path.exists(archive)
    old = os.path.join(campaign, PARAM_DIR.format(L=16), "_-1.74")
    assert float(coex_refit.latest_generation(old)[0]["mu"]) == -2.96

    args.yes = True
    coex_refit.do_promote(args)
    assert float(coex_refit.latest_generation(old)[0]["mu"]) == -2.9605      # rerun in place
    arch = os.path.join(archive, PARAM_DIR.format(L=16), "_-1.74")
    assert float(coex_refit.latest_generation(arch)[0]["mu"]) == -2.96       # old kept
    assert not os.path.exists(os.path.join(staging, PARAM_DIR.format(L=16), "_-1.74"))
    untouched = os.path.join(campaign, PARAM_DIR.format(L=16), "_-1.7")
    assert float(coex_refit.latest_generation(untouched)[0]["mu"]) == -2.87
    log = list(csv.DictReader(open(os.path.join(campaign, "promote_log.csv"))))
    assert len(log) == 2


def test_promote_refuses_when_archive_target_exists(campaign, tmp_path):
    work = tmp_path / "work"
    staging = str(work / "runs")
    archive = str(work / "archived_runs")
    make_run(staging, 16, -1.74, [0.0, 0.0], mu=-2.9605)
    os.makedirs(os.path.join(archive, PARAM_DIR.format(L=16), "_-1.74"))
    moves, problems = coex_refit.plan_promotion(campaign, staging, archive, [-1.74])
    assert any("already exists" in p for p in problems)


def test_promote_refuses_unconverged_without_force(campaign, tmp_path):
    work = tmp_path / "work"
    staging = str(work / "runs")
    make_run(staging, 16, -1.74, [0.0, 0.0], mu=-2.9605)
    eps_file = work / "rerun_eps.csv"
    eps_file.write_text("epsilon,worst_L,worst_m,failing_L\n-1.74,16,+0.2,16\n")
    args = argparse.Namespace(campaign=campaign, staging=staging, archive=str(work / "a"),
                              eps_file=str(eps_file), threshold=10.0, yes=True, force=False)
    with pytest.raises(SystemExit):
        coex_refit.do_promote(args)
    assert os.path.exists(os.path.join(staging, PARAM_DIR.format(L=16), "_-1.74"))


def test_eps_comes_from_data_not_folder_name(tmp_path):
    """Folder names keep 6 significant figures; the true eps must survive."""
    base = str(tmp_path / "SUSC_RUNS_Y")
    make_run(base, 16, -1.568011, [0.1, 0.12], mu=-3.0)
    assert os.path.basename(coex_refit.campaign_runs(base)[(16, -1.568011)]) == "_-1.56801"
    assert coex_refit.m_table(base)[0]["epsilon"] == -1.568011
