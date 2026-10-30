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


def write_coex(path, eps_values, mu_fn, err=1e-4):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epsilon", "delta_f", "delta_mu", "k", "scheme", "mu_coex_FITTED", "mu_coex_FITTED_error"])
        for e in eps_values:
            w.writerow([e, 0.0, 1.0, 1.0, "homo", mu_fn(e), err])


def test_fit_on_new_grid_outside_campaign(campaign, tmp_path):
    """--fit-min/max pick the coex points; --eps-min/max/step pick where mu is written."""
    from flexsim.susc.sweep import load_mu_map

    cubic = lambda e: 0.4 * (e + 1.7) ** 3 + 0.1 * (e + 1.7) ** 2 + 2.17 * e + 0.45
    coex = str(tmp_path / "manage.csv")
    write_coex(coex, [round(-1.95 + 0.01 * i, 4) for i in range(41)], cubic)
    out = str(tmp_path / "work")
    coex_refit.do_fit(argparse.Namespace(
        campaign=campaign, coex=coex, degree=3, max_fit_err=1e-3, out=out,
        fit_min=-1.90, fit_max=-1.60, eps_min=-1.900042, eps_max=-1.825042, eps_step=0.005))

    rows = list(csv.DictReader(open(os.path.join(out, "fitted_mu.csv"))))
    assert len(rows) == 16
    assert float(rows[0]["epsilon"]) == -1.900042 and float(rows[-1]["epsilon"]) == -1.825042
    for r in rows:
        assert abs(float(r["mu_coex_FITTED"]) - cubic(float(r["epsilon"])) ) < 1e-8
        assert r["mu_campaign"] == "" and r["shift"] == ""      # the campaign never ran these
    # The sweep must find a mu for every eps on the same grid it builds itself.
    from flexsim.grid import frange
    mu_map = load_mu_map(os.path.join(out, "fitted_mu.csv"), delta_f="0.0", delta_mu="1.0",
                         k="1.0", scheme="homo")
    assert all(round(e, 6) in mu_map for e in frange(-1.900042, -1.825042, 0.005))


def test_fit_grid_flags_must_come_together(campaign, tmp_path):
    coex = str(tmp_path / "manage.csv")
    write_coex(coex, [-1.80, -1.75, -1.70, -1.65], lambda e: 2 * e)
    with pytest.raises(SystemExit, match="--eps-min"):
        coex_refit.do_fit(argparse.Namespace(
            campaign=campaign, coex=coex, degree=1, max_fit_err=1e-3, out=str(tmp_path / "w"),
            fit_min=None, fit_max=None, eps_min=-1.8, eps_max=None, eps_step=None))


def test_combine_links_kept_points_and_lists_dropped(campaign, tmp_path):
    new = str(tmp_path / "NEW")
    make_run(new, 16, -1.85, [0.02, 0.0], mu=-3.2)       # kept
    make_run(new, 32, -1.85, [0.09, 0.07], mu=-3.2)      # dropped: |<m>| = 0.08
    out = str(tmp_path / "combined")
    args = argparse.Namespace(base=[campaign, new], max_abs_m=0.05, out=out)
    coex_refit.do_combine(args)

    kept = {(int(r["L"]), float(r["epsilon"])) for r in csv.DictReader(open(os.path.join(out, "kept_points.csv")))}
    dropped = {(int(r["L"]), float(r["epsilon"])) for r in csv.DictReader(open(os.path.join(out, "dropped_points.csv")))}
    assert kept == {(16, -1.70), (32, -1.70), (16, -1.85)}
    assert dropped == {(16, -1.80), (32, -1.80), (16, -1.74), (32, -1.74), (32, -1.85)}
    # The combined folder reads like a campaign, and only holds links.
    assert set(coex_refit.campaign_runs(out)) == kept
    link = os.path.join(out, PARAM_DIR.format(L=16), "_-1.85")
    assert os.path.islink(link) and os.path.realpath(link) == os.path.realpath(
        os.path.join(new, PARAM_DIR.format(L=16), "_-1.85"))

    coex_refit.do_combine(args)                          # rebuilding is fine
    assert os.path.isdir(os.path.join(new, PARAM_DIR.format(L=32), "_-1.85"))   # data untouched


def test_combine_refuses_overlap_and_real_folders(campaign, tmp_path):
    with pytest.raises(SystemExit, match="in both"):
        coex_refit.do_combine(argparse.Namespace(base=[campaign, campaign], max_abs_m=0.05,
                                                 out=str(tmp_path / "c1")))
    with pytest.raises(SystemExit, match="more than links"):
        coex_refit.do_combine(argparse.Namespace(base=[campaign], max_abs_m=0.05,
                                                 out=os.path.dirname(campaign)))
