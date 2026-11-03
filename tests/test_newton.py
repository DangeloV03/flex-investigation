"""Tests for flexsim.susc.newton: step rules, decisions, and a full chain on a fake simulator.

The fake simulator is a two-state (telegraph) process m = +-m_s with flip rates biased by
delta = beta*(mu - mu_true), so that <m> = m_s tanh(N m_s delta / 2), which is the
equilibrium form, with slope N m_s^2 / 2 at coexistence. It reads the real runner
command line that newton.runner_args builds and writes the same files the runner does.
"""

import csv
import math
import os

import numpy as np
import pytest

from flexsim.susc import newton
from flexsim.susc.layout import SUSCEPTIBILITY_CSV_FIELDS

L = 16
N = L * L
M_S = 0.7
MU_TRUE = -3.0


def cfg(**over):
    c = dict(newton.DEFAULT_CONFIG)
    c.update(eq0=0.0, prod0=4e4, eq_iter=0.0, seg_prod=2e4, cpus=16)
    c.update(over)
    return c


# ---------------------------------------------------------------------------
# Pure rules
# ---------------------------------------------------------------------------

def meas(m0, se=0.01, J=12.0, slope=1000.0):
    return {"m0": m0, "se": se, "J": J, "slope_var": slope}


def state_with(n_seg=1, n_iter=1, **c):
    its = [{"dir": f"iter_{i}", "mu": -3.0, "measurements": [meas(0.2)]} for i in range(n_iter - 1)]
    its.append({"dir": f"iter_{n_iter - 1}", "mu": -3.0, "measurements": [meas(0.0)] * n_seg})
    return {"config": cfg(**c), "iterations": its}


def test_step_uses_variance_slope_and_damping():
    its = [{"mu": -3.0, "measurements": []}]
    s = newton.newton_step(its, meas(0.2, slope=1000.0), cfg(tau=0.5))
    assert s["slope_kind"] == "variance"
    assert s["mu_next"] == pytest.approx(-3.0 - 0.5 * 0.2 / 1000.0)


def test_step_direction_follows_sign_of_m():
    its = [{"mu": -3.0, "measurements": []}]
    assert newton.newton_step(its, meas(+0.3), cfg())["mu_next"] < -3.0
    assert newton.newton_step(its, meas(-0.3), cfg())["mu_next"] > -3.0


def test_step_is_capped():
    its = [{"mu": -3.0, "measurements": []}]
    s = newton.newton_step(its, meas(0.5, slope=1.0), cfg(dmu_max=1e-3))
    assert s["step"] == pytest.approx(-1e-3)
    assert abs(s["step_raw"]) > 1e-3


def test_secant_used_when_clear_and_positive():
    its = [{"mu": -3.0, "measurements": [meas(0.40, se=0.01)]},
           {"mu": -3.001, "measurements": []}]
    s = newton.newton_step(its, meas(0.20, se=0.01, slope=100.0), cfg(tau=1.0, dmu_max=2e-3))
    assert s["slope_kind"] == "secant"
    assert s["slope"] == pytest.approx(200.0)
    assert s["mu_next"] == pytest.approx(-3.002)


def test_secant_used_even_when_small_but_skipped_when_wrong_sign():
    its = [{"mu": -3.0, "measurements": [meas(0.21, se=0.05)]},
           {"mu": -3.001, "measurements": []}]
    assert newton.newton_step(its, meas(0.20, se=0.05), cfg())["slope_kind"].startswith("secant")
    its[0]["measurements"] = [meas(0.10, se=0.001)]          # m went up while mu went down
    assert newton.newton_step(its, meas(0.20, se=0.001), cfg())["slope_kind"] == "variance"


def test_noisy_secant_is_clamped_to_fluctuation_slope():
    # smoke-test case: <m> -0.101 -> -0.097 over 7e-4 gives secant ~5 vs fluctuation ~70
    its = [{"mu": -3.2942, "measurements": [meas(-0.101)]},
           {"mu": -3.2935, "measurements": []}]
    s = newton.newton_step(its, meas(-0.097, slope=70.0), cfg())
    assert s["slope_kind"] == "secant-clamped" and s["slope"] == pytest.approx(35.0)
    its[0]["measurements"] = [meas(-0.9)]                    # absurdly steep secant
    s = newton.newton_step(its, meas(-0.097, slope=70.0), cfg())
    assert s["slope_kind"] == "secant-clamped" and s["slope"] == pytest.approx(350.0)


def test_default_cap_is_5e_4():
    its = [{"mu": -3.0, "measurements": []}]
    assert newton.newton_step(its, meas(0.5, slope=1.0), cfg())["step"] == pytest.approx(-5e-4)


def test_fallback_when_slope_zero():
    its = [{"mu": -3.0, "measurements": []}]
    s = newton.newton_step(its, meas(0.3, slope=0.0), cfg(dmu_max=1e-3))
    assert s["slope_kind"] == "fallback" and s["mu_next"] == pytest.approx(-3.001)


def test_judged_after_first_segment_even_with_few_jumps():
    st = state_with(n_seg=1)
    assert newton.choose_action(st, meas(0.5, J=0.5))[0] == "new_mu"


def test_any_m_over_tolerance_steps_regardless_of_se():
    st = state_with(n_seg=1)
    assert newton.choose_action(st, meas(0.08, se=0.06, J=4.0))[0] == "new_mu"
    assert newton.choose_action(st, meas(0.04, se=0.001, J=4.0))[0] == "continue"


def test_zero_jump_symmetric_start_is_not_converged():
    # half the replicas at +m_s, half at -m_s: <m> ~ 0 by construction, J = 0
    st = state_with(n_seg=5)
    assert newton.choose_action(st, meas(0.0, se=0.18, J=0.0))[0] == "continue"


def test_converged_needs_tolerance_and_target_jumps():
    st = state_with(n_seg=2)
    assert newton.choose_action(st, meas(0.03, J=12.0))[0] == "converged"
    assert newton.choose_action(st, meas(0.03, J=6.0))[0] == "continue"


def test_drift_out_at_target_triggers_new_step():
    st = state_with(n_seg=4)
    assert newton.choose_action(st, meas(0.12, se=0.02, J=11.0))[0] == "new_mu"


def test_max_iter_and_max_segments_fail():
    assert newton.choose_action(state_with(n_iter=3, max_iter=3), meas(0.3))[0] == "failed"
    assert newton.choose_action(state_with(n_seg=4, max_segments=4), meas(0.03, J=5.0))[0] == "failed"


def test_measure_slope_is_half_n_var_without_inert():
    rng = np.random.default_rng(0)
    series = []
    for _ in range(8):
        m = rng.choice([-M_S, M_S], size=500)
        series.append({"m": m, "occ": (1 + m) / 2})
    out = newton.measure_series(series, N)
    pooled = np.concatenate([s["m"] for s in series])
    assert out["slope_var"] == pytest.approx(N / 2 * pooled.var())


# ---------------------------------------------------------------------------
# Full chain on a fake simulator
# ---------------------------------------------------------------------------

def _arg(cmd, flag, n=1):
    i = cmd.index(flag)
    return cmd[i + 1] if n == 1 else cmd[i + 1:i + 1 + n]


def fake_runner(cmd, rng):
    """Honour the runner flags newton builds; write rows + timeseries like the real runner."""
    outdir = _arg(cmd, "--outdir")
    mu = float(_arg(cmd, "--mu"))
    n_samples = round(float(_arg(cmd, "--prod-time")) / newton.CHUNK_TIME)
    cpus = int(_arg(cmd, "--cpus"))
    resume_dir = _arg(cmd, "--resume-dir") if "--resume-dir" in cmd else None
    fresh = "--fresh-history" in cmd
    ids = None
    if "--resume-ids" in cmd:  # ids run until the next flag
        ids = []
        for x in cmd[cmd.index("--resume-ids") + 1:]:
            if x.startswith("--"):
                break
            ids.append(int(x))
    os.makedirs(outdir, exist_ok=True)
    data_csv = os.path.join(outdir, "susceptibility_data.csv")
    prior = list(csv.DictReader(open(data_csv))) if os.path.isfile(data_csv) else []
    next_id = max([int(r["id"]) for r in prior], default=-1) + 1

    delta = mu - MU_TRUE
    p_up = 0.04 * math.exp(+delta * N * M_S / 2)   # rate - -> +
    p_dn = 0.04 * math.exp(-delta * N * M_S / 2)   # rate + -> -
    rows = []
    slots = ids if ids is not None else [None] * cpus
    for slot, rid in enumerate(slots):
        hist = []
        if rid is not None:
            last = int(np.load(os.path.join(resume_dir, f"final_lattice_{rid}.npy")))
            if not fresh:
                hist = list(np.loadtxt(os.path.join(resume_dir, f"m_timeseries_{rid}.csv"),
                                       delimiter=",", skiprows=1, usecols=4, ndmin=1))
        else:
            last = 1 if slot % 2 == 0 else -1
        s, new = last, []
        for _ in range(n_samples):
            if s > 0 and rng.random() < min(1.0, p_dn):
                s = -1
            elif s < 0 and rng.random() < min(1.0, p_up):
                s = 1
            new.append(s * M_S + 0.02 * rng.standard_normal())
        m = np.array(hist + new)
        run_id = next_id + slot
        np.save(os.path.join(outdir, f"final_lattice_{run_id}.npy"), np.array(s))
        with open(os.path.join(outdir, f"m_timeseries_{run_id}.csv"), "w") as fh:
            fh.write("chunk,rho_bonding,rho_inert,rho_empty,m,energy\n")
            for i, v in enumerate(m):
                fh.write(f"{i},{(1 + v) / 2},0.0,{(1 - v) / 2},{v},0.0\n")
        row = {f: "" for f in SUSCEPTIBILITY_CSV_FIELDS}
        row.update(id=run_id, replica_id=slot, epsilon=-1.7, L=L, Lx=L, Ly=L, mu=mu,
                   m_mean=float(m.mean()), prod_time=len(m) * newton.CHUNK_TIME, beta=1.0)
        rows.append(row)
    with open(data_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUSCEPTIBILITY_CSV_FIELDS)
        if not prior:
            w.writeheader()
        w.writerows(rows)


def make_chain(tmp_path, mu0, **c):
    pair = str(tmp_path / "base" / "_16_16_S3_DF0.0_DMU2.0_K1.0" / "_-1.7")
    os.makedirs(pair)
    state = {"version": 1, "epsilon": -1.7, "L": L, "beta": 1.0,
             "physics": {"delta_f": 0.0, "delta_mu": 2.0, "k": 1.0, "scheme": "positive_drive"},
             "mu_start": mu0, "config": cfg(**c), "status": "running",
             "iterations": [{"dir": "iter_0", "mu": mu0, "measurements": [], "retries": 0}],
             "next": {"kind": "start"}, "jobs": []}
    newton.save_state(pair, state)
    return pair


def run_chain(pair, rng, max_steps=60):
    queue = []

    def submit(p, st):
        queue.append(newton.runner_args(p, st, st["config"]["cpus"]))

    submit(pair, newton.load_state(pair))
    for _ in range(max_steps):
        if not queue:
            break
        fake_runner(queue.pop(0), rng)
        if newton.decide(pair, submit=submit) != "running":
            break
    return newton.load_state(pair)


@pytest.mark.parametrize("offset", [+0.004, -0.006, 0.0])
def test_chain_converges_to_true_mu(tmp_path, offset):
    rng = np.random.default_rng(1)
    st = run_chain(make_chain(tmp_path, MU_TRUE + offset), rng)
    assert st["status"] == "converged"
    last = st["iterations"][-1]
    assert abs(last["measurements"][-1]["m0"]) <= 0.05
    assert last["measurements"][-1]["J"] >= 10
    # |<m>| <= 0.05 means |delta| <~ 8e-4 here, but with no SE test a point can pass on
    # noise at J ~ 10-16 (seen: true <m> ~ 0.11 measured as 0.047), so allow ~2x that.
    assert abs(last["mu"] - MU_TRUE) < 2.5e-3
    if offset != 0.0:
        assert len(st["iterations"]) >= 2
        assert abs(last["mu"] - MU_TRUE) < abs(offset)
    # each iteration's data was taken at that iteration's mu only
    pair = os.path.join(str(tmp_path), "base", "_16_16_S3_DF0.0_DMU2.0_K1.0", "_-1.7")
    for it in st["iterations"]:
        rows = list(csv.DictReader(open(os.path.join(pair, it["dir"], "susceptibility_data.csv"))))
        assert all(abs(float(r["mu"]) - it["mu"]) < 1e-9 for r in rows)


def test_stop_file_halts_chain(tmp_path):
    pair = make_chain(tmp_path, MU_TRUE)
    open(os.path.join(str(tmp_path), "base", "STOP"), "w").close()
    st = run_chain(pair, np.random.default_rng(2))
    assert st["status"] == "stopped"


def test_empty_segment_is_retried_then_fails(tmp_path):
    pair = make_chain(tmp_path, MU_TRUE, max_retries=1)
    calls = []
    assert newton.decide(pair, submit=lambda p, s: calls.append(1)) == "running"
    assert newton.decide(pair, submit=lambda p, s: calls.append(1)) == "failed"
    assert len(calls) == 1


def test_runner_args_per_action(tmp_path):
    pair = make_chain(tmp_path, -3.0)
    st = newton.load_state(pair)
    a = newton.runner_args(pair, st, 16)
    assert "--resume-dir" not in a and _arg(a, "--prod-time") == repr(4e4)
    st["iterations"].append({"dir": "iter_1", "mu": -3.001, "measurements": []})
    st["next"] = {"kind": "new_mu", "resume_ids": [3, 1]}
    a = newton.runner_args(pair, st, 16)
    assert "--fresh-history" in a
    assert _arg(a, "--resume-dir").endswith("iter_0") and _arg(a, "--outdir").endswith("iter_1")
    assert _arg(a, "--resume-ids", 2) == ["3", "1"]           # slot order kept
    assert int(_arg(a, "--seed-base")) == 7000 + newton.SEED_STRIDE
    st["next"] = {"kind": "continue", "resume_ids": [5]}
    a = newton.runner_args(pair, st, 16)
    assert "--fresh-history" not in a and _arg(a, "--resume-dir") == _arg(a, "--outdir")
