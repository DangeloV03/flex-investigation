#!/usr/bin/env python3
"""
End-to-end audit of the susceptibility coex job pipeline.

Traces: manage.csv → analyzer → susceptibility_coex_queue.json →
run_susceptibility_all.py → sbatch (flex_sim) → json_runner.py → output.csv

Usage (on Della):
    python scripts/trace_coex_pipeline.py
    python scripts/trace_coex_pipeline.py --run-analyzer-once   # one analyzer cycle
    python scripts/trace_coex_pipeline.py --run-dispatch-once     # one dispatch cycle
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzer import (
    N_INITIAL_MU_POINTS,
    N_REFINEMENT_POINTS,
    PSI_COEX_MAX,
    analyze_combo,
    build_curves,
    count_in_bracket,
    find_manage_row,
    has_phi_sign_change,
    interior_psi_minimum,
    is_psi_minimum_acceptable,
    min_psi_value,
    read_manage,
    refinement_mus,
    select_depth_first_combo,
    sign_change_bracket,
)
from combo_paths import COMBO_KEY_FIELDS, combo_dir_name, discover_combo_results
from queue_manifest import read_manifest
from susceptibility_paths import (
    COEX_MANIFEST,
    COEX_RESULTS_DIR,
    COEX_SAMPLES_DIR,
    MANAGE_CSV,
    patch_coex_job_json,
)

EXPECTED_COMMITS = (
    "094c1f4",  # results_base in analyzer enrich
    "9fc3641",  # dispatch-time patch
)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def check_processes() -> None:
    section("1. Long-running processes (BOTH required for Slurm jobs)")
    print(
        "Pipeline needs two daemons (see scripts/start_sus_coex_daemons.sh):\n"
        "  A) analyzer.py          → writes pending paths to susceptibility_coex_queue.json\n"
        "  B) run_susceptibility_all.py --phase coex → sbatch flex_sim from that queue\n"
    )
    for label, pattern in [
        ("coex dispatcher", "run_susceptibility_all.py --phase coex"),
        ("coex analyzer", "analyzer.py"),
    ]:
        out = run_cmd(["pgrep", "-af", pattern])
        lines = [ln for ln in out.splitlines() if "trace_coex_pipeline" not in ln]
        if lines:
            print(f"  OK  {label}:")
            for ln in lines:
                print(f"      {ln}")
        else:
            print(f"  MISSING  {label} — nothing will reach Slurm without it")

    tmux = run_cmd(["tmux", "list-sessions"])
    if "sus-coex" in tmux:
        print("\n  tmux session sus-coex exists:")
        for win in run_cmd(["tmux", "list-windows", "-t", "sus-coex"]).splitlines():
            print(f"      {win}")
        print(
            "\n  If processes above are MISSING but tmux exists, the session is STALE.\n"
            "  Fix: tmux kill-session -t sus-coex && ./scripts/start_sus_coex_daemons.sh"
        )
    else:
        print("\n  tmux session sus-coex: not running")
        print("  Start: ./scripts/start_sus_coex_daemons.sh")


def check_code_version() -> None:
    section("2. Code version (wrong-dir bug fixes)")
    head = run_cmd(["git", "rev-parse", "--short", "HEAD"])
    print(f"  HEAD: {head}")
    for commit in EXPECTED_COMMITS:
        ok = (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                cwd=ROOT,
            ).returncode
            == 0
        )
        print(f"  {'OK' if ok else 'MISSING'}  contains {commit}")


def check_queue() -> dict:
    section("3. Queue manifest (analyzer → dispatcher handoff)")
    print(f"  File: {COEX_MANIFEST}")
    m = read_manifest(COEX_MANIFEST)
    pending = m.get("pending", [])
    in_flight = m.get("in_flight", {})
    print(f"  pending={len(pending)}  in_flight={len(in_flight)}")
    if pending:
        print("  Next pending (up to 5):")
        for p in pending[:5]:
            exists = os.path.isfile(p)
            print(f"    {'OK' if exists else 'MISSING FILE'}  {p}")
    else:
        print(
            "  pending is empty → dispatcher has nothing to sbatch.\n"
            "  Either analyzer has not enqueued, or all jobs finished."
        )
    if in_flight:
        print("  in_flight:")
        for jid, p in list(in_flight.items())[:5]:
            print(f"    job {jid}: {p}")

    wrong = read_manifest("run_all_queue.json")
    leaked = [
        p for p in wrong.get("pending", []) + list(wrong.get("in_flight", {}).values())
        if "susceptibility_samples/coex" in p or "homo_Ly16_mu" in p
    ]
    if leaked:
        print(
            f"\n  LEAK: {len(leaked)} coex job path(s) found in run_all_queue.json "
            f"(analyzer prepend bug — coex dispatcher never sees these):"
        )
        for p in leaked[:5]:
            print(f"    {p}")
        print(
            "  Fix: git pull (prepend_pending path= fix), restart sus-coex, re-run analyzer once."
        )
    return m


def audit_json_paths(paths: list[str]) -> None:
    section("4. Job JSON path fields (where json_runner writes)")
    if not paths:
        samples = sorted(Path(COEX_SAMPLES_DIR).glob("homo_Ly16_mu*.json"))
        paths = [str(p) for p in samples[-3:]]
    if not paths:
        print("  No homo_Ly16_mu*.json files to inspect.")
        return
    for p in paths[:8]:
        if not os.path.isfile(p):
            print(f"  MISSING  {p}")
            continue
        with open(p) as f:
            j = json.load(f)
        rb = j.get("results_base")
        mc = j.get("manage_csv")
        ok = rb == COEX_RESULTS_DIR and mc == MANAGE_CSV
        print(
            f"  {'OK' if ok else 'BAD'}  {os.path.basename(p)}: "
            f"results_base={rb!r} manage_csv={mc!r}"
        )
        if not ok:
            patched = patch_coex_job_json(p)
            print(f"       patch_coex_job_json would fix: {patched}")


def count_outputs() -> None:
    section("5. Simulation outputs (correct vs wrong directory)")
    coex = int(
        subprocess.check_output(
            f"find {COEX_RESULTS_DIR} -name output.csv 2>/dev/null | wc -l",
            shell=True,
            text=True,
        ).strip()
        or 0
    )
    stray = int(
        subprocess.check_output(
            "find results -path '*160x16*' -name output.csv 2>/dev/null | wc -l",
            shell=True,
            text=True,
        ).strip()
        or 0
    )
    print(f"  {COEX_RESULTS_DIR}/.../output.csv  count={coex}")
    print(f"  results/.../160x16.../output.csv   count={stray} (coex stray; main campaign also uses results/)")
    print(
        f"\n  json_runner resolves outdir as:\n"
        f"    {{results_base}}/{{combo_dir}}/mu_sweeps/mu{{tag}}/output.csv\n"
        f"  Default results_base if missing: results/  ← the old bug"
    )


def plan_for_combo(epsilon: float, data: dict, row: dict) -> str:
    job = data["job"]
    tag = combo_dir_name(job)
    mu_vals, phi_vals, _, psi_vals, _ = build_curves(data["points"])
    n_points = len(data["points"])
    analyzed = bool(str(row.get("isAnalyzed", "")).strip())
    n_req = int(row.get("RequestForAdditionalData") or 0)

    if analyzed:
        return "already analyzed — no jobs"
    if n_points < N_INITIAL_MU_POINTS:
        return f"wait for initial batch ({n_points}/{N_INITIAL_MU_POINTS})"
    if not has_phi_sign_change(phi_vals):
        return "extend mu window (analyzer will enqueue)"
    if is_psi_minimum_acceptable(psi_vals):
        return "FINALIZE ONLY — psi OK, no Slurm needed (use finalize script or analyzer)"
    bracket = sign_change_bracket(mu_vals, phi_vals)
    if bracket is None:
        return "stuck — sign change but no bracket"
    n_new = len(refinement_mus(bracket[0], bracket[1], mu_vals))
    in_br = count_in_bracket(mu_vals, bracket[0], bracket[1])
    if n_new == 0:
        return (
            f"stuck — bad psi, no new mu to enqueue "
            f"(in_bracket={in_br}/{N_REFINEMENT_POINTS})"
        )
    return f"SHOULD ENQUEUE {n_new} Slurm job(s) (RequestForAdditionalData={n_req})"


def check_manage_and_plans() -> None:
    section("6. manage.csv rows → expected next action")
    grouped = discover_combo_results(COEX_RESULTS_DIR)
    rows = read_manage(MANAGE_CSV)
    if not rows:
        print(f"  No rows in {MANAGE_CSV}")
        return

    n_open = 0
    n_finalize = 0
    n_enqueue = 0
    for row in sorted(rows, key=lambda r: float(r["epsilon"])):
        combo = {f: row[f] for f in COMBO_KEY_FIELDS}
        key = tuple(str(combo[f]) for f in COMBO_KEY_FIELDS)
        data = grouped.get(key)
        eps = row["epsilon"]
        analyzed = str(row.get("isAnalyzed", "")).strip()
        if analyzed:
            continue
        n_open += 1
        if data is None:
            print(f"  eps={eps}: NO RESULT DATA on disk")
            continue
        plan = plan_for_combo(float(eps), data, row)
        n_points = len(data["points"])
        _, _, _, psi_vals, _ = build_curves(data["points"])
        psi_min = min_psi_value(psi_vals)
        print(
            f"  eps={eps}  n_mu={n_points}  min_psi={psi_min:.4f}  "
            f"req={row.get('RequestForAdditionalData', 0)}"
        )
        print(f"    → {plan}")
        if plan.startswith("FINALIZE"):
            n_finalize += 1
        elif plan.startswith("SHOULD ENQUEUE"):
            n_enqueue += 1

    if n_open == 0:
        print("  All rows analyzed — coex phase complete; empty queue is expected.")
    else:
        print(
            f"\n  Summary: {n_open} open row(s), "
            f"{n_enqueue} need Slurm, {n_finalize} need finalize-only"
        )
        if n_enqueue:
            print(
                "\n  If pending=0 but rows need Slurm, restart analyzer after git pull "
                "(old process may have stale in-memory skip state; fixed in latest analyzer.py)."
            )
        if n_finalize and not n_enqueue:
            print(
                "\n  NOTE: Empty queue + no flex_sim is EXPECTED right now.\n"
                "  Run: python scripts/finalize_susceptibility_coex.py"
            )


def check_depth_first_focus() -> None:
    section("7. Analyzer depth-first focus (only one epsilon per poll)")
    grouped = discover_combo_results(COEX_RESULTS_DIR)
    key = select_depth_first_combo(grouped, MANAGE_CSV)
    if key is None:
        print("  No unfinished combo — analyzer idle (sleep loop)")
    else:
        eps = grouped[key]["job"]["epsilon"]
        rows = read_manage(MANAGE_CSV)
        combo = {f: grouped[key]["job"][f] for f in COMBO_KEY_FIELDS}
        idx = find_manage_row(rows, combo)
        n_req = int(rows[idx].get("RequestForAdditionalData") or 0) if idx is not None else -1
        print(f"  Would focus epsilon={eps}  RequestForAdditionalData={n_req}")


def run_analyzer_once() -> None:
    section("8. Running ONE analyzer cycle (--run-analyzer-once)")
    pending_before = len(read_manifest(COEX_MANIFEST).get("pending", []))
    grouped = discover_combo_results(COEX_RESULTS_DIR)
    active_key = select_depth_first_combo(grouped, MANAGE_CSV)
    if active_key is None:
        print("  Nothing to analyze.")
        return
    pending_points: dict = {}
    processed: set = set()
    for combo_key, data in grouped.items():
        if combo_key != active_key:
            continue
        rows = read_manage(MANAGE_CSV)
        combo = {f: data["job"][f] for f in COMBO_KEY_FIELDS}
        idx = find_manage_row(rows, combo)
        if idx is not None and rows[idx].get("isAnalyzed", ""):
            print("  Focus combo already analyzed.")
            return
        print(f"  Calling analyze_combo for epsilon={data['job']['epsilon']}")
        analyze_combo(
            combo_key,
            data,
            MANAGE_CSV,
            COEX_RESULTS_DIR,
            COEX_SAMPLES_DIR,
            COEX_MANIFEST,
            pending_points,
        )
    pending_after = len(read_manifest(COEX_MANIFEST).get("pending", []))
    print(f"  Queue pending: {pending_before} → {pending_after}")


def run_dispatch_once() -> None:
    section("9. Running ONE dispatch cycle (--run-dispatch-once)")
    cmd = [sys.executable, "-u", "run_susceptibility_all.py", "--phase", "coex", "--once"]
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace susceptibility coex pipeline")
    parser.add_argument(
        "--run-analyzer-once",
        action="store_true",
        help="Run a single analyzer cycle and show queue delta",
    )
    parser.add_argument(
        "--run-dispatch-once",
        action="store_true",
        help="Run run_susceptibility_all.py --phase coex --once",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    print("Coex pipeline trace (cwd=%s)" % os.getcwd())

    check_processes()
    check_code_version()
    m = check_queue()
    audit_json_paths(m.get("pending", []) + list(m.get("in_flight", {}).values()))
    count_outputs()
    check_manage_and_plans()
    check_depth_first_focus()

    if args.run_analyzer_once:
        run_analyzer_once()
        check_queue()
    if args.run_dispatch_once:
        run_dispatch_once()
        check_queue()

    section("10. Slurm check")
    sq = run_cmd(["squeue", "-u", os.environ.get("USER", ""), "-n", "flex_sim"])
    if sq:
        print(sq)
    else:
        print("  No flex_sim jobs in squeue (normal if pending=0 or dispatch not running)")


if __name__ == "__main__":
    main()
