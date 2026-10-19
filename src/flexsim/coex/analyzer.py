"""
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
    python run/coex.py analyze [--results results] [--manage manage.csv] [--interval 10]

Modules:
  phi_psi.py        phi(mu), psi(mu) from output.csv files; per-combo plot
  criteria.py       thresholds and decision rules (resolved? which mu next? fit mu_coex)
  refine_steps.py   actions: queue more mu points, finalize a combo
  analyze_combo.py  the per-combo decision tree
  refine_jobs.py    write job JSONs and push them onto the queue
  manage_csv.py     read/update manage.csv
"""

from __future__ import annotations

import argparse
import time

from flexsim.coex.layout import COMBO_KEY_FIELDS, RESULTS_DIR, discover_combo_results
from flexsim.coex.manage_csv import find_manage_row, read_manage
from flexsim.coex.analyze_combo import analyze_combo, select_depth_first_combo


MANAGE_CSV = "manage.csv"
SAMPLES_DIR = "samples"
POLL_INTERVAL = 10.0  # seconds
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
                        help="Queue manifest for the dispatcher")
    parser.add_argument(
        "--depth-first",
        action="store_true",
        help="Finish mu refinement for one epsilon before starting the next",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one poll cycle then exit (debugging)",
    )
    args = parser.parse_args()

    mode = "depth-first" if args.depth_first else "breadth-first"
    print(f"[analyzer] Watching '{args.results}' every {args.interval}s "
          f"({mode}, Ctrl-C to stop)")

    pending_points: dict[tuple, int] = {}  # point count when last follow-up was queued
    last_focus_key: tuple[str, ...] | None = None

    while True:
        grouped = discover_combo_results(args.results)

        active_key = None
        if args.depth_first:
            active_key = select_depth_first_combo(grouped, args.manage)
            if active_key is None:
                time.sleep(args.interval)
                continue
            if active_key != last_focus_key:
                print(
                    f"[analyzer] depth-first: focusing "
                    f"epsilon={grouped[active_key]['job']['epsilon']}",
                )
                last_focus_key = active_key

        for combo_key, data in grouped.items():
            if args.depth_first and combo_key != active_key:
                continue

            rows = read_manage(args.manage)
            combo = {f: data["job"][f] for f in COMBO_KEY_FIELDS}
            idx = find_manage_row(rows, combo)
            if idx is not None and rows[idx].get("isAnalyzed", ""):
                pending_points.pop(combo_key, None)
                continue

            analyze_combo(
                combo_key, data, args.manage, args.results, args.samples, args.manifest,
                pending_points,
            )

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
