# coex/archive

One-off / diagnostic scripts retained for reference but no longer part of the
active pipeline. None are imported by any module, and none are referenced by the
docs. They were written against earlier campaign state, so some reference the old
`mu_coex_SIM` schema, stale tmux session names, or hardcoded CSVs — treat as
historical. If you resurrect one, move it back up to `coex/` (a couple compute
their repo root via `parents[1]`, which assumes they live directly under `coex/`).

| Script | What it did |
|--------|-------------|
| `audit_campaign.py` | Compare manage.csv, results/, and combo folder artifacts. |
| `clean_wrong_npy.py` | Delete final_lattice_*.npy for wrongly analyzed combos (old `mu_coex_SIM=NaN` rows). |
| `diagnose_susceptibility_analyzer.py` | Print why coex combos are / aren't ready for the analyzer. |
| `fitted_vs_sim.py` | One-off plot of fitted vs sim μ from a hardcoded `mu_coex_comparison.csv`. |
| `replot_from_results.py` | Regenerate φ/ψ plots + CSV from existing results/. |
| `reset_susceptibility_coex_analysis.py` | Clear coex analysis fields in a manage CSV so the analyzer re-runs. |

Active recovery/maintenance tools stayed in `coex/`:
`repair_queue.py`, `requeue_incomplete.py`, `retry_nan_combos.py`, `estimate_runtime.py`.
