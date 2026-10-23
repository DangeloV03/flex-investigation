#!/usr/bin/env bash
# Coex refit + rerun for SUSC_RUNS_POS_DMU3, the same procedure as S2A.
#
#   bash run/coex_refit_pos_dmu3.sh fit       # linear fit of mu_coex(eps) from COEX_RUNS_D3D/ly40
#   bash run/coex_refit_pos_dmu3.sh select    # eps with any L |<m>| >= 0.05
#   DRY_RUN=1 bash run/coex_refit_pos_dmu3.sh launch   # preview the jobs
#   bash run/coex_refit_pos_dmu3.sh all       # fit + select + launch
#   bash run/coex_refit_pos_dmu3.sh status | compare | promote [--yes]
#
# POS_DMU3: positive_drive, beta*delta_f = 0, k = 1, beta*delta_mu = 3, eps -1.613011 ... -1.413011
# (step 0.005), L = 16-64 only (the campaign was run without 96/128). Every pair converged,
# so no cold eps are skipped. Work dir: SUSC_RUNS_POS_DMU3_refit/.
# Any setting can still be overridden, e.g. DEGREE=2 bash run/coex_refit_pos_dmu3.sh fit
# See run/coex_refit_rerun.sh for what each step does.

set -euo pipefail
export CAMPAIGN="${CAMPAIGN:-SUSC_RUNS_POS_DMU3}"
export COEX="${COEX:-COEX_RUNS_D3D/ly40/manage.csv}"
export SIZES="${SIZES:-16 32 48 64}"
export MIN_EPS="${MIN_EPS:-none}"
exec bash "$(dirname "$0")/coex_refit_rerun.sh" "$@"
