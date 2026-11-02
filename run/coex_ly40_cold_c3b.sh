#!/usr/bin/env bash
# C3B (negative_drive, beta*delta_f = 0, k = 1, beta*delta_mu = -1.0): Ly = 40 coex on the cold side,
# eps -2.0813 ... -1.7813 at step 0.02 (16 eps), appended to COEX_RUNS_C3B/ly40. The existing Ly = 40 grid
# stops at -1.7613, before BC reaches its bimodal plateau, so the BC_max-vs-beta*eps inflection
# was not bracketed. Step 0.02 = 4x the original 0.005 grid, aligned to it.
#
#   bash run/coex_ly40_cold_c3b.sh all       # add the jobs and start (tmux coex-C3B-ly40)
#   bash run/coex_ly40_cold_c3b.sh status    # progress
#   bash run/coex_ly40_cold_c3b.sh stop      # stop this campaign's tmux session
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
# See run/coex_ly40_extend.sh for details.
export EXP=C3B SCHEME=negative_drive FLEX_SCHEME=3 DELTA_MU=-1.0
export EPS_MIN=-2.0813 EPS_MAX=-1.7813 EPS_STEP=0.02 MAX_CONCURRENT=${MAX_CONCURRENT:-20}
export WRAPPER="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
exec bash "$(dirname "$0")/coex_ly40_extend.sh" "$@"
