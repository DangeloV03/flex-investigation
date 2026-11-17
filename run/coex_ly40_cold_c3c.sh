#!/usr/bin/env bash
# C3C (negative_drive, beta*delta_f = 0, k = 1, beta*delta_mu = 2.0): Ly = 40 coex on the cold side,
# eps -2.088902 ... -1.908902 at step 0.02 (10 eps), appended to COEX_RUNS_C3C/ly40. The existing Ly = 40 grid
# stops at -1.888902, before BC reaches its bimodal plateau, so the BC_max-vs-beta*eps inflection
# was not bracketed. Step 0.02 = 4x the original 0.005 grid, aligned to it.
#
#   bash run/coex_ly40_cold_c3c.sh all       # add the jobs and start (tmux coex-C3C-ly40)
#   bash run/coex_ly40_cold_c3c.sh status    # progress
#   bash run/coex_ly40_cold_c3c.sh stop      # stop this campaign's tmux session
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
# See run/coex_ly40_extend.sh for details.
export EXP=C3C SCHEME=negative_drive FLEX_SCHEME=3 DELTA_MU=2.0
export EPS_MIN=-2.088902 EPS_MAX=-1.908902 EPS_STEP=0.02 MAX_CONCURRENT=${MAX_CONCURRENT:-20}
export WRAPPER="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
exec bash "$(dirname "$0")/coex_ly40_extend.sh" "$@"
