#!/usr/bin/env bash
# D3B (positive_drive, beta*delta_f = 0, k = 1, beta*delta_mu = -1.0): Ly = 40 coex on the cold side,
# eps -2.091185 ... -1.811185 at step 0.02 (15 eps), appended to COEX_RUNS_D3B/ly40. The existing Ly = 40 grid
# stops at -1.791185, before BC reaches its bimodal plateau, so the BC_max-vs-beta*eps inflection
# was not bracketed. Step 0.02 = 4x the original 0.005 grid, aligned to it.
#
#   bash run/coex_ly40_cold_d3b.sh all       # add the jobs and start (tmux coex-D3B-ly40)
#   bash run/coex_ly40_cold_d3b.sh status    # progress
#   bash run/coex_ly40_cold_d3b.sh stop      # stop this campaign's tmux session
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
# See run/coex_ly40_extend.sh for details.
export EXP=D3B SCHEME=positive_drive FLEX_SCHEME=2 DELTA_MU=-1.0
export EPS_MIN=-2.091185 EPS_MAX=-1.811185 EPS_STEP=0.02 MAX_CONCURRENT=${MAX_CONCURRENT:-20}
export WRAPPER="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
exec bash "$(dirname "$0")/coex_ly40_extend.sh" "$@"
