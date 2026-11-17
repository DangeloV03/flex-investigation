#!/usr/bin/env bash
# D3F (positive_drive, beta*delta_f = 0, k = 1, beta*delta_mu = -2): new campaign, Ly = 40 coex only,
# eps -2.2 ... -1.5 at step 0.02 (36 eps), in a new COEX_RUNS_D3F/ly40.
# Range: D3B (delta_mu = -1) has its BC inflection near -1.77 to -1.80, and more negative delta_mu moves
# it colder, so this brackets both BC plateaus with room on each side and gives enough fitted points
# for a cubic fit of mu_coex(eps).
#
#   bash run/coex_ly40_d3f.sh all       # make COEX_RUNS_D3F/ly40, add the jobs and start (tmux coex-D3F-ly40)
#   bash run/coex_ly40_d3f.sh status    # progress
#   bash run/coex_ly40_d3f.sh stop      # stop this campaign's tmux session
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
# See run/coex_ly40_extend.sh for details.
export EXP=D3F SCHEME=positive_drive FLEX_SCHEME=2 DELTA_MU=-2.0 NEW_BASE=1 NEW_CAMPAIGN=1
export EPS_MIN=-2.2 EPS_MAX=-1.5 EPS_STEP=0.02 MAX_CONCURRENT=${MAX_CONCURRENT:-30}
export WRAPPER="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
exec bash "$(dirname "$0")/coex_ly40_extend.sh" "$@"
