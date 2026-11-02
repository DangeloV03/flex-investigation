#!/usr/bin/env bash
# C2B (homo, beta*delta_f = 0, k = 1, beta*delta_mu = -1): first Ly = 40 coex for this campaign,
# eps -2.1005 ... -1.3805 at step 0.02 (37 eps, 4x the Ly = 16 grid's 0.005 and aligned to it),
# in a new COEX_RUNS_C2B/ly40 next to ly16. Ly = 16 stops at -1.7805, before BC reaches its bimodal
# plateau, so this also covers the cold side the BC-vs-beta*eps inflection needs.
#
#   bash run/coex_ly40_c2b.sh all       # make ly40, add the jobs and start (tmux coex-C2B-ly40)
#   bash run/coex_ly40_c2b.sh status    # progress
#   bash run/coex_ly40_c2b.sh stop      # stop this campaign's tmux session
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
# See run/coex_ly40_extend.sh for details.
export EXP=C2B SCHEME=homo FLEX_SCHEME=1 DELTA_MU=-1.0 NEW_BASE=1
export EPS_MIN=-2.1005 EPS_MAX=-1.3805 EPS_STEP=0.02 MAX_CONCURRENT=${MAX_CONCURRENT:-30}
export WRAPPER="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
exec bash "$(dirname "$0")/coex_ly40_extend.sh" "$@"
