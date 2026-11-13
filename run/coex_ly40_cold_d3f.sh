#!/usr/bin/env bash
# D3F (positive_drive, beta*delta_f = 0, k = 1, beta*delta_mu = -2): Ly = 40 coex on the cold side,
# eps -2.02 ... -1.90 at step 0.02 (7 eps), appended to COEX_RUNS_D3F/ly40. The existing fitted mu
# stops at -1.88, but SUSC_RUNS_POS_DMU-2_FITTED_COEX needs eps down to -2.0 (eps_c ~ -1.90).
# Tested on D3A-D3D: a cubic fit cut at -1.88 is off by 1e-3 - 3e-3 at -2.0, so the cold end needs data.
#
#   bash run/coex_ly40_cold_d3f.sh all       # add the jobs and start (tmux coex-D3F-ly40)
#   bash run/coex_ly40_cold_d3f.sh status    # progress
#   bash run/coex_ly40_cold_d3f.sh stop      # stop this campaign's tmux session
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
# See run/coex_ly40_extend.sh for details.
export EXP=D3F SCHEME=positive_drive FLEX_SCHEME=2 DELTA_MU=-2.0
export EPS_MIN=-2.02 EPS_MAX=-1.9 EPS_STEP=0.02 MAX_CONCURRENT=${MAX_CONCURRENT:-30}
export WRAPPER="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
exec bash "$(dirname "$0")/coex_ly40_extend.sh" "$@"
