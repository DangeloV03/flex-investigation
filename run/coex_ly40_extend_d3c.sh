#!/usr/bin/env bash
# D3C (positive_drive, beta*delta_f = 0, k = 1, beta*delta_mu = +2): Ly = 40 coex at
# eps -2.1 ... -1.6, step 0.01, appended to COEX_RUNS_D3C/ly40.
#
#   bash run/coex_ly40_extend_d3c.sh all       # add the jobs and start (tmux coex-D3C-ly40)
#   bash run/coex_ly40_extend_d3c.sh status    # progress
#   bash run/coex_ly40_extend_d3c.sh stop      # stop this campaign's tmux session
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
# See run/coex_ly40_extend.sh for details.
export EXP=D3C DELTA_MU=2.0
export WRAPPER="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
exec bash "$(dirname "$0")/coex_ly40_extend.sh" "$@"
