#!/usr/bin/env bash
# D3D (positive_drive, beta*delta_f = 0, k = 1, beta*delta_mu = +3): Ly = 40 coex at
# eps -2.1 ... -1.6, step 0.01, appended to COEX_RUNS_D3D/ly40.
#
#   bash run/coex_ly40_extend_d3d.sh all       # add the jobs and start (tmux coex-D3D-ly40)
#   bash run/coex_ly40_extend_d3d.sh status    # progress
#   bash run/coex_ly40_extend_d3d.sh stop      # stop this campaign's tmux session
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
# See run/coex_ly40_extend.sh for details.
export EXP=D3D DELTA_MU=3.0
export WRAPPER="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
exec bash "$(dirname "$0")/coex_ly40_extend.sh" "$@"
