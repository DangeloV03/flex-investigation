#!/usr/bin/env bash
# NEG_DMU_2.5 (negative_drive, beta*delta_f = 0, k = 1, beta*delta_mu = 2.5): new campaign, Ly = 40 coex only,
# eps -2.15 ... -1.29 at step 0.02, in a new COEX_NEG_DMU_2.5/ly40. Coex + later fitted-coex susceptibility.
# Range: union of C3C (dmu=2) and C3D (dmu=3) Ly=40 ranges; warm end = FLEX cutoff (mu_coex_FLEX > 0 above -1.29).
# Step 0.02 (coarser than the old 0.005) because mu_coex(eps) is used through a cubic fit.
#
#   bash run/coex_ly40_neg_dmu_2p5.sh all       # make COEX_NEG_DMU_2.5/ly40, add the jobs and start (tmux coex-NEG_DMU_2p5-ly40)
#   bash run/coex_ly40_neg_dmu_2p5.sh status    # progress
#   bash run/coex_ly40_neg_dmu_2p5.sh stop      # stop this campaign's tmux session
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
# See run/coex_ly40_extend.sh and run/runner.md for details.
export EXP=NEG_DMU_2.5 CAMPAIGN_DIR=COEX_NEG_DMU_2.5
export SCHEME=negative_drive FLEX_SCHEME=3 DELTA_MU=2.5 NEW_BASE=1 NEW_CAMPAIGN=1
export EPS_MIN=-2.15 EPS_MAX=-1.29 EPS_STEP=0.02 MAX_CONCURRENT=${MAX_CONCURRENT:-30}
export WRAPPER="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
exec bash "$(dirname "$0")/coex_ly40_extend.sh" "$@"
