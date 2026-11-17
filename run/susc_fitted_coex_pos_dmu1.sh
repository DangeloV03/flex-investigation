#!/usr/bin/env bash
# SUSC_RUNS_POS_DMU1_FITTED_COEX: positive_drive, beta*delta_f = 0, k = 1, beta*delta_mu = 1.0,
# at mu from a cubic fit of the D3A Ly=40 coex mu_coex(eps). No Newton correction.
#
#   bash run/susc_fitted_coex_pos_dmu1.sh fit | launch | status | analyze   (DRY_RUN=1 for launch preview)
#
# Grid: eps -1.8 ... -1.6 step 0.005; L = 16 ... 64. eps_c ~ -1.71 (D3A Ly40 BC 5/9 crossing -1.664 + homo slab offset).
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
export WORK="${WORK:-SUSC_RUNS_POS_DMU1_FITTED_COEX}"
export PHYS_CAMPAIGN=SUSC_RUNS_POS_DMU1 COEX=COEX_RUNS_D3A/ly40/manage.csv
export FIT_MIN="${FIT_MIN:--1.90}" FIT_MAX="${FIT_MAX:--1.55}"
export EPS_MIN="${EPS_MIN:--1.8}" EPS_MAX="${EPS_MAX:--1.6}" EPS_STEP=0.005
export SIZES="${SIZES:-16 32 48 64}"
export DELTA_F=0.0 DELTA_MU=1.0 K=1.0 SCHEME=positive_drive
exec bash "$(dirname "$0")/susc_fitted_coex.sh" "$@"
