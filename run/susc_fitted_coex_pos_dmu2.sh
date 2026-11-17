#!/usr/bin/env bash
# SUSC_RUNS_POS_DMU2_FITTED_COEX: positive_drive, beta*delta_f = 0, k = 1, beta*delta_mu = 2.0,
# at mu from a cubic fit of the D3C Ly=40 coex mu_coex(eps). No Newton correction.
#
#   bash run/susc_fitted_coex_pos_dmu2.sh fit | launch | status | analyze   (DRY_RUN=1 for launch preview)
#
# Grid: eps -1.8 ... -1.6 step 0.005; L = 16 ... 64. eps_c ~ -1.68 (POS_DMU2_shift U4 crossings -1.685/-1.678).
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
export WORK="${WORK:-SUSC_RUNS_POS_DMU2_FITTED_COEX}"
export PHYS_CAMPAIGN=SUSC_RUNS_POS_DMU2 COEX=COEX_RUNS_D3C/ly40/manage.csv
export FIT_MIN="${FIT_MIN:--1.90}" FIT_MAX="${FIT_MAX:--1.55}"
export EPS_MIN="${EPS_MIN:--1.8}" EPS_MAX="${EPS_MAX:--1.6}" EPS_STEP=0.005
export SIZES="${SIZES:-16 32 48 64}"
export DELTA_F=0.0 DELTA_MU=2.0 K=1.0 SCHEME=positive_drive
exec bash "$(dirname "$0")/susc_fitted_coex.sh" "$@"
