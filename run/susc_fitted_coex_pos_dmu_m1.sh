#!/usr/bin/env bash
# SUSC_RUNS_POS_DMU-1_FITTED_COEX: positive_drive, beta*delta_f = 0, k = 1, beta*delta_mu = -1.0,
# at mu from a cubic fit of the D3B Ly=40 coex mu_coex(eps). No Newton correction.
#
#   bash run/susc_fitted_coex_pos_dmu_m1.sh fit | launch | status | analyze   (DRY_RUN=1 for launch preview)
#
# Grid: eps -1.95 ... -1.6 step 0.005; L = 16 ... 64. Cold end extended: eps_c ~ -1.83 (D3B Ly40 crossing -1.786 + offset); D3B data to -2.09.
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
export WORK="${WORK:-SUSC_RUNS_POS_DMU-1_FITTED_COEX}"
export COEX=COEX_RUNS_D3B/ly40/manage.csv
export FIT_MIN="${FIT_MIN:--2.05}" FIT_MAX="${FIT_MAX:--1.55}"
export EPS_MIN="${EPS_MIN:--1.95}" EPS_MAX="${EPS_MAX:--1.6}" EPS_STEP=0.005
export SIZES="${SIZES:-16 32 48 64}"
export DELTA_F=0.0 DELTA_MU=-1.0 K=1.0 SCHEME=positive_drive
exec bash "$(dirname "$0")/susc_fitted_coex.sh" "$@"
