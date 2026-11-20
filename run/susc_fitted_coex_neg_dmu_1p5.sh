#!/usr/bin/env bash
# SUSC_RUNS_NEG_DMU_1.5_FITTED_COEX: negative_drive, beta*delta_f = 0, k = 1, beta*delta_mu = 1.5,
# at mu from a cubic fit of the COEX_NEG_DMU_1.5 Ly=40 coex mu_coex(eps). No Newton correction.
#
#   bash run/susc_fitted_coex_neg_dmu_1p5.sh fit | launch | status | analyze   (DRY_RUN=1 for launch preview)
#
# Grid: eps_c +- 0.1 with eps_c = -1.743 (COEX_NEG_DMU_1.5 Ly=40 BC = 5/9 crossing,
# plots/bc_all_coex/NEG_DMU_1.5), i.e. -1.843 ... -1.643 step 0.005 (41 eps); L = 16 ... 64.
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
export WORK="${WORK:-SUSC_RUNS_NEG_DMU_1.5_FITTED_COEX}"
export COEX=COEX_NEG_DMU_1.5/ly40/manage.csv
export FIT_MIN="${FIT_MIN:--1.99}" FIT_MAX="${FIT_MAX:--1.55}"
export EPS_MIN="${EPS_MIN:--1.843}" EPS_MAX="${EPS_MAX:--1.643}" EPS_STEP=0.005
export SIZES="${SIZES:-16 32 48 64}"
export DELTA_F=0.0 DELTA_MU=1.5 K=1.0 SCHEME=negative_drive
exec bash "$(dirname "$0")/susc_fitted_coex.sh" "$@"
