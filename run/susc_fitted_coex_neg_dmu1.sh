#!/usr/bin/env bash
# SUSC_RUNS_NEG_DMU1_FITTED_COEX: negative_drive, beta*delta_f = 0, k = 1, beta*delta_mu = 1,
# at mu from a cubic fit of the C3A Ly=40 coex mu_coex(eps). No Newton correction.
#
#   bash run/susc_fitted_coex_neg_dmu1.sh fit | launch | status | analyze   (DRY_RUN=1 for launch preview)
#
# Coex: COEX_RUNS_C3A/ly40/manage_recovered.csv = manage.csv with the 121 warm fits
# (eps -1.748718 ... -1.148718) that a generator bug wiped, put back from the analyzer log
# (logs/c3_auto_20260923T130922Z.log). They match the mu the old SUSC_RUNS_NEG_DMU1 ran at to 5e-7.
# Grid: eps_c +- 0.1 with eps_c = -1.723718 (C3A Ly=40 BC = 5/9 crossings -1.727 ... -1.717),
# i.e. -1.823718 ... -1.623718 step 0.005 (41 eps); L = 16 ... 64.
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
export WORK="${WORK:-SUSC_RUNS_NEG_DMU1_FITTED_COEX}"
export PHYS_CAMPAIGN=SUSC_RUNS_NEG_DMU1 COEX=COEX_RUNS_C3A/ly40/manage_recovered.csv
export FIT_MIN="${FIT_MIN:--1.89}" FIT_MAX="${FIT_MAX:--1.55}"
export EPS_MIN="${EPS_MIN:--1.823718}" EPS_MAX="${EPS_MAX:--1.623718}" EPS_STEP=0.005
export SIZES="${SIZES:-16 32 48 64}"
export DELTA_F=0.0 DELTA_MU=1.0 K=1.0 SCHEME=negative_drive
exec bash "$(dirname "$0")/susc_fitted_coex.sh" "$@"
