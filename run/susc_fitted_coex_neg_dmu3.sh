#!/usr/bin/env bash
# SUSC_RUNS_NEG_DMU3_FITTED_COEX: negative_drive, beta*delta_f = 0, k = 1, beta*delta_mu = 3,
# at mu from a cubic fit of COEX_RUNS_C3D/ly40 mu_coex(eps). No Newton correction.
#
#   bash run/susc_fitted_coex_neg_dmu3.sh fit | launch | status | analyze   (DRY_RUN=1 for launch preview)
#
# Grid: eps_c +- 0.1 with eps_c = -1.95772 (C3D Ly=40 BC = 5/9 crossings -1.960 ... -1.952),
# i.e. -2.05772 ... -1.85772 step 0.005 (41 eps, the old SUSC_RUNS_NEG_DMU3 grid; the two eps it
# skipped for bad coex fits are included, since the cubic fit smooths over them); L = 16 ... 64.
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
export WORK="${WORK:-SUSC_RUNS_NEG_DMU3_FITTED_COEX}"
export PHYS_CAMPAIGN=SUSC_RUNS_NEG_DMU3 COEX=COEX_RUNS_C3D/ly40/manage.csv
export FIT_MIN="${FIT_MIN:--2.13}" FIT_MAX="${FIT_MAX:--1.78}"
export EPS_MIN="${EPS_MIN:--2.05772}" EPS_MAX="${EPS_MAX:--1.85772}" EPS_STEP=0.005
export SIZES="${SIZES:-16 32 48 64}"
export DELTA_F=0.0 DELTA_MU=3.0 K=1.0 SCHEME=negative_drive
exec bash "$(dirname "$0")/susc_fitted_coex.sh" "$@"
