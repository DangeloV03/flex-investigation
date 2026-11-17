#!/usr/bin/env bash
# SUSC_RUNS_NEG_DMU-2_FITTED_COEX: negative_drive, beta*delta_f = 0, k = 1, beta*delta_mu = -2,
# at mu from a cubic fit of COEX_RUNS_C3F/ly40 mu_coex(eps). No Newton correction.
#
#   bash run/susc_fitted_coex_neg_dmu_m2.sh fit | launch | status | analyze   (DRY_RUN=1 for launch preview)
#
# Fit: C3F coex points (step 0.02) with eps in [-1.95, -1.60] (C3F's warm end is -1.60).
# No earlier susc campaign at this physics, so the physics comes from the settings below.
# Grid: C3F has no BC/criticality analysis. Its mu_coex sits a near-constant -0.107 below C3B's
# (dmu = -1) at every eps, so eps_c is taken as C3B's -1.7232: -1.8232 ... -1.6232 step 0.005
# (41 eps); L = 16 ... 64. Shift the grid (EPS_MIN/EPS_MAX) if the chi peak lands near an edge.
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
export WORK="${WORK:-SUSC_RUNS_NEG_DMU-2_FITTED_COEX}"
export COEX=COEX_RUNS_C3F/ly40/manage.csv
export FIT_MIN="${FIT_MIN:--1.95}" FIT_MAX="${FIT_MAX:--1.60}"
export EPS_MIN="${EPS_MIN:--1.8232}" EPS_MAX="${EPS_MAX:--1.6232}" EPS_STEP=0.005
export SIZES="${SIZES:-16 32 48 64}"
export DELTA_F=0.0 DELTA_MU=-2.0 K=1.0 SCHEME=negative_drive
exec bash "$(dirname "$0")/susc_fitted_coex.sh" "$@"
