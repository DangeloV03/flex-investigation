#!/usr/bin/env bash
# SUSC_RUNS_NEG_DMU-1_FITTED_COEX: negative_drive, beta*delta_f = 0, k = 1, beta*delta_mu = -1,
# at mu from a cubic fit of COEX_RUNS_C3B/ly40 mu_coex(eps). No Newton correction.
#
#   bash run/susc_fitted_coex_neg_dmu_m1.sh fit | launch | status | analyze   (DRY_RUN=1 for launch preview)
#
# Fit: C3B coex points with eps in [-1.95, -1.55]. No earlier susc campaign at this physics, so the
# physics comes from the settings below (no PHYS_CAMPAIGN).
# Grid: eps_c +- 0.1 with eps_c = -1.7232 (C3B Ly=40 BC = 5/9 crossing),
# i.e. -1.8232 ... -1.6232 step 0.005 (41 eps); L = 16 ... 64.
# (File is named _m1, not -1, so it doesn't clash with a local untracked neg_dmu-1.sh.)
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
export WORK="${WORK:-SUSC_RUNS_NEG_DMU-1_FITTED_COEX}"
export COEX=COEX_RUNS_C3B/ly40/manage.csv
export FIT_MIN="${FIT_MIN:--1.95}" FIT_MAX="${FIT_MAX:--1.55}"
export EPS_MIN="${EPS_MIN:--1.8232}" EPS_MAX="${EPS_MAX:--1.6232}" EPS_STEP=0.005
export SIZES="${SIZES:-16 32 48 64}"
export DELTA_F=0.0 DELTA_MU=-1.0 K=1.0 SCHEME=negative_drive
exec bash "$(dirname "$0")/susc_fitted_coex.sh" "$@"
