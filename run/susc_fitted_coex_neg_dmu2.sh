#!/usr/bin/env bash
# SUSC_RUNS_NEG_DMU2_FITTED_COEX: negative_drive, beta*delta_f = 0, k = 1, beta*delta_mu = 2,
# at mu from a cubic fit of COEX_RUNS_C3C/ly40 mu_coex(eps). No Newton correction.
#
#   bash run/susc_fitted_coex_neg_dmu2.sh fit | launch | status | analyze   (DRY_RUN=1 for launch preview)
#
# Fit: coex points with eps in [-1.95, -1.60]; 57/60 used, rms residual 1.8e-4. Fits over
# neighbouring windows agree to 3e-5 on the run grid.
# Grid: eps_c +- 0.1 with eps_c = -1.778902 (C3C Ly=40 BC crossing), i.e.
# -1.878902 ... -1.678902 step 0.005 (41 eps, the old SUSC_RUNS_NEG_DMU2 grid); L = 16 ... 64.
# (L = 96/128 can be added later with SIZES="96 128" FORCE=1 ... launch.)
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
export WORK="${WORK:-SUSC_RUNS_NEG_DMU2_FITTED_COEX}"
export PHYS_CAMPAIGN=SUSC_RUNS_NEG_DMU2 COEX=COEX_RUNS_C3C/ly40/manage.csv
export FIT_MIN=-1.95 FIT_MAX=-1.60
export EPS_MIN="${EPS_MIN:--1.878902}" EPS_MAX="${EPS_MAX:--1.678902}" EPS_STEP=0.005
export SIZES="${SIZES:-16 32 48 64}"
export DELTA_F=0.0 DELTA_MU=2.0 K=1.0 SCHEME=negative_drive
exec bash "$(dirname "$0")/susc_fitted_coex.sh" "$@"
