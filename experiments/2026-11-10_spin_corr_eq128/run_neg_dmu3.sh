#!/usr/bin/env bash
# 128x128 spin-spin correlations, negative_drive, beta*dmu = 3, k = 1, beta*df = 0, at mu from the
# cubic fit of COEX_RUNS_C3D/ly40 (SUSC_RUNS_NEG_DMU3_FITTED_COEX/fit.json; eps_c ~ -1.958).
# eps = -1.983 (ordered side) and -1.870 (disordered side). Same run settings as the eq runs.
#   bash experiments/2026-11-10_spin_corr_eq128/run_neg_dmu3.sh smoke | launch | status | analyze
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/WJACOBS/vd7294/flex-investigation}"
export DATA_ROOT
export OUT="${OUT:-$DATA_ROOT/SPIN_CORR_NEG_DMU3_128}"
export JOB_NAME=spincorr_neg3
export SCHEME=negative_drive DELTA_MU=3.0 DELTA_F=0.0 K=1.0
export MU_FIT_JSON="$DATA_ROOT/SUSC_RUNS_NEG_DMU3_FITTED_COEX/fit.json"
export EPS_LIST="${EPS_LIST:--1.983 -1.870}"
exec bash "$(dirname "$0")/submit.sh" "$@"
