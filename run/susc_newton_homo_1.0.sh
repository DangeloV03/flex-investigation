#!/usr/bin/env bash
# SUSC_RUNS_HOMO_1.0: Newton-mu susceptibility campaign for homo scheme 1,
# beta*delta_f = 0, k = 1, beta*delta_mu = 1 (the S2A physics).
#
#   bash run/susc_newton_homo_1.0.sh fit       # cubic fit of S2A's coex mu(eps) -> SUSC_RUNS_HOMO_1.0/fitted_mu.csv, fit.png
#   DRY_RUN=1 bash run/susc_newton_homo_1.0.sh launch   # preview the jobs
#   bash run/susc_newton_homo_1.0.sh launch    # one Newton chain per (eps, L)
#   bash run/susc_newton_homo_1.0.sh status | kick | stop | analyze
#
# Starting mu: degree-3 polynomial through COEX_RUNS_C2A/ly40 mu_coex(eps) (the coex runs
# S2A used), fitted over eps [-1.85, -1.55]; rms residual 1.7e-4, vs 5.6e-4 for the
# linear fit S2A_refit used. Grid = S2A's: eps -1.795 ... -1.600, step 0.005 (40 eps),
# L = 16 ... 128. Each (eps, L) then corrects mu by Newton steps (run/susc_newton.sh).
#
# Works from any clone or account: data lives in /scratch/gpfs/WJACOBS/vd7294/flex-investigation.
export CAMPAIGN=SUSC_RUNS_S2A COEX=COEX_RUNS_C2A/ly40/manage.csv
export DEGREE=3 FIT_MIN=-1.85 FIT_MAX=-1.55
export DELTA_F=0.0 DELTA_MU=1.0 K=1.0 SCHEME=homo
if [[ "${SMOKE:-0}" == 1 ]]; then export WORK="${WORK:-NEWTON_SMOKE_HOMO}"; else export WORK="${WORK:-SUSC_RUNS_HOMO_1.0}"; fi
export EPS_MIN="${EPS_MIN:--1.795}" EPS_MAX="${EPS_MAX:--1.600}" EPS_STEP="${EPS_STEP:-0.005}"
exec bash "$(dirname "$0")/susc_newton.sh" "$@"
