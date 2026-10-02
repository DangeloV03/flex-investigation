#!/usr/bin/env bash
# Pull SUSC_RUNS_* campaign trees from Della onto this machine.
# Intended to run on the wjacobs workstation via cron every 12 hours.
#
# First-time setup (on the workstation, not your laptop):
#   ./scripts/setup_workstation_della_key.sh
#   ssh-copy-id -i ~/.ssh/id_ed25519_della.pub vd7294@della.princeton.edu
#   ssh -i ~/.ssh/id_ed25519_della vd7294@della.princeton.edu 'echo ok'
#
# Cron (every 12 hours at 00:00 and 12:00 local):
#   0 0,12 * * * /home/dangelov/software/flex-investigation/scripts/sync_susc_from_della.sh >> /home/dangelov/software/flex-investigation/logs/rsync_della.log 2>&1

set -euo pipefail

DELLA_HOST="${DELLA_HOST:-vd7294@della.princeton.edu}"
DELLA_KEY="${DELLA_KEY:-$HOME/.ssh/id_ed25519_della}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/gpfs/WJACOBS/vd7294/flex-investigation}"
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/software/flex-investigation}"
LOG_DIR="${LOCAL_ROOT}/logs"

mkdir -p "$LOG_DIR" "$LOCAL_ROOT"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=30)
if [[ -f "$DELLA_KEY" ]]; then
    SSH_OPTS+=(-i "$DELLA_KEY")
fi

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) rsync Della → $(hostname -s) ==="
echo "remote: ${DELLA_HOST}:${REMOTE_ROOT}/SUSC_RUNS_*"
echo "local:  ${LOCAL_ROOT}/"

# Copy every SUSC_RUNS_* campaign (S1A, S1B, S2A, …). Skip timeseries PNGs
# (old S1A copies); new runs do not write them. --partial keeps interrupted
# transfers; -a preserves times; -z compresses over the WAN.
rsync -az --partial --human-readable --info=stats2 \
    -e "ssh ${SSH_OPTS[*]}" \
    --exclude 'm_timeseries_*.png' \
    --exclude '_scratch_*' \
    "${DELLA_HOST}:${REMOTE_ROOT}/SUSC_RUNS_*" \
    "${LOCAL_ROOT}/"

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) done ==="
du -sh "${LOCAL_ROOT}"/SUSC_RUNS_* 2>/dev/null || true
