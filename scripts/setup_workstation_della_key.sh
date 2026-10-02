#!/usr/bin/env bash
# Create a dedicated ed25519 key on THIS machine for passwordless SSH to Della.
# Run on the wjacobs workstation (the host that will rsync), not on your laptop.
#
# Does not overwrite ~/.ssh/id_ed25519 if that already exists.

set -euo pipefail

KEY="${HOME}/.ssh/id_ed25519_della"
NETID="${NETID:-vd7294}"
CLUSTER="${CLUSTER:-della.princeton.edu}"

mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"

if [[ -f "${KEY}" ]]; then
    echo "Key already exists: ${KEY}"
else
    ssh-keygen -t ed25519 -f "${KEY}" -N "" -C "${USER}@$(hostname -s)-della"
    echo "Created ${KEY} and ${KEY}.pub"
fi

chmod 600 "${KEY}"
chmod 644 "${KEY}.pub"

echo
echo "Public key:"
cat "${KEY}.pub"
echo
echo "Copy it to Della (you will type your NetID password and Duo once):"
echo "  ssh-copy-id -i ${KEY}.pub ${NETID}@${CLUSTER}"
echo
echo "Then test (should print hostname with no password):"
echo "  ssh -i ${KEY} ${NETID}@${CLUSTER} 'hostname'"
echo
echo "Install the 12-hour cron job:"
echo "  mkdir -p ${HOME}/software/flex-investigation/logs"
echo "  crontab -l 2>/dev/null | grep -v sync_susc_from_della.sh > /tmp/cron.\$\$ || true"
echo "  echo '0 0,12 * * * ${HOME}/software/flex-investigation/scripts/sync_susc_from_della.sh >> ${HOME}/software/flex-investigation/logs/rsync_della.log 2>&1' >> /tmp/cron.\$\$"
echo "  crontab /tmp/cron.\$\$; rm -f /tmp/cron.\$\$"
