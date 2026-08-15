#!/usr/bin/env bash
# Terminate the instance launched by start.sh. This is a destructive,
# billing-stopping action against a real AWS account — it does not ask for
# confirmation, matching start.sh's posture that both scripts are meant to
# be read before they are run.
set -euo pipefail

STATE_FILE="${STATE_FILE:-infra/.instance_id}"

if [[ ! -f "$STATE_FILE" ]]; then
  echo "No instance tracked in $STATE_FILE — nothing to stop." >&2
  exit 1
fi

INSTANCE_ID=$(cat "$STATE_FILE")
echo "Terminating $INSTANCE_ID..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null
aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"
rm -f "$STATE_FILE"
echo "Terminated. Billing stops now."
