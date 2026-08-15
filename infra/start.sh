#!/usr/bin/env bash
# Launch a single on-demand EC2 Graviton box, build/pull this repo's Docker
# image, and wait for the health check to pass before handing back control.
#
# Deliberately NOT a Terraform/CloudFormation stack: this is a start/stop
# pattern for a single box run on demand (batch scoring, a demo session),
# not standing infrastructure. Read before running against a real account —
# it launches a billed instance and does not ask for confirmation.
set -euo pipefail

INSTANCE_TYPE="${INSTANCE_TYPE:-t4g.large}"
AMI_ID="${AMI_ID:?Set AMI_ID to an Amazon Linux 2023 ARM64 AMI in your target region}"
KEY_NAME="${KEY_NAME:?Set KEY_NAME to an existing EC2 key pair}"
SECURITY_GROUP="${SECURITY_GROUP:?Set SECURITY_GROUP to a security group allowing inbound 7860}"
SUBNET_ID="${SUBNET_ID:?Set SUBNET_ID to a subnet with a route to the internet}"
STATE_FILE="${STATE_FILE:-infra/.instance_id}"

if [[ -f "$STATE_FILE" ]]; then
  echo "An instance is already tracked in $STATE_FILE — run stop.sh first, or remove the file if it's stale." >&2
  exit 1
fi

USER_DATA=$(cat <<'CLOUDINIT'
#!/bin/bash
set -euo pipefail
dnf install -y docker git
systemctl enable --now docker
git clone --depth 1 https://github.com/REPLACE_WITH_YOUR_REMOTE/SWE_Assessment_Ankit_Hooda.git /opt/app
cd /opt/app
docker build -t autoace-v2 .
docker run -d --name autoace -p 7860:7860 --restart unless-stopped autoace-v2
CLOUDINIT
)

echo "Launching $INSTANCE_TYPE..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SECURITY_GROUP" \
  --subnet-id "$SUBNET_ID" \
  --user-data "$USER_DATA" \
  --instance-market-options '{"MarketType":"spot"}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=autoace-v2-batch}]' \
  --query 'Instances[0].InstanceId' --output text)

echo "$INSTANCE_ID" > "$STATE_FILE"
echo "Instance $INSTANCE_ID launching. Waiting for it to reach 'running'..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "Instance running at $PUBLIC_IP. Container build + model load takes 1-3 minutes."
echo "Poll: curl http://$PUBLIC_IP:7860/health"
echo "When done: ./infra/stop.sh"
