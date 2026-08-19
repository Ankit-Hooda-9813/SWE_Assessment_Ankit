#!/usr/bin/env bash
# Run this yourself (not via Claude) after the Container App exists — it reads
# your local .env and pushes the values in as the app's live env vars/secrets.
# Usage: ./infra/azure_set_secrets.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo ".env not found in repo root" >&2
  exit 1
fi
set -a
source .env
set +a

: "${DASHBOARD_USER:?set in .env}"
: "${DASHBOARD_PASSWORD:?set in .env — do not leave as change-me}"
: "${SESSION_SECRET:?set in .env — do not leave as change-me-too}"
: "${AZURE_OPENAI_APIKEY:?set in .env}"
: "${AZURE_OPENAI_ENDPOINT:?set in .env}"

az containerapp update \
  --name autoace-voice-trial \
  --resource-group autoace-trial-rg \
  --set-env-vars \
    "AZURE_OPENAI_APIKEY=${AZURE_OPENAI_APIKEY}" \
    "AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}" \
    "AZURE_OPENAI_DEPLOYMENT=${AZURE_OPENAI_DEPLOYMENT:-gpt-5-mini}" \
    "GROQ_API_KEY=${GROK_API_KEY}" \
    "DASHBOARD_USER=${DASHBOARD_USER}" \
    "DASHBOARD_PASSWORD=${DASHBOARD_PASSWORD}" \
    "SESSION_SECRET=${SESSION_SECRET}" \
    "PRIVACY_MODE=${PRIVACY_MODE:-hybrid}" \
    "TONE_PROVIDER_ORDER=${TONE_PROVIDER_ORDER:-azure_openai,local}" \
    "SER_ENSEMBLE_ENABLED=${SER_ENSEMBLE_ENABLED:-true}"

echo "Secrets set. New revision will roll out automatically."
