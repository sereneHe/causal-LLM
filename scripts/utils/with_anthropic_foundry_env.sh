#!/bin/sh
set -eu

# Run a command with ANTHROPIC_FOUNDRY_* env loaded.
#
# Priority:
#   1) Load .env if present (gitignored)
#   2) Override ANTHROPIC_FOUNDRY_API_KEY from macOS Keychain if present
#
# Examples:
#   ./scripts/utils/with_anthropic_foundry_env.sh python3 scripts/utils/check_anthropic_foundry.py
#

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$REPO_ROOT"

if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a
  . ./.env
  set +a
fi

# Trim accidental whitespace/newlines (common when copying from web UIs)
if [ -n "${ANTHROPIC_FOUNDRY_API_KEY:-}" ]; then
  ANTHROPIC_FOUNDRY_API_KEY=$(printf '%s' "$ANTHROPIC_FOUNDRY_API_KEY" | tr -d '\r\n')
  export ANTHROPIC_FOUNDRY_API_KEY
fi
if [ -n "${ANTHROPIC_FOUNDRY_ENDPOINT:-}" ]; then
  ANTHROPIC_FOUNDRY_ENDPOINT=$(printf '%s' "$ANTHROPIC_FOUNDRY_ENDPOINT" | tr -d '\r\n')
  export ANTHROPIC_FOUNDRY_ENDPOINT
fi

if command -v security >/dev/null 2>&1; then
  keychain_value=$(security find-generic-password -a "$USER" -s "anthropic_foundry_api_key" -w 2>/dev/null || true)
  if [ -n "${keychain_value:-}" ]; then
    export ANTHROPIC_FOUNDRY_API_KEY=$(printf '%s' "$keychain_value" | tr -d '\r\n')
  fi
fi

if [ -z "${ANTHROPIC_FOUNDRY_API_KEY:-}" ]; then
  echo "Missing ANTHROPIC_FOUNDRY_API_KEY." 1>&2
  echo "- Put it in Keychain (recommended):" 1>&2
  echo "  security add-generic-password -a \"$USER\" -s \"anthropic_foundry_api_key\" -w \"<YOUR_KEY>\" -U" 1>&2
  echo "- Or create .env from .env.example" 1>&2
  exit 2
fi

# Basic sanity checks to avoid accidental misconfiguration
case "$ANTHROPIC_FOUNDRY_API_KEY" in
  http://*|https://*)
    echo "ANTHROPIC_FOUNDRY_API_KEY looks like a URL, not a key. Fix your .env/Keychain." 1>&2
    exit 2
    ;;
esac

if [ -n "${ANTHROPIC_FOUNDRY_ENDPOINT:-}" ]; then
  case "$ANTHROPIC_FOUNDRY_ENDPOINT" in
    *ai.azure.com/foundryProject*|*foundryProject/overview*)
      echo "ANTHROPIC_FOUNDRY_ENDPOINT is an Azure portal UI URL (ai.azure.com), not an inference API endpoint." 1>&2
      echo "Use a deployed inference endpoint like .../chat/completions or .../v1/messages." 1>&2
      exit 2
      ;;
  esac
fi

exec "$@"
