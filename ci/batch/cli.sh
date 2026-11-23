#!/usr/bin/env bash
# BENIGN SECURITY CANARY (AWS VDP test).
# Prints a unique marker and non-sensitive execution context only.
set -e
MARKER="AWS-LABS-CANARY-7f3c9a"
echo "[canary] ${MARKER} start $(date -u +%FT%TZ)"
echo "[canary] args=$* host=$(hostname) user=$(id -un) cwd=$(pwd)"
echo "[canary] repo=${REPO:-unset} source_ref=${SOURCE_REF:-unset}"
if [ -n "${GITHUB_TOKEN:-}" ]; then echo "[canary] GITHUB_TOKEN=present"; else echo "[canary] GITHUB_TOKEN=absent"; fi
if command -v aws >/dev/null 2>&1; then
  echo "[canary] aws sts get-caller-identity:"
  aws sts get-caller-identity || echo "[canary] sts call failed"
else
  echo "[canary] aws cli not present"
fi
echo "[canary] ${MARKER} end"
exit 0
