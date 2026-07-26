#!/usr/bin/env bash
# Thin wrapper around the interactive / one-click installer.
# Examples:
#   bash scripts/bootstrap.sh
#   bash scripts/bootstrap.sh -y
#   bash scripts/bootstrap.sh --dry-run -y
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python3 scripts/setup_quickstart.py "$@"
