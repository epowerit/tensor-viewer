#!/usr/bin/env bash
# Installs both projects once, when the codespace is created.
set -euo pipefail
cd "$(dirname "$0")/.."
pipx install uv
(cd backend && uv sync --locked --python 3.11)
(cd frontend && npm ci)
