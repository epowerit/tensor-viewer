#!/usr/bin/env bash
# Starts the backend and the frontend in the background each time the
# codespace starts; their output goes to /tmp/tensorviewer-*.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
setsid bash -c 'cd backend && uv run uvicorn tensorviewer.app:app --host 127.0.0.1 --port 8000' \
  >/tmp/tensorviewer-backend.log 2>&1 < /dev/null &
setsid bash -c 'cd frontend && npm run dev' \
  >/tmp/tensorviewer-frontend.log 2>&1 < /dev/null &
echo "TensorViewer is starting: the frontend opens on port 5173."
