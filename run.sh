#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "==> creating .venv"
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

[[ -f .env ]] && set -a && source .env && set +a

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "==> http://$HOST:$PORT"
exec ./.venv/bin/python -m uvicorn server.main:app --host "$HOST" --port "$PORT"
