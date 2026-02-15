#!/bin/sh
set -eu

echo "[entrypoint] Starting backend container..."

if [ "${RUN_MIGRATIONS:-1}" != "0" ]; then
  echo "[entrypoint] Running database migrations..."
  alembic upgrade head
else
  echo "[entrypoint] Skipping migrations (RUN_MIGRATIONS=0)."
fi

echo "[entrypoint] Launching uvicorn on 0.0.0.0:${PORT:-8000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
