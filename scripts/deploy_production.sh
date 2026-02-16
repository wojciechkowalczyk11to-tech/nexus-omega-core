#!/bin/sh
set -eu

# ──────────────────────────────────────────────────────────────────────
# deploy_production.sh – Build, start, and verify the production stack.
# Usage: ./scripts/deploy_production.sh
# ──────────────────────────────────────────────────────────────────────

COMPOSE_FILE="docker-compose.production.yml"
HEALTH_URL="http://localhost:8000/api/v1/health"
HEALTH_TIMEOUT=120
HEALTH_INTERVAL=3

log() { printf '[deploy] %s\n' "$*"; }

# ── prerequisite checks ─────────────────────────────────────────────
log "Checking prerequisites …"

if ! command -v docker >/dev/null 2>&1; then
  log "ERROR: docker is not installed or not in PATH."
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  log "ERROR: 'docker compose' plugin is not available."
  exit 1
fi

if [ ! -f ".env" ]; then
  log "ERROR: .env file not found. Copy .env.example and configure it:"
  log "  cp .env.example .env"
  exit 1
fi

if [ ! -f "$COMPOSE_FILE" ]; then
  log "ERROR: $COMPOSE_FILE not found. Run this script from the repository root."
  exit 1
fi

log "Prerequisites OK."

# ── pull / build ─────────────────────────────────────────────────────
log "Pulling base images and building services …"
docker compose -f "$COMPOSE_FILE" build

# ── start stack ──────────────────────────────────────────────────────
log "Starting production stack …"
docker compose -f "$COMPOSE_FILE" up -d

# ── wait for health ──────────────────────────────────────────────────
log "Waiting for backend health (timeout ${HEALTH_TIMEOUT}s) …"
elapsed=0
while [ "$elapsed" -lt "$HEALTH_TIMEOUT" ]; do
  if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
    log "Backend is healthy."
    break
  fi
  sleep "$HEALTH_INTERVAL"
  elapsed=$((elapsed + HEALTH_INTERVAL))
done

if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
  log "ERROR: Backend did not become healthy within ${HEALTH_TIMEOUT}s."
  log "Recent backend logs:"
  docker compose -f "$COMPOSE_FILE" logs --tail=30 backend || true
  exit 1
fi

# ── status ───────────────────────────────────────────────────────────
log "Container status:"
docker compose -f "$COMPOSE_FILE" ps

log "Health JSON:"
curl -s "$HEALTH_URL" | python3 -m json.tool 2>/dev/null || curl -s "$HEALTH_URL"

log "Production stack is running."
