#!/bin/sh
set -eu

# ──────────────────────────────────────────────────────────────────────
# restore_db.sh – Restore a PostgreSQL dump into the production database.
# Usage: ./scripts/restore_db.sh <backup_file>
# ──────────────────────────────────────────────────────────────────────

COMPOSE_FILE="docker-compose.production.yml"
CONTAINER="nexus-postgres"
DB_USER="jarvis"
DB_NAME="jarvis"

log() { printf '[restore] %s\n' "$*"; }

# ── argument check ───────────────────────────────────────────────────
if [ $# -lt 1 ]; then
  log "Usage: $0 <backup_file>"
  exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
  log "ERROR: Backup file not found: $BACKUP_FILE"
  exit 1
fi

# ── prerequisite checks ─────────────────────────────────────────────
if ! docker compose -f "$COMPOSE_FILE" ps --status running 2>/dev/null | grep -q "$CONTAINER"; then
  log "ERROR: Container '$CONTAINER' is not running."
  log "Start the stack first: docker compose -f $COMPOSE_FILE up -d"
  exit 1
fi

# ── restore ──────────────────────────────────────────────────────────
SIZE=$(wc -c < "$BACKUP_FILE" | tr -d ' ')
log "Restoring '$BACKUP_FILE' ($SIZE bytes) into database '$DB_NAME' …"
docker exec -i "$CONTAINER" psql -U "$DB_USER" "$DB_NAME" < "$BACKUP_FILE"

log "Restore complete."
