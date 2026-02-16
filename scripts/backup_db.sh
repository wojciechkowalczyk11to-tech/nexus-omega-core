#!/bin/sh
set -eu

# ──────────────────────────────────────────────────────────────────────
# backup_db.sh – Dump the production PostgreSQL database.
# Usage: ./scripts/backup_db.sh [output_file]
# Default output: backups/nexus_backup_<timestamp>.sql
# ──────────────────────────────────────────────────────────────────────

COMPOSE_FILE="docker-compose.production.yml"
CONTAINER="nexus-postgres"
DB_USER="jarvis"
DB_NAME="jarvis"
BACKUP_DIR="backups"

log() { printf '[backup] %s\n' "$*"; }

# ── prerequisite checks ─────────────────────────────────────────────
if ! docker compose -f "$COMPOSE_FILE" ps --status running 2>/dev/null | grep -q "$CONTAINER"; then
  log "ERROR: Container '$CONTAINER' is not running."
  log "Start the stack first: docker compose -f $COMPOSE_FILE up -d"
  exit 1
fi

# ── prepare output path ─────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="${1:-${BACKUP_DIR}/nexus_backup_${TIMESTAMP}.sql}"
OUTPUT_DIR=$(dirname "$OUTPUT")

if [ ! -d "$OUTPUT_DIR" ]; then
  log "Creating backup directory: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

# ── dump ─────────────────────────────────────────────────────────────
log "Dumping database '$DB_NAME' from container '$CONTAINER' …"
docker exec -t "$CONTAINER" pg_dump -U "$DB_USER" "$DB_NAME" > "$OUTPUT"

SIZE=$(wc -c < "$OUTPUT" | tr -d ' ')
log "Backup complete: $OUTPUT ($SIZE bytes)"
