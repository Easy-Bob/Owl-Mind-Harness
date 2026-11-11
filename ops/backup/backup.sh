#!/usr/bin/env bash
# Nightly backup (M4-G): a pg_dump of the whole database, keeping the last 14.
#
# Usage: ops/backup/backup.sh            (BACKUP_DIR defaults to ./backups, which git ignores)
# Schedule it nightly, for example with cron:  15 3 * * * cd /path/to/repo && ops/backup/backup.sh
# Backups hold student data: keep BACKUP_DIR on encrypted storage, never in git.
set -euo pipefail
cd "$(dirname "$0")/../.."

BACKUP_DIR=${BACKUP_DIR:-backups}
KEEP=${KEEP:-14}
DATABASE=${DATABASE:-owl}
mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"

file="$BACKUP_DIR/$DATABASE-$(date -u +%Y%m%dT%H%M%SZ).dump"
# Custom format: compressed, and pg_restore can restore one table if needed.
docker compose exec -T postgres pg_dump -U owl -d "$DATABASE" --format=custom > "$file.partial"
mv "$file.partial" "$file"
chmod 600 "$file"
echo "Backed up to $file ($(du -h "$file" | cut -f1))"

# Keep the newest $KEEP.
ls -1t "$BACKUP_DIR"/"$DATABASE"-*.dump | tail -n +"$((KEEP + 1))" | xargs -r rm --
