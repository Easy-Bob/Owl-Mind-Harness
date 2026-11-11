#!/usr/bin/env bash
# The restore drill (M4-G): restore the newest backup into a fresh database, check that the
# schema matches the code, count what came back, and run the integration tests against it.
#
# Usage: ops/backup/restore_drill.sh [backup file]
# It never touches the live database: everything goes into owl_restore_drill, dropped first.
set -euo pipefail
cd "$(dirname "$0")/../.."

BACKUP_DIR=${BACKUP_DIR:-backups}
file=${1:-$(ls -1t "$BACKUP_DIR"/*.dump | head -1)}
DRILL=owl_restore_drill
echo "Restoring $file into $DRILL"

docker compose exec -T postgres dropdb -U owl --if-exists "$DRILL"
docker compose exec -T postgres createdb -U owl "$DRILL"
docker compose exec -T postgres pg_restore -U owl -d "$DRILL" --no-owner --exit-on-error \
  < "$file"

export DATABASE_URL=postgresql://owl:owl_dev@localhost:5433/$DRILL
uv run alembic upgrade head > /dev/null  # A backup from an older release is upgraded, as in a deploy.
uv run alembic check
docker compose exec -T postgres psql -U owl -d "$DRILL" -tAc "
  SELECT 'students ' || count(*) FROM app.students
  UNION ALL SELECT 'threads ' || count(*) FROM app.threads
  UNION ALL SELECT 'runs ' || count(*) FROM app.runs
  UNION ALL SELECT 'checkpoints ' || count(*) FROM lg_checkpoint.checkpoints
  UNION ALL SELECT 'store items ' || count(*) FROM lg_store.store
  UNION ALL SELECT 'model calls ' || count(*) FROM ops.model_calls
  UNION ALL SELECT 'note chunks ' || count(*) FROM materials.chunks"
# The persistence tests restart containers and flush Redis, so they're left out here.
uv run pytest -q -m integration --ignore=tests/test_persistence_integration.py -p no:cacheprovider
echo "Restore drill passed: $file"
