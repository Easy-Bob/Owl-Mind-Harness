# Restore from a backup

Backups are made nightly by `ops/backup/backup.sh` (14 kept, in `BACKUP_DIR`). They hold student data: keep them on encrypted storage.

**Practice first.** `ops/backup/restore_drill.sh [file]` restores into a scratch database (`owl_restore_drill`), upgrades and checks the schema, counts rows, and runs the integration tests against it. Run it monthly, and before any real restore.

**A real restore:**
1. Stop the API and every worker, so nothing writes during the restore.
2. Keep the broken database for later: `docker compose exec -T postgres psql -U owl -d postgres -c "ALTER DATABASE owl RENAME TO owl_broken_$(date +%Y%m%d)"`.
3. `docker compose exec -T postgres createdb -U owl owl`, then `docker compose exec -T postgres pg_restore -U owl -d owl --no-owner --exit-on-error < <backup file>`.
4. `uv run alembic upgrade head && uv run alembic check`.
5. Start the workers and the API. Runs that were in flight at the backup's time are failed or resumed by the workers as usual.
6. Anything after the backup is lost: chats, plans and memory written since. Tell affected students.
