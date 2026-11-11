# Roll back a release

1. Check out the previous tag: `git checkout v<previous>`, then `uv sync --frozen` and rebuild `web/`.
2. If the release added a migration, downgrade it first, with the new code still checked out: `uv run alembic downgrade -1` (once per migration added). Every migration in this repo has a `downgrade`. Migrations that drop data say so; for those, restore instead ([restore.md](restore.md)).
3. Restart the workers, then the API, as in [deploy.md](deploy.md).
4. If a kept M4-D change caused it, its setting turns it off without a rollback: `FORCE_DELEGATION`, `DIRECT_SINGLE_TUTOR`, `QUIZ_REUSE`, `COMPOSITE_ANSWER`, `LARGE_MODEL`.
