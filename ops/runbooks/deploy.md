# Deploy a release

1. Check CI is green on the commit, and `uv run python -m owl_mind.evaluation.check_baseline` passes.
2. Back up first: `ops/backup/backup.sh`.
3. Pull the release: `git fetch --tags && git checkout v<version>`, then `uv sync --frozen` and `npm ci && npm run build` in `web/`.
4. Migrate: `uv run alembic upgrade head`, then `uv run alembic check` (must say no new operations).
5. Restart the workers one at a time (`uv run python -m owl_mind.worker.runs`, each with its own `WORKER_METRICS_PORT`). A run a stopped worker had is taken over and resumed after 10 minutes, so stopping one is safe.
6. Restart the API (`uv run uvicorn owl_mind.api.app:app --port 8010`).
7. Watch the Service health dashboard for 15 minutes: failed share, worker heartbeats, queue depth.
8. Send one test message from a test account and check it streams to the end.

If anything in steps 4 to 8 fails, roll back ([rollback.md](rollback.md)).
