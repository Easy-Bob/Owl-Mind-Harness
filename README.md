# Owl-Mind Harness

A multi-agent study assistant built on [DeepAgents](https://github.com/langchain-ai/deepagents).

**Status:** early development. The project scaffold and guard-rail tests are in place, and no study features exist yet.

## Development

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run ruff check .
uv run pytest
```

The default test run blocks network access and costs nothing. Settings are listed in `.env.example`.

### Local services

Postgres and Redis run in Docker, on ports 5433 and 6380 to avoid clashing with other local projects:

```bash
cp .env.example .env
docker compose up -d --wait postgres redis
uv run alembic upgrade head
uv run pytest -m integration
```

The integration tests flush Redis and restart the containers, and they refuse to run against any other address. Don't point them at data you want to keep.

`docker compose down -v` deletes all local data. Use `docker compose down` to keep it.

## Planned stack

- Python 3.14, DeepAgents and FastAPI
- React and Vite
- PostgreSQL with pgvector, and Redis
