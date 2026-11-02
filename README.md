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

Postgres, Redis, Prometheus and Grafana run in Docker, on ports 5433, 6380, 9091 and 3002 to avoid clashing with other local projects:

```bash
cp .env.example .env
docker compose up -d --wait
uv run alembic upgrade head
uv run pytest -m integration
```

The app itself also needs `JWT_SECRET` in `.env`. Accounts need `INVITE_CODES`, and `CONFIGURED_ADMINS` names the admins.

The integration tests flush Redis and restart the containers, and they refuse to run against any other address. Don't point them at data you want to keep.

`docker compose down -v` deletes all local data. Use `docker compose down` to keep it.

### Web app

You need Node (see `web/.nvmrc`). With the services running and `.env` set up, run these in two terminals:

```bash
uv run uvicorn owl_mind.api.app:app --port 8010
cd web && npm ci && npm run dev
```

Then open the address Vite prints. Set `COOKIE_SECURE=false` in `.env`, or login cookies won't work over plain HTTP. Checks:

```bash
cd web
npm run build
npm test
npx playwright install chromium
npm run e2e
```

`npm run e2e` drives a real browser against the real API, Postgres and Redis with a scripted model, so it costs nothing. It restarts the Postgres and Redis containers, and needs ports 8012, 4173 and 4174 free.

### Telemetry

Every model call is recorded in `ops.model_calls`, with metadata only and no message text. To run the parts that are not in Docker:

```bash
uv run python -m owl_mind.worker.telemetry
uv run uvicorn owl_mind.api.app:app --host 0.0.0.0 --port 8010
```

The first is the consumer that writes the rows. The second serves `/metrics`, which Prometheus scrapes. Port 8010 is open to your network while it runs. Grafana is at http://localhost:3002 (user `admin`, password `owl_dev`).

### Live model test

This test calls DeepSeek and costs a small amount, so it runs only on request and never in CI. Set `DEEPSEEK_API_KEY` in `.env`, then run:

```bash
uv run pytest -m live
```

## Planned stack

- Python 3.14, DeepAgents and FastAPI
- React and Vite
- PostgreSQL with pgvector, and Redis
