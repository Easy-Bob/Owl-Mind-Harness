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

## Planned stack

- Python 3.14, DeepAgents and FastAPI
- React and Vite
- PostgreSQL with pgvector, and Redis
