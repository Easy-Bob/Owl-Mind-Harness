"""Needs the compose services: docker compose up -d --wait postgres redis

Run with: uv run pytest -m integration
"""

import json
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import psycopg
import pytest
import redis

from owl_mind.core.config import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = {"lg_checkpoint", "lg_store", "app", "materials", "ops"}


def run(*command: str) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)


def alembic(*arguments: str) -> None:
    run(sys.executable, "-m", "alembic", *arguments)


def probe(action: str, thread_id: str) -> dict | None:
    done = run(sys.executable, "-m", "tests.persistence_probe", action, thread_id)
    return json.loads(done.stdout) if action == "read" else None


def query(sql: str) -> list[tuple]:
    with psycopg.connect(get_settings().database_url) as connection:
        return connection.execute(sql).fetchall()


def tables_in(schema: str) -> set[str]:
    rows = query(f"SELECT tablename FROM pg_tables WHERE schemaname = '{schema}'")
    return {row[0] for row in rows}


def assert_local_compose_services() -> None:
    """These tests flush Redis and restart containers, so never run them on anything else."""
    settings = get_settings()
    database, cache = urlsplit(settings.database_url), urlsplit(settings.redis_url)
    assert (database.hostname, database.port) == ("localhost", 5433), "not the compose Postgres"
    assert (cache.hostname, cache.port) == ("localhost", 6380), "not the compose Redis"


@pytest.fixture(scope="module", autouse=True)
def migrated():
    assert_local_compose_services()
    alembic("upgrade", "head")


def test_migration_creates_schemas_and_first_tables():
    rows = query("SELECT schema_name FROM information_schema.schemata")

    assert SCHEMAS <= {row[0] for row in rows}
    assert {"students", "threads", "runs"} <= tables_in("app")


def test_migrating_twice_is_safe():
    alembic("upgrade", "head")


def test_models_match_migrations():
    alembic("check")


def test_thread_and_store_survive_a_restart_of_process_and_containers():
    thread_id = str(uuid.uuid4())
    probe("write", thread_id)

    run("docker", "compose", "restart", "postgres", "redis")
    run("docker", "compose", "up", "-d", "--wait", "postgres", "redis")

    result = probe("read", thread_id)
    assert result["messages"] == ["hello", "reply 1"]
    assert result["store"] == {"text": "weak on recursion"}
    assert thread_id in result["search"]


def test_saver_and_store_tables_stay_in_their_schemas():
    probe("write", str(uuid.uuid4()))

    assert {"checkpoints", "checkpoint_migrations"} <= tables_in("lg_checkpoint")
    assert {"store", "store_vectors", "store_migrations"} <= tables_in("lg_store")
    assert tables_in("public") == set()
    assert "checkpoints" not in tables_in("lg_store")
    assert "store" not in tables_in("lg_checkpoint")


def test_store_has_a_vector_index():
    rows = query(
        "SELECT indexdef FROM pg_indexes"
        " WHERE schemaname = 'lg_store' AND tablename = 'store_vectors'"
    )

    assert any("hnsw" in row[0] for row in rows)


def test_flushing_redis_does_not_touch_history():
    thread_id = str(uuid.uuid4())
    probe("write", thread_id)

    redis.Redis.from_url(get_settings().redis_url).flushdb()

    assert probe("read", thread_id)["messages"] == ["hello", "reply 1"]


def test_unknown_thread_loads_as_empty():
    result = probe("read", str(uuid.uuid4()))

    assert result["messages"] == []
    assert result["store"] is None


def test_writing_the_same_thread_twice_keeps_both_turns():
    thread_id = str(uuid.uuid4())
    probe("write", thread_id)
    probe("write", thread_id)

    assert probe("read", thread_id)["messages"] == ["hello", "reply 1", "hello", "reply 3"]
