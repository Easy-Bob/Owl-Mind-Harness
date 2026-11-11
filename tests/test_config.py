import socket
from pathlib import Path

import pytest
from pytest_socket import SocketBlockedError

from owl_mind.core.config import ConfigError, get_settings

ENV_NAMES = (
    "DATABASE_URL",
    "REDIS_URL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "INTENT_LAYER3_ENABLED",
    "LIVE_AGENT_RUNS_ENABLED",
    "STUDENT_DAILY_TOKENS",
    "GLOBAL_DAILY_USD",
    "COMPOSITE_ANSWER",
    "LARGE_MODEL",
    "JWT_SECRET",
    "COOKIE_SECURE",
    "CONFIGURED_ADMINS",
    "INVITE_CODES",
    "MAX_UPLOAD_MB",
    "INGEST_TIMEOUT_SECONDS",
    "EMBEDDING_MODEL",
    "EMBEDDING_CACHE_DIR",
    "FORUM_REPO",
    "FORUM_CATEGORY",
    "FORUM_APP_ID",
    "FORUM_APP_PRIVATE_KEY_PATH",
    "FORUM_CLIENT_ID",
    "FORUM_CLIENT_SECRET",
    "FORUM_OAUTH_REDIRECT_URL",
    "FORUM_KEY_ENCRYPTION_KEY",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def set_required(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("REDIS_URL", "redis://cache")


def test_reads_settings_from_the_environment(monkeypatch):
    set_required(monkeypatch)

    settings = get_settings()

    assert settings.database_url == "postgresql://db"
    assert settings.redis_url == "redis://cache"
    assert settings.deepseek_api_key is None
    assert settings.deepseek_base_url == "https://api.deepseek.com"


def test_reads_settings_from_a_dotenv_file(tmp_path):
    (tmp_path / ".env").write_text("DATABASE_URL=postgresql://file\nREDIS_URL=redis://file\n")

    assert get_settings().database_url == "postgresql://file"


def test_missing_setting_is_named(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://cache")

    with pytest.raises(ConfigError, match="DATABASE_URL"):
        get_settings()


def test_blank_setting_counts_as_missing(monkeypatch):
    set_required(monkeypatch)
    monkeypatch.setenv("REDIS_URL", "")

    with pytest.raises(ConfigError, match="REDIS_URL"):
        get_settings()


def test_error_does_not_contain_setting_values(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")

    with pytest.raises(ConfigError) as caught:
        get_settings()

    assert "not-a-real-key" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_api_key_is_hidden_when_printed(monkeypatch):
    set_required(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")

    assert "not-a-real-key" not in repr(get_settings())


def test_example_file_lists_every_setting():
    example = Path(__file__).resolve().parent.parent / ".env.example"
    names = {
        line.split("=")[0]
        for line in example.read_text().splitlines()
        if line and not line.startswith("#")
    }

    assert names == set(ENV_NAMES)


def test_default_test_run_blocks_the_network():
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
