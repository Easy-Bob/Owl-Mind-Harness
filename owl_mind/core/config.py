"""The only module that reads the environment."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """A required setting is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    database_url: str
    redis_url: str
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    intent_layer3_enabled: bool = False
    live_agent_runs_enabled: bool = False
    jwt_secret: SecretStr | None = None
    cookie_secure: bool = True
    configured_admins: str = ""
    invite_codes: SecretStr = SecretStr("")
    max_upload_mb: int = 25
    ingest_timeout_seconds: int = 120
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_dir: Path = Path.home() / ".cache" / "owl-mind" / "fastembed"
    forum_repo: str = ""
    forum_category: str = "Q&A"
    forum_app_id: str | None = None
    forum_app_private_key_path: Path | None = None
    forum_client_id: str | None = None
    forum_client_secret: SecretStr | None = None
    forum_oauth_redirect_url: str = "http://localhost:5173/forum/account/callback"
    forum_key_encryption_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        # Name the settings only. The original error text can include their values.
        names = sorted({str(error["loc"][0]).upper() for error in exc.errors()})
        raise ConfigError(f"Missing or invalid settings: {', '.join(names)}") from None
