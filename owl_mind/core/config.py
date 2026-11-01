"""The only module that reads the environment."""

from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        # Name the settings only. The original error text can include their values.
        names = sorted({str(error["loc"][0]).upper() for error in exc.errors()})
        raise ConfigError(f"Missing or invalid settings: {', '.join(names)}") from None
