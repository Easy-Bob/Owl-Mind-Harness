"""The only module that reads the environment."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    # From the M3-E baseline: about 40 learn turns or 8 composite ones a day. Run budgets are in
    # core/limits.py.
    student_daily_tokens: int = 1_000_000
    global_daily_usd: float = 2.0
    # How a two-tutor answer is made: "join" puts the tutors' replies together in code, and
    # "rewrite" has the orchestrator write a new answer from them. Compared in M4-D.
    composite_answer: Literal["join", "rewrite"] = "join"
    # M4-D experiments. Kept ones are on; each setting turns its change off again.
    # Rejected: the orchestrator's first call must call a tutor (completion fell in both runs).
    force_delegation: bool = False
    # Kept: a one-tutor turn goes straight to that tutor, with no orchestrator (cost -17%).
    direct_single_tutor: bool = True
    # Kept: "Quiz me on X" shows a quiz already made from the same notes, with no agent run.
    quiz_reuse: bool = True
    # Who gets live agent answers (M4-G): emails, "@domain" and "invite:<code>", comma-separated.
    # Empty means everyone. See services/cohort.py.
    agent_cohort: SecretStr = SecretStr("")
    # Retention (M4-G): idle chats, and the access log and model calls, are deleted after these.
    retention_chat_days: int = 180
    retention_log_days: int = 400
    # Workers (M4-B, tuned in M4-E): runs at once per worker process, and its /metrics port
    # (0 for none). Each worker on one host needs its own port.
    # 16 from the load test: 4 capped two workers at about 111 turns a minute (evals/load).
    worker_concurrency: int = 16
    worker_metrics_port: int = 9101
    # Database connections per process, for each pool (SQLAlchemy, the saver, the store).
    # 5 held 32 runs at once in the load test with no errors; 12 changed nothing.
    db_pool_size: int = 5
    # The large tier's model. Changed only with the owner's approval of the M3-F comparison.
    large_model: Literal["deepseek-flash", "deepseek-v4-pro"] = "deepseek-flash"
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
