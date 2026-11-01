"""Alembic environment. DATABASE_URL comes from core/config.py."""

from alembic import context
from sqlalchemy import create_engine, text

from owl_mind.core.config import get_settings
from owl_mind.services.models import Base

SCHEMA = "app"


def sqlalchemy_url(database_url: str) -> str:
    """Select the psycopg 3 driver, which is the only Postgres driver installed."""
    return database_url.replace("postgresql://", "postgresql+psycopg://", 1)


def include_name(name, type_, parent_names):
    # Autogenerate looks only at app. The other schemas have their own owners.
    return name == SCHEMA if type_ == "schema" else True


def run_migrations() -> None:
    engine = create_engine(sqlalchemy_url(get_settings().database_url))
    with engine.connect() as connection:
        # The version table lives in app, so app must exist before the first migration.
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        connection.commit()
        context.configure(
            connection=connection,
            target_metadata=Base.metadata,
            version_table_schema=SCHEMA,
            include_schemas=True,
            include_name=include_name,
        )
        with context.begin_transaction():
            context.run_migrations()


run_migrations()
