"""FastAPI entry."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.api.routes import router
from owl_mind.core.config import get_settings
from owl_mind.core.events import RedisEventBus
from owl_mind.core.intent.classifier import IntentStep
from owl_mind.core.persistence import open_persistence
from owl_mind.core.runtime import Services
from owl_mind.services.models import sqlalchemy_url
from owl_mind.services.run_store import PostgresRunStore
from owl_mind.services.study_store import PostgresStudyStore

# Longer than the stream's idle wait, so that waiting isn't mistaken for a lost connection.
_REDIS_TIMEOUT_SECONDS = 15


@asynccontextmanager
async def _production_services() -> AsyncIterator[Services]:
    settings = get_settings()
    engine = create_async_engine(sqlalchemy_url(settings.database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    client = Redis.from_url(settings.redis_url, socket_timeout=_REDIS_TIMEOUT_SECONDS)
    try:
        async with open_persistence(settings.database_url) as persistence:
            yield Services(
                runs=PostgresRunStore(sessions),
                study=PostgresStudyStore(sessions),
                events=RedisEventBus(client),
                store=persistence.store,
                checkpointer=persistence.checkpointer,
                intent=IntentStep(),
            )
    finally:
        await client.aclose()
        await engine.dispose()


def create_app(services_factory: Callable = _production_services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.run_tasks = set()
        async with services_factory() as services:
            app.state.services = services
            # One process until M4-B, so any unfinished run was left by a restart.
            await services.runs.fail_unfinished_runs()
            yield
            for task in list(app.state.run_tasks):
                task.cancel()
            await asyncio.gather(*app.state.run_tasks, return_exceptions=True)

    application = FastAPI(title="Owl-Mind Harness", lifespan=lifespan)
    application.include_router(router)

    @application.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()
