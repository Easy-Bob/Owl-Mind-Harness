"""FastAPI entry."""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.api import materials  # noqa: F401  Registers its routes on admin_router.
from owl_mind.api.admin import admin_router
from owl_mind.api.auth import auth_router, check_jwt_secret
from owl_mind.api.forum_account import HideCallbackQuery
from owl_mind.api.forum_account import router as forum_account_router
from owl_mind.api.handoffs import router as handoffs_router
from owl_mind.api.routes import router
from owl_mind.core.budget import RedisBudgetLedger
from owl_mind.core.config import ConfigError, get_settings
from owl_mind.core.events import RedisEventBus
from owl_mind.core.intent.classifier import IntentStep
from owl_mind.core.model import get_chat_model
from owl_mind.core.persistence import open_persistence
from owl_mind.core.runtime import Services, read_history
from owl_mind.forum.accounts import ForumAccounts, GitHubOAuth, TokenCipher
from owl_mind.forum.accounts import PostgresAccountStore as PostgresForumAccountStore
from owl_mind.forum.github import GitHubForum
from owl_mind.forum.handoff import HandoffService
from owl_mind.forum.improve import Improver
from owl_mind.forum.store import PostgresHandoffStore
from owl_mind.materials.embedding import FastEmbedEmbedder
from owl_mind.materials.ingest import Ingestor
from owl_mind.materials.search import MaterialsSearch, RedisSearchCache
from owl_mind.materials.store import PostgresMaterialsStore
from owl_mind.materials.topics import TopicNamer
from owl_mind.services.accounts import PostgresAccountStore, RedisLoginLimiter
from owl_mind.services.admin_store import PostgresAdminStore
from owl_mind.services.approvals import PostgresApprovalStore
from owl_mind.services.memory import Memory, MemoryWriter, PostgresMemoryLog
from owl_mind.services.models import sqlalchemy_url
from owl_mind.services.run_events import RunEventLog
from owl_mind.services.run_store import PostgresRunStore
from owl_mind.services.study_store import PostgresStudyStore
from owl_mind.worker.jobs import expire_approvals, jobs  # noqa: F401  Imported from here.
from owl_mind.worker.runs import RedisRunQueue

# Longer than the stream's idle wait, so that waiting isn't mistaken for a lost connection.
_REDIS_TIMEOUT_SECONDS = 15
logger = logging.getLogger(__name__)


def _forum(settings) -> GitHubForum | None:
    """Without the forum settings, a draft still works and the student gets a copyable text."""
    if not (settings.forum_repo and settings.forum_app_id and settings.forum_app_private_key_path):
        return None
    return GitHubForum(
        settings.forum_repo, settings.forum_app_id, settings.forum_app_private_key_path
    )


def _forum_accounts(settings, sessions) -> ForumAccounts | None:
    """Connecting accounts needs the app's client ID and secret, and a key to encrypt tokens."""
    secret, key = settings.forum_client_secret, settings.forum_key_encryption_key
    if not (settings.forum_client_id and secret and key):
        return None
    try:
        cipher = TokenCipher.from_setting(key.get_secret_value())
    except ValueError:
        raise ConfigError("Missing or invalid settings: FORUM_KEY_ENCRYPTION_KEY") from None
    oauth = GitHubOAuth(
        settings.forum_client_id, secret.get_secret_value(), settings.forum_oauth_redirect_url
    )
    return ForumAccounts(PostgresForumAccountStore(sessions), oauth, cipher)


async def _in_process_jobs(services: Services) -> None:
    """With no workers, as in tests and single-process development, the API runs the jobs."""

    async def loop(interval: float, job) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                await job()
            except Exception as exc:
                logger.warning("Background job failed: %s", type(exc).__name__)

    await asyncio.gather(*(loop(interval, job) for _, interval, job in jobs(services)))


@asynccontextmanager
async def _production_services() -> AsyncIterator[Services]:
    settings = get_settings()
    check_jwt_secret(settings)
    engine = create_async_engine(
        sqlalchemy_url(settings.database_url),
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_pool_size,
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    client = Redis.from_url(settings.redis_url, socket_timeout=_REDIS_TIMEOUT_SECONDS)
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir)
    materials_store = PostgresMaterialsStore(sessions)
    # Topics are named only with a model key. Without one, imports still work.
    namer = TopicNamer(get_chat_model) if settings.deepseek_api_key else None
    ingestor = Ingestor(materials_store, embedder, settings.ingest_timeout_seconds, namer)
    search = MaterialsSearch(sessions, materials_store, embedder, RedisSearchCache(client))
    forum = _forum(settings)
    forum_accounts = _forum_accounts(settings, sessions)
    runs = PostgresRunStore(sessions)
    events = RunEventLog(runs)
    handoffs = HandoffService(
        PostgresHandoffStore(sessions),
        forum,
        settings.forum_category,
        forum_accounts,
        events=events,
    )
    try:
        # The store embeds file text with the same model as course search, for memory search.
        async with open_persistence(settings.database_url, embed=embedder.embed) as persistence:
            budgets = RedisBudgetLedger(client)
            memory = (
                MemoryWriter(
                    Memory(persistence.store),
                    PostgresMemoryLog(sessions),
                    lambda thread_id: read_history(persistence.checkpointer, thread_id),
                    get_chat_model,
                    budgets,
                    record=runs.record_event,
                )
                if settings.deepseek_api_key
                else None
            )
            yield Services(
                admin=PostgresAdminStore(sessions),
                accounts=PostgresAccountStore(sessions),
                limiter=RedisLoginLimiter(client),
                runs=runs,
                study=PostgresStudyStore(sessions),
                events=RedisEventBus(client),
                store=persistence.store,
                checkpointer=persistence.checkpointer,
                intent=IntentStep(),
                ingestor=ingestor,
                search=search,
                handoffs=handoffs,
                forum_accounts=forum_accounts,
                budgets=budgets,
                approvals=PostgresApprovalStore(sessions),
                memory=memory,
                # Runs go to workers (M4-B): uv run python -m owl_mind.worker.runs
                queue=RedisRunQueue(client),
                # Only with a model key: without one, the button is hidden.
                improver=Improver(handoffs.store, get_chat_model, events)
                if settings.deepseek_api_key
                else None,
            )
    finally:
        if forum is not None:
            await forum.aclose()
        if forum_accounts is not None:
            await forum_accounts.oauth.aclose()
        await ingestor.shutdown()
        await client.aclose()
        await engine.dispose()


def create_app(services_factory: Callable = _production_services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.run_tasks = set()
        async with services_factory() as services:
            app.state.services = services
            if services.ingestor is not None:
                # The file is only held in memory, so an interrupted import can't resume.
                await services.ingestor.store.fail_unfinished_jobs()
            in_process = None
            if services.queue is None:
                # No workers: any unfinished run was left by this process, and the jobs run here.
                await services.runs.fail_unfinished_runs()
                if services.handoffs is not None:
                    await services.handoffs.store.fail_unfinished()
                in_process = asyncio.create_task(_in_process_jobs(services))
            yield
            if in_process is not None:
                in_process.cancel()
            for task in list(app.state.run_tasks):
                task.cancel()
            await asyncio.gather(*app.state.run_tasks, return_exceptions=True)

    access_log = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, HideCallbackQuery) for f in access_log.filters):
        access_log.addFilter(HideCallbackQuery())
    application = FastAPI(title="Owl-Mind Harness", lifespan=lifespan)
    application.include_router(auth_router)
    application.include_router(admin_router)
    application.include_router(router)
    application.include_router(handoffs_router)
    application.include_router(forum_account_router)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # The default response echoes the submitted input, which here can be a password.
        fields = [{"field": list(error["loc"]), "type": error["type"]} for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": fields})

    @application.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()
