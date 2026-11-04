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
from owl_mind.core.runtime import Services
from owl_mind.forum.accounts import ForumAccounts, GitHubOAuth, TokenCipher
from owl_mind.forum.accounts import PostgresAccountStore as PostgresForumAccountStore
from owl_mind.forum.github import GitHubForum
from owl_mind.forum.handoff import HandoffService
from owl_mind.forum.improve import Improver
from owl_mind.forum.replies import ReplyPoller
from owl_mind.forum.store import PostgresHandoffStore
from owl_mind.materials.embedding import FastEmbedEmbedder
from owl_mind.materials.ingest import Ingestor
from owl_mind.materials.search import MaterialsSearch, RedisSearchCache
from owl_mind.materials.store import PostgresMaterialsStore
from owl_mind.services.accounts import PostgresAccountStore, RedisLoginLimiter
from owl_mind.services.admin_store import PostgresAdminStore
from owl_mind.services.approvals import PostgresApprovalStore
from owl_mind.services.models import sqlalchemy_url
from owl_mind.services.run_events import RunEventLog
from owl_mind.services.run_store import PostgresRunStore
from owl_mind.services.study_store import PostgresStudyStore

# Longer than the stream's idle wait, so that waiting isn't mistaken for a lost connection.
_REDIS_TIMEOUT_SECONDS = 15
_HANDOFF_RETRY_SECONDS = 60
_APPROVAL_CHECK_SECONDS = 300
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


async def expire_approvals(services: Services) -> int:
    """Expired approvals count as rejected: their runs stop and free the student's slot."""
    runs = await services.approvals.expire_due()
    for run_id in runs:
        await services.runs.record_event(
            run_id, "approval:expired", "approval", {"step": "expired"}
        )
        await services.runs.set_run_status(run_id, "stopped")
    return len(runs)


async def _approval_jobs(services: Services) -> None:
    while True:
        await asyncio.sleep(_APPROVAL_CHECK_SECONDS)
        try:
            await expire_approvals(services)
        except Exception as exc:
            logger.warning("Approval job failed: %s", type(exc).__name__)


async def _forum_jobs(services: Services) -> None:
    """Retries approved posts and checks posts for replies. Moves to workers in M4-B."""
    forum = services.handoffs.forum
    handoffs = services.handoffs
    poller = (
        ReplyPoller(handoffs.store, forum, events=handoffs.events) if forum is not None else None
    )
    while True:
        await asyncio.sleep(_HANDOFF_RETRY_SECONDS)
        try:
            await services.handoffs.retry_pending()
            if poller is not None:
                await poller.poll_due()
        except Exception as exc:
            logger.warning("Forum job failed: %s", type(exc).__name__)


@asynccontextmanager
async def _production_services() -> AsyncIterator[Services]:
    settings = get_settings()
    check_jwt_secret(settings)
    engine = create_async_engine(sqlalchemy_url(settings.database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    client = Redis.from_url(settings.redis_url, socket_timeout=_REDIS_TIMEOUT_SECONDS)
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir)
    materials_store = PostgresMaterialsStore(sessions)
    ingestor = Ingestor(materials_store, embedder, settings.ingest_timeout_seconds)
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
        async with open_persistence(settings.database_url) as persistence:
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
                budgets=RedisBudgetLedger(client),
                approvals=PostgresApprovalStore(sessions),
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
            # One process until M4-B, so any unfinished run was left by a restart.
            await services.runs.fail_unfinished_runs()
            if services.ingestor is not None:
                # The file is only held in memory, so an interrupted import can't resume.
                await services.ingestor.store.fail_unfinished_jobs()
            retrying = None
            expiring = (
                asyncio.create_task(_approval_jobs(services))
                if services.approvals is not None
                else None
            )
            if services.handoffs is not None:
                await services.handoffs.store.fail_unfinished()
                retrying = asyncio.create_task(_forum_jobs(services))
            yield
            for job in (retrying, expiring):
                if job is not None:
                    job.cancel()
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
