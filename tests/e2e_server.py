"""The real API on real Postgres and Redis, with a scripted model. Used by the browser tests.

Every model has its network layer replaced, so this can never reach DeepSeek or spend tokens.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.api.app import create_app
from owl_mind.core.budget import RedisBudgetLedger
from owl_mind.core.config import get_settings
from owl_mind.core.events import RedisEventBus
from owl_mind.core.intent.classifier import IntentStep
from owl_mind.core.model import get_chat_model
from owl_mind.core.persistence import open_persistence
from owl_mind.core.runtime import Services
from owl_mind.forum.github import ForumPost, Replies, SimilarPost
from owl_mind.forum.handoff import HandoffService
from owl_mind.forum.store import PostgresHandoffStore
from owl_mind.services.accounts import PostgresAccountStore, RedisLoginLimiter
from owl_mind.services.admin_store import PostgresAdminStore
from owl_mind.services.models import sqlalchemy_url
from owl_mind.services.run_events import RunEventLog
from owl_mind.services.run_store import PostgresRunStore
from owl_mind.services.study_store import PostgresStudyStore
from tests.fake_deepseek import USAGE, FakeDeepSeek, _chunk, completion

PORT = 8012
SLOW_WORDS = 30
SLOW_DELAY_SECONDS = 0.15
UNSAFE_ANSWER = (
    'Unsafe <script>window.pwned = true</script> <img src=x onerror="window.pwned = true"> text '
    "with **bold** kept."
)


def _answer_for(message: str) -> tuple[list[str], float]:
    if "slow" in message:
        return [f"word{index} " for index in range(SLOW_WORDS)], SLOW_DELAY_SECONDS
    if "unsafe" in message:
        return [UNSAFE_ANSWER], 0
    return [f"Answer to: {message}"], 0


class ScriptedDeepSeek(FakeDeepSeek):
    def __init__(self) -> None:
        super().__init__([])
        self.handler = self.respond

    def respond(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        message = next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")
        if not body.get("stream"):
            intents = [{"intent": "concept_explain", "confidence": 0.9}]
            reply = {"role": "assistant", "content": json.dumps({"intents": intents})}
            return httpx.Response(200, json=completion(reply))
        pieces, delay = _answer_for(message)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_stream(pieces, delay)
        )


async def _stream(pieces: list[str], delay: float):
    yield _line(_chunk({"role": "assistant", "content": ""}))
    for piece in pieces:
        await asyncio.sleep(delay)
        yield _line(_chunk({"content": piece}))
    yield _line(_chunk({}, finish_reason="stop"))
    yield _line(dict(_chunk({}), choices=[], usage=USAGE))
    yield b"data: [DONE]\n\n"


def _line(chunk: dict) -> bytes:
    return f"data: {json.dumps(chunk)}\n\n".encode()


class LocalForum:
    """Stands in for GitHub, so a browser test can never post to the real forum."""

    # Shaped like GitHub's, because Admin shows only GitHub discussion links.
    home_url = "https://github.com/e2e-owner/e2e-forum/discussions"

    def __init__(self) -> None:
        self.posts: list[tuple[str, str]] = []

    async def search_similar(self, text: str, limit: int = 3) -> list[SimilarPost]:
        return [SimilarPost("An older question", f"{self.home_url}/1")]

    async def post_question(
        self, category: str, title: str, body: str, user_token: str | None = None
    ) -> ForumPost:
        self.posts.append((title, body))
        return ForumPost(f"{self.home_url}/{len(self.posts)}", f"D_{len(self.posts)}")

    async def get_replies(self, node_id: str) -> Replies:
        return Replies([], True)


@asynccontextmanager
async def _services():
    settings = get_settings()
    fake = ScriptedDeepSeek()

    def get_model(tier, *, component, max_tokens):
        model = get_chat_model(tier, component=component, max_tokens=max_tokens)
        fake.attach(model)
        return model

    engine = create_async_engine(sqlalchemy_url(settings.database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    client = Redis.from_url(settings.redis_url, socket_timeout=15)
    runs = PostgresRunStore(sessions)
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
                intent=IntentStep(lambda tier, c, m: get_model(tier, component=c, max_tokens=m)),
                get_model=get_model,
                budgets=RedisBudgetLedger(client),
                handoffs=HandoffService(
                    PostgresHandoffStore(sessions), LocalForum(), "Q&A", events=RunEventLog(runs)
                ),
            )
    finally:
        await client.aclose()
        await engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    uvicorn.run(create_app(_services), host="127.0.0.1", port=PORT, log_level="warning")
