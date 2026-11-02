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
from owl_mind.core.config import get_settings
from owl_mind.core.events import RedisEventBus
from owl_mind.core.intent.classifier import IntentStep
from owl_mind.core.model import get_chat_model
from owl_mind.core.persistence import open_persistence
from owl_mind.core.runtime import Services
from owl_mind.services.accounts import PostgresAccountStore, RedisLoginLimiter
from owl_mind.services.admin_store import PostgresAdminStore
from owl_mind.services.models import sqlalchemy_url
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
    try:
        async with open_persistence(settings.database_url) as persistence:
            yield Services(
                admin=PostgresAdminStore(sessions),
                accounts=PostgresAccountStore(sessions),
                limiter=RedisLoginLimiter(client),
                runs=PostgresRunStore(sessions),
                study=PostgresStudyStore(sessions),
                events=RedisEventBus(client),
                store=persistence.store,
                checkpointer=persistence.checkpointer,
                intent=IntentStep(lambda tier, c, m: get_model(tier, component=c, max_tokens=m)),
                get_model=get_model,
            )
    finally:
        await client.aclose()
        await engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    uvicorn.run(create_app(_services), host="127.0.0.1", port=PORT, log_level="warning")
