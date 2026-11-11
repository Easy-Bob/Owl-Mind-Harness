"""Token budgets: per run, per student per day, and for the whole system per day.

Counters grow as each call's usage arrives, so a check never waits for the telemetry consumer.
Any doubt counts against spending: a call with no usage counts as its cap, and counters that
can't be read stop agent runs.
"""

import logging
import math
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol, runtime_checkable

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import LLMResult
from redis.asyncio import Redis

from owl_mind.core.config import get_settings

logger = logging.getLogger(__name__)

# US dollars per million tokens, as (input, output). The peak price with no cache discount,
# so the estimate can only be too high. The worker's ops.model_calls cost is the real one.
_PRICES = {
    "deepseek-flash": (0.30, 1.20),
    "deepseek-v4-pro": (1.32, 3.96),
}
_UNKNOWN_MODEL_PRICE = max(_PRICES.values())
# Counted for a call whose cap isn't known, such as one that set no max_tokens.
DEFAULT_OUTPUT_CAP = 4000
_COUNTER_TTL_SECONDS = 2 * 24 * 3600
# A run's own counter only matters while the run can still be resumed (M4-C).
_RUN_TTL_SECONDS = 24 * 3600

WIND_DOWN_SHARE = 0.7
FINAL_SHARE = 0.9


def micro_usd(model: str | None, input_tokens: int, output_tokens: int) -> int:
    """A cost estimate in millionths of a dollar, rounded up. A dollar per million tokens is
    a millionth of a dollar per token, so the rates apply as they are."""
    rate_in, rate_out = _PRICES.get(model or "", _UNKNOWN_MODEL_PRICE)
    return math.ceil(input_tokens * rate_in + output_tokens * rate_out)


class BudgetStop(Exception):
    """No more model calls for this run. Never turned into a tool result."""


class BudgetUnavailable(BudgetStop):
    """The counters couldn't be read or written, so no agent run may start or go on."""


class BudgetExceeded(BudgetStop):
    def __init__(self, scope: str) -> None:
        super().__init__(scope)
        self.scope = scope  # "run", "student" or "global"


@dataclass(frozen=True)
class Spend:
    student_tokens: int
    global_micro_usd: int


@runtime_checkable
class BudgetLedger(Protocol):
    async def spent(self, student_id: uuid.UUID, day: date) -> Spend: ...

    async def charge(
        self, student_id: uuid.UUID, day: date, tokens: int, cost_micro_usd: int
    ) -> None: ...


def _student_key(student_id: uuid.UUID, day: date) -> str:
    return f"budget:student:{day.isoformat()}:{student_id}"


def _global_key(day: date) -> str:
    return f"budget:global:{day.isoformat()}"


def _run_key(run_id: uuid.UUID) -> str:
    return f"budget:run:{run_id}"


class RedisBudgetLedger:
    """Daily counters in Redis. INCRBY is atomic, so two runs at once can't lose a charge."""

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def spent(self, student_id: uuid.UUID, day: date) -> Spend:
        student, total = await self._client.mget(_student_key(student_id, day), _global_key(day))
        return Spend(int(student or 0), int(total or 0))

    async def charge(
        self, student_id: uuid.UUID, day: date, tokens: int, cost_micro_usd: int
    ) -> None:
        async with self._client.pipeline(transaction=False) as pipe:
            pipe.incrby(_student_key(student_id, day), tokens)
            pipe.expire(_student_key(student_id, day), _COUNTER_TTL_SECONDS)
            pipe.incrby(_global_key(day), cost_micro_usd)
            pipe.expire(_global_key(day), _COUNTER_TTL_SECONDS)
            await pipe.execute()

    async def charge_run(self, run_id: uuid.UUID, tokens: int) -> None:
        async with self._client.pipeline(transaction=False) as pipe:
            pipe.incrby(_run_key(run_id), tokens)
            pipe.expire(_run_key(run_id), _RUN_TTL_SECONDS)
            await pipe.execute()

    async def run_spent(self, run_id: uuid.UUID) -> int:
        return int(await self._client.get(_run_key(run_id)) or 0)


class InMemoryBudgetLedger:
    """The same behavior for unit tests. Set down to make every read and write fail."""

    def __init__(self) -> None:
        self.students: dict[tuple[uuid.UUID, date], int] = {}
        self.totals: dict[date, int] = {}
        self.runs: dict[uuid.UUID, int] = {}
        self.down = False

    async def spent(self, student_id: uuid.UUID, day: date) -> Spend:
        if self.down:
            raise ConnectionError("ledger down")
        return Spend(self.students.get((student_id, day), 0), self.totals.get(day, 0))

    async def charge(
        self, student_id: uuid.UUID, day: date, tokens: int, cost_micro_usd: int
    ) -> None:
        if self.down:
            raise ConnectionError("ledger down")
        self.students[(student_id, day)] = self.students.get((student_id, day), 0) + tokens
        self.totals[day] = self.totals.get(day, 0) + cost_micro_usd

    async def charge_run(self, run_id: uuid.UUID, tokens: int) -> None:
        if self.down:
            raise ConnectionError("ledger down")
        self.runs[run_id] = self.runs.get(run_id, 0) + tokens

    async def run_spent(self, run_id: uuid.UUID) -> int:
        if self.down:
            raise ConnectionError("ledger down")
        return self.runs.get(run_id, 0)


def budget_step(share: float) -> str | None:
    """The step of the degrade ladder for the share of a run's budget used so far."""
    if share >= FINAL_SHARE:
        return "final"
    if share >= WIND_DOWN_SHARE:
        return "wind_down"
    return None


class RunMeter(AsyncCallbackHandler):
    """Counts one run's spend as each model call ends, and checks every budget before the next.

    It is passed in the run's config callbacks, so LangChain hands it every model call in the
    run: the intent step, the orchestrator, each tutor and each summary.
    """

    def __init__(
        self,
        ledger: BudgetLedger | None,
        student_id: uuid.UUID,
        run_tokens: int = 0,
        record: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
        run_id: uuid.UUID | None = None,
    ) -> None:
        settings = get_settings()
        self.ledger = ledger
        # Each run's spend is kept too, so a run another worker resumes keeps its count.
        self.run_id = run_id
        self.student_id = student_id
        # Set once the intent step has chosen the kind of request. Until then only the daily
        # budgets apply.
        self.run_tokens = run_tokens
        self.student_tokens = settings.student_daily_tokens
        self.global_micro_usd = round(settings.global_daily_usd * 1_000_000)
        self.day = datetime.now(UTC).date()
        self.used = 0
        # The ladder's step at the last check, so tools can be refused as well as hidden.
        self.step: str | None = None
        self._record = record
        self._steps: set[str] = set()
        self._estimates: dict[uuid.UUID, int] = {}
        self._broken = False

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        params = kwargs.get("invocation_params") or {}
        cap = params.get("max_tokens") or DEFAULT_OUTPUT_CAP
        self._estimates[run_id] = count_tokens_approximately(messages[0]) + cap

    async def on_llm_end(self, response: LLMResult, *, run_id, **kwargs) -> None:
        estimate = self._estimates.pop(run_id, DEFAULT_OUTPUT_CAP)
        message = getattr(response.generations[0][0], "message", None)
        usage = getattr(message, "usage_metadata", None) or {}
        model = (getattr(message, "response_metadata", None) or {}).get("model_name")
        if usage.get("input_tokens") is None or usage.get("output_tokens") is None:
            # Unknown is never zero: count the most the call could have used.
            await self._charge(estimate, micro_usd(model, estimate, 0))
            return
        tokens_in, tokens_out = usage["input_tokens"], usage["output_tokens"]
        await self._charge(tokens_in + tokens_out, micro_usd(model, tokens_in, tokens_out))

    async def on_llm_error(self, error: BaseException, *, run_id, **kwargs) -> None:
        # A failed call can still be billed for what it sent.
        estimate = self._estimates.pop(run_id, DEFAULT_OUTPUT_CAP)
        await self._charge(estimate, micro_usd(None, estimate, 0))

    async def _charge(self, tokens: int, cost: int) -> None:
        self.used += tokens
        try:
            if self.ledger is None:
                raise ConnectionError("no ledger")
            await self.ledger.charge(self.student_id, self.day, tokens, cost)
        except Exception as exc:
            # A charge that may not have counted stops the run at its next check.
            self._broken = True
            logger.warning("Budget charge failed: %s", type(exc).__name__)
            return
        if self.run_id is not None and hasattr(self.ledger, "charge_run"):
            try:
                await self.ledger.charge_run(self.run_id, tokens)
            except Exception as exc:
                logger.warning("Run spend not kept: %s", type(exc).__name__)

    async def restore(self, *run_ids: uuid.UUID | None) -> None:
        """Start from what these runs already spent: a resumed run doesn't get a fresh budget."""
        if self.ledger is None or not hasattr(self.ledger, "run_spent"):
            return
        for run_id in run_ids:
            if run_id is not None:
                try:
                    self.used += await self.ledger.run_spent(run_id)
                except Exception as exc:
                    # Unreadable means unknown, and unknown stops the run at its next check.
                    self._broken = True
                    logger.warning("Run spend unreadable: %s", type(exc).__name__)

    async def daily(self) -> None:
        """Raise unless the student and the system both have budget left today."""
        if self._broken or self.ledger is None:
            raise BudgetUnavailable("budget counters unavailable")
        try:
            spend = await self.ledger.spent(self.student_id, self.day)
        except Exception as exc:
            logger.warning("Budget read failed: %s", type(exc).__name__)
            raise BudgetUnavailable("budget counters unavailable") from None
        if spend.global_micro_usd >= self.global_micro_usd:
            raise BudgetExceeded("global")
        if spend.student_tokens >= self.student_tokens:
            raise BudgetExceeded("student")

    async def check(self) -> str | None:
        """Raise when any budget is used up. Otherwise return the ladder's step, if any."""
        await self.daily()
        share = self.used / self.run_tokens if self.run_tokens else 0.0
        if share >= 1:
            await self._note("stopped", share)
            raise BudgetExceeded("run")
        self.step = budget_step(share)
        if self.step:
            await self._note(self.step, share)
        return self.step

    async def _note(self, step: str, share: float) -> None:
        """Record each step once, for the Admin timeline."""
        if step in self._steps or self._record is None:
            return
        self._steps.add(step)
        await self._record("budget_step", {"step": step, "percent": min(round(share * 100), 999)})
