"""Builds one agent per turn. The only module that calls create_deep_agent."""

from collections.abc import Callable, Sequence

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import TAG_NOSTREAM
from langgraph.store.base import BaseStore

from owl_mind.backends.composite import SKILL_SOURCES, build_backend
from owl_mind.backends.permissions import ORCHESTRATOR_PERMISSIONS, TUTOR_PERMISSIONS
from owl_mind.core import prompts
from owl_mind.core.budget import RunMeter
from owl_mind.core.config import get_settings
from owl_mind.core.integrity import IntegrityCheck, needs_check
from owl_mind.core.intent.taxonomy import Subagent, TaskKind
from owl_mind.core.limits import RUN_LIMITS
from owl_mind.core.middleware import JoinAnswerMiddleware, ReportCheckMiddleware, turn_middleware
from owl_mind.core.model import ModelTier, get_chat_model
from owl_mind.core.subagents import TUTORS
from owl_mind.tools.context import RunContext
from owl_mind.tools.planner import get_due_topics

_ORCHESTRATOR_MAX_TOKENS = 1200
# The owner's choice. DeepAgents would wait for 85% of the model's 1,000,000-token window.
SUMMARY_TRIGGER = ("tokens", 24_000)
SUMMARY_KEEP = ("tokens", 4_000)
# Everything older than the kept part, up to the trigger. DeepAgents would read only 4,000.
_SUMMARY_INPUT_TOKENS = 20_000
_SUMMARY_MAX_TOKENS = 1000

# DeepAgents adds a general-purpose subagent with all the main agent's tools unless a profile
# turns it off. Left on, the model could use it to get around every tutor's scope.
register_harness_profile(
    "deepseek",
    HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools=frozenset({"execute"}),
    ),
)


class LiveAgentRunsDisabled(RuntimeError):
    """LIVE_AGENT_RUNS_ENABLED is off, so no agent runs."""


class BudgetedSummarization(SummarizationMiddleware):
    """DeepAgents' summarization with our trigger, checking the budgets before a summary call.

    It sits outside our middleware in DeepAgents' stack, so it checks for itself.
    """

    def __init__(self, meter: RunMeter, **kwargs) -> None:
        super().__init__(**kwargs)
        self.meter = meter

    @property
    def name(self) -> str:
        # The same name as DeepAgents' own, so create_deep_agent puts this one in its place.
        return "SummarizationMiddleware"

    async def awrap_model_call(self, request, handler):
        await self.meter.check()
        return await super().awrap_model_call(request, handler)


GetModel = Callable[..., BaseChatModel]


def build_owl_agent(
    task_kind: TaskKind,
    subagents: Sequence[Subagent],
    *,
    store: BaseStore,
    meter: RunMeter,
    checkpointer: BaseCheckpointSaver | None = None,
    get_model: GetModel = get_chat_model,
):
    """Assemble the setup for one turn. The model chooses the steps inside these limits."""
    if not get_settings().live_agent_runs_enabled:
        raise LiveAgentRunsDisabled("LIVE_AGENT_RUNS_ENABLED is off")

    limit = RUN_LIMITS[task_kind]
    backend = build_backend()

    def checks(orchestrator: bool):
        stack = turn_middleware(
            task_kind,
            subagents,
            limit.tool_calls,
            limit.seconds,
            meter,
            orchestrator=orchestrator,
        )
        summary_model = get_model(
            ModelTier.SMALL, component="summarization", max_tokens=_SUMMARY_MAX_TOKENS
        )
        # It runs inside the agent's model step, so untagged its text would stream to the
        # student as if it were the answer.
        summary_model.tags = [*(summary_model.tags or []), TAG_NOSTREAM]
        summarization = BudgetedSummarization(
            meter,
            model=summary_model,
            backend=backend,
            trigger=SUMMARY_TRIGGER,
            keep=SUMMARY_KEEP,
            trim_tokens_to_summarize=_SUMMARY_INPUT_TOKENS,
        )
        return [*stack, summarization]

    specs = [
        {
            "name": subagent.value,
            "description": TUTORS[subagent].description,
            "system_prompt": TUTORS[subagent].system_prompt,
            "tools": list(TUTORS[subagent].tools),
            "permissions": TUTOR_PERMISSIONS[subagent],
            "skills": SKILL_SOURCES,
            "middleware": checks(orchestrator=False),
            "model": get_model(
                ModelTier.SMALL,
                component=f"agent:{subagent.value}",
                max_tokens=TUTORS[subagent].max_tokens,
            ),
        }
        for subagent in subagents
    ]
    # Only a two-tutor turn has work to plan. Elsewhere write_todos would just add tokens.
    is_composite = task_kind is TaskKind.COMPOSITE
    todos = [TodoListMiddleware()] if is_composite else []
    joins = is_composite and get_settings().composite_answer == "join"
    join = [JoinAnswerMiddleware(tuple(s.value for s in subagents))] if joins else []
    # Checked turns are reviewed at the end. Elsewhere, a tutor's reply is checked on arrival.
    report_check = (
        [] if needs_check(subagents) else [ReportCheckMiddleware(IntegrityCheck(get_model))]
    )
    return create_deep_agent(
        model=get_model(
            ModelTier.LARGE if is_composite else ModelTier.SMALL,
            component="orchestrator",
            max_tokens=_ORCHESTRATOR_MAX_TOKENS,
        ),
        tools=[get_due_topics],
        system_prompt=prompts.ORCHESTRATOR,
        subagents=specs,
        middleware=[*join, *checks(orchestrator=True), *report_check, *todos],
        context_schema=RunContext,
        backend=backend,
        permissions=ORCHESTRATOR_PERMISSIONS,
        store=store,
        checkpointer=checkpointer,
    )
