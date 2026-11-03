"""Builds one agent per turn. The only module that calls create_deep_agent."""

from collections.abc import Callable, Sequence

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from owl_mind.backends.composite import SKILL_SOURCES, build_backend
from owl_mind.backends.permissions import ORCHESTRATOR_PERMISSIONS, TUTOR_PERMISSIONS
from owl_mind.core import prompts
from owl_mind.core.config import get_settings
from owl_mind.core.intent.taxonomy import Subagent, TaskKind
from owl_mind.core.limits import RUN_LIMITS
from owl_mind.core.middleware import turn_middleware
from owl_mind.core.model import ModelTier, get_chat_model
from owl_mind.core.subagents import TUTORS
from owl_mind.tools.context import RunContext
from owl_mind.tools.planner import get_due_topics

_ORCHESTRATOR_MAX_TOKENS = 1200

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
    """No live agent runs until M3-E delivers token budgets."""


GetModel = Callable[..., BaseChatModel]


def build_owl_agent(
    task_kind: TaskKind,
    subagents: Sequence[Subagent],
    *,
    store: BaseStore,
    checkpointer: BaseCheckpointSaver | None = None,
    get_model: GetModel = get_chat_model,
):
    """Assemble the setup for one turn. The model chooses the steps inside these limits."""
    if not get_settings().live_agent_runs_enabled:
        raise LiveAgentRunsDisabled("No live agent runs until M3-E delivers token budgets")

    limit = RUN_LIMITS[task_kind]

    def checks(orchestrator: bool):
        return turn_middleware(
            task_kind, subagents, limit.tool_calls, limit.seconds, orchestrator=orchestrator
        )

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
    return create_deep_agent(
        model=get_model(
            ModelTier.LARGE if is_composite else ModelTier.SMALL,
            component="orchestrator",
            max_tokens=_ORCHESTRATOR_MAX_TOKENS,
        ),
        tools=[get_due_topics],
        system_prompt=prompts.ORCHESTRATOR,
        subagents=specs,
        middleware=[*checks(orchestrator=True), *([TodoListMiddleware()] if is_composite else [])],
        context_schema=RunContext,
        backend=build_backend(),
        permissions=ORCHESTRATOR_PERMISSIONS,
        store=store,
        checkpointer=checkpointer,
    )
