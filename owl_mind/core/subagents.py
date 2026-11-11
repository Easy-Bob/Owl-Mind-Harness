"""Each tutor's prompt, tools and spending limit. Tool order is fixed to keep prompts cacheable."""

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from owl_mind.core import prompts
from owl_mind.core.intent.taxonomy import Subagent
from owl_mind.tools.concept import get_prerequisites
from owl_mind.tools.materials import materials_search
from owl_mind.tools.memory import search_memory
from owl_mind.tools.planner import (
    commit_study_plan,
    get_due_topics,
    preview_review_dates,
    propose_study_plan,
    reset_progress,
)
from owl_mind.tools.practice import analyze_complexity, build_hint
from owl_mind.tools.quiz import post_quiz


@dataclass(frozen=True)
class TutorSpec:
    description: str
    system_prompt: str
    tools: tuple[BaseTool, ...]
    max_tokens: int


# The scopes on their own, so a guard rail can compare them with what each tutor is built with.
TOOL_SCOPES: dict[Subagent, tuple[str, ...]] = {
    Subagent.CONCEPT_TUTOR: ("materials_search", "get_prerequisites"),
    Subagent.PRACTICE_COACH: ("materials_search", "build_hint", "analyze_complexity"),
    Subagent.STUDY_PLANNER: (
        "preview_review_dates",
        "get_due_topics",
        "propose_study_plan",
        "commit_study_plan",
        "reset_progress",
        "search_memory",
    ),
    Subagent.QUIZ_MASTER: ("materials_search", "post_quiz"),
}

TUTORS: dict[Subagent, TutorSpec] = {
    Subagent.CONCEPT_TUTOR: TutorSpec(
        "Explains computer-science concepts.",
        prompts.CONCEPT_TUTOR,
        (materials_search, get_prerequisites),
        1200,
    ),
    Subagent.PRACTICE_COACH: TutorSpec(
        "Guides a student through a problem with hints, without giving solutions.",
        prompts.PRACTICE_COACH,
        (materials_search, build_hint, analyze_complexity),
        1000,
    ),
    Subagent.STUDY_PLANNER: TutorSpec(
        "Builds study plans and schedules reviews.",
        prompts.STUDY_PLANNER,
        (
            preview_review_dates,
            get_due_topics,
            propose_study_plan,
            commit_study_plan,
            reset_progress,
            search_memory,
        ),
        800,
    ),
    Subagent.QUIZ_MASTER: TutorSpec(
        "Writes multiple-choice quizzes, with the answers at the end.",
        prompts.QUIZ_MASTER,
        (materials_search, post_quiz),
        1200,
    ),
}
