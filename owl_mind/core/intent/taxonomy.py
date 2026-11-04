"""The 19 intents and the setup each one leads to. An intent never holds agent steps."""

from dataclasses import dataclass
from enum import StrEnum


class Intent(StrEnum):
    CONCEPT_EXPLAIN = "concept_explain"
    CONCEPT_COMPARE = "concept_compare"
    MATERIAL_SEARCH = "material_search"
    PROBLEM_HELP = "problem_help"
    HOMEWORK_CHECK = "homework_check"
    CODE_REVIEW = "code_review"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    STUDY_PLAN = "study_plan"
    PROGRESS_QUERY = "progress_query"
    DEADLINE = "deadline"
    QUIZ_REQUEST = "quiz_request"
    ANSWER_SUBMISSION = "answer_submission"
    MOCK_EXAM = "mock_exam"
    EXPLAIN_BACK = "explain_back"
    MOTIVATION = "motivation"
    HUMAN_TUTOR = "human_tutor"
    GREETING = "greeting"
    FEEDBACK = "feedback"
    OTHER = "other"


class TaskKind(StrEnum):
    LEARN = "learn"
    PRACTICE = "practice"
    PLAN = "plan"
    ASSESS = "assess"
    SUPPORT = "support"
    COMPOSITE = "composite"


class Route(StrEnum):
    RULE = "rule"
    TEMPLATE = "template"
    ORCHESTRATED = "orchestrated"
    HANDOFF = "handoff"


class Subagent(StrEnum):
    CONCEPT_TUTOR = "concept_tutor"
    PRACTICE_COACH = "practice_coach"
    STUDY_PLANNER = "study_planner"
    QUIZ_MASTER = "quiz_master"


class Reply(StrEnum):
    """Fixed replies that need no model. Their wording belongs to the route that sends them."""

    EMPTY = "empty"
    GREETING = "greeting"
    THANKS = "thanks"
    OFF_TOPIC = "off_topic"
    DUE_TOPICS = "due_topics"
    PROGRESS = "progress"
    FEEDBACK = "feedback"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class IntentSpec:
    task_kind: TaskKind | None
    route: Route
    subagent: Subagent | None
    reply: Reply | None
    description: str
    example: str


def _learn(description: str, example: str) -> IntentSpec:
    return IntentSpec(
        TaskKind.LEARN, Route.ORCHESTRATED, Subagent.CONCEPT_TUTOR, None, description, example
    )


def _practice(description: str, example: str) -> IntentSpec:
    return IntentSpec(
        TaskKind.PRACTICE, Route.ORCHESTRATED, Subagent.PRACTICE_COACH, None, description, example
    )


def _plan(description: str, example: str) -> IntentSpec:
    return IntentSpec(
        TaskKind.PLAN, Route.ORCHESTRATED, Subagent.STUDY_PLANNER, None, description, example
    )


def _assess(description: str, example: str) -> IntentSpec:
    return IntentSpec(
        TaskKind.ASSESS, Route.ORCHESTRATED, Subagent.QUIZ_MASTER, None, description, example
    )


def _template(reply: Reply, description: str, example: str) -> IntentSpec:
    return IntentSpec(None, Route.TEMPLATE, None, reply, description, example)


INTENTS: dict[Intent, IntentSpec] = {
    Intent.CONCEPT_EXPLAIN: _learn(
        "explain a computer-science concept", "how does a hash table resolve collisions"
    ),
    Intent.CONCEPT_COMPARE: _learn("compare concepts", "stack or queue, when do I use each"),
    Intent.MATERIAL_SEARCH: _learn("find course material", "which lecture covers paging"),
    Intent.PROBLEM_HELP: _practice("stuck on a problem", "I can't start this graph problem"),
    Intent.HOMEWORK_CHECK: _practice("check an answer or proof", "did I solve exercise 2 right"),
    Intent.CODE_REVIEW: _practice("debug or review code", "why does my loop never stop"),
    Intent.COMPLEXITY_ANALYSIS: _practice("time or space cost", "what is the runtime of this code"),
    Intent.STUDY_PLAN: _plan("plan or schedule studying", "plan my revision for the final"),
    Intent.DEADLINE: _plan("a date or deadline", "my project is due on Monday"),
    Intent.PROGRESS_QUERY: _template(
        Reply.PROGRESS, "ask about their own progress", "what am I weak at so far"
    ),
    Intent.QUIZ_REQUEST: _assess("ask to be quizzed", "ask me some questions about sorting"),
    Intent.ANSWER_SUBMISSION: _assess("answer a quiz question", "I would say the answer is B"),
    Intent.MOCK_EXAM: _assess("ask for a full practice exam", "set me a timed midterm"),
    Intent.EXPLAIN_BACK: _assess(
        "explain a topic to be checked", "here is how I think DNS works, am I right"
    ),
    Intent.MOTIVATION: IntentSpec(
        TaskKind.SUPPORT,
        Route.ORCHESTRATED,
        None,
        None,
        "discouraged or stressed",
        "I don't think I can pass this class",
    ),
    Intent.HUMAN_TUTOR: IntentSpec(
        None, Route.HANDOFF, None, None, "ask for a human", "can a real tutor help me"
    ),
    Intent.GREETING: _template(Reply.GREETING, "say hello", "hey, how is it going"),
    Intent.FEEDBACK: _template(
        Reply.FEEDBACK, "comment on the assistant", "that last hint was unhelpful"
    ),
    # Owner's rule: anything that isn't computer science gets the off-topic reply, including
    # simple arithmetic and other subjects, so no agent runs on it.
    Intent.OTHER: _template(
        Reply.OFF_TOPIC,
        "not about computer science, such as everyday arithmetic, other subjects or general "
        "knowledge. Maths inside computer science, such as binary, logic or complexity, is not "
        "other",
        "tell me how to calculate 1 + 1",
    ),
}


def check_taxonomy() -> None:
    """Fail at startup if the table no longer covers every intent correctly."""
    missing = set(Intent) - set(INTENTS)
    if missing:
        raise ValueError(f"Intents with no setup: {sorted(missing)}")
    for intent, spec in INTENTS.items():
        if not spec.description or not spec.example:
            raise ValueError(f"{intent} needs a description and an example")
        if spec.route is Route.TEMPLATE and spec.reply is None:
            raise ValueError(f"{intent} uses the template route but has no reply")
        if spec.route is Route.ORCHESTRATED and spec.task_kind is None:
            raise ValueError(f"{intent} runs an agent but has no task_kind")


check_taxonomy()
