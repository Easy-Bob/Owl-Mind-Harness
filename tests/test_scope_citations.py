"""Offline: answers stay on what was asked, non-CS messages are refused, citations resolve."""

import pytest

from owl_mind.core import prompts, replies
from owl_mind.core.intent.classifier import SYSTEM_PROMPT
from owl_mind.core.intent.taxonomy import INTENTS, Intent, Reply
from owl_mind.materials.search import Citation
from tests.test_runtime import World, intent, start, stream, text_of
from tests.test_runtime import settings as runtime_settings  # noqa: F401  Applied to all.

PROMPTS = {
    "orchestrator": prompts.ORCHESTRATOR,
    "concept_tutor": prompts.CONCEPT_TUTOR,
    "practice_coach": prompts.PRACTICE_COACH,
    "study_planner": prompts.STUDY_PLANNER,
    "quiz_master": prompts.QUIZ_MASTER,
}


@pytest.mark.parametrize("name", PROMPTS)
def test_every_agent_prompt_has_the_shared_scope_rule(name):
    assert prompts.SCOPE_RULE in PROMPTS[name]


def test_non_cs_messages_are_classified_as_other_and_get_the_off_topic_reply():
    assert INTENTS[Intent.OTHER].reply is Reply.OFF_TOPIC
    assert "arithmetic" in INTENTS[Intent.OTHER].description
    assert "Anything else, however simple, is other." in SYSTEM_PROMPT


def test_an_other_message_gets_the_fixed_reply_and_no_agent():
    world = World(intent(other=0.9))
    with world.client() as client:
        events = stream(client, start(client, "tell me how to calculate 1 + 1")["run_id"])

    assert text_of(events) == replies.TEXT[Reply.OFF_TOPIC] + "\n\n" + replies.help_text([])
    assert len(world.fake.requests) == 1  # The intent call only.


# -- citations ---------------------------------------------------------------------


class Resolver:
    def __init__(self, known: dict[str, Citation]) -> None:
        self.known = known
        self.asked: list[list[str]] = []

    async def resolve(self, ids):
        self.asked.append(list(ids))
        return {i: self.known.get(i) for i in ids}


def citation_world(known=None) -> tuple[World, Resolver]:
    world = World()
    resolver = Resolver(known or {})
    world.services = world.services.__class__(**{**vars(world.services), "search": resolver})
    return world, resolver


def test_citations_resolve_to_title_location_and_url():
    known = {"a1": Citation("Lecture 6", "slide 3", "https://example.edu/l6")}
    world, resolver = citation_world(known)
    with world.client() as client:
        response = client.get("/citations", params={"ids": "a1,zz,a1"})

    assert response.json() == {
        "a1": {"title": "Lecture 6", "location": "slide 3", "url": "https://example.edu/l6"},
        "zz": None,
    }
    assert resolver.asked == [["a1", "zz"]]  # Asked once each, in order.


def test_too_many_citations_are_refused():
    world, _ = citation_world()
    with world.client() as client:
        response = client.get("/citations", params={"ids": ",".join(f"c{n}" for n in range(21))})

    assert response.status_code == 422


def test_citations_need_a_login():
    world, _ = citation_world()
    world.app.dependency_overrides.clear()
    with world.client() as client:
        assert client.get("/citations", params={"ids": "a1"}).status_code == 401


def test_without_course_search_every_citation_is_unknown():
    world = World()
    with world.client() as client:
        assert client.get("/citations", params={"ids": "a1"}).json() == {"a1": None}


# -- what a student can ask, with topics from the course notes --------------------------


class Topics:
    def __init__(self, topics: list[str]) -> None:
        self.topics = topics

    async def list_topics(self, limit: int = 40) -> list[str]:
        return self.topics

    async def fail_unfinished_jobs(self) -> int:
        return 0


class Ingestor:
    def __init__(self, topics: list[str]) -> None:
        self.store = Topics(topics)

    async def shutdown(self) -> None:
        pass


def topics_world(*replies_: dict, topics=("Bloom filters", "Chernoff bounds")) -> World:
    world = World(*replies_)
    world.services = world.services.__class__(
        **{**vars(world.services), "ingestor": Ingestor(list(topics))}
    )
    return world


def test_the_help_page_lists_what_to_ask_and_the_course_topics_with_no_model_call():
    world = topics_world()
    with world.client() as client:
        page = client.get("/help").json()

    assert page["topics"] == ["Bloom filters", "Chernoff bounds"]
    assert {"what": "Quiz you", "example": "give me a short quiz on sorting"} in page[
        "can_help_with"
    ]
    assert world.fake.requests == []


@pytest.mark.parametrize("reply", ["other", "unclear"])
def test_the_off_topic_and_clarifying_replies_end_with_the_help_and_topics(reply):
    world = topics_world(intent(other=0.9) if reply == "other" else intent(concept_explain=0.3))
    with world.client() as client:
        events = stream(client, start(client, "what is the capital of France")["run_id"])

    shown = text_of(events)
    assert "Topics in your course notes: Bloom filters, Chernoff bounds." in shown
    assert '- Quiz you: "give me a short quiz on sorting"' in shown
    assert len(world.fake.requests) == 1


def test_a_lookup_failure_only_drops_the_topics():
    class Broken(Topics):
        async def list_topics(self, limit: int = 40):
            raise ConnectionError("down")

    world = topics_world(intent(other=0.9))
    world.services.ingestor.store = Broken([])
    with world.client() as client:
        events = stream(client, start(client, "recommend a movie")["run_id"])

    assert "Topics in your course notes" not in text_of(events)
    assert "I can help you study computer science" in text_of(events)
