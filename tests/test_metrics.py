"""Offline: the M4-E metrics. Runs, the queue and spending are counted, labels hold no student
data, and every dashboard query names a metric that exists."""

import asyncio
import json
import re
import time
from pathlib import Path

from prometheus_client import REGISTRY

from owl_mind.core import metrics, prompts
from owl_mind.core.integrity import REVIEWER_PROMPT
from owl_mind.evaluation.fake_model import reply_for
from owl_mind.worker.runs import handle
from tests.test_filesystem import say, task
from tests.test_runtime import World, intent, start, stream
from tests.test_runtime import settings as runtime_settings  # noqa: F401  Applied to all.
from tests.test_workers import FakeQueue, message, queued_run, queued_world

DASHBOARDS = Path(__file__).resolve().parents[1] / "ops" / "grafana" / "dashboards"
# Label names a metric may have. None can hold an email, an ID or a student's text.
SAFE_LABELS = {
    "route", "task_kind", "status", "reason", "why", "step", "job", "result", "layer",
    "component", "model", "outcome", "kind",
}  # fmt: skip


def value(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_a_finished_run_is_counted_with_its_route_kind_and_status():
    labels = {"route": "orchestrated", "task_kind": "learn", "status": "completed", "reason": ""}
    before = value("owl_runs_finished_total", **labels)
    seconds = value("owl_run_seconds_count", route="orchestrated", task_kind="learn")
    world = World(intent(concept_explain=0.9), task("concept_tutor"), say("r"), say("Answer."))
    with world.client() as client:
        stream(client, start(client, "explain heaps")["run_id"])

    assert value("owl_runs_finished_total", **labels) == before + 1
    assert value("owl_run_seconds_count", route="orchestrated", task_kind="learn") == seconds + 1
    assert value("owl_runs_running") == 0


def test_a_stopped_run_is_counted_with_its_reason():
    labels = {"route": "template", "task_kind": "none", "status": "completed", "reason": ""}
    before = value("owl_runs_finished_total", **labels)
    world = World()
    with world.client() as client:
        stream(client, start(client, "hello")["run_id"])

    assert value("owl_runs_finished_total", **labels) == before + 1


def test_integrity_outcomes_and_budget_steps_are_counted_from_run_events():
    before = value("owl_integrity_total", result="blocked", layer="code")
    steps = value("owl_budget_steps_total", step="wind_down")

    metrics.observe_event("integrity", {"result": "blocked", "layer": "code", "reason": "x"})
    metrics.observe_event("budget_step", {"step": "wind_down", "percent": 71})
    metrics.observe_event("text_delta", {"text": "never a label"})

    assert value("owl_integrity_total", result="blocked", layer="code") == before + 1
    assert value("owl_budget_steps_total", step="wind_down") == steps + 1


def test_a_worker_measures_how_long_a_run_waited_in_the_queue():
    world, _ = queued_world()
    request = queued_run(world)
    request = request.__class__(**{**vars(request), "queued_at": time.time() - 3})
    count = value("owl_queue_wait_seconds_count")
    total = value("owl_queue_wait_seconds_sum")

    asyncio.run(handle(world.services, FakeQueue(), message(request)))

    assert value("owl_queue_wait_seconds_count") == count + 1
    assert value("owl_queue_wait_seconds_sum") - total >= 3


def test_a_dead_letter_is_counted():
    world, _ = queued_world()
    request = queued_run(world)
    before = value("owl_dead_letters_total", why="deliveries")

    asyncio.run(handle(world.services, FakeQueue(deliveries=4), message(request)))

    assert value("owl_dead_letters_total", why="deliveries") == before + 1


def test_no_metric_label_can_hold_student_data():
    for collector in REGISTRY.collect():
        for sample in collector.samples:
            if sample.name.startswith(("owl_", "llm_")):
                extra = set(sample.labels) - SAFE_LABELS - {"le"}
                assert not extra, (sample.name, extra)


def test_every_dashboard_query_names_a_metric_that_exists():
    known = {m.name for m in REGISTRY.collect()}
    boards = sorted(DASHBOARDS.glob("*.json"))
    assert [b.stem for b in boards] == ["owl-health", "owl-runs", "owl-spending"]
    for board in boards:
        for panel in json.loads(board.read_text())["panels"]:
            for target in panel["targets"]:
                for name in re.findall(r"\b(?:owl|llm)_[a-z_]+", target["expr"]):
                    base = re.sub(r"_(bucket|sum|count|total)$", "", name)
                    assert base in known or name in known, (board.stem, name)


def test_the_dashboards_are_built_from_their_script():
    from ops.grafana.build_dashboards import DASHBOARDS as BUILT

    for board in BUILT:
        assert json.loads((DASHBOARDS / f"{board['uid']}.json").read_text()) == board


def test_the_fake_model_plays_one_learn_turn():
    def body(system: str, *messages: dict) -> dict:
        return {"messages": [{"role": "system", "content": system}, *messages]}

    intent_reply = json.loads(reply_for(body("Classify a student's message into..."))["content"])
    delegate = reply_for(body(prompts.ORCHESTRATOR, {"role": "user", "content": "hi"}))
    answer = reply_for(body(prompts.ORCHESTRATOR, {"role": "tool", "content": "report"}))

    assert intent_reply["intents"][0]["intent"] == "concept_explain"
    assert delegate["tool_calls"][0]["function"]["name"] == "task"
    assert "[cite:" in answer["content"]
    assert '"hint"' in reply_for(body(REVIEWER_PROMPT))["content"]
