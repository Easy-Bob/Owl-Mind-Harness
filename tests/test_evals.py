"""Offline: the eval corpus is well-formed, the rules agree with it, and the check works."""

import json

import pytest

from owl_mind.core.intent.rules import match_rule
from owl_mind.core.intent.taxonomy import Intent
from owl_mind.evaluation import check_baseline
from owl_mind.evaluation.corpus import ATTACK, DIALOGUE, FOLLOW_UPS, MEMORY, VALID_REQUESTS
from owl_mind.evaluation.corpus_intent import INTENT


def test_every_case_has_a_unique_id_and_known_labels():
    ids = [c[0] for c in INTENT] + [c[0] for c in FOLLOW_UPS] + [c[0] for c in DIALOGUE]
    ids += [c[0] for c in MEMORY]
    assert len(ids) == len(set(ids))
    known = {i.value for i in Intent}
    assert all(accepted <= known and accepted for _, _, accepted in INTENT)
    assert len(INTENT) >= 60 and len(DIALOGUE) >= 20
    assert len(ATTACK) == 40 and len(VALID_REQUESTS) == 8 and len(MEMORY) == 10


def test_at_least_20_dialogue_cases_are_two_tutor_turns_for_m4_d():
    assert sum(kind.value == "composite" for _, _, kind, _ in DIALOGUE) >= 10
    assert {kind.value for _, _, kind, _ in DIALOGUE} >= {
        "learn", "practice", "plan", "assess", "support", "composite"
    }  # fmt: skip


@pytest.mark.parametrize(("case_id", "message", "accepted"), INTENT, ids=[c[0] for c in INTENT])
def test_the_rules_never_contradict_the_corpus(case_id, message, accepted):
    """The rule-only part of the intent evals, run in CI with no model call."""
    match = match_rule(message)
    if match is not None and match.intent is not None:
        assert match.intent.value in accepted


# -- the baseline check ---------------------------------------------------------------------

BASELINE = {
    "prompt_version": "old",
    "judge_calibrated": False,
    "intent": {"accuracy": 0.99, "follow_up_accuracy": 1.0},
    "dialogue": {"completion": 1.0, "citation_rate": 0.9, "judge": {"correctness": 4.5}},
    "adversarial": {"leak_rate": 0.0, "valid_blocked_rate": 0.0},
    "memory": {"private_kept_rate": 0.0},
}


def results(**changes) -> dict:
    report = json.loads(json.dumps(BASELINE))
    report["prompt_version"] = check_baseline.PROMPT_VERSION
    for path, value in changes.items():
        part, score = path.split("__")
        report[part][score] = value
    return report


def test_results_as_good_as_the_baseline_pass():
    assert check_baseline.problems(results(), BASELINE) == []


def test_a_drop_within_its_margin_passes_and_past_it_fails():
    assert check_baseline.problems(results(intent__accuracy=0.97), BASELINE) == []
    [problem] = check_baseline.problems(results(intent__accuracy=0.95), BASELINE)
    assert "intent.accuracy went from 0.99 to 0.95" in problem


def test_any_leak_fails():
    [problem] = check_baseline.problems(results(adversarial__leak_rate=0.025), BASELINE)
    assert "leak_rate" in problem


def test_results_for_an_older_prompt_version_fail():
    report = results()
    report["prompt_version"] = "something-else"

    [problem] = check_baseline.problems(report, BASELINE)
    assert "No eval results for prompt version" in problem


def test_judge_scores_count_only_once_the_judge_is_calibrated():
    worse = results()
    worse["dialogue"]["judge"] = {"correctness": 2.0}

    assert check_baseline.problems(worse, BASELINE) == []
    calibrated = dict(BASELINE, judge_calibrated=True)
    assert "judge.correctness" in check_baseline.problems(worse, calibrated)[0]


def test_judge_agreement_with_owner_labels(tmp_path):
    sheet = tmp_path / "sheet.json"
    rows = [
        {"judge": {"correctness": 5, "faithfulness": 4, "scope": 5}, "owner": 5},
        {"judge": {"correctness": 2, "faithfulness": 2, "scope": 2}, "owner": 5},
        {"judge": {"correctness": 4}, "owner": None},
    ]
    sheet.write_text(json.dumps(rows))

    assert check_baseline.agreement(sheet) == 0.5
    sheet.write_text(json.dumps(rows[2:]))
    assert check_baseline.agreement(sheet) is None


def test_the_committed_results_pass_against_the_pinned_baseline():
    path, baseline = check_baseline.latest(), check_baseline.BASELINE
    if path is None or not baseline.exists():
        pytest.skip("No results pinned yet")
    report = json.loads(path.read_text())
    assert check_baseline.problems(report, json.loads(baseline.read_text())) == []
