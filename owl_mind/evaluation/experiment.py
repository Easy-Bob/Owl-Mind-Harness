"""Runs one M4-D experiment: the eval parts with one setting changed, against the baseline.

Results go to evals/experiments/<date>-<name>.json, apart from evals/results, so the CI check
keeps reading the pinned results. A kept change is pinned afterwards with check_baseline --pin.

Usage, with agent runs on and the course materials imported:
    LIVE_AGENT_RUNS_ENABLED=true uv run python -m owl_mind.evaluation.experiment rewrite
    ... experiment <name> dialogue adversarial      (only these parts)
"""

import asyncio
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

from owl_mind.core.config import get_settings
from owl_mind.evaluation import evals
from owl_mind.evaluation.check_baseline import BASELINE, CHECKS, _value, problems

EXPERIMENTS_DIR = evals.REPO / "evals" / "experiments"

# Each experiment: the settings it changes, and the parts it's judged on by default.
EXPERIMENTS: dict[str, tuple[dict[str, Any], list[str]]] = {
    "control": ({}, ["dialogue", "adversarial", "quiz_repeat"]),
    "rewrite": ({"composite_answer": "rewrite"}, ["dialogue", "adversarial"]),
    "forced_delegation": ({"force_delegation": True}, ["dialogue", "adversarial"]),
    "direct": ({"direct_single_tutor": True}, ["dialogue", "adversarial"]),
    "quiz_reuse": ({"quiz_reuse": True}, ["quiz_repeat", "dialogue"]),
    "large_pro": ({"large_model": "deepseek-v4-pro"}, ["dialogue", "adversarial"]),
}


def path_for(name: str) -> Path:
    return EXPERIMENTS_DIR / f"{date.today().isoformat()}-{name}.json"


def compare(results: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """One line per score: baseline, experiment, and the change."""
    lines = []
    for part, score, direction, margin in CHECKS:
        old, new = _value(baseline, part, score), _value(results, part, score)
        if part in results and old is not None and new is not None:
            better = "higher" if direction == "up" else "lower"
            lines.append(
                f"{part}.{score}: {old} -> {new} ({new - old:+.4g}, {better} is better, "
                f"margin {margin})"
            )
    return lines


def _settings_of(name: str) -> tuple[dict[str, Any], list[str]]:
    """A repeat is named <experiment>-r<n>, and saved apart from the first run."""
    return EXPERIMENTS[re.sub(r"-r\d+$", "", name)]


async def run(name: str, parts: list[str]) -> dict[str, Any]:
    settings = get_settings()
    changes, _ = _settings_of(name)
    before = {key: getattr(settings, key) for key in changes}
    for key, value in changes.items():
        setattr(settings, key, value)  # For this process only.
    try:
        report = await evals.run(parts, path_for(name))
    finally:
        for key, value in before.items():
            setattr(settings, key, value)
    report["experiment"] = {"name": name, "settings": changes}
    path_for(name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    if not get_settings().live_agent_runs_enabled:
        sys.exit("LIVE_AGENT_RUNS_ENABLED is off, so there's nothing to evaluate.")
    name = sys.argv[1]
    parts = sys.argv[2:] or _settings_of(name)[1]
    report = asyncio.run(run(name, parts))
    baseline = json.loads(BASELINE.read_text())
    for line in compare(report, baseline):
        print(line)
    # The prompt version may differ on purpose; only score drops matter here.
    found = [p for p in problems(report, baseline) if "prompt version" not in p]
    print("\n".join(found) if found else "No score dropped past its margin.")


if __name__ == "__main__":
    main()
