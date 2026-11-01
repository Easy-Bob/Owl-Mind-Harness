"""Each rule must hold on this repo and must fail on a deliberate violation."""

from pathlib import Path

import pytest

from tests.guard_rails import (
    ALL_CHECKS,
    check_ascii_source,
    check_env_access,
    check_model_construction,
    check_no_top_level_mcp,
    check_sampling_params,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def write(root: Path, relative: str, source: bytes | str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = source.encode() if isinstance(source, str) else source
    path.write_bytes(data)


@pytest.mark.parametrize("check", ALL_CHECKS, ids=lambda check: check.__name__)
def test_repo_follows_the_rule(check):
    assert [str(violation) for violation in check(REPO_ROOT)] == []


@pytest.mark.parametrize(
    ("check", "relative", "source"),
    [
        (
            check_model_construction,
            "owl_mind/a.py",
            "from langchain.chat_models import init_chat_model\n",
        ),
        (check_model_construction, "owl_mind/a.py", "from langchain_openai import ChatOpenAI\n"),
        (check_model_construction, "owl_mind/a.py", "import langchain_openai.chat_models\n"),
        (check_model_construction, "owl_mind/a.py", "import openai\n"),
        (check_model_construction, "owl_mind/a.py", "tools.init_chat_model('x')\n"),
        (check_model_construction, "migrations/a.py", "import anthropic\n"),
        (check_sampling_params, "owl_mind/a.py", "build(temperature=0.2)\n"),
        (check_sampling_params, "owl_mind/a.py", "settings = {'top_p': 0.9}\n"),
        (check_sampling_params, "owl_mind/a.py", "def build(top_k): ...\n"),
        (check_sampling_params, "owl_mind/a.py", "model.temperature\n"),
        (check_sampling_params, "owl_mind/core/model.py", "build(temperature=0.2)\n"),
        (check_env_access, "owl_mind/a.py", "import os\nos.environ['KEY']\n"),
        (check_env_access, "owl_mind/a.py", "import os\nos.getenv('KEY')\n"),
        (check_env_access, "owl_mind/a.py", "import os as o\no.environ['KEY']\n"),
        (check_env_access, "owl_mind/a.py", "from os import environ\n"),
        (check_no_top_level_mcp, "mcp/__init__.py", ""),
        (check_no_top_level_mcp, "mcp.py", ""),
        (check_ascii_source, "owl_mind/a.py", "name = 'caf\u00e9'\n"),
        (check_ascii_source, "tests/a.py", "# caf\u00e9\n"),
        (check_model_construction, "owl_mind/a.py", "def broken(:\n"),
        (check_sampling_params, "owl_mind/a.py", "def broken(:\n"),
        (check_env_access, "owl_mind/a.py", "def broken(:\n"),
    ],
)
def test_rule_reports_a_deliberate_violation(tmp_path, check, relative, source):
    write(tmp_path, relative, source)

    violations = check(tmp_path)

    assert len(violations) == 1
    assert relative.startswith(violations[0].path)


@pytest.mark.parametrize(
    ("check", "relative", "source"),
    [
        (
            check_model_construction,
            "owl_mind/core/model.py",
            "from langchain.chat_models import init_chat_model\n",
        ),
        (
            check_model_construction,
            "owl_mind/a.py",
            "from langchain_core.messages import ChatMessage\n",
        ),
        (check_env_access, "owl_mind/core/config.py", "import os\nos.environ['KEY']\n"),
        (check_env_access, "owl_mind/a.py", "environ = {}\n"),
        (check_sampling_params, "owl_mind/a.py", "# temperature is never set\n"),
        (check_no_top_level_mcp, "owl_mind/mcp_servers/__init__.py", ""),
        (check_ascii_source, ".venv/lib/a.py", "# caf\u00e9\n"),
    ],
)
def test_rule_allows_legal_code(tmp_path, check, relative, source):
    write(tmp_path, relative, source)

    assert check(tmp_path) == []
