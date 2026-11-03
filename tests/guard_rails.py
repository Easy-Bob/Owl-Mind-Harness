"""Project rules, checked on the syntax tree so that comments and prose do not trip them."""

import ast
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from owl_mind.core.telemetry import TokenAccounting

PRODUCT_DIRS = ("owl_mind", "migrations")
SKIPPED_DIRS = {".git", ".venv", "__pycache__", "build", "dist", "node_modules"}

MODEL_MODULE = "owl_mind/core/model.py"
CONFIG_MODULE = "owl_mind/core/config.py"
FACTORY_MODULE = "owl_mind/core/factory.py"
FORUM_PACKAGE = "owl_mind.forum"
# The fixed handoff path. No agent, tool or subagent module may reach the forum.
FORUM_USERS = {
    "owl_mind/api/app.py",
    "owl_mind/api/handoffs.py",
    "owl_mind/api/forum_account.py",
    "owl_mind/core/runtime.py",
}

PROVIDER_MODULES = {
    "anthropic",
    "langchain.chat_models",
    "langchain_anthropic",
    "langchain_deepseek",
    "langchain_google_genai",
    "langchain_openai",
    "openai",
}
SAMPLING_PARAMS = {"temperature", "top_p", "top_k"}
ENV_READERS = {"environ", "getenv"}


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


def _python_files(root: Path, dirs: tuple[str, ...] | None = None) -> Iterator[Path]:
    bases = [root] if dirs is None else [root / name for name in dirs]
    for base in bases:
        for path in sorted(base.rglob("*.py")):
            if not SKIPPED_DIRS.intersection(path.relative_to(root).parts):
                yield path


def _product_trees(root: Path, rule: str) -> Iterator[tuple[str, ast.AST | None, Violation | None]]:
    """Yield each product file's tree. A file that cannot be parsed is a violation, not a skip."""
    for path in _python_files(root, PRODUCT_DIRS):
        relative = path.relative_to(root).as_posix()
        try:
            yield relative, ast.parse(path.read_bytes()), None
        except (SyntaxError, ValueError) as exc:
            line = getattr(exc, "lineno", None) or 1
            yield relative, None, Violation(rule, relative, line, "file cannot be parsed")


def _is_provider_module(module: str) -> bool:
    return any(module == name or module.startswith(f"{name}.") for name in PROVIDER_MODULES)


def _model_construction(node: ast.AST) -> str | None:
    match node:
        case ast.Import(names=names):
            for alias in names:
                if _is_provider_module(alias.name):
                    return f"imports {alias.name}"
        case ast.ImportFrom(module=module, names=names):
            if module and _is_provider_module(module):
                return f"imports {module}"
            if any(alias.name == "init_chat_model" for alias in names):
                return "imports init_chat_model"
        case ast.Name(id="init_chat_model") | ast.Attribute(attr="init_chat_model"):
            return "uses init_chat_model"
    return None


def check_model_construction(root: Path) -> list[Violation]:
    """Chat models are built only in core/model.py."""
    rule = "model-door"
    found = []
    for path, tree, error in _product_trees(root, rule):
        if error:
            found.append(error)
        elif path != MODEL_MODULE:
            for node in ast.walk(tree):
                if message := _model_construction(node):
                    found.append(Violation(rule, path, node.lineno, message))
    return found


def _sampling_param(node: ast.AST) -> str | None:
    match node:
        case (
            ast.keyword(arg=name) | ast.Name(id=name) | ast.Attribute(attr=name) | ast.arg(arg=name)
        ):
            pass
        case ast.Constant(value=str() as name):
            pass
        case _:
            return None
    return name if name in SAMPLING_PARAMS else None


def check_sampling_params(root: Path) -> list[Violation]:
    """temperature, top_p and top_k appear nowhere."""
    rule = "sampling-params"
    found = []
    for path, tree, error in _product_trees(root, rule):
        if error:
            found.append(error)
            continue
        for node in ast.walk(tree):
            if name := _sampling_param(node):
                found.append(Violation(rule, path, node.lineno, f"uses {name}"))
    return found


def _env_read(node: ast.AST) -> str | None:
    match node:
        case ast.Attribute(attr=name) if name in ENV_READERS:
            return f"reads {name}"
        case ast.ImportFrom(module="os", names=names):
            for alias in names:
                if alias.name in ENV_READERS:
                    return f"imports {alias.name}"
    return None


def check_env_access(root: Path) -> list[Violation]:
    """The environment is read only in core/config.py."""
    rule = "env-access"
    found = []
    for path, tree, error in _product_trees(root, rule):
        if error:
            found.append(error)
        elif path != CONFIG_MODULE:
            for node in ast.walk(tree):
                if message := _env_read(node):
                    found.append(Violation(rule, path, node.lineno, message))
    return found


def check_no_top_level_mcp(root: Path) -> list[Violation]:
    """A top-level mcp package would shadow the official MCP SDK."""
    return [
        Violation("no-top-level-mcp", name, 1, "shadows the MCP SDK")
        for name in ("mcp", "mcp.py")
        if (root / name).exists()
    ]


def check_ascii_source(root: Path) -> list[Violation]:
    """Python source files are ASCII only."""
    found = []
    for path in _python_files(root):
        for number, line in enumerate(path.read_bytes().splitlines(), start=1):
            if not line.isascii():
                relative = path.relative_to(root).as_posix()
                found.append(Violation("ascii-source", relative, number, "non-ASCII byte"))
                break
    return found


def check_agent_construction(root: Path) -> list[Violation]:
    """Agents are built only in core/factory.py."""
    rule = "agent-factory"
    found = []
    for path, tree, error in _product_trees(root, rule):
        if error:
            found.append(error)
        elif path != FACTORY_MODULE:
            for node in ast.walk(tree):
                match node:
                    case ast.Name(id="create_deep_agent") | ast.Attribute(attr="create_deep_agent"):
                        found.append(Violation(rule, path, node.lineno, "uses create_deep_agent"))
                    case ast.ImportFrom(names=names) if any(
                        alias.name == "create_deep_agent" for alias in names
                    ):
                        found.append(Violation(rule, path, node.lineno, "imports it"))
    return found


def check_forum_access(root: Path) -> list[Violation]:
    """Only the fixed handoff path imports the forum client, so an agent can never post."""
    rule = "forum-access"
    found = []
    for path, tree, error in _product_trees(root, rule):
        if error:
            found.append(error)
        elif path not in FORUM_USERS and not path.startswith("owl_mind/forum/"):
            for node in ast.walk(tree):
                match node:
                    case ast.ImportFrom(module=module) if module and (
                        module == FORUM_PACKAGE or module.startswith(FORUM_PACKAGE + ".")
                    ):
                        found.append(Violation(rule, path, node.lineno, "imports the forum"))
                    case ast.Import(names=names) if any(
                        alias.name.startswith(FORUM_PACKAGE) for alias in names
                    ):
                        found.append(Violation(rule, path, node.lineno, "imports the forum"))
    return found


FORUM_MODEL_MODULE = "owl_mind/forum/improve.py"
MODEL_NAMES = {"get_model", "get_chat_model", "ModelTier", "ainvoke", "invoke"}


def check_handoff_model_calls(root: Path) -> list[Violation]:
    """On the handoff path, only the clicked "Improve with AI" rewrite may call a model."""
    rule = "handoff-model-call"
    found = []
    for path, tree, error in _product_trees(root, rule):
        if error:
            found.append(error)
        elif path.startswith("owl_mind/forum/") and path != FORUM_MODEL_MODULE:
            for node in ast.walk(tree):
                match node:
                    case ast.Name(id=name) | ast.Attribute(attr=name) if name in MODEL_NAMES:
                        found.append(Violation(rule, path, node.lineno, f"uses {name}"))
                    case ast.ImportFrom(module=module) if module and module.startswith(
                        "owl_mind.core.model"
                    ):
                        found.append(Violation(rule, path, node.lineno, "imports the model door"))
    return found


REQUIRED_CHECKS = (
    "BudgetMiddleware",
    "ToolErrorMiddleware",
    "ResultCapMiddleware",
    "ToolArgumentMiddleware",
    "ContextMiddleware",
    "BudgetedSummarization",
)


def check_agent_middleware(built: dict) -> list[Violation]:
    """Every agent the factory builds has the M3-A and M3-E checks, and the orchestrator caps
    fan-out. The budget check comes first, then tool errors, which are caught around the rest.

    built is what the factory passed to create_deep_agent.
    """
    found = []
    agents = [("orchestrator", built.get("middleware", []), True)] + [
        (spec["name"], spec.get("middleware", []), False) for spec in built.get("subagents", [])
    ]
    for name, middleware, is_orchestrator in agents:
        # Joining answers makes no call and no tool call, so it may sit outside every check.
        present = [
            type(m).__name__ for m in middleware if type(m).__name__ != "JoinAnswerMiddleware"
        ]
        required = [*REQUIRED_CHECKS, *(["FanOutMiddleware"] if is_orchestrator else [])]
        for check in required:
            if check not in present:
                found.append(Violation("agent-middleware", name, 0, f"missing {check}"))
        if present[:2] != ["BudgetMiddleware", "ToolErrorMiddleware"]:
            found.append(
                Violation("agent-middleware", name, 0, "budget and tool errors aren't outermost")
            )
    return found


def check_models_from_door(models: list) -> list[Violation]:
    """Every model given to an agent has token accounting, which only core/model.py attaches."""
    return [
        Violation("model-from-door", "<test run>", 0, f"{type(model).__name__} has no accounting")
        for model in models
        if not any(isinstance(callback, TokenAccounting) for callback in model.callbacks or [])
    ]


def check_tool_scopes(declared: dict, built: dict) -> list[Violation]:
    """Each subagent's declared tool scope equals the tools it is built with."""
    return [
        Violation("tool-scope", str(name), 0, f"declared {declared.get(name)}, built {tools}")
        for name, tools in built.items()
        if list(declared.get(name, ())) != list(tools)
    ]


def check_activity_event_types(types) -> list[Violation]:
    """session_started and message_sent are the only activity event types."""
    extra = set(types) ^ {"session_started", "message_sent"}
    if not extra:
        return []
    return [Violation("activity-types", "<config>", 0, f"unexpected: {sorted(extra)}")]


def _schema_routes(client) -> list[tuple[str, str]]:
    """Every route in the public API schema, with a real-looking id in each path parameter."""
    routes = []
    for path, methods in client.get("/openapi.json").json()["paths"].items():
        concrete = re.sub(r"{[^}]+}", "00000000-0000-0000-0000-000000000000", path)
        routes += [(method.upper(), concrete) for method in methods]
    return routes


def check_admin_routes(student_client) -> list[Violation]:
    """Call every route under /admin as a logged-in student. Each must answer 403."""
    return [
        Violation("admin-only", path, 0, f"{method} answered {status} to a student")
        for method, path in _schema_routes(student_client)
        if path.startswith("/admin")
        and (status := student_client.request(method, path).status_code) != 403
    ]


def check_login_required(anonymous_client, public_paths: set[str]) -> list[Violation]:
    """Call every route with no login. Each must answer 401, unless it is on the public list."""
    return [
        Violation("login-required", path, 0, f"{method} answered {status} with no login")
        for method, path in _schema_routes(anonymous_client)
        if path not in public_paths
        and (status := anonymous_client.request(method, path).status_code) != 401
    ]


def check_no_model_calls(model_requests: list, route: str) -> list[Violation]:
    """A rule-settled message makes no model call at all."""
    if not model_requests:
        return []
    return [Violation("no-model", route, 0, f"made {len(model_requests)} model calls")]


def check_access_logged(rows_before: int, rows_after: int) -> list[Violation]:
    """Opening one user's detail writes exactly one access log row."""
    if rows_after == rows_before + 1:
        return []
    return [Violation("access-log", "<test run>", 0, f"{rows_after - rows_before} rows written")]


def check_every_call_recorded(model_calls: int, recorded: int) -> list[Violation]:
    """Every model call in a test run produces exactly one telemetry record."""
    if model_calls == recorded:
        return []
    message = f"{model_calls} model calls but {recorded} records"
    return [Violation("every-call-recorded", "<test run>", 0, message)]


def check_telemetry_failure_is_isolated(model_call: Callable[[], object]) -> list[Violation]:
    """Run a model call while telemetry is failing. The call itself must still succeed."""
    try:
        model_call()
    except Exception as exc:
        message = f"call failed with {type(exc).__name__}"
        return [Violation("telemetry-isolated", "<test run>", 0, message)]
    return []


ALL_CHECKS = (
    check_model_construction,
    check_sampling_params,
    check_env_access,
    check_no_top_level_mcp,
    check_ascii_source,
    check_agent_construction,
    check_forum_access,
    check_handoff_model_calls,
)
