"""Project rules, checked on the syntax tree so that comments and prose do not trip them."""

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PRODUCT_DIRS = ("owl_mind", "migrations")
SKIPPED_DIRS = {".git", ".venv", "__pycache__", "build", "dist", "node_modules"}

MODEL_MODULE = "owl_mind/core/model.py"
CONFIG_MODULE = "owl_mind/core/config.py"

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
        case ast.keyword(arg=name) | ast.Name(id=name) | ast.Attribute(attr=name) | ast.arg(
            arg=name
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


ALL_CHECKS = (
    check_model_construction,
    check_sampling_params,
    check_env_access,
    check_no_top_level_mcp,
    check_ascii_source,
)
