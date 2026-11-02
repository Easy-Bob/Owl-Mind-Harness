"""Removes private information from a forum draft, and stops GitHub from turning text into links.

This is best-effort. Names in particular can't be found reliably, which is why the student
always previews and approves the exact text before anything is posted.
"""

import re
from collections import Counter
from dataclasses import dataclass

GRADED_QUESTION_CHARS = 400
CODE_LINES = 10
GRADED_CODE_LINES = 5

_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[email removed]"),
    (
        "secret",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{10,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
            r"|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|eyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,})"
        ),
        "[secret removed]",
    ),
    (
        "secret",
        re.compile(r"(?i)\b(password|passwd|pwd|api[_-]?key|token|secret)\b(\s*[:=]\s*)(?!\[)\S+"),
        r"\1\2[secret removed]",
    ),
    ("student id", re.compile(r"\b[Ss]\d{7,9}\b|\b\d{9}\b"), "[id removed]"),
    ("netid", re.compile(r"\(\s*[a-z]{2,4}[0-9]{1,3}\s*\)"), "[id removed]"),
    (
        "phone",
        re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\w)"),
        "[phone removed]",
    ),
    # Only explicit introductions: "I am stuck" mustn't lose a word.
    (
        "name",
        re.compile(r"(?i:\b(my name is|name:))\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"),
        r"\1 [name removed]",
    ),
]

_CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Redacted:
    text: str
    removed: dict[str, int]


def redact(text: str, known_values: list[str] = ()) -> Redacted:
    """Remove private details, plus any exact values known to belong to the student."""
    counts: Counter[str] = Counter()
    for value in known_values:
        if value and len(value) >= 3:
            pattern = re.compile(re.escape(value), re.IGNORECASE)
            text, found = pattern.subn("[removed]", text)
            counts["known detail"] += found
    for kind, pattern, replacement in _RULES:
        text, found = pattern.subn(replacement, text)
        counts[kind] += found
    return Redacted(text, {kind: n for kind, n in counts.items() if n})


def trim_code(text: str, max_lines: int) -> str:
    """Keep only the start of each code block, so a post never carries a whole solution."""

    def shorten(match: re.Match[str]) -> str:
        lines = match.group(1).splitlines()
        if len(lines) <= max_lines:
            return match.group(0)
        kept = "\n".join(lines[:max_lines])
        return f"```\n{kept}\n... ({len(lines) - max_lines} more lines not shown)\n```"

    return _CODE_BLOCK.sub(shorten, text)


def for_graded_problem(text: str) -> str:
    """A graded problem's statement and code mustn't be posted, so keep only a short excerpt."""
    text = trim_code(text, GRADED_CODE_LINES)
    if len(text) > GRADED_QUESTION_CHARS:
        text = text[:GRADED_QUESTION_CHARS].rsplit(" ", 1)[0] + " ... (shortened: graded problem)"
    return text


def escape_markdown(text: str) -> str:
    """Stop GitHub from making mentions, issue links, images or HTML out of student text."""
    zero_width = "\u200b"
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"@(?=\w)", "@" + zero_width, text)
    text = re.sub(r"#(?=\d)", "#" + zero_width, text)
    return text.replace("![", "!" + zero_width + "[")
