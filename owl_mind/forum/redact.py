"""Removes private information from a forum draft, and stops GitHub from turning text into links.

This is best-effort. Names in particular can't be found reliably, which is why the student
always previews and approves the exact text before anything is posted.
"""

import re

from owl_mind.services.privacy import Redacted, redact  # noqa: F401  Imported from here.

_CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

GRADED_QUESTION_CHARS = 400
CODE_LINES = 10
GRADED_CODE_LINES = 5


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
