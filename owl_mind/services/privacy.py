"""Removing private details from text that leaves a student's own chat: forum posts and memory.

This is best-effort. Names in particular can't be found reliably, so anything shown to others
is previewed and approved, and memory never quotes the student.
"""

import re
from collections import Counter
from dataclasses import dataclass

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


def known_values(email: str) -> list[str]:
    """The student's own email and the name parts of it are always removed, wherever they appear."""
    local = email.split("@", 1)[0]
    parts = re.split(r"[._+-]", local)
    return [email, local, *(part for part in parts if len(part) >= 3)]
