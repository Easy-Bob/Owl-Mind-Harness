"""Splits extracted sections into overlapping chunks that remember where they came from."""

import re
from dataclasses import dataclass

from owl_mind.materials.extract import Section

TARGET_CHARS = 1200
OVERLAP_CHARS = 200

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_SOLUTION_NAME = re.compile(
    r"(?<![a-z])(solutions?|solns?|answers?[\s_-]*keys?|answers)(?![a-z])", re.IGNORECASE
)
_SOLUTION_HEADING = re.compile(r"^\s*(solutions?|answer\s*key|answers)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    position: int
    location: str
    heading: str
    text: str


def looks_like_solution(filename: str, sections: list[Section]) -> bool:
    """A first check only. The uploader's declaration is the real safeguard."""
    if _SOLUTION_NAME.search(filename):
        return True
    opening = "\n".join(s.heading + "\n" + s.text for s in sections[:1])[:300]
    return any(_SOLUTION_HEADING.match(line) for line in opening.splitlines()[:5])


def _pieces(text: str) -> list[str]:
    """Paragraphs, with long ones split into sentences and very long sentences cut."""
    pieces = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if len(paragraph) <= TARGET_CHARS:
            pieces.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            pieces += [
                sentence[i : i + TARGET_CHARS] for i in range(0, len(sentence), TARGET_CHARS)
            ]
    return [piece for piece in pieces if piece]


def _overlap(text: str) -> str:
    tail = text[-OVERLAP_CHARS:]
    # Start the overlap at a word, so a chunk never begins mid-word.
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


def chunk_sections(sections: list[Section]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        current = ""
        for piece in _pieces(section.text):
            if current and len(current) + len(piece) + 1 > TARGET_CHARS:
                chunks.append(Chunk(len(chunks), section.location, section.heading, current))
                current = _overlap(current)
            current = f"{current}\n{piece}".strip() if current else piece
        if current:
            chunks.append(Chunk(len(chunks), section.location, section.heading, current))
    return chunks
