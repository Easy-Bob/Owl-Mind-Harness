"""Text extraction for course files. Files are untrusted, so parsing runs in its own process."""

import io
import logging
import multiprocessing
import queue
import re
import time
from dataclasses import dataclass
from pathlib import PurePath

SUPPORTED = {".pdf", ".pptx", ".docx", ".md", ".tex", ".latex"}
LEGACY = {".ppt": ".pptx", ".doc": ".docx"}
_MEMORY_LIMIT_BYTES = 2 * 1024**3


class ExtractionError(Exception):
    """The file can't be imported. The message is safe to show the uploader."""


@dataclass(frozen=True)
class Section:
    location: str
    heading: str
    text: str


def file_format(filename: str) -> str:
    suffix = PurePath(filename).suffix.lower()
    if suffix in LEGACY:
        raise ExtractionError(f"{suffix} files aren't supported. Save it as {LEGACY[suffix]}.")
    if suffix not in SUPPORTED:
        raise ExtractionError(f"Unsupported file type: {suffix or 'none'}")
    return suffix


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")[:200]


def _pdf(data: bytes) -> list[Section]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ExtractionError("The PDF is password-protected.")
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except PdfReadError as exc:
        raise ExtractionError("The PDF couldn't be read.") from exc
    return [
        Section(f"page {number}", _first_line(text), text)
        for number, text in enumerate(pages, start=1)
        if text
    ]


def _pptx(data: bytes) -> list[Section]:
    from pptx import Presentation

    sections = []
    for number, slide in enumerate(Presentation(io.BytesIO(data)).slides, start=1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if shape.has_table:
                parts += [" | ".join(cell.text for cell in row.cells) for row in shape.table.rows]
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"Notes: {notes}")
        text = "\n".join(part.strip() for part in parts if part.strip())
        if text:
            title = slide.shapes.title.text if slide.shapes.title is not None else ""
            sections.append(Section(f"slide {number}", title.strip() or _first_line(text), text))
    return sections


def _docx(data: bytes) -> list[Section]:
    from docx import Document

    document = Document(io.BytesIO(data))
    sections: list[Section] = []
    heading, lines = "", []

    def close() -> None:
        text = "\n".join(lines).strip()
        if text:
            sections.append(Section(f"section {len(sections) + 1}", heading, text))

    for paragraph in document.paragraphs:
        if paragraph.style.name.lower().startswith("heading") and paragraph.text.strip():
            close()
            heading, lines = paragraph.text.strip(), []
        elif paragraph.text.strip():
            lines.append(paragraph.text)
    for table in document.tables:
        lines += [" | ".join(cell.text for cell in row.cells) for row in table.rows]
    close()
    return sections


def _split_on_headings(text: str, pattern: re.Pattern[str]) -> list[Section]:
    sections: list[Section] = []
    heading, lines = "", []
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            if "\n".join(lines).strip():
                sections.append(
                    Section(f"section {len(sections) + 1}", heading, "\n".join(lines).strip())
                )
            heading, lines = match.group("title").strip(), []
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append(Section(f"section {len(sections) + 1}", heading, "\n".join(lines).strip()))
    return sections


_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+)$")
_LATEX_HEADING = re.compile(r"^\s*\\(?:sub)*section\*?\{(?P<title>[^}]*)\}")
_LATEX_COMMENT = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?")


def _markdown(data: bytes) -> list[Section]:
    return _split_on_headings(data.decode("utf-8", errors="replace"), _MARKDOWN_HEADING)


def _latex(data: bytes) -> list[Section]:
    text = _LATEX_COMMENT.sub("", data.decode("utf-8", errors="replace"))
    body = text.split("\\begin{document}", 1)[-1].split("\\end{document}", 1)[0]
    sections = _split_on_headings(body, _LATEX_HEADING)
    # Commands are dropped and their arguments kept, which leaves readable text and the math.
    return [
        Section(
            s.location, s.heading, _LATEX_COMMAND.sub("", s.text).replace("{", "").replace("}", "")
        )
        for s in sections
    ]


_EXTRACTORS = {
    ".pdf": _pdf,
    ".pptx": _pptx,
    ".docx": _docx,
    ".md": _markdown,
    ".tex": _latex,
    ".latex": _latex,
}


# Control characters such as NUL appear in real PDFs, and Postgres can't store NUL in text.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# Scribe notes name their student authors, with NetIDs, after a "Scribe By:" label. Names must
# not reach the model or appear in citations. The names often continue onto the next lines, so
# everything from the label to the last NetID shortly after it is removed.
_SCRIBE_LABEL = re.compile(r"\bscribe[ds]?\s*(?:by)?\s*:", re.IGNORECASE)
_NETID = re.compile(r"\(\s*[a-z]{2,4}[0-9]{1,3}\s*\)")
_CREDIT_REACH = 300


def _remove_scribe_credits(text: str) -> str:
    while label := _SCRIBE_LABEL.search(text):
        window = text[label.end() : label.end() + _CREDIT_REACH]
        netids = list(_NETID.finditer(window))
        if netids:
            end = label.end() + netids[-1].end()
        else:
            # With no NetID to mark the end, remove the label's own line, including its break.
            newline = text.find("\n", label.end())
            end = len(text) if newline == -1 else newline + 1
        text = text[: label.start()] + text[end:]
    # A NetID left on its own still identifies a student.
    return _NETID.sub("", text)


def _clean_text(text: str) -> str:
    return _remove_scribe_credits(_CONTROL.sub("", text))


def _clean(section: Section) -> Section:
    return Section(section.location, _clean_text(section.heading), _clean_text(section.text))


# Lecture notes put the course, lecturer and student authors in a header before section "1".
# Names in that header often have no label or NetID, so the header is dropped as a whole.
_HEADER_MARKER = re.compile(r"\b(lecturer|instructors?|scribe[ds]?)\b", re.IGNORECASE)
_FIRST_SECTION = re.compile(r"^[ \t]*1(?:\.1)?(?:[ \t]+\S|[ \t]*$)", re.MULTILINE)
_HEADER_REACH = 600


def _drop_notes_header(sections: list[Section]) -> list[Section]:
    if not sections:
        return sections
    first = sections[0]
    marker = _HEADER_MARKER.search(first.text[:_HEADER_REACH])
    start = _FIRST_SECTION.search(first.text, marker.end()) if marker else None
    if start is None or start.start() > _HEADER_REACH:
        return sections
    body = first.text[start.start() :].lstrip()
    return [Section(first.location, _first_line(body), body), *sections[1:]]


def extract(data: bytes, file_type: str) -> list[Section]:
    try:
        sections = _drop_notes_header([_clean(s) for s in _EXTRACTORS[file_type](data)])
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError("The file couldn't be read. It may be damaged.") from exc
    if not any(section.text.strip() for section in sections):
        raise ExtractionError("No text was found. A scanned document needs text, not images.")
    return sections


def _child(data: bytes, file_type: str, results) -> None:
    # Damaged files make the PDF library log hundreds of repair warnings.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (_MEMORY_LIMIT_BYTES, _MEMORY_LIMIT_BYTES))
    except ImportError, ValueError, OSError:
        pass  # Not every system supports a memory limit. The time limit still applies.
    try:
        results.put(("ok", extract(data, file_type)))
    except ExtractionError as exc:
        results.put(("refused", str(exc)))
    except MemoryError:
        results.put(("refused", "The file needs too much memory to read."))


def extract_isolated(
    data: bytes, file_type: str, timeout_seconds: float, target=_child
) -> list[Section]:
    """Parse in a separate process, so a hostile or broken file can't hang or crash the app."""
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(target=target, args=(data, file_type, results), daemon=True)
    process.start()
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                status, value = results.get(timeout=0.2)
                break
            except queue.Empty:
                if not process.is_alive() and results.empty():
                    raise ExtractionError("Reading the file failed.") from None
                if time.monotonic() > deadline:
                    raise ExtractionError("Reading the file took too long.") from None
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
    if status == "refused":
        raise ExtractionError(value)
    return value
