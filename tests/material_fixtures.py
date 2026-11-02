"""Small original course files, built in code, so tests never need real course material."""

import io
import os
import time

from docx import Document
from pptx import Presentation
from pptx.util import Inches


def pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF with one line of text per page."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + index * 2} 0 R" for index in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    font = 3 + len(pages) * 2
    for index, text in enumerate(pages):
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            f" /Resources << /Font << /F1 {font} 0 R >> >> /Contents {4 + index * 2} 0 R >>"
        )
        objects.append(f"<< /Length {len(content)} >>\nstream\n{content.decode()}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = io.BytesIO(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n{body}\nendobj\n".encode())
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    out.write("".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    )
    return out.getvalue()


def pptx(slides: list[tuple[str, str]], notes: str = "") -> bytes:
    deck = Presentation()
    for title, body in slides:
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    if notes:
        deck.slides[0].notes_slide.notes_text_frame.text = notes
    table_slide = deck.slides.add_slide(deck.slide_layouts[5])
    table_slide.shapes.title.text = "Costs"
    table = table_slide.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    for row, values in enumerate([("Structure", "Query"), ("Bloom filter", "O(k)")]):
        for column, value in enumerate(values):
            table.cell(row, column).text = value
    out = io.BytesIO()
    deck.save(out)
    return out.getvalue()


def docx(sections: list[tuple[str, str]]) -> bytes:
    document = Document()
    for heading, body in sections:
        document.add_heading(heading, level=1)
        document.add_paragraph(body)
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


MARKDOWN = b"""# Hashing

A hash function maps keys to slots.

## Collisions

Chaining keeps a list in each slot.
"""

LATEX = rb"""\documentclass{article}
\begin{document}
\section{Tail bounds}
Markov's inequality gives $P(X \geq a) \leq E[X]/a$. % a comment that must not appear
\subsection{Chernoff}
\textbf{Chernoff bounds} are much tighter for sums of independent variables.
\end{document}
"""


def slow_child(data, file_type, results):
    """A parser that never finishes."""
    time.sleep(60)


def crashing_child(data, file_type, results):
    """A parser that dies without reporting."""
    os._exit(1)
