"""Page-text extraction that preserves tables as markdown.

Both pipelines (RAG chunking + PageIndex tree-builder) read page text. By
default pymupdf's get_text() flattens tables into rows of space-separated
cell values, which is fine for prose but destroys the structure an LLM
needs to answer questions like "what was R&D spend in 2024 vs 2023?"

This module wraps pymupdf so each page's text contains an inline markdown
table at the right offset whenever a table is detected. No new
dependencies — pymupdf has built-in table detection.
"""
from __future__ import annotations
from pathlib import Path

import pymupdf


def _table_to_markdown(table) -> str:
    rows = table.extract() or []
    if not rows:
        return ""
    # Use the first row as the header. If cells are empty/None, replace with "—".
    def norm(cell):
        if cell is None:
            return ""
        return str(cell).replace("|", "\\|").replace("\n", " ").strip()

    header = [norm(c) for c in rows[0]]
    body = [[norm(c) for c in r] for r in rows[1:]]
    if not body and len(header) <= 1:
        return ""  # not really a table

    n_cols = max(len(header), max((len(r) for r in body), default=0))
    header = header + [""] * (n_cols - len(header))
    body = [r + [""] * (n_cols - len(r)) for r in body]

    sep = ["---"] * n_cols
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(sep) + " |"]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _extract_page(page) -> str:
    """Return the page's text with detected tables replaced by markdown.

    Strategy: get the raw text, then for each detected table substitute the
    table's bounding-box text region with the markdown rendering. We find
    tables first because table cells appear in the raw text in reading order
    but unstructured.
    """
    # 1) Detect tables (returns a TableFinder).
    try:
        tables = page.find_tables()
        table_list = list(tables) if tables else []
    except Exception:
        table_list = []

    text = page.get_text("text") or ""

    if not table_list:
        return text

    # 2) Build markdown for each table; pull out the text the table occupies
    #    using its bounding-box and remove it from the raw text. Then append
    #    the markdown tables after the prose.
    markdowns: list[str] = []
    consumed: list[str] = []
    for t in table_list:
        md = _table_to_markdown(t)
        if not md:
            continue
        markdowns.append(md)
        # Best-effort: pull the text within the table's bbox so we can strip it
        # from the prose flow (avoids the same data appearing twice).
        try:
            bbox = t.bbox  # (x0, y0, x1, y1)
            consumed.append(page.get_text("text", clip=bbox) or "")
        except Exception:
            pass

    cleaned = text
    for chunk in consumed:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Only strip if it's a clean substring — otherwise leave the prose alone.
        if chunk in cleaned:
            cleaned = cleaned.replace(chunk, "")

    parts = [cleaned.strip()] + [
        f"\n\n[TABLE]\n{md}\n[/TABLE]\n" for md in markdowns
    ]
    return "\n\n".join(p for p in parts if p)


def extract_pages_with_tables(pdf_path: Path | str) -> list[str]:
    """Return [page_text_with_tables_as_markdown, …] for a PDF."""
    doc = pymupdf.open(str(pdf_path))
    out = [_extract_page(p) for p in doc]
    doc.close()
    return out
