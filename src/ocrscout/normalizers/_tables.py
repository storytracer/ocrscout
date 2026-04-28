"""Shared table parsers used by multiple normalizers.

- ``parse_html_table`` — handles HTML ``<table>`` payloads (dots-ocr,
  lighton-ocr2, glm-ocr's Table Recognition mode).
- ``parse_pipe_table`` — handles GitHub-flavoured Markdown pipe tables.
- ``parse_otsl_table`` — handles DocTags / OTSL ``<fcel>...<nl>`` fragments
  emitted by PaddleOCR-VL's Table Recognition mode (and likely future
  DocTags-native models).
- ``parse_table_payload`` — auto-detects the format and dispatches.

All variants return a ``TableData`` ready to feed into
``DoclingDocument.add_table``. Parsing failures fall through to an empty
``TableData`` rather than raising — per-block resilience is the contract.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Any

from docling_core.types.doc.document import TableCell, TableData

log = logging.getLogger(__name__)

_OTSL_MARKERS = ("<fcel", "<ecel", "<lcel", "<ucel", "<xcel", "<otsl", "<ched", "<rhed")


def parse_table_payload(text: str) -> TableData:
    """Pick the right parser by looking at the payload, then call it.

    Detection precedence:

    1. OTSL — any of ``<fcel>``/``<ecel>``/``<lcel>``/``<otsl>`` markers
       (DocTags table grammar; PaddleOCR-VL Table Recognition mode).
    2. HTML — contains a ``<table>`` (or ``<tr>`` / ``<td>``) tag.
    3. Pipe table — fallback.

    Empty input returns an empty ``TableData``.
    """
    stripped = (text or "").strip()
    if not stripped:
        return TableData(table_cells=[], num_rows=0, num_cols=0)
    lower = stripped.lower()
    if any(m in lower for m in _OTSL_MARKERS):
        return parse_otsl_table(stripped)
    if "<table" in lower or "<tr" in lower or "<td" in lower:
        return parse_html_table(stripped)
    return parse_pipe_table(stripped)


def parse_otsl_table(otsl_text: str) -> TableData:
    """Parse a DocTags / OTSL table fragment into a ``TableData``.

    Delegates to ``docling_core.types.doc.document.parse_otsl_table_content``.
    Failures degrade to an empty ``TableData`` so a malformed cell doesn't
    drop the whole document.
    """
    try:
        from docling_core.types.doc.document import parse_otsl_table_content
    except ImportError:  # pragma: no cover
        log.warning("docling-core has no parse_otsl_table_content; returning empty table")
        return TableData(table_cells=[], num_rows=0, num_cols=0)
    try:
        return parse_otsl_table_content(otsl_text)
    except Exception as e:  # noqa: BLE001
        log.warning("OTSL parse failed (%s); returning empty table", e)
        return TableData(table_cells=[], num_rows=0, num_cols=0)


def parse_html_table(html_text: str) -> TableData:
    """Parse a simple HTML ``<table>`` payload into a ``TableData``.

    Handles ``<thead>``/``<tbody>``, ``<th>``/``<td>``, and ``rowspan``/
    ``colspan`` attributes. Inline tags inside cells are stripped to text.
    Cells in ``<thead>`` (or any ``<th>``) are marked ``column_header=True``.
    """
    parser = _HtmlTableParser()
    parser.feed(html_text)
    parser.close()
    return parser.build()


def parse_pipe_table(text: str) -> TableData:
    """Parse a GFM-style pipe table into a ``TableData``.

    The first non-delimiter row is treated as the column header. Rows shorter
    than the widest row are padded with empty cells. Delimiter rows of the
    form ``|---|---|`` (with optional ``:`` for alignment) are skipped.
    """
    lines = [s for s in (ln.strip() for ln in text.splitlines()) if s]
    rows: list[list[str]] = []
    for ln in lines:
        if not _looks_like_pipe_row(ln):
            continue
        cells = _split_pipe_row(ln)
        if _is_delim_cells(cells):
            continue
        rows.append(cells)

    if not rows:
        return TableData(table_cells=[], num_rows=0, num_cols=0)

    num_cols = max(len(r) for r in rows)
    cells: list[TableCell] = []
    for r, row in enumerate(rows):
        for c, txt in enumerate(row):
            cells.append(
                TableCell(
                    text=txt,
                    start_row_offset_idx=r,
                    end_row_offset_idx=r + 1,
                    start_col_offset_idx=c,
                    end_col_offset_idx=c + 1,
                    column_header=(r == 0),
                )
            )
    return TableData(table_cells=cells, num_rows=len(rows), num_cols=num_cols)


def looks_like_pipe_table(lines: list[str], start: int) -> bool:
    """Whether ``lines[start]`` and ``lines[start+1]`` form a pipe-table head.

    Used by line-based parsers to decide whether to enter pipe-table mode.
    Requires a header row plus a delimiter row immediately below.
    """
    if start + 1 >= len(lines):
        return False
    head = lines[start].strip()
    delim = lines[start + 1].strip()
    if not (_looks_like_pipe_row(head) and _looks_like_pipe_row(delim)):
        return False
    return _is_delim_cells(_split_pipe_row(delim))


def is_pipe_row(line: str) -> bool:
    return _looks_like_pipe_row(line.strip())


_DELIM_CELL = re.compile(r":?-{2,}:?")


def _looks_like_pipe_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def _is_delim_cells(cells: list[str]) -> bool:
    return bool(cells) and all(_DELIM_CELL.fullmatch(c) for c in cells if c) and any(
        c for c in cells
    )


def _split_pipe_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


class _HtmlTableParser(HTMLParser):
    """Collect rows of (text, colspan, rowspan, is_header) from a <table>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._rows: list[list[tuple[str, int, int, bool]]] = []
        self._current_row: list[tuple[str, int, int, bool]] | None = None
        self._cell_buf: list[str] | None = None
        self._cell_colspan = 1
        self._cell_rowspan = 1
        self._cell_is_header = False
        self._in_thead = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "thead":
            self._in_thead = True
        elif tag == "tr":
            self._current_row = []
        elif tag in ("td", "th") and self._current_row is not None:
            attr = dict(attrs)
            self._cell_buf = []
            self._cell_colspan = _to_int(attr.get("colspan"), 1)
            self._cell_rowspan = _to_int(attr.get("rowspan"), 1)
            self._cell_is_header = tag == "th" or self._in_thead

    def handle_endtag(self, tag: str) -> None:
        if tag == "thead":
            self._in_thead = False
        elif tag == "tr":
            if self._current_row is not None:
                self._rows.append(self._current_row)
                self._current_row = None
        elif tag in ("td", "th") and self._cell_buf is not None and self._current_row is not None:
            text = " ".join("".join(self._cell_buf).split())
            self._current_row.append(
                (text, self._cell_colspan, self._cell_rowspan, self._cell_is_header)
            )
            self._cell_buf = None

    def handle_data(self, data: str) -> None:
        if self._cell_buf is not None:
            self._cell_buf.append(data)

    def close(self) -> None:
        # VLM outputs frequently emit `<table><tr><td>...</table>` without the
        # matching `</td>`/`</tr>` (especially when a layout-aware backend
        # routes a non-table region through a "Table Recognition:" prompt and
        # the model dumps prose into one giant cell). Without this flush,
        # `build()` only sees rows that hit an explicit `</tr>` and silently
        # drops the entire payload — observed on glm-ocr-layout under the
        # ``layout_chat`` backend.
        super().close()
        if self._cell_buf is not None and self._current_row is not None:
            text = " ".join("".join(self._cell_buf).split())
            self._current_row.append(
                (text, self._cell_colspan, self._cell_rowspan, self._cell_is_header)
            )
            self._cell_buf = None
        if self._current_row:
            self._rows.append(self._current_row)
            self._current_row = None

    def build(self) -> TableData:
        cells: list[TableCell] = []
        occupied: set[tuple[int, int]] = set()
        max_cols = 0
        for r, row in enumerate(self._rows):
            c = 0
            for text, colspan, rowspan, is_header in row:
                while (r, c) in occupied:
                    c += 1
                cells.append(
                    TableCell(
                        text=text,
                        start_row_offset_idx=r,
                        end_row_offset_idx=r + rowspan,
                        start_col_offset_idx=c,
                        end_col_offset_idx=c + colspan,
                        row_span=rowspan,
                        col_span=colspan,
                        column_header=is_header,
                    )
                )
                for rr in range(r, r + rowspan):
                    for cc in range(c, c + colspan):
                        occupied.add((rr, cc))
                c += colspan
                if c > max_cols:
                    max_cols = c
        return TableData(table_cells=cells, num_rows=len(self._rows), num_cols=max_cols)


def _to_int(value: Any, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default
