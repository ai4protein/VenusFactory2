"""Bounded UTF-8 previews through the session's anchored file descriptor."""
from __future__ import annotations

import codecs
import csv
import hashlib
import io
import json
import os
import re
from pathlib import Path

from utils.scoped_files import open_file


def _validate_window(offset_bytes, max_bytes):
    if type(offset_bytes) is not int or offset_bytes < 0:
        raise ValueError("offset_bytes must be a nonnegative integer")
    if type(max_bytes) is not int or not 256 <= max_bytes <= 200000:
        raise ValueError("max_bytes must be between 256 and 200000")


def text_window(workspace, path, *, offset_bytes=0, max_bytes=20000):
    _validate_window(offset_bytes, max_bytes)
    with open_file(workspace, path) as stream:
        size = os.fstat(stream.fileno()).st_size
        if offset_bytes > size:
            raise ValueError("offset_bytes is beyond this file")
        full_hash = None
        if size <= 200000:
            whole = stream.read(200001)
            if len(whole) != size:
                raise ValueError("File changed while reading; read it again")
            full_hash = hashlib.sha256(whole).hexdigest()
            raw = whole[offset_bytes:offset_bytes+max_bytes]
        else:
            stream.seek(offset_bytes)
            raw = stream.read(min(max_bytes, size-offset_bytes))
    if b"\x00" in raw:
        raise ValueError("Binary file: inspect it with an appropriate scientific tool or isolated Python")
    try:
        decoder = codecs.getincrementaldecoder("utf-8")()
        text = decoder.decode(raw, final=offset_bytes+len(raw) >= size)
    except UnicodeDecodeError as exc:
        raise ValueError("Use UTF-8 text and the returned next_offset_bytes; inspect other encodings with isolated Python") from exc
    pending = decoder.getstate()[0]
    used = len(raw)-len(pending)
    end = offset_bytes+used
    truncated = offset_bytes != 0 or end < size
    result = {"path": path, "text": text, "bytes": size,
              "offset_bytes": offset_bytes, "truncated": truncated,
              "next_offset_bytes": end if end < size else None}
    result["slice_sha256" if truncated else "sha256"] = hashlib.sha256(raw[:used]).hexdigest()
    if full_hash is not None:
        result["file_sha256"] = full_hash
    if end < size:
        result["next_step"] = "Read the next window if needed; use isolated Python for full-table analysis."
    return result


TABLE_SUFFIXES = frozenset({".csv", ".tsv", ".tab"})
TABLE_PREFIX_BYTES = 65536
SAMPLE_ROWS = 3
SAMPLE_COLUMNS = 32
SAMPLE_CELL_CHARS = 80
SAMPLE_COLUMN_CHARS = 128


def _protected_table_notice(text):
    """A protected CSV initially contains a JSON receipt, not data rows."""
    stripped = text.lstrip("\ufeff \r\n\t")
    if not stripped.startswith("{"):
        return False
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        # A small explicit byte budget can truncate the notice itself.
        return bool(re.match(r'^\{\s*"data_access"\s*:\s*"protected_research_table"\s*[,}]', stripped))
    return isinstance(value, dict) and value.get("data_access") == "protected_research_table"


def file_preview(workspace, path, *, offset_bytes=0, max_bytes=20000, view="auto"):
    """Default to a small table sample; raw text remains explicitly available."""
    _validate_window(offset_bytes, max_bytes)
    if view not in ("auto", "text"):
        raise ValueError("view must be auto or text")
    suffix = Path(path).suffix.lower()
    table = view == "auto" and offset_bytes == 0 and suffix in TABLE_SUFFIXES
    window = text_window(workspace, path, offset_bytes=offset_bytes,
                         max_bytes=min(max_bytes, TABLE_PREFIX_BYTES) if table else max_bytes)
    if not table or _protected_table_notice(window["text"]):
        return window
    return _table_sample(window, suffix)


def _table_sample(window, suffix):
    text = window["text"].lstrip("\ufeff")
    prefix_truncated = window["truncated"]
    incomplete_line = False
    if prefix_truncated and not text.endswith(("\n", "\r")):
        # Do not present an unquoted row cut by the byte budget as a whole row.
        end = max(text.rfind("\n"), text.rfind("\r"))
        text = text[:end + 1]
        incomplete_line = True
    delimiter = "\t" if suffix in {".tsv", ".tab"} else ","
    if suffix == ".csv" and text:
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t").delimiter
        except csv.Error:
            pass

    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    header, records, parse_issue = [], [], None
    try:
        header = next(reader, [])
        for row in reader:
            if row:
                records.append(row)
            if len(records) > SAMPLE_ROWS:
                break
    except csv.Error:
        parse_issue = ("The bounded prefix ends inside a record; that record was omitted."
                       if prefix_truncated else "Malformed delimited text; inspect the file with isolated Python.")

    rows = records[:SAMPLE_ROWS]
    clipped_columns = [i for i, value in enumerate(header[:SAMPLE_COLUMNS])
                       if len(value) > SAMPLE_COLUMN_CHARS]
    clipped_cells = [[i, j] for i, row in enumerate(rows)
                     for j, value in enumerate(row[:SAMPLE_COLUMNS]) if len(value) > SAMPLE_CELL_CHARS]
    omitted_columns = max(0, max([len(header), *(len(row) for row in rows)]) - SAMPLE_COLUMNS)
    result = {
        "path": window["path"], "preview_kind": "table", "bytes": window["bytes"],
        "delimiter": delimiter, "column_count": len(header),
        "columns": [value[:SAMPLE_COLUMN_CHARS] for value in header[:SAMPLE_COLUMNS]],
        "sample_rows": [[value[:SAMPLE_CELL_CHARS] for value in row[:SAMPLE_COLUMNS]] for row in rows],
        "rows_shown": len(rows), "sample_row_field_counts": [len(row) for row in rows],
        "source_prefix_bytes": len(window["text"].encode("utf-8")),
        "source_prefix_sha256": window.get("slice_sha256", window.get("sha256")),
        "truncated": bool(prefix_truncated or len(records) > SAMPLE_ROWS or omitted_columns
                          or clipped_columns or clipped_cells or parse_issue),
        "next_step": ("Use isolated Python to analyze the full table. This preview contains only sample rows. "
                      "For a targeted raw-text window, use view='text'."),
    }
    if omitted_columns:
        result["omitted_columns"] = omitted_columns
    if clipped_columns or clipped_cells:
        result["clipped_column_indexes"] = clipped_columns
        result["clipped_cell_indexes"] = clipped_cells
        result["character_limits"] = {"column": SAMPLE_COLUMN_CHARS, "cell": SAMPLE_CELL_CHARS}
    if incomplete_line:
        result["incomplete_line_omitted"] = True
    if parse_issue:
        result["parse_notice"] = parse_issue
    return result
