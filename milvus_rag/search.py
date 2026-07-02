"""Semantic Search + regex Grep + Fetch implementations.

Each function renders its result as a single plain-text string the LLM
can consume verbatim. Grep output mirrors Claude Code's Grep so a
CC-trained model reads results zero-shot; Search hits use a compact
per-row shape distinct enough that the model won't confuse channels.
"""

from __future__ import annotations

import json
import re
from typing import Any

from milvus_rag.corpus import CorpusError, MilvusCorpus


_MAX_HEAD_LIMIT = 1000
_MAX_PREVIEW_CHARS = 4000


# ---------------------------------------------------------------------------
# Semantic Search — vector ANN
# ---------------------------------------------------------------------------


def render_semantic_search(
    corpus: MilvusCorpus,
    query: str,
    k: int = 10,
    filter_expr: str | None = None,
) -> str:
    """Run Milvus ANN and format the top-k hits as a text block."""
    if not query or not query.strip():
        return "Search: `query` is required."
    k = max(1, min(int(k or 10), 100))
    try:
        rows = corpus.vector_search(query, k=k, filter_expr=filter_expr)
    except CorpusError as exc:
        return f"Search failed: {exc}"

    if not rows:
        return "No matches."

    text_field = corpus.milvus.text_field
    pk_field = corpus.pk_field
    lines: list[str] = [f"Found {len(rows)} matches (semantic, top-{k}):"]
    for i, row in enumerate(rows, start=1):
        pk = row.get(pk_field, "?")
        score = row.get("_score")
        score_s = f"{score:.4f}" if isinstance(score, float) else "?"
        body = str(row.get(text_field, "") or "").strip()
        body_preview = body[:_MAX_PREVIEW_CHARS]
        meta = _format_extra_metadata(row, skip={text_field, pk_field, "_score"})
        lines.append(f"\n[{i}] pk={pk} score={score_s}{meta}")
        lines.append(body_preview)
        if len(body) > _MAX_PREVIEW_CHARS:
            lines.append(f"[...truncated — Fetch(pk={pk!r}) for full text]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grep — regex text matching (Python-side over Milvus row stream)
# ---------------------------------------------------------------------------


def render_grep(
    corpus: MilvusCorpus,
    pattern: str,
    *,
    filter_expr: str | None = None,
    output_mode: str = "files_with_matches",
    before_context: int = 0,
    after_context: int = 0,
    context: int = 0,
    show_line_numbers: bool = True,
    case_insensitive: bool = False,
    multiline: bool = False,
    head_limit: int = 250,
    offset: int = 0,
) -> str:
    """Regex-scan text across the corpus.

    Uses Python's :mod:`re` module over rows streamed from Milvus so
    the full regex grammar is available (character classes, backrefs,
    multiline etc.) rather than the substring-only Milvus ``like``.

    Output mirrors Claude Code's Grep:

    * ``files_with_matches`` (default): ``pk_1\\npk_2\\n...``
    * ``content``: ``pk:Lline:content`` per match, with -A / -B / -C
      producing context blocks separated by ``--``.
    * ``count``: ``pk_1: N matches`` per matching row.

    ``pk`` is the Milvus primary-key value. The model treats it as an
    opaque handle it can feed back into ``Fetch`` for full row detail.
    """
    if not pattern or not pattern.strip():
        return "Grep: `pattern` is required."
    if output_mode not in ("content", "files_with_matches", "count"):
        return (
            f"Grep: unknown output_mode={output_mode!r}. Use "
            "'content', 'files_with_matches', or 'count'."
        )
    head_limit = max(0, min(int(head_limit or 0), _MAX_HEAD_LIMIT))
    if head_limit == 0:
        head_limit = _MAX_HEAD_LIMIT
    offset = max(0, int(offset or 0))

    flags = 0
    if case_insensitive:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.MULTILINE | re.DOTALL
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return f"Grep: invalid regex: {exc}"

    ctx_before = max(0, int(before_context or context or 0))
    ctx_after = max(0, int(after_context or context or 0))

    text_field = corpus.milvus.text_field
    pk_field = corpus.pk_field

    files_with_matches: list[Any] = []
    counts: list[tuple[Any, int]] = []
    content_lines: list[str] = []
    emitted = 0
    scanned = 0

    try:
        row_iter = corpus.iter_rows(
            filter_expr=filter_expr, limit=_MAX_HEAD_LIMIT * 4,
        )
    except CorpusError as exc:
        return f"Grep failed: {exc}"

    for row in row_iter:
        scanned += 1
        text = row.get(text_field) or ""
        if not isinstance(text, str):
            text = str(text)
        if not text:
            continue

        pk = row.get(pk_field, "?")
        if multiline:
            matches = list(regex.finditer(text))
            if not matches:
                continue
            if output_mode == "files_with_matches":
                files_with_matches.append(pk)
                emitted += 1
            elif output_mode == "count":
                counts.append((pk, len(matches)))
                emitted += 1
            else:  # content
                for m in matches:
                    line_no = text.count("\n", 0, m.start()) + 1
                    line_start = text.rfind("\n", 0, m.start()) + 1
                    line_end = text.find("\n", m.end())
                    if line_end < 0:
                        line_end = len(text)
                    matched_line = text[line_start:line_end]
                    prefix = f"{pk}:L{line_no}:" if show_line_numbers else f"{pk}:"
                    content_lines.append(f"{prefix}{matched_line}")
                    emitted += 1
                    if emitted >= head_limit + offset:
                        break
        else:
            lines = text.splitlines()
            matching_line_idxs = [
                i for i, ln in enumerate(lines) if regex.search(ln)
            ]
            if not matching_line_idxs:
                continue
            if output_mode == "files_with_matches":
                files_with_matches.append(pk)
                emitted += 1
            elif output_mode == "count":
                counts.append((pk, len(matching_line_idxs)))
                emitted += 1
            else:  # content
                ranges = _merge_context_ranges(
                    matching_line_idxs, ctx_before, ctx_after, len(lines),
                )
                for r_idx, (lo, hi) in enumerate(ranges):
                    if r_idx > 0:
                        content_lines.append("--")
                    for j in range(lo, hi + 1):
                        prefix = f"{pk}:L{j+1}:" if show_line_numbers else f"{pk}:"
                        content_lines.append(f"{prefix}{lines[j].rstrip()}")
                emitted += len(matching_line_idxs)

        if emitted >= head_limit + offset:
            break

    if output_mode == "files_with_matches":
        window = files_with_matches[offset : offset + head_limit]
        if not window:
            return f"No matches (scanned {scanned} rows)."
        return "\n".join(str(pk) for pk in window)

    if output_mode == "count":
        window = counts[offset : offset + head_limit]
        if not window:
            return f"No matches (scanned {scanned} rows)."
        return "\n".join(f"{pk}: {n} matches" for pk, n in window)

    # content mode
    window = content_lines[offset : offset + head_limit]
    if not window:
        return f"No matches (scanned {scanned} rows)."
    return "\n".join(window)


def _merge_context_ranges(
    match_idxs: list[int],
    before: int,
    after: int,
    n_lines: int,
) -> list[tuple[int, int]]:
    """Collapse overlapping ``[idx-before, idx+after]`` windows."""
    ranges: list[list[int]] = []
    for idx in match_idxs:
        lo = max(0, idx - before)
        hi = min(n_lines - 1, idx + after)
        if ranges and lo <= ranges[-1][1] + 1:
            ranges[-1][1] = max(ranges[-1][1], hi)
        else:
            ranges.append([lo, hi])
    return [(lo, hi) for lo, hi in ranges]


# ---------------------------------------------------------------------------
# Fetch — direct primary-key lookup
# ---------------------------------------------------------------------------


def render_fetch(corpus: MilvusCorpus, pk: Any) -> str:
    """Return every non-vector column for one PK as a labelled block."""
    if pk is None or (isinstance(pk, str) and not pk.strip()):
        return "Fetch: `pk` is required."
    try:
        row = corpus.fetch(pk)
    except CorpusError as exc:
        return f"Fetch failed: {exc}"
    if row is None:
        return f"Fetch: no row with pk={pk!r}."
    text_field = corpus.milvus.text_field
    lines: list[str] = []
    body = row.pop(text_field, "")
    for key, val in row.items():
        lines.append(f"{key}: {_short(val)}")
    lines.append("")
    lines.append(f"{text_field}:")
    lines.append(str(body).strip())
    return "\n".join(lines)


def _short(v: Any, limit: int = 300) -> str:
    """Render a scalar / small collection to one displayable line."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        s = str(v)
    else:
        try:
            s = json.dumps(v, ensure_ascii=False)
        except Exception:
            s = str(v)
    return s if len(s) <= limit else s[:limit] + "…"


def _format_extra_metadata(row: dict[str, Any], skip: set[str]) -> str:
    """Format non-text non-pk columns as ``key=val`` pairs on one line."""
    parts: list[str] = []
    for k, v in row.items():
        if k in skip:
            continue
        parts.append(f"{k}={_short(v, limit=80)}")
    return (" " + " ".join(parts)) if parts else ""


__all__ = [
    "render_fetch",
    "render_grep",
    "render_semantic_search",
]
