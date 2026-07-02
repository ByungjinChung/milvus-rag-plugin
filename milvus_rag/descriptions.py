"""Tool descriptions surfaced via ``tools/list``.

Keep in lockstep with the schemas registered in server.py. These are
the strings the LLM sees to decide which tool to call.
"""

from __future__ import annotations


SEARCH_DESCRIPTION = (
    "Semantic search over the Milvus corpus using vector similarity.\n\n"
    "Usage:\n"
    "- Use for natural-language questions and conceptual queries.\n"
    "  Examples: \"how does authentication work here\", "
    "\"why did the parser fail\", \"policy on refunds\".\n"
    "- Embeds `query` via the configured OpenAI-compatible endpoint, "
    "then runs Milvus ANN against the collection's vector field.\n"
    "- Returns the top-k rows ranked by similarity, with a `pk` handle "
    "you can feed to `Fetch` for full row detail.\n"
    "- Prefer `Grep` when you know the exact string, identifier, error "
    "code, or a regex pattern — semantic search may miss rare tokens.\n"
    "- Optional `filter` is a raw Milvus filter expression scoped by "
    "any column your schema has, e.g. `source == \"guide.pdf\"`. Leave "
    "empty to search the whole collection."
)


GREP_DESCRIPTION = (
    "Regex text matching across the corpus — ripgrep semantics over "
    "the Milvus text column. Model-agnostic: works whether or not the "
    "collection has vectors or an embedding endpoint is configured.\n\n"
    "Usage:\n"
    "- Use for exact identifiers, error codes, code definitions, rare "
    "tokens the embedding model may not capture, or any query where "
    "you already know the string / pattern.\n"
    "- `pattern` is a full Python regex — supports character classes, "
    "backrefs, groups.\n"
    "- Output modes:\n"
    "    * `files_with_matches` (default) — one `pk` per matching row.\n"
    "    * `content` — matched lines as `pk:Lline:content`, with -A/-B/"
    "-C context supported. Set `-n=false` to omit line numbers.\n"
    "    * `count` — `pk: N matches` per row.\n"
    "- `-i` for case-insensitive, `multiline: true` for cross-line "
    "patterns.\n"
    "- `head_limit` (default 250) caps the number of returned entries; "
    "`offset` (default 0) skips leading entries for paging.\n"
    "- Optional `filter` is a raw Milvus filter expression to scope "
    "the scan (same shape as Search's `filter`)."
)


FETCH_DESCRIPTION = (
    "Return every non-vector column of a single row by primary key.\n\n"
    "Usage:\n"
    "- Call after `Search` or `Grep` returns a `pk` handle you want to "
    "expand — full text and all metadata columns come back.\n"
    "- `pk` accepts either the integer or string form the collection "
    "uses; pass the exact value the previous tool returned."
)


__all__ = [
    "FETCH_DESCRIPTION",
    "GREP_DESCRIPTION",
    "SEARCH_DESCRIPTION",
]
