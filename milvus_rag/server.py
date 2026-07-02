"""MCP stdio server — exposes Search / Grep / Fetch over JSON-RPC 2.0.

Wire format: one JSON message per line, UTF-8, ``\\n``-terminated.
Supported methods: ``initialize``, ``tools/list``, ``tools/call``,
``ping``, plus notification acknowledgements.

stderr is redirected to ``os.devnull`` at startup so chatty output
cannot block the pipe (the client never drains it).
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional — fall back to plain os.environ
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:  # type: ignore
        return False

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env", override=False)

from milvus_rag.config import EmbedConfig, MilvusConfig  # noqa: E402
from milvus_rag.corpus import CorpusError, MilvusCorpus  # noqa: E402
from milvus_rag.descriptions import (  # noqa: E402
    FETCH_DESCRIPTION,
    GREP_DESCRIPTION,
    SEARCH_DESCRIPTION,
)
from milvus_rag.search import (  # noqa: E402
    render_fetch,
    render_grep,
    render_semantic_search,
)


PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "milvus-rag-mcp"
SERVER_VERSION = "0.1.0"


def _debug_log(line: str) -> None:
    """Append ``line`` to ``MILVUS_RAG_DEBUG_LOG`` if set."""
    path = os.environ.get("MILVUS_RAG_DEBUG_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# ---------------------------------------------------------------------------
# Tool schemas advertised via tools/list
# ---------------------------------------------------------------------------


_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "Search",
        "description": SEARCH_DESCRIPTION,
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language query to embed and vector-search "
                        "the corpus with."
                    ),
                },
                "k": {
                    "type": "integer",
                    "description": (
                        "Number of top results to return. Default 10, "
                        "max 100."
                    ),
                },
                "filter": {
                    "type": "string",
                    "description": (
                        "Optional raw Milvus filter expression to scope the "
                        "search, e.g. `source == \"guide.pdf\"`."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "Grep",
        "description": GREP_DESCRIPTION,
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Python regex. Supports full grammar — character "
                        "classes, groups, backrefs, alternation."
                    ),
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["files_with_matches", "content", "count"],
                    "description": (
                        "Result shape. Defaults to `files_with_matches`."
                    ),
                },
                "-A": {
                    "type": "integer",
                    "description": (
                        "Lines after each match (content mode only)."
                    ),
                },
                "-B": {
                    "type": "integer",
                    "description": (
                        "Lines before each match (content mode only)."
                    ),
                },
                "-C": {
                    "type": "integer",
                    "description": "Alias for context around each match.",
                },
                "context": {
                    "type": "integer",
                    "description": (
                        "Lines before AND after each match (content mode)."
                    ),
                },
                "-n": {
                    "type": "boolean",
                    "description": (
                        "Show line numbers in content mode. Default true."
                    ),
                },
                "-i": {
                    "type": "boolean",
                    "description": "Case-insensitive match. Default false.",
                },
                "multiline": {
                    "type": "boolean",
                    "description": (
                        "Enable multiline mode: `.` matches newlines and "
                        "patterns can span lines. Default false."
                    ),
                },
                "head_limit": {
                    "type": "integer",
                    "description": (
                        "Cap on returned entries. Default 250, max 1000."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "Skip N leading entries before applying head_limit."
                    ),
                },
                "filter": {
                    "type": "string",
                    "description": (
                        "Optional raw Milvus filter expression to scope the "
                        "scan."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Fetch",
        "description": FETCH_DESCRIPTION,
        "inputSchema": {
            "type": "object",
            "properties": {
                "pk": {
                    "description": (
                        "Primary key value (integer or string) returned by "
                        "a prior Search / Grep call."
                    ),
                },
            },
            "required": ["pk"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def _call_tool(corpus: MilvusCorpus, name: str, args: dict[str, Any]) -> str:
    if name == "Search":
        return render_semantic_search(
            corpus,
            query=str(args.get("query") or ""),
            k=int(args.get("k") or 10),
            filter_expr=args.get("filter") or None,
        )
    if name == "Grep":
        return render_grep(
            corpus,
            pattern=str(args.get("pattern") or ""),
            filter_expr=args.get("filter") or None,
            output_mode=str(args.get("output_mode") or "files_with_matches"),
            before_context=int(args.get("-B") or 0),
            after_context=int(args.get("-A") or 0),
            context=int(args.get("context") or args.get("-C") or 0),
            show_line_numbers=(
                True if args.get("-n") is None else bool(args.get("-n"))
            ),
            case_insensitive=bool(args.get("-i") or False),
            multiline=bool(args.get("multiline") or False),
            head_limit=int(args.get("head_limit") or 250),
            offset=int(args.get("offset") or 0),
        )
    if name == "Fetch":
        return render_fetch(corpus, args.get("pk"))
    raise ValueError(f"unknown tool: {name!r}")


# ---------------------------------------------------------------------------
# JSON-RPC message handling
# ---------------------------------------------------------------------------


def _response(rpc_id: Any, result: Any) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": rpc_id, "result": result}) + "\n"


def _error(rpc_id: Any, code: int, message: str) -> str:
    return json.dumps({
        "jsonrpc": "2.0", "id": rpc_id,
        "error": {"code": code, "message": message},
    }) + "\n"


def _handle(corpus_holder: dict, msg: dict[str, Any]) -> str | None:
    method = msg.get("method")
    rpc_id = msg.get("id")
    params = msg.get("params") or {}

    if rpc_id is None:
        if method == "notifications/cancelled":
            _debug_log(f"notifications/cancelled params={params}")
        return None

    if method == "initialize":
        return _response(rpc_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })

    if method == "ping":
        return _response(rpc_id, {})

    if method == "tools/list":
        return _response(rpc_id, {"tools": _TOOL_SCHEMAS})

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        _debug_log(f"tools/call name={name!r} args={args!r}")
        if not isinstance(name, str):
            return _error(rpc_id, INVALID_PARAMS, "missing tool name")
        try:
            corpus = corpus_holder["corpus"]
            text = _call_tool(corpus, name, args)
            _debug_log(f"  -> len={len(text)} preview={text[:120]!r}")
            return _response(rpc_id, {
                "content": [{"type": "text", "text": text}],
                "isError": False,
            })
        except CorpusError as exc:
            return _response(rpc_id, {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            })
        except Exception as exc:  # noqa: BLE001
            _debug_log(f"  -> EXCEPTION {type(exc).__name__}: {exc}")
            return _error(
                rpc_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}",
            )

    return _error(rpc_id, METHOD_NOT_FOUND, f"unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    """Read JSON-RPC lines from stdin, write responses to stdout."""
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115 — lifetime = process

    milvus_cfg = MilvusConfig.from_env()
    embed_cfg = EmbedConfig.from_env()
    errs = milvus_cfg.validate()
    if errs:
        # Emit a startup-shaped notification the client will surface to
        # the user before any tools/list attempt.
        sys.stdout.write(_error(
            None, INVALID_REQUEST,
            "milvus-rag config error: " + "; ".join(errs),
        ))
        sys.stdout.flush()
        return 2

    corpus_holder: dict[str, MilvusCorpus] = {}

    def ensure_corpus() -> MilvusCorpus:
        if "corpus" not in corpus_holder:
            corpus_holder["corpus"] = MilvusCorpus(milvus_cfg, embed_cfg)
        return corpus_holder["corpus"]

    class _LazyDict(dict):
        def __getitem__(self, key: str) -> Any:
            if key == "corpus":
                return ensure_corpus()
            return super().__getitem__(key)

    holder = _LazyDict()

    while True:
        line = sys.stdin.readline()
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(_error(None, PARSE_ERROR, "invalid JSON"))
            sys.stdout.flush()
            continue
        try:
            out = _handle(holder, msg)
        except Exception:  # noqa: BLE001
            out = _error(
                msg.get("id"), INTERNAL_ERROR,
                f"server error: {traceback.format_exc(limit=1)}",
            )
        if out is not None:
            sys.stdout.write(out)
            sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
