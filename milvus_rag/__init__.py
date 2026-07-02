"""milvus-rag — general-purpose agentic RAG plugin for OpenAI Codex.

Connects any Milvus collection with a text column and (optionally) a
vector column to the Codex agent through three tools:

    Search — semantic vector search via an OpenAI-compatible embedding
             endpoint (BGE-M3, OpenAI, vLLM, TEI — whatever you point
             MILVUS_RAG_EMBED_URL at).
    Grep   — regex over the text column, ripgrep-style output.
    Fetch  — return every non-vector column of a single row by PK.

Schema-agnostic: no assumption about doc / page / chunk metadata.
Whatever columns exist flow through unchanged.

Entry points:
    milvus-rag            (console script; used by `uvx` from Codex)
    python -m milvus_rag  (equivalent; used for local dev)
"""
