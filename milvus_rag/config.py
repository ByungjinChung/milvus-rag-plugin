"""Environment-driven configuration for the milvus-rag plugin.

Two logical config blocks:

    MilvusConfig  — connection + which columns to read.
    EmbedConfig   — OpenAI-compatible embedding endpoint used by
                    semantic Search. Absent config means semantic
                    Search is unavailable; Grep still works.

Env var names use the ``MILVUS_RAG_*`` prefix. ``RETRIEVAL_CORE_*``
aliases from the retrieval-plugin upstream are accepted for backward
compatibility on the four Milvus connection fields.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(*names: str, default: str = "") -> str:
    """Return the first non-empty env value among ``names``."""
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return default


@dataclass(frozen=True)
class MilvusConfig:
    """Milvus connection + column pointers."""

    uri: str
    collection: str
    db_name: str
    user: str
    password: str
    token: str
    text_field: str
    vector_field: str

    @classmethod
    def from_env(cls) -> MilvusConfig:
        return cls(
            uri=_env("MILVUS_RAG_URI", "RETRIEVAL_CORE_MILVUS_URI"),
            collection=_env(
                "MILVUS_RAG_COLLECTION", "RETRIEVAL_CORE_MILVUS_COLLECTION",
            ),
            db_name=_env(
                "MILVUS_RAG_DB_NAME", "RETRIEVAL_CORE_MILVUS_DB_NAME",
            ),
            user=_env("MILVUS_RAG_USER", "RETRIEVAL_CORE_MILVUS_USER"),
            password=_env(
                "MILVUS_RAG_PASSWORD", "RETRIEVAL_CORE_MILVUS_PASSWORD",
            ),
            token=_env("MILVUS_RAG_TOKEN", "RETRIEVAL_CORE_MILVUS_TOKEN"),
            text_field=_env("MILVUS_RAG_TEXT_FIELD", default="text"),
            vector_field=_env("MILVUS_RAG_VECTOR_FIELD", default=""),
        )

    def validate(self) -> list[str]:
        """Return a list of missing-required error messages (empty on OK)."""
        errs: list[str] = []
        if not self.uri:
            errs.append("MILVUS_RAG_URI is required")
        if not self.collection:
            errs.append("MILVUS_RAG_COLLECTION is required")
        if not self.text_field:
            errs.append("MILVUS_RAG_TEXT_FIELD is required")
        return errs


@dataclass(frozen=True)
class EmbedConfig:
    """OpenAI-compatible embedding endpoint config."""

    url: str
    model: str
    api_key: str
    metric: str
    timeout_s: float

    @classmethod
    def from_env(cls) -> EmbedConfig:
        return cls(
            url=_env("MILVUS_RAG_EMBED_URL"),
            model=_env("MILVUS_RAG_EMBED_MODEL"),
            api_key=_env("MILVUS_RAG_EMBED_API_KEY"),
            metric=_env("MILVUS_RAG_METRIC", default="COSINE").upper(),
            timeout_s=float(_env("MILVUS_RAG_EMBED_TIMEOUT_S", default="30")),
        )

    @property
    def enabled(self) -> bool:
        """True when semantic Search is usable — endpoint is set."""
        return bool(self.url)


__all__ = ["EmbedConfig", "MilvusConfig"]
