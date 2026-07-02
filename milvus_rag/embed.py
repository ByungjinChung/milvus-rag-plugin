"""OpenAI-compatible embedding client.

Single-purpose: POST a text query to ``{EMBED_URL}`` (an OpenAI-shaped
``/v1/embeddings`` endpoint), return the vector. Works against any
compatible server — OpenAI, vLLM, llama.cpp, LM Studio, TEI, Together,
etc. The model name is a request-body value the server treats as-is;
we do not validate it locally because the pool of legitimate names is
provider-specific.

Uses ``urllib`` (stdlib) so the plugin doesn't pull ``httpx`` or
``requests`` into the install closure. This keeps ``uvx`` cold-start
fast and avoids version conflicts with a user's existing Python env.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from milvus_rag.config import EmbedConfig


class EmbedError(RuntimeError):
    """Raised when the embedding endpoint fails or returns an unusable body."""


def embed_query(query: str, config: EmbedConfig) -> list[float]:
    """Return the embedding vector for ``query`` from the OpenAI-compat endpoint.

    Request shape (identical to OpenAI ``/v1/embeddings``):

        POST {url}
        {
            "input": "<query>",
            "model": "<config.model>"        # sent as-is; blank if unset
        }

    ``Authorization: Bearer <api_key>`` is added when ``api_key`` is
    non-empty (required by OpenAI / Together / etc.; usually optional
    for a self-hosted llama.cpp / vLLM / TEI).

    Response is parsed as ``{"data": [{"embedding": [...]}]}``. Anything
    else raises :class:`EmbedError` so the caller surfaces a clear
    reason instead of a downstream Milvus shape error.
    """
    if not config.enabled:
        raise EmbedError(
            "Semantic search is not configured — set MILVUS_RAG_EMBED_URL "
            "(and optionally MILVUS_RAG_EMBED_MODEL / _API_KEY / _TIMEOUT_S). "
            "Grep still works without an embedding endpoint."
        )

    body: dict[str, Any] = {"input": query}
    if config.model:
        body["model"] = config.model
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    req = urllib.request.Request(
        config.url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise EmbedError(
            f"embedding endpoint HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise EmbedError(f"embedding endpoint unreachable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EmbedError(f"embedding endpoint returned non-JSON: {exc}") from exc

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise EmbedError(
            f"embedding endpoint returned no data: {str(payload)[:200]}"
        )
    vec = data[0].get("embedding")
    if not isinstance(vec, list) or not vec:
        raise EmbedError(
            f"embedding endpoint returned no vector: {str(data[0])[:200]}"
        )
    if not all(isinstance(x, (int, float)) for x in vec):
        raise EmbedError("embedding endpoint returned non-numeric vector")
    return [float(x) for x in vec]


__all__ = ["EmbedError", "embed_query"]
