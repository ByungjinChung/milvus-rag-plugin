"""Milvus corpus wrapper — connection, row iteration, PK lookup, vector ANN.

Intentionally schema-agnostic. The plugin does not assume the collection
carries ``doc`` / ``page`` metadata — the only column the wrapper truly
needs is ``text_field`` (returned by every tool) and, for semantic
Search, ``vector_field``. Any other columns the schema exposes flow
through unchanged; tools surface them to the model as-is.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterator

from milvus_rag.config import EmbedConfig, MilvusConfig
from milvus_rag.embed import EmbedError, embed_query


_ITER_BATCH_SIZE = 1000
_DEFAULT_LIMIT = 250


class CorpusError(RuntimeError):
    """Raised when a Milvus call fails or preconditions aren't met."""


class MilvusCorpus:
    """Thin wrapper over a single Milvus collection.

    A dedicated connection alias is opened on the first query and held
    for process lifetime. Schema is loaded once and cached so tool
    responses can name concrete columns (``pk``, ``text``, whatever the
    collection also has).
    """

    def __init__(self, milvus: MilvusConfig, embed: EmbedConfig) -> None:
        self.milvus = milvus
        self.embed = embed
        self._alias: str | None = None
        self._collection: Any = None
        self._pk_field: str | None = None
        self._all_fields: tuple[str, ...] = ()

    # -- connection / schema ------------------------------------------

    def _ensure_connected(self) -> Any:
        """Open the connection on first use; cache the ``Collection`` handle."""
        if self._collection is not None:
            return self._collection
        try:
            from pymilvus import Collection, connections
        except ImportError as exc:
            raise CorpusError(
                "pymilvus is required. Install with `pip install pymilvus`."
            ) from exc

        alias = f"milvus-rag-{uuid.uuid4().hex[:8]}"
        kwargs: dict[str, Any] = {"alias": alias, "uri": self.milvus.uri}
        if self.milvus.db_name:
            kwargs["db_name"] = self.milvus.db_name
        if self.milvus.token:
            kwargs["token"] = self.milvus.token
        if self.milvus.user:
            kwargs["user"] = self.milvus.user
        if self.milvus.password:
            kwargs["password"] = self.milvus.password
        try:
            connections.connect(**kwargs)
        except Exception as exc:  # pymilvus surfaces many concrete types
            raise CorpusError(
                f"failed to connect to Milvus at {self.milvus.uri}: {exc}"
            ) from exc

        try:
            collection = Collection(self.milvus.collection, using=alias)
            collection.load()
        except Exception as exc:
            raise CorpusError(
                f"failed to open collection {self.milvus.collection!r}: {exc}"
            ) from exc

        self._alias = alias
        self._collection = collection
        self._all_fields = tuple(f.name for f in collection.schema.fields)
        for f in collection.schema.fields:
            if getattr(f, "is_primary", False):
                self._pk_field = f.name
                break
        if self._pk_field is None:
            raise CorpusError(
                f"collection {self.milvus.collection!r} has no primary-key "
                "field; milvus-rag requires one for Fetch"
            )
        if self.milvus.text_field not in self._all_fields:
            raise CorpusError(
                f"MILVUS_RAG_TEXT_FIELD={self.milvus.text_field!r} not found "
                f"in collection. Available fields: {list(self._all_fields)}"
            )
        return collection

    @property
    def pk_field(self) -> str:
        self._ensure_connected()
        assert self._pk_field is not None
        return self._pk_field

    @property
    def all_fields(self) -> tuple[str, ...]:
        self._ensure_connected()
        return self._all_fields

    def _output_fields(self) -> list[str]:
        """Return every non-vector field. Vectors are opaque to the model.

        Auto-detects vector columns from the schema's dtype so we skip
        them even when the user hasn't set ``MILVUS_RAG_VECTOR_FIELD``
        (e.g. a Grep-only user pointing at a Milvus collection that
        happens to carry embeddings). Milvus' vector dtypes all contain
        ``VECTOR`` in their name (``FLOAT_VECTOR``, ``BINARY_VECTOR``,
        ``SPARSE_FLOAT_VECTOR``, ``FLOAT16_VECTOR``, ``BFLOAT16_VECTOR``).
        """
        assert self._collection is not None
        skip: set[str] = set()
        for f in self._collection.schema.fields:
            # DataType is IntEnum; ``name`` gives us the label
            # ("FLOAT_VECTOR", "SPARSE_FLOAT_VECTOR", etc.) rather than
            # the numeric ``str(f.dtype)`` = "101" that hides it.
            dtype_label = getattr(f.dtype, "name", "") or repr(f.dtype)
            if "VECTOR" in dtype_label.upper():
                skip.add(f.name)
        if self.milvus.vector_field:
            skip.add(self.milvus.vector_field)
        return [f for f in self._all_fields if f not in skip]

    # -- primary-key fetch --------------------------------------------

    def fetch(self, pk: Any) -> dict[str, Any] | None:
        """Return every non-vector column for a single row, or None."""
        collection = self._ensure_connected()
        expr = self._pk_expr(pk)
        rows = collection.query(
            expr=expr, output_fields=self._output_fields(), limit=1,
        )
        if not rows:
            return None
        return dict(rows[0])

    def _pk_expr(self, pk: Any) -> str:
        """Build a Milvus filter expression that matches a single PK."""
        if isinstance(pk, (int, float)) and not isinstance(pk, bool):
            return f"{self.pk_field} == {int(pk)}"
        text = str(pk).replace("\\", "\\\\").replace('"', '\\"')
        return f'{self.pk_field} == "{text}"'

    # -- row iteration (for Grep) -------------------------------------

    def iter_rows(
        self,
        filter_expr: str | None = None,
        limit: int = _DEFAULT_LIMIT,
    ) -> Iterator[dict[str, Any]]:
        """Stream rows matching an optional Milvus filter expression.

        When ``filter_expr`` is empty we scan the collection via a
        ``pk >= 0`` catch-all (for numeric PKs) or ``pk != ""`` (for
        string PKs). ``limit`` caps the total scan so a Grep against a
        multi-million-row corpus doesn't stall — callers pass their own
        ``head_limit`` down as ``limit`` here.
        """
        collection = self._ensure_connected()
        expr = filter_expr.strip() if filter_expr else self._catchall_expr()
        output_fields = self._output_fields()

        iterator_factory = getattr(collection, "query_iterator", None)
        if iterator_factory is not None:
            iterator = iterator_factory(
                expr=expr,
                output_fields=output_fields,
                batch_size=_ITER_BATCH_SIZE,
                limit=limit,
            )
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    for row in batch:
                        yield dict(row)
            finally:
                closer = getattr(iterator, "close", None)
                if callable(closer):
                    closer()
            return

        rows = collection.query(
            expr=expr, output_fields=output_fields, limit=limit,
        )
        for row in rows:
            yield dict(row)

    def _catchall_expr(self) -> str:
        """Return a filter that matches every row regardless of PK type."""
        # A numeric PK accepts ``>= 0`` cheaply. For a VARCHAR PK we use
        # ``!= ""`` which is the closest cheap ``true`` we can express
        # in Milvus filter syntax. Detect via first ``query`` schema.
        # The schema has already been loaded in ``_ensure_connected``,
        # so we can peek at the pk field's dtype directly.
        pk = self.pk_field
        if not self._collection:
            return f"{pk} >= 0"
        for f in self._collection.schema.fields:
            if f.name == pk:
                # DataType.VARCHAR / STRING are 21 / 20 in pymilvus.
                dtype_str = str(f.dtype).upper()
                if "VARCHAR" in dtype_str or "STRING" in dtype_str:
                    return f'{pk} != ""'
                break
        return f"{pk} >= 0"

    # -- vector ANN search --------------------------------------------

    def vector_search(
        self,
        query: str,
        k: int,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """Embed ``query`` and run Milvus ANN.

        Returns rows sorted by score (best first) with a synthetic
        ``_score`` field appended. All non-vector columns are included
        so the LLM sees whatever metadata the collection carries.

        Raises :class:`CorpusError` when the collection has no vector
        field configured (``MILVUS_RAG_VECTOR_FIELD`` unset) or when
        the embedding endpoint fails.
        """
        collection = self._ensure_connected()
        if not self.milvus.vector_field:
            raise CorpusError(
                "MILVUS_RAG_VECTOR_FIELD is not set — Search cannot embed "
                "a query. Use Grep for text matching, or set the vector "
                "column name in env."
            )
        if self.milvus.vector_field not in self._all_fields:
            raise CorpusError(
                f"MILVUS_RAG_VECTOR_FIELD={self.milvus.vector_field!r} not "
                f"found in collection. Available fields: {list(self._all_fields)}"
            )
        try:
            vec = embed_query(query, self.embed)
        except EmbedError as exc:
            raise CorpusError(str(exc)) from exc

        search_params = {"metric_type": self.embed.metric}
        kwargs: dict[str, Any] = {
            "data": [vec],
            "anns_field": self.milvus.vector_field,
            "param": search_params,
            "limit": max(1, int(k)),
            "output_fields": self._output_fields(),
        }
        if filter_expr and filter_expr.strip():
            kwargs["expr"] = filter_expr.strip()
        try:
            results = collection.search(**kwargs)
        except Exception as exc:
            raise CorpusError(f"Milvus search failed: {exc}") from exc

        rows: list[dict[str, Any]] = []
        for hit in results[0]:
            row = dict(hit.entity.fields) if hasattr(hit.entity, "fields") else {}
            if not row:
                # Older pymilvus exposes fields via ``.get``.
                row = {f: hit.entity.get(f) for f in self._output_fields()}
            row["_score"] = float(hit.distance)
            row.setdefault(self.pk_field, hit.id)
            rows.append(row)
        return rows

    # -- cleanup -------------------------------------------------------

    def close(self) -> None:
        if self._alias is None:
            return
        try:
            from pymilvus import connections
            connections.disconnect(alias=self._alias)
        except Exception:
            pass
        self._alias = None
        self._collection = None


__all__ = ["CorpusError", "MilvusCorpus"]
