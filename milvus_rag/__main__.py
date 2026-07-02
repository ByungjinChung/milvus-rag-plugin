"""Process entry point — ``python -m milvus_rag``.

Codex CLI spawns this via the ``[mcp_servers.corpus]`` block in
``~/.codex/config.toml``. Self-bootstraps ``sys.path`` so the package
resolves whether Codex launched us with ``-m`` (already on
``PYTHONPATH``) or by absolute file path (path not primed yet).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from milvus_rag.server import main  # noqa: E402

raise SystemExit(main())
