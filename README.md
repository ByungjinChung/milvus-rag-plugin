# Milvus RAG for Codex

Milvus RAG is an OpenAI Codex plugin that connects a Milvus collection to Codex as a live document corpus. It exposes MCP tools for semantic search, regex search, exact row fetching, collection browsing, and coverage checks.

## Install from Codex

Add this repository as a Codex plugin marketplace:

```text
/plugin marketplace add ByungjinChung/milvus-rag-plugin
```

Then install the plugin from that marketplace:

```text
/plugin install milvus-rag@milvus-rag-plugin
```

Codex will install the plugin metadata from this repository. The plugin MCP server runs through `uvx`, so the first launch downloads the Python package and dependencies in an isolated environment.

## Requirements

- OpenAI Codex with plugin support.
- `uv` available on your PATH.
- Milvus 2.4 or newer.
- A Milvus collection with at least one text field.
- For semantic search, a vector field and an OpenAI-compatible embeddings endpoint.

Install `uv`:

```powershell
winget install --id astral-sh.uv
```

On macOS:

```bash
brew install uv
```

On Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Configure Milvus

After installation, set the MCP environment values in your Codex config or through the plugin settings:

```toml
[mcp_servers.corpus.env]
MILVUS_RAG_URI = "http://127.0.0.1:19530"
MILVUS_RAG_COLLECTION = "my_documents"
MILVUS_RAG_TEXT_FIELD = "text"
MILVUS_RAG_METRIC = "COSINE"
```

To enable semantic search, also configure:

```toml
[mcp_servers.corpus.env]
MILVUS_RAG_VECTOR_FIELD = "dense"
MILVUS_RAG_EMBED_URL = "http://127.0.0.1:8080/v1/embeddings"
MILVUS_RAG_EMBED_MODEL = "bge-m3"
MILVUS_RAG_EMBED_API_KEY = ""
```

## Environment Variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `MILVUS_RAG_URI` | Yes | `http://127.0.0.1:19530` | Milvus server URI. |
| `MILVUS_RAG_COLLECTION` | Yes | `pages` | Collection to query. |
| `MILVUS_RAG_TEXT_FIELD` | Yes | `text` | Field containing searchable text. |
| `MILVUS_RAG_DB_NAME` | No | | Milvus database name. |
| `MILVUS_RAG_USER` | No | | Milvus username. |
| `MILVUS_RAG_PASSWORD` | No | | Milvus password. |
| `MILVUS_RAG_TOKEN` | No | | Milvus token. |
| `MILVUS_RAG_VECTOR_FIELD` | Search only | | Vector field used for semantic search. |
| `MILVUS_RAG_EMBED_URL` | Search only | | OpenAI-compatible `/v1/embeddings` endpoint. |
| `MILVUS_RAG_EMBED_MODEL` | No | | Embedding model name sent in the request body. |
| `MILVUS_RAG_EMBED_API_KEY` | No | | Bearer token for the embeddings endpoint. |
| `MILVUS_RAG_METRIC` | No | `COSINE` | Milvus metric: `COSINE`, `IP`, or `L2`. |
| `MILVUS_RAG_EMBED_TIMEOUT_S` | No | `30` | Embedding request timeout in seconds. |

## Tools

`Search(query, k=10, filter=None)` performs semantic vector search. It requires `MILVUS_RAG_VECTOR_FIELD` and `MILVUS_RAG_EMBED_URL`.

`Grep(pattern, filter=None, output_mode="files_with_matches", ...)` performs regex search over text fields without embeddings.

`Fetch(pk)` fetches one exact Milvus row by primary key.

`Browse(glob="*")` lists corpus entries by document-like names when the collection has suitable metadata.

`Coverage(pattern)` counts matches across documents to check whether a query is broad, narrow, or missing.

## Local Development

```powershell
git clone https://github.com/ByungjinChung/milvus-rag-plugin
cd milvus-rag-plugin
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Run the MCP server locally:

```powershell
python -m milvus_rag
```

You can also register the local server manually:

```powershell
codex mcp add corpus -- .\.venv\Scripts\python.exe -m milvus_rag
```

## Marketplace Layout

This repository is also a Codex marketplace. The marketplace manifest lives at `.agents/plugins/marketplace.json` and points to `plugins/milvus-rag`, which contains the Codex plugin metadata.

## License

MIT
