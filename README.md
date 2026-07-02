# milvus-rag

**OpenAI Codex용 범용 Agentic RAG 플러그인.**
어떤 도메인의 Milvus 컬렉션이든 코덱스에 붙여 라이브 코퍼스로 만든다. 스키마에 대한 가정이 없어서 페이지 기반 PDF든, 토큰 청크든, 웹 스크래핑 덤프든 그대로 붙는다.

---

## 노출되는 3개 툴

| 툴 | 언제 씀 | 요구 사항 |
|---|---|---|
| **`Search`** | "왜 이렇게 동작해?" 같은 자연어 질문. 개념 매칭. | 벡터 필드 + 임베딩 엔드포인트 |
| **`Grep`** | 정확한 문자열, 오류 코드, 함수명, rare token, 정규식 | 텍스트 필드만 |
| **`Fetch`** | Search/Grep 결과의 `pk`로 특정 청크 전체 조회 | 텍스트 필드만 |

**핵심**: Search는 임베딩 필요, Grep은 임베딩 없이 동작. 임베딩 엔드포인트 안 만들어도 Grep + Fetch로 통상 사용 가능.

---

## 요구사항

- **Milvus 2.4+** — 텍스트 컬럼 하나만 있으면 됨. 벡터 컬럼도 있으면 Search까지 활성화.
- **OpenAI Codex CLI** — 플러그인 지원 버전 (2026-03 이후).
- **[`uv`](https://docs.astral.sh/uv/)** — 코덱스가 파이썬 서버 부팅할 때 사용.
  - macOS: `brew install uv`
  - Windows: `winget install --id astral-sh.uv`
  - Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **(Search만 필요)** OpenAI-compatible embedding endpoint — llama.cpp / vLLM / TEI / OpenAI 등. 인덱싱 시 쓴 그 모델을 그대로 노출해야 함.

---

## 마켓플레이스 설치 (권장)

코덱스에서:

```
/plugin marketplace add ByungjinChung/milvus-rag-plugin
/plugin install milvus-rag@milvus-rag-plugin
```

두 줄이면 끝. 사용자 `~/.codex/config.toml`에 `corpus` MCP 서버 블록이 자동 등록됨. `uv`가 첫 실행 시 파이썬 + pymilvus를 격리 설치.

---

## 사용자별 Milvus 연결 설정

사용자 `~/.codex/config.toml`의 플러그인 스코프 블록을 오버라이드:

```toml
[mcp_servers.corpus.env]
MILVUS_RAG_URI = "http://milvus.internal:19530"
MILVUS_RAG_COLLECTION = "my_documents"
MILVUS_RAG_TEXT_FIELD = "text"

# 시맨틱 Search까지 쓰려면:
MILVUS_RAG_VECTOR_FIELD = "dense"
MILVUS_RAG_EMBED_URL = "http://embed.internal:8080/v1/embeddings"
MILVUS_RAG_EMBED_MODEL = "bge-m3"
MILVUS_RAG_METRIC = "COSINE"
```

또는 셸에서 export → 플러그인이 자동 forwarding (`.mcp.json`의 `env_vars`에 나열된 모든 변수).

---

## 환경변수 전체 목록

### Milvus 접속

| 변수 | 필수 | 설명 |
|---|---|---|
| `MILVUS_RAG_URI` | ★ | Milvus 서버 URI |
| `MILVUS_RAG_COLLECTION` | ★ | 컬렉션 이름 |
| `MILVUS_RAG_DB_NAME` | – | Multi-DB Milvus만 |
| `MILVUS_RAG_USER` | – | Basic auth |
| `MILVUS_RAG_PASSWORD` | – | Basic auth |
| `MILVUS_RAG_TOKEN` | – | Token auth |

### 필드 매핑

| 변수 | 필수 | 기본값 | 설명 |
|---|---|---|---|
| `MILVUS_RAG_TEXT_FIELD` | ★ | `text` | 검색·반환할 텍스트 컬럼 |
| `MILVUS_RAG_VECTOR_FIELD` | (Search만) | – | 벡터 컬럼. 비어 있으면 Search 비활성 |

### 시맨틱 Search용 임베딩

| 변수 | 필수 | 기본값 | 설명 |
|---|---|---|---|
| `MILVUS_RAG_EMBED_URL` | (Search만) | – | OpenAI-compat `/v1/embeddings` 엔드포인트 |
| `MILVUS_RAG_EMBED_MODEL` | – | – | 요청 body의 `model` 값. 서버가 요구할 때만 |
| `MILVUS_RAG_EMBED_API_KEY` | – | – | `Authorization: Bearer <key>` |
| `MILVUS_RAG_METRIC` | – | `COSINE` | 인덱싱 시 metric. `COSINE`/`IP`/`L2` |
| `MILVUS_RAG_EMBED_TIMEOUT_S` | – | `30` | 임베딩 요청 타임아웃(초) |

### 진단

| 변수 | 설명 |
|---|---|
| `MILVUS_RAG_DEBUG_LOG` | 파일 경로. 모든 JSON-RPC 호출을 append |
| `LANG=C` | 비영어 Windows 로케일에서 MCP 재접속 매칭용 (기본값) |

---

## 툴 상세

### `Search(query, k=10, filter=None)`

시맨틱 벡터 검색.

- `query` (**필수**): 자연어 쿼리
- `k` (옵션, 기본 10, 최대 100): 반환 개수
- `filter` (옵션): 원시 Milvus 표현식. 예: `source == "guide.pdf"`

**동작**: `query`를 `EMBED_URL`로 임베딩 → Milvus `search(anns_field=VECTOR_FIELD, metric=METRIC)` → top-k. 반환 각 행에 `_score` 및 `pk` 포함.

### `Grep(pattern, filter=None, output_mode="files_with_matches", ...)`

정규식 텍스트 매칭 (임베딩 불필요).

- `pattern` (**필수**): 파이썬 정규식. 문자 클래스, 그룹, 백래퍼런스, 교대 지원
- `output_mode`: `files_with_matches` (기본, pk 목록) / `content` (매치 라인) / `count`
- `-A` / `-B` / `-C` / `context`: 매치 전후 라인 컨텍스트 (content 모드)
- `-n`: 라인 번호 표시 (기본 true)
- `-i`: 대소문자 무시
- `multiline`: 여러 줄 걸친 패턴
- `head_limit` (기본 250, 최대 1000): 결과 개수 상한
- `offset`: paging용
- `filter`: Search와 동일한 Milvus 표현식

**동작**: Milvus에서 row를 스트리밍으로 받아 Python 정규식으로 매칭. 청크 단위로 필터.

### `Fetch(pk)`

- `pk` (**필수**): Search/Grep이 반환한 primary key 값

**동작**: 해당 row의 벡터 제외 모든 컬럼 반환.

---

## 인덱싱 가이드

플러그인은 인덱싱 방법을 강요하지 않습니다. **필드 하나(텍스트)만** 있으면 Grep/Fetch 동작. 벡터 컬럼까지 있으면 Search 활성.

**LangChain 예시**:

```python
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings

vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="my_documents",
    connection_args={"uri": "http://localhost:19530"},
)
```
→ 이 컬렉션은 필드가 `text` (텍스트) + `vector` (임베딩)로 자동 생성됨. 그대로 붙음:
```toml
MILVUS_RAG_TEXT_FIELD = "text"
MILVUS_RAG_VECTOR_FIELD = "vector"
MILVUS_RAG_EMBED_MODEL = "text-embedding-3-small"
MILVUS_RAG_EMBED_URL = "https://api.openai.com/v1/embeddings"
MILVUS_RAG_EMBED_API_KEY = "sk-..."
```

**로컬 BGE-M3 + llama.cpp**:

```bash
llama-server --embedding \
    -m D:/models/bge-m3-Q4_K_M.gguf \
    -c 8192 --port 8080
```

```toml
MILVUS_RAG_EMBED_URL = "http://localhost:8080/v1/embeddings"
MILVUS_RAG_EMBED_MODEL = "bge-m3"
MILVUS_RAG_METRIC = "COSINE"
```

---

## 마켓플레이스 없이 쓰려면

```powershell
git clone https://github.com/ByungjinChung/milvus-rag-plugin
cd milvus-rag-plugin
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

copy .env.example .env
# .env 편집

# 코덱스에 등록:
codex mcp add corpus -- .\.venv\Scripts\python.exe -m milvus_rag
```

---

## 라이선스

MIT

## 참고

- [OpenAI Codex 플러그인 문서](https://developers.openai.com/codex/plugins)
- [Codex 플러그인 빌드 가이드](https://developers.openai.com/codex/plugins/build)
- [Codex MCP 설정](https://developers.openai.com/codex/mcp)
- [`uv` 문서](https://docs.astral.sh/uv/)
- [Milvus 문서](https://milvus.io/docs)
