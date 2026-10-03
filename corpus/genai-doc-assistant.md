<h1 align="center">GenAI Document Assistant</h1>

<p align="center">
  <b>A full-stack, production-shaped RAG application</b><br>
  Retrieval-augmented chat · router-agent · grounded answers with citations · streaming · evaluation
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/RAG-vector%20search-4285F4">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white">
  <img src="https://img.shields.io/badge/tests-39%20passing-brightgreen">
  <img src="https://img.shields.io/badge/license-MIT-green">
</p>

A complete **Retrieval-Augmented Generation** system you can run end-to-end with
**zero credentials**: ingest documents, ask questions in a web chat UI, and get
**grounded answers with citations** — plus a router-agent that sends math to a
calculator, streaming responses, and a real **evaluation harness**. Swap the
offline reader for **Gemini / OpenAI** (and the embedder for **Vertex AI**) with
one line — the interfaces are identical.

![chat](docs/chat.png)

## Why this is more than a demo

- **Grounded, not hallucinated** — the default reader is an *extractive* LLM
  that answers **only** from retrieved context and **abstains** when the context
  doesn't support an answer. Every answer carries citations back to the source
  chunk.
- **Real RAG internals, from scratch** — sentence-aware chunking, a stateless
  **feature-hashing embedder**, a cosine **vector store** with persistence, and
  a **re-ranking retriever** that blends vector similarity with keyword overlap.
- **Agentic routing** — a `RouterAgent` decides per query: arithmetic → a safe
  **AST calculator**; everything else → RAG. The route is returned to the UI.
- **Streaming** — `/api/chat/stream` serves tokens over **Server-Sent Events**.
- **Measured quality** — an eval harness reports **retrieval hit-rate, MRR,
  faithfulness, and answer-relevance**, gated in CI.
- **Deployable** — FastAPI backend, single-page chat frontend, **Dockerfile +
  docker-compose** (app **+ Postgres/pgvector**), and a **GitHub Actions** matrix.
- **Swap offline → production without touching code** — every layer is chosen by
  an environment variable: **hashing → sentence-transformers** embeddings,
  **in-memory → pgvector** store, **extractive → Ollama/Gemini/OpenAI**
  generation. Tests and local dev stay zero-dependency; production flips env vars.

## Results (offline extractive mode, sample corpus)

```
Retrieval hit-rate : 1.00     eval set gated in CI (python -m app.eval.run)
Mean reciprocal rank: 1.00
Faithfulness       : 1.00     (answers grounded in retrieved context)
Backend tests      : 39 passing
Load test          : 123 req/s @ 50 concurrent, 0 errors, p50 286ms (1 worker)
```

The eval set (`data/eval_set.json`) runs on every push and **fails the build**
if retrieval or grounding regress. Load-test numbers are reproducible with
`k6 run docs/load_test.js` — see **[deploy/DEPLOY.md](deploy/DEPLOY.md)**.

## Quickstart

```bash
git clone https://github.com/riteshmamidi0905-lab/genai-doc-assistant.git
cd genai-doc-assistant

pip install -e "backend[dev]"
uvicorn app.main:app --app-dir backend --reload     # open http://localhost:8000/
cd backend && pytest -q                              # 29 passing tests

# or the whole thing in Docker:
docker compose up --build                            # http://localhost:8000/
```

Ask the assistant (grounded in the bundled Acme Cloud docs):

```bash
curl -s localhost:8000/api/chat -H 'content-type: application/json' \
     -d '{"message":"How is data encrypted at rest?"}' | python -m json.tool
# → "Acme Cloud encrypts all data at rest using AES-256 …"  route: rag  + citations
```

## Production backends (swap offline → real, no code change)

Every layer is selected from the environment by `app/backends.py`. The defaults
are offline and deterministic so tests and local dev need **zero credentials**;
production sets a few env vars to get real semantic retrieval, a persistent
vector database, and a generative model.

| Layer | Env var | Offline default | Production options |
| :-- | :-- | :-- | :-- |
| Embeddings | `EMBEDDING_BACKEND` | `hashing` (feature-hashing) | `sentence-transformers` (MiniLM), `vertex`, `openai` |
| Vector store | `VECTOR_BACKEND` (+ `DATABASE_URL`) | `memory` (cosine, in-process) | `pgvector` (Postgres, IVFFlat cosine index) |
| Generation | `LLM_BACKEND` | `extractive` (grounded, offline) | `ollama` (local & free), `gemini`, `openai` |

```bash
# Real semantic retrieval + persistent pgvector + a free local generative model:
pip install -e "backend[prod]"                 # sentence-transformers + pgvector + …
docker compose up -d db                         # Postgres/pgvector
ollama pull llama3.2 && ollama serve            # free local model (optional)

export EMBEDDING_BACKEND=sentence-transformers
export VECTOR_BACKEND=pgvector
export DATABASE_URL=postgresql://genai:genai@localhost:5432/genai
export LLM_BACKEND=ollama                        # or gemini / openai
uvicorn app.main:app --app-dir backend
```

`GET /api/health` reports exactly which backends are live:

```json
{ "status": "ok", "chunks": 42,
  "backends": { "embedding_backend": "sentence-transformers",
                "vector_backend": "pgvector", "llm_backend": "ollama" } }
```

The entire stack also comes up with one command via `docker compose up --build`
(app + pgvector), with the production backends pre-wired in `docker-compose.yml`.

## API

| Method | Path | Purpose |
| :-- | :-- | :-- |
| `POST` | `/api/chat` | grounded answer + route + citations + trace |
| `POST` | `/api/chat/stream` | token stream (Server-Sent Events) |
| `POST` | `/api/ingest` | add raw text to the knowledge base |
| `POST` | `/api/ingest/file` | upload a `.txt`/`.pdf`; extract → chunk → index |
| `GET`  | `/api/health` | chunk count + active backends (embedder/store/LLM) |
| `GET`  | `/metrics` | Prometheus metrics (JSON fallback if client absent) |
| `GET`  | `/` | web chat UI |

## Security & observability

Production-shaped operations, all opt-in from the environment so tests and local
dev stay open:

- **API-key auth** — set `API_KEYS=key1,key2`; protected routes then require an
  `X-API-Key` header (401 otherwise). Unset ⇒ open for local dev.
- **Rate limiting** — set `RATE_LIMIT_PER_MIN=60`; a per-caller sliding window
  returns `429` past the budget. (In-process by default; swap the `RateLimiter`
  boundary for Redis to scale horizontally.)
- **Structured logging** — every request logs JSON (`method`, `path`, `status`,
  `latency_ms`) to stdout, ready for any log pipeline.
- **Metrics** — `/metrics` exposes Prometheus counters for request volume,
  latency, answer route (rag/calculator), grounding rate, and approximate token
  usage — point Grafana at it for live dashboards. Every response also carries an
  `X-Response-Time-ms` header.

## Deploy

One live HTTPS URL, backed by persistent pgvector, on any of three hosts:

- **Render** — `render.yaml` provisions the app **and** a pgvector Postgres in
  one blueprint (simplest).
- **Fly.io** — `fly launch` + `fly postgres attach` + `fly deploy`.
- **Cloud Run** — `deploy/cloudrun.yaml` (Knative) + Cloud SQL.

Full step-by-step, env vars, and verification in
**[deploy/DEPLOY.md](deploy/DEPLOY.md)**.

## Project layout

```
backend/app/
  backends.py    config-driven selection of embedder / store / LLM
  security.py    API-key auth + per-caller rate limiting
  observability.py  structured logs · Prometheus /metrics · request middleware
  rag/         chunking · embeddings (hashing + sentence-transformers) ·
               vector_store (in-memory) · pg_vector_store (Postgres) ·
               ingest · retriever
  agent/       tools (calculator, document_search) · RouterAgent
  llm/         base · extractive (offline) · local (Ollama) · cloud (Gemini/OpenAI)
  services/    chat orchestration
  eval/        metrics (hit-rate, MRR, faithfulness, relevance) · harness ·
               run.py (CI quality gate)
  main.py      FastAPI app (chat, streaming, ingest, upload, health, metrics) + UI
backend/tests/ 39 passing tests
data/eval_set.json   evaluation set gated in CI
deploy/        DEPLOY.md · cloudrun.yaml    ·   fly.toml · render.yaml (repo root)
docs/          architecture.md · load_test.js (k6)
frontend/      single-page chat UI
data/sample_docs/  the demo knowledge base
docs/architecture.md · Dockerfile · docker-compose.yml · .github/workflows/ci.yml
```

See **[docs/architecture.md](docs/architecture.md)** for the full design.

## License

MIT © Ritesh Mamidi
