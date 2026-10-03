# Architecture

## Overview

The GenAI Document Assistant is a retrieval-augmented chat application. A user
question is routed by a lightweight agent to either a **calculator tool** or the
**RAG pipeline**, which retrieves supporting passages and produces a grounded,
cited answer.

```
                         ┌─────────────────────────── FastAPI backend ───────────────────────────┐
  Browser (chat UI) ──►  │  /api/chat            RouterAgent ─┬─ calculator ─────────► answer      │
        ▲                │  /api/chat/stream (SSE)            │                                    │
        │                │                                    └─ RAG ─► Retriever ─► Reader LLM ──┐ │
        └── answer + ────┤  /api/ingest                              (VectorStore)   (Extractive/ │ │
            citations    │  /api/health                                               Gemini/OpenAI)│
                         └───────────────────────────────────────────────────────────────────────┘
```

## Components

**RAG pipeline** (`app/rag/`)
- `chunking` — sentence-aware, overlapping chunks (~480 chars).
- `embeddings` — `HashingEmbedder` (stateless, offline, signed feature hashing
  over word + char-trigram tokens, L2-normalised) or `CloudEmbedder` (Vertex AI).
- `vector_store` — in-memory cosine search with JSON persistence; the interface
  mirrors FAISS / pgvector / Pinecone.
- `retriever` — over-fetch then blend vector score with keyword overlap
  (re-ranking) for better precision@k.

**Agent** (`app/agent/`)
- `RouterAgent` decides per query: arithmetic → `calculator` (safe AST eval);
  otherwise → RAG. Tools include a `document_search` tool over the retriever.

**Reader LLM** (`app/llm/`)
- `ExtractiveLLM` — deterministic, offline, **grounded**: ranks context
  sentences against the question, composes an answer, attaches citations, and
  **abstains** when the context is insufficient (no hallucination).
- `GeminiLLM` / `OpenAILLM` — generative backends behind the same interface.

**Orchestration** (`app/services/chat.py`)
- Ties store + retriever + agent + reader into one `answer()` / `stream()` call
  returning answer, route, citations, and trace.

**Evaluation** (`app/eval/`)
- Retrieval hit-rate & MRR, plus faithfulness and answer-relevance — no LLM
  judge required, so quality is measured deterministically in CI.

**API** (`app/main.py`) — `/api/chat`, `/api/chat/stream` (SSE),
`/api/ingest`, `/api/health`, and the static chat UI at `/`.

## Design choices

- **Grounded by default** — the offline reader only answers from retrieved
  context and abstains otherwise, so the app is safe to demo without a
  hallucinating model.
- **Everything swappable** — embedder, reader LLM, and vector store are all
  interfaces; the offline defaults keep the app runnable with zero credentials,
  and the cloud implementations drop in for production.
- **Measurable** — the eval harness turns "seems good" into hit-rate,
  faithfulness, and relevance numbers gated in CI.
