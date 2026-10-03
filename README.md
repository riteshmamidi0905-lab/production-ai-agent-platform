# Production AI Agent Platform

A research/analysis agent built as a real system, not a notebook: it **plans** a request, **calls tools** (retrieval, calculator, MCP, an approval-gated write), **retrieves evidence** with hybrid search on PostgreSQL + pgvector, **writes a cited report**, **verifies its own claims**, and is **evaluated automatically** with a regression gate. Every step streams to a live operations UI as structured events (never hidden reasoning).

```
request → guard → plan → [parallel tool steps ⇄ approvals] → evidence → synthesize → verify (repair) → cited result
                                  │                                   │
                      SSE / WebSocket event stream      Postgres: runs, events, traces, approvals, vectors
```

## Run it

**No Docker (embedded Postgres + pgvector):**
```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python scripts/dev_server.py            # indexes ./corpus, serves API + UI on http://127.0.0.1:8000
```
**Docker:**
```bash
cp .env.example .env
docker compose up -d --build db redis api && docker compose run --rm ingest
python scripts/smoke.py http://localhost:8000
```
Then open `/`. Submit a request, tick “save report” to see a consequential tool pause for your approval.

**Models.** The default `offline` provider is a *deterministic heuristic, not a language model* (rule-based planner + extractive, citation-bearing writer). It lets the whole platform run and be tested with no key. For real models set `AP_OPENAI_API_KEY` (any OpenAI-compatible endpoint), `AP_OLLAMA_*` (free, local) or `AP_ANTHROPIC_API_KEY`; add a vendor by implementing one method (`generate_structured`) and calling `register()`. Keys stay server-side.

## What is in it

| Concern | Implementation |
|---|---|
| API | FastAPI, async end to end; REST + **SSE** (`Last-Event-ID` resume) + **WebSocket**; optional API key |
| Orchestration | Explicit validated state machine (`created→guarding→planning→executing→synthesizing→verifying→completed/failed/cancelled`); independent plan steps run **concurrently**, dependent ones wait |
| Tools | Typed (Pydantic) args, risk classes (`read` / `network` / `consequential`), per-call timeout, bounded retries, metrics. Built-ins: `search_corpus`, `calculator` (AST, no `eval`), `fetch_url` (off by default, TLS + allow-list), `save_report` |
| **MCP** | JSON-RPC 2.0 server at `POST /mcp` + stdio (`initialize`, `tools/list`, `tools/call`); consequential tools are *never* exposed. MCP **client** imports external servers' tools (anything without `readOnlyHint` is treated as consequential) |
| Retrieval | Markdown-aware chunking → embeddings → **pgvector HNSW cosine** + **Postgres full-text (GIN)** in parallel → **Reciprocal Rank Fusion** → term-coverage rerank; per-passage vector / lexical / fused scores are surfaced |
| State | PostgreSQL: runs, append-only events, traces, approvals, conversations, documents/chunks, eval history. Runs survive restarts as records; in-flight runs are marked `interrupted` at startup |
| Redis | Optional: cross-worker event fan-out, approval decisions and retrieval cache; in-process fallback with the same interface |
| Guardrails | Input prompt-injection screening; retrieved text is **untrusted** (flagged passages are shown but never reach the synthesizer); secret redaction; tool allow-list; human approval for consequential actions (timeout ⇒ deny); per-run **cost budget** |
| Evidence & citations | Every claim cites evidence ids (`[e3]`); a verifier checks citation validity and claim support, makes one repair pass, then **removes** what it cannot support rather than shipping it |
| Reliability | Retries with backoff for transient provider errors, per-call timeouts, ordered **provider fallback chain**, schema-repair round-trip, invalid-plan fallback, idempotency keys, cancellation |
| Observability | Per-run **trace spans**, token usage, latency, estimated cost, tool-call metrics, **failure classification** (15 classes), Prometheus `/metrics` |
| Prompts | Versioned registry; each run records `name@version#hash` of every prompt it used |
| Evaluation | 15-case labelled dataset, 6 metric families, independent of the online verifier, regression gate with stored baseline, run from CLI, CI and the UI |

## Measured results (offline provider, this repo's own dataset)

Produced by `python -m agentplatform.cli --embedded-pg eval --repeats 2` against real Postgres/pgvector, stored in `agentplatform/eval/baselines/offline.json`:

| groundedness | citation F1 | task completion | tool success | instruction adherence | unsupported numbers | consistency | latency p50 / p95 |
|---|---|---|---|---|---|---|---|
| 1.000 | 0.882 | 0.967 | 1.000 | 1.000 | 0.000 | 1.000 | 5.6 / 10.8 ms |

**Read these honestly.** The offline provider is extractive, so groundedness is high by construction and consistency is 1.0 because it is deterministic. The informative numbers are retrieval-driven: citation F1 (0.88) and task completion (0.97, one required fact missing in `metric-count`). The dataset was used while developing the chunker/writer (three real defects were found and fixed this way), so treat these as development-set numbers, not a held-out benchmark. They say nothing about any language model's quality; run the suite with a real provider to get those numbers. Details: [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Tests

`pytest -q` → 78 tests against **real PostgreSQL 16 + pgvector 0.6** (embedded via `pgserver`, no mocks for the DB), a real uvicorn server for SSE/WebSocket, fakeredis for the Redis path, and `httpx.MockTransport` for provider HTTP. Mutation-checked: disabling approvals, the verifier or retrieval fusion each turns the suite red. CI also runs the evaluation regression gate and a Docker Compose smoke test.

## Docs
[`ARCHITECTURE.md`](docs/ARCHITECTURE.md) (diagrams) · [`DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) (trade-offs, scaling, interview Q&A) · [`EVALUATION.md`](docs/EVALUATION.md) · [`NEXT_PROJECTS.md`](docs/NEXT_PROJECTS.md) (voice agent, MAREF platform)

## Known limitations
- Real-model providers (OpenAI-compatible, Anthropic) are tested against mocked HTTP only: no live API call was made while building this.
- `Dockerfile`/`docker-compose.yml` were written but **not built** in the authoring environment (no Docker available); CI builds them.
- Redis behaviour is tested with fakeredis, not a real server.
- Embeddings default to a deterministic hashing embedder (lexical-semantic, no model download). Plug a real embedder via `ProviderEmbedder` and set `AP_EMBEDDING_DIM`.
- Run tasks live in the API process; a crash interrupts in-flight runs (recorded, not resumed). A durable queue/worker split is the next scaling step (see design doc).
- Token counts are estimated for the offline provider; costs use editable list-price estimates (`AP_PRICING_JSON`).
