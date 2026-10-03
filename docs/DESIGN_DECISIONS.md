# Design decisions, trade-offs and scaling (interview notes)

## Why these choices
- **Explicit state machine over a free-form agent loop.** Allowed transitions are a table; illegal ones raise. Every transition is an event. That makes runs resumable-in-principle, debuggable, testable and gives clear failure classes. Cost: less flexible than an open ReAct loop; mitigated by a plan with dependencies and parallel steps.
- **Plan-then-execute with a validated, fallback-able plan.** Plans are structured (`depends_on` DAG, tool names checked against the registry, acyclicity checked). A bad plan degrades to a safe single-retrieval plan instead of failing the run. Trade-off: up-front planning can't adapt mid-run; an observe-and-replan loop is a natural extension (planner is already a prompt + schema).
- **Structured outputs everywhere (Pydantic).** Providers return validated `Plan`/`Report` or raise `SchemaError`; the OpenAI adapter does one *repair round-trip* with the validation error. Prevents silent garbage reaching the verifier.
- **Postgres for everything stateful, including vectors.** One system to back up and reason about; transactions across chunks/documents; `pgvector` HNSW + `tsvector` GIN give hybrid search without a second datastore. Trade-off: a dedicated vector DB wins at 10⁸+ vectors or exotic filtering; at the target scale (≤ tens of millions) Postgres is simpler and cheaper.
- **Hybrid retrieval with RRF.** Dense and lexical failures are different (paraphrase vs exact identifier). RRF needs no score calibration between legs. A light coverage rerank adds precision. Both legs run concurrently.
- **Events are the product surface.** Persist-then-publish with a per-run sequence; SSE/WS replay from the DB so reconnects are lossless. The UI is just a reducer over events, so anything the UI shows is auditable.
- **Verification is separate from generation, and separate from evaluation.** Online verifier: citation validity + claim support, one repair, then *strip*. Offline evaluator: stricter claim rule (+ number grounding) written independently so the system isn't graded by its own checker.
- **Approvals are part of execution, not UI sugar.** Consequential tools suspend the step; decisions travel over pub/sub (so any API replica can answer); timeout means deny. `save_report` is a *finalisation* action the planner cannot schedule.
- **Untrusted retrieval.** Retrieved text is data: injection patterns are flagged, shown to the user, and withheld from the synthesizer. This is a mitigation, not a proof; a determined attacker needs layered defences (see below).
- **Cost control.** Per-run budget stop, per-provider pricing table, usage on every call, fallback to cheaper/local providers.

## Failure handling matrix
| Failure | Handling | Class |
|---|---|---|
| Provider 429/5xx/timeout | backoff retries → next provider in chain | `provider_error` / `provider_timeout` |
| Model returns bad JSON | one repair round-trip → fallback | `schema_validation` |
| Invalid plan | `plan.fallback` safe plan | `invalid_plan` (event) |
| Tool timeout / error | retry if retryable, step marked failed; run continues with other evidence | `tool_timeout` / `tool_error` |
| Bad tool args | rejected before execution | `tool_args_invalid` |
| Approval denied / expired | optional action skipped; verified report kept | `approval_denied` / `approval_timeout` |
| Unsupported claims | repair → strip → fail if nothing survives | `verification_failed` |
| Budget exceeded | stop immediately | `budget_exceeded` |
| Process crash | run marked `interrupted` at startup | — |

## Scaling path (what I would do next, in order)
1. **Split API and workers**: enqueue runs (Postgres `SKIP LOCKED` or a broker), stateless workers, leases + heartbeats so in-flight runs resume instead of being marked interrupted. The event hub + approval pub/sub already work across processes via Redis.
2. **Provider streaming**: stream tokens for the synthesizer (the event contract already has `report.*`).
3. **Real embeddings + reranker** (cross-encoder), embedding cache keyed by content hash, `ef_search` tuning, partitioned `chunks` per tenant.
4. **Multi-tenancy & authz**: per-tenant corpora, scoped API keys/JWT, row-level security; per-tenant budgets and rate limits.
5. **OpenTelemetry export** of the existing span model; alerting on `agent_failures_total{failure_class}` and budget burn.
6. **Replan loop** (observe → decide) with a step budget.

## Likely interview questions
- *How do you stop prompt injection via retrieved docs?* Untrusted-data handling + flagging + exclusion + approval gate on side-effecting tools + tool allow-lists + no tool can be planned that writes. Residual risk: subtle semantic injection; mitigations are least-privilege tools and human approval.
- *How do you know it works?* Labelled dataset, independent metrics, regression gate in CI; but be explicit: the offline provider only validates the plumbing, and the dataset was a dev set.
- *What breaks at 100× load?* Single-process run tasks, Postgres connection pool, HNSW build/recall settings, embedding throughput. See scaling path.
- *Exactly-once?* Events have a unique `(run_id, seq)`; idempotency keys dedupe submissions; approvals can be decided once (conditional UPDATE).
- *Why not LangChain/LangGraph?* The value here is the explicit contracts (state, events, approvals, evals); thin purpose-built code keeps them visible and testable. The provider/tool interfaces are small enough to adapt either way.
