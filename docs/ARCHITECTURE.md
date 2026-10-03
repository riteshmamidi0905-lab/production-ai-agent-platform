# Architecture

## Components
```mermaid
flowchart LR
  UI[Operations UI<br/>SSE EventSource] -->|REST| API
  CLI[CLI / CI eval] --> Container
  MCPc[External MCP clients] -->|JSON-RPC /mcp| API
  subgraph Process[API process]
    API[FastAPI<br/>REST · SSE · WS · /mcp · /metrics] --> Container[Container<br/>composition root]
    Container --> Engine[AgentEngine<br/>state machine]
    Engine --> Exec[ToolExecutor<br/>validate · approve · timeout · retry]
    Engine --> Prov[ProviderPool<br/>fallback chain]
    Engine --> Hub[EventHub<br/>seq · persist · fan-out]
    Exec --> Reg[ToolRegistry<br/>builtin + MCP-imported]
    Reg --> Ret[HybridRetriever]
    Exec --> Brk[ApprovalBroker]
  end
  Prov --> P1[offline heuristic]
  Prov --> P2[OpenAI-compatible / Ollama]
  Prov --> P3[Anthropic]
  Ret --> PG[(PostgreSQL 16<br/>pgvector HNSW · tsvector GIN<br/>runs · events · traces · approvals)]
  Hub --> PG
  Hub <--> Cache[(Redis or in-memory<br/>pub/sub · cache)]
  Brk <--> Cache
  Reg -.->|import tools| MCPs[External MCP servers]
```

## One run
```mermaid
sequenceDiagram
  participant U as UI/API client
  participant E as Engine
  participant L as Provider
  participant T as Tools
  participant H as Human
  U->>E: POST /v1/runs (query, constraints)
  E-->>U: 202 run_id (SSE: run.created …)
  E->>E: guarding — injection / secret screen
  E->>L: plan (prompt planner@v)
  L-->>E: Plan{steps, depends_on} (validated, else safe fallback)
  par independent steps
    E->>T: search_corpus (vector ∥ full-text → RRF)
  and
    E->>T: search_corpus (scoped to doc)
  end
  T-->>E: evidence (flagged passages excluded)
  E->>L: synthesize (evidence only)
  L-->>E: Report{claims[evidence_ids]}
  E->>E: verify → repair once → strip unsupported
  opt constraints.save_report
    E->>T: save_report (consequential)
    T->>H: approval_required (run suspended)
    H-->>T: approve / deny / timeout⇒deny
  end
  E-->>U: run.completed (usage, latency, verification)
```

## State machine
```mermaid
stateDiagram-v2
  [*] --> created
  created --> guarding
  guarding --> planning
  guarding --> failed: guardrail_block
  planning --> executing
  executing --> synthesizing
  executing --> failed: no_evidence / tool_*
  synthesizing --> verifying
  verifying --> synthesizing: repair pass
  verifying --> completed
  verifying --> failed: verification_failed
  created --> cancelled
  planning --> cancelled
  executing --> cancelled
  completed --> [*]
  failed --> [*]
  cancelled --> [*]
```
Approval waits happen *inside* `executing`/after `verifying` as events (`tool.approval_required` / `tool.approved|denied`); `interrupted` is assigned at startup to runs that were in flight when the process died.

## Data model
```mermaid
erDiagram
  documents ||--o{ chunks : has
  conversations ||--o{ messages : has
  conversations ||--o{ runs : groups
  runs ||--o{ events : appends
  runs ||--|| run_traces : traced_by
  runs ||--o{ approvals : gates
  eval_runs { text id text provider jsonb metrics jsonb cases jsonb gate }
  chunks { text id vector embedding tsvector tsv }
  runs { text state text failure_class jsonb plan jsonb report jsonb usage jsonb prompt_versions }
  events { int seq text type jsonb data }
```

## Event contract (what the UI renders)
`run.created · state.changed · guardrail.input|evidence · plan.created|fallback · step.started|completed · tool.requested|approval_required|approved|denied|completed|failed|retrying · evidence.added|pruned · usage.updated · provider.retrying|error|fallback · verification.completed · report.title|claim|limitation|final|saved|not_saved · run.completed|failed|cancelled`
Events carry a gap-free per-run `seq`; clients resume with `Last-Event-ID` / `?after=`. They contain plans, tool I/O, evidence and verdicts, never model reasoning.

## Package map
`api/` HTTP surface · `orchestrator/` engine, executor, approvals, events, verifier · `providers/` LLM adapters, pricing, registry · `retrieval/` chunking, embeddings, hybrid search, ingestion · `tools/` registry, built-ins, MCP · `storage/` schema + repo · `eval/` dataset, metrics, runner, baselines · `prompts/` versioned templates · `ui/` operations console.
