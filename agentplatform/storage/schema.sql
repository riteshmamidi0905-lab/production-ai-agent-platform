CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id          text PRIMARY KEY,
  title       text NOT NULL,
  source      text NOT NULL DEFAULT 'corpus',
  content_sha text NOT NULL,
  meta        jsonb NOT NULL DEFAULT '{}',
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id        text PRIMARY KEY,
  doc_id    text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal   int  NOT NULL,
  text      text NOT NULL,
  embedding vector({dim}) NOT NULL,
  tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin(tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS conversations (
  id         text PRIMARY KEY,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS messages (
  id              bigserial PRIMARY KEY,
  conversation_id text NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role            text NOT NULL,
  content         text NOT NULL,
  run_id          text,
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
  id               text PRIMARY KEY,
  conversation_id  text REFERENCES conversations(id) ON DELETE SET NULL,
  query            text NOT NULL,
  provider         text NOT NULL,
  state            text NOT NULL,
  constraints      jsonb NOT NULL,
  plan             jsonb,
  report           jsonb,
  verification     jsonb,
  usage            jsonb NOT NULL DEFAULT '{}',
  latency_ms       double precision,
  failure_class    text,
  error            text,
  prompt_versions  jsonb NOT NULL DEFAULT '{}',
  tool_calls       jsonb NOT NULL DEFAULT '[]',
  idempotency_key  text UNIQUE,
  created_at       double precision NOT NULL,
  finished_at      double precision
);
CREATE INDEX IF NOT EXISTS runs_created_idx ON runs(created_at DESC);

CREATE TABLE IF NOT EXISTS events (
  run_id text NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  seq    int  NOT NULL,
  type   text NOT NULL,
  ts     double precision NOT NULL,
  data   jsonb NOT NULL DEFAULT '{}',
  PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS run_traces (
  run_id text PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
  spans  jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
  id           text PRIMARY KEY,
  run_id       text NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  tool         text NOT NULL,
  args         jsonb NOT NULL,
  risk         text NOT NULL,
  status       text NOT NULL DEFAULT 'pending',
  requested_at double precision NOT NULL,
  decided_at   double precision,
  reason       text
);
CREATE INDEX IF NOT EXISTS approvals_run_idx ON approvals(run_id);

CREATE TABLE IF NOT EXISTS eval_runs (
  id         text PRIMARY KEY,
  created_at double precision NOT NULL,
  dataset    text NOT NULL,
  provider   text NOT NULL,
  config     jsonb NOT NULL,
  metrics    jsonb NOT NULL,
  cases      jsonb NOT NULL,
  gate       jsonb
);
