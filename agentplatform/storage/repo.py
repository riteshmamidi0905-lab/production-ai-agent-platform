"""Persistence for runs, events, approvals, conversations, documents and evaluation history."""
from __future__ import annotations

import time
from typing import Any

from ..models import Event, RunRecord, RunState
from .db import J, Database


class Repo:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---- runs ----
    async def create_run(self, r: RunRecord, idempotency_key: str | None) -> bool:
        n = await self.db.execute(
            """INSERT INTO runs(id, conversation_id, query, provider, state, constraints, usage, prompt_versions, tool_calls, idempotency_key, created_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
            r.id, r.conversation_id, r.query, r.provider, r.state.value, J(r.constraints.model_dump()), J(r.usage.model_dump()), J({}), J([]), idempotency_key, r.created_at)
        return n == 1

    async def run_by_key(self, key: str) -> str | None:
        row = await self.db.one("SELECT id FROM runs WHERE idempotency_key=%s", key)
        return row["id"] if row else None

    async def save_run(self, r: RunRecord) -> None:
        await self.db.execute(
            """UPDATE runs SET state=%s, plan=%s, report=%s, verification=%s, usage=%s, latency_ms=%s, failure_class=%s, error=%s,
               prompt_versions=%s, tool_calls=%s, finished_at=%s WHERE id=%s""",
            r.state.value, J(r.plan.model_dump()) if r.plan else None, J(r.report.model_dump()) if r.report else None,
            J(r.verification.model_dump()) if r.verification else None, J(r.usage.model_dump()), r.latency_ms,
            r.failure_class.value if r.failure_class else None, r.error, J(r.prompt_versions), J([t.model_dump() for t in r.tool_calls]), r.finished_at, r.id)

    @staticmethod
    def _run(row: dict[str, Any]) -> RunRecord:
        return RunRecord.model_validate({**row, "constraints": row["constraints"], "tool_calls": row["tool_calls"] or [], "usage": row["usage"] or {}})

    async def get_run(self, run_id: str) -> RunRecord | None:
        row = await self.db.one("SELECT * FROM runs WHERE id=%s", run_id)
        return self._run(row) if row else None

    async def list_runs(self, limit: int = 50) -> list[RunRecord]:
        return [self._run(r) for r in await self.db.fetch("SELECT * FROM runs ORDER BY created_at DESC LIMIT %s", limit)]

    async def recover_interrupted(self) -> int:
        """Runs that were mid-flight when the process died cannot be resumed (in-memory approvals/tasks are gone): mark them so
        clients stop waiting and the cause is explicit."""
        n = await self.db.execute("UPDATE runs SET state=%s, error='process restarted while run was in flight', finished_at=%s WHERE state NOT IN ('completed','failed','cancelled','interrupted')",
                                  RunState.INTERRUPTED.value, time.time())
        await self.db.execute("UPDATE approvals SET status='expired', decided_at=%s WHERE status='pending'", time.time())
        return n

    # ---- events / traces ----
    async def append_event(self, e: Event) -> None:
        await self.db.execute("INSERT INTO events(run_id, seq, type, ts, data) VALUES (%s,%s,%s,%s,%s)", e.run_id, e.seq, e.type, e.ts, J(e.data))

    async def events_after(self, run_id: str, after: int = 0) -> list[Event]:
        rows = await self.db.fetch("SELECT * FROM events WHERE run_id=%s AND seq>%s ORDER BY seq", run_id, after)
        return [Event.model_validate(r) for r in rows]

    async def save_trace(self, run_id: str, spans: list[dict]) -> None:
        await self.db.execute("INSERT INTO run_traces(run_id, spans) VALUES (%s,%s) ON CONFLICT (run_id) DO UPDATE SET spans=EXCLUDED.spans", run_id, J(spans))

    async def get_trace(self, run_id: str) -> list[dict] | None:
        row = await self.db.one("SELECT spans FROM run_traces WHERE run_id=%s", run_id)
        return row["spans"] if row else None

    # ---- approvals ----
    async def create_approval(self, aid: str, run_id: str, tool: str, args: dict, risk: str) -> None:
        await self.db.execute("INSERT INTO approvals(id, run_id, tool, args, risk, requested_at) VALUES (%s,%s,%s,%s,%s,%s)", aid, run_id, tool, J(args), risk, time.time())

    async def decide_approval(self, aid: str, status: str, reason: str | None) -> dict | None:
        rows = await self.db.fetch("UPDATE approvals SET status=%s, decided_at=%s, reason=%s WHERE id=%s AND status='pending' RETURNING *", status, time.time(), reason, aid)
        return rows[0] if rows else None

    async def get_approval(self, aid: str) -> dict | None:
        return await self.db.one("SELECT * FROM approvals WHERE id=%s", aid)

    async def run_approvals(self, run_id: str) -> list[dict]:
        return await self.db.fetch("SELECT * FROM approvals WHERE run_id=%s ORDER BY requested_at", run_id)

    # ---- conversations ----
    async def ensure_conversation(self, cid: str) -> None:
        await self.db.execute("INSERT INTO conversations(id) VALUES (%s) ON CONFLICT DO NOTHING", cid)

    async def add_message(self, cid: str, role: str, content: str, run_id: str | None) -> None:
        await self.db.execute("INSERT INTO messages(conversation_id, role, content, run_id) VALUES (%s,%s,%s,%s)", cid, role, content, run_id)

    async def messages(self, cid: str, limit: int = 8) -> list[dict]:
        rows = await self.db.fetch("SELECT role, content, run_id FROM messages WHERE conversation_id=%s ORDER BY id DESC LIMIT %s", cid, limit)
        return list(reversed(rows))

    # ---- documents ----
    async def doc_sha(self, doc_id: str) -> str | None:
        row = await self.db.one("SELECT content_sha FROM documents WHERE id=%s", doc_id)
        return row["content_sha"] if row else None

    async def list_documents(self) -> list[dict]:
        return await self.db.fetch("SELECT d.id, d.title, d.source, d.meta, count(c.id)::int AS chunks FROM documents d LEFT JOIN chunks c ON c.doc_id=d.id GROUP BY d.id ORDER BY d.id")

    async def get_chunk(self, chunk_id: str) -> dict | None:
        return await self.db.one("SELECT c.id, c.doc_id, c.ordinal, c.text, d.title FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.id=%s", chunk_id)

    # ---- eval history ----
    async def save_eval(self, rec: dict) -> None:
        await self.db.execute("INSERT INTO eval_runs(id, created_at, dataset, provider, config, metrics, cases, gate) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                              rec["id"], rec["created_at"], rec["dataset"], rec["provider"], J(rec["config"]), J(rec["metrics"]), J(rec["cases"]), J(rec.get("gate")))

    async def list_evals(self, limit: int = 20) -> list[dict]:
        return await self.db.fetch("SELECT id, created_at, dataset, provider, metrics, gate, config FROM eval_runs ORDER BY created_at DESC LIMIT %s", limit)

    async def get_eval(self, eid: str) -> dict | None:
        return await self.db.one("SELECT * FROM eval_runs WHERE id=%s", eid)
