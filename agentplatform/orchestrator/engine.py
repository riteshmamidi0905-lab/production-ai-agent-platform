"""The agent engine: an explicit, validated state machine
   created → guarding → planning → executing → synthesizing → verifying → (repair pass) → completed | failed | cancelled.
Every transition, plan, tool call, piece of evidence, usage update and verdict is emitted as a structured event (no hidden reasoning)."""
from __future__ import annotations

import asyncio
import time

from ..cache import Cache
from ..config import Settings
from ..guardrails import check_input, redact, screen_evidence
from ..models import (Evidence, FailureClass, Plan, PlanStep, Report, RunRecord, RunRequest, RunState, Verification, new_id)
from ..prompts import RenderedPrompt, render
from ..providers.base import ProviderError, ProviderResult, SchemaError, T, with_retries
from ..providers.registry import ProviderPool
from ..retrieval.hybrid import HybridRetriever
from ..storage.repo import Repo
from ..telemetry import METRICS, Tracer
from ..tools.base import ToolContext, ToolRegistry
from .approvals import ApprovalBroker
from .events import EventHub
from .executor import ToolExecutor, ToolFailure
from .verify import strip_unsupported, verify

S = RunState
ALLOWED: dict[RunState, set[RunState]] = {
    S.CREATED: {S.GUARDING, S.CANCELLED, S.FAILED},
    S.GUARDING: {S.PLANNING, S.FAILED, S.CANCELLED},
    S.PLANNING: {S.EXECUTING, S.FAILED, S.CANCELLED},
    S.EXECUTING: {S.SYNTHESIZING, S.FAILED, S.CANCELLED},
    S.SYNTHESIZING: {S.VERIFYING, S.FAILED, S.CANCELLED},
    S.VERIFYING: {S.SYNTHESIZING, S.COMPLETED, S.FAILED, S.CANCELLED},
}


class RunFailure(Exception):
    def __init__(self, cls: FailureClass, message: str):
        super().__init__(message)
        self.cls = cls


class AgentEngine:
    def __init__(self, *, settings: Settings, repo: Repo, retriever: HybridRetriever, providers: ProviderPool, registry: ToolRegistry,
                 cache: Cache, hub: EventHub, broker: ApprovalBroker) -> None:
        self.s, self.repo, self.retriever, self.providers, self.registry = settings, repo, retriever, providers, registry
        self.cache, self.hub, self.broker = cache, hub, broker
        self.executor = ToolExecutor(registry, settings, broker)
        self.tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------ public
    async def submit(self, req: RunRequest) -> RunRecord:
        if req.idempotency_key and (existing := await self.repo.run_by_key(req.idempotency_key)):
            return await self.repo.get_run(existing)  # type: ignore[return-value]
        provider = req.provider or self.s.default_provider
        self.providers.get(provider)                      # fail fast (400) if unknown / unconfigured
        rec = RunRecord(id=new_id("run"), conversation_id=req.conversation_id, query=req.query, provider=provider, state=S.CREATED,
                        constraints=req.constraints, created_at=time.time())
        if req.conversation_id:
            await self.repo.ensure_conversation(req.conversation_id)
        if not await self.repo.create_run(rec, req.idempotency_key):
            return await self.repo.get_run(await self.repo.run_by_key(req.idempotency_key))  # type: ignore[arg-type]
        await self.hub.emit(rec.id, "run.created", query=rec.query, provider=provider, constraints=rec.constraints.model_dump())
        self.tasks[rec.id] = asyncio.create_task(self._run(rec), name=rec.id)
        self.tasks[rec.id].add_done_callback(lambda _t, rid=rec.id: self.tasks.pop(rid, None))
        return rec

    async def cancel(self, run_id: str) -> bool:
        t = self.tasks.get(run_id)
        if not t:
            return False
        t.cancel()
        return True

    async def wait(self, run_id: str) -> None:
        t = self.tasks.get(run_id)
        if t:
            await asyncio.wait([t])

    # ------------------------------------------------------------------ state machine
    async def _state(self, rec: RunRecord, new: RunState) -> None:
        if new not in ALLOWED.get(rec.state, set()):
            raise RuntimeError(f"illegal transition {rec.state.value} → {new.value}")
        old, rec.state = rec.state, new
        await self.repo.save_run(rec)
        await self.hub.emit(rec.id, "state.changed", **{"from": old.value, "to": new.value})

    async def _run(self, rec: RunRecord) -> None:
        tracer, t0 = Tracer(), time.perf_counter()
        emit = lambda type, **d: self.hub.emit(rec.id, type, **d)  # noqa: E731
        try:
            async with tracer.span("run", run_id=rec.id, provider=rec.provider):
                await self._guard(rec)
                plan = await self._plan(rec, tracer)
                evidence = await self._execute(rec, plan, tracer)
                report = await self._synthesize(rec, evidence, tracer)
                await self._finalize_actions(rec, report, tracer)
            rec.state, rec.latency_ms, rec.finished_at = S.COMPLETED, (time.perf_counter() - t0) * 1000, time.time()
            await self.repo.save_run(rec)
            METRICS.inc("agent_runs_total", status="completed", provider=rec.provider)
            METRICS.observe("agent_run_latency_ms", rec.latency_ms, provider=rec.provider)
            await self._remember(rec)
            await emit("run.completed", latency_ms=round(rec.latency_ms, 1), usage=rec.usage.model_dump(), verification=rec.verification.model_dump() if rec.verification else None,
                       tool_calls=len(rec.tool_calls), report_title=rec.report.title if rec.report else None)
        except RunFailure as f:
            await self._fail(rec, f.cls, str(f), t0)
        except asyncio.CancelledError:
            rec.state, rec.failure_class, rec.finished_at, rec.latency_ms = S.CANCELLED, FailureClass.CANCELLED, time.time(), (time.perf_counter() - t0) * 1000
            await asyncio.shield(self.repo.save_run(rec))
            METRICS.inc("agent_runs_total", status="cancelled", provider=rec.provider)
            await asyncio.shield(emit("run.cancelled"))
        except Exception as exc:  # noqa: BLE001 - last line of defence: a bug must still end the run cleanly
            await self._fail(rec, FailureClass.INTERNAL, f"{type(exc).__name__}: {exc}"[:400], t0)
        finally:
            await self.repo.save_trace(rec.id, [s.to_dict() for s in tracer.spans])

    async def _fail(self, rec: RunRecord, cls: FailureClass, msg: str, t0: float) -> None:
        rec.state, rec.failure_class, rec.error = S.FAILED, cls, msg
        rec.finished_at, rec.latency_ms = time.time(), (time.perf_counter() - t0) * 1000
        await self.repo.save_run(rec)
        METRICS.inc("agent_runs_total", status="failed", provider=rec.provider)
        METRICS.inc("agent_failures_total", failure_class=cls.value)
        await self.hub.emit(rec.id, "run.failed", failure_class=cls.value, error=msg, latency_ms=round(rec.latency_ms, 1), usage=rec.usage.model_dump())

    # ------------------------------------------------------------------ stages
    async def _guard(self, rec: RunRecord) -> None:
        await self._state(rec, S.GUARDING)
        g = check_input(rec.query)
        await self.hub.emit(rec.id, "guardrail.input", allowed=g.allowed, reasons=g.reasons)
        if not g.allowed:
            raise RunFailure(FailureClass.GUARDRAIL_BLOCK, "request blocked by input guardrails: " + "; ".join(g.reasons))

    async def _llm(self, rec: RunRecord, prompt: RenderedPrompt, schema: type[T], tracer: Tracer) -> ProviderResult[T]:
        """Provider call with timeout, transient-error retries, then fallback through the configured chain; tracks usage/cost/budget."""
        chain = self.providers.chain(rec.provider)
        last: Exception | None = None
        rec.prompt_versions[prompt.name] = prompt.ref
        for i, prov in enumerate(chain):
            try:
                async with tracer.span(f"llm:{prompt.name}", provider=prov.name, model=prov.model, prompt=prompt.ref) as sp:
                    async def once() -> ProviderResult[T]:
                        async with asyncio.timeout(self.s.provider_timeout_s):
                            return await prov.generate_structured(prompt, schema)

                    def on_retry(n: int, exc: Exception) -> None:
                        asyncio.get_running_loop().create_task(self.hub.emit(rec.id, "provider.retrying", provider=prov.name, attempt=n, error=str(exc)[:160]))
                    res = await with_retries(once, retries=self.s.provider_retries, on_retry=on_retry)
                    sp.attrs.update(prompt_tokens=res.usage.prompt_tokens, completion_tokens=res.usage.completion_tokens, cost_usd=res.usage.cost_usd)
                rec.usage.add(res.usage)
                METRICS.record_usage(prov.name, res.usage)
                METRICS.observe("agent_llm_latency_ms", res.latency_ms, provider=prov.name, prompt=prompt.name)
                await self.hub.emit(rec.id, "usage.updated", provider=prov.name, model=res.model, prompt=prompt.ref, call_latency_ms=round(res.latency_ms, 1), **res.usage.model_dump(), total=rec.usage.model_dump())
                if rec.usage.cost_usd > self.s.run_budget_usd:
                    raise RunFailure(FailureClass.BUDGET_EXCEEDED, f"estimated cost ${rec.usage.cost_usd:.4f} exceeded budget ${self.s.run_budget_usd:.2f}")
                return res
            except RunFailure:
                raise
            except (ProviderError, TimeoutError) as exc:
                last = exc
                kind = "timeout" if isinstance(exc, TimeoutError) else "schema" if isinstance(exc, SchemaError) else "error"
                METRICS.inc("agent_provider_errors_total", provider=prov.name, kind=kind)
                await self.hub.emit(rec.id, "provider.error", provider=prov.name, kind=kind, error=str(exc)[:200])
                if i + 1 < len(chain):
                    await self.hub.emit(rec.id, "provider.fallback", **{"from": prov.name, "to": chain[i + 1].name})
        cls = FailureClass.PROVIDER_TIMEOUT if isinstance(last, TimeoutError) else FailureClass.SCHEMA_VALIDATION if isinstance(last, SchemaError) else FailureClass.PROVIDER_ERROR
        raise RunFailure(cls, f"all providers failed: {last}")

    async def _context(self, rec: RunRecord) -> str:
        if not rec.conversation_id:
            return ""
        msgs = await self.repo.messages(rec.conversation_id, 6)
        return "\n".join(f"{m['role']}: {m['content'][:300]}" for m in msgs)

    @staticmethod
    def _plan_valid(plan: Plan, registry: ToolRegistry, max_steps: int) -> str | None:
        if len(plan.steps) > max_steps:
            return f"plan has {len(plan.steps)} steps (max {max_steps})"
        ids = [s.id for s in plan.steps]
        if len(set(ids)) != len(ids):
            return "duplicate step ids"
        for st in plan.steps:
            if registry.get(st.tool) is None or st.tool == "save_report":
                return f"step {st.id} uses unavailable tool '{st.tool}'"
            if any(d not in ids or d == st.id for d in st.depends_on):
                return f"step {st.id} has invalid dependencies"
        done: set[str] = set()                              # acyclic? repeatedly peel steps whose deps are satisfied
        remaining = list(plan.steps)
        while remaining:
            ready = [s for s in remaining if all(d in done for d in s.depends_on)]
            if not ready:
                return "dependency cycle"
            done |= {s.id for s in ready}
            remaining = [s for s in remaining if s.id not in done]
        return None

    async def _plan(self, rec: RunRecord, tracer: Tracer) -> Plan:
        await self._state(rec, S.PLANNING)
        docs = [{"doc_id": d["id"], "title": d["title"]} for d in await self.repo.list_documents()]
        max_steps = min(self.s.max_plan_steps, rec.constraints.max_steps)
        prompt = render("planner", query=rec.query, max_steps=max_steps, tools=self.registry.for_planner(), tools_json=self.registry.for_planner(), corpus=docs, corpus_json=docs[:40], context=await self._context(rec))
        async with tracer.span("plan"):
            try:
                plan = (await self._llm(rec, prompt, Plan, tracer)).parsed
                problem = self._plan_valid(plan, self.registry, max_steps)
            except RunFailure as f:
                if f.cls in (FailureClass.BUDGET_EXCEEDED,):
                    raise
                plan, problem = None, f"planner unavailable: {f}"
            if problem:                                     # fallback: a safe single-retrieval plan keeps the run useful
                await self.hub.emit(rec.id, "plan.fallback", reason=problem)
                plan = Plan(summary="Fallback plan: one retrieval over the full request, then a cited report.",
                            steps=[PlanStep(id="s1", goal="Retrieve evidence for the request", tool="search_corpus", args={"query": rec.query})])
        rec.plan = plan
        await self.repo.save_run(rec)
        await self.hub.emit(rec.id, "plan.created", plan=plan.model_dump())
        return plan

    async def _execute(self, rec: RunRecord, plan: Plan, tracer: Tracer) -> dict[str, Evidence]:
        await self._state(rec, S.EXECUTING)
        evidence: dict[str, Evidence] = {}
        by_chunk: dict[str, str] = {}
        failures: list[ToolFailure] = []
        done: set[str] = set()
        remaining = list(plan.steps)
        ctx = ToolContext(run_id=rec.id, retriever=self.retriever, settings=self.s)
        async with tracer.span("execute"):
            while remaining:
                layer = [s for s in remaining if all(d in done for d in s.depends_on)]

                async def run_step(st: PlanStep):
                    await self.hub.emit(rec.id, "step.started", step_id=st.id, goal=st.goal, tool=st.tool)
                    try:
                        res, tc = await self.executor.call(st.tool, st.args, ctx, emit=lambda t, **d: self.hub.emit(rec.id, t, **d), tracer=tracer,
                                                           approval_mode=rec.constraints.require_approval_for, step_id=st.id)
                        rec.tool_calls.append(tc)
                        await self.hub.emit(rec.id, "step.completed", step_id=st.id, status="ok")
                        return st, res
                    except ToolFailure as tf:
                        rec.tool_calls.append(tf.record); failures.append(tf)
                        await self.hub.emit(rec.id, "step.completed", step_id=st.id, status=tf.record.status)
                        return st, None
                results = await asyncio.gather(*(run_step(s) for s in layer))
                for st, res in results:                     # deterministic ordering: plan order, then rank
                    done.add(st.id)
                    if not res:
                        continue
                    for item in res.evidence:
                        if item.chunk_id in by_chunk:
                            ex = evidence[by_chunk[item.chunk_id]]
                            ex.fused_score = max(ex.fused_score, item.fused_score)
                            continue
                        eid = f"e{len(evidence) + 1}"
                        flags = screen_evidence(item.text)
                        ev = Evidence(id=eid, doc_id=item.doc_id, chunk_id=item.chunk_id, title=item.title, text=item.text, source=item.source, vector_score=item.vector_score,
                                      lexical_score=item.lexical_score, fused_score=item.fused_score, step_id=st.id, untrusted_flags=flags)
                        evidence[eid] = ev; by_chunk[item.chunk_id] = eid
                        await self.hub.emit(rec.id, "evidence.added", **ev.model_dump())
                        if flags:
                            await self.hub.emit(rec.id, "guardrail.evidence", evidence_id=eid, flags=flags, action="excluded from synthesis")
                remaining = [s for s in remaining if s.id not in done]
        usable = {k: v for k, v in evidence.items() if not v.untrusted_flags}
        if not usable:
            cls = failures[-1].failure if failures else FailureClass.NO_EVIDENCE
            raise RunFailure(cls, "no usable evidence: " + (str(failures[-1]) if failures else "retrieval returned nothing"))
        if len(usable) > self.s.max_evidence:
            keep = dict(sorted(usable.items(), key=lambda kv: -kv[1].fused_score)[: self.s.max_evidence])
            await self.hub.emit(rec.id, "evidence.pruned", kept=sorted(keep), dropped=sorted(set(usable) - set(keep)))
            usable = keep
        return usable

    async def _synthesize(self, rec: RunRecord, evidence: dict[str, Evidence], tracer: Tracer) -> Report:
        block = "\n\n".join(f"[{e.id}] ({e.title}) {e.text}" for e in evidence.values())
        feedback = ""
        report: Report | None = None
        ver: Verification | None = None
        async with tracer.span("synthesize_verify"):
            for attempt in range(2):
                await self._state(rec, S.SYNTHESIZING)
                prompt = render("synthesizer", query=rec.query, evidence_block=block, constraints_json=rec.constraints.model_dump(), feedback=feedback or "none",
                                evidence=[e.model_dump() for e in evidence.values()], constraints=rec.constraints.model_dump())
                report = (await self._llm(rec, prompt, Report, tracer)).parsed
                report = report.model_copy(update={"answer": redact(report.answer)})
                await self._state(rec, S.VERIFYING)
                ver = verify(report, evidence, require_citations=rec.constraints.require_citations)
                await self.hub.emit(rec.id, "verification.completed", attempt=attempt + 1, **ver.model_dump())
                if ver.ok or (not report.claims and report.limitations):
                    break
                feedback = "Unsupported or badly cited claims: " + " | ".join(ver.unsupported_claims[:5]) + "; invalid citations: " + ", ".join(ver.invalid_citations)
            assert report is not None and ver is not None
            abstained = not report.claims and bool(report.limitations)
            if not ver.ok and not abstained:                # repair failed: remove what cannot be supported, never ship it
                report = strip_unsupported(report, ver.unsupported_claims, ver.invalid_citations)
                ver = verify(report, evidence, require_citations=rec.constraints.require_citations)
                ver.repaired = True
                await self.hub.emit(rec.id, "verification.completed", attempt=3, **ver.model_dump())
                if not ver.ok:
                    if not report.claims:
                        raise RunFailure(FailureClass.VERIFICATION_FAILED, "no claim survived verification")
            if abstained:
                ver = Verification(ok=True, groundedness=1.0, citation_validity=1.0, notes=["abstained: evidence did not address the request"])
        rec.report, rec.verification = report, ver
        await self.repo.save_run(rec)
        await self.hub.emit(rec.id, "report.title", title=report.title)
        for c in report.claims:
            await self.hub.emit(rec.id, "report.claim", text=c.text, evidence_ids=c.evidence_ids)
        for l in report.limitations:
            await self.hub.emit(rec.id, "report.limitation", text=l)
        await self.hub.emit(rec.id, "report.final", answer=report.answer, title=report.title)
        return report

    async def _finalize_actions(self, rec: RunRecord, report: Report, tracer: Tracer) -> None:
        if not rec.constraints.save_report or not rec.report:
            return
        ctx = ToolContext(run_id=rec.id, retriever=self.retriever, settings=self.s)
        try:
            res, tc = await self.executor.call("save_report", {"title": report.title, "markdown": report.answer}, ctx, emit=lambda t, **d: self.hub.emit(rec.id, t, **d), tracer=tracer,
                                               approval_mode=rec.constraints.require_approval_for, step_id="final")
            rec.tool_calls.append(tc)
            await self.hub.emit(rec.id, "report.saved", **res.data)
        except ToolFailure as tf:                           # a denied/failed optional action never discards a verified report
            rec.tool_calls.append(tf.record)
            await self.hub.emit(rec.id, "report.not_saved", reason=tf.failure.value, error=str(tf)[:200])

    async def _remember(self, rec: RunRecord) -> None:
        if rec.conversation_id and rec.report:
            await self.repo.add_message(rec.conversation_id, "user", rec.query, rec.id)
            await self.repo.add_message(rec.conversation_id, "assistant", f"{rec.report.title}: " + " ".join(c.text for c in rec.report.claims[:3]), rec.id)
