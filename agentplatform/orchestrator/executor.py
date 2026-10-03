"""Tool execution with argument validation, risk-based approval gating, per-call timeout, bounded retries and full metrics."""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from pydantic import ValidationError

from ..config import Settings
from ..models import FailureClass, ToolCallRecord, new_id
from ..telemetry import METRICS, Tracer
from ..tools.base import Risk, ToolContext, ToolError, ToolRegistry, ToolResult
from .approvals import ApprovalBroker

Emit = Callable[..., Awaitable[object]]


class ToolFailure(Exception):
    def __init__(self, record: ToolCallRecord, failure: FailureClass):
        super().__init__(record.error or record.status)
        self.record, self.failure = record, failure


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, settings: Settings, broker: ApprovalBroker) -> None:
        self.registry, self.settings, self.broker = registry, settings, broker

    def needs_approval(self, risk: Risk, mode: str) -> bool:
        return mode == "all_tools" or (mode == "consequential" and risk is Risk.CONSEQUENTIAL)

    async def call(self, tool_name: str, raw_args: dict, ctx: ToolContext, *, emit: Emit, tracer: Tracer, approval_mode: str,
                   step_id: str | None = None) -> tuple[ToolResult, ToolCallRecord]:
        cid = new_id("tc")
        tool = self.registry.get(tool_name)
        t0 = time.perf_counter()

        def record(status, attempts=1, error=None, preview=None) -> ToolCallRecord:
            ms = (time.perf_counter() - t0) * 1000
            METRICS.inc("agent_tool_calls_total", tool=tool_name, status=status)
            METRICS.observe("agent_tool_latency_ms", ms, tool=tool_name)
            return ToolCallRecord(id=cid, tool=tool_name, args=raw_args, status=status, duration_ms=round(ms, 2), attempts=attempts, error=error, result_preview=preview)

        if tool is None:
            rec = record("invalid_args", error=f"unknown tool '{tool_name}'")
            await emit("tool.failed", call_id=cid, tool=tool_name, step_id=step_id, status=rec.status, error=rec.error)
            raise ToolFailure(rec, FailureClass.TOOL_ARGS_INVALID)
        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            rec = record("invalid_args", error=str(exc.errors()[0]["msg"]) + " @ " + ".".join(map(str, exc.errors()[0]["loc"])))
            await emit("tool.failed", call_id=cid, tool=tool_name, step_id=step_id, status=rec.status, error=rec.error)
            raise ToolFailure(rec, FailureClass.TOOL_ARGS_INVALID) from exc
        await emit("tool.requested", call_id=cid, tool=tool_name, args=raw_args, risk=tool.risk.value, step_id=step_id)
        if self.needs_approval(tool.risk, approval_mode):
            async def created(aid: str) -> None:
                await emit("tool.approval_required", call_id=cid, approval_id=aid, tool=tool_name, args=raw_args, risk=tool.risk.value, step_id=step_id,
                           expires_in_s=self.settings.approval_timeout_s)
            async with tracer.span("approval", tool=tool_name):
                d = await self.broker.request(ctx.run_id, tool_name, raw_args, tool.risk.value, self.settings.approval_timeout_s, created)
            if not d.approved:
                rec = record("denied", error=d.reason or "denied")
                await emit("tool.denied", call_id=cid, tool=tool_name, step_id=step_id, reason=d.reason, timed_out=d.timed_out)
                raise ToolFailure(rec, FailureClass.APPROVAL_TIMEOUT if d.timed_out else FailureClass.APPROVAL_DENIED)
            await emit("tool.approved", call_id=cid, tool=tool_name, step_id=step_id, reason=d.reason)
            t0 = time.perf_counter()                        # latency excludes human wait time
        attempts = 0
        last: Exception | None = None
        async with tracer.span(f"tool:{tool_name}", call_id=cid, step=step_id) as sp:
            while attempts <= self.settings.tool_retries:
                attempts += 1
                try:
                    async with asyncio.timeout(self.settings.tool_timeout_s):
                        res = await tool.run(args, ctx)
                    rec = record("ok", attempts, preview=res.text[:200])
                    sp.attrs.update(attempts=attempts, status="ok")
                    await emit("tool.completed", call_id=cid, tool=tool_name, step_id=step_id, duration_ms=rec.duration_ms, attempts=attempts, summary=res.text[:300], evidence_count=len(res.evidence))
                    return res, rec
                except TimeoutError as exc:
                    last = exc
                    kind, fc, retry = "timeout", FailureClass.TOOL_TIMEOUT, True
                except ToolError as exc:
                    last = exc
                    kind, fc, retry = "error", FailureClass.TOOL_ERROR, exc.retryable
                except Exception as exc:  # noqa: BLE001 - tool bugs must never crash the run
                    last = exc
                    kind, fc, retry = "error", FailureClass.TOOL_ERROR, False
                if not retry or attempts > self.settings.tool_retries:
                    break
                await emit("tool.retrying", call_id=cid, tool=tool_name, attempt=attempts, error=str(last)[:200])
                await asyncio.sleep(0.2 * attempts)
        msg = f"{type(last).__name__}: {last}"[:300] if kind == "error" else f"timed out after {self.settings.tool_timeout_s}s"
        rec = record(kind, attempts, error=msg)
        await emit("tool.failed", call_id=cid, tool=tool_name, step_id=step_id, status=kind, error=msg, attempts=attempts)
        raise ToolFailure(rec, fc)
