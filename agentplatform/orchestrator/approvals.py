"""Human-in-the-loop approvals. A consequential tool call creates a persisted approval and *suspends that step* until a human
decides (or the timeout expires -> denied). Decisions travel over the cache pub/sub so any API worker can resolve any run."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..cache import Cache
from ..models import new_id
from ..storage.repo import Repo


@dataclass
class Decision:
    approved: bool
    reason: str | None = None
    timed_out: bool = False


class ApprovalBroker:
    def __init__(self, repo: Repo, cache: Cache) -> None:
        self.repo, self.cache = repo, cache

    async def request(self, run_id: str, tool: str, args: dict, risk: str, timeout: float, on_created) -> Decision:
        aid = new_id("apr")
        await self.repo.create_approval(aid, run_id, tool, args, risk)
        ready = asyncio.Event()
        result: dict = {}

        async def listen() -> None:
            async for m in self.cache.subscribe(f"approval:{aid}"):
                result.update(m); ready.set(); return
        task = asyncio.create_task(listen())
        await asyncio.sleep(0)
        await on_created(aid)                              # emit tool.approval_required only once we are listening
        try:
            await asyncio.wait_for(ready.wait(), timeout=timeout)
            return Decision(bool(result["approved"]), result.get("reason"))
        except asyncio.TimeoutError:
            await self.repo.decide_approval(aid, "expired", "approval timeout")
            return Decision(False, "approval timed out", timed_out=True)
        finally:
            task.cancel()

    async def decide(self, aid: str, approve: bool, reason: str | None = None) -> dict | None:
        row = await self.repo.decide_approval(aid, "approved" if approve else "denied", reason)
        if row:
            await self.cache.publish(f"approval:{aid}", {"approved": approve, "reason": reason})
        return row
