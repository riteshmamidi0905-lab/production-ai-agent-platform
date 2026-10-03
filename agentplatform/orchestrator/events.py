"""Run event hub: persist every event (append-only, per-run sequence), fan out live via the cache (Redis or memory), and replay
from the database so SSE/WebSocket clients can reconnect with `after_seq`."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from ..cache import Cache
from ..models import Event
from ..storage.repo import Repo

TERMINAL = {"run.completed", "run.failed", "run.cancelled", "run.interrupted"}


class EventHub:
    def __init__(self, repo: Repo, cache: Cache) -> None:
        self.repo, self.cache = repo, cache
        self._seq: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def emit(self, run_id: str, type: str, **data: Any) -> Event:
        lock = self._locks.setdefault(run_id, asyncio.Lock())
        async with lock:                                   # one writer per run keeps seq gap-free and ordered
            seq = self._seq.get(run_id, 0) + 1
            self._seq[run_id] = seq
            ev = Event(run_id=run_id, seq=seq, type=type, data=data)
            await self.repo.append_event(ev)
        await self.cache.publish(f"run:{run_id}", ev.model_dump())
        if type in TERMINAL:
            self._seq.pop(run_id, None); self._locks.pop(run_id, None)
        return ev

    async def stream(self, run_id: str, after: int = 0, idle_timeout: float = 600) -> AsyncIterator[Event]:
        """Replay persisted events > after, then follow live ones until a terminal event. Subscribes before replaying so nothing is missed."""
        queue: asyncio.Queue[Event] = asyncio.Queue()

        async def pump() -> None:
            async for m in self.cache.subscribe(f"run:{run_id}"):
                await queue.put(Event.model_validate(m))
        task = asyncio.create_task(pump())
        await asyncio.sleep(0)                             # let the subscription register
        try:
            last = after
            for ev in await self.repo.events_after(run_id, after):
                last = ev.seq
                yield ev
                if ev.type in TERMINAL:
                    return
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=idle_timeout)
                except asyncio.TimeoutError:
                    return
                if ev.seq <= last:
                    continue
                last = ev.seq
                yield ev
                if ev.type in TERMINAL:
                    return
        finally:
            task.cancel()
