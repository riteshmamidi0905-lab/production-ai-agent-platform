"""Cache + pub/sub. Redis when configured (shared across workers); in-process fallback otherwise, same interface."""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import Any, AsyncIterator


class Cache:
    kind = "memory"

    def __init__(self) -> None:
        self._kv: dict[str, tuple[float, str]] = {}
        self._subs: dict[str, list[asyncio.Queue[str]]] = defaultdict(list)

    async def get(self, key: str) -> Any | None:
        item = self._kv.get(key)
        if not item or item[0] < time.time():
            self._kv.pop(key, None)
            return None
        return json.loads(item[1])

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        self._kv[key] = (time.time() + ttl, json.dumps(value))

    async def publish(self, channel: str, message: dict) -> None:
        payload = json.dumps(message)
        for q in list(self._subs[channel]):
            q.put_nowait(payload)

    async def subscribe(self, channel: str) -> AsyncIterator[dict]:
        q: asyncio.Queue[str] = asyncio.Queue()
        self._subs[channel].append(q)
        try:
            while True:
                yield json.loads(await q.get())
        finally:
            self._subs[channel].remove(q)

    async def close(self) -> None:
        pass


class RedisCache(Cache):
    kind = "redis"

    def __init__(self, url: str, client: Any | None = None) -> None:
        super().__init__()
        import redis.asyncio as aioredis
        self.r = client or aioredis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> Any | None:
        v = await self.r.get(f"ap:{key}")
        return json.loads(v) if v is not None else None

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        await self.r.set(f"ap:{key}", json.dumps(value), ex=ttl)

    async def publish(self, channel: str, message: dict) -> None:
        await self.r.publish(f"ap:{channel}", json.dumps(message))

    async def subscribe(self, channel: str) -> AsyncIterator[dict]:
        ps = self.r.pubsub()
        await ps.subscribe(f"ap:{channel}")
        try:
            async for m in ps.listen():
                if m.get("type") == "message":
                    yield json.loads(m["data"])
        finally:
            await ps.unsubscribe(f"ap:{channel}")
            await ps.aclose()

    async def close(self) -> None:
        await self.r.aclose()


def make_cache(redis_url: str | None) -> Cache:
    return RedisCache(redis_url) if redis_url else Cache()
