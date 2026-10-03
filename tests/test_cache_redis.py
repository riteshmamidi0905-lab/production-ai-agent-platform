import asyncio

import fakeredis.aioredis

from agentplatform.cache import RedisCache
from agentplatform.models import RunRequest


async def test_redis_cache_kv_and_pubsub():
    c = RedisCache("redis://fake", client=fakeredis.aioredis.FakeRedis(decode_responses=True))
    await c.set("k", {"a": 1}, ttl=5)
    assert await c.get("k") == {"a": 1} and await c.get("missing") is None
    got = []
    async def sub():
        async for m in c.subscribe("ch"):
            got.append(m); return
    t = asyncio.create_task(sub()); await asyncio.sleep(0.1)
    await c.publish("ch", {"x": 2}); await asyncio.wait_for(t, 3)
    assert got == [{"x": 2}]
    await c.close()


async def test_event_stream_and_approvals_work_over_redis_pubsub(container):
    """Swap the container's cache for the Redis implementation: events and approval decisions must still flow."""
    old = (container.cache, container.hub.cache, container.broker.cache, container.engine.cache)
    r = RedisCache("redis://fake", client=fakeredis.aioredis.FakeRedis(decode_responses=True))
    container.cache = container.hub.cache = container.broker.cache = container.engine.cache = r
    try:
        from agentplatform.models import RunConstraints
        rec = await container.engine.submit(RunRequest(query="How does the real-time streaming pipeline handle late events?", constraints=RunConstraints(save_report=True)))
        seen = []
        async for ev in container.hub.stream(rec.id):
            seen.append(ev.type)
            if ev.type == "tool.approval_required":
                await container.broker.decide(ev.data["approval_id"], True)
        assert seen[0] == "run.created" and seen[-1] == "run.completed" and "report.saved" in seen
    finally:
        container.cache, container.hub.cache, container.broker.cache, container.engine.cache = old
        await r.close()
