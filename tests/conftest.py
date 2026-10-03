"""Shared fixtures. Integration tests run against a REAL PostgreSQL 16 + pgvector (embedded via `pgserver`), a real HTTP server for
SSE/WebSocket, and fakeredis for the Redis code path. Nothing about the database layer is mocked."""
from __future__ import annotations

import asyncio
import socket
import tempfile
from pathlib import Path

import pgserver
import pytest
import pytest_asyncio
import uvicorn

from agentplatform.api.app import create_app
from agentplatform.config import Settings
from agentplatform.container import Container

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def pg():
    srv = pgserver.get_server(tempfile.mkdtemp(prefix="agentplatform-pg-"))
    yield srv
    srv.cleanup()


@pytest_asyncio.fixture(scope="session")
async def container(pg):
    settings = Settings(database_url=pg.get_uri(), reports_dir=tempfile.mkdtemp(prefix="reports-"), approval_timeout_s=5, tool_timeout_s=3)
    c = await Container.build(settings)
    await c.ingestor.ingest_dir(ROOT / "corpus")
    yield c
    await c.close()


@pytest_asyncio.fixture(autouse=True)
async def clean(request):
    yield
    if "container" in request.fixturenames:
        c = request.getfixturevalue("container")
        await c.db.execute("TRUNCATE runs, conversations RESTART IDENTITY CASCADE")
        c.cache._kv.clear()
        c.settings.approval_timeout_s, c.settings.tool_timeout_s, c.settings.run_budget_usd = 5, 3, 1.0
        c.settings.provider_fallbacks = ["offline"]


async def run_to_end(c: Container, req, approve: bool | None = None):
    """Submit a run, optionally auto-answering approvals, wait for completion, return (record, events)."""
    from agentplatform.models import RunRequest
    rec = await c.engine.submit(req if isinstance(req, RunRequest) else RunRequest(**req))
    async def approver():
        async for ev in c.hub.stream(rec.id):
            if ev.type == "tool.approval_required" and approve is not None:
                await c.broker.decide(ev.data["approval_id"], approve, "test")
    t = asyncio.create_task(approver())
    await asyncio.wait_for(c.engine.wait(rec.id), 30)
    await t
    return await c.repo.get_run(rec.id), await c.repo.events_after(rec.id)


@pytest_asyncio.fixture(scope="session")
async def server(container):
    """A real uvicorn server on a free port serving the app over the shared container (for SSE / WebSocket tests)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    async def factory():
        return container
    app = create_app(container.settings, container_factory=factory)
    # the app's lifespan would close the shared container; shield it
    async def no_close(): return None
    orig = container.close
    container.close = no_close  # type: ignore[method-assign]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    task = asyncio.create_task(srv.serve())
    for _ in range(100):
        if srv.started:
            break
        await asyncio.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    await task
    container.close = orig  # type: ignore[method-assign]
