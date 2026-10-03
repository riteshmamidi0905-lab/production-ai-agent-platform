"""FastAPI application: REST + SSE + WebSocket execution streaming, approvals, documents, MCP endpoint, metrics and the operations UI."""
from __future__ import annotations

import asyncio
import hmac
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..container import Container
from ..config import Settings, get_settings
from ..models import RunRequest
from ..prompts import catalog
from ..providers.base import ProviderError
from ..telemetry import METRICS

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


class ApprovalDecision(BaseModel):
    approve: bool
    reason: str | None = Field(default=None, max_length=300)


class DocumentIn(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,80}$")
    title: str = Field(max_length=200)
    text: str = Field(min_length=30, max_length=500_000)


def create_app(settings: Settings | None = None, container_factory=None) -> FastAPI:
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.c = await (container_factory() if container_factory else Container.build(s))
        try:
            yield
        finally:
            await app.state.c.close()

    app = FastAPI(title="Production AI Agent Platform", version=__version__, lifespan=lifespan)
    if s.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"])

    def C(request: Request) -> Container:
        return request.app.state.c

    def auth(request: Request, api_key: str | None = Query(default=None)) -> None:
        if not s.api_key:
            return
        given = request.headers.get("x-api-key") or api_key or ""
        if not hmac.compare_digest(given, s.api_key):
            raise HTTPException(401, "invalid or missing API key")

    # ------------------------------------------------------------- health / ops
    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    async def readyz(c: Container = Depends(C)):
        try:
            await c.db.fetch("SELECT 1")
            db_ok = True
        except Exception:  # noqa: BLE001
            db_ok = False
        body = {"database": db_ok, "cache": c.cache.kind, "providers": c.providers.available(), "tools": c.registry.names()}
        return JSONResponse(body, status_code=200 if db_ok else 503)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return METRICS.render()

    # ------------------------------------------------------------- runs
    @app.post("/v1/runs", status_code=202, dependencies=[Depends(auth)])
    async def create_run(req: RunRequest, c: Container = Depends(C)):
        try:
            rec = await c.engine.submit(req)
        except ProviderError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"run_id": rec.id, "state": rec.state.value, "events_url": f"/v1/runs/{rec.id}/events"}

    @app.get("/v1/runs", dependencies=[Depends(auth)])
    async def list_runs(limit: int = Query(30, ge=1, le=200), c: Container = Depends(C)):
        return [{"id": r.id, "query": r.query, "state": r.state.value, "provider": r.provider, "latency_ms": r.latency_ms, "cost_usd": r.usage.cost_usd,
                 "failure_class": r.failure_class.value if r.failure_class else None, "created_at": r.created_at} for r in await c.repo.list_runs(limit)]

    @app.get("/v1/runs/{run_id}", dependencies=[Depends(auth)])
    async def get_run(run_id: str, c: Container = Depends(C)):
        r = await c.repo.get_run(run_id)
        if not r:
            raise HTTPException(404, "run not found")
        return r.model_dump(mode="json")

    @app.post("/v1/runs/{run_id}/cancel", dependencies=[Depends(auth)])
    async def cancel(run_id: str, c: Container = Depends(C)):
        return {"cancelled": await c.engine.cancel(run_id)}

    @app.get("/v1/runs/{run_id}/trace", dependencies=[Depends(auth)])
    async def trace(run_id: str, c: Container = Depends(C)):
        spans = await c.repo.get_trace(run_id)
        if spans is None:
            raise HTTPException(404, "trace not available (run still in flight or unknown)")
        return {"run_id": run_id, "spans": spans}

    @app.get("/v1/runs/{run_id}/events", dependencies=[Depends(auth)])
    async def sse(run_id: str, request: Request, after: int = 0, c: Container = Depends(C)):
        if not await c.repo.get_run(run_id):
            raise HTTPException(404, "run not found")
        after = max(after, int(request.headers.get("last-event-id", 0) or 0))

        async def gen():
            async for ev in c.hub.stream(run_id, after):
                yield f"id: {ev.seq}\nevent: {ev.type}\ndata: {json.dumps(ev.model_dump())}\n\n"
                if await request.is_disconnected():
                    return
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.websocket("/v1/runs/{run_id}/ws")
    async def ws(websocket: WebSocket, run_id: str, after: int = 0, api_key: str | None = None):
        c: Container = websocket.app.state.c
        if s.api_key and not hmac.compare_digest(api_key or "", s.api_key):
            await websocket.close(code=4401); return
        await websocket.accept()
        if not await c.repo.get_run(run_id):
            await websocket.close(code=4404); return
        try:
            async for ev in c.hub.stream(run_id, after):
                await websocket.send_json(ev.model_dump())
            await websocket.close()
        except WebSocketDisconnect:
            return

    # ------------------------------------------------------------- approvals
    @app.get("/v1/runs/{run_id}/approvals", dependencies=[Depends(auth)])
    async def approvals(run_id: str, c: Container = Depends(C)):
        return await c.repo.run_approvals(run_id)

    @app.post("/v1/approvals/{approval_id}", dependencies=[Depends(auth)])
    async def decide(approval_id: str, d: ApprovalDecision, c: Container = Depends(C)):
        row = await c.broker.decide(approval_id, d.approve, d.reason)
        if not row:
            existing = await c.repo.get_approval(approval_id)
            raise HTTPException(409 if existing else 404, "approval already decided or expired" if existing else "approval not found")
        return {"id": approval_id, "status": row["status"]}

    # ------------------------------------------------------------- knowledge
    @app.post("/v1/documents", status_code=201, dependencies=[Depends(auth)])
    async def add_document(d: DocumentIn, c: Container = Depends(C)):
        try:
            return await c.ingestor.ingest_text(d.id, d.title, d.text)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/documents", dependencies=[Depends(auth)])
    async def documents(c: Container = Depends(C)):
        return await c.repo.list_documents()

    @app.get("/v1/chunks/{chunk_id:path}", dependencies=[Depends(auth)])
    async def chunk(chunk_id: str, c: Container = Depends(C)):
        row = await c.repo.get_chunk(chunk_id)
        if not row:
            raise HTTPException(404, "chunk not found")
        return row

    @app.get("/v1/conversations/{cid}", dependencies=[Depends(auth)])
    async def conversation(cid: str, c: Container = Depends(C)):
        return {"id": cid, "messages": await c.repo.messages(cid, 50)}

    # ------------------------------------------------------------- introspection
    @app.get("/v1/providers", dependencies=[Depends(auth)])
    async def providers(c: Container = Depends(C)):
        out = []
        for n in c.providers.available():
            p = c.providers.get(n)
            out.append({"name": n, "model": p.model, "deterministic": p.deterministic})
        return {"default": s.default_provider, "fallbacks": s.provider_fallbacks, "providers": out}

    @app.get("/v1/prompts", dependencies=[Depends(auth)])
    async def prompts():
        return catalog()

    @app.get("/v1/tools", dependencies=[Depends(auth)])
    async def tools(c: Container = Depends(C)):
        return c.registry.descriptors()

    # ------------------------------------------------------------- evaluation
    @app.post("/v1/evals/run", status_code=202, dependencies=[Depends(auth)])
    async def run_eval(request: Request, provider: str | None = None, repeats: int = Query(2, ge=1, le=5), c: Container = Depends(C)):
        from ..eval.runner import run_and_store
        eid = f"eval_{int(time.time())}"
        task = asyncio.create_task(run_and_store(c, provider or s.default_provider, repeats=repeats, eval_id=eid))
        request.app.state.eval_tasks = getattr(request.app.state, "eval_tasks", set()) | {task}
        task.add_done_callback(lambda t: request.app.state.eval_tasks.discard(t))
        return {"eval_id": eid}

    @app.get("/v1/evals", dependencies=[Depends(auth)])
    async def evals(c: Container = Depends(C)):
        return await c.repo.list_evals()

    @app.get("/v1/evals/{eid}", dependencies=[Depends(auth)])
    async def eval_detail(eid: str, c: Container = Depends(C)):
        row = await c.repo.get_eval(eid)
        if not row:
            raise HTTPException(404, "evaluation not found (it may still be running)")
        return row

    # ------------------------------------------------------------- MCP
    @app.post("/mcp", dependencies=[Depends(auth)])
    async def mcp(request: Request, c: Container = Depends(C)):
        try:
            msg = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}, status_code=400)
        resp = await c.mcp.handle(msg)
        return JSONResponse(resp) if resp is not None else PlainTextResponse("", status_code=202)

    # ------------------------------------------------------------- UI
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(UI_DIR / "index.html")

    @app.get("/ui/{name}", include_in_schema=False)
    async def ui_asset(name: str):
        p = (UI_DIR / name).resolve()
        if UI_DIR not in p.parents or not p.is_file():
            raise HTTPException(404)
        return FileResponse(p)

    return app


app = create_app() if False else None  # `uvicorn agentplatform.api.app:create_app --factory`
