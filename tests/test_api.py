import asyncio
import json

import httpx
import websockets

Q = "How does the real-time streaming pipeline handle late events and bound state?"


async def sse_events(client: httpx.AsyncClient, url: str, **kw):
    out = []
    async with client.stream("GET", url, **kw) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        cur = {}
        async for line in r.aiter_lines():
            if line.startswith("id:"): cur["id"] = int(line[3:])
            elif line.startswith("event:"): cur["event"] = line[6:].strip()
            elif line.startswith("data:"): cur["data"] = json.loads(line[5:])
            elif not line and cur:
                out.append(cur); cur = {}
    return out


async def test_health_ready_metrics_ui(server):
    async with httpx.AsyncClient(base_url=server) as c:
        assert (await c.get("/healthz")).json()["status"] == "ok"
        ready = (await c.get("/readyz")).json()
        assert ready["database"] is True and "search_corpus" in ready["tools"] and "offline" in ready["providers"]
        assert "<title>" in (await c.get("/")).text
        assert (await c.get("/ui/../config.py")).status_code in (404, 422)
        r = await c.get("/v1/providers"); assert r.json()["providers"][0]["name"] == "offline" and r.json()["providers"][0]["deterministic"] is True
        assert {p["name"] for p in (await c.get("/v1/prompts")).json()} == {"planner", "synthesizer"}
        assert any(t["name"] == "save_report" and t["risk"] == "consequential" for t in (await c.get("/v1/tools")).json())


async def test_create_run_stream_sse_and_fetch(server):
    async with httpx.AsyncClient(base_url=server, timeout=30) as c:
        r = await c.post("/v1/runs", json={"query": Q}); assert r.status_code == 202
        rid = r.json()["run_id"]
        evs = await sse_events(c, f"/v1/runs/{rid}/events")
        types = [e["event"] for e in evs]
        assert types[0] == "run.created" and types[-1] == "run.completed" and "plan.created" in types and "evidence.added" in types
        assert [e["id"] for e in evs] == list(range(1, len(evs) + 1))
        run = (await c.get(f"/v1/runs/{rid}")).json()
        assert run["state"] == "completed" and run["report"]["claims"] and run["verification"]["ok"]
        # reconnect with Last-Event-ID resumes after the given sequence
        tail = await sse_events(c, f"/v1/runs/{rid}/events", headers={"Last-Event-ID": str(len(evs) - 2)})
        assert [e["id"] for e in tail] == [len(evs) - 1, len(evs)]
        assert (await c.get(f"/v1/runs/{rid}/trace")).json()["spans"]
        assert any(x["id"] == rid for x in (await c.get("/v1/runs")).json())
        chunk_id = next(e["data"]["data"]["chunk_id"] for e in evs if e["event"] == "evidence.added")
        assert (await c.get(f"/v1/chunks/{chunk_id}")).json()["text"]


async def test_websocket_streams_same_events(server):
    async with httpx.AsyncClient(base_url=server, timeout=30) as c:
        rid = (await c.post("/v1/runs", json={"query": Q})).json()["run_id"]
    got = []
    async with websockets.connect(server.replace("http", "ws") + f"/v1/runs/{rid}/ws") as ws:
        async for m in ws:
            got.append(json.loads(m))
    assert got[0]["type"] == "run.created" and got[-1]["type"] == "run.completed" and [g["seq"] for g in got] == list(range(1, len(got) + 1))


async def test_http_approval_flow(server):
    async with httpx.AsyncClient(base_url=server, timeout=30) as c:
        rid = (await c.post("/v1/runs", json={"query": Q, "constraints": {"save_report": True}})).json()["run_id"]
        async def approve():
            for _ in range(100):
                rows = (await c.get(f"/v1/runs/{rid}/approvals")).json()
                if rows:
                    r1 = await c.post(f"/v1/approvals/{rows[0]['id']}", json={"approve": True, "reason": "looks fine"})
                    r2 = await c.post(f"/v1/approvals/{rows[0]['id']}", json={"approve": False})
                    return r1, r2
                await asyncio.sleep(0.05)
        task = asyncio.create_task(approve())
        evs = await sse_events(c, f"/v1/runs/{rid}/events")
        r1, r2 = await task
        assert r1.status_code == 200 and r1.json()["status"] == "approved" and r2.status_code == 409    # second decision rejected
        assert "report.saved" in [e["event"] for e in evs]
        assert (await c.post("/v1/approvals/apr_nope", json={"approve": True})).status_code == 404


async def test_validation_and_unknown_provider(server):
    async with httpx.AsyncClient(base_url=server) as c:
        assert (await c.post("/v1/runs", json={"query": "x"})).status_code == 422
        assert (await c.post("/v1/runs", json={"query": "valid enough question", "provider": "nonexistent"})).status_code == 400
        assert (await c.post("/v1/runs", json={"query": "valid enough question", "provider": "openai"})).status_code == 400   # key not configured
        assert (await c.get("/v1/runs/run_missing")).status_code == 404


async def test_documents_endpoint_ingests_and_makes_searchable(server):
    async with httpx.AsyncClient(base_url=server, timeout=30) as c:
        body = {"id": "api-doc", "title": "API doc", "text": "# API doc\n\nThe zanzibar widget calibration procedure requires exactly seven steps and a torque wrench."}
        assert (await c.post("/v1/documents", json=body)).json()["status"] == "indexed"
        rid = (await c.post("/v1/runs", json={"query": "How many steps does the zanzibar widget calibration procedure require?"})).json()["run_id"]
        evs = await sse_events(c, f"/v1/runs/{rid}/events")
        assert any(e["event"] == "evidence.added" and e["data"]["data"]["doc_id"] == "api-doc" for e in evs)
        assert (await c.post("/v1/documents", json={"id": "bad id!", "title": "t", "text": "x" * 40})).status_code == 422
        assert any(d["id"] == "api-doc" for d in (await c.get("/v1/documents")).json())


async def test_mcp_endpoint_over_http(server):
    async with httpx.AsyncClient(base_url=server) as c:
        r = await c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "calculator", "arguments": {"expression": "12*12"}}})
        assert r.json()["result"]["content"][0]["text"].endswith("= 144")
        assert (await c.post("/mcp", content="{nope", headers={"content-type": "application/json"})).status_code == 400


async def test_metrics_endpoint_exposes_tool_and_run_counters(server):
    async with httpx.AsyncClient(base_url=server, timeout=30) as c:
        rid = (await c.post("/v1/runs", json={"query": Q})).json()["run_id"]
        await sse_events(c, f"/v1/runs/{rid}/events")
        m = (await c.get("/metrics")).text
        assert 'agent_runs_total{provider="offline",status="completed"}' in m and 'agent_tool_calls_total{status="ok",tool="search_corpus"}' in m
        assert "agent_run_latency_ms_bucket" in m and "agent_llm_prompt_tokens_total" in m


async def test_api_key_protection(container):
    from agentplatform.api.app import create_app
    from agentplatform.config import Settings
    s = Settings(**{**container.settings.model_dump(), "api_key": "sekret"})
    async def factory(): return container
    app = create_app(s, container_factory=factory)
    app.state.c = container                                                   # ASGITransport does not run lifespan
    t = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=t, base_url="http://t") as c:
        assert (await c.get("/healthz")).status_code == 200                      # health stays open for orchestrators
        assert (await c.get("/v1/runs")).status_code == 401
        assert (await c.get("/v1/runs", headers={"x-api-key": "wrong"})).status_code == 401
        assert (await c.get("/v1/runs", headers={"x-api-key": "sekret"})).status_code == 200
