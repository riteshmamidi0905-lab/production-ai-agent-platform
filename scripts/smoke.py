"""End-to-end smoke test against a running deployment: python scripts/smoke.py http://localhost:8000
Ingests nothing itself (run `docker compose run --rm ingest` first); submits a run, approves its save, checks the result."""
import json
import sys
import time

import httpx

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
with httpx.Client(base_url=base, timeout=60) as c:
    for _ in range(60):
        try:
            if c.get("/readyz").status_code == 200: break
        except httpx.HTTPError: pass
        time.sleep(1)
    else:
        sys.exit("service never became ready")
    docs = c.get("/v1/documents").json()
    assert docs, "no documents indexed — run the ingest step first"
    rid = c.post("/v1/runs", json={"query": "How does the real-time streaming pipeline handle late events and bound state?", "constraints": {"save_report": True}}).json()["run_id"]
    approved, final = False, None
    with c.stream("GET", f"/v1/runs/{rid}/events") as r:
        for line in r.iter_lines():
            if not line.startswith("data:"): continue
            ev = json.loads(line[5:])
            if ev["type"] == "tool.approval_required" and not approved:
                assert c.post(f"/v1/approvals/{ev['data']['approval_id']}", json={"approve": True}).status_code == 200; approved = True
            if ev["type"] in ("run.completed", "run.failed"):
                final = ev; break
    assert final and final["type"] == "run.completed", final
    run = c.get(f"/v1/runs/{rid}").json()
    assert run["report"]["claims"] and run["verification"]["ok"] and approved
    print(f"smoke OK: run {rid} completed in {run['latency_ms']:.0f} ms, {len(run['report']['claims'])} cited claims, {len(docs)} docs, groundedness {run['verification']['groundedness']}")
