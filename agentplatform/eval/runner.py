"""Runs the labelled dataset end-to-end through the real engine (real Postgres, real retrieval, real tools) and scores the artefacts."""
from __future__ import annotations

import asyncio
import json
import time
from importlib import resources
from typing import Any

from ..container import Container
from ..models import RunConstraints, RunRequest
from .metrics import CaseResult, aggregate, consistency, score_case

HIGHER_IS_BETTER = ("groundedness", "citation_f1", "task_completion", "tool_success", "instruction_adherence", "consistency")


def load_dataset(name: str = "research_v1") -> list[dict]:
    text = resources.files("agentplatform.eval").joinpath(f"datasets/{name}.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_baseline(provider: str) -> dict | None:
    p = resources.files("agentplatform.eval").joinpath(f"baselines/{provider}.json")
    return json.loads(p.read_text()) if p.is_file() else None


async def _approve_all(c: Container, run_id: str) -> None:
    async for ev in c.hub.stream(run_id):
        if ev.type == "tool.approval_required":
            await c.broker.decide(ev.data["approval_id"], True, "auto-approved by eval harness")


async def run_case(c: Container, case: dict, provider: str) -> CaseResult:
    cons = RunConstraints(**case.get("constraints", {}))
    rec = await c.engine.submit(RunRequest(query=case["query"], provider=provider, constraints=cons))
    approver = asyncio.create_task(_approve_all(c, rec.id))
    await c.engine.wait(rec.id)
    await approver
    run = (await c.repo.get_run(rec.id)).model_dump(mode="json")  # type: ignore[union-attr]
    evidence = {e.data["id"]: e.data for e in await c.repo.events_after(rec.id) if e.type == "evidence.added"}
    return score_case(case, run, evidence)


async def run_eval(c: Container, provider: str, *, repeats: int = 2, dataset: str = "research_v1", only: list[str] | None = None) -> dict[str, Any]:
    cases = [x for x in load_dataset(dataset) if not only or x["id"] in only]
    all_results: list[CaseResult] = []
    cons: dict[str, float | None] = {}
    per_case: list[dict] = []
    for case in cases:
        reps = [await run_case(c, case, provider) for _ in range(repeats)]
        cons[case["id"]] = consistency(reps) if not case.get("expect") else None
        all_results.append(reps[0])                         # headline scores use the first repetition; the others feed consistency
        per_case.append({**reps[0].to_dict(), "consistency": cons[case["id"]], "repeats": repeats})
    metrics = aggregate(all_results, cons)
    prov = c.providers.get(provider)
    return {"dataset": dataset, "provider": provider, "model": prov.model, "deterministic_provider": prov.deterministic, "repeats": repeats,
            "metrics": metrics, "cases": per_case, "corpus_docs": len(await c.repo.list_documents())}


def gate(metrics: dict, baseline: dict | None, tolerance: float = 0.02, p95_cap_ms: float = 5000.0) -> dict[str, Any]:
    """Regression gate: no tracked metric may fall more than `tolerance` below the stored baseline; p95 latency has an absolute cap."""
    if not baseline:
        return {"passed": True, "baseline": None, "failures": [], "note": "no baseline stored for this provider"}
    failures = []
    for k in HIGHER_IS_BETTER:
        cur, base = metrics.get(k), baseline["metrics"].get(k)
        if cur is not None and base is not None and cur < base - tolerance:
            failures.append({"metric": k, "baseline": base, "current": cur, "delta": round(cur - base, 4)})
    if metrics["latency_p95_ms"] > p95_cap_ms:
        failures.append({"metric": "latency_p95_ms", "baseline": p95_cap_ms, "current": metrics["latency_p95_ms"], "delta": None})
    return {"passed": not failures, "baseline": baseline["metrics"], "tolerance": tolerance, "failures": failures}


async def run_and_store(c: Container, provider: str, *, repeats: int = 2, eval_id: str | None = None) -> dict:
    result = await run_eval(c, provider, repeats=repeats)
    g = gate(result["metrics"], load_baseline(provider))
    rec = {"id": eval_id or f"eval_{int(time.time())}", "created_at": time.time(), "dataset": result["dataset"], "provider": provider,
           "config": {k: result[k] for k in ("model", "deterministic_provider", "repeats", "corpus_docs")}, "metrics": result["metrics"], "cases": result["cases"], "gate": g}
    await c.repo.save_eval(rec)
    return rec
