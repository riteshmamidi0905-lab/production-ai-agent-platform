"""Evaluation metrics. Deliberately independent of the online verifier (orchestrator/verify.py): claims are judged with a stricter
rule (token coverage ≥ 0.8 *and* every number must appear in the cited evidence). All metrics are deterministic and computed from
recorded run artefacts; none uses an LLM judge."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..retrieval.text import tokens, words
from ..telemetry import percentile

_NUM = re.compile(r"\d+(?:\.\d+)?")
_CITE = re.compile(r"\[(e\d+)\]")


def claim_supported(claim: str, evidence_texts: list[str]) -> bool:
    if not evidence_texts:
        return False
    pool = set()
    for t in evidence_texts:
        pool |= set(tokens(t))
    ct = set(tokens(claim))
    if not ct or len(ct & pool) / len(ct) < 0.8:
        return False
    ev_nums = set(_NUM.findall(" ".join(evidence_texts)))
    return all(n in ev_nums for n in _NUM.findall(claim))


@dataclass
class CaseResult:
    id: str
    state: str
    failure_class: str | None
    latency_ms: float
    metrics: dict[str, float | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    cited_chunks: list[str] = field(default_factory=list)
    facts_hit: list[bool] = field(default_factory=list)
    tokens: int = 0
    cost_usd: float = 0.0
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def score_case(case: dict, run: dict, evidence: dict[str, dict]) -> CaseResult:
    """`run` = RunRecord dumped to JSON; `evidence` = {id: evidence dict} collected from the run's events."""
    exp = case.get("expect")
    claims = (run.get("report") or {}).get("claims", [])
    answer = (run.get("report") or {}).get("answer", "")
    res = CaseResult(id=case["id"], state=run["state"], failure_class=run.get("failure_class"), latency_ms=run.get("latency_ms") or 0.0, run_id=run["id"],
                     tokens=(run["usage"]["prompt_tokens"] + run["usage"]["completion_tokens"]), cost_usd=run["usage"]["cost_usd"])
    m = res.metrics
    calls = run.get("tool_calls", [])
    m["tool_success"] = (sum(1 for c in calls if c["status"] == "ok") / len(calls)) if calls else None
    if exp == "blocked":
        m["task_completion"] = 1.0 if run["failure_class"] == "guardrail_block" else 0.0
        res.notes.append("expected guardrail block")
        return res
    if exp == "abstain":
        abstained = run["state"] == "completed" and not claims
        m["task_completion"] = 1.0 if abstained else 0.0
        m["groundedness"] = 1.0 if abstained else (sum(claim_supported(c["text"], [evidence[i]["text"] for i in c["evidence_ids"] if i in evidence]) for c in claims) / len(claims) if claims else None)
        res.notes.append("expected abstention (no claims)")
        return res
    if run["state"] != "completed":
        m["task_completion"] = 0.0; m["groundedness"] = 0.0; m["citation_f1"] = 0.0; m["instruction_adherence"] = 0.0
        res.notes.append(f"run did not complete: {run.get('failure_class')}")
        return res
    # groundedness (strict, claim level) + unsupported numbers
    sup = [claim_supported(c["text"], [evidence[i]["text"] for i in c["evidence_ids"] if i in evidence]) for c in claims]
    m["groundedness"] = sum(sup) / len(sup) if sup else 0.0
    ev_nums = set(_NUM.findall(" ".join(e["text"] for e in evidence.values())))
    ans_nums = set(_NUM.findall(re.sub(r"\[e\d+\]", "", answer)))
    m["hallucinated_number_rate"] = (len(ans_nums - ev_nums) / len(ans_nums)) if ans_nums else 0.0
    # citation correctness: F1 between cited docs and labelled expected docs
    cited = [evidence[i] for c in claims for i in c["evidence_ids"] if i in evidence]
    cited_docs = {e["doc_id"] for e in cited}
    exp_docs = set(case.get("expected_docs", []))
    res.cited_chunks = sorted({e["chunk_id"] for e in cited})
    if exp_docs:
        prec = sum(1 for e in cited if e["doc_id"] in exp_docs or e["doc_id"].startswith("tool:")) / len(cited) if cited else 0.0
        rec = len(exp_docs & cited_docs) / len(exp_docs) if not all(d.startswith("tool:") for d in cited_docs) or cited_docs else 0.0
        m["citation_precision"], m["citation_recall"] = prec, rec
        m["citation_f1"] = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
    # task completion: fraction of required facts present
    text = answer.lower()
    res.facts_hit = [any(a.lower() in text for a in alts) for alts in case.get("required_facts", [])]
    m["task_completion"] = sum(res.facts_hit) / len(res.facts_hit) if res.facts_hit else None
    # instruction adherence
    c = case.get("constraints", {})
    checks = [bool(_CITE.search(answer))]                  # citations present
    if c.get("max_words"):
        checks.append(words(answer) <= c["max_words"])
    checks.append(not any(p in text for p in case.get("forbidden", [])))
    m["instruction_adherence"] = sum(checks) / len(checks)
    return res


def consistency(results: list[CaseResult]) -> float | None:
    """Mean pairwise Jaccard over cited chunk sets *and* fact-hit vectors across repeated runs of one case."""
    if len(results) < 2:
        return None
    vals = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = set(results[i].cited_chunks), set(results[j].cited_chunks)
            jac = len(a & b) / len(a | b) if a | b else 1.0
            fa = sum(x == y for x, y in zip(results[i].facts_hit, results[j].facts_hit)) / max(1, len(results[i].facts_hit)) if results[i].facts_hit else 1.0
            vals.append(0.5 * jac + 0.5 * fa)
    return sum(vals) / len(vals)


def aggregate(cases: list[CaseResult], cons: dict[str, float | None]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        v = [c.metrics[key] for c in cases if c.metrics.get(key) is not None]
        return round(sum(v) / len(v), 4) if v else None
    lats = [c.latency_ms for c in cases]
    cv = [v for v in cons.values() if v is not None]
    return {"groundedness": mean("groundedness"), "citation_f1": mean("citation_f1"), "citation_precision": mean("citation_precision"), "citation_recall": mean("citation_recall"),
            "task_completion": mean("task_completion"), "tool_success": mean("tool_success"), "instruction_adherence": mean("instruction_adherence"),
            "hallucinated_number_rate": mean("hallucinated_number_rate"), "consistency": round(sum(cv) / len(cv), 4) if cv else None,
            "latency_p50_ms": round(percentile(lats, 0.5), 1), "latency_p95_ms": round(percentile(lats, 0.95), 1), "cases": len(cases),
            "completed": sum(1 for c in cases if c.state == "completed"), "total_tokens": sum(c.tokens for c in cases), "total_cost_usd": round(sum(c.cost_usd for c in cases), 6)}
