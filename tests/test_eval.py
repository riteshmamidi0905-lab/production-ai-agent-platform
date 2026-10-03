
from agentplatform.eval.metrics import CaseResult, aggregate, claim_supported, consistency, score_case
from agentplatform.eval.runner import gate, load_baseline, load_dataset, run_eval

ev = {"e1": {"id": "e1", "doc_id": "d1", "chunk_id": "d1::0", "text": "The pipeline keeps state for two minutes using a watermark and 3 windows."},
      "e2": {"id": "e2", "doc_id": "d2", "chunk_id": "d2::0", "text": "Unrelated content about gardening."}}


def run(claims, answer, state="completed", fc=None, calls=None):
    return {"id": "r", "state": state, "failure_class": fc, "latency_ms": 10, "report": {"claims": claims, "answer": answer}, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0},
            "tool_calls": calls if calls is not None else [{"status": "ok"}, {"status": "error"}]}


def test_claim_support_is_strict_about_numbers_and_coverage():
    t = [ev["e1"]["text"]]
    assert claim_supported("The pipeline keeps state for two minutes using a watermark.", t)
    assert not claim_supported("The pipeline keeps state for two minutes using a watermark and 9 windows.", t)        # invented number
    assert not claim_supported("Kafka brokers replicate partitions across racks.", t)
    assert not claim_supported("anything", [])


def test_score_case_metrics_are_computed_from_artifacts():
    case = {"id": "c", "expected_docs": ["d1"], "required_facts": [["watermark"], ["cassandra", "sink"]], "constraints": {"max_words": 20}}
    good = run([{"text": "The pipeline keeps state for two minutes using a watermark.", "evidence_ids": ["e1"]}], "- The pipeline keeps state for two minutes using a watermark. [e1]")
    r = score_case(case, good, ev).metrics
    assert r["groundedness"] == 1.0 and r["citation_f1"] == 1.0 and r["task_completion"] == 0.5 and r["instruction_adherence"] == 1.0 and r["tool_success"] == 0.5
    bad = run([{"text": "Kafka brokers replicate partitions.", "evidence_ids": ["e2"]}], "Kafka brokers replicate 99 partitions. [e2]")
    b = score_case(case, bad, ev).metrics
    assert b["groundedness"] == 0.0 and b["citation_f1"] == 0.0 and b["hallucinated_number_rate"] == 1.0 and b["task_completion"] == 0.0
    nocite = score_case(case, run([{"text": "x", "evidence_ids": []}], "no citation here"), ev).metrics
    assert nocite["instruction_adherence"] < 1.0


def test_abstain_blocked_and_failed_expectations():
    abst = score_case({"id": "a", "expect": "abstain"}, run([], "none"), ev).metrics
    assert abst["task_completion"] == 1.0
    assert score_case({"id": "a", "expect": "abstain"}, run([{"text": "t", "evidence_ids": ["e1"]}], "x"), ev).metrics["task_completion"] == 0.0
    assert score_case({"id": "b", "expect": "blocked"}, run([], "", state="failed", fc="guardrail_block"), ev).metrics["task_completion"] == 1.0
    assert score_case({"id": "b", "expect": "blocked"}, run([], ""), ev).metrics["task_completion"] == 0.0
    failed = score_case({"id": "f", "expected_docs": ["d1"]}, run([], "", state="failed", fc="provider_error"), ev)
    assert failed.metrics["task_completion"] == 0.0 and failed.failure_class == "provider_error"


def test_consistency_is_one_for_identical_runs_and_drops_for_different():
    a = CaseResult("c", "completed", None, 1, cited_chunks=["x", "y"], facts_hit=[True, True])
    b = CaseResult("c", "completed", None, 1, cited_chunks=["x", "y"], facts_hit=[True, True])
    c = CaseResult("c", "completed", None, 1, cited_chunks=["z"], facts_hit=[True, False])
    assert consistency([a, b]) == 1.0 and consistency([a, c]) < 0.5 and consistency([a]) is None


def test_aggregate_and_latency_percentiles():
    rs = [CaseResult(str(i), "completed", None, float(i * 10), metrics={"groundedness": 1.0 if i else 0.0}) for i in range(1, 5)]
    agg = aggregate(rs, {})
    assert agg["latency_p50_ms"] == 25.0 and agg["latency_p95_ms"] > 35 and agg["groundedness"] == 1.0 and agg["consistency"] is None


def test_gate_flags_regressions_and_respects_tolerance():
    base = {"metrics": {"groundedness": 1.0, "task_completion": 0.9, "consistency": 1.0}}
    assert gate({"groundedness": 0.99, "task_completion": 0.9, "consistency": 1.0, "latency_p95_ms": 10}, base)["passed"]
    g = gate({"groundedness": 0.8, "task_completion": 0.9, "consistency": 1.0, "latency_p95_ms": 10}, base)
    assert not g["passed"] and g["failures"][0]["metric"] == "groundedness" and g["failures"][0]["delta"] == -0.2
    assert not gate({"groundedness": 1.0, "latency_p95_ms": 99999}, base)["passed"]
    assert gate({"groundedness": 0}, None)["passed"]


def test_dataset_is_well_formed():
    ds = load_dataset()
    assert len(ds) >= 15 and len({c["id"] for c in ds}) == len(ds)
    assert all("query" in c and ("required_facts" in c or c.get("expect")) for c in ds)


async def test_full_eval_matches_stored_baseline_and_detects_injected_regression(container):
    result = await run_eval(container, "offline", repeats=2)
    m = result["metrics"]
    assert m["cases"] == len(load_dataset()) and m["completed"] >= m["cases"] - 1                   # the injection case fails by design
    assert gate(m, load_baseline("offline"))["passed"], gate(m, load_baseline("offline"))
    # regression drill: break retrieval so it returns an unrelated document -> the gate must trip
    real = container.retriever.search
    async def broken(query, k=6, doc_ids=None):
        rows = await real("gardening quokka marsupial", k=k, doc_ids=["spark-data-lakehouse"])
        return rows
    container.retriever.search = broken
    try:
        bad = await run_eval(container, "offline", repeats=1)
    finally:
        container.retriever.search = real
    g = gate(bad["metrics"], load_baseline("offline"))
    assert not g["passed"] and {f["metric"] for f in g["failures"]} & {"task_completion", "citation_f1"}
