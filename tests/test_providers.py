import json

import httpx
import pytest

from agentplatform.models import Plan, Report
from agentplatform.prompts import catalog, render
from agentplatform.providers.anthropic import AnthropicProvider
from agentplatform.providers.base import ProviderError, SchemaError, with_retries
from agentplatform.providers.openai_compat import OpenAICompatProvider, extract_json
from agentplatform.providers.pricing import estimate

PLAN = {"summary": "s", "steps": [{"id": "s1", "goal": "g", "tool": "search_corpus", "args": {"query": "q"}}]}


def completion(content, p=100, c=50):
    return {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": p, "completion_tokens": c}}


def prov(handler, **kw):
    return OpenAICompatProvider(name="openai", base_url="http://x/v1", model="gpt-4o-mini", api_key="k", transport=httpx.MockTransport(handler), **kw)


async def test_openai_compat_structured_output_usage_and_cost():
    seen = {}
    def h(req):
        seen["body"], seen["auth"] = json.loads(req.content), req.headers["authorization"]
        return httpx.Response(200, json=completion("```json\n" + json.dumps(PLAN) + "\n```", 1000, 200))
    p = prov(h)
    r = await p.generate_structured(render("planner", query="q", max_steps=3, tools=[], tools_json=[], corpus=[], corpus_json=[], context=""), Plan)
    assert r.parsed.steps[0].tool == "search_corpus" and r.usage.prompt_tokens == 1000 and not r.usage.estimated
    assert r.usage.cost_usd == pytest.approx(estimate("gpt-4o-mini", 1000, 200)) and r.usage.cost_usd > 0
    assert seen["auth"] == "Bearer k" and seen["body"]["response_format"]["type"] == "json_schema" and seen["body"]["temperature"] == 0


async def test_openai_compat_repairs_invalid_structure_once():
    calls = []
    def h(req):
        calls.append(json.loads(req.content)["messages"])
        return httpx.Response(200, json=completion("not json at all" if len(calls) == 1 else json.dumps(PLAN)))
    r = await prov(h).generate_structured(render("planner", query="q", max_steps=3, tools=[], tools_json=[], corpus=[], corpus_json=[], context=""), Plan)
    assert len(calls) == 2 and "did not match the schema" in calls[1][-1]["content"] and r.usage.prompt_tokens == 200  # both attempts are billed/counted


async def test_openai_compat_gives_up_with_schema_error():
    p = prov(lambda req: httpx.Response(200, json=completion("{}")))
    with pytest.raises(SchemaError):
        await p.generate_structured(render("planner", query="q", max_steps=3, tools=[], tools_json=[], corpus=[], corpus_json=[], context=""), Plan)


@pytest.mark.parametrize("status,transient", [(429, True), (503, True), (400, False), (401, False)])
async def test_error_classification(status, transient):
    p = prov(lambda req: httpx.Response(status, text="no"))
    with pytest.raises(ProviderError) as ei:
        await p.generate_structured(render("planner", query="q", max_steps=3, tools=[], tools_json=[], corpus=[], corpus_json=[], context=""), Plan)
    assert ei.value.transient is transient


async def test_with_retries_only_retries_transient():
    n = {"a": 0, "b": 0}
    async def flaky():
        n["a"] += 1
        if n["a"] < 3:
            raise ProviderError("x", transient=True)
        return "ok"
    assert await with_retries(flaky, retries=2, base_delay=0.001) == "ok" and n["a"] == 3
    async def hard():
        n["b"] += 1
        raise ProviderError("x", transient=False)
    with pytest.raises(ProviderError):
        await with_retries(hard, retries=5, base_delay=0.001)
    assert n["b"] == 1


async def test_anthropic_tool_use_structured_output():
    seen = {}
    def h(req):
        seen["body"], seen["h"] = json.loads(req.content), req.headers
        return httpx.Response(200, json={"content": [{"type": "tool_use", "name": "return_result", "input": {"title": "t", "answer": "a [e1]", "claims": [{"text": "a", "evidence_ids": ["e1"]}]}}], "usage": {"input_tokens": 10, "output_tokens": 5}})
    p = AnthropicProvider(api_key="k", model="claude-test", transport=httpx.MockTransport(h))
    r = await p.generate_structured(render("synthesizer", query="q", evidence_block="", constraints_json={}, feedback="none", evidence=[], constraints={}), Report)
    assert r.parsed.claims[0].evidence_ids == ["e1"] and seen["body"]["tool_choice"] == {"type": "tool", "name": "return_result"} and seen["h"]["x-api-key"] == "k"


def test_extract_json_variants():
    assert extract_json('prefix {"a": 1} suffix') == '{"a": 1}'
    with pytest.raises(SchemaError):
        extract_json("nothing here")


def test_prompt_registry_versions_and_hashes():
    cat = {c["name"]: c for c in catalog()}
    assert {"planner", "synthesizer"} <= set(cat) and all(c["sha"] and c["version"] for c in cat.values())
    a = render("synthesizer", query="q", evidence_block="[e1] x", constraints_json={}, feedback="none", evidence=[], constraints={})
    assert a.ref.startswith("synthesizer@") and "[e1] x" in a.user and "{{" not in a.user
