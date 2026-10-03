"""OpenAI-compatible chat provider (works with OpenAI, Azure-style gateways, Ollama, vLLM, LM Studio, OpenRouter...).
Structured output uses `response_format: json_schema` and falls back to prompting for JSON; invalid structure triggers one
repair round-trip with the validation error. Usage comes from the API response. Keys never leave the server."""
from __future__ import annotations

import json
import re
import time

import httpx
from pydantic import ValidationError

from ..prompts import RenderedPrompt
from .base import LLMProvider, ProviderError, ProviderResult, SchemaError, T
from .pricing import usage as mk_usage


def extract_json(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    if text.startswith("{"):
        return text
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        return text[a:b + 1]
    raise SchemaError("no JSON object in model output")


class OpenAICompatProvider(LLMProvider):
    def __init__(self, *, name: str, base_url: str, model: str, api_key: str | None = None, timeout: float = 60.0,
                 transport: httpx.AsyncBaseTransport | None = None, json_schema_mode: bool = True, embedding_model: str | None = None) -> None:
        self.name, self.model, self.base_url = name, model, base_url.rstrip("/")
        self._json_schema_mode, self._embedding_model = json_schema_mode, embedding_model
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, body: dict) -> dict:
        try:
            r = await self._client.post(f"{self.base_url}{path}", json=body)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.name} timeout", transient=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} transport error: {exc}", transient=True) from exc
        if r.status_code in (408, 409, 429) or r.status_code >= 500:
            raise ProviderError(f"{self.name} HTTP {r.status_code}", transient=True, status=r.status_code)
        if r.status_code >= 400:
            raise ProviderError(f"{self.name} HTTP {r.status_code}: {r.text[:200]}", transient=False, status=r.status_code)
        return r.json()

    async def generate_structured(self, prompt: RenderedPrompt, schema: type[T]) -> ProviderResult[T]:
        t0 = time.perf_counter()
        messages = [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user}]
        body: dict = {"model": self.model, "messages": messages, "temperature": 0}
        if self._json_schema_mode:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": False}}
        else:
            body["response_format"] = {"type": "json_object"}
        total_p = total_c = 0
        last_err: Exception | None = None
        for attempt in range(2):                      # second pass = repair round-trip
            data = await self._post("/chat/completions", body)
            u = data.get("usage") or {}
            total_p += int(u.get("prompt_tokens", 0)); total_c += int(u.get("completion_tokens", 0))
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            try:
                parsed = schema.model_validate_json(extract_json(content))
                return ProviderResult(parsed=parsed, usage=mk_usage(self.model, total_p, total_c, estimated=not u), latency_ms=(time.perf_counter() - t0) * 1000, model=self.model, raw=content)
            except (ValidationError, SchemaError, json.JSONDecodeError) as exc:
                last_err = exc
                body["messages"] = messages + [{"role": "assistant", "content": content[:4000]},
                                               {"role": "user", "content": f"Your reply did not match the schema ({str(exc)[:400]}). Return ONLY valid JSON for the schema."}]
        raise SchemaError(f"{self.name}: structured output invalid after repair: {str(last_err)[:200]}")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._embedding_model:
            raise NotImplementedError("no embedding model configured")
        data = await self._post("/embeddings", {"model": self._embedding_model, "input": texts})
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
