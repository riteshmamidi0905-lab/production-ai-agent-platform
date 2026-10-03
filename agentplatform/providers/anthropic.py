"""Anthropic Messages API provider. Structured output is obtained by forcing a single tool call whose input_schema is the
Pydantic JSON schema (the standard way to get validated JSON from Claude)."""
from __future__ import annotations

import time

import httpx
from pydantic import ValidationError

from ..prompts import RenderedPrompt
from .base import LLMProvider, ProviderError, ProviderResult, SchemaError, T
from .pricing import usage as mk_usage


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, *, api_key: str, model: str, timeout: float = 60.0, base_url: str = "https://api.anthropic.com",
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.model, self.base_url = model, base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport,
                                         headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate_structured(self, prompt: RenderedPrompt, schema: type[T]) -> ProviderResult[T]:
        t0 = time.perf_counter()
        body = {"model": self.model, "max_tokens": 2048, "temperature": 0, "system": prompt.system,
                "messages": [{"role": "user", "content": prompt.user}],
                "tools": [{"name": "return_result", "description": f"Return the {schema.__name__}.", "input_schema": schema.model_json_schema()}],
                "tool_choice": {"type": "tool", "name": "return_result"}}
        try:
            r = await self._client.post(f"{self.base_url}/v1/messages", json=body)
        except httpx.TimeoutException as exc:
            raise ProviderError("anthropic timeout", transient=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"anthropic transport error: {exc}", transient=True) from exc
        if r.status_code in (408, 429, 529) or r.status_code >= 500:
            raise ProviderError(f"anthropic HTTP {r.status_code}", transient=True, status=r.status_code)
        if r.status_code >= 400:
            raise ProviderError(f"anthropic HTTP {r.status_code}: {r.text[:200]}", status=r.status_code)
        data = r.json()
        block = next((b for b in data.get("content", []) if b.get("type") == "tool_use"), None)
        if block is None:
            raise SchemaError("anthropic: no tool_use block in response")
        try:
            parsed = schema.model_validate(block["input"])
        except ValidationError as exc:
            raise SchemaError(f"anthropic: structured output invalid: {str(exc)[:200]}") from exc
        u = data.get("usage", {})
        return ProviderResult(parsed=parsed, usage=mk_usage(self.model, int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)), estimated=False),
                              latency_ms=(time.perf_counter() - t0) * 1000, model=self.model)
