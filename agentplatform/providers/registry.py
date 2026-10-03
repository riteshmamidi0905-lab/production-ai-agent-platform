"""Provider registry + fallback chain. Register a new provider with `register("name", factory)`; nothing else changes."""
from __future__ import annotations

from typing import Callable

from ..config import Settings
from .base import LLMProvider, ProviderError
from .offline import OfflineProvider

_FACTORIES: dict[str, Callable[[Settings], LLMProvider]] = {}


def register(name: str, factory: Callable[[Settings], LLMProvider]) -> None:
    _FACTORIES[name] = factory


def _openai(s: Settings) -> LLMProvider:
    from .openai_compat import OpenAICompatProvider
    if not s.openai_api_key:
        raise ProviderError("AP_OPENAI_API_KEY is not set")
    return OpenAICompatProvider(name="openai", base_url=s.openai_base_url, model=s.openai_model, api_key=s.openai_api_key, timeout=s.provider_timeout_s)


def _ollama(s: Settings) -> LLMProvider:
    from .openai_compat import OpenAICompatProvider
    return OpenAICompatProvider(name="ollama", base_url=s.ollama_base_url, model=s.ollama_model, timeout=s.provider_timeout_s, json_schema_mode=False)


def _anthropic(s: Settings) -> LLMProvider:
    from .anthropic import AnthropicProvider
    if not s.anthropic_api_key:
        raise ProviderError("AP_ANTHROPIC_API_KEY is not set")
    return AnthropicProvider(api_key=s.anthropic_api_key, model=s.anthropic_model, timeout=s.provider_timeout_s)


register("offline", lambda s: OfflineProvider())
register("openai", _openai)
register("ollama", _ollama)
register("anthropic", _anthropic)


class ProviderPool:
    """Resolves providers lazily and caches them; `chain(name)` yields the requested provider then the configured fallbacks."""

    def __init__(self, settings: Settings) -> None:
        self.settings, self._cache = settings, {}

    def get(self, name: str) -> LLMProvider:
        if name not in self._cache:
            if name not in _FACTORIES:
                raise ProviderError(f"unknown provider '{name}'. Available: {sorted(_FACTORIES)}")
            self._cache[name] = _FACTORIES[name](self.settings)
        return self._cache[name]

    def available(self) -> list[str]:
        out = []
        for n in _FACTORIES:
            try:
                self.get(n); out.append(n)
            except ProviderError:
                pass
        return out

    def chain(self, name: str) -> list[LLMProvider]:
        names = [name] + [n for n in self.settings.provider_fallbacks if n != name]
        chain: list[LLMProvider] = []
        for n in names:
            try:
                chain.append(self.get(n))
            except ProviderError:
                if n == name:
                    raise
        return chain

    async def aclose(self) -> None:
        for p in self._cache.values():
            close = getattr(p, "aclose", None)
            if close:
                await close()
