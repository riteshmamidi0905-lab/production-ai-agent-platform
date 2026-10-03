"""Provider abstraction. The orchestrator only ever calls `generate_structured`; adding a model/vendor means
implementing this one interface (plus optionally `embed`) and registering it in registry.py."""
from __future__ import annotations

import abc
import asyncio
import random
from typing import Awaitable, Callable, Generic, TypeVar

from pydantic import BaseModel

from ..models import Usage
from ..prompts import RenderedPrompt

T = TypeVar("T", bound=BaseModel)


class ProviderError(Exception):
    def __init__(self, message: str, *, transient: bool = False, status: int | None = None):
        super().__init__(message)
        self.transient = transient
        self.status = status


class SchemaError(ProviderError):
    """The model answered, but not with the requested structure."""


class ProviderResult(BaseModel, Generic[T]):
    parsed: T
    usage: Usage
    latency_ms: float
    model: str
    raw: str | None = None


class LLMProvider(abc.ABC):
    name: str = "base"
    model: str = ""
    deterministic: bool = False  # True for the offline provider: lets evals state what their numbers mean

    @abc.abstractmethod
    async def generate_structured(self, prompt: RenderedPrompt, schema: type[T]) -> ProviderResult[T]: ...

    async def embed(self, texts: list[str]) -> list[list[float]]:  # optional capability
        raise NotImplementedError(f"{self.name} does not provide embeddings")


async def with_retries(fn: Callable[[], Awaitable[T]], *, retries: int, base_delay: float = 0.4,
                       on_retry: Callable[[int, Exception], None] | None = None) -> T:
    """Exponential backoff with jitter, only for errors flagged transient."""
    attempt = 0
    while True:
        try:
            return await fn()
        except ProviderError as exc:
            if not exc.transient or attempt >= retries:
                raise
            attempt += 1
            if on_retry:
                on_retry(attempt, exc)
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)) * (0.75 + random.random() / 2))
