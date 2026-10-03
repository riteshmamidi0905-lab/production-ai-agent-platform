"""Embedders. The default is a deterministic signed feature-hashing embedder (words + bigrams + char trigrams): no
model download, reproducible in CI. It is lexical-semantic at best; swap in a real embedding provider via
ProviderEmbedder (the schema dimension comes from AP_EMBEDDING_DIM)."""
from __future__ import annotations

import abc
import hashlib
import math

from .text import tokens


class Embedder(abc.ABC):
    dim: int
    name: str

    @abc.abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder(Embedder):
    name = "hashing-v1"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        toks = tokens(text)
        feats = list(toks) + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        low = text.lower()
        feats += ["#" + low[i:i + 3] for i in range(max(0, len(low) - 2))]
        return feats

    def embed_one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for f in self._features(text):
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class ProviderEmbedder(Embedder):
    def __init__(self, provider, dim: int) -> None:  # provider: LLMProvider with .embed
        self.provider, self.dim, self.name = provider, dim, f"{provider.name}-embed"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = await self.provider.embed(texts)
        if vecs and len(vecs[0]) != self.dim:
            raise ValueError(f"embedding dim {len(vecs[0])} != configured AP_EMBEDDING_DIM {self.dim}")
        return vecs
