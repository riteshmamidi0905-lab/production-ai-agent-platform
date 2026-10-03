"""Hybrid retrieval over PostgreSQL: pgvector cosine (HNSW) + full-text rank (GIN), fused with Reciprocal Rank Fusion,
then lightly re-ranked by query-term coverage. Both legs run concurrently. Results are cached (Redis or memory)."""
from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any

from ..cache import Cache
from ..storage.db import Database, vec
from .embeddings import Embedder
from .text import tokens

RRF_K = 60


def or_tsquery(query: str) -> str | None:
    terms = []
    for t in dict.fromkeys(tokens(query)):
        t = re.sub(r"[^a-z0-9]", " ", t).strip().replace(" ", "")
        if len(t) > 1:
            terms.append(t)
    return " | ".join(terms[:24]) or None


class HybridRetriever:
    def __init__(self, db: Database, embedder: Embedder, cache: Cache, *, candidates: int = 24) -> None:
        self.db, self.embedder, self.cache, self.candidates = db, embedder, cache, candidates

    async def _vector(self, qvec: list[float], doc_ids: list[str] | None) -> list[dict[str, Any]]:
        where, args = ("WHERE c.doc_id = ANY(%s)", [doc_ids]) if doc_ids else ("", [])
        return await self.db.fetch(
            f"""SELECT c.id, c.doc_id, c.ordinal, c.text, d.title, 1 - (c.embedding <=> %s::vector) AS score
                FROM chunks c JOIN documents d ON d.id=c.doc_id {where}
                ORDER BY c.embedding <=> %s::vector LIMIT %s""", vec(qvec), *args, vec(qvec), self.candidates)

    async def _lexical(self, query: str, doc_ids: list[str] | None) -> list[dict[str, Any]]:
        tsq = or_tsquery(query)
        if not tsq:
            return []
        where, args = ("AND c.doc_id = ANY(%s)", [doc_ids]) if doc_ids else ("", [])
        return await self.db.fetch(
            f"""SELECT c.id, c.doc_id, c.ordinal, c.text, d.title, ts_rank_cd(c.tsv, to_tsquery('english', %s), 32) AS score
                FROM chunks c JOIN documents d ON d.id=c.doc_id
                WHERE c.tsv @@ to_tsquery('english', %s) {where}
                ORDER BY score DESC LIMIT %s""", tsq, tsq, *args, self.candidates)

    async def search(self, query: str, k: int = 6, doc_ids: list[str] | None = None) -> list[dict[str, Any]]:
        key = "ret:" + hashlib.sha1(f"{self.embedder.name}|{query}|{sorted(doc_ids or [])}|{k}".encode()).hexdigest()
        if (hit := await self.cache.get(key)) is not None:
            return [{**h, "cached": True} for h in hit]
        qvec = (await self.embedder.embed([query]))[0]
        vrows, lrows = await asyncio.gather(self._vector(qvec, doc_ids), self._lexical(query, doc_ids))
        fused: dict[str, dict[str, Any]] = {}
        for rank, r in enumerate(vrows):
            e = fused.setdefault(r["id"], {**{x: r[x] for x in ("id", "doc_id", "ordinal", "text", "title")}, "vector_score": None, "lexical_score": None, "rrf": 0.0})
            e["vector_score"] = float(r["score"]); e["rrf"] += 1.0 / (RRF_K + rank + 1)
        for rank, r in enumerate(lrows):
            e = fused.setdefault(r["id"], {**{x: r[x] for x in ("id", "doc_id", "ordinal", "text", "title")}, "vector_score": None, "lexical_score": None, "rrf": 0.0})
            e["lexical_score"] = float(r["score"]); e["rrf"] += 1.0 / (RRF_K + rank + 1)
        if not fused:
            return []
        top_rrf = max(e["rrf"] for e in fused.values())
        qt = set(tokens(query))
        for e in fused.values():
            cover = len(qt & set(tokens(e["text"]))) / (len(qt) + 1e-9)
            e["fused"] = round(0.7 * (e["rrf"] / top_rrf) + 0.3 * cover, 5)
        out = sorted(fused.values(), key=lambda e: -e["fused"])[:k]
        await self.cache.set(key, out, ttl=600)
        return out
