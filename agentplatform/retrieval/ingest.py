"""Document ingestion: chunk -> embed (batched) -> upsert. Idempotent via content hash (unchanged docs are skipped)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..storage.db import J, vec
from ..storage.repo import Repo
from .chunking import chunk_markdown
from .embeddings import Embedder


class Ingestor:
    def __init__(self, repo: Repo, embedder: Embedder) -> None:
        self.repo, self.embedder = repo, embedder

    async def ingest_text(self, doc_id: str, title: str, text: str, *, source: str = "corpus", meta: dict | None = None, force: bool = False) -> dict:
        sha = hashlib.sha256(text.encode()).hexdigest()
        if not force and await self.repo.doc_sha(doc_id) == sha:
            return {"doc_id": doc_id, "status": "unchanged", "chunks": 0}
        chunks = chunk_markdown(text)
        if not chunks:
            raise ValueError(f"{doc_id}: no indexable content")
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), 64):
            vectors += await self.embedder.embed([c.text for c in chunks[i:i + 64]])
        async with self.repo.db.pool.connection() as c, c.transaction():
            await c.execute("DELETE FROM documents WHERE id=%s", (doc_id,))
            await c.execute("INSERT INTO documents(id, title, source, content_sha, meta) VALUES (%s,%s,%s,%s,%s)", (doc_id, title, source, sha, J(meta or {})))
            async with c.cursor() as cur:
                await cur.executemany("INSERT INTO chunks(id, doc_id, ordinal, text, embedding) VALUES (%s,%s,%s,%s,%s::vector)",
                                      [(f"{doc_id}::{ch.ordinal}", doc_id, ch.ordinal, ch.text, vec(v)) for ch, v in zip(chunks, vectors)])
        return {"doc_id": doc_id, "status": "indexed", "chunks": len(chunks)}

    @staticmethod
    def title_of(text: str, default: str) -> str:
        """First level-1 heading that is not inside a code fence."""
        fenced = False
        for ln in text.splitlines():
            if ln.strip().startswith("```"):
                fenced = not fenced
            elif not fenced and ln.startswith("# "):
                return ln[2:].strip()
        return default

    async def ingest_dir(self, path: str | Path) -> list[dict]:
        out = []
        for p in sorted(Path(path).glob("*.md")):
            text = p.read_text()
            m = self.title_of(text, p.stem)
            out.append(await self.ingest_text(p.stem, m, text, meta={"file": p.name}))
        return out
