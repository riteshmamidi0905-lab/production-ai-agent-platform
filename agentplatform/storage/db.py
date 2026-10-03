"""Async PostgreSQL access (psycopg 3 pool). The schema is created idempotently at startup; vector dimension is configurable."""
from __future__ import annotations

from importlib import resources
from typing import Any, Sequence

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool


def vec(v: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


class Database:
    def __init__(self, url: str, *, min_size: int = 1, max_size: int = 10) -> None:
        self.url = url
        self.pool = AsyncConnectionPool(url, min_size=min_size, max_size=max_size, open=False, kwargs={"row_factory": dict_row, "autocommit": True})

    async def open(self, embedding_dim: int) -> None:
        await self.pool.open(wait=True, timeout=30)
        ddl = resources.files("agentplatform.storage").joinpath("schema.sql").read_text().replace("{dim}", str(int(embedding_dim)))
        async with self.pool.connection() as c:
            await c.execute(ddl)
            row = await (await c.execute("SELECT atttypmod AS d FROM pg_attribute WHERE attrelid='chunks'::regclass AND attname='embedding'")).fetchone()
            if row and row["d"] not in (-1, embedding_dim):
                raise RuntimeError(f"existing chunks.embedding is vector({row['d']}) but AP_EMBEDDING_DIM={embedding_dim}; re-create the table or fix the setting")

    async def close(self) -> None:
        await self.pool.close()

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        async with self.pool.connection() as c:
            cur = await c.execute(sql, args or None)
            return await cur.fetchall() if cur.description else []

    async def one(self, sql: str, *args: Any) -> dict[str, Any] | None:
        rows = await self.fetch(sql, *args)
        return rows[0] if rows else None

    async def execute(self, sql: str, *args: Any) -> int:
        async with self.pool.connection() as c:
            cur = await c.execute(sql, args or None)
            return cur.rowcount


def J(obj: Any) -> Jsonb:
    return Jsonb(obj)
