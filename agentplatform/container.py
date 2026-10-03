"""Composition root: builds every component once; used by the API, the CLI and the evaluation runner so they share identical wiring."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from .cache import Cache, make_cache
from .config import Settings, get_settings
from .orchestrator.approvals import ApprovalBroker
from .orchestrator.engine import AgentEngine
from .orchestrator.events import EventHub
from .providers.registry import ProviderPool
from .retrieval.embeddings import Embedder, HashingEmbedder
from .retrieval.hybrid import HybridRetriever
from .retrieval.ingest import Ingestor
from .storage.db import Database
from .storage.repo import Repo
from .tools.base import ToolContext, ToolRegistry
from .tools.builtin import default_registry
from .tools.mcp import MCPClient, MCPServer


@dataclass
class Container:
    settings: Settings
    db: Database
    repo: Repo
    cache: Cache
    embedder: Embedder
    retriever: HybridRetriever
    ingestor: Ingestor
    providers: ProviderPool
    registry: ToolRegistry
    hub: EventHub
    broker: ApprovalBroker
    engine: AgentEngine
    mcp: MCPServer
    mcp_clients: list[MCPClient]

    @classmethod
    async def build(cls, settings: Settings | None = None, *, cache: Cache | None = None, embedder: Embedder | None = None, db: Database | None = None,
                    fetch_transport: httpx.AsyncBaseTransport | None = None, registry: ToolRegistry | None = None) -> "Container":
        s = settings or get_settings()
        db = db or Database(s.database_url)
        await db.open(s.embedding_dim)
        repo, cache = Repo(db), cache or make_cache(s.redis_url)
        embedder = embedder or HashingEmbedder(s.embedding_dim)
        retriever = HybridRetriever(db, embedder, cache, candidates=s.retrieval_candidates)
        providers = ProviderPool(s)
        registry = registry or default_registry(fetch_transport)
        clients: list[MCPClient] = []
        for i, url in enumerate(s.mcp_servers):              # import tools from external MCP servers (risk-gated like any other tool)
            c = MCPClient(url, label=f"srv{i}")
            try:
                await c.import_into(registry)
                clients.append(c)
            except Exception:  # noqa: BLE001 - a dead MCP server must not stop the platform from starting
                await c.aclose()
        hub, broker = EventHub(repo, cache), ApprovalBroker(repo, cache)
        engine = AgentEngine(settings=s, repo=repo, retriever=retriever, providers=providers, registry=registry, cache=cache, hub=hub, broker=broker)
        mcp = MCPServer(registry, lambda: ToolContext(run_id="mcp", retriever=retriever, settings=s))
        await repo.recover_interrupted()
        return cls(s, db, repo, cache, embedder, retriever, Ingestor(repo, embedder), providers, registry, hub, broker, engine, mcp, clients)

    async def close(self) -> None:
        for t in list(self.engine.tasks.values()):
            t.cancel()
        await self.providers.aclose()
        for c in self.mcp_clients:
            await c.aclose()
        await self.cache.close()
        await self.db.close()
