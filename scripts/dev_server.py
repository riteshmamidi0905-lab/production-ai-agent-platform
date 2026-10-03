"""Zero-dependency local run (no Docker): embedded PostgreSQL+pgvector, corpus ingested, API + UI on http://127.0.0.1:8000.
    pip install -e ".[dev]" && python scripts/dev_server.py
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run from any cwd without an install

import pgserver
import uvicorn

from agentplatform.api.app import create_app
from agentplatform.config import Settings
from agentplatform.container import Container


async def main() -> None:
    srv = pgserver.get_server(os.environ.get("AP_DEV_PGDATA") or tempfile.mkdtemp(prefix="agentplatform-dev-"))
    settings = Settings(database_url=srv.get_uri(), reports_dir="./reports")
    container = await Container.build(settings)
    print("indexed:", [(r["doc_id"], r["status"]) for r in await container.ingestor.ingest_dir("corpus")])

    async def factory() -> Container:
        return container
    port = int(os.environ.get("PORT", "8000"))
    await uvicorn.Server(uvicorn.Config(create_app(settings, container_factory=factory), host="127.0.0.1", port=port, log_level="info")).serve()
    await container.close()
    srv.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
