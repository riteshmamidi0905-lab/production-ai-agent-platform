"""MCP over stdio: `python -m agentplatform.mcp_stdio` (newline-delimited JSON-RPC 2.0 on stdin/stdout).

Uses the same tool registry and the same read-only exposure rule as POST /mcp: consequential tools are not listed or
callable. Needs a reachable database (AP_DATABASE_URL), like the rest of the platform. Logs go to stderr, never stdout.
"""
from __future__ import annotations

import asyncio

from .config import Settings
from .container import Container


async def amain() -> None:
    c = await Container.build(Settings())
    try:
        await c.mcp.serve_stdio()
    finally:
        await c.close()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
