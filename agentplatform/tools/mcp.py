"""MCP-compatible tool layer. Server: exposes the registry over JSON-RPC 2.0 (`initialize`, `tools/list`, `tools/call`, `ping`)
on POST /mcp and on stdio. Client: imports tools from external MCP servers as first-class (risk-gated) registry tools."""
from __future__ import annotations

import json
import sys
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from .base import Risk, Tool, ToolContext, ToolError, ToolRegistry, ToolResult

PROTOCOL_VERSION = "2025-03-26"


class MCPServer:
    def __init__(self, registry: ToolRegistry, ctx_factory, *, name: str = "agentplatform", version: str = "0.1.0", expose_consequential: bool = False) -> None:
        self.registry, self.ctx_factory, self.info = registry, ctx_factory, {"name": name, "version": version}
        self.expose_consequential = expose_consequential  # consequential tools are NOT exposed over MCP unless explicitly enabled

    def _visible(self):
        return [t for t in self.registry._tools.values() if self.expose_consequential or t.risk is not Risk.CONSEQUENTIAL]

    async def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        def ok(result): return {"jsonrpc": "2.0", "id": mid, "result": result}
        def err(code, message): return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}
        if msg.get("jsonrpc") != "2.0" or not method:
            return err(-32600, "invalid request")
        if mid is None:                                    # notification
            return None
        if method == "initialize":
            return ok({"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {"listChanged": False}}, "serverInfo": self.info})
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": [{"name": t.name, "description": t.description, "inputSchema": t.args_model.model_json_schema(),
                                  "annotations": {"readOnlyHint": t.read_only_hint}} for t in self._visible()]})
        if method == "tools/call":
            tool = next((t for t in self._visible() if t.name == params.get("name")), None)
            if tool is None:
                return err(-32602, f"unknown tool: {params.get('name')}")
            try:
                args = tool.args_model.model_validate(params.get("arguments") or {})
                res = await tool.run(args, self.ctx_factory())
                return ok({"content": [{"type": "text", "text": res.text}], "structuredContent": res.data, "isError": False})
            except ToolError as exc:
                return ok({"content": [{"type": "text", "text": str(exc)}], "isError": True})
            except Exception as exc:  # noqa: BLE001 - validation errors etc. are reported in-band per MCP convention
                return ok({"content": [{"type": "text", "text": f"invalid arguments or failure: {str(exc)[:300]}"}], "isError": True})
        return err(-32601, f"method not found: {method}")

    async def serve_stdio(self) -> None:                   # `python -m agentplatform.mcp_stdio`
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                resp = await self.handle(json.loads(line))
            except json.JSONDecodeError:
                resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n"); sys.stdout.flush()


class _Passthrough(BaseModel):
    model_config = ConfigDict(extra="allow")


class MCPRemoteTool(Tool):
    """Adapter for a tool on a remote MCP server. Unless the server marks it readOnlyHint, it is treated as CONSEQUENTIAL."""

    def __init__(self, client: "MCPClient", desc: dict[str, Any]) -> None:
        self.client, self.desc = client, desc
        self.name = f"mcp:{client.label}:{desc['name']}"
        self.description = desc.get("description", "")
        self.args_model = _Passthrough
        ro = bool((desc.get("annotations") or {}).get("readOnlyHint"))
        self.risk, self.read_only_hint = (Risk.NETWORK if ro else Risk.CONSEQUENTIAL), ro

    def descriptor(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "risk": self.risk.value, "inputSchema": self.desc.get("inputSchema", {})}

    async def run(self, args: Any, ctx: ToolContext) -> ToolResult:
        out = await self.client.call(self.desc["name"], args.model_dump())
        text = " ".join(c.get("text", "") for c in out.get("content", []))
        if out.get("isError"):
            raise ToolError(text or "remote tool error")
        return ToolResult(text=text[:2000], data=out.get("structuredContent") or {})


class MCPClient:
    def __init__(self, url: str, label: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.url, self.label = url, label
        self._http = httpx.AsyncClient(timeout=15, transport=transport)
        self._id = 0

    async def _rpc(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        r = await self._http.post(self.url, json={"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise ToolError(f"MCP error {body['error']['code']}: {body['error']['message']}")
        return body["result"]

    async def connect(self) -> list[dict]:
        await self._rpc("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "agentplatform", "version": "0.1.0"}})
        return (await self._rpc("tools/list"))["tools"]

    async def call(self, name: str, arguments: dict) -> dict:
        return await self._rpc("tools/call", {"name": name, "arguments": arguments})

    async def import_into(self, registry: ToolRegistry) -> list[str]:
        names = []
        for d in await self.connect():
            t = MCPRemoteTool(self, d)
            registry.register(t); names.append(t.name)
        return names

    async def aclose(self) -> None:
        await self._http.aclose()
