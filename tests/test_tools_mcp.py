import httpx
import pytest

from agentplatform.tools.base import ToolContext, ToolError
from agentplatform.tools.builtin import FetchUrl, SaveReport, default_registry, safe_eval
from agentplatform.tools.mcp import MCPClient, MCPServer


@pytest.mark.parametrize("expr,val", [("2+3*4", 14), ("(600/5000 - 500/5000) * 100", 2.0), ("-2**2", -4), ("7//2", 3)])
def test_calculator_values(expr, val):
    assert safe_eval(expr) == pytest.approx(val)


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "1/0", "2**9999", "a+1", "[1,2]"])
def test_calculator_rejects_everything_else(expr):
    with pytest.raises((ToolError, Exception)):
        safe_eval(expr)


def ctx(c, tmp=None):
    return ToolContext(run_id="t", retriever=c.retriever, settings=c.settings)


async def test_save_report_writes_inside_reports_dir(container):
    r = await SaveReport().run(SaveReport.args_model(title="../../etc/passwd evil", markdown="body"), ctx(container))
    from pathlib import Path
    p = Path(r.data["path"]).resolve()
    assert Path(container.settings.reports_dir).resolve() in p.parents and "passwd" in p.name and ".." not in p.name


async def test_fetch_url_disabled_and_allowlist(container):
    t = FetchUrl(httpx.MockTransport(lambda req: httpx.Response(200, text="<html><script>x</script><p>Hello   world</p></html>")))
    args = FetchUrl.args_model(url="https://example.org/a")
    with pytest.raises(ToolError, match="disabled"):
        await t.run(args, ctx(container))
    container.settings.enable_fetch_url = True
    try:
        with pytest.raises(ToolError, match="allow-list"):
            await t.run(args, ctx(container))
        container.settings.fetch_allowlist = ["example.org"]
        r = await t.run(args, ctx(container))
        assert "Hello world" in r.evidence[0].text and "script" not in r.evidence[0].text
        with pytest.raises(ToolError, match="allow-list"):          # http (non-TLS) is refused even for allowed hosts
            await t.run(FetchUrl.args_model(url="http://example.org/a"), ctx(container))
    finally:
        container.settings.enable_fetch_url, container.settings.fetch_allowlist = False, []


async def test_mcp_server_roundtrip_and_client_import(container):
    server = MCPServer(container.registry, lambda: ctx(container))
    init = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["capabilities"]["tools"] is not None
    tools = (await server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"search_corpus", "calculator"} <= names and "save_report" not in names     # consequential tools are not exposed over MCP
    call = await server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "calculator", "arguments": {"expression": "6*7"}}})
    assert call["result"]["content"][0]["text"].endswith("= 42") and call["result"]["isError"] is False
    bad = await server.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "calculator", "arguments": {"expression": "1/0"}}})
    assert bad["result"]["isError"] is True
    assert (await server.handle({"jsonrpc": "2.0", "id": 5, "method": "nope"}))["error"]["code"] == -32601
    assert await server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    async def asgi(request: httpx.Request) -> httpx.Response:      # an MCP client talking to our own server (HTTP JSON-RPC)
        import json
        resp = await server.handle(json.loads(request.content))
        return httpx.Response(200, json=resp)
    client = MCPClient("http://mcp.local/mcp", "self", transport=httpx.MockTransport(asgi))
    reg = default_registry()
    imported = await client.import_into(reg)
    assert "mcp:self:calculator" in imported
    remote = reg.get("mcp:self:calculator")
    assert remote.risk.value == "network"                          # readOnlyHint -> not consequential
    out = await remote.run(remote.args_model(expression="2+2"), ctx(container))
    assert out.text.endswith("= 4")
    await client.aclose()


async def test_mcp_remote_tool_without_readonly_hint_is_consequential():
    from agentplatform.tools.mcp import MCPRemoteTool
    t = MCPRemoteTool(MCPClient("http://x", "x"), {"name": "delete_everything", "description": "d", "inputSchema": {}})
    assert t.risk.value == "consequential"
