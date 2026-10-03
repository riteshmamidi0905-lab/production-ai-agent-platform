"""Built-in tools: corpus search (hybrid retrieval), a safe calculator, allow-listed URL fetch, report saving (approval-gated)."""
from __future__ import annotations

import ast
import operator as op
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from .base import EvidenceItem, Risk, Tool, ToolContext, ToolError, ToolRegistry, ToolResult


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    k: int = Field(default=6, ge=1, le=12)
    doc_ids: list[str] | None = None


class SearchCorpus(Tool):
    name = "search_corpus"
    description = "Hybrid (vector + full-text) search over the indexed document corpus. Returns ranked passages with scores."
    args_model = SearchArgs

    async def run(self, args: SearchArgs, ctx: ToolContext) -> ToolResult:
        rows = await ctx.retriever.search(args.query, k=args.k, doc_ids=args.doc_ids)
        ev = [EvidenceItem(doc_id=r["doc_id"], chunk_id=r["id"], title=r["title"], text=r["text"], vector_score=r["vector_score"],
                           lexical_score=r["lexical_score"], fused_score=r["fused"]) for r in rows]
        return ToolResult(text=f"{len(ev)} passage(s) from {len({e.doc_id for e in ev})} document(s)", data={"count": len(ev), "cached": any(r.get('cached') for r in rows)}, evidence=ev)


_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}


def safe_eval(expr: str) -> float:
    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 64:
                raise ToolError("exponent too large")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ToolError("unsupported expression (numbers and + - * / // % ** only)")
    try:
        return ev(ast.parse(expr.strip(), mode="eval").body)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except SyntaxError as exc:
        raise ToolError("invalid expression") from exc


class CalcArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


class Calculator(Tool):
    name = "calculator"
    description = "Evaluate an arithmetic expression exactly (AST-walked; no eval)."
    args_model = CalcArgs

    async def run(self, args: CalcArgs, ctx: ToolContext) -> ToolResult:
        v = safe_eval(args.expression)
        text = f"{args.expression.strip()} = {v:.10g}"
        return ToolResult(text=text, data={"value": v}, evidence=[EvidenceItem(doc_id="tool:calculator", chunk_id=f"calc:{abs(hash(args.expression)) % 10**8}", title="Calculator", text=text, source="tool:calculator", fused_score=1.0)])


class FetchArgs(BaseModel):
    url: str = Field(max_length=500)


class FetchUrl(Tool):
    name = "fetch_url"
    description = "Fetch a web page from an allow-listed host and return its text. Disabled unless AP_ENABLE_FETCH_URL=true."
    args_model = FetchArgs
    risk = Risk.NETWORK

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def run(self, args: FetchArgs, ctx: ToolContext) -> ToolResult:
        s = ctx.settings
        host = urlparse(args.url).hostname or ""
        if not s.enable_fetch_url:
            raise ToolError("fetch_url is disabled on this deployment")
        if urlparse(args.url).scheme != "https" or host not in s.fetch_allowlist:
            raise ToolError(f"host '{host}' is not on the allow-list")
        async with httpx.AsyncClient(timeout=10, follow_redirects=False, transport=self._transport) as c:
            try:
                r = await c.get(args.url, headers={"User-Agent": "agentplatform/0.1"})
            except httpx.HTTPError as exc:
                raise ToolError(f"fetch failed: {exc}", retryable=True) from exc
        if r.status_code >= 500:
            raise ToolError(f"upstream HTTP {r.status_code}", retryable=True)
        if r.status_code >= 400:
            raise ToolError(f"upstream HTTP {r.status_code}")
        text = re.sub(r"\s+", " ", re.sub(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", " ", r.text[:400_000], flags=re.S)).strip()[:6000]
        return ToolResult(text=f"fetched {len(text)} chars from {host}", evidence=[EvidenceItem(doc_id=f"web:{host}", chunk_id=f"web:{abs(hash(args.url)) % 10**8}", title=host, text=text, source="tool:fetch_url", fused_score=0.5)])


class SaveArgs(BaseModel):
    title: str = Field(max_length=200)
    markdown: str = Field(max_length=60_000)


class SaveReport(Tool):
    name = "save_report"
    description = "Persist the final report as a Markdown file. Consequential: always requires human approval unless approvals are disabled."
    args_model = SaveArgs
    risk = Risk.CONSEQUENTIAL
    read_only_hint = False

    async def run(self, args: SaveArgs, ctx: ToolContext) -> ToolResult:
        d = Path(ctx.settings.reports_dir)
        d.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", args.title.lower()).strip("-")[:60] or "report"
        path = d / f"{ctx.run_id}-{slug}.md"
        path.write_text(f"# {args.title}\n\n{args.markdown}\n")
        return ToolResult(text=f"saved {path.name} ({path.stat().st_size} bytes)", data={"path": str(path)})


def default_registry(fetch_transport: httpx.AsyncBaseTransport | None = None) -> ToolRegistry:
    r = ToolRegistry()
    for t in (SearchCorpus(), Calculator(), FetchUrl(fetch_transport), SaveReport()):
        r.register(t)
    return r
