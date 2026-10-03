"""Deterministic, offline *heuristic* provider. It is NOT a language model: it plans with rules and writes extractive,
citation-bearing reports from the evidence it is given. It exists so the whole platform (orchestration, retrieval, tools,
approvals, streaming, evaluation, CI) runs end-to-end with no API key, and so tests are reproducible. Scores measured with it
describe the platform's retrieval/verification plumbing, not language-model quality."""
from __future__ import annotations

import re
import time
from typing import Any

from ..models import Claim, Plan, PlanStep, Report, Usage
from ..prompts import RenderedPrompt
from ..retrieval.text import sentences, tokens, words
from .base import LLMProvider, ProviderResult, SchemaError, T
from .pricing import usage as mk_usage

_ARITH = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?(?:\s*[-+*/x×^%]\s*\d[\d,]*(?:\.\d+)?)+)")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class OfflineProvider(LLMProvider):
    name = "offline"
    model = "offline-heuristic-v1"
    deterministic = True

    async def generate_structured(self, prompt: RenderedPrompt, schema: type[T]) -> ProviderResult[T]:
        t0 = time.perf_counter()
        if schema is Plan:
            parsed: Any = self._plan(prompt.variables)
        elif schema is Report:
            parsed = self._report(prompt.variables)
        else:
            raise SchemaError(f"offline provider cannot produce {schema.__name__}")
        text = parsed.model_dump_json()
        usage: Usage = mk_usage(self.model, max(1, len(prompt.system + prompt.user) // 4), max(1, len(text) // 4), estimated=True)
        return ProviderResult(parsed=parsed, usage=usage, latency_ms=(time.perf_counter() - t0) * 1000, model=self.model, raw=text)

    # ---------------- planning ----------------
    def _plan(self, v: dict[str, Any]) -> Plan:
        query: str = v["query"]
        max_steps: int = int(v.get("max_steps", 6))
        corpus: list[dict[str, str]] = v.get("corpus", [])
        tools = {t["name"] for t in v.get("tools", [])}
        nq = _norm(query)
        mentioned = []
        for d in corpus:
            variants = {_norm(d["doc_id"]), _norm(d.get("title", ""))}
            if any(x and x in nq for x in variants):
                mentioned.append(d["doc_id"])
        steps: list[PlanStep] = []
        if len(mentioned) >= 2:                               # comparison: one scoped retrieval per named document
            for doc in mentioned[: max_steps - 1]:
                steps.append(PlanStep(id=f"s{len(steps)+1}", goal=f"Retrieve evidence about the request from {doc}", tool="search_corpus",
                                      args={"query": query, "doc_ids": [doc]}))
        parts = [p.strip() for p in re.split(r"[?;]| and also | as well as ", query) if len(p.strip()) > 12]
        if len(mentioned) < 2 and len(parts) > 1:             # multi-part question: one retrieval per part
            for p in parts[: max_steps - 1]:
                steps.append(PlanStep(id=f"s{len(steps)+1}", goal=f"Retrieve evidence for: {p[:120]}", tool="search_corpus", args={"query": p}))
        if len(steps) < max_steps:                            # always one unscoped retrieval over the whole request
            steps.append(PlanStep(id=f"s{len(steps)+1}", goal="Retrieve evidence for the full request", tool="search_corpus", args={"query": query}))
        expr = self._expression(query)
        if expr and "calculator" in tools and len(steps) < max_steps:
            steps.append(PlanStep(id=f"s{len(steps)+1}", goal=f"Compute {expr}", tool="calculator", args={"expression": expr}))
        steps = steps[:max_steps] or [PlanStep(id="s1", goal="Search the corpus", tool="search_corpus", args={"query": query})]
        n_search = sum(1 for s in steps if s.tool == "search_corpus")
        summary = f"Run {n_search} retrieval step(s)" + (" and a calculation" if any(s.tool == "calculator" for s in steps) else "") + ", then write a cited report from the evidence."
        return Plan(summary=summary, steps=steps)

    @staticmethod
    def _expression(query: str) -> str | None:
        """Longest balanced arithmetic expression in the request (digits, + - * / ^ % and parentheses)."""
        best = None
        for m in re.finditer(r"[\d(][\d\s+\-*/().^,x×%]*[\d)]", query):
            cand = m.group(0).strip()
            if not re.search(r"\d\s*[-+*/^x×%]\s*[\d(]|\)\s*[-+*/^x×%]", cand) or cand.count("(") != cand.count(")"):
                continue
            if best is None or len(cand) > len(best):
                best = cand
        return best.replace(",", "").replace("x", "*").replace("×", "*").replace("^", "**") if best else None

    # ---------------- synthesis ----------------
    def _report(self, v: dict[str, Any]) -> Report:
        query: str = v["query"]
        evidence: list[dict[str, Any]] = v.get("evidence", [])
        c = v.get("constraints", {})
        max_words = max(15, (c.get("max_words") or 220) - 14)   # leave room for the summary line and [eN] markers
        qt = set(tokens(query))
        title = "Research report: " + (query[:70].rstrip(" ?.") + ("…" if len(query) > 70 else ""))
        if not evidence:
            return Report(title=title, answer="No supporting evidence was found in the corpus for this request.", claims=[],
                          limitations=["No evidence retrieved; nothing can be claimed."])
        cands: list[tuple[float, str, str, str]] = []
        for ev in evidence:
            if ev.get("untrusted_flags"):
                continue
            if ev["source"].startswith("tool:"):
                cands.append((9.0, ev["text"], ev["id"], ev["doc_id"]))
                continue
            for s in sentences(re.sub(r"^\[[^\]]*\]\s*", "", ev["text"])):       # drop the "[heading > path]" chunk prefix
                st = set(tokens(s))
                if len(st) < 4 or sum(s.count(ch) for ch in '{}[]`=<>|') / max(1, len(s)) > 0.05:   # skip code/markup-like fragments
                    continue
                overlap = len(qt & st) / (len(qt) + 1e-9)
                if overlap > 0:
                    cands.append((overlap + 0.15 * float(ev.get("fused_score", 0.0)), s, ev["id"], ev["doc_id"]))
        cands.sort(key=lambda x: -x[0])
        picked: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        per_doc: dict[str, int] = {}
        total = 0
        for _, s, eid, doc in cands:
            key = " ".join(sorted(tokens(s)))[:140]
            if key in seen or per_doc.get(doc, 0) >= 3:
                continue
            w = words(s)
            if total + w > max_words and picked:
                continue
            picked.append((s, eid, doc)); seen.add(key); per_doc[doc] = per_doc.get(doc, 0) + 1; total += w
            if len(picked) >= 8:
                break
        if not picked:
            return Report(title=title, answer="The retrieved passages do not address this request.", claims=[],
                          limitations=["Retrieved evidence had no overlap with the request."])
        lines = ["**Summary.** Passages from the corpus that address the request:", ""]
        claims: list[Claim] = []
        for s, eid, _ in picked:
            s_clean = re.sub(r"\*\*|__|`", "", s).strip()
            lines.append(f"- {s_clean} [{eid}]")
            claims.append(Claim(text=s_clean, evidence_ids=[eid]))
        limitations = []
        if len(picked) < 3:
            limitations.append("Few supporting passages were found; coverage of the request may be incomplete.")
        covered = qt & {t for s, _, _ in picked for t in tokens(s)}
        if qt and len(covered) / len(qt) < 0.4:
            limitations.append("Several terms in the request do not appear in the retrieved passages.")
        return Report(title=title, answer="\n".join(lines), claims=claims, limitations=limitations)
