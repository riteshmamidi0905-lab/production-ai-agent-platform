"""Lightweight tracing + metrics. Spans are persisted per run; metrics are exposed in Prometheus text format."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from .models import Usage


@dataclass
class Span:
    id: str
    name: str
    start: float
    parent: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "parent": self.parent, "start": self.start,
                "duration_ms": None if self.duration_ms is None else round(self.duration_ms, 3),
                "status": self.status, "attrs": self.attrs}


class Tracer:
    """Per-run tracer. Not thread-shared; one instance per run, safe across asyncio tasks of that run."""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._stack: list[str] = []
        self._n = 0

    @asynccontextmanager
    async def span(self, name: str, **attrs: Any):
        self._n += 1
        sp = Span(id=f"s{self._n}", name=name, start=time.time(), parent=self._stack[-1] if self._stack else None, attrs=dict(attrs))
        self.spans.append(sp)
        t0 = time.perf_counter()
        self._stack.append(sp.id)
        try:
            yield sp
        except BaseException as exc:  # noqa: BLE001 - recorded then re-raised
            sp.status = "error"
            sp.attrs["error"] = f"{type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            self._stack.pop()
            sp.duration_ms = (time.perf_counter() - t0) * 1000


class Metrics:
    """Process-wide counters/histograms exported at /metrics (Prometheus exposition format)."""

    BUCKETS = (50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000)

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self.hist: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(list)

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        self.counters[(name, tuple(sorted(labels.items())))] += value

    def observe(self, name: str, value: float, **labels: str) -> None:
        self.hist[(name, tuple(sorted(labels.items())))].append(value)

    def record_usage(self, provider: str, u: Usage) -> None:
        self.inc("agent_llm_prompt_tokens_total", u.prompt_tokens, provider=provider)
        self.inc("agent_llm_completion_tokens_total", u.completion_tokens, provider=provider)
        self.inc("agent_llm_cost_usd_total", u.cost_usd, provider=provider)

    @staticmethod
    def _fmt(labels: tuple[tuple[str, str], ...]) -> str:
        return "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""

    def render(self) -> str:
        out: list[str] = []
        for (name, labels), v in sorted(self.counters.items()):
            out.append(f"{name}{self._fmt(labels)} {v:g}")
        for (name, labels), vals in sorted(self.hist.items()):
            for b in self.BUCKETS:
                lb = labels + (("le", str(b)),)
                out.append(f"{name}_bucket{self._fmt(lb)} {sum(1 for x in vals if x <= b)}")
            out.append(f'{name}_bucket{self._fmt(labels + (("le", "+Inf"),))} {len(vals)}')
            out.append(f"{name}_sum{self._fmt(labels)} {sum(vals):.3f}")
            out.append(f"{name}_count{self._fmt(labels)} {len(vals)}")
        return "\n".join(out) + "\n"


METRICS = Metrics()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
