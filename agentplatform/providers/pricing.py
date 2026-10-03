"""Cost estimation. Prices are USD per 1M tokens (prompt, completion) and are *editable estimates*:
override or extend with AP_PRICING_JSON='{"model-name":[in,out]}'. Unknown models cost 0 and are flagged."""
from __future__ import annotations

import json
import os

from ..models import Usage

_DEFAULT: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "offline-heuristic-v1": (0.0, 0.0),
    "llama3.1": (0.0, 0.0),  # local
}


def table() -> dict[str, tuple[float, float]]:
    t = dict(_DEFAULT)
    extra = os.environ.get("AP_PRICING_JSON")
    if extra:
        t.update({k: (float(v[0]), float(v[1])) for k, v in json.loads(extra).items()})
    return t


def estimate(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = table().get(model, (0.0, 0.0))
    return round((prompt_tokens * pin + completion_tokens * pout) / 1_000_000, 8)


def usage(model: str, prompt_tokens: int, completion_tokens: int, *, estimated: bool) -> Usage:
    return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, calls=1,
                 cost_usd=estimate(model, prompt_tokens, completion_tokens), estimated=estimated)
