"""Versioned prompt registry. Every run records the exact prompt name@version#hash it used."""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_DIR = Path(__file__).parent


class PromptTemplate(BaseModel):
    name: str
    version: str
    system: str
    user: str
    notes: str = ""

    @property
    def sha(self) -> str:
        return hashlib.sha256((self.system + "\n" + self.user).encode()).hexdigest()[:10]

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}#{self.sha}"


class RenderedPrompt(BaseModel):
    name: str
    version: str
    sha: str
    system: str
    user: str
    variables: dict[str, Any] = {}  # structured inputs, used by the offline provider (no LLM to read the text)

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}#{self.sha}"


@lru_cache
def load_all() -> dict[str, PromptTemplate]:
    out: dict[str, PromptTemplate] = {}
    for p in sorted(_DIR.glob("*.json")):
        t = PromptTemplate.model_validate_json(p.read_text())
        out[t.name] = t
    return out


_VAR = re.compile(r"\{\{(\w+)\}\}")


def render(name: str, **variables: Any) -> RenderedPrompt:
    t = load_all()[name]

    def sub(text: str) -> str:
        return _VAR.sub(lambda m: variables[m.group(1)] if isinstance(variables.get(m.group(1)), str) else json.dumps(variables.get(m.group(1)), ensure_ascii=False), text)

    return RenderedPrompt(name=t.name, version=t.version, sha=t.sha, system=sub(t.system), user=sub(t.user), variables=variables)


def catalog() -> list[dict[str, str]]:
    return [{"name": t.name, "version": t.version, "sha": t.sha, "notes": t.notes} for t in load_all().values()]
