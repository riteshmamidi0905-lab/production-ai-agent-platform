"""Tool abstraction: every tool declares a typed argument model (JSON-schema derived), a risk class and a result. The same
registry feeds the planner prompt, the executor (validation/timeouts/approvals) and the MCP server."""
from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class Risk(str, enum.Enum):
    READ = "read"                    # side-effect free
    NETWORK = "network"              # reads external resources (allowlisted)
    CONSEQUENTIAL = "consequential"  # writes/changes something: requires human approval by default


class EvidenceItem(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    text: str
    source: str = "corpus"
    vector_score: float | None = None
    lexical_score: float | None = None
    fused_score: float = 0.0


class ToolResult(BaseModel):
    text: str                                  # short human/LLM-readable result
    data: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)


@dataclass
class ToolContext:
    run_id: str
    retriever: Any
    settings: Any
    extras: dict[str, Any] = field(default_factory=dict)


class ToolError(Exception):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class Tool(abc.ABC):
    name: str
    description: str
    args_model: type[BaseModel]
    risk: Risk = Risk.READ
    read_only_hint: bool = True

    @abc.abstractmethod
    async def run(self, args: Any, ctx: ToolContext) -> ToolResult: ...

    def descriptor(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "risk": self.risk.value, "inputSchema": self.args_model.model_json_schema()}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def descriptors(self) -> list[dict[str, Any]]:
        return [t.descriptor() for t in self._tools.values()]

    def for_planner(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "risk": t.risk.value, "args": t.args_model.model_json_schema().get("properties", {})}
                for t in self._tools.values() if t.name != "save_report"]  # save_report is a finalisation action, never planned
