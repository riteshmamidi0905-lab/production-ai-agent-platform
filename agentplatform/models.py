"""Pydantic domain models. Everything crossing a boundary (API, provider, tool, DB) is validated through these."""
from __future__ import annotations

import enum
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class RunState(str, enum.Enum):
    CREATED = "created"
    GUARDING = "guarding"
    PLANNING = "planning"
    EXECUTING = "executing"
    AWAITING_APPROVAL = "awaiting_approval"
    SYNTHESIZING = "synthesizing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"  # process died mid-run; recovered at startup

    @property
    def terminal(self) -> bool:
        return self in {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED, RunState.INTERRUPTED}


class FailureClass(str, enum.Enum):
    GUARDRAIL_BLOCK = "guardrail_block"
    PROVIDER_ERROR = "provider_error"
    PROVIDER_TIMEOUT = "provider_timeout"
    SCHEMA_VALIDATION = "schema_validation"
    INVALID_PLAN = "invalid_plan"
    TOOL_ERROR = "tool_error"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_ARGS_INVALID = "tool_args_invalid"
    NO_EVIDENCE = "no_evidence"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_TIMEOUT = "approval_timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    VERIFICATION_FAILED = "verification_failed"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


class RunConstraints(BaseModel):
    max_steps: int = Field(default=6, ge=1, le=12)
    require_citations: bool = True
    max_words: int | None = Field(default=None, ge=20, le=2000)
    format: Literal["report", "answer"] = "report"
    require_approval_for: Literal["consequential", "all_tools", "none"] = "consequential"
    save_report: bool = False  # asks the agent to persist the result with the consequential `save_report` tool


class RunRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000)
    provider: str | None = None
    conversation_id: str | None = None
    constraints: RunConstraints = Field(default_factory=RunConstraints)
    idempotency_key: str | None = Field(default=None, max_length=120)

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


# ---------- planning ----------
class PlanStep(BaseModel):
    id: str
    goal: str = Field(max_length=300)
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """A structured plan. `summary` is a short, user-safe description, never hidden reasoning."""
    summary: str = Field(max_length=500)
    steps: list[PlanStep] = Field(min_length=1)


# ---------- evidence + report ----------
class Evidence(BaseModel):
    id: str  # e1, e2, ... stable within a run; this is what citations reference
    doc_id: str
    chunk_id: str
    title: str
    text: str
    source: str = "corpus"  # corpus | tool:<name>
    vector_score: float | None = None
    lexical_score: float | None = None
    fused_score: float = 0.0
    step_id: str | None = None
    untrusted_flags: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class Report(BaseModel):
    title: str
    answer: str  # markdown; inline citations as [e1]
    claims: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class Verification(BaseModel):
    ok: bool
    groundedness: float
    citation_validity: float
    unsupported_claims: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    repaired: bool = False
    notes: list[str] = Field(default_factory=list)


# ---------- usage / telemetry ----------
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    estimated: bool = True  # True when token counts are approximated (offline provider)

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls
        self.cost_usd = round(self.cost_usd + other.cost_usd, 8)
        self.estimated = self.estimated and other.estimated


class ToolCallRecord(BaseModel):
    id: str
    tool: str
    args: dict[str, Any]
    status: Literal["ok", "error", "timeout", "denied", "invalid_args"]
    duration_ms: float
    attempts: int = 1
    error: str | None = None
    result_preview: str | None = None


class Event(BaseModel):
    run_id: str
    seq: int
    type: str
    ts: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    id: str
    conversation_id: str | None = None
    query: str
    provider: str
    state: RunState
    constraints: RunConstraints
    plan: Plan | None = None
    report: Report | None = None
    verification: Verification | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float | None = None
    failure_class: FailureClass | None = None
    error: str | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    created_at: float
    finished_at: float | None = None
