"""Typed, environment-driven configuration. Secrets only ever come from the environment (see .env.example)."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AP_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://agent:agent@localhost:5432/agent"
    redis_url: str | None = None  # optional: cross-process event fan-out + retrieval cache
    api_key: str | None = None  # optional shared secret for the HTTP API (X-API-Key / ?api_key=)

    # providers (keys are server-side only)
    default_provider: str = "offline"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"
    provider_fallbacks: list[str] = Field(default_factory=lambda: ["offline"])

    # retrieval
    embedding_dim: int = 384
    retrieval_top_k: int = 6
    retrieval_candidates: int = 24

    # orchestration limits
    max_plan_steps: int = 6
    max_evidence: int = 10
    provider_timeout_s: float = 60.0
    tool_timeout_s: float = 20.0
    approval_timeout_s: float = 300.0
    provider_retries: int = 2
    tool_retries: int = 1
    run_budget_usd: float = 1.0  # hard stop on estimated spend per run

    # tools
    reports_dir: str = "./reports"
    enable_fetch_url: bool = False
    fetch_allowlist: list[str] = Field(default_factory=list)
    mcp_servers: list[str] = Field(default_factory=list)  # URLs of external MCP servers to import tools from

    cors_origins: list[str] = Field(default_factory=list)


@lru_cache
def get_settings() -> Settings:
    return Settings()
