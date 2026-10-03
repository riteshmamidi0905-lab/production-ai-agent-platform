"""Guardrails: input screening, untrusted-evidence flagging, output redaction, tool policy."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .retrieval.text import injection_flags

_SECRETS = [re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), re.compile(r"\bAKIA[0-9A-Z]{16}\b"), re.compile(r"\bghp_[A-Za-z0-9]{30,}"),
            re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), re.compile(r"\bxox[bpa]-[A-Za-z0-9-]{10,}")]


@dataclass
class GuardResult:
    allowed: bool
    reasons: list[str]


def check_input(query: str) -> GuardResult:
    reasons = [f"prompt-injection pattern: {f}" for f in injection_flags(query)]
    if any(rx.search(query) for rx in _SECRETS):
        reasons.append("input contains what looks like a credential")
    return GuardResult(not reasons, reasons)


def screen_evidence(text: str) -> list[str]:
    """Retrieved text is untrusted data. Flagged passages are still shown to the user but never fed to the synthesizer."""
    return injection_flags(text)


def redact(text: str) -> str:
    for rx in _SECRETS:
        text = rx.sub("[REDACTED]", text)
    return text
