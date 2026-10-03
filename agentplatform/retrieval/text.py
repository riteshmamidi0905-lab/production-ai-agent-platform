"""Shared text utilities (tokenisation, sentence splitting, injection screening)."""
from __future__ import annotations

import re

STOP = frozenset("""a an the and or but if then else of to in on at for with without by from is are was were be been being this that
these those it its as into over under i you he she they we me him her them my your our their what which who whom how why when where do does
did can could would should will about between also not no than so such""".split())
_WORD = re.compile(r"[a-z0-9][a-z0-9_\-./]*[a-z0-9]|[a-z0-9]")
_SENT = re.compile(r"(?<=[.!?])[ \t]+(?=[A-Z0-9(\[`*#-])|\n+")


def tokens(text: str, drop_stop: bool = True) -> list[str]:
    toks = _WORD.findall(text.lower())
    return [t for t in toks if t not in STOP] if drop_stop else toks


def sentences(text: str) -> list[str]:
    out = []
    for part in _SENT.split(text.strip()):
        s = re.sub(r"\s+", " ", part).strip()
        if len(s) >= 3:
            out.append(s)
    return out


def words(text: str) -> int:
    return len(text.split())


INJECTION_PATTERNS = [
    (re.compile(r"\bignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)", re.I), "ignore-previous"),
    (re.compile(r"\b(disregard|forget) (all |your |the )?(previous |prior )?(instructions|rules|system prompt)", re.I), "disregard-instructions"),
    (re.compile(r"\byou are now\b.{0,40}\b(unrestricted|jailbroken|dan|developer mode)", re.I), "persona-override"),
    (re.compile(r"\b(reveal|print|show|repeat)\b.{0,30}\b(system prompt|hidden instructions|your instructions)", re.I), "prompt-exfiltration"),
    (re.compile(r"</?(system|assistant|tool)>", re.I), "role-tag-injection"),
    (re.compile(r"\bcall (the )?(save_report|fetch_url|http_fetch|delete|send_email)\b.{0,60}\b(without|skip|bypass)\b", re.I), "tool-abuse"),
]


def injection_flags(text: str) -> list[str]:
    return [label for rx, label in INJECTION_PATTERNS if rx.search(text)]
