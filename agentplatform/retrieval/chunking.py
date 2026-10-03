"""Markdown-aware chunking: split on headings, pack sentences to ~target chars with one-sentence overlap, keep code
blocks atomic (truncated), strip badges/HTML/images. Each chunk is prefixed with its heading path so it is self-describing."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .text import sentences

_HEAD = re.compile(r"^(#{1,4})\s+(.*)$")
_NOISE = [re.compile(r"!\[[^\]]*\]\([^)]*\)"), re.compile(r"<[^>]+>"), re.compile(r"^\s*\[!\[.*$", re.M), re.compile(r"^\s*>\s*\*\*▶.*$", re.M)]


@dataclass
class Chunk:
    ordinal: int
    text: str


def clean(md: str) -> str:
    for rx in _NOISE:
        md = rx.sub("", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)       # links -> text
    return re.sub(r"\n{3,}", "\n\n", md).strip()


_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")


def _units(para: str) -> list[str]:
    """Split a paragraph into self-contained units: each list item and table row is its own unit; prose is sentence-split."""
    units: list[str] = []
    prose: list[str] = []

    def flush() -> None:
        if prose:
            units.extend(sentences(" ".join(prose)))
            prose.clear()
    for line in para.splitlines():
        if not line.strip() or re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", line):
            continue
        m = _ITEM.match(line)
        if m:
            flush(); units.append(re.sub(r"\s+", " ", m.group(1)).strip())
        elif line.lstrip().startswith("|"):
            flush(); units.append(" — ".join(c.strip() for c in line.strip().strip("|").split("|") if c.strip()))
        elif units and line.startswith((" ", "\t")) and not prose:   # wrapped continuation of the previous list item
            units[-1] += " " + line.strip()
        else:
            prose.append(line.strip())
    flush()
    return [u for u in units if u]


def chunk_markdown(md: str, target_chars: int = 650, overlap_sentences: int = 1) -> list[Chunk]:
    md = clean(md)
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    buf: list[str] = []
    in_code = False
    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(path), body))
        buf.clear()
    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else _HEAD.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    chunks: list[Chunk] = []
    for head, body in sections:
        prefix = f"[{head}] " if head else ""
        # separate fenced code from prose
        pieces = re.split(r"(```.*?```)", body, flags=re.S)
        sents: list[str] = []
        for p in pieces:
            if p.startswith("```"):
                code = re.sub(r"\s+", " ", p.strip("`").strip())
                sents.append(code[:300])
            else:
                for para in re.split(r"\n\s*\n", p):
                    sents.extend(_units(para))
        cur: list[str] = []
        size = 0
        for s in sents:
            if cur and size + len(s) > target_chars:
                chunks.append(Chunk(len(chunks), prefix + "\n".join(cur)))
                cur = cur[-overlap_sentences:] if overlap_sentences else []
                size = sum(len(x) for x in cur)
            cur.append(s); size += len(s)
        if cur:
            chunks.append(Chunk(len(chunks), prefix + "\n".join(cur)))
    return [c for c in chunks if len(c.text) > 25]
