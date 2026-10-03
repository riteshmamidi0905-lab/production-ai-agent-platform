"""Online verification of the generated report: citation validity + claim support. (The offline evaluation suite uses a separate,
stricter implementation in eval/metrics.py so the platform does not grade itself with its own checker.)"""
from __future__ import annotations

import re

from ..models import Evidence, Report, Verification
from ..retrieval.text import tokens

_CITE = re.compile(r"\[(e\d+)\]")


def support_ratio(claim: str, evidence_texts: list[str]) -> float:
    ct = set(tokens(claim))
    if not ct:
        return 1.0
    pool = set()
    for t in evidence_texts:
        pool |= set(tokens(t))
    return len(ct & pool) / len(ct)


def verify(report: Report, evidence: dict[str, Evidence], *, require_citations: bool, threshold: float = 0.7) -> Verification:
    invalid: list[str] = []
    unsupported: list[str] = []
    valid_cites = ok_claims = 0
    total_cites = 0
    for c in report.claims:
        ids = c.evidence_ids
        good = [i for i in ids if i in evidence]
        total_cites += len(ids); valid_cites += len(good)
        invalid += [i for i in ids if i not in evidence]
        if not good:
            if require_citations:
                unsupported.append(c.text)
            continue
        if support_ratio(c.text, [evidence[i].text for i in good]) >= threshold:
            ok_claims += 1
        else:
            unsupported.append(c.text)
    for m in _CITE.findall(report.answer):                  # inline citations must also resolve
        total_cites += 1
        if m in evidence:
            valid_cites += 1
        else:
            invalid.append(m)
    n = len(report.claims)
    grounded = ok_claims / n if n else 0.0
    cite_valid = valid_cites / total_cites if total_cites else (0.0 if require_citations else 1.0)
    notes = []
    if not n:
        notes.append("report contains no claims")
    ok = bool(n) and grounded >= 0.8 and not invalid and (cite_valid >= 1.0 or not require_citations)
    return Verification(ok=ok, groundedness=round(grounded, 4), citation_validity=round(cite_valid, 4), unsupported_claims=unsupported, invalid_citations=sorted(set(invalid)), notes=notes)


def strip_unsupported(report: Report, unsupported: list[str], invalid: list[str]) -> Report:
    """Last-resort repair: remove unsupported claims and the answer lines that carry them, and record the omission."""
    bad = set(unsupported)
    lines = [ln for ln in report.answer.splitlines() if not any(b and b[:60] in ln for b in bad) and not any(f"[{i}]" in ln for i in invalid)]
    kept = [c for c in report.claims if c.text not in bad and all(i not in invalid for i in c.evidence_ids)]
    lim = list(report.limitations)
    if len(kept) != len(report.claims):
        lim.append(f"{len(report.claims) - len(kept)} claim(s) were removed because they were not supported by the retrieved evidence.")
    return Report(title=report.title, answer="\n".join(lines), claims=kept, limitations=lim)
