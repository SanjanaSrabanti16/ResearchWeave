"""Cheap, deterministic evidence selection before local insight generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.document import ParsedPaper

FIELD_POOLS = (
    "contributions",
    "overview",
    "design_methods",
    "data_methods",
    "system_methods",
    "evaluation_methods",
    "evaluation",
    "findings",
    "discussion",
    "limitations",
    "future_work",
    "methods",
    "gaps",
)
POOL_BUDGETS = {
    "contributions": 4,
    "overview": 6,
    "design_methods": 4,
    "data_methods": 4,
    "system_methods": 6,
    "evaluation_methods": 4,
    "evaluation": 8,
    "findings": 10,
    "discussion": 5,
    "limitations": 6,
    "future_work": 5,
    "methods": 14,
    "gaps": 6,
}
GUARANTEED_POOL_COUNTS = {
    "contributions": 2,
    "overview": 2,
    "design_methods": 1,
    "data_methods": 1,
    "system_methods": 1,
    "evaluation_methods": 1,
    "evaluation": 3,
    "discussion": 1,
    "limitations": 2,
    "future_work": 2,
}
HEADING_SIGNALS = {
    "contributions": ("abstract", "introduction", "contribution", "conclusion"),
    "overview": ("abstract", "introduction", "background", "conclusion", "summary"),
    "design_methods": ("method", "design", "methodology", "requirements"),
    "data_methods": ("method", "data", "dataset", "harmonization", "processing"),
    "system_methods": ("method", "system", "architecture", "algorithm", "model", "panel"),
    "evaluation_methods": (
        "evaluation",
        "experiment",
        "user study",
        "case study",
        "expert feedback",
    ),
    "evaluation": (
        "evaluation",
        "result",
        "case study",
        "investigation",
        "expert feedback",
    ),
    "methods": (
        "method",
        "model",
        "architecture",
        "approach",
        "algorithm",
        "attention",
        "data",
        "training",
        "experiment",
        "optimizer",
        "regularization",
        "encoding",
        "design",
    ),
    "findings": (
        "result",
        "evaluation",
        "experiment",
        "finding",
        "performance",
        "analysis",
        "conclusion",
        "benchmark",
        "translation",
        "parsing",
    ),
    "discussion": ("discussion", "conclusion"),
    "limitations": ("limitation", "constraint", "threat"),
    "future_work": ("future work", "future direction", "further work"),
    "gaps": ("limitation", "discussion", "conclusion", "future", "open question"),
}
TEXT_SIGNALS = {
    "contributions": (
        r"\b(?:main|key|primary) contributions?\b",
        r"\b(?:our contributions?|we contribute|contributions? of this work)\b",
        r"\bcontributions?\b.{0,80}\b(?:1|one)[.)]",
    ),
    "overview": (
        r"\bwe (?:propose|present|introduce|address|show|demonstrate)\b",
        r"\b(?:problem|challenge|contribution|important|significant)\b",
    ),
    "design_methods": (
        r"\b(?:we (?:followed|used|conducted)|design methodology|design process)\b",
        r"\b(?:activity-centered|human-centered|participatory design|prototype|interview|"
        r"requirements?)\b",
    ),
    "data_methods": (
        r"\b(?:we (?:collected|processed|harmonized|reconciled|labeled)|data processing)\b",
        r"\b(?:dataset|data source|harmoniz|preprocess|normaliz|standardiz)\w*\b",
    ),
    "system_methods": (
        r"\b(?:we (?:implemented|developed|built|designed)|consists of|architecture|algorithm)\b",
        r"\b(?:system|model|encoding|classifier|nearest neighbou?r|interface|panel)\b",
    ),
    "evaluation_methods": (
        r"\b(?:we (?:evaluated|conducted|report)|evaluation process|case stud|user stud)\w*\b",
        r"\b(?:participants?|domain experts?|procedure|screen sharing|note-taking|interview)\b",
    ),
    "evaluation": (
        r"\b(?:results? show|we (?:found|observed|report)|case stud|evaluation)\w*\b",
        r"\b(?:domain experts?|participants?|expert feedback|investigation)\b",
    ),
    "methods": (
        r"\b(?:we use|we employ|we train|consists of|comprises|architecture|algorithm)\b",
        r"\b(?:dataset|training data|optimizer|layer|evaluation setup)\b",
    ),
    "findings": (
        r"\b(?:we find|we show|results show|outperform|achiev|improv|score)\w*\b",
        r"\b\d+(?:\.\d+)?\s*(?:%|bleu|f1|accuracy|points?)\b",
    ),
    "discussion": (
        r"\b(?:we (?:conclude|found|show|demonstrate|recommend)|our results|case studies)\b",
        r"\b(?:in conclusion|overall|discussion|implications?)\b",
    ),
    "limitations": (
        r"\b(?:limitation|constraint|threat|cannot|could not|did not|lack of|limited)\b",
        r"\b(?:overplotting|under-represented|not available|unavailable|left-censor)\w*\b",
    ),
    "future_work": (
        r"\b(?:future (?:work|investigation|research|deployment|extension|data collection))\b",
        r"\b(?:further deployment|we (?:plan|will|intend|hope) to|marked .{0,80} future)\b",
    ),
    "gaps": (
        r"\b(?:limitation|future work|we plan|we hope|remains|further work)\b",
        r"\b(?:cannot|does not|not yet|open question)\b",
    ),
}


@dataclass(frozen=True)
class EvidenceSelection:
    chunks: list[tuple[str, str, str]]
    pools: dict[str, list[str]]


def evidence_catalog(selection: EvidenceSelection) -> dict[str, tuple[str, str, str]]:
    """Create compact IDs for deterministic, verbatim excerpts from selected chunks."""
    catalog: dict[str, tuple[str, str, str]] = {}
    pools_by_chunk: dict[str, set[str]] = {}
    for pool, chunk_ids in selection.pools.items():
        for chunk_id in chunk_ids:
            pools_by_chunk.setdefault(chunk_id, set()).add(pool)
    for heading, chunk_id, text in selection.chunks:
        passages = []
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
            passages.extend(re.split(r";\s+(?=(?:and\s+)?(?:\d+|[a-z])[.)])", sentence))
        passages = [part.strip() for part in passages if len(part.strip()) >= 20]
        if not passages:
            passages = [text.strip()]
        chunk_pools = pools_by_chunk.get(chunk_id, set())
        forced_patterns = [
            signal
            for pool in (
                "contributions",
                "evaluation_methods",
                "evaluation",
                "discussion",
                "limitations",
                "future_work",
            )
            if pool in chunk_pools
            for signal in TEXT_SIGNALS[pool]
        ]
        forced = {
            index
            for index, passage in enumerate(passages)
            if any(re.search(pattern, passage.casefold()) for pattern in forced_patterns)
            or (
                "contributions" in chunk_pools
                and re.match(r"^(?:and\s+)?(?:\d+|[a-z])[.)]", passage.casefold())
            )
        }
        ranked = sorted(
            enumerate(passages),
            key=lambda pair: (
                -(
                    2 * bool(re.search(r"\d", pair[1]))
                    + bool(
                        re.search(r"\b(?:we|model|method|result|attention|train)\b", pair[1], re.I)
                    )
                ),
                pair[0],
            ),
        )
        picked = sorted({0, *forced, *(index for index, _ in ranked[:2])})[:8]
        for index in picked:
            quote = passages[index][:280].strip().rstrip(" ,;:")
            if len(quote) < 20:
                continue
            evidence_id = f"E{len(catalog) + 1:02d}"
            catalog[evidence_id] = (chunk_id, quote, heading)
    return catalog


def _score(pool: str, heading: str, text: str, is_abstract: bool) -> int:
    heading_lower = heading.casefold()
    text_lower = text.casefold()
    score = 8 if pool in {"overview", "contributions"} and is_abstract else 0
    score += 5 * sum(signal in heading_lower for signal in HEADING_SIGNALS[pool])
    score += 3 * sum(bool(re.search(signal, text_lower)) for signal in TEXT_SIGNALS[pool])
    if pool == "contributions" and re.search(
        r"\b(?:main|key|primary) contributions?\b", text_lower
    ):
        score += 20
    if len(text.strip()) < 80:
        score -= 4
    if "equal contribution" in text_lower or "work performed while" in text_lower:
        score -= 10
    return score


def select_evidence(
    document: ParsedPaper, max_chunks: int = 25, max_chars: int = 13000
) -> EvidenceSelection:
    records: list[tuple[str, str, str, bool]] = [
        ("Abstract", chunk.id, chunk.text, True)
        for chunk in document.abstract_chunks
        if chunk.text.strip()
    ]
    records.extend(
        (section.heading or "Untitled section", chunk.id, chunk.text, False)
        for section in document.sections
        for chunk in section.chunks
        if chunk.text.strip()
    )
    if not records:
        return EvidenceSelection(chunks=[], pools={pool: [] for pool in FIELD_POOLS})

    pools: dict[str, list[str]] = {pool: [] for pool in FIELD_POOLS}
    for pool in FIELD_POOLS:
        ranked = sorted(
            records,
            key=lambda record: (
                -_score(pool, record[0], record[2], record[3]),
                records.index(record),
            ),
        )
        headings: set[str] = set()
        for diverse_only in (True, False):
            for heading, chunk_id, text, is_abstract in ranked:
                if len(pools[pool]) >= POOL_BUDGETS[pool]:
                    break
                if _score(pool, heading, text, is_abstract) <= 0:
                    continue
                if chunk_id in pools[pool] or (diverse_only and heading in headings):
                    continue
                pools[pool].append(chunk_id)
                headings.add(heading)

    by_id = {chunk_id: (heading, text) for heading, chunk_id, text, _ in records}
    chosen: set[str] = set()
    selected_chars = 0

    def add(chunk_id: str) -> None:
        nonlocal selected_chars
        if chunk_id in chosen or chunk_id not in by_id or len(chosen) >= max_chunks:
            return
        text = by_id[chunk_id][1]
        if selected_chars + len(text) > max_chars:
            return
        chosen.add(chunk_id)
        selected_chars += len(text)

    for pool, count in GUARANTEED_POOL_COUNTS.items():
        for chunk_id in pools[pool][:count]:
            add(chunk_id)
    for pool in FIELD_POOLS:
        for chunk_id in pools[pool]:
            add(chunk_id)

    for _heading, chunk_id, text, _ in records:
        if len(chosen) >= min(15, len(records)):
            break
        if (
            chunk_id not in chosen
            and len(chosen) < max_chunks
            and selected_chars + len(text) <= max_chars
        ):
            chosen.add(chunk_id)
            selected_chars += len(text)
    return EvidenceSelection(
        chunks=[
            (heading, chunk_id, text)
            for heading, chunk_id, text, _ in records
            if chunk_id in chosen
        ],
        pools=pools,
    )
