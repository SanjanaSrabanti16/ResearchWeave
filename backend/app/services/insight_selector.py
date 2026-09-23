"""Cheap, deterministic evidence selection before local insight generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.document import ParsedPaper

FIELD_POOLS = (
    "problem",
    "problem_gaps",
    "explicit_contributions",
    "contributions",
    "overview",
    "design_methods",
    "data_methods",
    "system_methods",
    "evaluation_methods",
    "evaluation",
    "evaluation_subsections",
    "findings",
    "discussion",
    "discussion_limitations",
    "discussion_deployment",
    "limitations",
    "future_work",
    "audience",
    "methods",
    "gaps",
)
POOL_BUDGETS = {
    "problem": 8,
    "problem_gaps": 8,
    "explicit_contributions": 12,
    "contributions": 4,
    "overview": 6,
    "design_methods": 4,
    "data_methods": 4,
    "system_methods": 6,
    "evaluation_methods": 4,
    "evaluation": 8,
    "evaluation_subsections": 16,
    "findings": 10,
    "discussion": 5,
    "discussion_limitations": 8,
    "discussion_deployment": 8,
    "limitations": 6,
    "future_work": 5,
    "audience": 6,
    "methods": 14,
    "gaps": 6,
}
HEADING_SIGNALS = {
    "problem": (
        "abstract",
        "introduction",
        "background",
        "motivation",
        "problem",
        "challenge",
        "conclusion",
    ),
    "problem_gaps": (
        "abstract",
        "introduction",
        "background",
        "motivation",
        "problem",
        "challenge",
        "conclusion",
    ),
    "explicit_contributions": ("contribution", "introduction", "conclusion"),
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
    "evaluation_subsections": (
        "evaluation",
        "result",
        "experiment",
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
    "discussion_limitations": ("discussion", "conclusion"),
    "discussion_deployment": ("discussion", "conclusion"),
    "limitations": ("limitation", "constraint", "threat"),
    "future_work": ("future work", "future direction", "further work"),
    "audience": ("evaluation", "expert feedback", "deployment", "users", "participants"),
    "gaps": ("limitation", "discussion", "conclusion", "future", "open question"),
}
TEXT_SIGNALS = {
    "problem": (
        r"\b(?:not known|unknown|unclear|poorly understood|not well understood|understudied|"
        r"remains? (?:unclear|unknown)|little is known|unresolved)\b",
        r"\b(?:has not been (?:studied|analy[sz]ed)|never before|challenge|difficult|difficulty|"
        r"impractical|lack of|insufficient|existing work (?:does|did) not|need to|we aim to|"
        r"we seek to|research question|open question)\b",
    ),
    "problem_gaps": (
        r"\b(?:not known|unknown|unclear|poorly understood|not well understood|understudied|"
        r"remains? (?:unclear|unknown)|little is known|unresolved)\b",
        r"\b(?:has not been (?:studied|analy[sz]ed)|never before|challenge|difficult|difficulty|"
        r"impractical|lack of|existing work (?:does|did) not)\b",
    ),
    "explicit_contributions": (
        r"\b(?:main|key|primary) contributions?\b",
        r"\b(?:our contributions?|we contribute|contributions? of this work)\b",
    ),
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
    "evaluation_subsections": (
        r"\b(?:found|observed|revealed|showed|confirmed|determined|noticed|surprised|apparent|"
        r"became apparent|appeared|indicated|suggests?|demonstrated)\w*\b",
        r"\b(?:case stud|evaluation|results?|expert feedback|investigation)\w*\b",
    ),
    "methods": (
        r"\b(?:we use|we employ|we train|consists of|comprises|architecture|algorithm)\b",
        r"\b(?:dataset|training data|optimizer|layer|evaluation setup)\b",
    ),
    "findings": (
        r"\b(?:we find|we show|results show|found|observed|revealed|showed|confirmed|"
        r"determined|noticed|surprised|apparent|became apparent|appeared|indicated|suggests?|"
        r"demonstrated|outperform|achiev|improv|score)\w*\b",
        r"\b\d+(?:\.\d+)?\s*(?:%|bleu|f1|accuracy|points?)\b",
    ),
    "discussion": (
        r"\b(?:we (?:conclude|found|show|demonstrate|recommend)|our results|case studies)\b",
        r"\b(?:in conclusion|overall|discussion|implications?)\b",
    ),
    "discussion_limitations": (
        r"\b(?:limited|limitation|constraint|issue|under-?represented|overplotting|missing|"
        r"unavailable|bias|censoring|left-censor|scalability)\w*\b",
        r"\b(?:could not|cannot|did not|lack of|may require|require(?:s|d)? .{0,40}"
        r"modifications?)\b",
    ),
    "discussion_deployment": (
        r"\b(?:generali[sz]ability|deployment|extend|extension|broader|additional datasets?|"
        r"other cohorts?|other domains?|multi-(?:dataset|cohort))\w*\b",
        r"\b(?:future|further (?:investigation|deployment)|future collection|"
        r"could be extended|can extend)\b",
    ),
    "limitations": (
        r"\b(?:limited|limitation|constraint|issue|missing|unavailable|could not|cannot|however|"
        r"threat|did not|lack of|may require|require(?:s|d)? .{0,40} modifications?)\b",
        r"\b(?:overplotting|under-?represented|scalability|bias|censoring|left-censor)\w*\b",
    ),
    "future_work": (
        r"\b(?:future (?:work|investigation|research|deployment|extension|data collection|"
        r"direction))\b",
        r"\b(?:further (?:work|investigation|deployment)|additional data|broader deployment|"
        r"we (?:plan|intend|hope) to|we will|marked .{0,80} future|could be extended|can extend)\b",
    ),
    "audience": (
        r"\b(?:domain experts?|clinicians?|researchers?|analysts?|practitioners?|users?)\b",
        r"\b(?:collaborators?|participants?|stakeholders?|intended for|deployed)\b",
    ),
    "gaps": (
        r"\b(?:limitation|future work|we plan|we hope|remains|further work)\b",
        r"\b(?:cannot|does not|not yet|open question)\b",
    ),
}

_FUTURE_ACTION_SIGNALS = (
    r"\b(?:future (?:work|investigation|research|deployment|extension|data collection|"
    r"direction))\b",
    r"\b(?:further (?:work|investigation|deployment)|broader deployment)\b",
    r"\bfuture .{0,40}(?:additional data|data collection)\b",
    r"\bwe\s+(?:"
    r"will\s+(?:conduct|evaluate|explore|investigate|study|collect|deploy|extend|develop|"
    r"implement|integrate|assess|examine|test|build|create|adapt|apply|compare|analy[sz]e)|"
    r"plan\s+(?:to|on)\s+\w+|intend\s+to\s+\w+|aim\s+to\s+\w+|"
    r"envision(?:\s+\w+){0,3}|hope\s+to\s+\w+|seek\s+to\s+\w+)\b",
    r"\bfuture\s+(?:work|research|directions?|extensions?)\s+"
    r"(?:will|could|may|might|includes?|will include)\b",
    r"\b(?:marked .{0,80} future|could be extended|can extend)\b",
)
_NON_AUTHOR_FUTURE_SIGNALS = (
    r"\b(?:participants?|interviewees?|experts?|users?|reviewers?)\s+"
    r"(?:suggested|recommended|requested|proposed|mentioned|expressed|wanted|asked)\b",
    r"\b(?:users?|readers?|practitioners?)\s+(?:should|could|may|might)\b",
    r"\bwe\s+recommend\s+(?:that\s+)?(?:users?|readers?|practitioners?)\b",
    r"\b(?:might|may)\s+fare\s+better\b",
    r"\b(?:prior|previous)\s+(?:work|stud(?:y|ies))\b.{0,100}\b(?:could|will|may|might)\b",
    r"\b[A-Z][A-Za-z-]+\s+et\s+al\.\b.{0,100}\b(?:could|will|may|might)\b",
)
_METHOD_DESCRIPTION_SIGNALS = (
    r"\bwe (?:use|used|apply|applied|decided to use|calculate|calculated|compute|computed|"
    r"implement|implemented|develop|developed|design|designed)\b",
    r"\b(?:algorithm|architecture|distance metric|implementation|procedure|consists of|"
    r"comprises)\b",
)
_RESULT_OBSERVATION_SIGNALS = (
    r"\b(?:we|the (?:analysis|comparison|evaluation|experiment|results?))\s+"
    r"(?:found|observed|revealed|showed|confirmed|determined|noticed|indicated|demonstrated|"
    r"identified|categorized|characterized)\b",
    r"\bresults? (?:show|showed|indicate|indicated|demonstrate|demonstrated)\b",
    r"\b(?:was|were|is|are) (?:higher|lower|more|less|earlier|later|longer|shorter|better|"
    r"worse|larger|smaller)\b",
    r"\b(?:outperform|achiev|improv|score)\w*\b",
    r"\b(?:found|observed|revealed|showed|confirmed|determined|noticed|indicated|"
    r"demonstrated|identified|categorized|characterized)\b",
    r"\b(?:experts?|participants?|users?|respondents?)\s+"
    r"(?:found|reported|noted|observed|highlighted|appreciated|preferred|valued|agreed)\b",
    r"\b(?:themes?|patterns?)\s+(?:emerged|were identified|were observed)\b",
)
_FINDING_HEADING = re.compile(
    r"\b(?:evaluation|results?|findings?|experiments?|case stud(?:y|ies)|analysis|"
    r"expert feedback|user study|survey|discussion)\b",
    re.I,
)
_NON_FINDING_HEADING = re.compile(
    r"\b(?:introduction|background|related work|design goals?|method(?:s|ology)?|"
    r"implementation|architecture|system design)\b",
    re.I,
)
_CAPABILITY_ONLY = re.compile(
    r"\b(?:system|framework|tool|method|approach|grammar|platform|model)\s+"
    r"(?:can|supports?|enables?|allows?|provides?|offers?|is designed to)\b",
    re.I,
)


def supports_future_work(text: str) -> bool:
    """Require an explicit forward action, investigation, extension, or deployment."""
    if any(re.search(signal, text, re.I) for signal in _NON_AUTHOR_FUTURE_SIGNALS):
        return False
    return any(re.search(signal, text, re.I) for signal in _FUTURE_ACTION_SIGNALS)


def supports_finding(text: str, headings: tuple[str, ...] = ()) -> bool:
    """Require current-paper result/observation evidence, including qualitative findings."""
    if any(re.search(signal, text, re.I) for signal in _RESULT_OBSERVATION_SIGNALS):
        return True
    if any(re.search(signal, text, re.I) for signal in _METHOD_DESCRIPTION_SIGNALS):
        return False
    if _CAPABILITY_ONLY.search(text):
        return False
    if headings and all(_NON_FINDING_HEADING.search(heading) for heading in headings):
        return False
    return bool(headings and any(_FINDING_HEADING.search(heading) for heading in headings))


@dataclass(frozen=True)
class EvidenceSelection:
    chunks: list[tuple[str, str, str]]
    pools: dict[str, list[str]]
    priority_passages: dict[str, list[str]] = field(default_factory=dict)


_LIST_MARKER = re.compile(r"(?<!\w)(?:\(\d{1,3}\)|\d{1,3}[.)]|[a-z][.)]|[•▪◦])\s+", re.I)


def _contribution_list_items(text: str) -> list[str]:
    """Return every item from an author-declared numbered or bulleted contribution list."""
    if not any(
        re.search(pattern, text, re.I) for pattern in TEXT_SIGNALS["explicit_contributions"]
    ):
        return []
    markers = list(_LIST_MARKER.finditer(text))
    if len(markers) < 2:
        bullet_lines = [
            line.strip()
            for line in text.splitlines()
            if re.match(r"^[-*]\s+\S", line.strip()) is not None
        ]
        return bullet_lines if len(bullet_lines) >= 2 else []
    items = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        item = text[marker.start() : end].strip().strip(";").strip()
        if len(item) >= 20:
            items.append(item)
    return items


def _passages(text: str) -> list[str]:
    passages = []
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
        passages.extend(re.split(r";\s+(?=(?:and\s+)?(?:\d+|[a-z])[.)])", sentence))
    return [part.strip() for part in passages if len(part.strip()) >= 20] or [text.strip()]


_BOUNDARY_HEADING = re.compile(
    r"\b(?:limitations?|constraints?|discussion|conclusions?|future\s+"
    r"(?:work|directions?|research|extensions?))\b",
    re.I,
)
_EXPLICIT_LIMITATION_HEADING = re.compile(r"\b(?:limitations?|constraints?|threats?)\b", re.I)
_EXPLICIT_FUTURE_HEADING = re.compile(
    r"\bfuture\s+(?:work|directions?|research|extensions?)\b", re.I
)
_BARE_BOUNDARY_PASSAGE = re.compile(
    r"^\s*(?:our\s+)?(?:main\s+|key\s+)?(?:limitations?|constraints?|future\s+"
    r"(?:work|directions?|research)|limitations?\s+and\s+future\s+work)"
    r"(?:\s+(?:are|include|are\s+as\s+follows|include\s+the\s+following))?\s*[.:]?\s*$",
    re.I,
)
_CONCRETE_LIMITATION = re.compile(
    r"\b(?:cannot|could\s+not|did\s+not|does\s+not|has\s+not|have\s+not|not\s+yet|"
    r"lack(?:s|ed|ing)?|limited|limitation|constraint|restrict(?:s|ed|ing)?|"
    r"unavailable|incomplete|slow|scalability|bias(?:ed)?|no\s+guaranteed?|"
    r"memory\s+(?:cost|requirement|footprint)|overhead|trade-?off)\b",
    re.I,
)


def _substantive_boundary_passage(passage: str) -> bool:
    return (
        len(re.findall(r"\b\w+\b", passage)) >= 6
        and _BARE_BOUNDARY_PASSAGE.fullmatch(passage) is None
    )


def reserve_boundary_passages(
    selection: EvidenceSelection,
    *,
    limitation_limit: int = 3,
    future_limit: int = 3,
) -> EvidenceSelection:
    """Reserve a small set of concrete end-section boundary sentences before catalog ranking."""
    limitations: list[tuple[str, str]] = []
    future: list[tuple[str, str]] = []
    seen: set[str] = set()
    for heading, chunk_id, text in selection.chunks:
        if _BOUNDARY_HEADING.search(heading) is None:
            continue
        explicit_limitation = _EXPLICIT_LIMITATION_HEADING.search(heading) is not None
        explicit_future = _EXPLICIT_FUTURE_HEADING.search(heading) is not None
        for passage in _passages(text):
            normalized = " ".join(re.sub(r"\W+", " ", passage.casefold()).split())
            if not normalized or normalized in seen or not _substantive_boundary_passage(passage):
                continue
            if (explicit_limitation or _CONCRETE_LIMITATION.search(passage)) and (
                _CONCRETE_LIMITATION.search(passage) is not None
            ):
                limitations.append((chunk_id, passage))
                seen.add(normalized)
                continue
            if (explicit_future or re.search(r"\b(?:discussion|conclusion)\b", heading, re.I)) and (
                supports_future_work(passage)
            ):
                future.append((chunk_id, passage))
                seen.add(normalized)

    priority: dict[str, list[str]] = {}
    for chunk_id, passage in [
        *limitations[:limitation_limit],
        *future[:future_limit],
    ]:
        priority.setdefault(chunk_id, []).append(passage)
    return EvidenceSelection(
        chunks=selection.chunks,
        pools=selection.pools,
        priority_passages=priority,
    )


def _truncate_excerpt(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text.rstrip(" ,;:")
    prefix = text[: limit + 1]
    boundary = max(prefix.rfind(mark) for mark in (". ", "! ", "? ", "; ", ": "))
    if boundary < int(limit * 0.6):
        boundary = prefix.rfind(" ", 0, limit + 1)
    if boundary <= 0:
        boundary = limit
    return text[:boundary].strip().rstrip(" ,;:")


def evidence_catalog(selection: EvidenceSelection) -> dict[str, tuple[str, str, str]]:
    """Create compact IDs after required category and section coverage has been reserved."""
    catalog: dict[str, tuple[str, str, str]] = {}
    seen_quotes: set[str] = set()
    pools_by_chunk: dict[str, set[str]] = {}
    for pool, chunk_ids in selection.pools.items():
        for chunk_id in chunk_ids:
            pools_by_chunk.setdefault(chunk_id, set()).add(pool)

    def add(chunk_id: str, quote: str, heading: str, limit: int = 280) -> None:
        quote = _truncate_excerpt(quote, limit)
        normalized = " ".join(re.sub(r"\W+", " ", quote.casefold()).split())
        if len(quote) < 20 or not normalized or normalized in seen_quotes:
            return
        seen_quotes.add(normalized)
        evidence_id = f"E{len(catalog) + 1:02d}"
        catalog[evidence_id] = (chunk_id, quote, heading)

    for heading, chunk_id, text in selection.chunks:
        passages = _passages(text)
        priority_passages = selection.priority_passages.get(chunk_id, [])
        for passage in priority_passages:
            add(chunk_id, passage, heading)
        chunk_pools = pools_by_chunk.get(chunk_id, set())
        forced_patterns = [
            signal
            for pool in (
                "problem",
                "problem_gaps",
                "contributions",
                "evaluation_methods",
                "evaluation",
                "evaluation_subsections",
                "findings",
                "discussion",
                "discussion_limitations",
                "discussion_deployment",
                "limitations",
                "future_work",
                "audience",
            )
            if pool in chunk_pools
            for signal in TEXT_SIGNALS[pool]
        ]
        forced = {
            index
            for index, passage in enumerate(passages)
            if any(re.search(pattern, passage.casefold()) for pattern in forced_patterns)
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
        required = {0, *forced}
        optional = [index for index, _ in ranked if index not in required]
        picked = sorted(
            required | set(optional[: max(0, 8 - len(required) - len(priority_passages))])
        )
        for index in picked:
            add(chunk_id, passages[index], heading)
        if "explicit_contributions" in chunk_pools:
            for item in _contribution_list_items(text):
                add(chunk_id, item, heading, limit=500)
    return catalog


def _score(pool: str, heading: str, text: str, is_abstract: bool) -> int:
    heading_lower = heading.casefold()
    text_lower = text.casefold()
    score = (
        8 if pool in {"problem", "problem_gaps", "overview", "contributions"} and is_abstract else 0
    )
    score += 5 * sum(signal in heading_lower for signal in HEADING_SIGNALS[pool])
    text_signal_weight = 6 if pool in {"problem", "problem_gaps"} else 3
    score += text_signal_weight * sum(
        bool(re.search(signal, text_lower)) for signal in TEXT_SIGNALS[pool]
    )
    if pool in {"explicit_contributions", "contributions"} and re.search(
        r"\b(?:main|key|primary) contributions?\b", text_lower
    ):
        score += 20
    if len(text.strip()) < 80:
        score -= 4
    if "equal contribution" in text_lower or "work performed while" in text_lower:
        score -= 10
    return score


def select_evidence(document: ParsedPaper) -> EvidenceSelection:
    """Index all paper chunks into semantic pools without applying a shared prompt budget."""
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
        if pool in {
            "problem_gaps",
            "explicit_contributions",
            "evaluation_subsections",
            "discussion_limitations",
            "discussion_deployment",
        }:
            continue
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

    pools["explicit_contributions"] = [
        chunk_id
        for _heading, chunk_id, text, _is_abstract in records
        if _contribution_list_items(text)
    ][: POOL_BUDGETS["explicit_contributions"]]

    gap_records = [
        record
        for record in records
        if any(re.search(signal, record[2], re.I) for signal in TEXT_SIGNALS["problem_gaps"])
    ]
    pools["problem_gaps"] = [
        record[1]
        for record in sorted(
            gap_records,
            key=lambda record: (
                -_score("problem_gaps", record[0], record[2], record[3]),
                records.index(record),
            ),
        )[: POOL_BUDGETS["problem_gaps"]]
    ]

    evaluation_section_ids: list[str] = []
    active_evaluation_level: int | None = None
    evaluation_heading = re.compile(
        r"\b(?:evaluation|results?|experiments?|case stud(?:y|ies)|investigation|"
        r"expert feedback)\b",
        re.I,
    )
    for section in document.sections:
        heading = section.heading or ""
        matches = evaluation_heading.search(heading) is not None
        if matches:
            if active_evaluation_level is None or section.level <= active_evaluation_level:
                active_evaluation_level = section.level
        elif active_evaluation_level is not None and section.level <= active_evaluation_level:
            active_evaluation_level = None
        if not matches and active_evaluation_level is None:
            continue
        candidates = [chunk for chunk in section.chunks if chunk.text.strip()]
        finding_candidates = [
            chunk
            for chunk in candidates
            if any(
                re.search(signal, chunk.text, re.I)
                for signal in TEXT_SIGNALS["evaluation_subsections"]
            )
        ]
        if not finding_candidates and candidates:
            finding_candidates = [
                max(
                    candidates,
                    key=lambda chunk: (
                        _score("findings", heading, chunk.text, False),
                        -candidates.index(chunk),
                    ),
                )
            ]
        phase_limit = 8
        if len(finding_candidates) > phase_limit:
            protected = {0, len(finding_candidates) // 2, len(finding_candidates) - 1}
            ranked_indexes = sorted(
                range(len(finding_candidates)),
                key=lambda index: (
                    -_score("findings", heading, finding_candidates[index].text, False),
                    index,
                ),
            )
            for index in ranked_indexes:
                if len(protected) >= phase_limit:
                    break
                protected.add(index)
            finding_candidates = [
                chunk for index, chunk in enumerate(finding_candidates) if index in protected
            ]
        evaluation_section_ids.extend(chunk.id for chunk in finding_candidates)
    pools["evaluation_subsections"] = list(dict.fromkeys(evaluation_section_ids))

    discussion_heading = re.compile(r"\b(?:discussion|conclusion)\b", re.I)
    pools["discussion_limitations"] = [
        chunk_id
        for heading, chunk_id, text, _is_abstract in records
        if discussion_heading.search(heading)
        and any(re.search(signal, text, re.I) for signal in TEXT_SIGNALS["discussion_limitations"])
    ][: POOL_BUDGETS["discussion_limitations"]]
    pools["discussion_deployment"] = [
        chunk_id
        for heading, chunk_id, text, _is_abstract in records
        if discussion_heading.search(heading)
        and any(re.search(signal, text, re.I) for signal in TEXT_SIGNALS["discussion_deployment"])
    ][: POOL_BUDGETS["discussion_deployment"]]

    return EvidenceSelection(
        chunks=[(heading, chunk_id, text) for heading, chunk_id, text, _ in records],
        pools=pools,
    )


def focus_evidence(
    selection: EvidenceSelection,
    reservations: tuple[tuple[str, int | None], ...],
    fill_pools: tuple[str, ...],
    *,
    max_chunks: int,
    max_chars: int,
) -> EvidenceSelection:
    """Build a compact stage pack, reserving required semantic coverage before filling."""
    by_id = {chunk_id: (heading, text) for heading, chunk_id, text in selection.chunks}
    chosen: list[str] = []
    chosen_set: set[str] = set()
    selected_chars = 0

    def reserve(chunk_id: str) -> None:
        nonlocal selected_chars
        if chunk_id in chosen_set or chunk_id not in by_id:
            return
        chosen.append(chunk_id)
        chosen_set.add(chunk_id)
        selected_chars += len(by_id[chunk_id][1])

    for pool, quota in reservations:
        candidates = selection.pools[pool]
        for chunk_id in candidates if quota is None else candidates[:quota]:
            reserve(chunk_id)

    for pool in fill_pools:
        for chunk_id in selection.pools[pool]:
            if chunk_id in chosen_set or chunk_id not in by_id:
                continue
            text = by_id[chunk_id][1]
            if len(chosen) >= max_chunks or selected_chars + len(text) > max_chars:
                continue
            reserve(chunk_id)

    return EvidenceSelection(
        chunks=[chunk for chunk in selection.chunks if chunk[1] in chosen_set],
        pools=selection.pools,
        priority_passages={
            chunk_id: passages
            for chunk_id, passages in selection.priority_passages.items()
            if chunk_id in chosen_set
        },
    )
