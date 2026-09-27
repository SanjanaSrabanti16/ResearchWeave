from __future__ import annotations

import re
from dataclasses import dataclass

from app.llm.base import ContextStrategy, LLMProviderCapabilities
from app.models.document import ParsedPaper
from app.models.insights import EVIDENCE_QUOTE_MAX_LENGTH
from app.services.insight_selector import explicit_contribution_chunk_ids


@dataclass(frozen=True)
class EvidenceExcerpt:
    evidence_id: str
    chunk_id: str
    section_id: str
    heading: str
    quote: str
    rhetorical_role: str | None = None


@dataclass(frozen=True)
class PaperContextPacket:
    packet_id: str
    purpose: str
    excerpts: tuple[EvidenceExcerpt, ...]
    strategy: ContextStrategy = "full_document"

    @property
    def character_count(self) -> int:
        return sum(len(excerpt.quote) for excerpt in self.excerpts)


_EXCLUDED_HEADING = re.compile(r"\b(?:references|bibliography|acknowledg(?:e)?ments?)\b", re.I)
_METADATA_MARKER = re.compile(
    r"\b(?:corresponding author|e-?mail|department of|research institute|state key laboratory)\b|"
    r"[\w.+-]+@[\w.-]+",
    re.I,
)


def _citeable_spans(
    text: str,
    limit: int = EVIDENCE_QUOTE_MAX_LENGTH,
) -> tuple[str, ...]:
    """Split source text into deterministic, bounded, exact-substring evidence spans."""
    spans: list[str] = []
    cursor = 0
    source_end = len(text)
    while cursor < source_end:
        while cursor < source_end and text[cursor].isspace():
            cursor += 1
        if cursor >= source_end:
            break
        remaining = source_end - cursor
        if remaining <= limit:
            span = text[cursor:source_end].rstrip()
            if span:
                spans.append(span)
            break

        window_end = cursor + limit
        minimum_boundary = cursor + limit // 2
        boundary_candidates: list[int] = []
        for match in re.finditer(r"\n\s*\n|(?<=[.!?])\s+", text[cursor:window_end]):
            absolute_end = cursor + match.start()
            if absolute_end >= minimum_boundary:
                boundary_candidates.append(absolute_end)
        if boundary_candidates:
            span_end = boundary_candidates[-1]
        else:
            whitespace = max(
                text.rfind(" ", minimum_boundary, window_end + 1),
                text.rfind("\n", minimum_boundary, window_end + 1),
                text.rfind("\t", minimum_boundary, window_end + 1),
            )
            span_end = whitespace if whitespace > cursor else window_end
        span = text[cursor:span_end].rstrip()
        if not span:
            span_end = min(cursor + limit, source_end)
            span = text[cursor:span_end]
        spans.append(span)
        cursor = span_end
    return tuple(spans)


def _role_for_chunk(chunk_id: str, header_ids: set[str], item_ids: set[str]) -> str | None:
    if chunk_id in item_ids:
        return "explicit_current_paper_contribution_item"
    if chunk_id in header_ids:
        return "explicit_current_paper_contribution_header"
    return None


def _paper_records(document: ParsedPaper) -> list[tuple[str, str, str, str, str | None]]:
    """Return citeable paper chunks once, in parsed reading order, without references."""
    contribution_headers, contribution_items = explicit_contribution_chunk_ids(document)
    header_ids = set(contribution_headers)
    item_ids = set(contribution_items)
    records: list[tuple[str, str, str, str, str | None]] = []
    for chunk in document.abstract_chunks:
        if chunk.text.strip():
            records.append((chunk.id, chunk.section_id, "Abstract", chunk.text, None))
    for section in document.sections:
        heading = section.heading or "Untitled section"
        if _EXCLUDED_HEADING.search(heading):
            continue
        for chunk in section.chunks:
            text = chunk.text.strip()
            if not text or (_METADATA_MARKER.search(text) and len(text) < 500):
                continue
            records.append(
                (
                    chunk.id,
                    section.id,
                    heading,
                    chunk.text,
                    _role_for_chunk(chunk.id, header_ids, item_ids),
                )
            )
    return records


def _full_document_packet(document: ParsedPaper) -> PaperContextPacket:
    """Expose the complete parsed paper exactly once in reading order."""
    excerpts: list[EvidenceExcerpt] = []
    for chunk_id, section_id, heading, text, rhetorical_role in _paper_records(document):
        for quote in _citeable_spans(text):
            excerpts.append(
                EvidenceExcerpt(
                    evidence_id=f"P1-E{len(excerpts) + 1:03d}",
                    chunk_id=chunk_id,
                    section_id=section_id,
                    heading=heading,
                    quote=quote,
                    rhetorical_role=rhetorical_role,
                )
            )
    return PaperContextPacket(
        packet_id="P1",
        purpose="full parsed paper in reading order",
        excerpts=tuple(excerpts),
        strategy="full_document",
    )


_HIERARCHICAL_PHASES = (
    "introduction and research problem",
    "methods and approach",
    "results and evaluation",
    "discussion and implications",
    "limitations, open issues, and future work",
    "conclusion",
)


def _hierarchical_phase(heading: str) -> str:
    normalized = heading.casefold()
    if re.search(r"\bconclusions?\b", normalized):
        return _HIERARCHICAL_PHASES[5]
    if re.search(
        r"\b(?:limitations?|constraints?|future|open (?:issues?|questions?|research)|"
        r"outlooks?|challenges?|threats?)\b",
        normalized,
    ):
        return _HIERARCHICAL_PHASES[4]
    if re.search(r"\b(?:discussion|implications?|lessons learned)\b", normalized):
        return _HIERARCHICAL_PHASES[3]
    if re.search(
        r"\b(?:results?|evaluation|experiments?|case stud(?:y|ies)|findings?|analysis|"
        r"user stud(?:y|ies)|expert feedback)\b",
        normalized,
    ):
        return _HIERARCHICAL_PHASES[2]
    if re.search(
        r"\b(?:abstract|introduction|background|motivation|overview|problem|research question)\b",
        normalized,
    ):
        return _HIERARCHICAL_PHASES[0]
    return _HIERARCHICAL_PHASES[1]


def _hierarchical_packets(document: ParsedPaper, batch_chars: int) -> list[PaperContextPacket]:
    """Read every non-reference section through ordered, purpose-specific note packets."""
    grouped: dict[str, list[tuple[str, str, str, str, str | None]]] = {
        phase: [] for phase in _HIERARCHICAL_PHASES
    }
    for record in _paper_records(document):
        grouped[_hierarchical_phase(record[2])].append(record)

    packets: list[PaperContextPacket] = []
    for phase in _HIERARCHICAL_PHASES:
        pending: list[EvidenceExcerpt] = []
        pending_chars = 0
        for chunk_id, section_id, heading, text, rhetorical_role in grouped[phase]:
            for quote in _citeable_spans(text):
                cost = len(quote) + len(heading) + 32
                if pending and pending_chars + cost > batch_chars:
                    packet_id = f"P{len(packets) + 1}"
                    packets.append(
                        PaperContextPacket(
                            packet_id=packet_id,
                            purpose=phase,
                            excerpts=tuple(
                                EvidenceExcerpt(
                                    evidence_id=f"{packet_id}-E{index:03d}",
                                    chunk_id=item.chunk_id,
                                    section_id=item.section_id,
                                    heading=item.heading,
                                    quote=item.quote,
                                    rhetorical_role=item.rhetorical_role,
                                )
                                for index, item in enumerate(pending, start=1)
                            ),
                            strategy="hierarchical_sections",
                        )
                    )
                    pending = []
                    pending_chars = 0
                pending.append(
                    EvidenceExcerpt(
                        evidence_id="pending",
                        chunk_id=chunk_id,
                        section_id=section_id,
                        heading=heading,
                        quote=quote,
                        rhetorical_role=rhetorical_role,
                    )
                )
                pending_chars += cost
        if pending:
            packet_id = f"P{len(packets) + 1}"
            packets.append(
                PaperContextPacket(
                    packet_id=packet_id,
                    purpose=phase,
                    excerpts=tuple(
                        EvidenceExcerpt(
                            evidence_id=f"{packet_id}-E{index:03d}",
                            chunk_id=item.chunk_id,
                            section_id=item.section_id,
                            heading=item.heading,
                            quote=item.quote,
                            rhetorical_role=item.rhetorical_role,
                        )
                        for index, item in enumerate(pending, start=1)
                    ),
                    strategy="hierarchical_sections",
                )
            )
    return packets


def build_paper_context_packets(
    document: ParsedPaper,
    capabilities: LLMProviderCapabilities,
    batch_chars: int,
) -> list[PaperContextPacket]:
    """Build deterministic provider-independent packets from ParsedPaper."""
    if capabilities.context_strategy == "full_document":
        packet = _full_document_packet(document)
        return [packet] if packet.excerpts else []
    packet_chars = min(batch_chars, capabilities.max_context_chars // 2)
    return _hierarchical_packets(document, packet_chars)
