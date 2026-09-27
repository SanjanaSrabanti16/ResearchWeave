import json
import re
from types import SimpleNamespace

import httpx
import openai
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.genai import errors
from pydantic import BaseModel

from app.api.routes import router
from app.llm import (
    EVLGemmaProvider,
    GeminiProvider,
    LLMAuthenticationError,
    LLMOutputBudgetExceeded,
    LLMOutputError,
    LLMProvider,
    LLMProviderCapabilities,
    LLMProviderRegistry,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMRequestContext,
    LLMTimeoutError,
    OllamaProvider,
    UnknownLLMProviderError,
)
from app.models.document import DocumentChunk, DocumentSection, ParsedPaper, SourcePDF
from app.models.insights import (
    EVIDENCE_QUOTE_MAX_LENGTH,
    PAPER_OVERVIEW_MAX_LENGTH,
    EvidenceClaim,
    EvidenceReference,
    InsightClaim,
    InsightFields,
    ValidatedEvidenceLedger,
)
from app.services.insight_cache import InsightCache
from app.services.insight_context import (
    EvidenceExcerpt,
    PaperContextPacket,
    _citeable_spans,
    build_paper_context_packets,
)
from app.services.insight_service import InsightOutputError, validate_insights
from app.services.paper_understanding_service import (
    _EVIDENCE_SYSTEM_PROMPT,
    _FULL_DOCUMENT_EVIDENCE_PASSES,
    _SYNTHESIS_SYSTEM_PROMPT,
    PIPELINE_VERSION,
    PaperUnderstandingDiagnostics,
    PaperUnderstandingService,
    _evidence_prompt,
    _EvidenceExtraction,
    _EvidenceExtractionEnvelope,
    _GeneratedEvidenceClaim,
    _GeneratedOverviewClaim,
    _GeneratedSynthesisClaim,
    _parse_evidence_extraction,
    _parse_research_synthesis,
    _preserve_explicit_author_claims,
    _ResearchSynthesis,
    _ResearchSynthesisEnvelope,
    _resolve_generated_evidence,
    _resolve_synthesis,
    _synthesis_prompt,
)


def _is_evidence_schema(schema: type[BaseModel]) -> bool:
    return issubclass(schema, _EvidenceExtractionEnvelope)


def research_paper() -> ParsedPaper:
    abstract = "Current tools cannot organize complex research evidence. We address this gap."
    sections = [
        ("method", "Approach", "We developed a systems framework with five functional subsystems."),
        (
            "contribution",
            "Contributions",
            "We contribute a taxonomy of twelve agentic design patterns.",
        ),
        (
            "evaluation",
            "Evaluation",
            "We evaluated the framework through a ReAct case study.",
        ),
        (
            "result",
            "Results",
            "Our evaluation found that the patterns explained ReAct coordination behavior.",
        ),
        (
            "audience",
            "Discussion",
            "The framework is designed for AI system researchers.",
        ),
        (
            "limitation",
            "Limitations",
            "We did not evaluate the framework with deployed production agents.",
        ),
        (
            "future",
            "Future Work",
            "In future work, we will evaluate the framework on additional agent systems.",
        ),
    ]
    return ParsedPaper(
        paper_id="agentic-patterns",
        title="Agentic Design Patterns: A System-Theoretic Framework",
        authors=["A. Researcher"],
        abstract=abstract,
        abstract_chunks=[
            DocumentChunk(
                id="abstract-1",
                section_id="abstract",
                text=abstract,
                start_char=0,
                end_char=len(abstract),
            )
        ],
        sections=[
            DocumentSection(
                id=section_id,
                heading=heading,
                level=1,
                text=text,
                chunks=[
                    DocumentChunk(
                        id=f"chunk-{section_id}",
                        section_id=section_id,
                        text=text,
                        start_char=0,
                        end_char=len(text),
                    )
                ],
            )
            for section_id, heading, text in sections
        ],
        parser="grobid",
        parser_version="0.9.1-crf+frontmatter-v1",
        source_pdf=SourcePDF(
            acquisition_method="arxiv",
            sha256="a" * 64,
            size_bytes=1000,
        ),
    )


def test_long_source_chunk_creates_stable_exact_bounded_citeable_spans() -> None:
    source = "\n\n".join(
        f"Sentence {index} describes a grounded architectural detail with exact provenance."
        for index in range(1, 19)
    )
    document = ParsedPaper(
        paper_id="long-paper",
        title="Long evidence paper",
        authors=["A. Researcher"],
        sections=[
            DocumentSection(
                id="methods",
                heading="Methods and Evaluation",
                level=1,
                text=source,
                chunks=[
                    DocumentChunk(
                        id="long-chunk",
                        section_id="methods",
                        text=source,
                        start_char=0,
                        end_char=len(source),
                    )
                ],
            )
        ],
        parser="grobid",
        parser_version="0.9.1",
        source_pdf=SourcePDF(
            acquisition_method="arxiv",
            sha256="f" * 64,
            size_bytes=100,
        ),
    )

    first = build_paper_context_packets(
        document,
        GeminiProvider.capabilities,
        batch_chars=13000,
    )
    second = build_paper_context_packets(
        document,
        EVLGemmaProvider.capabilities,
        batch_chars=13000,
    )

    assert first == second
    assert len(first[0].excerpts) >= 3
    assert [item.evidence_id for item in first[0].excerpts] == [
        f"P1-E{index:03d}" for index in range(1, len(first[0].excerpts) + 1)
    ]
    for excerpt in first[0].excerpts:
        assert len(excerpt.quote) <= EVIDENCE_QUOTE_MAX_LENGTH
        assert excerpt.quote in source
        assert excerpt.chunk_id == "long-chunk"
        assert excerpt.section_id == "methods"
        assert EvidenceReference(chunk_id=excerpt.chunk_id, quote=excerpt.quote)


def test_all_provider_context_strategies_use_bounded_exact_source_spans() -> None:
    document = research_paper()
    source_by_chunk = {
        chunk.id: chunk.text
        for chunk in document.abstract_chunks
        + [chunk for section in document.sections for chunk in section.chunks]
    }

    for capabilities in (
        OllamaProvider.capabilities,
        GeminiProvider.capabilities,
        EVLGemmaProvider.capabilities,
    ):
        packets = build_paper_context_packets(document, capabilities, batch_chars=13000)
        assert packets
        for packet in packets:
            for excerpt in packet.excerpts:
                assert len(excerpt.quote) <= EVIDENCE_QUOTE_MAX_LENGTH
                assert excerpt.quote in source_by_chunk[excerpt.chunk_id]
                EvidenceReference(chunk_id=excerpt.chunk_id, quote=excerpt.quote)


def _split_contribution_paper(*, heading: str = "Introduction") -> ParsedPaper:
    texts = [
        "The primary contributions are outlined as:",
        "• We identify a structured security taxonomy for adaptive networks.",
        "• We outline open research issues that guide subsequent investigations.",
    ]
    chunks = [
        DocumentChunk(
            id=f"contribution-{index}",
            section_id="introduction",
            text=text,
            start_char=sum(len(item) + 1 for item in texts[:index]),
            end_char=sum(len(item) + 1 for item in texts[:index]) + len(text),
        )
        for index, text in enumerate(texts)
    ]
    conclusion = "In this paper, we proposed the taxonomy and outlined open research issues."
    return ParsedPaper(
        paper_id="split-contributions",
        title="A Generic Adaptive-Network Study",
        sections=[
            DocumentSection(
                id="introduction",
                heading=heading,
                level=1,
                text="\n".join(texts),
                chunks=chunks,
            ),
            DocumentSection(
                id="open-issues",
                heading="Open Research Issues and Outlook",
                level=1,
                text="Future efforts should establish shared testbeds for realistic evaluation.",
                chunks=[
                    DocumentChunk(
                        id="open-issues-1",
                        section_id="open-issues",
                        text=(
                            "Future efforts should establish shared testbeds for realistic "
                            "evaluation."
                        ),
                        start_char=0,
                        end_char=79,
                    )
                ],
            ),
            DocumentSection(
                id="conclusion",
                heading="Conclusion",
                level=1,
                text=conclusion,
                chunks=[
                    DocumentChunk(
                        id="conclusion-1",
                        section_id="conclusion",
                        text=conclusion,
                        start_char=0,
                        end_char=len(conclusion),
                    )
                ],
            ),
        ],
        parser="grobid",
        parser_version="0.9.1",
        source_pdf=SourcePDF(
            acquisition_method="arxiv",
            sha256="c" * 64,
            size_bytes=100,
        ),
    )


def _context_strategy_paper() -> ParsedPaper:
    abstract = "Existing static tools cannot adapt to changing threats."
    sections = [
        (
            "introduction",
            "Introduction",
            [
                "The primary contributions are outlined as:",
                "• We introduce an adaptive security framework.",
            ],
        ),
        ("methods", "Methods and Approach", ["We implement a three-stage analysis pipeline."]),
        (
            "results",
            "Results and Evaluation",
            ["The evaluation found 84 percent accuracy on the test dataset."],
        ),
        ("discussion", "Discussion", ["The result supports adaptive threat analysis."]),
        (
            "limitations",
            "Limitations and Open Issues",
            [
                "The study did not evaluate production deployments.",
                "Future work will evaluate additional environments.",
            ],
        ),
        ("conclusion", "Conclusion", ["We conclude that the framework merits further study."]),
        ("references", "References", ["[1] External Author. An unrelated cited paper."]),
    ]
    document_sections = []
    for section_id, heading, texts in sections:
        offset = 0
        chunks = []
        for index, text in enumerate(texts):
            chunks.append(
                DocumentChunk(
                    id=f"{section_id}-{index}",
                    section_id=section_id,
                    text=text,
                    start_char=offset,
                    end_char=offset + len(text),
                )
            )
            offset += len(text) + 1
        document_sections.append(
            DocumentSection(
                id=section_id,
                heading=heading,
                level=1,
                text="\n".join(texts),
                chunks=chunks,
            )
        )
    return ParsedPaper(
        paper_id="context-strategy-paper",
        title="Context Strategy Paper",
        abstract=abstract,
        abstract_chunks=[
            DocumentChunk(
                id="abstract-1",
                section_id="abstract",
                text=abstract,
                start_char=0,
                end_char=len(abstract),
            )
        ],
        sections=document_sections,
        parser="grobid",
        parser_version="0.9.1",
        source_pdf=SourcePDF(
            acquisition_method="arxiv",
            sha256="d" * 64,
            size_bytes=100,
        ),
    )


def test_full_document_mode_preserves_complete_reading_order_and_excludes_references() -> None:
    document = _context_strategy_paper()
    packet = build_paper_context_packets(
        document,
        EVLGemmaProvider.capabilities,
        batch_chars=13000,
    )[0]
    assert packet.strategy == "full_document"
    assert [excerpt.chunk_id for excerpt in packet.excerpts] == [
        "abstract-1",
        "introduction-0",
        "introduction-1",
        "methods-0",
        "results-0",
        "discussion-0",
        "limitations-0",
        "limitations-1",
        "conclusion-0",
    ]
    assert all(excerpt.chunk_id != "references-0" for excerpt in packet.excerpts)
    assert all(len(excerpt.quote) <= EVIDENCE_QUOTE_MAX_LENGTH for excerpt in packet.excerpts)


def test_full_document_prompt_preserves_headings_and_split_contribution_structure() -> None:
    document = _context_strategy_paper()
    packet = build_paper_context_packets(
        document,
        GeminiProvider.capabilities,
        batch_chars=13000,
    )[0]
    prompt = _evidence_prompt(document, packet)
    headings = [
        "## Abstract",
        "## Introduction",
        "## Methods and Approach",
        "## Results and Evaluation",
        "## Discussion",
        "## Limitations and Open Issues",
        "## Conclusion",
    ]
    positions = [prompt.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "The primary contributions are outlined as:" in prompt
    assert "We introduce an adaptive security framework." in prompt
    assert "rhetorical_role=explicit_current_paper_contribution_header" in prompt
    assert "rhetorical_role=explicit_current_paper_contribution_item" in prompt
    assert "External Author" not in prompt


def test_provider_context_capabilities_are_explicit_and_model_appropriate() -> None:
    assert EVLGemmaProvider.capabilities.context_strategy == "full_document"
    assert GeminiProvider.capabilities.context_strategy == "full_document"
    assert OllamaProvider.capabilities.context_strategy == "hierarchical_sections"


def test_context_delivery_strategy_does_not_change_final_insight_schema() -> None:
    document = _context_strategy_paper()
    full_packets = build_paper_context_packets(
        document,
        EVLGemmaProvider.capabilities,
        batch_chars=13000,
    )
    hierarchical_packets = build_paper_context_packets(
        document,
        OllamaProvider.capabilities,
        batch_chars=13000,
    )
    assert full_packets[0].strategy != hierarchical_packets[0].strategy
    assert set(InsightFields.model_fields) == {
        "paper_overview",
        "research_problem",
        "methods",
        "key_contributions",
        "evaluation",
        "main_findings",
        "why_it_matters",
        "target_audience",
        "limitations",
        "future_work",
    }


def test_hierarchical_mode_covers_each_research_story_phase_without_references() -> None:
    document = _context_strategy_paper()
    packets = build_paper_context_packets(
        document,
        OllamaProvider.capabilities,
        batch_chars=13000,
    )
    assert all(packet.strategy == "hierarchical_sections" for packet in packets)
    assert [packet.purpose for packet in packets] == [
        "introduction and research problem",
        "methods and approach",
        "results and evaluation",
        "discussion and implications",
        "limitations, open issues, and future work",
        "conclusion",
    ]
    delivered = [excerpt.chunk_id for packet in packets for excerpt in packet.excerpts]
    assert set(delivered) == {
        "abstract-1",
        "introduction-0",
        "introduction-1",
        "methods-0",
        "results-0",
        "discussion-0",
        "limitations-0",
        "limitations-1",
        "conclusion-0",
    }
    assert len(delivered) == len(set(delivered))


def test_split_current_paper_contribution_list_is_labeled_and_preserved() -> None:
    document = _split_contribution_paper()
    packet = build_paper_context_packets(
        document,
        EVLGemmaProvider.capabilities,
        batch_chars=13000,
    )[0]
    contribution_items = [
        excerpt
        for excerpt in packet.excerpts
        if excerpt.rhetorical_role == "explicit_current_paper_contribution_item"
    ]
    assert [item.chunk_id for item in contribution_items] == [
        "contribution-1",
        "contribution-2",
    ]

    diagnostics = PaperUnderstandingDiagnostics()
    generated = _preserve_explicit_author_claims(
        packet,
        _EvidenceExtraction(claims=[]),
        diagnostics,
    )
    ledger = _resolve_generated_evidence(
        [(packet, generated)],
        document,
        StoryProvider("evl_gemma", "gemma4"),
        diagnostics,
    )

    contributions = [claim for claim in ledger.claims if claim.category == "contribution"]
    assert len(contributions) == 2
    assert diagnostics.preserved_author_claims == 2
    assert {claim.evidence_refs[0].chunk_id for claim in contributions} == {
        "contribution-1",
        "contribution-2",
    }


def test_explicit_source_contribution_survives_when_provider_paraphrase_is_rejected() -> None:
    document = _split_contribution_paper()
    packet = build_paper_context_packets(
        document,
        EVLGemmaProvider.capabilities,
        batch_chars=13000,
    )[0]
    item = next(excerpt for excerpt in packet.excerpts if excerpt.chunk_id == "contribution-1")
    provider_claim = _GeneratedEvidenceClaim(
        category="contribution",
        claim="The paper contributes an unsupported clinical intervention taxonomy.",
        evidence_ids=[item.evidence_id],
    )
    diagnostics = PaperUnderstandingDiagnostics()
    generated = _preserve_explicit_author_claims(
        packet,
        _EvidenceExtraction(claims=[provider_claim]),
        diagnostics,
    )
    ledger = _resolve_generated_evidence(
        [(packet, generated)],
        document,
        StoryProvider("evl_gemma", "gemma4"),
        diagnostics,
    )
    retained = [claim for claim in ledger.claims if claim.category == "contribution"]
    assert any("structured security taxonomy" in claim.claim for claim in retained)
    assert all("clinical intervention" not in claim.claim for claim in retained)


def test_provider_and_preserved_contribution_with_same_evidence_are_deduplicated() -> None:
    document = _split_contribution_paper()
    packet = build_paper_context_packets(
        document,
        EVLGemmaProvider.capabilities,
        batch_chars=13000,
    )[0]
    item = next(excerpt for excerpt in packet.excerpts if excerpt.chunk_id == "contribution-1")
    provider_claim = _GeneratedEvidenceClaim(
        category="contribution",
        claim="We identify a structured security taxonomy for adaptive networks.",
        evidence_ids=[item.evidence_id],
    )
    diagnostics = PaperUnderstandingDiagnostics()
    generated = _preserve_explicit_author_claims(
        packet,
        _EvidenceExtraction(claims=[provider_claim]),
        diagnostics,
    )
    ledger = _resolve_generated_evidence(
        [(packet, generated)],
        document,
        StoryProvider("evl_gemma", "gemma4"),
        diagnostics,
    )
    retained = [claim for claim in ledger.claims if claim.category == "contribution"]
    assert (
        len([claim for claim in retained if claim.evidence_refs[0].chunk_id == item.chunk_id]) == 1
    )
    assert any(
        item["reason"] in {"duplicate_claim", "duplicate_evidence_claim"}
        for item in diagnostics.rejected_evidence
    )


def test_related_work_contribution_list_is_not_labeled_as_current_paper() -> None:
    packet = build_paper_context_packets(
        _split_contribution_paper(heading="Related Work"),
        EVLGemmaProvider.capabilities,
        batch_chars=13000,
    )[0]
    assert not any(excerpt.rhetorical_role for excerpt in packet.excerpts)
    generated = _preserve_explicit_author_claims(
        packet,
        _EvidenceExtraction(claims=[]),
        PaperUnderstandingDiagnostics(),
    )
    assert generated.claims == []


def test_open_issues_and_conclusion_receive_large_context_coverage() -> None:
    packet = build_paper_context_packets(
        _split_contribution_paper(),
        EVLGemmaProvider.capabilities,
        batch_chars=13000,
    )[0]
    chunk_ids = {excerpt.chunk_id for excerpt in packet.excerpts}
    assert "open-issues-1" in chunk_ids
    assert "conclusion-1" in chunk_ids


def test_explicit_limitation_survives_but_silence_does_not_create_one() -> None:
    document = research_paper()
    packet = build_paper_context_packets(
        document,
        EVLGemmaProvider.capabilities,
        batch_chars=13000,
    )[0]
    limitation_excerpt = next(
        excerpt for excerpt in packet.excerpts if excerpt.chunk_id == "chunk-limitation"
    )
    generated = _EvidenceExtraction(
        claims=[
            _GeneratedEvidenceClaim(
                category="limitation",
                claim="We did not evaluate the framework with deployed production agents.",
                evidence_ids=[limitation_excerpt.evidence_id],
            )
        ]
    )
    ledger = _resolve_generated_evidence(
        [(packet, generated)],
        document,
        StoryProvider("evl_gemma", "gemma4"),
        PaperUnderstandingDiagnostics(),
    )
    assert [claim.category for claim in ledger.claims] == ["limitation"]
    assert (
        _preserve_explicit_author_claims(
            packet,
            _EvidenceExtraction(claims=[]),
            PaperUnderstandingDiagnostics(),
        ).claims
        == []
    )


def test_synthesis_restores_every_grounded_source_declared_contribution() -> None:
    document = _split_contribution_paper()
    packet = build_paper_context_packets(
        document,
        EVLGemmaProvider.capabilities,
        batch_chars=13000,
    )[0]
    generated_evidence = _preserve_explicit_author_claims(
        packet,
        _EvidenceExtraction(claims=[]),
        PaperUnderstandingDiagnostics(),
    )
    ledger = _resolve_generated_evidence(
        [(packet, generated_evidence)],
        document,
        StoryProvider("evl_gemma", "gemma4"),
        PaperUnderstandingDiagnostics(),
    )
    first = ledger.claims[0]
    generated = _ResearchSynthesis(
        key_contributions=[
            _GeneratedSynthesisClaim(
                claim=first.claim,
                evidence_claim_ids=[first.claim_id],
            )
        ]
    )
    diagnostics = PaperUnderstandingDiagnostics()

    result = _resolve_synthesis(generated, ledger, document, diagnostics)

    assert len(result.key_contributions) == 2
    assert {item.evidence[0].chunk_id for item in result.key_contributions} == {
        "contribution-1",
        "contribution-2",
    }
    assert diagnostics.restored_explicit_contributions == 1


def test_grounded_open_challenge_is_visible_without_becoming_a_self_limitation() -> None:
    document = _split_contribution_paper()
    challenge = (
        "Designing a secure and trustworthy architecture for adaptive networks remains a "
        "challenge. Future research can focus on interoperable defenses."
    )
    document.sections[1] = DocumentSection(
        id="open-issues",
        heading="Open Research Issues and Outlook",
        level=1,
        text=challenge,
        chunks=[
            DocumentChunk(
                id="open-issues-1",
                section_id="open-issues",
                text=challenge,
                start_char=0,
                end_char=len(challenge),
            )
        ],
    )
    ledger = ValidatedEvidenceLedger(
        claims=[
            EvidenceClaim(
                claim_id="EC001",
                category="future_work",
                claim="Research into interoperable defenses for trustworthy adaptive networks.",
                evidence_refs=[EvidenceReference(chunk_id="open-issues-1", quote=challenge)],
                provider="evl_gemma",
            )
        ]
    )
    diagnostics = PaperUnderstandingDiagnostics()

    result = _resolve_synthesis(_ResearchSynthesis(), ledger, document, diagnostics)

    assert len(result.limitations) == 1
    assert result.limitations[0].claim.startswith(
        "The paper does not explicitly frame this as a study limitation"
    )
    assert result.limitations[0].evidence[0].quote == challenge
    assert result.future_work
    assert diagnostics.restored_open_challenges == 1


def test_plain_future_direction_is_not_recast_as_an_open_challenge() -> None:
    document = _split_contribution_paper()
    future = "Future work will evaluate additional datasets."
    ledger = ValidatedEvidenceLedger(
        claims=[
            EvidenceClaim(
                claim_id="EC001",
                category="future_work",
                claim="Evaluation on additional datasets.",
                evidence_refs=[EvidenceReference(chunk_id="open-issues-1", quote=future)],
                provider="evl_gemma",
            )
        ]
    )

    result = _resolve_synthesis(
        _ResearchSynthesis(), ledger, document, PaperUnderstandingDiagnostics()
    )

    assert result.limitations == []


def test_prompts_require_researcher_explanation_depth_without_maximal_brevity() -> None:
    document = _split_contribution_paper()
    packet = build_paper_context_packets(
        document,
        EVLGemmaProvider.capabilities,
        batch_chars=13000,
    )[0]
    evidence_prompt = _EVIDENCE_SYSTEM_PROMPT + _evidence_prompt(document, packet)
    ledger = ValidatedEvidenceLedger(
        claims=[
            EvidenceClaim(
                claim_id="EC001",
                category="contribution",
                claim="We identify a structured security taxonomy for adaptive networks.",
                evidence_refs=[
                    EvidenceReference(
                        chunk_id="contribution-1",
                        quote=(
                            "• We identify a structured security taxonomy for adaptive networks."
                        ),
                    )
                ],
                provider="evl_gemma",
            )
        ]
    )
    synthesis_prompt = _SYNTHESIS_SYSTEM_PROMPT + _synthesis_prompt(document, ledger)
    assert "every major distinct item" in evidence_prompt
    assert "classify that evidence as approach_method" in evidence_prompt
    assert "knowledgeable researcher" in synthesis_prompt
    assert "information-dense paragraph" in synthesis_prompt
    assert "including important components, stages, data, taxonomies, or procedures" in (
        synthesis_prompt
    )
    assert "specific quantitative or qualitative outcomes" in synthesis_prompt
    assert "do not optimize for short one-line summaries" in synthesis_prompt
    assert "maximally brief" not in synthesis_prompt


def test_overview_can_cite_nine_validated_claims_for_research_story_depth() -> None:
    overview = _GeneratedOverviewClaim(
        claim="A grounded, information-dense research overview.",
        evidence_claim_ids=[f"EC{index:03d}" for index in range(1, 10)],
    )
    assert len(overview.evidence_claim_ids) == 9


def test_citeable_span_split_avoids_midword_boundaries_when_whitespace_exists() -> None:
    source = " ".join(f"token{index}" for index in range(180))
    spans = _citeable_spans(source, limit=80)

    assert len(spans) > 1
    assert all(len(span) <= 80 and span in source for span in spans)
    assert "".join(spans) != source
    assert " ".join(spans) == source


class StoryProvider(LLMProvider):
    capabilities = LLMProviderCapabilities(
        structured_json=True,
        context_strategy="full_document",
        cloud=True,
        max_context_chars=100000,
        max_extraction_packets=1,
    )

    def __init__(
        self,
        provider_id: str,
        model_id: str,
        *,
        configured: bool = True,
        include_bad_claim: bool = False,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self._configured = configured
        self.include_bad_claim = include_bad_claim
        self.prompts: list[str] = []
        self.contexts: list[LLMRequestContext | None] = []

    @property
    def configured(self) -> bool:
        return self._configured

    @staticmethod
    def _evidence_ids(prompt: str) -> dict[str, str]:
        ids: dict[str, str] = {}
        matches = list(re.finditer(r"\[(P\d+-E\d+)\].*?\n([^\n]+)", prompt))
        for match in matches:
            text = match.group(2)
            for key in (
                "Current tools",
                "We developed",
                "We contribute",
                "We evaluated",
                "Our evaluation found",
                "designed for",
                "did not evaluate",
                "future work",
            ):
                if key in text:
                    ids[key] = match.group(1)
        return ids

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        max_output_tokens: int,
        context=None,
    ) -> str:
        del system_prompt, max_output_tokens
        self.prompts.append(user_prompt)
        self.contexts.append(context)
        if _is_evidence_schema(schema):
            ids = self._evidence_ids(user_prompt)
            claims = [
                {
                    "category": "problem",
                    "claim": "Current tools cannot organize complex research evidence.",
                    "evidence_ids": [ids["Current tools"]],
                    "explicitness": "explicit",
                },
                {
                    "category": "approach_method",
                    "claim": (
                        "The authors developed a systems framework with five functional subsystems."
                    ),
                    "evidence_ids": [ids["We developed"]],
                    "explicitness": "explicit",
                },
                {
                    "category": "contribution",
                    "claim": "The paper contributes a taxonomy of twelve agentic design patterns.",
                    "evidence_ids": [ids["We contribute"]],
                    "explicitness": "explicit",
                },
                {
                    "category": "evaluation",
                    "claim": "The framework was evaluated through a ReAct case study.",
                    "evidence_ids": [ids["We evaluated"]],
                    "explicitness": "explicit",
                },
                {
                    "category": "finding",
                    "claim": "The patterns explained ReAct coordination behavior.",
                    "evidence_ids": [ids["Our evaluation found"]],
                    "explicitness": "explicit",
                },
                {
                    "category": "audience",
                    "claim": "The framework is designed for AI system researchers.",
                    "evidence_ids": [ids["designed for"]],
                    "explicitness": "explicit",
                },
                {
                    "category": "limitation",
                    "claim": "The framework was not evaluated with deployed production agents.",
                    "evidence_ids": [ids["did not evaluate"]],
                    "explicitness": "explicit",
                },
                {
                    "category": "future_work",
                    "claim": "Future evaluation on additional agent systems.",
                    "evidence_ids": [ids["future work"]],
                    "explicitness": "explicit",
                },
            ]
            if self.include_bad_claim:
                claims.append(
                    {
                        "category": "future_work",
                        "claim": "Invent a clinical deployment.",
                        "evidence_ids": ["P1-E999"],
                        "explicitness": "synthesized",
                    }
                )
            allowed_match = re.search(r"Emit ONLY these categories: ([^.]+)\.", user_prompt)
            if allowed_match:
                allowed = {item.strip() for item in allowed_match.group(1).split(",")}
                claims = [claim for claim in claims if claim["category"] in allowed]
            return json.dumps({"claims": claims})

        ledger = {
            category: claim_id
            for claim_id, category in re.findall(r"\[(EC\d+)\] category=([a-z_]+)", user_prompt)
        }
        empty = {name: [] for name in schema.model_fields}
        empty["paper_overview"] = [
            {
                "claim": (
                    "The paper addresses complex research evidence with a systems framework "
                    "evaluated through a ReAct case study."
                ),
                "evidence_claim_ids": [
                    ledger["problem"],
                    ledger["approach_method"],
                    ledger["evaluation"],
                ],
            }
        ]
        for field_name, category in (
            ("research_problem", "problem"),
            ("methods", "approach_method"),
            ("key_contributions", "contribution"),
            ("evaluation", "evaluation"),
            ("main_findings", "finding"),
            ("target_audience", "audience"),
            ("limitations", "limitation"),
            ("future_work", "future_work"),
        ):
            match = re.search(rf"\[{ledger[category]}\].*?claim=([^\n]+)", user_prompt)
            empty[field_name] = [
                {
                    "claim": match.group(1),
                    "evidence_claim_ids": [ledger[category]],
                }
            ]
        empty["why_it_matters"] = [
            {
                "claim": (
                    "The systems framework addresses complex research evidence by explaining "
                    "agent coordination behavior."
                ),
                "evidence_claim_ids": [ledger["problem"], ledger["finding"]],
            }
        ]
        return json.dumps(empty)


class StructuralClaimsProvider(LLMProvider):
    provider_id = "structural-test"
    model_id = "test-model"
    capabilities = LLMProviderCapabilities(
        structured_json=True,
        context_strategy="full_document",
        cloud=False,
        max_context_chars=100000,
        max_extraction_packets=1,
    )

    def __init__(self, all_invalid: bool = False) -> None:
        self.all_invalid = all_invalid
        self.prompts: list[str] = []

    @property
    def configured(self) -> bool:
        return True

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        max_output_tokens: int,
        context=None,
    ) -> str:
        del system_prompt, max_output_tokens, context
        self.prompts.append(user_prompt)
        if _is_evidence_schema(schema):
            if "pass_a_core_research_design" not in user_prompt:
                return '{"claims":[]}'
            evidence_id = StoryProvider._evidence_ids(user_prompt)["Current tools"]
            invalid = {
                "category": "approach_method",
                "claim": "Structurally invalid method must not reach validation.",
                "evidence_ids": [evidence_id] * 5,
                "explicitness": "explicit",
            }
            claims = [invalid]
            if not self.all_invalid:
                claims.insert(
                    0,
                    {
                        "category": "problem",
                        "claim": "Current tools cannot organize complex research evidence.",
                        "evidence_ids": [evidence_id],
                        "explicitness": "explicit",
                    },
                )
            return json.dumps({"claims": claims})

        claim_id = re.search(r"\[(EC\d+)\] category=problem", user_prompt).group(1)
        output = {name: [] for name in schema.model_fields}
        output["research_problem"] = [
            {
                "claim": "Current tools cannot organize complex research evidence.",
                "evidence_claim_ids": [claim_id],
            }
        ]
        return json.dumps(output)


class OneFailedPassProvider(StoryProvider):
    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        max_output_tokens: int,
        context=None,
    ) -> str:
        if context is not None and context.extraction_pass == "pass_b_evaluation_results":
            self.prompts.append(user_prompt)
            self.contexts.append(context)
            raise LLMOutputBudgetExceeded("synthetic pass budget")
        if _is_evidence_schema(schema):
            return await super().generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
                max_output_tokens=max_output_tokens,
                context=context,
            )

        self.prompts.append(user_prompt)
        self.contexts.append(context)
        ledger = {
            category: claim_id
            for claim_id, category in re.findall(r"\[(EC\d+)\] category=([a-z_]+)", user_prompt)
        }
        output = {name: [] for name in schema.model_fields}
        output["paper_overview"] = [
            {
                "claim": (
                    "The paper addresses complex research evidence with a systems framework "
                    "and a taxonomy of agentic design patterns."
                ),
                "evidence_claim_ids": [
                    ledger["problem"],
                    ledger["approach_method"],
                    ledger["contribution"],
                ],
            }
        ]
        return json.dumps(output)


def test_provider_registry_registers_all_providers_and_rejects_unknown() -> None:
    http_client = httpx.AsyncClient()
    registry = LLMProviderRegistry(
        [
            OllamaProvider(http_client, "http://localhost:11434", "qwen3:1.7b"),
            GeminiProvider(None, "gemini-3.5-flash", client=SimpleNamespace()),
            EVLGemmaProvider(
                None,
                "https://sage200.evl.uic.edu",
                "gemma4",
                client=SimpleNamespace(),
            ),
        ]
    )
    assert registry.get("ollama").provider_id == "ollama"
    assert registry.get("gemini").provider_id == "gemini"
    assert registry.get("evl_gemma").provider_id == "evl_gemma"
    with pytest.raises(UnknownLLMProviderError):
        registry.get("unknown")


class _TinySchema(BaseModel):
    value: str


class _FakeGeminiModels:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.result)


class _SequencedFakeGeminiModels:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


async def _no_sleep(_delay: float) -> None:
    return None


def _gemini_with(models: _FakeGeminiModels) -> GeminiProvider:
    return GeminiProvider(
        "configured-test-key",
        "gemini-3.5-flash",
        client=SimpleNamespace(aio=SimpleNamespace(models=models)),
        sleep=_no_sleep,
        jitter=lambda: 0.0,
    )


@pytest.mark.asyncio
async def test_gemini_missing_key_success_and_malformed_output() -> None:
    missing = GeminiProvider(None, "gemini-3.5-flash")
    with pytest.raises(LLMProviderUnavailableError, match="GEMINI_API_KEY"):
        await missing.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )

    models = _FakeGeminiModels('{"value":"ok"}')
    provider = _gemini_with(models)
    assert json.loads(
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )
    ) == {"value": "ok"}
    assert models.calls[0]["model"] == "gemini-3.5-flash"
    config = models.calls[0]["config"]
    assert config.response_json_schema is not None
    assert "additionalProperties" not in json.dumps(config.response_json_schema)

    malformed = _gemini_with(_FakeGeminiModels("not-json"))
    with pytest.raises(LLMOutputError, match=r"structured.?output"):
        await malformed.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", [_EvidenceExtractionEnvelope, _ResearchSynthesisEnvelope])
async def test_gemini_sanitizes_current_common_schema_only_at_adapter_boundary(
    schema: type[BaseModel],
) -> None:
    content = '{"claims":[]}' if schema is _EvidenceExtractionEnvelope else "{}"
    models = _FakeGeminiModels(content)
    provider = _gemini_with(models)

    await provider.generate_structured(
        system_prompt="system",
        user_prompt="user",
        schema=schema,
        max_output_tokens=20,
    )

    sent_schema = json.dumps(models.calls[0]["config"].response_json_schema)
    assert "additionalProperties" not in sent_schema
    assert "minLength" not in sent_schema
    assert "maxLength" not in sent_schema
    assert '"default"' not in sent_schema
    assert "minItems" in sent_schema
    assert "maxItems" in sent_schema
    assert "minLength" in json.dumps(schema.model_json_schema())


@pytest.mark.asyncio
async def test_gemini_timeout_rate_limit_and_secret_redaction(caplog) -> None:
    timeout_provider = _gemini_with(_FakeGeminiModels(error=TimeoutError()))
    with pytest.raises(LLMTimeoutError):
        await timeout_provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )

    secret = "must-never-appear"
    rate_error = errors.APIError(429, {"message": f"rate limited {secret}"})
    limited = GeminiProvider(
        secret,
        "gemini-3.5-flash",
        client=SimpleNamespace(aio=SimpleNamespace(models=_FakeGeminiModels(error=rate_error))),
        sleep=_no_sleep,
        jitter=lambda: 0.0,
    )
    with pytest.raises(LLMRateLimitError) as caught:
        await limited.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )
    assert secret not in str(caught.value)
    assert secret not in caplog.text

    auth = _gemini_with(
        _FakeGeminiModels(
            error=errors.APIError(
                400,
                {
                    "error": {
                        "message": f"bad key {secret}",
                        "details": [{"reason": "API_KEY_INVALID"}],
                    }
                },
            )
        )
    )
    with pytest.raises(LLMAuthenticationError, match="authentication failed") as auth_error:
        await auth.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )
    assert secret not in str(auth_error.value)

    unavailable = _gemini_with(
        _FakeGeminiModels(error=errors.APIError(503, {"message": f"down {secret}"}))
    )
    with pytest.raises(LLMProviderUnavailableError, match="unavailable") as unavailable_error:
        await unavailable.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )
    assert secret not in str(unavailable_error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [429, 503])
async def test_gemini_retries_transient_errors_with_bounded_backoff(code: int) -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    models = _SequencedFakeGeminiModels(
        [
            errors.APIError(code, {"message": "transient"}),
            errors.APIError(code, {"message": "transient"}),
            '{"value":"ok"}',
        ]
    )
    provider = GeminiProvider(
        "configured-test-key",
        "gemini-3.5-flash",
        client=SimpleNamespace(aio=SimpleNamespace(models=models)),
        sleep=record_sleep,
        jitter=lambda: 0.25,
    )

    result = await provider.generate_structured(
        system_prompt="system",
        user_prompt="user",
        schema=_TinySchema,
        max_output_tokens=20,
    )

    assert json.loads(result) == {"value": "ok"}
    assert len(models.calls) == 3
    assert delays == [2.25, 4.25]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "expected"),
    [(400, LLMOutputError), (401, LLMAuthenticationError)],
)
async def test_gemini_does_not_retry_deterministic_errors(
    code: int, expected: type[Exception]
) -> None:
    models = _SequencedFakeGeminiModels([errors.APIError(code, {"message": "deterministic"})])
    provider = GeminiProvider(
        "configured-test-key",
        "gemini-3.5-flash",
        client=SimpleNamespace(aio=SimpleNamespace(models=models)),
    )

    with pytest.raises(expected):
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )

    assert len(models.calls) == 1


@pytest.mark.asyncio
async def test_gemini_retry_attempts_are_capped_at_three() -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    models = _SequencedFakeGeminiModels(
        [errors.APIError(503, {"message": "unavailable"}) for _ in range(3)]
    )
    provider = GeminiProvider(
        "configured-test-key",
        "gemini-3.5-flash",
        client=SimpleNamespace(aio=SimpleNamespace(models=models)),
        sleep=record_sleep,
        jitter=lambda: 0.0,
    )

    with pytest.raises(LLMProviderUnavailableError):
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )

    assert len(models.calls) == 3
    assert delays == [2.0, 4.0]


class _FakeOpenAICompletions:
    def __init__(self, result: str | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.result))]
        )


class _SequencedFakeOpenAICompletions:
    def __init__(self, results: list[str | None]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results[len(self.calls) - 1]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=result), finish_reason="stop")]
        )


class _LengthLimitedOpenAICompletions:
    def __init__(self, result: str | None) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.result),
                    finish_reason="length",
                )
            ]
        )


def _evl_with(completions, api_key: str = "test-key", diagnostic_dir=None) -> EVLGemmaProvider:
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return EVLGemmaProvider(
        api_key,
        "https://sage200.evl.uic.edu",
        "gemma4",
        client=client,
        diagnostic_dir=diagnostic_dir,
    )


def _openai_status_error(error_type, status_code: int, message: str) -> Exception:
    response = SimpleNamespace(
        status_code=status_code,
        request=SimpleNamespace(),
        headers={},
    )
    return error_type(message, response=response, body={"error": message})


@pytest.mark.asyncio
async def test_evl_gemma_missing_key_success_and_malformed_output() -> None:
    missing = EVLGemmaProvider(None, "https://sage200.evl.uic.edu", "gemma4")
    assert missing.configured is False
    with pytest.raises(LLMProviderUnavailableError, match="EVL_GEMMA_API_KEY"):
        await missing.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )

    valid_envelopes = (
        '{"value":"ok"}',
        '```json\n{"value":"ok"}\n```',
        '```\n{"value":"ok"}\n```',
    )
    valid_calls = []
    for content in valid_envelopes:
        completions = _FakeOpenAICompletions(content)
        provider = _evl_with(completions)
        assert json.loads(
            await provider.generate_structured(
                system_prompt="system",
                user_prompt="user",
                schema=_TinySchema,
                max_output_tokens=20,
            )
        ) == {"value": "ok"}
        valid_calls.extend(completions.calls)
    call = valid_calls[0]
    assert call["model"] == "gemma4"
    assert call["temperature"] == 0
    assert call["max_tokens"] == 20
    assert call["response_format"] == {"type": "json_object"}
    assert '"properties":{"value"' in call["messages"][1]["content"]
    assert "trailing commas" in call["messages"][1]["content"]
    assert "never exceed an array's maxItems" in call["messages"][1]["content"]

    extra_text = _evl_with(_FakeOpenAICompletions('Here is JSON:\n```json\n{"value":"ok"}\n```'))
    with pytest.raises(LLMOutputError, match=r"structured.?output"):
        await extra_text.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )

    malformed = _evl_with(_FakeOpenAICompletions('{"value":"ok",}'))
    with pytest.raises(LLMOutputError, match=r"structured.?output"):
        await malformed.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )

    schema_invalid = _evl_with(_FakeOpenAICompletions('{"wrong":"field"}'))
    with pytest.raises(LLMOutputError, match="invalid output"):
        await schema_invalid.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )


@pytest.mark.asyncio
async def test_evl_corrects_malformed_serialization_once_without_changing_fields() -> None:
    original = '{"value":"preserved",}'
    corrected = '{"value":"preserved"}'
    completions = _SequencedFakeOpenAICompletions([original, corrected])
    provider = _evl_with(completions)

    result = await provider.generate_structured(
        system_prompt="system",
        user_prompt="user",
        schema=_TinySchema,
        max_output_tokens=40,
    )

    assert json.loads(result) == {"value": "preserved"}
    assert len(completions.calls) == 2
    assert provider.last_format_correction_used is True
    assert provider.last_generation_call_count == 2
    correction = completions.calls[1]["messages"]
    assert (
        "Preserve the semantic content, claim text, categories, and evidence IDs"
        in correction[0]["content"]
    )
    assert original in correction[1]["content"]


@pytest.mark.asyncio
async def test_evl_correction_preserves_claim_category_and_evidence_ids() -> None:
    expected = {
        "claims": [
            {
                "category": "finding",
                "claim": "The case study identified a coordination failure.",
                "evidence_ids": ["P1-E001", "P1-E002"],
                "explicitness": "explicit",
            }
        ]
    }
    original = json.dumps(expected)[:-1] + ",}"
    corrected = json.dumps(expected)
    completions = _SequencedFakeOpenAICompletions([original, corrected])
    provider = _evl_with(completions)

    result = await provider.generate_structured(
        system_prompt="system",
        user_prompt="user",
        schema=_EvidenceExtractionEnvelope,
        max_output_tokens=200,
    )

    assert json.loads(result) == expected
    assert len(completions.calls) == 2


@pytest.mark.asyncio
async def test_evl_format_correction_failure_stops_after_one_attempt() -> None:
    completions = _SequencedFakeOpenAICompletions(['{"value":"ok",}', '{"value":"ok",}'])
    provider = _evl_with(completions)

    with pytest.raises(LLMOutputError, match="correction failed"):
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=40,
        )

    assert len(completions.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ['{"claims":[', None])
async def test_evl_length_finish_is_output_budget_exceeded_without_correction(
    content: str | None,
    tmp_path,
) -> None:
    completions = _LengthLimitedOpenAICompletions(content)
    provider = _evl_with(completions, diagnostic_dir=tmp_path)

    with pytest.raises(LLMOutputBudgetExceeded, match="output budget"):
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_EvidenceExtractionEnvelope,
            max_output_tokens=40,
            context=LLMRequestContext(
                stage="evidence_extraction",
                paper_id="paper-1",
                paper_title="Paper",
                run_id="budget-run",
                extraction_pass="pass_a_core_research_design",
            ),
        )

    assert len(completions.calls) == 1
    assert provider.last_format_correction_used is False
    artifact = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert artifact["json_parse_error"] == "output_budget_exceeded"
    assert artifact["extraction_pass"] == "pass_a_core_research_design"


@pytest.mark.asyncio
async def test_evl_schema_failure_is_not_format_corrected() -> None:
    completions = _SequencedFakeOpenAICompletions(['{"wrong":"field"}'])
    provider = _evl_with(completions)

    with pytest.raises(LLMOutputError, match="structurally invalid"):
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=40,
        )

    assert len(completions.calls) == 1


@pytest.mark.asyncio
async def test_evl_diagnostic_artifacts_never_store_secret(tmp_path) -> None:
    secret = "credential-must-not-be-recorded"
    completions = _SequencedFakeOpenAICompletions(['{"value":"ok",}', '{"value":"ok",}'])
    provider = _evl_with(completions, api_key=secret, diagnostic_dir=tmp_path)

    with pytest.raises(LLMOutputError):
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=40,
            context=LLMRequestContext(
                stage="evidence_extraction",
                paper_id="paper-1",
                paper_title="Safe title",
                run_id="safe-run",
            ),
        )

    artifacts = list(tmp_path.glob("*.json"))
    assert len(artifacts) == 2
    assert all(secret not in artifact.read_text(encoding="utf-8") for artifact in artifacts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            _openai_status_error(openai.AuthenticationError, 401, "bad credential"),
            LLMAuthenticationError,
        ),
        (
            _openai_status_error(openai.RateLimitError, 429, "limited"),
            LLMRateLimitError,
        ),
        (openai.APITimeoutError(SimpleNamespace()), LLMTimeoutError),
        (
            _openai_status_error(openai.InternalServerError, 503, "unavailable"),
            LLMProviderUnavailableError,
        ),
    ],
)
async def test_evl_gemma_maps_common_errors(error: Exception, expected: type[Exception]) -> None:
    provider = _evl_with(_FakeOpenAICompletions(error=error))
    with pytest.raises(expected):
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )


@pytest.mark.asyncio
async def test_evl_gemma_never_exposes_secret(caplog) -> None:
    secret = "must-never-appear"
    error = _openai_status_error(openai.AuthenticationError, 401, f"bad {secret}")
    provider = _evl_with(_FakeOpenAICompletions(error=error), api_key=secret)
    with pytest.raises(LLMAuthenticationError) as caught:
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema=_TinySchema,
            max_output_tokens=20,
        )
    assert secret not in str(caught.value)
    assert secret not in caplog.text


def _raw_evidence_claim(
    *,
    category: str = "problem",
    claim: str = "Current tools cannot organize complex research evidence.",
    evidence_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "category": category,
        "claim": claim,
        "evidence_ids": evidence_ids or ["P1-E001"],
        "explicitness": "explicit",
    }


def test_independent_claim_parsing_keeps_all_valid_claims() -> None:
    diagnostics = PaperUnderstandingDiagnostics()
    content = json.dumps(
        {
            "claims": [
                _raw_evidence_claim(),
                _raw_evidence_claim(
                    category="contribution",
                    claim="The paper contributes a reusable framework.",
                    evidence_ids=["P1-E002"],
                ),
            ]
        }
    )

    result = _parse_evidence_extraction(content, "test", diagnostics)

    assert len(result.claims) == 2
    assert diagnostics.raw_evidence_claims == 2
    assert diagnostics.structurally_valid_claims == 2
    assert diagnostics.structurally_rejected_claims == 0


def test_independent_claim_parsing_rejects_too_many_evidence_ids_only() -> None:
    diagnostics = PaperUnderstandingDiagnostics()
    content = json.dumps(
        {
            "claims": [
                _raw_evidence_claim(),
                _raw_evidence_claim(
                    category="contribution",
                    claim="This claim cites too many excerpts.",
                    evidence_ids=[f"P1-E00{index}" for index in range(1, 6)],
                ),
            ]
        }
    )

    result = _parse_evidence_extraction(content, "test", diagnostics)

    assert [claim.category for claim in result.claims] == ["problem"]
    assert diagnostics.structurally_rejected_claims == 1
    assert diagnostics.rejected_evidence[-1]["reason"] == "too_many_evidence_ids"


def test_independent_claim_parsing_rejects_invalid_category_only() -> None:
    diagnostics = PaperUnderstandingDiagnostics()
    content = json.dumps(
        {
            "claims": [
                _raw_evidence_claim(),
                _raw_evidence_claim(category="invented_category"),
            ]
        }
    )

    result = _parse_evidence_extraction(content, "test", diagnostics)

    assert len(result.claims) == 1
    assert diagnostics.rejected_evidence[-1]["reason"] == "invalid_category"


def test_independent_synthesis_parsing_rejects_oversized_item_only() -> None:
    diagnostics = PaperUnderstandingDiagnostics()
    content = json.dumps(
        {
            "research_problem": [
                {
                    "claim": "A concise supported problem.",
                    "evidence_claim_ids": ["EC001"],
                },
                {
                    "claim": "x" * 501,
                    "evidence_claim_ids": ["EC001"],
                },
            ]
        }
    )

    result = _parse_research_synthesis(content, "test", diagnostics)

    assert [item.claim for item in result.research_problem] == ["A concise supported problem."]
    assert diagnostics.raw_synthesis_claims == 2
    assert diagnostics.structurally_rejected_synthesis_claims == 1
    assert diagnostics.rejected_evidence[-1]["reason"] == "synthesis_claim_too_long"


def test_synthesis_field_overflow_rejects_only_extra_item() -> None:
    diagnostics = PaperUnderstandingDiagnostics()
    items = [{"claim": f"Problem {index}", "evidence_claim_ids": ["EC001"]} for index in range(5)]

    result = _parse_research_synthesis(json.dumps({"research_problem": items}), "test", diagnostics)

    assert len(result.research_problem) == 4
    assert diagnostics.raw_synthesis_claims == 5
    assert diagnostics.structurally_rejected_synthesis_claims == 1
    assert diagnostics.rejected_evidence[-1]["reason"] == "too_many_synthesis_items"


@pytest.mark.parametrize("content", ["not-json", '{"research_problem": "not-a-list"}'])
def test_invalid_synthesis_response_envelope_fails_whole_response(content: str) -> None:
    with pytest.raises(InsightOutputError, match="invalid synthesis response envelope"):
        _parse_research_synthesis(content, "test", PaperUnderstandingDiagnostics())


@pytest.mark.parametrize("content", ["not-json", "{}"])
def test_invalid_evidence_response_envelope_fails_whole_response(content: str) -> None:
    with pytest.raises(InsightOutputError, match="invalid evidence response envelope"):
        _parse_evidence_extraction(content, "test", PaperUnderstandingDiagnostics())


@pytest.mark.asyncio
async def test_all_structurally_invalid_claims_fail_extraction(tmp_path) -> None:
    provider = StructuralClaimsProvider(all_invalid=True)
    service = PaperUnderstandingService(
        LLMProviderRegistry([provider], default_provider=provider.provider_id),
        InsightCache(tmp_path),
    )
    diagnostics = PaperUnderstandingDiagnostics()

    with pytest.raises(InsightOutputError, match="no structurally valid evidence claims"):
        await service.extract(research_paper(), diagnostics=diagnostics)

    assert diagnostics.raw_evidence_claims == 1
    assert diagnostics.structurally_valid_claims == 0
    assert diagnostics.structurally_rejected_claims == 1
    assert diagnostics.rejected_evidence[0]["reason"] == "too_many_evidence_ids"


@pytest.mark.asyncio
async def test_only_structurally_valid_claims_reach_validation_and_synthesis(tmp_path) -> None:
    provider = StructuralClaimsProvider()
    service = PaperUnderstandingService(
        LLMProviderRegistry([provider], default_provider=provider.provider_id),
        InsightCache(tmp_path),
    )
    diagnostics = PaperUnderstandingDiagnostics()

    result = await service.extract(research_paper(), diagnostics=diagnostics)

    assert diagnostics.raw_evidence_claims == 2
    assert diagnostics.structurally_valid_claims == 1
    assert diagnostics.structurally_rejected_claims == 1
    assert diagnostics.validated_evidence_claims == 1
    assert "Structurally invalid method" not in provider.prompts[-1]
    assert "Current tools cannot organize" in provider.prompts[-1]
    assert result.insights.research_problem


@pytest.mark.asyncio
async def test_evidence_first_pipeline_validates_before_synthesis_and_maps_refs(tmp_path) -> None:
    provider = StoryProvider("gemini", "gemini-3.5-flash", include_bad_claim=True)
    service = PaperUnderstandingService(
        LLMProviderRegistry([provider], default_provider="gemini"),
        InsightCache(tmp_path),
    )
    diagnostics = PaperUnderstandingDiagnostics()
    result = await service.extract(research_paper(), diagnostics=diagnostics)

    assert diagnostics.model_calls == 4
    assert diagnostics.raw_evidence_claims == 9
    assert diagnostics.validated_evidence_claims == 8
    assert any(item["reason"] == "unknown_evidence_id" for item in diagnostics.rejected_evidence)
    assert "Invent a clinical deployment" not in provider.prompts[-1]
    assert result.insights.paper_overview
    assert result.insights.evaluation
    assert result.insights.future_work
    assert all(
        reference.chunk_id.startswith(("abstract", "chunk-"))
        for field_name in InsightFields.model_fields
        for claim in getattr(result.insights, field_name)
        for reference in claim.evidence
    )


@pytest.mark.asyncio
async def test_full_document_semantic_passes_reuse_identical_complete_evidence(tmp_path) -> None:
    provider = StoryProvider("evl_gemma", "gemma4")
    service = PaperUnderstandingService(
        LLMProviderRegistry([provider], default_provider=provider.provider_id),
        InsightCache(tmp_path),
    )
    diagnostics = PaperUnderstandingDiagnostics()

    result = await service.extract(research_paper(), diagnostics=diagnostics)

    evidence_prompts = provider.prompts[:3]
    evidence_payloads = [
        prompt.split("Paper text with citeable evidence locations:\n", maxsplit=1)[1].split(
            "\n\nUnderstand the paper as research.", maxsplit=1
        )[0]
        for prompt in evidence_prompts
    ]
    assert evidence_payloads[0] == evidence_payloads[1] == evidence_payloads[2]
    evidence_ids = [re.findall(r"\[(P\d+-E\d+)\]", payload) for payload in evidence_payloads]
    assert evidence_ids[0] == evidence_ids[1] == evidence_ids[2]
    assert "We contribute a taxonomy" in evidence_payloads[0]
    assert "problem, approach_method, contribution" in evidence_prompts[0]
    assert "evaluation, finding, significance" in evidence_prompts[1]
    assert "limitation, future_work, audience" in evidence_prompts[2]
    assert [context.extraction_pass for context in provider.contexts[:3]] == [
        extraction_pass.pass_id for extraction_pass in _FULL_DOCUMENT_EVIDENCE_PASSES
    ]
    assert [item["status"] for item in diagnostics.extraction_passes] == ["ok", "ok", "ok"]
    assert diagnostics.model_calls == 4
    assert result.insights.key_contributions


@pytest.mark.asyncio
async def test_failed_full_document_pass_keeps_valid_sibling_passes(tmp_path) -> None:
    provider = OneFailedPassProvider("evl_gemma", "gemma4")
    service = PaperUnderstandingService(
        LLMProviderRegistry([provider], default_provider=provider.provider_id),
        InsightCache(tmp_path),
    )
    diagnostics = PaperUnderstandingDiagnostics()

    result = await service.extract(research_paper(), diagnostics=diagnostics)

    assert diagnostics.partial is True
    assert diagnostics.failed_extraction_passes == [
        {
            "pass_id": "pass_b_evaluation_results",
            "reason": "output_budget_exceeded",
            "message": "synthetic pass budget",
        }
    ]
    assert result.insights.paper_overview
    assert result.insights.methods
    assert result.insights.key_contributions
    assert not result.insights.evaluation
    assert "category=evaluation" not in provider.prompts[-1]


def test_invented_chunk_bad_quote_prior_work_and_future_work_are_rejected() -> None:
    document = research_paper()
    bad = InsightFields(
        methods=[
            InsightClaim(
                claim="Invented method.",
                evidence=[{"chunk_id": "missing", "quote": "not present"}],
            )
        ]
    )
    assert validate_insights(bad, document).methods == []

    packet = PaperContextPacket(
        packet_id="P1",
        purpose="synthetic",
        excerpts=(
            EvidenceExcerpt(
                evidence_id="P1-E001",
                chunk_id="chunk-method",
                section_id="method",
                heading="Related Work",
                quote="Their work proposed an external taxonomy.",
            ),
            EvidenceExcerpt(
                evidence_id="P1-E002",
                chunk_id="chunk-evaluation",
                section_id="evaluation",
                heading="Methods",
                quote="We evaluated the framework through a ReAct case study.",
            ),
        ),
    )
    generated = _EvidenceExtraction(
        claims=[
            _GeneratedEvidenceClaim(
                category="contribution",
                claim="Their work proposed an external taxonomy.",
                evidence_ids=["P1-E001"],
            ),
            _GeneratedEvidenceClaim(
                category="future_work",
                claim="Future evaluation through a ReAct case study.",
                evidence_ids=["P1-E002"],
            ),
        ]
    )
    diagnostics = PaperUnderstandingDiagnostics()
    provider = StoryProvider("gemini", "gemini-3.5-flash")
    ledger = _resolve_generated_evidence([(packet, generated)], document, provider, diagnostics)
    assert ledger.claims == []
    assert {item["reason"] for item in diagnostics.rejected_evidence} >= {
        "prior_work_attribution",
        "future_work_guard",
    }


def test_oversized_evidence_reference_rejects_claim_without_crashing() -> None:
    document = research_paper()
    packet = PaperContextPacket(
        packet_id="P1",
        purpose="synthetic",
        excerpts=(
            EvidenceExcerpt(
                evidence_id="P1-E001",
                chunk_id="chunk-method",
                section_id="method",
                heading="Approach",
                quote="x" * 501,
            ),
        ),
    )
    generated = _EvidenceExtraction(
        claims=[
            _GeneratedEvidenceClaim(
                category="approach_method",
                claim="The paper uses a systems framework.",
                evidence_ids=["P1-E001"],
            )
        ]
    )
    diagnostics = PaperUnderstandingDiagnostics()
    provider = StoryProvider("gemini", "gemini-3.5-flash")

    ledger = _resolve_generated_evidence([(packet, generated)], document, provider, diagnostics)

    assert ledger.claims == []
    assert diagnostics.rejected_evidence == [
        {
            "category": "approach_method",
            "claim": "The paper uses a systems framework.",
            "reason": "evidence_quote_too_long",
        }
    ]


def test_cache_separates_provider_model_and_pipeline_version(tmp_path) -> None:
    cache = InsightCache(tmp_path)
    fingerprint = "f" * 64
    ollama = cache.key(fingerprint, "qwen3:1.7b", PIPELINE_VERSION, "ollama")
    gemini = cache.key(fingerprint, "qwen3:1.7b", PIPELINE_VERSION, "gemini")
    evl_gemma = cache.key(fingerprint, "gemma4", PIPELINE_VERSION, "evl_gemma")
    other_model = cache.key(fingerprint, "gemini-other", PIPELINE_VERSION, "gemini")
    other_version = cache.key(fingerprint, "qwen3:1.7b", "next-version", "ollama")
    assert len({ollama, gemini, evl_gemma, other_model, other_version}) == 5


def test_synthesis_cannot_create_evidence_and_missing_categories_are_allowed() -> None:
    document = research_paper()
    reference = EvidenceReference(
        chunk_id="chunk-method",
        quote="We developed a systems framework with five functional subsystems.",
    )
    ledger = ValidatedEvidenceLedger(
        claims=[
            EvidenceClaim(
                claim_id="EC001",
                category="approach_method",
                claim="The authors developed a systems framework.",
                evidence_refs=[reference],
                section_id="method",
                provider="gemini",
            )
        ]
    )
    generated = _ResearchSynthesis(
        paper_overview=[
            _GeneratedOverviewClaim(claim="An invented overview.", evidence_claim_ids=["EC999"])
        ]
    )
    diagnostics = PaperUnderstandingDiagnostics()

    result = _resolve_synthesis(generated, ledger, document, diagnostics)

    assert result.paper_overview == []
    assert result.methods[0].evidence == [reference]
    assert result.future_work == []
    assert diagnostics.rejected_evidence[0]["reason"] == "unknown_validated_claim_id"


def test_long_grounded_overview_is_supported_without_widening_evidence_quotes() -> None:
    document = research_paper()
    references = [
        EvidenceReference(
            chunk_id="abstract-1",
            quote="Current tools cannot organize complex research evidence. We address this gap.",
        ),
        EvidenceReference(
            chunk_id="chunk-method",
            quote="We developed a systems framework with five functional subsystems.",
        ),
        EvidenceReference(
            chunk_id="chunk-evaluation",
            quote="We evaluated the framework through a ReAct case study.",
        ),
        EvidenceReference(
            chunk_id="chunk-result",
            quote="Our evaluation found that the patterns explained ReAct coordination behavior.",
        ),
    ]
    categories = ["problem", "approach_method", "evaluation", "finding"]
    ledger = ValidatedEvidenceLedger(
        claims=[
            EvidenceClaim(
                claim_id=f"EC{index:03d}",
                category=category,
                claim=reference.quote,
                evidence_refs=[reference],
                provider="evl_gemma",
            )
            for index, (category, reference) in enumerate(
                zip(categories, references, strict=True), start=1
            )
        ]
    )
    overview = (
        "Current tools cannot organize complex research evidence, so the paper develops a systems "
        "framework with five functional subsystems. It evaluates the framework through a ReAct "
        "case study and finds that the patterns explain ReAct coordination behavior. "
        + "The grounded research story remains specific and evidence-linked. "
        * 7
    )
    assert 500 < len(overview) <= PAPER_OVERVIEW_MAX_LENGTH
    generated = _ResearchSynthesis(
        paper_overview=[
            _GeneratedOverviewClaim(
                claim=overview,
                evidence_claim_ids=[claim.claim_id for claim in ledger.claims],
            )
        ]
    )

    result = _resolve_synthesis(generated, ledger, document, PaperUnderstandingDiagnostics())

    assert result.paper_overview[0].claim == overview.strip()
    assert {ref.chunk_id for ref in result.paper_overview[0].evidence} == {
        ref.chunk_id for ref in references
    }
    assert all(len(ref.quote) <= EVIDENCE_QUOTE_MAX_LENGTH for ref in references)


def test_explicit_significance_alone_can_support_why_it_matters() -> None:
    document = research_paper()
    quote = (
        "This framework provides a shared language for researchers to understand agent "
        "coordination."
    )
    document.sections.append(
        DocumentSection(
            id="significance",
            heading="Discussion",
            level=1,
            text=quote,
            chunks=[
                DocumentChunk(
                    id="chunk-significance",
                    section_id="significance",
                    text=quote,
                    start_char=0,
                    end_char=len(quote),
                )
            ],
        )
    )
    ledger = ValidatedEvidenceLedger(
        claims=[
            EvidenceClaim(
                claim_id="EC001",
                category="significance",
                claim=quote,
                evidence_refs=[EvidenceReference(chunk_id="chunk-significance", quote=quote)],
                provider="evl_gemma",
            )
        ]
    )
    generated = _ResearchSynthesis(
        why_it_matters=[_GeneratedSynthesisClaim(claim=quote, evidence_claim_ids=["EC001"])]
    )

    result = _resolve_synthesis(generated, ledger, document, PaperUnderstandingDiagnostics())

    assert len(result.why_it_matters) == 1


def test_overview_with_unsupported_content_is_rejected() -> None:
    document = research_paper()
    reference = EvidenceReference(
        chunk_id="chunk-method",
        quote="We developed a systems framework with five functional subsystems.",
    )
    ledger = ValidatedEvidenceLedger(
        claims=[
            EvidenceClaim(
                claim_id="EC001",
                category="approach_method",
                claim="The authors developed a systems framework.",
                evidence_refs=[reference],
                provider="evl_gemma",
            ),
            EvidenceClaim(
                claim_id="EC002",
                category="evaluation",
                claim="The framework was evaluated through a ReAct case study.",
                evidence_refs=[
                    EvidenceReference(
                        chunk_id="chunk-evaluation",
                        quote="We evaluated the framework through a ReAct case study.",
                    )
                ],
                provider="evl_gemma",
            ),
        ]
    )
    generated = _ResearchSynthesis(
        paper_overview=[
            _GeneratedOverviewClaim(
                claim=(
                    "The clinical trial cured cancer, eliminated mortality, and guaranteed "
                    "universal deployment success."
                ),
                evidence_claim_ids=["EC001", "EC002"],
            )
        ]
    )

    result = _resolve_synthesis(generated, ledger, document, PaperUnderstandingDiagnostics())

    assert result.paper_overview == []


def test_unsupported_why_it_matters_hype_is_rejected() -> None:
    document = research_paper()
    reference = EvidenceReference(
        chunk_id="chunk-method",
        quote="We developed a systems framework with five functional subsystems.",
    )
    ledger = ValidatedEvidenceLedger(
        claims=[
            EvidenceClaim(
                claim_id="EC001",
                category="significance",
                claim="The framework organizes five functional subsystems.",
                evidence_refs=[reference],
                provider="evl_gemma",
            )
        ]
    )
    generated = _ResearchSynthesis(
        why_it_matters=[
            _GeneratedSynthesisClaim(
                claim="This will revolutionize medicine and guarantee better patient outcomes.",
                evidence_claim_ids=["EC001"],
            )
        ]
    )

    result = _resolve_synthesis(generated, ledger, document, PaperUnderstandingDiagnostics())

    assert len(result.why_it_matters) == 1
    assert result.why_it_matters[0].claim == "The framework organizes five functional subsystems."
    assert "revolutionize" not in result.why_it_matters[0].claim


def test_internal_ledger_markers_are_removed_without_losing_evidence() -> None:
    document = research_paper()
    reference = EvidenceReference(
        chunk_id="chunk-method",
        quote="We developed a systems framework with five functional subsystems.",
    )
    ledger = ValidatedEvidenceLedger(
        claims=[
            EvidenceClaim(
                claim_id="EC001",
                category="approach_method",
                claim="The paper develops a systems framework.",
                evidence_refs=[reference],
                provider="evl_gemma",
            )
        ]
    )
    generated = _ResearchSynthesis(
        methods=[
            _GeneratedSynthesisClaim(
                claim="The paper develops a systems framework [EC001].",
                evidence_claim_ids=["EC001"],
            )
        ]
    )

    result = _resolve_synthesis(generated, ledger, document, PaperUnderstandingDiagnostics())

    assert result.methods[0].claim == "The paper develops a systems framework."
    assert result.methods[0].evidence == [reference]


def _api_app(
    tmp_path,
    gemini_configured: bool = True,
    evl_gemma_configured: bool = True,
) -> FastAPI:
    ollama = StoryProvider("ollama", "qwen3:1.7b")
    gemini = StoryProvider("gemini", "gemini-3.5-flash", configured=gemini_configured)
    evl_gemma = StoryProvider("evl_gemma", "gemma4", configured=evl_gemma_configured)
    registry = LLMProviderRegistry([ollama, gemini, evl_gemma])
    app = FastAPI()
    app.state.llm_registry = registry
    app.state.insight_service = PaperUnderstandingService(registry, InsightCache(tmp_path))
    app.include_router(router)
    return app


def test_api_old_request_explicit_providers_status_and_unavailable(tmp_path) -> None:
    document = research_paper().model_dump(mode="json")
    with TestClient(_api_app(tmp_path / "configured")) as client:
        old = client.post("/api/papers/insights", json={"document": document})
        gemini = client.post(
            "/api/papers/insights", json={"document": document, "provider": "gemini"}
        )
        evl_gemma = client.post(
            "/api/papers/insights", json={"document": document, "provider": "evl_gemma"}
        )
        status = client.get("/api/llm/providers")
    assert old.status_code == 200 and old.json()["model"] == "qwen3:1.7b"
    assert gemini.status_code == 200 and gemini.json()["model"] == "gemini-3.5-flash"
    assert evl_gemma.status_code == 200 and evl_gemma.json()["model"] == "gemma4"
    assert {item["provider_id"] for item in status.json()["providers"]} == {
        "ollama",
        "gemini",
        "evl_gemma",
    }
    evl_status = next(
        item for item in status.json()["providers"] if item["provider_id"] == "evl_gemma"
    )
    assert evl_status == {
        "provider_id": "evl_gemma",
        "model": "gemma4",
        "configured": True,
        "cloud": True,
    }

    with TestClient(_api_app(tmp_path / "unavailable", gemini_configured=False)) as client:
        unavailable = client.post(
            "/api/papers/insights", json={"document": document, "provider": "gemini"}
        )
    assert unavailable.status_code == 503
    assert "not configured" in unavailable.json()["detail"]

    with TestClient(_api_app(tmp_path / "evl-unavailable", evl_gemma_configured=False)) as client:
        unavailable = client.post(
            "/api/papers/insights", json={"document": document, "provider": "evl_gemma"}
        )
    assert unavailable.status_code == 503
    assert "not configured" in unavailable.json()["detail"]
