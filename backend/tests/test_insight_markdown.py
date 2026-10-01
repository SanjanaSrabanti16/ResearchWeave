from __future__ import annotations

# ruff: noqa: E501 -- representative provider prose is intentionally not source-wrapped.
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from app.llm import (
    LLMOutputBudgetExceeded,
    LLMProvider,
    LLMProviderCapabilities,
    LLMProviderRegistry,
    LLMRequestContext,
)
from app.llm.evl_gemma import EVLGemmaProvider
from app.llm.gemini import GeminiProvider
from app.llm.ollama import OllamaProvider
from app.models.document import (
    DocumentChunk,
    DocumentSection,
    ParsedPaper,
    SourcePDF,
)
from app.services.insight_cache import InsightCache
from app.services.insight_context import (
    full_research_context,
    hierarchical_research_contexts,
    select_research_context_strategy,
)
from app.services.insight_markdown import (
    SHARED_RESEARCH_SYSTEM_PROMPT,
    parse_research_markdown,
    suspiciously_incomplete,
    validate_markdown_insights,
)
from app.services.insight_service import InsightOutputError
from app.services.paper_understanding_service import (
    PIPELINE_VERSION,
    PaperUnderstandingDiagnostics,
    PaperUnderstandingService,
)


def _chunk(chunk_id: str, section_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        section_id=section_id,
        text=text,
        start_char=0,
        end_char=len(text),
    )


def _document() -> ParsedPaper:
    abstract = _chunk(
        "abstract-1",
        "abstract",
        "Cross-center clinical data are heterogeneous and difficult to compare.",
    )
    intro = _chunk(
        "intro-1",
        "intro",
        "Existing tools do not connect demographic, socioeconomic, and clinical factors.",
    )
    methods = _chunk(
        "methods-1",
        "methods",
        "We harmonize two datasets and build a coordinated visual analytics system.",
    )
    contribution = _chunk(
        "contrib-1",
        "intro",
        "Our contributions include data harmonization, visual design, and expert evaluation.",
    )
    evaluation = _chunk(
        "eval-1",
        "evaluation",
        "Two clinician case studies and expert interviews evaluated the system.",
    )
    findings = _chunk(
        "finding-1",
        "results",
        "Experts identified cohort differences and an incomplete follow-up artifact.",
    )
    discussion = _chunk(
        "discussion-1",
        "discussion",
        "The system supports population-level and patient-level hypothesis exploration.",
    )
    limitation = _chunk(
        "limit-1",
        "discussion",
        "Limited expert availability prevented a participatory design process.",
    )
    future = _chunk(
        "future-1",
        "conclusion",
        "Future deployments will investigate additional standardized cohorts.",
    )
    return ParsedPaper(
        paper_id="paper-1",
        title="A Tale of Two Centers",
        authors=["A. Researcher"],
        abstract=abstract.text,
        abstract_chunks=[abstract],
        sections=[
            DocumentSection(
                id="intro",
                heading="Introduction and Contributions",
                level=1,
                text=f"{intro.text}\n{contribution.text}",
                chunks=[intro, contribution],
            ),
            DocumentSection(
                id="methods",
                heading="Methods and Design",
                level=1,
                text=methods.text,
                chunks=[methods],
            ),
            DocumentSection(
                id="evaluation",
                heading="Evaluation",
                level=1,
                text=evaluation.text,
                chunks=[evaluation],
            ),
            DocumentSection(
                id="results",
                heading="Results and Case Studies",
                level=1,
                text=findings.text,
                chunks=[findings],
            ),
            DocumentSection(
                id="discussion",
                heading="Discussion and Limitations",
                level=1,
                text=f"{discussion.text}\n{limitation.text}",
                chunks=[discussion, limitation],
            ),
            DocumentSection(
                id="conclusion",
                heading="Conclusion and Future Work",
                level=1,
                text=future.text,
                chunks=[future],
            ),
            DocumentSection(
                id="references",
                heading="References",
                level=1,
                text="Prior work that must not be treated as paper content.",
                chunks=[_chunk("ref-1", "references", "Prior work bibliography entry.")],
            ),
        ],
        parser="grobid",
        parser_version="0.9.1",
        source_pdf=SourcePDF(
            acquisition_method="upload",
            acquisition_provenance="upload",
            sha256="a" * 64,
            size_bytes=1234,
        ),
    )


MARKDOWN = """## Paper Overview
The paper connects heterogeneous cross-center data to a coordinated visual analytics workflow and evaluates it with clinicians. [chunk:abstract-1] [chunk:eval-1]

## Research Problem
- Existing tools do not connect the necessary factors. [chunk:intro-1]

## Methods / Approach
- The authors harmonize two datasets and build coordinated views. [chunk:methods-1]

## Key Contributions
- The work contributes harmonization, visual design, and expert evaluation. [chunk:contrib-1]

## Evaluation
Two clinician case studies and interviews evaluate the system. [chunk:eval-1]

## Main Findings
Experts found cohort differences and a follow-up artifact. [chunk:finding-1]

## Why It Matters
The workflow supports population- and patient-level hypothesis exploration. [chunk:discussion-1]

## Target Audience
The paper does not explicitly specify a target audience.

## Limitations / Open Challenges
The study lacked a participatory design process because experts had limited availability. [chunk:limit-1]

## Future Work
The authors plan deployment on additional standardized cohorts. [chunk:future-1]
"""


class MarkdownProvider(LLMProvider):
    def __init__(
        self,
        provider_id: str,
        capabilities: LLMProviderCapabilities,
        *,
        fail_first: bool = False,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = "test-model"
        self.capabilities = capabilities
        self.calls: list[dict[str, Any]] = []
        self.fail_first = fail_first

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
        context: LLMRequestContext | None = None,
    ) -> str:
        raise AssertionError("The production Markdown path must not request JSON")

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        context: LLMRequestContext | None = None,
    ) -> str:
        self.calls.append(
            {
                "system": system_prompt,
                "prompt": user_prompt,
                "tokens": max_output_tokens,
                "context": context,
            }
        )
        if self.fail_first and len(self.calls) == 1:
            raise LLMOutputBudgetExceeded("test truncation")
        if "PAPER SEGMENT" in user_prompt:
            return "Method and evaluation notes [chunk:methods-1] [chunk:eval-1]"
        return MARKDOWN


FULL_CAPABILITIES = LLMProviderCapabilities(
    structured_json=True,
    context_strategy="full_document",
    cloud=True,
    max_context_chars=180000,
    max_extraction_packets=1,
)
LOCAL_CAPABILITIES = LLMProviderCapabilities(
    structured_json=True,
    context_strategy="hierarchical_sections",
    cloud=False,
    max_context_chars=26000,
    max_extraction_packets=12,
)


def test_markdown_parser_accepts_heading_aliases_paragraphs_and_bullets() -> None:
    raw = MARKDOWN.replace("## Methods / Approach", "**Methodology**:").replace(
        "## Main Findings", "Results:"
    )
    insights = parse_research_markdown(raw, _document())
    assert "harmonize" in insights.methods[0].claim
    assert "cohort differences" in insights.main_findings[0].claim
    assert insights.methods[0].evidence[0].chunk_id == "methods-1"


def test_markdown_parser_recovers_unknown_subheadings_and_strips_harmless_html() -> None:
    raw = MARKDOWN.replace(
        "- The authors harmonize two datasets",
        "### Architecture detail\n<p>The authors harmonize two datasets</p>",
    )
    insights = parse_research_markdown(raw, _document())
    assert "Architecture detail" not in insights.methods[0].claim
    assert "harmonize" in insights.methods[0].claim
    assert "<p>" not in insights.methods[0].claim


def test_grouped_chunk_citations_are_removed_and_all_evidence_is_retained() -> None:
    raw = MARKDOWN.replace(
        "[chunk:abstract-1] [chunk:eval-1]",
        "[chunk:abstract-1, chunk:eval-1]",
    )
    insights = parse_research_markdown(raw, _document())
    overview = insights.paper_overview[0]
    assert "[chunk:" not in overview.claim
    assert [reference.chunk_id for reference in overview.evidence] == [
        "abstract-1",
        "eval-1",
    ]


def test_explicit_contribution_list_anchor_preserves_distinct_items() -> None:
    document = _document()
    raw = MARKDOWN.replace(
        "- The work contributes harmonization, visual design, and expert evaluation. "
        "[chunk:contrib-1]",
        "The authors explicitly list two contributions: [chunk:contrib-1]\n"
        "- Multi-site data harmonization.\n"
        "- Expert evaluation of the visual design.",
    )
    parsed = parse_research_markdown(raw, document)
    validated = validate_markdown_insights(parsed, document)
    assert [item.claim for item in validated.key_contributions] == [
        "Multi-site data harmonization.",
        "Expert evaluation of the visual design.",
    ]
    assert all(item.evidence for item in validated.key_contributions)


def test_markdown_parser_rejects_unsafe_markup_and_ellipsis_overview() -> None:
    with pytest.raises(InsightOutputError, match="unsafe markup"):
        parse_research_markdown(MARKDOWN + "<script>alert(1)</script>", _document())
    with pytest.raises(InsightOutputError, match="ellipsis"):
        parse_research_markdown(
            MARKDOWN.replace(
                "and evaluates it with clinicians. [chunk:abstract-1] [chunk:eval-1]",
                "and evaluates it...",
            ),
            _document(),
        )


def test_unknown_citation_is_not_exposed_as_broken_evidence() -> None:
    insights = parse_research_markdown(
        MARKDOWN.replace("[chunk:intro-1]", "[chunk:not-real]"), _document()
    )
    assert insights.research_problem[0].evidence == []
    assert validate_markdown_insights(insights, _document()) == insights


def test_complete_context_excludes_bibliography_and_keeps_all_useful_chunks() -> None:
    context = full_research_context(_document())
    assert "[chunk:abstract-1]" in context.text
    assert "[chunk:future-1]" in context.text
    assert "Prior work bibliography" not in context.text
    assert "ref-1" not in context.chunk_ids


def test_context_strategy_uses_actual_safe_fit() -> None:
    document = _document()
    assert select_research_context_strategy(document, FULL_CAPABILITIES) == "full_document"
    assert select_research_context_strategy(document, LOCAL_CAPABILITIES) == "hierarchical_sections"
    high_context_local = LLMProviderCapabilities(
        structured_json=True,
        context_strategy="hierarchical_sections",
        cloud=False,
        max_context_chars=200000,
        max_extraction_packets=12,
    )
    assert select_research_context_strategy(document, high_context_local) == "full_document"


def test_hierarchical_context_preserves_semantic_section_order_and_overlap() -> None:
    parts = hierarchical_research_contexts(_document(), max_part_chars=350)
    purposes = [part.purpose for part in parts]
    assert purposes.index("introduction and research problem") < purposes.index(
        "methods and approach"
    )
    assert any("evaluation" in purpose for purpose in purposes)
    assert all(part.text for part in parts)
    assert {"abstract-1", "methods-1", "future-1"}.issubset(
        {chunk_id for part in parts for chunk_id in part.chunk_ids}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["evl_gemma", "openai_compatible", "gemini"])
async def test_full_document_providers_share_one_markdown_contract(
    tmp_path: Any, provider_id: str
) -> None:
    provider = MarkdownProvider(provider_id, FULL_CAPABILITIES)
    service = PaperUnderstandingService(
        LLMProviderRegistry([provider], default_provider=provider_id),
        InsightCache(tmp_path),
    )
    diagnostics = PaperUnderstandingDiagnostics()
    result = await service.extract(_document(), diagnostics=diagnostics)
    assert result.extraction_version == PIPELINE_VERSION
    assert len(provider.calls) == 1
    assert "[chunk:future-1]" in provider.calls[0]["prompt"]
    assert "Prior work bibliography" not in provider.calls[0]["prompt"]
    assert diagnostics.context_strategy == "full_document"
    assert result.insights.methods and result.insights.key_contributions


@pytest.mark.asyncio
async def test_ollama_strategy_uses_section_notes_then_same_final_schema(tmp_path: Any) -> None:
    provider = MarkdownProvider("ollama", LOCAL_CAPABILITIES)
    service = PaperUnderstandingService(
        LLMProviderRegistry([provider], default_provider="ollama"),
        InsightCache(tmp_path),
        batch_chars=5000,
    )
    diagnostics = PaperUnderstandingDiagnostics()
    result = await service.extract(_document(), diagnostics=diagnostics)
    assert diagnostics.context_strategy == "hierarchical_sections"
    assert len(provider.calls) > 1
    assert sum("PAPER SEGMENT" in call["prompt"] for call in provider.calls) >= 1
    assert (
        result.insights.model_dump() == parse_research_markdown(MARKDOWN, _document()).model_dump()
    )


@pytest.mark.asyncio
async def test_output_budget_gets_only_one_bounded_recovery(tmp_path: Any) -> None:
    provider = MarkdownProvider("evl_gemma", FULL_CAPABILITIES, fail_first=True)
    service = PaperUnderstandingService(
        LLMProviderRegistry([provider], default_provider="evl_gemma"),
        InsightCache(tmp_path),
    )
    diagnostics = PaperUnderstandingDiagnostics()
    result = await service.extract(_document(), diagnostics=diagnostics)
    assert result.insights.paper_overview
    assert len(provider.calls) == 2
    assert diagnostics.completeness_recovery_used is True


def test_gold_style_regression_requires_methods_contributions_and_findings() -> None:
    document = _document()
    complete = parse_research_markdown(MARKDOWN, document)
    assert suspiciously_incomplete(complete, document) is False
    shallow = complete.model_copy(
        update={"methods": [], "key_contributions": [], "main_findings": []}
    )
    assert suspiciously_incomplete(shallow, document) is True
    assert "knowledgeable researcher" in SHARED_RESEARCH_SYSTEM_PROMPT
    assert "Do not optimize for one-line summaries" in SHARED_RESEARCH_SYSTEM_PROMPT


class _OpenAITextCompletions:
    def __init__(self, finish_reason: str = "stop") -> None:
        self.finish_reason = finish_reason
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=MARKDOWN),
                    finish_reason=self.finish_reason,
                )
            ]
        )


@pytest.mark.asyncio
async def test_evl_openai_compatible_text_mode_does_not_request_json() -> None:
    completions = _OpenAITextCompletions()
    provider = EVLGemmaProvider(
        "configured",
        "https://example.test",
        "gemma4",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    assert (
        await provider.generate_text(
            system_prompt="system",
            user_prompt="user",
            max_output_tokens=7000,
        )
        == MARKDOWN
    )
    assert "response_format" not in completions.calls[0]


class _GeminiTextModels:
    def __init__(self, finish_reason: str = "STOP") -> None:
        self.finish_reason = finish_reason
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            text=MARKDOWN,
            candidates=[SimpleNamespace(finish_reason=self.finish_reason)],
        )


@pytest.mark.asyncio
async def test_gemini_text_mode_uses_markdown_without_json_schema() -> None:
    models = _GeminiTextModels()
    provider = GeminiProvider(
        "configured",
        "gemini-test",
        client=SimpleNamespace(aio=SimpleNamespace(models=models)),
    )
    assert (
        await provider.generate_text(
            system_prompt="system",
            user_prompt="user",
            max_output_tokens=7000,
        )
        == MARKDOWN
    )
    config = models.calls[0]["config"]
    assert getattr(config, "response_json_schema", None) is None
    assert getattr(config, "response_mime_type", None) is None


@pytest.mark.asyncio
async def test_ollama_text_mode_omits_structured_format() -> None:
    requests: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"content": MARKDOWN}, "done_reason": "stop"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaProvider(client, "http://ollama.test", "qwen3:1.7b")
        assert (
            await provider.generate_text(
                system_prompt="system",
                user_prompt="user",
                max_output_tokens=7000,
            )
            == MARKDOWN
        )
    assert "format" not in requests[0]
