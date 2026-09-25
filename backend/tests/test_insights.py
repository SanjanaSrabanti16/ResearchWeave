import json
import re
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.main import create_app
from app.models.document import DocumentChunk, DocumentSection, ParsedPaper, SourcePDF
from app.models.insights import INSIGHT_FIELDS, InsightClaim, InsightFields, InsightsResponse
from app.services.insight_cache import InsightCache
from app.services.insight_diagnostics import InsightDiagnostics
from app.services.insight_selector import (
    evidence_catalog,
    focus_evidence,
    reserve_boundary_passages,
    select_evidence,
    supports_finding,
    supports_future_work,
)
from app.services.insight_service import (
    EXTRACTION_VERSION,
    InsightInputError,
    InsightOutputError,
    InsightService,
    OllamaTimeoutError,
    OllamaUnavailableError,
    document_fingerprint,
    explicit_claim_rejection_reason,
    explicit_claim_supported,
    filter_synthesis_support,
    merge_insights,
    quote_in_chunk,
    restore_selected_chunk_prefixes,
    synthesis_claim_supported,
    validate_insights,
)


def paper() -> ParsedPaper:
    return ParsedPaper(
        paper_id="paper-1",
        title="Attention Is All You Need",
        abstract="A transformer architecture is proposed.",
        abstract_chunks=[
            DocumentChunk(
                id="abstract-1",
                section_id="abstract",
                text="A transformer architecture is proposed.",
                start_char=0,
                end_char=39,
            )
        ],
        sections=[
            DocumentSection(
                id="section-1",
                heading="Method",
                level=1,
                text="The model uses self-attention layers.",
                chunks=[
                    DocumentChunk(
                        id="method-1",
                        section_id="section-1",
                        text="The model uses self-attention layers.",
                        start_char=0,
                        end_char=37,
                    )
                ],
            )
        ],
        parser="grobid",
        parser_version="0.9.1-crf",
        source_pdf=SourcePDF(acquisition_method="arxiv", sha256="a" * 64, size_bytes=100),
    )


def empty_output() -> dict[str, list[dict[str, object]]]:
    return {field: [] for field in INSIGHT_FIELDS}


def method_output() -> dict[str, list[dict[str, object]]]:
    output = empty_output()
    output["methods"] = [
        {
            "claim": "The method uses self-attention layers.",
            "evidence": [{"chunk_id": "method-1", "quote": "the model uses self attention layers"}],
        }
    ]
    return output


def focused_response(
    request: httpx.Request, output: dict[str, list[dict[str, object]]]
) -> httpx.Response:
    payload = json.loads(request.content)
    fields = payload["format"]["properties"]
    prompt = payload["messages"][1]["content"]
    multiple_evidence = "and evidence_ids" in prompt
    catalog = {
        match.group(1): match.group(2)
        for match in re.finditer(r"^\[(E\d+)\] .*?: (.+)$", prompt, re.MULTILINE)
    }

    def evidence_id(item: dict[str, object]) -> str:
        quote = str(item["evidence"][0]["quote"])
        for identifier, excerpt in catalog.items():
            if quote_in_chunk(quote, excerpt):
                return identifier
        return "E999"

    return httpx.Response(
        200,
        json={
            "message": {
                "content": json.dumps(
                    {
                        field: [
                            (
                                {
                                    "claim": item["claim"],
                                    "evidence_ids": [evidence_id(item)],
                                }
                                if multiple_evidence
                                else {
                                    "claim": item["claim"],
                                    "evidence_id": evidence_id(item),
                                }
                            )
                            for item in output[field]
                        ]
                        for field in fields
                    }
                )
            }
        },
    )


@pytest.mark.asyncio
async def test_valid_structured_response_uses_schema_and_grounded_chunks(tmp_path) -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return focused_response(request, method_output())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = InsightService(
            client, "http://localhost:11434", "qwen3:4b", InsightCache(tmp_path)
        )
        response = await service.extract(paper())

    assert response.insights.methods[0].claim == "The method uses self-attention layers."
    assert response.insights.methods[0].evidence[0].chunk_id == "method-1"
    assert response.model == "qwen3:4b"
    assert response.extraction_version == EXTRACTION_VERSION
    assert response.cached is False
    assert requests[0]["model"] == "qwen3:4b"
    assert requests[0]["stream"] is False
    assert set(requests[0]["format"]["properties"]) == {
        "research_problem",
        "key_contributions",
        "methods",
    }
    assert set(requests[1]["format"]["properties"]) == {
        "main_findings",
        "limitations",
        "future_work",
    }
    assert set(requests[2]["format"]["properties"]) == {
        "why_it_matters",
        "target_audience",
    }
    assert requests[0]["think"] is False
    assert requests[0]["options"]["num_predict"] == 1400
    prompt = requests[0]["messages"][1]["content"]
    assert "Attention Is All You Need" in prompt
    assert "A transformer architecture is proposed" in prompt
    assert "Method" in prompt
    assert "[E01]" in prompt
    assert "Return only JSON matching the schema, no prose" in prompt
    system_prompt = requests[0]["messages"][0]["content"]
    assert "do not stop after the first valid claim" in system_prompt
    assert "not quotas" in system_prompt
    assert "limitations 0-8, future_work 0-8" in system_prompt
    properties = InsightFields.model_json_schema()["properties"]
    assert all("maxItems" not in properties[field] for field in INSIGHT_FIELDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not JSON", '{"methods": []}'])
async def test_malformed_or_schema_invalid_ollama_response_is_rejected(tmp_path, content) -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"message": {"content": content}})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        service = InsightService(
            client, "http://localhost:11434", "qwen3:4b", InsightCache(tmp_path)
        )
        with pytest.raises(InsightOutputError, match="invalid structured"):
            await service.extract(paper())


@pytest.mark.asyncio
async def test_nonexistent_id_bad_quote_and_unsupported_claim_are_removed(tmp_path) -> None:
    document = paper()
    finding = "Evaluation results showed that experts completed the comparison task."
    document.sections.append(
        DocumentSection(
            id="evaluation-results",
            heading="Evaluation Results",
            level=1,
            text=finding,
            chunks=[
                DocumentChunk(
                    id="finding-1",
                    section_id="evaluation-results",
                    text=finding,
                    start_char=0,
                    end_char=len(finding),
                )
            ],
        )
    )
    output = method_output()
    output["methods"].extend(
        [
            {
                "claim": "Another unsupported method.",
                "evidence": [
                    {"chunk_id": "method-1", "quote": "A result that is not in the paper."}
                ],
            },
        ]
    )
    output["limitations"] = [
        {
            "claim": "The paper has no limitations.",
            "evidence": [{"chunk_id": "missing", "quote": "none"}],
        }
    ]
    output["main_findings"] = [
        {
            "claim": "Experts completed the comparison task.",
            "evidence": [{"chunk_id": "finding-1", "quote": finding}],
        }
    ]
    transport = httpx.MockTransport(lambda request: focused_response(request, output))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await InsightService(
            client, "http://localhost:11434", "qwen3:4b", InsightCache(tmp_path)
        ).extract(document)

    assert len(result.insights.methods) == 1
    assert [ref.chunk_id for ref in result.insights.methods[0].evidence] == ["method-1"]
    assert result.insights.limitations == []
    assert quote_in_chunk("Self-attention layers", "The model uses self-attention layers.")
    assert not quote_in_chunk("attentionless", "The model uses self-attention layers.")


@pytest.mark.asyncio
async def test_cache_hit_skips_ollama_and_key_varies_by_model_and_version(tmp_path) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return focused_response(request, method_output())

    cache = InsightCache(tmp_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = InsightService(client, "http://localhost:11434", "qwen3:4b", cache)
        first = await service.extract(paper())
        second = await service.extract(paper())

    assert calls == 3
    assert not first.cached and second.cached
    assert second.insights == first.insights
    fingerprint = document_fingerprint(paper())
    assert cache.key(fingerprint, "qwen3:4b", EXTRACTION_VERSION) != cache.key(
        fingerprint, "other-model", EXTRACTION_VERSION
    )
    assert cache.key(fingerprint, "qwen3:4b", EXTRACTION_VERSION) != cache.key(
        fingerprint, "qwen3:4b", "m2b-v2"
    )


@pytest.mark.asyncio
async def test_cached_unsupported_evidence_is_not_served(tmp_path) -> None:
    cache = InsightCache(tmp_path)
    bad = method_output()
    bad["methods"][0]["evidence"] = [{"chunk_id": "missing", "quote": "invented quote"}]
    key = cache.key(document_fingerprint(paper()), "qwen3:4b", EXTRACTION_VERSION)
    cache.put(key, InsightFields.model_validate(bad))
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return focused_response(request, method_output())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await InsightService(
            client, "http://localhost:11434", "qwen3:4b", cache
        ).extract(paper())
    assert calls == 3
    assert response.cached is False
    assert response.insights.methods[0].evidence[0].chunk_id == "method-1"


@pytest.mark.asyncio
async def test_selected_evidence_uses_three_calls(tmp_path) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return focused_response(request, method_output())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = InsightService(
            client, "http://localhost:11434", "qwen3:4b", InsightCache(tmp_path)
        )
        result = await service.extract(paper())

    assert calls == 3
    assert len(result.insights.methods) == 1


@pytest.mark.asyncio
async def test_multiple_supported_claims_from_three_focused_calls_are_preserved(tmp_path) -> None:
    calls = 0
    document = paper()
    abstract_text = (
        "Recurrent models compute sequentially. Attention can improve parallel training."
    )
    method_text = "The model uses self-attention layers. It also uses positional encodings."
    document.abstract = abstract_text
    document.abstract_chunks[0].text = abstract_text
    document.abstract_chunks[0].end_char = len(abstract_text)
    document.sections[0].text = method_text
    document.sections[0].chunks[0].text = method_text
    document.sections[0].chunks[0].end_char = len(method_text)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        prompt = json.loads(request.content)["messages"][1]["content"]
        output = empty_output()
        fields = json.loads(request.content)["format"]["properties"]
        if "research_problem" in fields:
            assert "[E01]" in prompt and "Recurrent models compute sequentially" in prompt
        output["research_problem"] = [
            {
                "claim": "Recurrent models compute sequentially.",
                "evidence": [
                    {
                        "chunk_id": "abstract-1",
                        "quote": "Recurrent models compute sequentially.",
                    }
                ],
            },
            {
                "claim": "Attention can improve parallel training.",
                "evidence": [
                    {
                        "chunk_id": "abstract-1",
                        "quote": "Attention can improve parallel training.",
                    }
                ],
            },
        ]
        output["methods"] = [
            {
                "claim": "The model uses self-attention layers.",
                "evidence": [
                    {"chunk_id": "method-1", "quote": "The model uses self-attention layers."}
                ],
            },
            {
                "claim": "The model uses positional encodings.",
                "evidence": [
                    {"chunk_id": "method-1", "quote": "It also uses positional encodings."}
                ],
            },
        ]
        return focused_response(request, output)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await InsightService(
            client, "http://localhost:11434", "qwen3:4b", InsightCache(tmp_path)
        ).extract(document)

    assert calls == 3
    assert len(result.insights.research_problem) == 2
    assert len(result.insights.methods) == 2
    for field in ("research_problem", "methods"):
        assert all(item.evidence for item in getattr(result.insights, field))


def test_selector_prefers_method_and_result_sections_deterministically() -> None:
    document = paper()
    document.sections.append(
        DocumentSection(
            id="section-2",
            heading="Results and Evaluation",
            level=1,
            text="Results show a 2.0 BLEU improvement in translation.",
            chunks=[
                DocumentChunk(
                    id="result-1",
                    section_id="section-2",
                    text="Results show a 2.0 BLEU improvement in translation.",
                    start_char=0,
                    end_char=53,
                )
            ],
        )
    )
    first = select_evidence(document)
    second = select_evidence(document)
    assert first == second
    assert "method-1" in first.pools["methods"]
    assert "result-1" in first.pools["findings"]
    assert len({chunk_id for _, chunk_id, _ in first.chunks}) == len(first.chunks)


def test_explicit_numbered_contributions_are_not_partially_lost() -> None:
    document = paper()
    contribution_text = (
        "The main contributions of this work are: 1) a problem characterization; "
        "2) a harmonized dataset; 3) a visual analysis design; "
        "4) an implemented analysis system; and 5) an expert evaluation."
    )
    document.sections.append(
        DocumentSection(
            id="contributions",
            heading="Introduction",
            level=1,
            text=contribution_text,
            chunks=[
                DocumentChunk(
                    id="contribution-list",
                    section_id="contributions",
                    text=contribution_text,
                    start_char=0,
                    end_char=len(contribution_text),
                )
            ],
        )
    )

    selection = select_evidence(document)
    core = focus_evidence(
        selection,
        (("explicit_contributions", None),),
        (),
        max_chunks=1,
        max_chars=1,
    )
    quotes = [quote for _, quote, _ in evidence_catalog(core).values()]

    assert "contribution-list" in selection.pools["contributions"]
    assert "contribution-list" in {chunk_id for _heading, chunk_id, _text in core.chunks}
    for phrase in (
        "1) a problem characterization",
        "2) a harmonized dataset",
        "3) a visual analysis design",
        "4) an implemented analysis system",
        "5) an expert evaluation",
    ):
        assert any(phrase in quote for quote in quotes)


def test_problem_pack_retains_explicit_research_gap() -> None:
    document = paper()
    gap = (
        "It remains unknown which multivariate factors drive cross-center outcome disparities, "
        "and existing approaches cannot reconcile the heterogeneous data."
    )
    document.sections.append(
        DocumentSection(
            id="introduction-gap",
            heading="Introduction",
            level=1,
            text=gap,
            chunks=[
                DocumentChunk(
                    id="problem-gap",
                    section_id="introduction-gap",
                    text=gap,
                    start_char=0,
                    end_char=len(gap),
                )
            ],
        )
    )

    selection = select_evidence(document)
    core = focus_evidence(
        selection,
        (("problem", 1),),
        ("overview",),
        max_chunks=1,
        max_chars=1,
    )

    assert "problem-gap" in selection.pools["problem"]
    assert "problem-gap" in selection.pools["problem_gaps"]
    assert "problem-gap" in {chunk_id for _heading, chunk_id, _text in core.chunks}


def test_strong_problem_gap_survives_per_chunk_excerpt_limit() -> None:
    document = paper()
    motivation = "Cancer outcomes vary substantially across several patient populations."
    background = "Researchers collect clinical and demographic measurements at every center."
    filler = " ".join(f"Background factor {number} is commonly reported." for number in range(10))
    gap = (
        "It is not known how these factors influence outcomes, and cross-center data has never "
        "before been analyzed."
    )
    text = f"{motivation} {background} {filler} {gap}"
    document.sections.append(
        DocumentSection(
            id="long-introduction",
            heading="Introduction",
            level=1,
            text=text,
            chunks=[
                DocumentChunk(
                    id="long-problem-gap",
                    section_id="long-introduction",
                    text=text,
                    start_char=0,
                    end_char=len(text),
                )
            ],
        )
    )

    selection = select_evidence(document)
    core = focus_evidence(
        selection,
        (("problem_gaps", 1),),
        (),
        max_chunks=1,
        max_chars=1,
    )
    quotes = [quote for _chunk_id, quote, _heading in evidence_catalog(core).values()]

    assert any("not known" in quote and "never before" in quote for quote in quotes)
    assert any(motivation in quote for quote in quotes)


def test_multiple_evaluation_subsections_are_guaranteed_to_findings_call() -> None:
    document = paper()
    evaluation_sections = (
        (
            "case-study-a",
            "Case Study: Cohort Comparison",
            "We observed meaningful differences between the two patient cohorts.",
        ),
        (
            "expert-feedback",
            "Expert Feedback",
            "Experts found that the coordinated views supported rapid hypothesis testing.",
        ),
    )
    for chunk_id, heading, text in evaluation_sections:
        document.sections.append(
            DocumentSection(
                id=f"section-{chunk_id}",
                heading=heading,
                level=2,
                text=text,
                chunks=[
                    DocumentChunk(
                        id=chunk_id,
                        section_id=f"section-{chunk_id}",
                        text=text,
                        start_char=0,
                        end_char=len(text),
                    )
                ],
            )
        )

    selection = select_evidence(document)
    findings = focus_evidence(
        selection,
        (("evaluation_subsections", None),),
        ("findings",),
        max_chunks=1,
        max_chars=1,
    )

    expected = {chunk_id for chunk_id, _heading, _text in evaluation_sections}
    assert expected <= set(selection.pools["evaluation_subsections"])
    assert expected <= {chunk_id for _, chunk_id, _ in findings.chunks}


def test_long_case_study_preserves_early_middle_and_late_finding_phases() -> None:
    document = paper()
    finding_texts = (
        "The team observed an early demographic difference between cohorts.",
        "The analysis revealed a difference in insurance coverage.",
        "Investigators noticed a distinct smoking pattern.",
        "It became apparent that follow-up periods differed.",
        "The comparison confirmed a treatment-stage difference.",
        "The patient analysis determined that care was similar after matching.",
        "Expert feedback showed rapid hypothesis testing in the late phase.",
    )
    document.sections.append(
        DocumentSection(
            id="long-case-study",
            heading="Case Study Investigation",
            level=1,
            text=" ".join(finding_texts),
            chunks=[
                DocumentChunk(
                    id=f"phase-{index}",
                    section_id="long-case-study",
                    text=text,
                    start_char=0,
                    end_char=len(text),
                )
                for index, text in enumerate(finding_texts)
            ],
        )
    )

    selection = select_evidence(document)
    findings = focus_evidence(
        selection,
        (("evaluation_subsections", None),),
        (),
        max_chunks=1,
        max_chars=1,
    )
    selected = {chunk_id for _heading, chunk_id, _text in findings.chunks}

    assert len({f"phase-{index}" for index in range(len(finding_texts))} & selected) >= 3
    assert {"phase-0", "phase-3", "phase-6"} <= selected


def test_participatory_design_limitation_is_not_future_work() -> None:
    document = paper()
    limitation_text = (
        "The approach did not use participatory design because domain experts had limited "
        "availability, constraining opportunities for complex encoding design."
    )
    document.sections.append(
        DocumentSection(
            id="discussion",
            heading="Discussion and Conclusion",
            level=1,
            text=limitation_text,
            chunks=[
                DocumentChunk(
                    id="participatory-limitation",
                    section_id="discussion",
                    text=limitation_text,
                    start_char=0,
                    end_char=len(limitation_text),
                )
            ],
        )
    )

    selection = select_evidence(document)

    assert "participatory-limitation" in selection.pools["limitations"]
    assert "participatory-limitation" not in selection.pools["future_work"]
    assert not supports_future_work(limitation_text)


def test_discussion_tail_reserves_limitations_scalability_and_deployment() -> None:
    document = paper()
    chunks = (
        (
            "discussion-overview",
            "The discussion summarizes the evaluation and its principal observations.",
        ),
        (
            "discussion-scalability",
            "Scalability is limited for underrepresented groups, and overplotting becomes an "
            "issue for large datasets.",
        ),
        (
            "discussion-scope",
            "Simultaneous comparison of several cohorts may require substantial modifications.",
        ),
        (
            "discussion-deployment",
            "For further deployment, the approach can extend to additional datasets, other "
            "cohorts, and broader domains.",
        ),
    )
    document.sections.append(
        DocumentSection(
            id="discussion-tail",
            heading="Discussion and Conclusion",
            level=1,
            text=" ".join(text for _chunk_id, text in chunks),
            chunks=[
                DocumentChunk(
                    id=chunk_id,
                    section_id="discussion-tail",
                    text=text,
                    start_char=0,
                    end_char=len(text),
                )
                for chunk_id, text in chunks
            ],
        )
    )

    selection = select_evidence(document)
    boundaries = focus_evidence(
        selection,
        (("discussion_limitations", None), ("discussion_deployment", None)),
        (),
        max_chunks=1,
        max_chars=1,
    )
    selected = {chunk_id for _heading, chunk_id, _text in boundaries.chunks}

    assert {"discussion-scalability", "discussion-scope"} <= set(
        selection.pools["discussion_limitations"]
    )
    assert "discussion-deployment" in selection.pools["discussion_deployment"]
    assert {
        "discussion-scalability",
        "discussion-scope",
        "discussion-deployment",
    } <= selected
    assert supports_future_work(chunks[-1][1])
    assert not supports_future_work(chunks[2][1])


def test_unsupported_why_it_matters_synthesis_is_rejected() -> None:
    document = paper()
    quote = "Socioeconomic and cultural factors affect individual health but remain understudied."
    document.abstract_chunks[0].text = quote
    document.abstract_chunks[0].end_char = len(quote)
    unsupported = InsightClaim(
        claim=(
            "Studying these factors will improve diagnosis and treatment and produce equitable "
            "healthcare."
        ),
        evidence=[{"chunk_id": "abstract-1", "quote": quote}],
    )
    insights = InsightFields.empty()
    insights.main_findings = [
        InsightClaim(
            claim="Socioeconomic and cultural factors remain understudied.",
            evidence=[{"chunk_id": "abstract-1", "quote": quote}],
        )
    ]
    insights.why_it_matters = [unsupported]

    assert not synthesis_claim_supported(unsupported, "why_it_matters")
    assert filter_synthesis_support(insights, document).why_it_matters == []


def test_why_it_matters_prompt_uses_validated_story_not_generic_introduction() -> None:
    document = paper()
    generic_intro = "Visualization is an increasingly important topic across many applications."
    document.abstract = generic_intro
    document.abstract_chunks[0].text = generic_intro
    document.abstract_chunks[0].end_char = len(generic_intro)
    contribution = "We contribute a coordinated visual analytics system for cohort comparison."
    finding = "Experts found that the system enabled rapid formation and testing of hypotheses."
    additions = (
        ("contribution-evidence", "Contributions", contribution),
        ("finding-evidence", "Expert Feedback", finding),
    )
    for chunk_id, heading, text in additions:
        document.sections.append(
            DocumentSection(
                id=f"section-{chunk_id}",
                heading=heading,
                level=1,
                text=text,
                chunks=[
                    DocumentChunk(
                        id=chunk_id,
                        section_id=f"section-{chunk_id}",
                        text=text,
                        start_char=0,
                        end_char=len(text),
                    )
                ],
            )
        )
    story = InsightFields.empty()
    story.key_contributions = [
        InsightClaim(
            claim="The work contributes a coordinated cohort-comparison system.",
            evidence=[{"chunk_id": "contribution-evidence", "quote": contribution}],
        )
    ]
    story.main_findings = [
        InsightClaim(
            claim="Experts rapidly formed and tested hypotheses.",
            evidence=[{"chunk_id": "finding-evidence", "quote": finding}],
        )
    ]
    index = select_evidence(document)
    context = focus_evidence(
        index,
        (("audience", 2), ("discussion", 2), ("evaluation", 2)),
        ("audience", "discussion", "evaluation"),
        max_chunks=10,
        max_chars=6000,
    )
    catalog = InsightService._catalog(document, context, story)
    prompt = InsightService._prompt(document, catalog, "why_it_matters and target_audience", story)

    assert "Already validated research-story claims" in prompt
    assert story.key_contributions[0].claim in prompt
    assert story.main_findings[0].claim in prompt
    assert generic_intro not in prompt


def test_explicit_future_deployment_remains_future_work() -> None:
    document = paper()
    future_text = (
        "For future deployment, we plan to extend the system to additional patient datasets."
    )
    document.sections.append(
        DocumentSection(
            id="future",
            heading="Discussion and Conclusion",
            level=1,
            text=future_text,
            chunks=[
                DocumentChunk(
                    id="future-deployment",
                    section_id="future",
                    text=future_text,
                    start_char=0,
                    end_char=len(future_text),
                )
            ],
        )
    )
    selection = select_evidence(document)
    insights = InsightFields.empty()
    insights.future_work = [
        InsightClaim(
            claim="Extend the system to additional patient datasets.",
            evidence=[{"chunk_id": "future-deployment", "quote": future_text}],
        )
    ]

    assert "future-deployment" in selection.pools["future_work"]
    assert supports_future_work(future_text)
    assert len(filter_synthesis_support(insights, document).future_work) == 1


def test_completed_analysis_is_not_supported_as_future_work() -> None:
    completed = (
        "The teams then examined additional grocery-store data and observed a cohort difference."
    )

    assert not supports_future_work(completed)
    assert supports_finding(completed)


def test_method_description_is_not_supported_as_a_finding() -> None:
    assert not supports_finding("We applied the K-nearest neighbor algorithm to patient records.")
    assert not supports_finding("We decided to use cosine distance as the distance metric.")
    assert not supports_finding("We computed similarity with the proposed algorithm.")
    assert supports_finding("The comparison showed similar treatment after matching patients.")


def _validated_explicit_claim(field: str, heading: str, quote: str, claim: str) -> bool:
    document = paper()
    document.sections.append(
        DocumentSection(
            id="semantic-evidence-section",
            heading=heading,
            level=1,
            text=quote,
            chunks=[
                DocumentChunk(
                    id="semantic-evidence",
                    section_id="semantic-evidence-section",
                    text=quote,
                    start_char=0,
                    end_char=len(quote),
                )
            ],
        )
    )
    insights = InsightFields.empty()
    setattr(
        insights,
        field,
        [
            InsightClaim(
                claim=claim,
                evidence=[{"chunk_id": "semantic-evidence", "quote": quote}],
            )
        ],
    )
    return bool(getattr(validate_insights(insights, document), field))


def test_reversed_entity_comparison_is_rejected() -> None:
    quote = "Group A had higher survival than Group B."
    assert not _validated_explicit_claim(
        "main_findings",
        "Results",
        quote,
        "Group B had higher survival than Group A.",
    )


def test_correct_entity_comparison_is_retained() -> None:
    quote = "Group A had higher survival than Group B."
    assert _validated_explicit_claim(
        "main_findings",
        "Results",
        quote,
        "Group A had higher survival than Group B.",
    )


def test_higher_lower_reversal_is_rejected() -> None:
    quote = "Group A had higher accuracy than Group B."
    assert not _validated_explicit_claim(
        "main_findings",
        "Results",
        quote,
        "Group A had lower accuracy than Group B.",
    )


def test_earlier_later_reversal_is_rejected() -> None:
    quote = "Cohort Alpha presented earlier than Cohort Beta."
    assert not _validated_explicit_claim(
        "main_findings",
        "Case Study Results",
        quote,
        "Cohort Alpha presented later than Cohort Beta.",
    )


def test_longer_shorter_reversal_is_rejected() -> None:
    quote = "Population North had longer follow-up than Population South."
    assert not _validated_explicit_claim(
        "main_findings",
        "Evaluation Results",
        quote,
        "Population North had shorter follow-up than Population South.",
    )


def test_unqualified_claim_cannot_strengthen_seemed_result() -> None:
    quote = "Group A seemed to have better survival than Group B."
    assert not _validated_explicit_claim(
        "main_findings",
        "Results",
        quote,
        "Group A had better survival than Group B.",
    )
    assert _validated_explicit_claim(
        "main_findings",
        "Results",
        quote,
        "Group A seemed to have better survival than Group B.",
    )


def test_unsupported_new_attribute_is_rejected() -> None:
    quote = "Group A had higher survival than Group B."
    assert not _validated_explicit_claim(
        "main_findings",
        "Results",
        quote,
        "Group A required more feeding tubes than Group B.",
    )


def test_unsupported_named_subject_is_rejected() -> None:
    quote = "The study examined Population Alpha."
    assert not _validated_explicit_claim(
        "limitations",
        "Limitations",
        quote,
        "Population Beta had incomplete records.",
    )


def test_related_work_cannot_become_current_paper_finding() -> None:
    quote = "A prior study found that Group A had higher accuracy than Group B."
    assert not _validated_explicit_claim(
        "main_findings",
        "Related Work and Background",
        quote,
        "Group A had higher accuracy than Group B.",
    )


def test_method_evidence_cannot_become_finding() -> None:
    quote = "We decided to use cosine distance as the distance metric."
    assert not _validated_explicit_claim(
        "main_findings",
        "Evaluation",
        quote,
        "Cosine distance improved classification accuracy.",
    )


def test_actual_result_statement_remains_a_finding() -> None:
    quote = "The comparison showed Group A had higher accuracy than Group B."
    assert _validated_explicit_claim(
        "main_findings",
        "Evaluation Results",
        quote,
        "Group A had higher accuracy than Group B.",
    )


def test_explicit_numbered_contribution_survives_semantic_validation() -> None:
    quote = "The main contributions are: 1) a harmonized multi-site dataset."
    assert _validated_explicit_claim(
        "key_contributions",
        "Introduction",
        quote,
        "The work contributes a harmonized multi-site dataset.",
    )


def test_genuine_future_action_survives_semantic_validation() -> None:
    quote = "For future work, we will extend the system to additional datasets."
    assert _validated_explicit_claim(
        "future_work",
        "Discussion and Future Work",
        quote,
        "The authors will extend the system to additional datasets.",
    )


def test_conditional_future_deployment_allows_concise_action_phrase() -> None:
    quote = "For future deployment, the approach could be deployed to Cohort Beta."
    assert _validated_explicit_claim(
        "future_work",
        "Future Work",
        quote,
        "Deployment to Cohort Beta.",
    )


def test_conditional_dataset_extension_allows_concise_extension_claim() -> None:
    quote = "The approach could be extended to additional datasets."
    assert _validated_explicit_claim(
        "future_work",
        "Future Work",
        quote,
        "Extension to additional datasets.",
    )


def test_future_work_preserves_material_scope_condition() -> None:
    quote = (
        "For future deployment, the system could support pairwise comparison with other "
        "cohorts, assuming the datasets are standardized."
    )
    assert _validated_explicit_claim(
        "future_work",
        "Future Work",
        quote,
        "Deployment to additional standardized cohorts.",
    )
    assert not _validated_explicit_claim(
        "future_work",
        "Future Work",
        quote,
        "Deployment for unrestricted comparison across any number of cohorts.",
    )


def test_conditional_future_evidence_cannot_promise_outcome() -> None:
    quote = "The approach could be extended to additional datasets."
    assert not _validated_explicit_claim(
        "future_work",
        "Future Work",
        quote,
        "Extension to additional datasets will improve outcomes.",
    )


def test_conditional_future_evidence_cannot_support_invented_action() -> None:
    quote = "The approach could be extended to additional datasets."
    assert not _validated_explicit_claim(
        "future_work",
        "Future Work",
        quote,
        "Deployment of a new mobile application.",
    )


def test_completed_analysis_cannot_become_future_work() -> None:
    quote = "The team examined an additional dataset and observed a performance difference."
    assert not _validated_explicit_claim(
        "future_work",
        "Results",
        quote,
        "The team will examine an additional dataset.",
    )


def test_distinct_method_categories_are_preserved_in_technical_pack() -> None:
    document = paper()
    categories = [
        ("design", "Design Methodology", "We used interviews and parallel prototyping."),
        ("data", "Data Harmonization", "We harmonized two datasets and reconciled units."),
        ("system", "System Architecture", "We implemented a nearest neighbor algorithm."),
        (
            "evaluation-method",
            "Evaluation",
            "We conducted two case studies with domain experts using screen sharing.",
        ),
    ]
    for chunk_id, heading, text in categories:
        document.sections.append(
            DocumentSection(
                id=f"section-{chunk_id}",
                heading=heading,
                level=1,
                text=text,
                chunks=[
                    DocumentChunk(
                        id=chunk_id,
                        section_id=f"section-{chunk_id}",
                        text=text,
                        start_char=0,
                        end_char=len(text),
                    )
                ],
            )
        )

    selection = select_evidence(document)
    technical = focus_evidence(
        selection,
        (
            ("design_methods", 1),
            ("data_methods", 1),
            ("system_methods", 1),
            ("evaluation_methods", 1),
        ),
        ("methods",),
        max_chunks=1,
        max_chars=1,
    )
    selected_ids = {chunk_id for _, chunk_id, _ in technical.chunks}

    assert {chunk_id for chunk_id, _, _ in categories} <= selected_ids


def test_stage_budget_cannot_eliminate_reserved_category_coverage() -> None:
    document = paper()
    categories = (
        ("data-required", "Data Harmonization", "We harmonized heterogeneous records."),
        ("design-required", "Design Process", "We conducted interviews with domain experts."),
        ("system-required", "System", "We implemented coordinated visual analysis views."),
        ("evaluation-required", "Evaluation", "We conducted expert case-study sessions."),
    )
    for chunk_id, heading, sentence in categories:
        text = f"{sentence} " + "Supporting context. " * 30
        document.sections.append(
            DocumentSection(
                id=f"section-{chunk_id}",
                heading=heading,
                level=1,
                text=text,
                chunks=[
                    DocumentChunk(
                        id=chunk_id,
                        section_id=f"section-{chunk_id}",
                        text=text,
                        start_char=0,
                        end_char=len(text),
                    )
                ],
            )
        )

    selection = select_evidence(document)
    core = focus_evidence(
        selection,
        (
            ("data_methods", 1),
            ("design_methods", 1),
            ("system_methods", 1),
            ("evaluation_methods", 1),
        ),
        (),
        max_chunks=1,
        max_chars=1,
    )

    assert {chunk_id for chunk_id, _heading, _text in categories} <= {
        chunk_id for _heading, chunk_id, _text in core.chunks
    }


def test_dropped_chunk_prefix_is_restored_only_for_unique_selected_id() -> None:
    output = InsightFields.empty()
    output.research_problem = [
        InsightClaim(
            claim="A transformer architecture is proposed.",
            evidence=[{"chunk_id": "abc123", "quote": "transformer architecture"}],
        )
    ]
    corrected = restore_selected_chunk_prefixes(output, {"chunk-abc123"})
    assert corrected.research_problem[0].evidence[0].chunk_id == "chunk-abc123"
    assert output.research_problem[0].evidence[0].chunk_id == "abc123"
    document = paper()
    document.abstract_chunks[0].id = "chunk-abc123"
    assert len(validate_insights(corrected, document, {"chunk-abc123"}).research_problem) == 1
    unchanged = restore_selected_chunk_prefixes(output, {"chunk-other"})
    assert unchanged.research_problem[0].evidence[0].chunk_id == "abc123"


@pytest.mark.asyncio
async def test_ollama_timeout_has_distinct_error(tmp_path) -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
        service = InsightService(
            client, "http://localhost:11434", "qwen3:4b", InsightCache(tmp_path)
        )
        with pytest.raises(OllamaTimeoutError, match="timed out"):
            await service.extract(paper())


def test_merge_preserves_distinct_findings_and_merges_only_near_duplicates() -> None:
    def claim(text: str, chunk_id: str, quote: str) -> InsightClaim:
        return InsightClaim.model_validate(
            {"claim": text, "evidence": [{"chunk_id": chunk_id, "quote": quote}]}
        )

    first = InsightFields.empty()
    first.methods = [
        claim("The model uses self-attention layers in the encoder.", "method-1", "A quote."),
    ]
    first.main_findings = [
        claim("The model improves BLEU by 2.0 points.", "results-1", "A result quote."),
    ]
    second = InsightFields.empty()
    second.methods = [
        claim("The model uses self-attention layers in the encoder stack.", "method-1", "A quote."),
        claim(
            "The model uses self-attention layers in the encoder stack.",
            "method-2",
            "Another quote.",
        ),
    ]
    second.main_findings = [
        claim("The model improves BLEU by 3.0 points.", "results-1", "A result quote."),
    ]

    merged = merge_insights([first, second])

    assert len(merged.methods) == 1
    assert merged.methods[0].claim == "The model uses self-attention layers in the encoder stack."
    assert [ref.chunk_id for ref in merged.methods[0].evidence] == ["method-1", "method-2"]
    assert len(merged.main_findings) == 2


def test_merge_consolidates_repeated_measurement_and_applies_global_soft_limits() -> None:
    def claim(text: str, chunk_id: str) -> InsightClaim:
        return InsightClaim.model_validate(
            {"claim": text, "evidence": [{"chunk_id": chunk_id, "quote": "Reported result."}]}
        )

    first = InsightFields.empty()
    first.main_findings = [
        claim("Transformer achieves 28.4 BLEU on German translation.", "abstract-1"),
    ]
    first.methods = [
        claim(f"Distinct method component {number} is used.", f"method-{number}")
        for number in range(10)
    ]
    second = InsightFields.empty()
    second.main_findings = [
        claim("The Transformer achieves 28.4 BLEU on German translation.", "results-1"),
        claim("The Transformer achieves 41.8 BLEU on French translation.", "results-2"),
    ]

    merged = merge_insights([first, second])

    assert len(merged.main_findings) == 2
    assert {ref.chunk_id for ref in merged.main_findings[0].evidence} == {
        "abstract-1",
        "results-1",
    }
    assert len(merged.methods) == 8
    assert merged.methods[0].claim.endswith("0 is used.")
    assert merged.methods[-1].claim.endswith("7 is used.")
    assert merged.limitations == []


def test_merge_keeps_distinct_method_locations() -> None:
    first = InsightFields.empty()
    first.methods = [
        InsightClaim.model_validate(
            {
                "claim": "The encoder uses masked self-attention.",
                "evidence": [{"chunk_id": "method-1", "quote": "An attention quote."}],
            }
        )
    ]
    second = InsightFields.empty()
    second.methods = [
        InsightClaim.model_validate(
            {
                "claim": "The decoder uses masked self-attention.",
                "evidence": [{"chunk_id": "method-2", "quote": "A different attention quote."}],
            }
        )
    ]

    assert len(merge_insights([first, second]).methods) == 2


@pytest.mark.asyncio
async def test_ollama_unavailable_and_document_without_chunks(tmp_path) -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as client:
        service = InsightService(
            client, "http://localhost:11434", "qwen3:4b", InsightCache(tmp_path)
        )
        with pytest.raises(OllamaUnavailableError, match="unavailable"):
            await service.extract(paper())
        empty = paper().model_copy(update={"abstract_chunks": [], "sections": []})
        with pytest.raises(InsightInputError, match="no evidence chunks"):
            await service.extract(empty)
        duplicate = paper().model_copy(deep=True)
        duplicate.sections[0].chunks[0].id = "abstract-1"
        with pytest.raises(InsightInputError, match="duplicate evidence chunk IDs"):
            await service.extract(duplicate)


def test_insights_api_serializes_success_and_maps_errors(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}",
        parsed_document_cache_dir=str(tmp_path / "parsed"),
        insight_cache_dir=str(tmp_path / "insights"),
    )
    app = create_app(settings)
    result = InsightsResponse(
        paper_id="paper-1",
        document_fingerprint="b" * 64,
        model="qwen3:4b",
        extraction_version=EXTRACTION_VERSION,
        cached=False,
        insights=InsightFields.model_validate(method_output()),
    )

    async def success(_: ParsedPaper) -> InsightsResponse:
        return result

    async def unavailable(_: ParsedPaper) -> InsightsResponse:
        raise OllamaUnavailableError("Local Ollama is unavailable")

    async def invalid(_: ParsedPaper) -> InsightsResponse:
        raise InsightOutputError("Ollama returned invalid structured insights")

    async def timed_out(_: ParsedPaper) -> InsightsResponse:
        raise OllamaTimeoutError("Local Ollama extraction timed out")

    async def no_chunks(_: ParsedPaper) -> InsightsResponse:
        raise InsightInputError("Parsed paper has no evidence chunks to extract from")

    with TestClient(app) as client:
        app.state.insight_service = SimpleNamespace(extract=success)
        response = client.post("/api/papers/insights", json={"document": paper().model_dump()})
        app.state.insight_service = SimpleNamespace(extract=unavailable)
        unavailable_response = client.post(
            "/api/papers/insights", json={"document": paper().model_dump()}
        )
        app.state.insight_service = SimpleNamespace(extract=invalid)
        invalid_response = client.post(
            "/api/papers/insights", json={"document": paper().model_dump()}
        )
        app.state.insight_service = SimpleNamespace(extract=no_chunks)
        no_chunks_response = client.post(
            "/api/papers/insights", json={"document": paper().model_dump()}
        )
        app.state.insight_service = SimpleNamespace(extract=timed_out)
        timeout_response = client.post(
            "/api/papers/insights", json={"document": paper().model_dump()}
        )

    assert response.status_code == 200
    assert response.json()["insights"]["methods"][0]["evidence"][0]["chunk_id"] == "method-1"
    assert unavailable_response.status_code == 503
    assert invalid_response.status_code == 502
    assert no_chunks_response.status_code == 422
    assert timeout_response.status_code == 504


def test_insights_stream_reports_real_stages_and_result(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'stream.sqlite3'}",
        parsed_document_cache_dir=str(tmp_path / "parsed"),
        insight_cache_dir=str(tmp_path / "insights"),
    )
    app = create_app(settings)

    async def success(_: ParsedPaper, on_progress) -> InsightsResponse:
        await on_progress("Selecting evidence")
        await on_progress("Generating core research story (1/3)")
        await on_progress("Generating evidence and boundaries (2/3)")
        await on_progress("Generating grounded synthesis (3/3)")
        await on_progress("Validating evidence")
        return InsightsResponse(
            paper_id="paper-1",
            document_fingerprint="b" * 64,
            model="qwen3:4b",
            extraction_version=EXTRACTION_VERSION,
            cached=False,
            insights=InsightFields.model_validate(method_output()),
        )

    with TestClient(app) as client:
        app.state.insight_service = SimpleNamespace(extract=success)
        response = client.post(
            "/api/papers/insights/stream", json={"document": paper().model_dump()}
        )

    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert [event["stage"] for event in events[:-1]] == [
        "Selecting evidence",
        "Generating core research story (1/3)",
        "Generating evidence and boundaries (2/3)",
        "Generating grounded synthesis (3/3)",
        "Validating evidence",
    ]
    assert events[-1]["type"] == "result"
    assert events[-1]["data"]["insights"]["methods"][0]["evidence"][0]["chunk_id"] == "method-1"


def test_ollama_configuration_is_local_only() -> None:
    settings = Settings(_env_file=None)
    assert settings.ollama_model == "qwen3:1.7b"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_timeout_seconds == 180
    assert settings.ollama_batch_chars == 13000
    with pytest.raises(ValidationError, match="local HTTP Ollama address"):
        Settings(_env_file=None, ollama_base_url="https://cloud.example/api")


@pytest.mark.asyncio
async def test_diagnostics_are_off_by_default_and_preserve_response_schema(tmp_path) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return focused_response(request, method_output())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await InsightService(
            client, "http://localhost:11434", "qwen3:1.7b", InsightCache(tmp_path)
        ).extract(paper())

    assert calls == 3
    assert set(response.model_dump()) == {
        "paper_id",
        "document_fingerprint",
        "model",
        "extraction_version",
        "cached",
        "insights",
    }


@pytest.mark.asyncio
async def test_diagnostics_record_evidence_candidates_and_actual_dispositions(tmp_path) -> None:
    document = paper()
    future_text = "The current system uses self-attention layers."
    document.sections.append(
        DocumentSection(
            id="future-section",
            heading="Future Work",
            level=1,
            text=future_text,
            chunks=[
                DocumentChunk(
                    id="future-1",
                    section_id="future-section",
                    text=future_text,
                    start_char=0,
                    end_char=len(future_text),
                )
            ],
        )
    )
    output = method_output()
    output["future_work"] = [
        {
            "claim": "Extend the current system.",
            "evidence": [{"chunk_id": "future-1", "quote": future_text}],
        }
    ]
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return focused_response(request, output)

    diagnostics = InsightDiagnostics()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await InsightService(
            client, "http://localhost:11434", "qwen3:1.7b", InsightCache(tmp_path)
        ).extract(document, diagnostics=diagnostics)

    trace = diagnostics.to_dict()
    stages = {stage["stage_id"]: stage for stage in trace["stages"]}
    core_candidates = stages["core_story"]["candidate_claims"]
    boundary_candidates = stages["evidence_boundaries"]["candidate_claims"]
    retained = next(candidate for candidate in core_candidates if candidate["field"] == "methods")
    rejected = next(
        candidate for candidate in boundary_candidates if candidate["field"] == "future_work"
    )

    assert calls == 3
    assert stages["core_story"]["selected_evidence"]
    assert stages["core_story"]["raw_model_output"] is not None
    assert retained["final_retained"] is True
    assert rejected["evidence_resolution"]["reason"] == "future_work_guard"
    assert rejected["final_retained"] is False
    assert result.insights.future_work == []
    assert trace["final_output"]["counts"]["methods"] == 1


def test_diagnostic_reason_reporting_preserves_validator_boolean_outcome() -> None:
    quote = "Group A had higher accuracy than Group B."
    claim = InsightClaim(
        claim="Group A had lower accuracy than Group B.",
        evidence=[{"chunk_id": "result-1", "quote": quote}],
    )
    headings = {"result-1": "Results"}

    reason = explicit_claim_rejection_reason(claim, "main_findings", headings)

    assert reason == "comparison_direction_mismatch"
    assert explicit_claim_supported(claim, "main_findings", headings) is (reason is None)


def neutral_paper() -> ParsedPaper:
    document = paper()
    document.title = "Synthetic Systems Study"
    document.abstract = "A synthetic study evaluates a configurable analysis workflow."
    document.abstract_chunks[0].text = document.abstract
    document.abstract_chunks[0].end_char = len(document.abstract)
    document.sections[0].text = "The workflow uses a configurable processing stage."
    document.sections[0].chunks[0].text = document.sections[0].text
    document.sections[0].chunks[0].end_char = len(document.sections[0].text)
    return document


def _append_section(document: ParsedPaper, chunk_id: str, heading: str, text: str) -> None:
    document.sections.append(
        DocumentSection(
            id=f"section-{chunk_id}",
            heading=heading,
            level=1,
            text=text,
            chunks=[
                DocumentChunk(
                    id=chunk_id,
                    section_id=f"section-{chunk_id}",
                    text=text,
                    start_char=0,
                    end_char=len(text),
                )
            ],
        )
    )


def _reserved_passages(document: ParsedPaper) -> list[str]:
    selection = reserve_boundary_passages(select_evidence(document))
    return [passage for passages in selection.priority_passages.values() for passage in passages]


def test_explicit_limitation_section_reserves_concrete_sentence() -> None:
    document = neutral_paper()
    limitation = "The prototype does not support concurrent updates from multiple clients."
    _append_section(document, "limitation", "Limitations", limitation)

    assert limitation in _reserved_passages(document)


def test_explicit_future_section_reserves_concrete_sentence() -> None:
    document = neutral_paper()
    future = "We will evaluate the workflow with participants from several organizations."
    _append_section(document, "future", "Future Work", future)

    assert future in _reserved_passages(document)


def test_combined_boundary_section_reserves_limitation_and_future_action() -> None:
    document = neutral_paper()
    limitation = "The current implementation cannot process concurrent event streams."
    future = "We plan to evaluate a distributed implementation in future work."
    _append_section(
        document,
        "combined-boundaries",
        "Discussion, Limitations, and Future Work",
        f"{limitation} {future}",
    )

    reserved = _reserved_passages(document)

    assert limitation in reserved
    assert future in reserved


def test_boundary_reservation_skips_bare_headers_and_is_bounded_and_deduplicated() -> None:
    document = neutral_paper()
    limitation_sentences = [
        f"The prototype does not support processing mode {index} under constrained memory."
        for index in range(5)
    ]
    future_sentences = [
        f"We will evaluate processing mode {index} with additional participants."
        for index in range(5)
    ]
    text = " ".join(
        [
            "Our limitations are as follows.",
            *limitation_sentences,
            "Future Work.",
            *future_sentences,
            future_sentences[0],
        ]
    )
    _append_section(
        document,
        "bounded-boundaries",
        "Discussion, Limitations, and Future Work",
        text,
    )

    reserved = _reserved_passages(document)

    assert len(reserved) == 8
    assert len(set(reserved)) == 8
    assert "Our limitations are as follows." not in reserved
    assert "Future Work." not in reserved


@pytest.mark.parametrize(
    "text",
    [
        "For future work, we will conduct a user study with independent participants.",
        "We plan to evaluate the workflow under additional operating conditions.",
        "We envision extending the analysis to additional datasets.",
    ],
)
def test_author_forward_actions_are_recognized_as_future_work(text: str) -> None:
    assert supports_future_work(text)


@pytest.mark.parametrize(
    "text",
    [
        "We conducted a user study with independent participants.",
        "The evaluation showed that the workflow reduced processing time.",
        "Users should consider applying a different workflow in this situation.",
        "Experts suggested that future versions should include configurable dashboards.",
    ],
)
def test_non_author_or_completed_actions_are_not_future_work(text: str) -> None:
    assert not supports_future_work(text)


def test_prior_work_cannot_validate_current_paper_contribution() -> None:
    quote = "Their work introduces a reusable taxonomy for interactive systems."
    assert not _validated_explicit_claim(
        "key_contributions",
        "Background",
        quote,
        "The paper contributes a reusable taxonomy for interactive systems.",
    )


def test_prior_work_cannot_validate_current_paper_limitation() -> None:
    quote = "Their method requires manual configuration and cannot process streaming data."
    assert not _validated_explicit_claim(
        "limitations",
        "Related Work",
        quote,
        "The current method requires manual configuration.",
    )


def test_current_paper_contribution_remains_valid_with_prior_work_contrast() -> None:
    quote = (
        "Prior work introduces static views [1], but we introduce an interactive workflow "
        "for configurable analysis."
    )
    assert _validated_explicit_claim(
        "key_contributions",
        "Introduction",
        quote,
        "We introduce an interactive workflow for configurable analysis.",
    )


def test_bare_contribution_header_cannot_support_detailed_claim() -> None:
    assert not _validated_explicit_claim(
        "key_contributions",
        "Introduction",
        "Our key contributions are:",
        "The paper contributes an interactive workflow.",
    )


def test_substantive_contribution_after_header_remains_valid() -> None:
    quote = "Our key contributions are: 1) We introduce an interactive analysis workflow."
    assert _validated_explicit_claim(
        "key_contributions",
        "Introduction",
        quote,
        "We introduce an interactive analysis workflow.",
    )


def test_introductory_capability_is_not_a_main_finding() -> None:
    quote = "The framework supports interactive comparison across configurable datasets."
    assert not _validated_explicit_claim(
        "main_findings",
        "Introduction",
        quote,
        "The framework supports interactive comparison across configurable datasets.",
    )


@pytest.mark.parametrize(
    ("heading", "quote", "claim"),
    [
        (
            "Evaluation Results",
            "The evaluation showed that the method reduced runtime by 20 percent.",
            "The method reduced runtime by 20 percent.",
        ),
        (
            "Expert Feedback",
            "Participants reported that the workflow simplified comparative analysis.",
            "Participants reported simpler comparative analysis.",
        ),
        (
            "Survey Analysis",
            "The analysis identified three recurring design patterns across reviewed systems.",
            "The analysis identified three recurring design patterns.",
        ),
    ],
)
def test_quantitative_and_qualitative_results_remain_findings(
    heading: str, quote: str, claim: str
) -> None:
    assert _validated_explicit_claim("main_findings", heading, quote, claim)


def test_catalog_truncation_stops_at_word_boundary_and_remains_quote_matchable() -> None:
    document = neutral_paper()
    source = " ".join(f"descriptiveword{index}" for index in range(40))
    _append_section(document, "long-excerpt", "Evaluation Results", source)
    catalog = evidence_catalog(select_evidence(document))
    quote = next(
        quote for chunk_id, quote, _heading in catalog.values() if chunk_id == "long-excerpt"
    )

    assert len(quote) <= 280
    assert quote_in_chunk(quote, source)
    assert len(quote) == len(source) or source[len(quote)].isspace()


def test_relational_associated_with_phrase_is_not_uncertainty() -> None:
    quote = "The method has no guaranteed bound for error associated with approximation bias."
    assert _validated_explicit_claim(
        "limitations",
        "Limitations",
        quote,
        "The method has no guaranteed bound for error associated with approximation bias.",
    )


def test_genuine_uncertainty_must_still_be_preserved() -> None:
    quote = "The workflow may reduce processing time under stable input conditions."
    assert not _validated_explicit_claim(
        "main_findings",
        "Evaluation Results",
        quote,
        "The workflow reduces processing time under stable input conditions.",
    )
    assert _validated_explicit_claim(
        "main_findings",
        "Evaluation Results",
        quote,
        "The workflow may reduce processing time under stable input conditions.",
    )


def _validated_story_with_why(
    claim: str, evidence: list[tuple[str, str]]
) -> tuple[ParsedPaper, InsightFields]:
    document = neutral_paper()
    problem = "Existing processing is too slow for interactive analysis."
    contribution = "We introduce a cache that supports interactive analysis."
    finding = (
        "The evaluation found that the cache reduced runtime while maintaining comparable "
        "output quality."
    )
    for chunk_id, heading, text in (
        ("story-problem", "Introduction", problem),
        ("story-contribution", "Contributions", contribution),
        ("story-finding", "Evaluation Results", finding),
    ):
        _append_section(document, chunk_id, heading, text)
    insights = InsightFields.empty()
    insights.research_problem = [
        InsightClaim(
            claim="Existing processing is too slow for interactive analysis.",
            evidence=[{"chunk_id": "story-problem", "quote": problem}],
        )
    ]
    insights.key_contributions = [
        InsightClaim(
            claim="The work introduces a cache for interactive analysis.",
            evidence=[{"chunk_id": "story-contribution", "quote": contribution}],
        )
    ]
    insights.main_findings = [
        InsightClaim(
            claim="The cache reduced runtime while maintaining comparable output quality.",
            evidence=[{"chunk_id": "story-finding", "quote": finding}],
        )
    ]
    insights.why_it_matters = [
        InsightClaim(
            claim=claim,
            evidence=[{"chunk_id": chunk_id, "quote": quote} for chunk_id, quote in evidence],
        )
    ]
    return document, insights


def test_why_it_matters_uses_minimum_union_of_validated_upstream_evidence() -> None:
    contribution = "We introduce a cache that supports interactive analysis."
    finding = (
        "The evaluation found that the cache reduced runtime while maintaining comparable "
        "output quality."
    )
    document, insights = _validated_story_with_why(
        "The cache makes interactive analysis faster while maintaining comparable output quality.",
        [
            ("story-problem", "Existing processing is too slow for interactive analysis."),
            ("story-contribution", contribution),
            ("story-finding", finding),
        ],
    )

    retained = filter_synthesis_support(insights, document).why_it_matters

    assert len(retained) == 1
    retained_ids = {reference.chunk_id for reference in retained[0].evidence}
    assert len(retained_ids) == 2
    assert "story-finding" in retained_ids
    assert retained_ids <= {"story-problem", "story-contribution", "story-finding"}


def test_why_it_matters_rejects_unsupported_new_concepts() -> None:
    document, insights = _validated_story_with_why(
        "The cache reduces hospital costs and guarantees improved clinical diagnosis.",
        [
            (
                "story-finding",
                "The evaluation found that the cache reduced runtime while maintaining comparable "
                "output quality.",
            )
        ],
    )

    assert filter_synthesis_support(insights, document).why_it_matters == []


def test_why_it_matters_preserves_upstream_uncertainty() -> None:
    document = neutral_paper()
    quote = "The evaluation suggests that the workflow may reduce processing time."
    _append_section(document, "uncertain-finding", "Evaluation Results", quote)
    insights = InsightFields.empty()
    insights.main_findings = [
        InsightClaim(
            claim="The workflow may reduce processing time.",
            evidence=[{"chunk_id": "uncertain-finding", "quote": quote}],
        )
    ]
    insights.why_it_matters = [
        InsightClaim(
            claim="The workflow will reduce processing time.",
            evidence=[{"chunk_id": "uncertain-finding", "quote": quote}],
        )
    ]

    assert filter_synthesis_support(insights, document).why_it_matters == []


def test_why_it_matters_can_remain_empty_without_validated_story_support() -> None:
    document = neutral_paper()
    discussion = "Interactive analysis can be useful in several settings."
    _append_section(document, "discussion-only", "Discussion", discussion)
    insights = InsightFields.empty()
    insights.why_it_matters = [
        InsightClaim(
            claim="Interactive analysis is useful in several settings.",
            evidence=[{"chunk_id": "discussion-only", "quote": discussion}],
        )
    ]

    assert filter_synthesis_support(insights, document).why_it_matters == []


@pytest.mark.asyncio
async def test_v15_synthesis_uses_evidence_ids_without_an_extra_model_call(tmp_path) -> None:
    calls: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        return focused_response(request, empty_output())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await InsightService(
            client, "http://localhost:11434", "qwen3:1.7b", InsightCache(tmp_path)
        ).extract(neutral_paper())

    assert len(calls) == 3
    synthesis_schema = calls[2]["format"]
    assert "evidence_ids" in json.dumps(synthesis_schema)
    assert "smallest set of 1-3 evidence IDs" in calls[2]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_diagnostics_record_updated_prior_work_rejection_reason(tmp_path) -> None:
    document = neutral_paper()
    prior = "Their work introduces a static taxonomy for configurable interfaces."
    _append_section(document, "prior-contribution", "Introduction", prior)
    output = empty_output()
    output["key_contributions"] = [
        {
            "claim": "The paper contributes a static taxonomy for configurable interfaces.",
            "evidence": [{"chunk_id": "prior-contribution", "quote": prior}],
        }
    ]
    method_quote = document.sections[0].chunks[0].text
    output["methods"] = [
        {
            "claim": "The workflow uses a configurable processing stage.",
            "evidence": [{"chunk_id": "method-1", "quote": method_quote}],
        }
    ]
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return focused_response(request, output)

    diagnostics = InsightDiagnostics()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await InsightService(
            client, "http://localhost:11434", "qwen3:1.7b", InsightCache(tmp_path)
        ).extract(document, diagnostics=diagnostics)

    candidate = next(
        candidate
        for stage in diagnostics.to_dict()["stages"]
        for candidate in stage["candidate_claims"]
        if candidate["field"] == "key_contributions"
    )
    assert calls == 3
    assert result.insights.key_contributions == []
    assert candidate["validation"][0]["reason"] == "prior_work_attribution"
