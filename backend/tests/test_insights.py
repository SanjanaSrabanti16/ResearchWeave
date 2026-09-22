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
from app.services.insight_selector import evidence_catalog, select_evidence
from app.services.insight_service import (
    EXTRACTION_VERSION,
    InsightInputError,
    InsightOutputError,
    InsightService,
    OllamaTimeoutError,
    OllamaUnavailableError,
    document_fingerprint,
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
                            {
                                "claim": item["claim"],
                                "evidence_id": evidence_id(item),
                            }
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
        "main_findings",
        "why_it_matters",
        "target_audience",
    }
    assert set(requests[1]["format"]["properties"]) == {
        "methods",
        "limitations",
        "future_work",
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
    transport = httpx.MockTransport(lambda request: focused_response(request, output))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await InsightService(
            client, "http://localhost:11434", "qwen3:4b", InsightCache(tmp_path)
        ).extract(paper())

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

    assert calls == 2
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
    assert calls == 2
    assert response.cached is False
    assert response.insights.methods[0].evidence[0].chunk_id == "method-1"


@pytest.mark.asyncio
async def test_selected_evidence_uses_two_calls(tmp_path) -> None:
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

    assert calls == 2
    assert len(result.insights.methods) == 1


@pytest.mark.asyncio
async def test_multiple_supported_claims_from_two_focused_calls_are_preserved(tmp_path) -> None:
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

    assert calls == 2
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
    quotes = [quote for _, quote, _ in evidence_catalog(selection).values()]

    assert "contribution-list" in selection.pools["contributions"]
    for phrase in (
        "1) a problem characterization",
        "2) a harmonized dataset",
        "3) a visual analysis design",
        "4) an implemented analysis system",
        "5) an expert evaluation",
    ):
        assert any(phrase in quote for quote in quotes)


def test_evaluation_section_is_guaranteed_to_findings_call() -> None:
    document = paper()
    evaluation_text = (
        "We report results from two case studies with domain experts. "
        "The evaluation found that experts rapidly formed and tested hypotheses."
    )
    document.sections.append(
        DocumentSection(
            id="evaluation",
            heading="Evaluation and Case Studies",
            level=1,
            text=evaluation_text,
            chunks=[
                DocumentChunk(
                    id="evaluation-1",
                    section_id="evaluation",
                    text=evaluation_text,
                    start_char=0,
                    end_char=len(evaluation_text),
                )
            ],
        )
    )

    selection = select_evidence(document)
    overview = InsightService._focus_selection(
        selection,
        (
            ("contributions", 4),
            ("evaluation", 7),
            ("discussion", 4),
            ("overview", 6),
            ("findings", 8),
        ),
        18,
    )

    assert "evaluation-1" in selection.pools["evaluation"]
    assert "evaluation-1" in {chunk_id for _, chunk_id, _ in overview.chunks}


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
    assert len(filter_synthesis_support(insights, document).future_work) == 1


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
    technical = InsightService._focus_selection(
        selection,
        (
            ("design_methods", 3),
            ("data_methods", 3),
            ("system_methods", 5),
            ("evaluation_methods", 3),
            ("limitations", 5),
            ("future_work", 4),
            ("methods", 10),
            ("gaps", 4),
        ),
        18,
    )
    selected_ids = {chunk_id for _, chunk_id, _ in technical.chunks}

    assert {chunk_id for chunk_id, _, _ in categories} <= selected_ids


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
        await on_progress("Generating grounded insights (1/2)")
        await on_progress("Generating grounded insights (2/2)")
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
        "Generating grounded insights (1/2)",
        "Generating grounded insights (2/2)",
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
