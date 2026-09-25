import inspect
import json
import re

import httpx
import pytest

from app.models.document import DocumentChunk, DocumentSection, ParsedPaper, SourcePDF
from app.models.insights import INSIGHT_FIELDS, InsightClaim, InsightFields, InsightsResponse
from app.services.insight_cache import InsightCache
from app.services.insight_diagnostics import InsightDiagnostics
from app.services.insight_selector import (
    evidence_catalog,
    reserve_boundary_passages,
    select_evidence,
    supports_future_work,
)
from app.services.insight_service import (
    EXTRACTION_VERSION,
    InsightService,
    explicit_claim_rejection_reason,
    filter_synthesis_support,
    quote_in_chunk,
    synthesis_claim_supported,
)


def _document(*sections: tuple[str, str, str]) -> ParsedPaper:
    return ParsedPaper(
        paper_id="synthetic-paper",
        title="A Configurable Analysis Workflow",
        abstract="The study examines a configurable analysis workflow.",
        abstract_chunks=[
            DocumentChunk(
                id="abstract",
                section_id="abstract",
                text="The study examines a configurable analysis workflow.",
                start_char=0,
                end_char=54,
            )
        ],
        sections=[
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
            for chunk_id, heading, text in sections
        ],
        parser="grobid",
        parser_version="0.9.1-crf+frontmatter-v1",
        source_pdf=SourcePDF(acquisition_method="upload", sha256="b" * 64, size_bytes=256),
    )


def _explicit(field: str, heading: str, evidence: str, claim: str) -> bool:
    item = InsightClaim(
        claim=claim,
        evidence=[{"chunk_id": "evidence", "quote": evidence}],
    )
    return explicit_claim_rejection_reason(item, field, {"evidence": heading}) is None


def _reserved(heading: str, text: str) -> list[str]:
    selection = reserve_boundary_passages(
        _selection := select_evidence(_document(("b", heading, text)))
    )
    assert _selection.chunks
    return [passage for passages in selection.priority_passages.values() for passage in passages]


def _why_story(
    claim: str,
    cited: tuple[str, ...],
    *,
    include_problem: bool = True,
    include_contribution: bool = True,
    include_finding: bool = True,
) -> tuple[ParsedPaper, InsightFields]:
    problem = "Manual review is too slow for interactive monitoring."
    contribution = "We introduce an indexed cache that enables interactive monitoring."
    finding = "The evaluation found lower latency while preserving output quality."
    document = _document(
        ("problem", "Introduction", problem),
        ("contribution", "Contributions", contribution),
        ("finding", "Evaluation Results", finding),
    )
    insights = InsightFields.empty()
    if include_problem:
        insights.research_problem = [
            InsightClaim(
                claim="Manual review is too slow for interactive monitoring.",
                evidence=[{"chunk_id": "problem", "quote": problem}],
            )
        ]
    if include_contribution:
        insights.key_contributions = [
            InsightClaim(
                claim="The work introduces an indexed cache for interactive monitoring.",
                evidence=[{"chunk_id": "contribution", "quote": contribution}],
            )
        ]
    if include_finding:
        insights.main_findings = [
            InsightClaim(
                claim="The cache had lower latency while preserving output quality.",
                evidence=[{"chunk_id": "finding", "quote": finding}],
            )
        ]
    quote_by_id = {
        "problem": problem,
        "contribution": contribution,
        "finding": finding,
    }
    insights.why_it_matters = [
        InsightClaim(
            claim=claim,
            evidence=[{"chunk_id": chunk_id, "quote": quote_by_id[chunk_id]} for chunk_id in cited],
        )
    ]
    return document, insights


def test_v16_01_their_work_is_not_current_contribution() -> None:
    evidence = "Their work introduces a reusable taxonomy for monitoring tasks."
    assert not _explicit("key_contributions", "Introduction", evidence, evidence)


def test_v16_02_named_external_authors_are_not_current_contribution() -> None:
    evidence = "Smith et al. propose a reusable taxonomy for monitoring tasks."
    assert not _explicit("key_contributions", "Introduction", evidence, evidence)


def test_v16_03_prior_work_is_not_current_limitation() -> None:
    evidence = "Previous work does not support concurrent monitoring."
    assert not _explicit("limitations", "Discussion", evidence, evidence)


def test_v16_04_prior_work_is_not_current_finding() -> None:
    evidence = "Their study found lower accuracy under noisy inputs."
    assert not _explicit("main_findings", "Introduction", evidence, evidence)


def test_v16_05_prior_contrast_preserves_current_contribution() -> None:
    evidence = "Unlike prior work, we introduce an indexed workflow for live monitoring."
    assert _explicit("key_contributions", "Introduction", evidence, evidence)


def test_v16_06_citation_does_not_invalidate_current_evidence() -> None:
    evidence = "We introduce an indexed workflow for live monitoring [12]."
    assert _explicit("key_contributions", "Introduction", evidence, evidence)


def test_v16_07_mixed_sentence_attributes_current_clause_correctly() -> None:
    evidence = "Previous work uses static tables; our method introduces interactive views."
    claim = "Our method introduces interactive views."
    assert _explicit("key_contributions", "Introduction", evidence, claim)


def test_v16_08_need_alone_cannot_support_specific_contribution() -> None:
    evidence = "There is a need for a structured approach to monitoring."
    claim = "We introduce a declarative monitoring grammar."
    assert not _explicit("key_contributions", "Introduction", evidence, claim)


def test_v16_09_background_alone_cannot_support_current_contribution() -> None:
    evidence = "Structured monitoring grammars organize complex design choices."
    claim = "We contribute a structured monitoring grammar."
    assert not _explicit("key_contributions", "Background", evidence, claim)


def test_v16_10_current_introduction_supports_contribution() -> None:
    evidence = "We introduce a declarative monitoring grammar."
    assert _explicit("key_contributions", "Introduction", evidence, evidence)


def test_v16_11_systematic_analysis_is_valid_non_artifact_contribution() -> None:
    evidence = "We conduct a systematic analysis of interaction patterns."
    claim = "A systematic analysis of interaction patterns."
    assert _explicit("key_contributions", "Introduction", evidence, claim)


def test_v16_12_contribution_header_alone_is_not_support() -> None:
    evidence = "Our key contributions are:"
    claim = "We introduce an indexed monitoring workflow."
    assert not _explicit("key_contributions", "Introduction", evidence, claim)


def test_v16_13_adjacent_contribution_item_is_preserved() -> None:
    text = "Our contributions are. We introduce an indexed workflow for live monitoring."
    catalog = evidence_catalog(select_evidence(_document(("c", "Introduction", text))))
    quotes = [quote for chunk_id, quote, _heading in catalog.values() if chunk_id == "c"]
    assert any("indexed workflow" in quote for quote in quotes)


def test_v16_14_contribution_items_are_bounded_and_deduplicated() -> None:
    items = [f"{index}) We introduce analysis component {index}." for index in range(1, 9)]
    text = "Our contributions include: " + " ".join([*items, items[0]])
    catalog = evidence_catalog(select_evidence(_document(("c", "Introduction", text))))
    quotes = [quote for chunk_id, quote, _heading in catalog.values() if chunk_id == "c"]
    component_quotes = [
        quote for quote in quotes if re.match(r"\d+\)\s+We introduce analysis component", quote)
    ]
    assert len(component_quotes) <= 6
    assert len(component_quotes) == len(set(component_quotes))


def test_v16_15_comparison_setup_does_not_support_outperformance() -> None:
    evidence = "We compare the workflow against a baseline configuration."
    claim = "The workflow outperforms the baseline configuration."
    assert not _explicit("main_findings", "Evaluation", evidence, claim)


def test_v16_16_runtime_measurement_setup_does_not_support_decrease() -> None:
    evidence = "We evaluated runtime across three configurations."
    claim = "Runtime decreased across the configurations."
    assert not _explicit("main_findings", "Evaluation", evidence, claim)


def test_v16_17_explicit_comparative_outcome_is_a_finding() -> None:
    evidence = "The workflow had lower latency than the baseline."
    assert _explicit("main_findings", "Results", evidence, evidence)


def test_v16_18_quantitative_outcome_is_a_finding() -> None:
    evidence = "Accuracy improved to 84 percent after calibration."
    assert _explicit("main_findings", "Results", evidence, evidence)


def test_v16_19_qualitative_expert_observation_is_a_finding() -> None:
    evidence = "Experts reported that the workflow simplified comparative analysis."
    assert _explicit("main_findings", "Expert Feedback", evidence, evidence)


def test_v16_20_survey_result_is_a_finding() -> None:
    evidence = "The survey identified three recurring coordination barriers."
    assert _explicit("main_findings", "Survey Results", evidence, evidence)


def test_v16_21_use_case_capability_is_not_a_finding() -> None:
    evidence = "The use case shows how analysts can compare configurable records."
    assert not _explicit("main_findings", "Case Study", evidence, evidence)


def test_v16_22_method_description_remains_a_method_not_a_finding() -> None:
    evidence = "We apply a nearest-neighbor algorithm to compare records."
    assert _explicit("methods", "Methods", evidence, evidence)
    assert not _explicit("main_findings", "Methods", evidence, evidence)


def test_v16_23_conclusion_restatement_of_outcome_is_a_finding() -> None:
    evidence = "We show that calibration reduced error across all tested conditions."
    assert _explicit("main_findings", "Conclusion", evidence, evidence)


def test_v16_24_limitation_section_reserves_concrete_sentence() -> None:
    sentence = "The prototype does not support concurrent updates from multiple clients."
    assert sentence in _reserved("Limitations", sentence)


def test_v16_25_combined_discussion_conclusion_reserves_limitation() -> None:
    sentence = "The current evaluation was not conducted outside the laboratory."
    assert sentence in _reserved("Discussion and Conclusion", sentence)


def test_v16_26_bare_limitation_heading_does_not_consume_slot() -> None:
    assert "Our limitations are as follows." not in _reserved(
        "Limitations", "Our limitations are as follows."
    )


def test_v16_27_explicit_we_did_not_statement_is_a_limitation() -> None:
    evidence = "We did not conduct a formal user study."
    claim = "The work did not include a formal user study."
    assert _explicit("limitations", "Limitations", evidence, claim)


def test_v16_28_explicit_not_evaluated_statement_is_a_limitation() -> None:
    evidence = "Scalability was not evaluated beyond two concurrent streams."
    assert _explicit("limitations", "Limitations", evidence, evidence)


def test_v16_29_silence_does_not_create_a_limitation() -> None:
    evidence = "We evaluated accuracy across two configurations."
    claim = "Scalability was not evaluated."
    assert not _explicit("limitations", "Evaluation", evidence, claim)


def test_v16_30_prior_negative_statement_is_not_current_limitation() -> None:
    evidence = "Their method was not evaluated with concurrent streams."
    assert not _explicit("limitations", "Related Work", evidence, evidence)


def test_v16_31_author_will_conduct_action_is_future_work() -> None:
    assert supports_future_work("We will conduct a broader field evaluation.")


def test_v16_32_author_plan_to_integrate_is_future_work() -> None:
    assert supports_future_work("We plan to integrate additional datasets.")


def test_v16_33_author_intent_to_evaluate_is_future_work() -> None:
    assert supports_future_work("We intend to evaluate the workflow in new settings.")


def test_v16_34_conditional_future_strength_is_preserved() -> None:
    evidence = "The approach could be extended to new sites if records are standardized."
    assert _explicit("future_work", "Future Work", evidence, "Extension to standardized sites.")
    assert not _explicit(
        "future_work",
        "Future Work",
        evidence,
        "The extension will improve outcomes at every site.",
    )


def test_v16_35_participant_suggestion_is_not_author_future_work() -> None:
    assert not supports_future_work("Participants suggested extending the dashboard.")


def test_v16_36_user_recommendation_is_not_author_future_work() -> None:
    assert not supports_future_work("Users should extend the workflow to new datasets.")


def test_v16_37_completed_action_is_not_future_work() -> None:
    assert not supports_future_work("We conducted a broader field evaluation.")


def test_v16_38_prior_paper_plan_is_not_current_future_work() -> None:
    assert not supports_future_work("Previous work plans to integrate additional datasets.")


def test_v16_39_plain_contribution_alone_is_not_why_it_matters() -> None:
    document, insights = _why_story(
        "The work introduces an indexed cache.",
        ("contribution",),
        include_problem=False,
        include_finding=False,
    )
    assert filter_synthesis_support(insights, document).why_it_matters == []


def test_v16_40_plain_capability_alone_is_not_why_it_matters() -> None:
    document, insights = _why_story(
        "The cache enables interactive monitoring.",
        ("contribution",),
        include_problem=False,
        include_finding=False,
    )
    assert filter_synthesis_support(insights, document).why_it_matters == []


def test_v16_41_problem_and_contribution_support_significance() -> None:
    document, insights = _why_story(
        "The indexed workflow addresses slow manual review for interactive monitoring.",
        ("problem", "contribution"),
        include_finding=False,
    )
    assert len(filter_synthesis_support(insights, document).why_it_matters) == 1


def test_v16_42_contribution_and_finding_support_significance() -> None:
    document, insights = _why_story(
        "The cache enables interactive monitoring with lower latency while preserving quality.",
        ("contribution", "finding"),
        include_problem=False,
    )
    assert len(filter_synthesis_support(insights, document).why_it_matters) == 1


def test_v16_43_three_story_components_support_significance() -> None:
    document, insights = _why_story(
        "The cache addresses slow review by enabling lower-latency interactive monitoring.",
        ("problem", "contribution", "finding"),
    )
    assert len(filter_synthesis_support(insights, document).why_it_matters) == 1


def test_v16_44_why_uses_minimum_multi_evidence_union() -> None:
    document, insights = _why_story(
        "The indexed workflow addresses slow manual review for interactive monitoring.",
        ("problem", "contribution", "finding"),
    )
    retained = filter_synthesis_support(insights, document).why_it_matters
    assert len(retained) == 1
    assert len(retained[0].evidence) == 2


def test_v16_45_unsupported_impact_is_rejected() -> None:
    document, insights = _why_story(
        "The cache guarantees safer medical decisions and lower costs.",
        ("problem", "contribution", "finding"),
    )
    assert filter_synthesis_support(insights, document).why_it_matters == []


def test_v16_46_causal_strengthening_is_rejected() -> None:
    document, insights = _why_story(
        "The cache will cause universally better decisions.",
        ("problem", "contribution", "finding"),
    )
    assert filter_synthesis_support(insights, document).why_it_matters == []


def test_v16_47_why_preserves_uncertainty() -> None:
    problem = "Manual review may be too slow for interactive monitoring."
    finding = "The evaluation suggests the cache may reduce monitoring latency."
    document = _document(
        ("problem", "Introduction", problem),
        ("finding", "Results", finding),
    )
    insights = InsightFields.empty()
    insights.research_problem = [
        InsightClaim(claim=problem, evidence=[{"chunk_id": "problem", "quote": problem}])
    ]
    insights.main_findings = [
        InsightClaim(claim=finding, evidence=[{"chunk_id": "finding", "quote": finding}])
    ]
    insights.why_it_matters = [
        InsightClaim(
            claim="The cache may reduce latency that constrains interactive monitoring.",
            evidence=[
                {"chunk_id": "problem", "quote": problem},
                {"chunk_id": "finding", "quote": finding},
            ],
        )
    ]
    assert len(filter_synthesis_support(insights, document).why_it_matters) == 1


def test_v16_48_one_upstream_component_leaves_why_empty() -> None:
    document, insights = _why_story(
        "The cache enables interactive monitoring.",
        ("contribution",),
        include_problem=False,
        include_finding=False,
    )
    assert filter_synthesis_support(insights, document).why_it_matters == []


def test_v16_49_but_exception_cannot_be_erased() -> None:
    evidence = "Overall runtime increased with load, but cache timing was not affected."
    claim = "Cache timing increased with load."
    assert not _explicit("main_findings", "Results", evidence, claim)


def test_v16_50_although_distinction_cannot_be_reversed() -> None:
    evidence = "Although Group A had lower accuracy, Group B had higher accuracy."
    claim = "Group A had higher accuracy than Group B."
    assert not _explicit("main_findings", "Results", evidence, claim)


def test_v16_51_does_not_affect_exception_is_preserved() -> None:
    evidence = "Memory pressure increased, but cache timing was not affected."
    claim = "Memory pressure increased cache timing."
    assert not _explicit("main_findings", "Results", evidence, claim)


def test_v16_52_incidental_detail_need_not_be_copied() -> None:
    evidence = "Using a blue display, the evaluation found that the workflow reduced latency."
    claim = "The workflow reduced latency."
    assert _explicit("main_findings", "Results", evidence, claim)


def test_v16_53_explicit_intended_user_role_is_accepted() -> None:
    item = InsightClaim(
        claim="The intended users are clinical analysts.",
        evidence=[{"chunk_id": "a", "quote": "The system is intended for clinical analysts."}],
    )
    assert synthesis_claim_supported(item, "target_audience")


def test_v16_54_domain_relevance_is_not_an_audience() -> None:
    item = InsightClaim(
        claim="The audience is clinical analysts.",
        evidence=[{"chunk_id": "a", "quote": "The topic is relevant to clinical analysts."}],
    )
    assert not synthesis_claim_supported(item, "target_audience")


def test_v16_55_related_work_users_are_not_current_audience() -> None:
    item = InsightClaim(
        claim="The intended users are logistics analysts.",
        evidence=[{"chunk_id": "a", "quote": "Prior work is used by logistics analysts."}],
    )
    assert not synthesis_claim_supported(item, "target_audience")


def test_v16_56_explicit_primary_target_users_are_accepted() -> None:
    item = InsightClaim(
        claim="The primary target users are operations engineers.",
        evidence=[{"chunk_id": "a", "quote": "Primary target users include operations engineers."}],
    )
    assert synthesis_claim_supported(item, "target_audience")


def test_v16_57_catalog_excerpt_ends_at_word_boundary() -> None:
    source = " ".join(f"descriptiveword{index}" for index in range(50))
    catalog = evidence_catalog(select_evidence(_document(("long", "Results", source))))
    quote = next(quote for chunk_id, quote, _heading in catalog.values() if chunk_id == "long")
    assert quote_in_chunk(quote, source)
    assert len(quote) == len(source) or source[len(quote)].isspace()


def test_v16_58_genuine_uncertainty_is_protected() -> None:
    evidence = "The workflow may reduce latency under stable inputs."
    assert not _explicit("main_findings", "Results", evidence, "The workflow reduces latency.")
    assert _explicit("main_findings", "Results", evidence, evidence)


def test_v16_59_associated_with_is_not_treated_as_epistemic_hedge() -> None:
    evidence = "The method has no error bound associated with approximation bias."
    assert _explicit("limitations", "Limitations", evidence, evidence)


def test_v16_60_diagnostics_are_optional_and_off_by_default() -> None:
    parameter = inspect.signature(InsightService.extract).parameters["diagnostics"]
    assert parameter.default is None
    assert isinstance(InsightDiagnostics(), InsightDiagnostics)


def test_v16_61_public_insight_response_schema_is_unchanged() -> None:
    assert set(InsightsResponse.model_json_schema()["properties"]) == {
        "paper_id",
        "document_fingerprint",
        "model",
        "extraction_version",
        "cached",
        "insights",
    }


@pytest.mark.asyncio
async def test_v16_62_three_call_architecture_is_unchanged(tmp_path) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        fields = payload["format"]["properties"]
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps({field: [] for field in fields})}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await InsightService(
            client,
            "http://localhost:11434",
            "qwen3:1.7b",
            InsightCache(tmp_path),
        ).extract(_document(("method", "Methods", "We use an indexed processing workflow.")))

    assert calls == 3
    assert response.extraction_version == EXTRACTION_VERSION


def test_v16_63_parser_metadata_round_trip_is_unchanged() -> None:
    document = _document(("method", "Methods", "We use an indexed processing workflow."))
    restored = ParsedPaper.model_validate(document.model_dump())
    assert restored.parser == "grobid"
    assert restored.parser_version == "0.9.1-crf+frontmatter-v1"
    assert set(INSIGHT_FIELDS) == set(InsightFields.model_fields)
