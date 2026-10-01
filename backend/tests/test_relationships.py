from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.db.database import create_session_factory, initialize_database
from app.llm import LLMProviderRegistry
from app.main import create_app
from app.models.document import (
    DocumentChunk,
    DocumentReference,
    DocumentSection,
    ParsedPaper,
    SourcePDF,
)
from app.models.insights import EvidenceReference, InsightClaim, InsightFields
from app.models.relationships import (
    RelationshipEvidence,
    RelationshipProposalRequest,
    RelationshipReviewRequest,
    RelationshipType,
)
from app.services.insight_cache import InsightCache
from app.services.insight_service import document_fingerprint
from app.services.paper_understanding_service import PIPELINE_VERSION
from app.services.parsed_document_cache import ParsedDocumentCache
from app.services.relationship_service import (
    RELATIONSHIP_PIPELINE_VERSION,
    RelationshipError,
    RelationshipPaperState,
    RelationshipService,
    RelationshipStateResolver,
    RelationshipStore,
)


def _document(paper_id: str, marker: str) -> ParsedPaper:
    text = f"Paper {marker} studies shared agent workflows with a grounded evaluation."
    chunk = DocumentChunk(
        id=f"{paper_id}-chunk",
        section_id=f"{paper_id}-section",
        text=text,
        start_char=0,
        end_char=len(text),
    )
    return ParsedPaper(
        paper_id=paper_id,
        title=f"Paper {marker}",
        authors=[f"Author {marker}"],
        abstract=text,
        sections=[
            DocumentSection(
                id=f"{paper_id}-section",
                heading="Evaluation",
                level=1,
                text=text,
                chunks=[chunk],
            )
        ],
        references=[DocumentReference(id=f"{paper_id}-ref", raw_text="UNUSED_BIBLIOGRAPHY_TEXT")],
        parser="grobid",
        parser_version="0.9.1",
        source_pdf=SourcePDF(
            acquisition_method="upload",
            sha256=marker.casefold() * 64,
            size_bytes=100,
        ),
    )


def _insights(document: ParsedPaper) -> InsightFields:
    chunk = document.sections[0].chunks[0]
    claim = InsightClaim(
        claim=f"{document.title} evaluates shared agent workflows.",
        evidence=[EvidenceReference(chunk_id=chunk.id, quote=chunk.text)],
    )
    return InsightFields(methods=[claim], evaluation=[claim], main_findings=[claim])


def _state(paper_id: str, marker: str) -> RelationshipPaperState:
    document = _document(paper_id, marker)
    insights = _insights(document)
    return RelationshipPaperState(
        paper_id=paper_id,
        document=document,
        document_fingerprint=document_fingerprint(document),
        insights=insights,
        insight_provider="ollama",
        insight_model="qwen3:1.7b",
        extraction_version=PIPELINE_VERSION,
        evidence_ids=frozenset({document.sections[0].chunks[0].id}),
        evidence=(
            RelationshipEvidence(
                paper_id=paper_id,
                evidence_id=document.sections[0].chunks[0].id,
                insight_field="methods",
                claim=insights.methods[0].claim,
                quote=insights.methods[0].evidence[0].quote,
                section_id=document.sections[0].id,
                section_heading="Evaluation",
            ),
        ),
    )


def _response(
    *,
    relationship_type: str = "shared_method",
    source_id: str = "paper-a-chunk",
    target_id: str = "paper-b-chunk",
    summary: str = "Both papers evaluate grounded agent workflows.",
) -> str:
    return json.dumps(
        {
            "relationships": [
                {
                    "relationship_type": relationship_type,
                    "summary": summary,
                    "source_evidence_ids": [source_id],
                    "target_evidence_ids": [target_id],
                }
            ]
        }
    )


class FakeProvider:
    provider_id = "ollama"
    model_id = "qwen3:1.7b"
    configured = True
    capabilities = SimpleNamespace(cloud=False)

    def __init__(self, response: str | None = None) -> None:
        self.response = response or _response()
        self.calls: list[dict[str, object]] = []

    async def generate_structured(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.response

    async def aclose(self) -> None:
        return None


class FakeResolver:
    def __init__(self, states: dict[str, RelationshipPaperState]) -> None:
        self.states = states

    def resolve(self, paper_id: str) -> RelationshipPaperState:
        state = self.states.get(paper_id)
        if state is None:
            raise RelationshipError("PAPER_NOT_FOUND", "not found", 404)
        return state


class FakeSemantics:
    def __init__(self, similarity: float = 0.8) -> None:
        self.similarity = similarity
        self.calls = 0

    def paper_similarity_matrix(self, papers):
        self.calls += 1
        assert [paper.id for paper in papers] == ["paper-a", "paper-b"]
        return np.asarray([[1.0, self.similarity], [self.similarity, 1.0]])


def _store(tmp_path) -> RelationshipStore:
    factory = create_session_factory(f"sqlite:///{tmp_path / 'relationships.sqlite3'}")
    initialize_database(factory)
    return RelationshipStore(factory)


def _service(tmp_path, provider: FakeProvider | None = None, states=None):
    provider = provider or FakeProvider()
    states = states or {"paper-a": _state("paper-a", "a"), "paper-b": _state("paper-b", "b")}
    semantics = FakeSemantics()
    service = RelationshipService(
        state_resolver=FakeResolver(states),
        registry=LLMProviderRegistry([provider]),
        semantics=semantics,
        store=_store(tmp_path),
    )
    return service, provider, semantics


def _propose(service: RelationshipService, source="paper-a", target="paper-b", types=None):
    return asyncio.run(
        service.propose(
            RelationshipProposalRequest(
                source_paper_id=source,
                target_paper_id=target,
                requested_relationship_types=types,
            )
        )
    )


def test_proposal_requires_two_distinct_papers(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    with pytest.raises(RelationshipError, match="two different") as exc:
        _propose(service, "paper-a", "paper-a")
    assert exc.value.code == "INVALID_RELATIONSHIP_FORMAT"


def test_pair_is_normalized_to_stable_canonical_order(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    proposal = _propose(service, "paper-b", "paper-a")
    assert (proposal.source_paper_id, proposal.target_paper_id) == ("paper-a", "paper-b")
    assert {item.paper_id for item in proposal.evidence_source} == {"paper-a"}
    assert {item.paper_id for item in proposal.evidence_target} == {"paper-b"}


def test_provider_receives_validated_insights_with_separate_namespaces(tmp_path) -> None:
    service, provider, _ = _service(tmp_path)
    _propose(service)
    prompt = str(provider.calls[0]["user_prompt"])
    assert "Paper A canonical_id=paper-a" in prompt
    assert "Paper B canonical_id=paper-b" in prompt
    assert "evidence_id=paper-a-chunk" in prompt
    assert "evidence_id=paper-b-chunk" in prompt
    assert "references" not in prompt.casefold()


def test_provider_uses_strict_relationship_schema(tmp_path) -> None:
    service, provider, _ = _service(tmp_path)
    _propose(service)
    assert provider.calls[0]["schema"].__name__ == "GeneratedRelationshipResponse"


def test_valid_evidence_from_both_papers_is_retained(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    item = _propose(service).relationships[0]
    assert item.source_evidence_ids == ["paper-a-chunk"]
    assert item.target_evidence_ids == ["paper-b-chunk"]


def test_wrong_source_evidence_is_rejected_while_supported_sibling_survives(tmp_path) -> None:
    payload = json.loads(_response())
    payload["relationships"].insert(
        0,
        {
            "relationship_type": "shared_method",
            "summary": "Wrong ownership.",
            "source_evidence_ids": ["paper-b-chunk"],
            "target_evidence_ids": ["paper-b-chunk"],
        },
    )
    service, _, _ = _service(tmp_path, FakeProvider(json.dumps(payload)))
    proposal = _propose(service)
    assert len(proposal.relationships) == 1
    assert proposal.diagnostics[0].reason == "unsupported_source_evidence_reference"


def test_wrong_target_evidence_is_rejected(tmp_path) -> None:
    service, _, _ = _service(tmp_path, FakeProvider(_response(target_id="missing")))
    with pytest.raises(RelationshipError) as exc:
        _propose(service)
    assert exc.value.code == "UNSUPPORTED_EVIDENCE_REFERENCE"


def test_all_unsupported_evidence_fails_clearly(tmp_path) -> None:
    service, _, _ = _service(
        tmp_path,
        FakeProvider(_response(source_id="missing-a", target_id="missing-b")),
    )
    with pytest.raises(RelationshipError, match="No evidence-grounded"):
        _propose(service)


def test_evidence_ids_without_claim_support_do_not_validate_relationship(tmp_path) -> None:
    service, _, _ = _service(
        tmp_path,
        FakeProvider(_response(summary="Both papers improve marine biology outcomes.")),
    )
    with pytest.raises(RelationshipError, match="No evidence-grounded") as exc:
        _propose(service)
    assert exc.value.code == "RELATIONSHIP_GENERATION_FAILED"


@pytest.mark.parametrize("raw", ["not-json", '{"relationships":'])
def test_malformed_provider_output_fails_cleanly(tmp_path, raw) -> None:
    service, _, _ = _service(tmp_path, FakeProvider(raw))
    with pytest.raises(RelationshipError) as exc:
        _propose(service)
    assert exc.value.code == "INVALID_RELATIONSHIP_FORMAT"


def test_invalid_relationship_enum_fails_structural_validation(tmp_path) -> None:
    service, _, _ = _service(tmp_path, FakeProvider(_response(relationship_type="invented")))
    with pytest.raises(RelationshipError) as exc:
        _propose(service)
    assert exc.value.code == "INVALID_RELATIONSHIP_FORMAT"


@pytest.mark.parametrize("missing_field", ["source_evidence_ids", "target_evidence_ids"])
def test_relationship_missing_bilateral_evidence_is_rejected(tmp_path, missing_field) -> None:
    payload = json.loads(_response())
    payload["relationships"][0].pop(missing_field)
    service, _, _ = _service(tmp_path, FakeProvider(json.dumps(payload)))
    with pytest.raises(RelationshipError) as exc:
        _propose(service)
    assert exc.value.code == "INVALID_RELATIONSHIP_FORMAT"


def test_confidence_is_deterministic_and_not_provider_supplied(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    item = _propose(service).relationships[0]
    assert item.confidence == 0.68


def test_requested_relationship_type_is_sent_and_retained(tmp_path) -> None:
    service, provider, _ = _service(tmp_path)
    proposal = _propose(service, types=[RelationshipType.SHARED_METHOD])
    assert proposal.relationship_types == [RelationshipType.SHARED_METHOD]
    assert "shared_method" in str(provider.calls[0]["user_prompt"])


def test_unrequested_relationship_type_is_rejected(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    with pytest.raises(RelationshipError) as exc:
        _propose(service, types=[RelationshipType.CONTRASTING_RESULT])
    assert exc.value.code == "RELATIONSHIP_GENERATION_FAILED"


def test_identical_proposal_request_is_cached(tmp_path) -> None:
    service, provider, semantics = _service(tmp_path)
    first = _propose(service)
    second = _propose(service)
    assert second.proposal_id == first.proposal_id
    assert second.cached is True
    assert len(provider.calls) == 1
    assert semantics.calls == 1


def test_reverse_pair_order_hits_same_cache_entry(tmp_path) -> None:
    service, provider, _ = _service(tmp_path)
    first = _propose(service, "paper-a", "paper-b")
    second = _propose(service, "paper-b", "paper-a")
    assert second.proposal_id == first.proposal_id
    assert len(provider.calls) == 1


def test_pair_lookup_returns_current_cached_proposal_without_model_call(tmp_path) -> None:
    service, provider, semantics = _service(tmp_path)
    proposal = _propose(service)
    provider.calls.clear()
    semantics.calls = 0

    found = asyncio.run(service.lookup_pair("paper-a", "paper-b"))

    assert found is not None
    assert found.proposal_id == proposal.proposal_id
    assert found.cached is True
    assert provider.calls == []
    assert semantics.calls == 0


def test_reverse_pair_lookup_uses_same_normalized_cache_entry(tmp_path) -> None:
    service, provider, _ = _service(tmp_path)
    proposal = _propose(service)
    provider.calls.clear()
    found = asyncio.run(service.lookup_pair("paper-b", "paper-a"))
    assert found is not None
    assert found.proposal_id == proposal.proposal_id
    assert provider.calls == []


def test_pair_lookup_returns_none_without_generating(tmp_path) -> None:
    service, provider, semantics = _service(tmp_path)
    assert asyncio.run(service.lookup_pair("paper-a", "paper-b")) is None
    assert provider.calls == []
    assert semantics.calls == 0


def test_pair_lookup_does_not_return_stale_fingerprint_entry(tmp_path) -> None:
    service, provider, _ = _service(tmp_path)
    _propose(service)
    provider.calls.clear()
    service.state_resolver.states["paper-a"] = _state("paper-a", "c")
    assert asyncio.run(service.lookup_pair("paper-a", "paper-b")) is None
    assert provider.calls == []


def test_pair_lookup_requires_current_provider_and_model_cache_key(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    _propose(service)
    changed_provider = FakeProvider()
    changed_provider.model_id = "new-model"
    service.registry = LLMProviderRegistry([changed_provider])
    assert asyncio.run(service.lookup_pair("paper-a", "paper-b")) is None
    assert changed_provider.calls == []


def test_changed_document_fingerprint_invalidates_cache(tmp_path) -> None:
    service, provider, _ = _service(tmp_path)
    first = _propose(service)
    changed = _state("paper-a", "c")
    changed = RelationshipPaperState(**{**changed.__dict__, "paper_id": "paper-a"})
    service.state_resolver.states["paper-a"] = changed
    provider.response = _response(source_id="paper-a-chunk")
    second = _propose(service)
    assert second.proposal_id != first.proposal_id
    assert len(provider.calls) == 2


def test_pipeline_version_is_recorded(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    assert _propose(service).relationship_pipeline_version == RELATIONSHIP_PIPELINE_VERSION


def test_missing_canonical_paper_is_structured_error(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    with pytest.raises(RelationshipError) as exc:
        _propose(service, "missing", "paper-b")
    assert exc.value.code == "PAPER_NOT_FOUND"


def test_missing_target_paper_is_structured_error(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    with pytest.raises(RelationshipError) as exc:
        _propose(service, "paper-a", "missing")
    assert exc.value.code == "PAPER_NOT_FOUND"


def test_state_resolver_reports_paper_not_found(tmp_path) -> None:
    resolver = RelationshipStateResolver(tmp_path / "parsed", InsightCache(tmp_path / "insights"))
    with pytest.raises(RelationshipError) as exc:
        resolver.resolve("missing")
    assert exc.value.code == "PAPER_NOT_FOUND"


def test_state_resolver_reports_parsed_paper_missing(tmp_path) -> None:
    document = _document("paper-a", "a")
    insight_cache = InsightCache(tmp_path / "insights")
    insight_cache.put_current(
        paper_id="paper-a",
        document_fingerprint=document_fingerprint(document),
        provider_id="ollama",
        model="qwen3:1.7b",
        extraction_version=PIPELINE_VERSION,
        insights=_insights(document),
    )
    resolver = RelationshipStateResolver(tmp_path / "parsed", insight_cache)
    with pytest.raises(RelationshipError) as exc:
        resolver.resolve("paper-a")
    assert exc.value.code == "PARSED_PAPER_MISSING"


@pytest.mark.parametrize("missing_id", ["paper-a", "paper-b"])
def test_service_preserves_parsed_missing_error_for_either_endpoint(tmp_path, missing_id) -> None:
    class ParsedMissingResolver(FakeResolver):
        def resolve(self, paper_id):
            if paper_id == missing_id:
                raise RelationshipError("PARSED_PAPER_MISSING", f"missing {paper_id}", 409)
            return super().resolve(paper_id)

    service, _, _ = _service(tmp_path)
    service.state_resolver = ParsedMissingResolver(
        {"paper-a": _state("paper-a", "a"), "paper-b": _state("paper-b", "b")}
    )
    with pytest.raises(RelationshipError) as exc:
        _propose(service)
    assert exc.value.code == "PARSED_PAPER_MISSING"


def test_state_resolver_reports_current_analysis_missing(tmp_path) -> None:
    document = _document("paper-a", "a")
    ParsedDocumentCache(tmp_path / "parsed").put_paper_state(document)
    resolver = RelationshipStateResolver(tmp_path / "parsed", InsightCache(tmp_path / "insights"))
    with pytest.raises(RelationshipError) as exc:
        resolver.resolve("paper-a")
    assert exc.value.code == "CURRENT_ANALYSIS_MISSING"


@pytest.mark.parametrize(
    ("fingerprint", "version"),
    [("f" * 64, PIPELINE_VERSION), (None, "old-pipeline")],
)
def test_state_resolver_reports_stale_analysis(tmp_path, fingerprint, version) -> None:
    document = _document("paper-a", "a")
    ParsedDocumentCache(tmp_path / "parsed").put_paper_state(document)
    insight_cache = InsightCache(tmp_path / "insights")
    insight_cache.put_current(
        paper_id="paper-a",
        document_fingerprint=fingerprint or document_fingerprint(document),
        provider_id="ollama",
        model="qwen3:1.7b",
        extraction_version=version,
        insights=_insights(document),
    )
    with pytest.raises(RelationshipError) as exc:
        RelationshipStateResolver(tmp_path / "parsed", insight_cache).resolve("paper-a")
    assert exc.value.code == "STALE_ANALYSIS"


def test_state_resolver_rejects_noncurrent_provider_model_version(tmp_path) -> None:
    document = _document("paper-a", "a")
    ParsedDocumentCache(tmp_path / "parsed").put_paper_state(document)
    insight_cache = InsightCache(tmp_path / "insights")
    insight_cache.put_current(
        paper_id="paper-a",
        document_fingerprint=document_fingerprint(document),
        provider_id="ollama",
        model="old-model",
        extraction_version=PIPELINE_VERSION,
        insights=_insights(document),
    )
    registry = SimpleNamespace(
        statuses=lambda: [
            {
                "provider_id": "ollama",
                "model": "qwen3:1.7b",
                "configured": True,
            }
        ]
    )
    resolver = RelationshipStateResolver(tmp_path / "parsed", insight_cache, registry)
    with pytest.raises(RelationshipError) as exc:
        resolver.resolve("paper-a")
    assert exc.value.code == "STALE_ANALYSIS"


@pytest.mark.parametrize("stale_id", ["paper-a", "paper-b"])
def test_service_preserves_stale_error_for_either_endpoint(tmp_path, stale_id) -> None:
    class StaleResolver(FakeResolver):
        def resolve(self, paper_id):
            if paper_id == stale_id:
                raise RelationshipError("STALE_ANALYSIS", f"stale {paper_id}", 409)
            return super().resolve(paper_id)

    service, _, _ = _service(tmp_path)
    service.state_resolver = StaleResolver(
        {"paper-a": _state("paper-a", "a"), "paper-b": _state("paper-b", "b")}
    )
    with pytest.raises(RelationshipError) as exc:
        _propose(service)
    assert exc.value.code == "STALE_ANALYSIS"


def test_missing_proposal_is_structured_error(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    with pytest.raises(RelationshipError) as exc:
        asyncio.run(service.get("missing"))
    assert exc.value.code == "PROPOSAL_NOT_FOUND"


@pytest.mark.parametrize("decision", ["accepted", "rejected"])
def test_accept_and_reject_reviews_preserve_original_proposal(tmp_path, decision) -> None:
    service, _, _ = _service(tmp_path)
    proposal = _propose(service)
    review = asyncio.run(
        service.review(proposal.proposal_id, RelationshipReviewRequest(decision=decision))
    )
    assert review.original_summary == proposal.summary
    assert asyncio.run(service.get(proposal.proposal_id)).summary == proposal.summary


def test_edited_review_changes_only_types_and_summary(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    proposal = _propose(service)
    review = asyncio.run(
        service.review(
            proposal.proposal_id,
            RelationshipReviewRequest(
                decision="edited",
                edited_relationship_types=[RelationshipType.RELATED_APPLICATION],
                edited_summary="The papers address related application settings.",
                reviewer_note="Human clarification",
            ),
        )
    )
    assert review.edited_relationship_types == [RelationshipType.RELATED_APPLICATION]
    assert review.edited_summary == "The papers address related application settings."
    assert not hasattr(review, "source_evidence_ids")


def test_edited_review_requires_an_edit() -> None:
    with pytest.raises(ValidationError):
        RelationshipReviewRequest(decision="edited")


@pytest.mark.parametrize("field", ["edited_summary", "reviewer_note"])
def test_review_rejects_html_injection(field) -> None:
    payload = {"decision": "edited", "edited_summary": "Safe edit"}
    payload[field] = "<script>alert(1)</script>"
    with pytest.raises(ValidationError, match="HTML"):
        RelationshipReviewRequest(**payload)


def test_double_submit_returns_same_review(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    proposal = _propose(service)
    request = RelationshipReviewRequest(decision="accepted", reviewer_note="Reviewed")
    first = asyncio.run(service.review(proposal.proposal_id, request))
    second = asyncio.run(service.review(proposal.proposal_id, request))
    assert second.review_id == first.review_id
    assert second.review_version == 1


def test_review_history_is_append_only_and_versioned(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    proposal = _propose(service)
    asyncio.run(
        service.review(proposal.proposal_id, RelationshipReviewRequest(decision="accepted"))
    )
    asyncio.run(
        service.review(
            proposal.proposal_id,
            RelationshipReviewRequest(decision="rejected", reviewer_note="Second review"),
        )
    )
    history = asyncio.run(service.history(proposal.proposal_id))
    assert history.proposal.proposal_id == proposal.proposal_id
    assert [review.review_version for review in history.reviews] == [1, 2]
    assert [review.decision for review in history.reviews] == ["accepted", "rejected"]


def test_proposal_and_reviews_survive_store_restart(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    proposal = _propose(service)
    review = asyncio.run(
        service.review(proposal.proposal_id, RelationshipReviewRequest(decision="accepted"))
    )
    restarted = RelationshipStore(
        create_session_factory(f"sqlite:///{tmp_path / 'relationships.sqlite3'}")
    )
    assert restarted.get_proposal(proposal.proposal_id) is not None
    assert restarted.review_history(proposal.proposal_id)[0].review_id == review.review_id


def test_persisted_proposal_does_not_duplicate_full_parsed_document(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    proposal = _propose(service)
    database = tmp_path / "relationships.sqlite3"
    assert proposal.proposal_id.encode() in database.read_bytes()
    assert b"UNUSED_BIBLIOGRAPHY_TEXT" not in database.read_bytes()


def test_proposal_retains_compact_bilateral_evidence_provenance(tmp_path) -> None:
    service, _, _ = _service(tmp_path)
    proposal = _propose(service)
    assert proposal.evidence_source[0].paper_id == "paper-a"
    assert proposal.evidence_source[0].section_heading == "Evaluation"
    assert proposal.evidence_target[0].paper_id == "paper-b"
    assert proposal.evidence_target[0].evidence_id == "paper-b-chunk"


def test_duplicate_generated_relationship_is_deduplicated(tmp_path) -> None:
    payload = json.loads(_response())
    payload["relationships"].append(payload["relationships"][0])
    service, _, _ = _service(tmp_path, FakeProvider(json.dumps(payload)))
    proposal = _propose(service)
    assert len(proposal.relationships) == 1
    assert proposal.diagnostics[0].reason == "duplicate_relationship"


def test_api_propose_get_review_and_history_use_persistent_service(tmp_path) -> None:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}",
        parsed_document_cache_dir=str(tmp_path / "parsed"),
        insight_cache_dir=str(tmp_path / "insights"),
    )
    app = create_app(settings)
    service, _, _ = _service(tmp_path / "service")
    with TestClient(app) as client:
        app.state.relationship_service = service
        proposed = client.post(
            "/api/relationships/propose",
            json={"source_paper_id": "paper-a", "target_paper_id": "paper-b"},
        )
        assert proposed.status_code == 200
        proposal_id = proposed.json()["proposal_id"]
        assert client.get(f"/api/relationships/{proposal_id}").status_code == 200
        reviewed = client.post(
            f"/api/relationships/{proposal_id}/review", json={"decision": "accepted"}
        )
        assert reviewed.status_code == 200
        history = client.get(f"/api/relationships/{proposal_id}/history")
        assert history.json()["reviews"][0]["decision"] == "accepted"


def test_api_pair_lookup_returns_cached_or_null_without_generation(tmp_path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}")
    app = create_app(settings)
    service, provider, _ = _service(tmp_path / "service")
    with TestClient(app) as client:
        app.state.relationship_service = service
        absent = client.get(
            "/api/relationships/by-pair",
            params={"source_paper_id": "paper-a", "target_paper_id": "paper-b"},
        )
        assert absent.status_code == 200
        assert absent.json() is None
        assert absent.headers["cache-control"] == "no-store"
        proposal = _propose(service)
        provider.calls.clear()
        found = client.get(
            "/api/relationships/by-pair",
            params={"source_paper_id": "paper-b", "target_paper_id": "paper-a"},
        )
    assert found.status_code == 200
    assert found.json()["proposal_id"] == proposal.proposal_id
    assert provider.calls == []


def test_api_returns_stable_structured_error(tmp_path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}")
    app = create_app(settings)
    service, _, _ = _service(tmp_path / "service")
    with TestClient(app) as client:
        app.state.relationship_service = service
        response = client.get("/api/relationships/missing")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "PROPOSAL_NOT_FOUND"


def test_api_invalid_review_uses_stable_error_code(tmp_path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}")
    app = create_app(settings)
    service, _, _ = _service(tmp_path / "service")
    proposal = _propose(service)
    with TestClient(app) as client:
        app.state.relationship_service = service
        response = client.post(
            f"/api/relationships/{proposal.proposal_id}/review",
            json={"decision": "edited"},
        )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_REVIEW"
