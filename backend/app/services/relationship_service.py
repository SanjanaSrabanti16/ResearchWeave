from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.tables import RelationshipProposalEntry, RelationshipReviewEntry
from app.llm import LLMProvider, LLMProviderError, LLMProviderRegistry
from app.models.document import ParsedPaper
from app.models.insights import InsightFields
from app.models.paper import Paper
from app.models.relationships import (
    AnalysisProvenance,
    GeneratedRelationshipResponse,
    RelationshipDiagnostic,
    RelationshipEvidence,
    RelationshipItem,
    RelationshipProposal,
    RelationshipProposalRequest,
    RelationshipReview,
    RelationshipReviewHistory,
    RelationshipReviewRequest,
    RelationshipType,
)
from app.services.graph_semantics import GraphSemanticsService, GraphSemanticsUnavailableError
from app.services.insight_cache import CurrentInsightState, InsightCache
from app.services.insight_markdown import validate_markdown_insights
from app.services.insight_service import document_fingerprint
from app.services.paper_understanding_service import PIPELINE_VERSION
from app.services.parsed_document_cache import ParsedDocumentCache

RELATIONSHIP_PIPELINE_VERSION = "m4.1-v1"
RELATIONSHIP_MAX_OUTPUT_TOKENS = 4096
logger = logging.getLogger(__name__)
_SUPPORT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "both",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "paper",
        "papers",
        "study",
        "studies",
        "that",
        "the",
        "their",
        "these",
        "this",
        "to",
        "using",
        "with",
    }
)

_SYSTEM_PROMPT = """You compare two scholarly papers using only the supplied, already-validated
paper insights and their evidence excerpts. Return strict JSON matching the schema. Every
relationship item must cite one or more evidence IDs from Paper A and one or more evidence IDs
from Paper B. Never use outside knowledge. Never invent an evidence ID. Describe only a direct,
specific relationship supported by both cited evidence sets. Keep summaries factual and concise.
Do not include HTML."""


class RelationshipError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class RelationshipPaperState:
    paper_id: str
    document: ParsedPaper
    document_fingerprint: str
    insights: InsightFields
    insight_provider: str
    insight_model: str
    extraction_version: str
    evidence_ids: frozenset[str]
    evidence: tuple[RelationshipEvidence, ...]

    @property
    def provenance(self) -> AnalysisProvenance:
        return AnalysisProvenance(
            document_fingerprint=self.document_fingerprint,
            provider=self.insight_provider,
            model=self.insight_model,
            extraction_version=self.extraction_version,
        )


def _all_document_chunks(document: ParsedPaper) -> dict[str, str]:
    chunks = {chunk.id: chunk.text for section in document.sections for chunk in section.chunks}
    chunks.update({chunk.id: chunk.text for chunk in document.abstract_chunks})
    return chunks


def _insight_evidence_ids(insights: InsightFields) -> frozenset[str]:
    return frozenset(
        evidence.chunk_id
        for field_name in type(insights).model_fields
        for claim in getattr(insights, field_name)
        for evidence in claim.evidence
    )


def _relationship_evidence(
    document: ParsedPaper,
    insights: InsightFields,
) -> tuple[RelationshipEvidence, ...]:
    chunks = {
        chunk.id: (chunk, section.heading)
        for section in document.sections
        for chunk in section.chunks
    }
    chunks.update({chunk.id: (chunk, "Abstract") for chunk in document.abstract_chunks})
    evidence_items: list[RelationshipEvidence] = []
    seen: set[tuple[str, str, str]] = set()
    for field_name in type(insights).model_fields:
        for claim in getattr(insights, field_name):
            for evidence in claim.evidence:
                key = (evidence.chunk_id, claim.claim, evidence.quote)
                if key in seen:
                    continue
                seen.add(key)
                chunk_info = chunks.get(evidence.chunk_id)
                chunk, heading = chunk_info if chunk_info is not None else (None, None)
                evidence_items.append(
                    RelationshipEvidence(
                        paper_id=document.paper_id,
                        evidence_id=evidence.chunk_id,
                        insight_field=field_name,
                        claim=claim.claim,
                        quote=evidence.quote,
                        section_id=chunk.section_id if chunk is not None else None,
                        section_heading=heading,
                        page_start=chunk.page_start if chunk is not None else None,
                        page_end=chunk.page_end if chunk is not None else None,
                    )
                )
    return tuple(evidence_items)


class RelationshipStateResolver:
    """Resolve only backend-authoritative parsed and current M2 analysis state."""

    def __init__(
        self,
        parsed_document_directory: str,
        insight_cache: InsightCache,
        registry: LLMProviderRegistry | None = None,
    ) -> None:
        self.parsed_cache = ParsedDocumentCache(parsed_document_directory)
        self.insight_cache = insight_cache
        self.registry = registry

    def resolve(self, paper_id: str) -> RelationshipPaperState:
        document = self.parsed_cache.resolve_paper_states({paper_id}).get(paper_id)
        latest = self.insight_cache.get_latest_current(paper_id)
        if document is None:
            code = "PARSED_PAPER_MISSING" if latest is not None else "PAPER_NOT_FOUND"
            message = (
                f"Paper {paper_id} exists, but no parsed paper is available"
                if latest is not None
                else f"Paper {paper_id} was not found in backend state"
            )
            raise RelationshipError(code, message, 409 if latest is not None else 404)
        fingerprint = document_fingerprint(document)
        if latest is None:
            raise RelationshipError(
                "CURRENT_ANALYSIS_MISSING",
                f"Paper {paper_id} has no current evidence-grounded M2 analysis",
                409,
            )
        self._require_current(latest, fingerprint, self.registry)
        validated = validate_markdown_insights(latest.insights, document)
        if validated != latest.insights:
            raise RelationshipError(
                "STALE_ANALYSIS",
                "The current paper analysis no longer validates against the parsed paper",
                409,
            )
        evidence_ids = _insight_evidence_ids(validated)
        if not evidence_ids:
            raise RelationshipError(
                "CURRENT_ANALYSIS_MISSING",
                f"Paper {paper_id}'s current M2 analysis contains no validated evidence",
                409,
            )
        return RelationshipPaperState(
            paper_id=paper_id,
            document=document,
            document_fingerprint=fingerprint,
            insights=validated,
            insight_provider=latest.provider_id,
            insight_model=latest.model,
            extraction_version=latest.extraction_version,
            evidence_ids=evidence_ids,
            evidence=_relationship_evidence(document, validated),
        )

    @staticmethod
    def _require_current(
        state: CurrentInsightState,
        fingerprint: str,
        registry: LLMProviderRegistry | None,
    ) -> None:
        if (
            state.document_fingerprint != fingerprint
            or state.extraction_version != PIPELINE_VERSION
        ):
            raise RelationshipError(
                "STALE_ANALYSIS",
                "The stored M2 analysis does not match the current document or pipeline",
                409,
            )
        if registry is not None:
            configured_versions = {
                (str(status["provider_id"]), str(status["model"]))
                for status in registry.statuses()
                if bool(status["configured"])
            }
            if (state.provider_id, state.model) not in configured_versions:
                raise RelationshipError(
                    "STALE_ANALYSIS",
                    "The stored M2 analysis provider/model is not current",
                    409,
                )


class RelationshipValidator:
    """Deterministic ownership, provenance, and support checks for generated relationships."""

    @staticmethod
    def validate_pair(source_paper_id: str, target_paper_id: str) -> None:
        if source_paper_id == target_paper_id:
            raise RelationshipError(
                "INVALID_RELATIONSHIP_FORMAT",
                "A relationship requires two different canonical papers",
                422,
            )

    @staticmethod
    def confidence(
        semantic_similarity: float,
        source_evidence_count: int,
        target_evidence_count: int,
    ) -> float:
        evidence_support = min(1.0, min(source_evidence_count, target_evidence_count) / 2)
        return round((0.6 * semantic_similarity) + (0.4 * evidence_support), 6)

    def validate_generated(
        self,
        generated: GeneratedRelationshipResponse,
        *,
        source: RelationshipPaperState,
        target: RelationshipPaperState,
        semantic_similarity: float,
        requested_types: frozenset[RelationshipType] | None,
    ) -> tuple[list[RelationshipItem], list[RelationshipDiagnostic]]:
        retained: list[RelationshipItem] = []
        diagnostics: list[RelationshipDiagnostic] = []
        seen: set[tuple[object, ...]] = set()
        for index, item in enumerate(generated.relationships):
            reason = self._unsupported_reason(item, source, target, requested_types)
            if reason is not None:
                diagnostics.append(RelationshipDiagnostic(item_index=index, reason=reason))
                continue
            source_ids = list(dict.fromkeys(item.source_evidence_ids))
            target_ids = list(dict.fromkeys(item.target_evidence_ids))
            signature = (
                item.relationship_type,
                re.sub(r"\s+", " ", item.summary).strip().casefold(),
                tuple(sorted(source_ids)),
                tuple(sorted(target_ids)),
            )
            if signature in seen:
                diagnostics.append(
                    RelationshipDiagnostic(item_index=index, reason="duplicate_relationship")
                )
                continue
            seen.add(signature)
            retained.append(
                RelationshipItem(
                    **item.model_dump(exclude={"source_evidence_ids", "target_evidence_ids"}),
                    source_evidence_ids=source_ids,
                    target_evidence_ids=target_ids,
                    confidence=self.confidence(
                        semantic_similarity,
                        len(source_ids),
                        len(target_ids),
                    ),
                )
            )
        return retained, diagnostics

    @staticmethod
    def _unsupported_reason(
        item: Any,
        source: RelationshipPaperState,
        target: RelationshipPaperState,
        requested_types: frozenset[RelationshipType] | None,
    ) -> str | None:
        if requested_types is not None and item.relationship_type not in requested_types:
            return "relationship_type_not_requested"
        if not set(item.source_evidence_ids).issubset(source.evidence_ids):
            return "unsupported_source_evidence_reference"
        if not set(item.target_evidence_ids).issubset(target.evidence_ids):
            return "unsupported_target_evidence_reference"
        summary_terms = RelationshipValidator._support_terms(item.summary)
        source_terms = RelationshipValidator._evidence_terms(source, item.source_evidence_ids)
        target_terms = RelationshipValidator._evidence_terms(target, item.target_evidence_ids)
        if not summary_terms.intersection(source_terms):
            return "summary_not_supported_by_source_evidence"
        if not summary_terms.intersection(target_terms):
            return "summary_not_supported_by_target_evidence"
        supported_terms = summary_terms.intersection(source_terms.union(target_terms))
        if summary_terms and len(supported_terms) / len(summary_terms) < 0.5:
            return "summary_contains_unsupported_concepts"
        return None

    @staticmethod
    def _support_terms(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.casefold())
            if len(token) >= 4 and token not in _SUPPORT_STOPWORDS
        }

    @classmethod
    def _evidence_terms(
        cls,
        state: RelationshipPaperState,
        evidence_ids: list[str],
    ) -> set[str]:
        selected = set(evidence_ids)
        return cls._support_terms(
            " ".join(
                f"{evidence.claim} {evidence.quote}"
                for evidence in state.evidence
                if evidence.evidence_id in selected
            )
        )


class RelationshipStore:
    """Persistent proposal and append-only review storage without duplicating paper text."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def get_by_cache_key(self, cache_key: str) -> RelationshipProposal | None:
        with self.session_factory() as session:
            entry = session.scalar(
                select(RelationshipProposalEntry).where(
                    RelationshipProposalEntry.cache_key == cache_key
                )
            )
            return self._proposal(entry, cached=True) if entry is not None else None

    def get_proposal(self, proposal_id: str) -> RelationshipProposal | None:
        with self.session_factory() as session:
            entry = session.get(RelationshipProposalEntry, proposal_id)
            return self._proposal(entry, cached=False) if entry is not None else None

    def put_proposal(self, cache_key: str, proposal: RelationshipProposal) -> RelationshipProposal:
        payload = proposal.model_copy(update={"cached": False}).model_dump_json()
        entry = RelationshipProposalEntry(
            proposal_id=proposal.proposal_id,
            cache_key=cache_key,
            source_paper_id=proposal.source_paper_id,
            target_paper_id=proposal.target_paper_id,
            payload=payload,
            created_at=proposal.created_at.replace(tzinfo=None),
        )
        with self.session_factory() as session:
            session.add(entry)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                existing = session.scalar(
                    select(RelationshipProposalEntry).where(
                        RelationshipProposalEntry.cache_key == cache_key
                    )
                )
                if existing is not None:
                    return self._proposal(existing, cached=True)
                raise
        return proposal

    def put_review(
        self,
        proposal: RelationshipProposal,
        request: RelationshipReviewRequest,
    ) -> RelationshipReview:
        request_json = json.dumps(request.model_dump(mode="json"), sort_keys=True)
        request_hash = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        with self.session_factory() as session:
            existing = session.scalar(
                select(RelationshipReviewEntry).where(
                    RelationshipReviewEntry.proposal_id == proposal.proposal_id,
                    RelationshipReviewEntry.request_hash == request_hash,
                )
            )
            if existing is not None:
                return RelationshipReview.model_validate_json(existing.payload)
            latest_version = session.scalar(
                select(func.max(RelationshipReviewEntry.review_version)).where(
                    RelationshipReviewEntry.proposal_id == proposal.proposal_id
                )
            )
            created_at = datetime.now(UTC)
            review = RelationshipReview(
                review_id=str(uuid.uuid4()),
                proposal_id=proposal.proposal_id,
                review_version=(latest_version or 0) + 1,
                decision=request.decision,
                original_relationship_types=proposal.relationship_types,
                original_summary=proposal.summary,
                relationship_pipeline_version=proposal.relationship_pipeline_version,
                edited_relationship_types=request.edited_relationship_types,
                edited_summary=request.edited_summary,
                reviewer_note=request.reviewer_note,
                created_at=created_at,
            )
            session.add(
                RelationshipReviewEntry(
                    review_id=review.review_id,
                    proposal_id=proposal.proposal_id,
                    request_hash=request_hash,
                    review_version=review.review_version,
                    payload=review.model_dump_json(),
                    created_at=created_at.replace(tzinfo=None),
                )
            )
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                concurrent = session.scalar(
                    select(RelationshipReviewEntry).where(
                        RelationshipReviewEntry.proposal_id == proposal.proposal_id,
                        RelationshipReviewEntry.request_hash == request_hash,
                    )
                )
                if concurrent is not None:
                    return RelationshipReview.model_validate_json(concurrent.payload)
                raise
            return review

    def review_history(self, proposal_id: str) -> list[RelationshipReview]:
        with self.session_factory() as session:
            entries = session.scalars(
                select(RelationshipReviewEntry)
                .where(RelationshipReviewEntry.proposal_id == proposal_id)
                .order_by(
                    RelationshipReviewEntry.review_version,
                    RelationshipReviewEntry.created_at,
                )
            ).all()
            return [RelationshipReview.model_validate_json(entry.payload) for entry in entries]

    @staticmethod
    def _proposal(entry: RelationshipProposalEntry, *, cached: bool) -> RelationshipProposal:
        proposal = RelationshipProposal.model_validate_json(entry.payload)
        return proposal.model_copy(update={"cached": cached})


def _relationship_context(state: RelationshipPaperState, label: str) -> str:
    chunks = _all_document_chunks(state.document)
    lines = [
        f"Paper {label} canonical_id={state.paper_id}",
        f"Paper {label} title={state.document.title or '[untitled]'}",
        f"Paper {label} document_fingerprint={state.document_fingerprint}",
    ]
    for field_name in type(state.insights).model_fields:
        claims = getattr(state.insights, field_name)
        for claim_index, claim in enumerate(claims, start=1):
            lines.append(f"{field_name}[{claim_index}] claim={claim.claim}")
            for evidence in claim.evidence:
                text = chunks.get(evidence.chunk_id)
                if text is None:
                    continue
                lines.append(
                    f"  evidence_id={evidence.chunk_id} quote={json.dumps(evidence.quote)}"
                )
    return "\n".join(lines)


def _user_prompt(
    source: RelationshipPaperState,
    target: RelationshipPaperState,
    requested_types: frozenset[RelationshipType] | None,
) -> str:
    allowed = requested_types or frozenset(RelationshipType)
    type_text = ", ".join(sorted(item.value for item in allowed))
    return (
        f"Allowed relationship_type values for this request: {type_text}.\n"
        "Paper A and Paper B are strictly separate evidence namespaces. Put only Paper A IDs in "
        "source_evidence_ids and only Paper B IDs in target_evidence_ids. Each relationship must "
        "be supported by both papers. Do not infer a relationship from titles alone.\n\n"
        f"{_relationship_context(source, 'A')}\n\n{_relationship_context(target, 'B')}"
    )


def _cache_key(
    source: RelationshipPaperState,
    target: RelationshipPaperState,
    provider: LLMProvider,
    requested_types: frozenset[RelationshipType] | None,
) -> str:
    payload = {
        "source_paper_id": source.paper_id,
        "target_paper_id": target.paper_id,
        "source_fingerprint": source.document_fingerprint,
        "target_fingerprint": target.document_fingerprint,
        "pipeline_version": RELATIONSHIP_PIPELINE_VERSION,
        "provider": provider.provider_id,
        "model": provider.model_id,
        "requested_types": sorted(item.value for item in requested_types or []),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _proposal_summary(items: list[RelationshipItem]) -> str:
    parts: list[str] = []
    size = 0
    for item in items:
        addition = item.summary.strip()
        separator = " " if parts else ""
        if size + len(separator) + len(addition) > 2400:
            break
        parts.append(addition)
        size += len(separator) + len(addition)
    return " ".join(parts)


class RelationshipService:
    def __init__(
        self,
        *,
        state_resolver: RelationshipStateResolver,
        registry: LLMProviderRegistry,
        semantics: GraphSemanticsService,
        store: RelationshipStore,
        validator: RelationshipValidator | None = None,
    ) -> None:
        self.state_resolver = state_resolver
        self.registry = registry
        self.semantics = semantics
        self.store = store
        self.validator = validator or RelationshipValidator()

    async def propose(self, request: RelationshipProposalRequest) -> RelationshipProposal:
        self.validator.validate_pair(request.source_paper_id, request.target_paper_id)
        source_id, target_id = sorted((request.source_paper_id, request.target_paper_id))
        source, target = await asyncio.gather(
            asyncio.to_thread(self.state_resolver.resolve, source_id),
            asyncio.to_thread(self.state_resolver.resolve, target_id),
        )
        try:
            provider = self.registry.get()
        except LLMProviderError as exc:
            raise RelationshipError("RELATIONSHIP_GENERATION_FAILED", str(exc), 503) from exc
        requested_types = (
            frozenset(request.requested_relationship_types)
            if request.requested_relationship_types is not None
            else None
        )
        cache_key = _cache_key(source, target, provider, requested_types)
        cached = await asyncio.to_thread(self.store.get_by_cache_key, cache_key)
        if cached is not None:
            return cached
        try:
            semantic_similarity = await asyncio.to_thread(
                self._semantic_similarity, source.document, target.document
            )
        except GraphSemanticsUnavailableError as exc:
            logger.warning("Relationship similarity failed: %s", type(exc).__name__)
            raise RelationshipError(
                "RELATIONSHIP_GENERATION_FAILED",
                "Paper similarity could not be computed",
                503,
            ) from exc
        try:
            raw = await provider.generate_structured(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_user_prompt(source, target, requested_types),
                schema=GeneratedRelationshipResponse,
                max_output_tokens=RELATIONSHIP_MAX_OUTPUT_TOKENS,
            )
        except LLMProviderError as exc:
            logger.warning(
                "Relationship generation failed for provider=%s model=%s: %s",
                provider.provider_id,
                provider.model_id,
                type(exc).__name__,
            )
            raise RelationshipError(
                "RELATIONSHIP_GENERATION_FAILED",
                "The relationship provider could not generate a response",
                502,
            ) from exc
        try:
            generated = GeneratedRelationshipResponse.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "Relationship output validation failed for provider=%s model=%s: %s",
                provider.provider_id,
                provider.model_id,
                type(exc).__name__,
            )
            raise RelationshipError(
                "INVALID_RELATIONSHIP_FORMAT",
                "The relationship provider returned invalid structured output",
                502,
            ) from exc
        items, diagnostics = self.validator.validate_generated(
            generated,
            source=source,
            target=target,
            semantic_similarity=semantic_similarity,
            requested_types=requested_types,
        )
        if not items:
            code = (
                "UNSUPPORTED_EVIDENCE_REFERENCE"
                if any("evidence_reference" in item.reason for item in diagnostics)
                else "RELATIONSHIP_GENERATION_FAILED"
            )
            raise RelationshipError(
                code,
                "No evidence-grounded relationship items survived validation",
                502,
            )
        if diagnostics:
            logger.info(
                "Relationship validation retained=%d rejected=%d reasons=%s",
                len(items),
                len(diagnostics),
                sorted({item.reason for item in diagnostics}),
            )
        relationship_types = list(dict.fromkeys(item.relationship_type for item in items))
        proposal = RelationshipProposal(
            proposal_id=str(uuid.uuid4()),
            source_paper_id=source.paper_id,
            target_paper_id=target.paper_id,
            semantic_similarity=semantic_similarity,
            relationship_types=relationship_types,
            relationships=items,
            summary=_proposal_summary(items),
            evidence_source=self._used_evidence(source, items, source_side=True),
            evidence_target=self._used_evidence(target, items, source_side=False),
            confidence=round(sum(item.confidence for item in items) / len(items), 6),
            source_analysis=source.provenance,
            target_analysis=target.provenance,
            relationship_provider=provider.provider_id,
            relationship_model=provider.model_id,
            relationship_pipeline_version=RELATIONSHIP_PIPELINE_VERSION,
            created_at=datetime.now(UTC),
            diagnostics=diagnostics,
        )
        return await asyncio.to_thread(self.store.put_proposal, cache_key, proposal)

    async def lookup_pair(
        self,
        source_paper_id: str,
        target_paper_id: str,
    ) -> RelationshipProposal | None:
        """Read a current cached pair proposal without invoking similarity or an LLM."""
        self.validator.validate_pair(source_paper_id, target_paper_id)
        source_id, target_id = sorted((source_paper_id, target_paper_id))
        source, target = await asyncio.gather(
            asyncio.to_thread(self.state_resolver.resolve, source_id),
            asyncio.to_thread(self.state_resolver.resolve, target_id),
        )
        try:
            provider = self.registry.get()
        except LLMProviderError as exc:
            raise RelationshipError("RELATIONSHIP_GENERATION_FAILED", str(exc), 503) from exc
        cache_key = _cache_key(source, target, provider, requested_types=None)
        return await asyncio.to_thread(self.store.get_by_cache_key, cache_key)

    async def get(self, proposal_id: str) -> RelationshipProposal:
        proposal = await asyncio.to_thread(self.store.get_proposal, proposal_id)
        if proposal is None:
            raise RelationshipError("PROPOSAL_NOT_FOUND", "Proposal not found", 404)
        return proposal

    async def review(
        self,
        proposal_id: str,
        request: RelationshipReviewRequest,
    ) -> RelationshipReview:
        proposal = await self.get(proposal_id)
        try:
            return await asyncio.to_thread(self.store.put_review, proposal, request)
        except (IntegrityError, ValueError) as exc:
            raise RelationshipError(
                "INVALID_REVIEW", "The relationship review could not be stored", 422
            ) from exc

    async def history(self, proposal_id: str) -> RelationshipReviewHistory:
        proposal = await self.get(proposal_id)
        reviews = await asyncio.to_thread(self.store.review_history, proposal_id)
        return RelationshipReviewHistory(
            proposal_id=proposal_id,
            proposal=proposal,
            reviews=reviews,
        )

    @staticmethod
    def _used_evidence(
        state: RelationshipPaperState,
        items: list[RelationshipItem],
        *,
        source_side: bool,
    ) -> list[RelationshipEvidence]:
        used_ids = {
            evidence_id
            for item in items
            for evidence_id in (
                item.source_evidence_ids if source_side else item.target_evidence_ids
            )
        }
        return [evidence for evidence in state.evidence if evidence.evidence_id in used_ids]

    def _semantic_similarity(self, source: ParsedPaper, target: ParsedPaper) -> float:
        papers = [
            Paper(
                id=source.paper_id,
                title=source.title or source.paper_id,
                abstract=source.abstract,
                authors=source.authors,
            ),
            Paper(
                id=target.paper_id,
                title=target.title or target.paper_id,
                abstract=target.abstract,
                authors=target.authors,
            ),
        ]
        matrix = self.semantics.paper_similarity_matrix(papers)
        return round(max(0.0, min(1.0, float(matrix[0, 1]))), 6)


__all__ = [
    "RELATIONSHIP_PIPELINE_VERSION",
    "RelationshipError",
    "RelationshipPaperState",
    "RelationshipService",
    "RelationshipStateResolver",
    "RelationshipStore",
    "RelationshipValidator",
]
