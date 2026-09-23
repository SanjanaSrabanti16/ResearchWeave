from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models.document import ParsedPaper
from app.models.insights import INSIGHT_FIELDS, InsightClaim, InsightFields
from app.services.insight_selector import EvidenceSelection


class InsightDiagnostics:
    """Opt-in recorder for insight extraction decisions; never used by default."""

    def __init__(self) -> None:
        self.trace: dict[str, Any] = {
            "trace_schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "extraction": {},
            "stages": [],
            "final_output": None,
        }

    def start_extraction(
        self,
        document: ParsedPaper,
        document_fingerprint: str,
        model: str,
        extraction_version: str,
    ) -> None:
        self.trace["extraction"] = {
            "paper_id": document.paper_id,
            "paper_title": document.title,
            "document_fingerprint": document_fingerprint,
            "model": model,
            "extraction_version": extraction_version,
            "cached": False,
        }

    def record_cache_hit(self) -> None:
        self.trace["extraction"]["cached"] = True

    def start_stage(
        self,
        stage_id: str,
        stage_name: str,
        focus: str,
        document: ParsedPaper,
        selection: EvidenceSelection,
        catalog: dict[str, tuple[str, str, str]],
    ) -> None:
        section_ids = {chunk.id: "abstract" for chunk in document.abstract_chunks}
        section_ids.update(
            (chunk.id, section.id) for section in document.sections for chunk in section.chunks
        )
        routing_by_chunk: dict[str, list[str]] = {}
        selected_ids = {chunk_id for _heading, chunk_id, _text in selection.chunks}
        for pool, chunk_ids in selection.pools.items():
            for chunk_id in chunk_ids:
                if chunk_id in selected_ids:
                    routing_by_chunk.setdefault(chunk_id, []).append(pool)
        self.trace["stages"].append(
            {
                "stage_id": stage_id,
                "stage_name": stage_name,
                "focus": focus,
                "selected_evidence": [
                    {
                        "evidence_id": evidence_id,
                        "chunk_id": chunk_id,
                        "section_id": section_ids.get(chunk_id),
                        "section_title": heading,
                        "routing_categories": sorted(routing_by_chunk.get(chunk_id, [])),
                        "quote": quote,
                    }
                    for evidence_id, (chunk_id, quote, heading) in catalog.items()
                ],
                "raw_model_output": None,
                "candidate_claims": [],
            }
        )

    def _stage(self, stage_id: str) -> dict[str, Any]:
        return next(stage for stage in self.trace["stages"] if stage["stage_id"] == stage_id)

    def record_raw_model_output(self, stage_id: str, content: str) -> None:
        self._stage(stage_id)["raw_model_output"] = content

    def record_candidate(
        self,
        stage_id: str,
        field: str,
        index: int,
        claim: str,
        evidence_ids: list[str],
    ) -> str:
        candidate_id = f"{stage_id}:{field}:{index + 1}"
        self._stage(stage_id)["candidate_claims"].append(
            {
                "candidate_id": candidate_id,
                "field": field,
                "claim": claim,
                "referenced_evidence_ids": evidence_ids,
                "evidence_resolution": None,
                "validation": [],
                "final_retained": None,
            }
        )
        return candidate_id

    def record_evidence_resolution(
        self,
        stage_id: str,
        candidate_id: str,
        *,
        retained: bool,
        reason: str | None,
        resolved_evidence: list[dict[str, str]] | None = None,
    ) -> None:
        candidate = self._candidate(stage_id, candidate_id)
        candidate["evidence_resolution"] = {
            "retained": retained,
            "rule": "evidence_id_resolution",
            "reason": reason,
            "resolved_evidence": resolved_evidence,
        }

    def _candidate(self, stage_id: str, candidate_id: str) -> dict[str, Any]:
        return next(
            candidate
            for candidate in self._stage(stage_id)["candidate_claims"]
            if candidate["candidate_id"] == candidate_id
        )

    @staticmethod
    def _claim_matches(candidate: dict[str, Any], field: str, item: InsightClaim) -> bool:
        if candidate["field"] != field or candidate["claim"] != item.claim:
            return False
        resolved = (candidate.get("evidence_resolution") or {}).get("resolved_evidence") or []
        return not resolved or any(
            reference.chunk_id == evidence.get("chunk_id")
            and reference.quote == evidence.get("quote")
            for reference in item.evidence
            for evidence in resolved
        )

    def record_validation(
        self,
        field: str,
        item: InsightClaim,
        *,
        retained: bool,
        rule: str,
        reason: str | None,
    ) -> None:
        for stage in self.trace["stages"]:
            for candidate in stage["candidate_claims"]:
                if self._claim_matches(candidate, field, item):
                    candidate["validation"].append(
                        {"retained": retained, "rule": rule, "reason": reason}
                    )
                    return

    def finalize(self, insights: InsightFields) -> None:
        final_items: list[dict[str, Any]] = []
        for field in INSIGHT_FIELDS:
            for item in getattr(insights, field):
                provenance = []
                for stage in self.trace["stages"]:
                    for candidate in stage["candidate_claims"]:
                        if self._claim_matches(candidate, field, item):
                            candidate["final_retained"] = True
                            provenance.append(candidate["candidate_id"])
                final_items.append(
                    {
                        "field": field,
                        "claim": item.claim,
                        "evidence": [reference.model_dump() for reference in item.evidence],
                        "candidate_ids": provenance,
                    }
                )
        for stage in self.trace["stages"]:
            for candidate in stage["candidate_claims"]:
                if candidate["final_retained"] is None:
                    candidate["final_retained"] = False
                    validations = candidate["validation"]
                    resolution = candidate["evidence_resolution"] or {}
                    if resolution.get("retained") is False:
                        candidate["final_disposition"] = resolution.get("reason")
                    elif any(event["retained"] is False for event in validations):
                        candidate["final_disposition"] = next(
                            event["reason"] for event in validations if event["retained"] is False
                        )
                    else:
                        candidate["final_disposition"] = "deduplication_or_coverage_limit"
                else:
                    candidate["final_disposition"] = "retained"
        self.trace["final_output"] = {
            "counts": {field: len(getattr(insights, field)) for field in INSIGHT_FIELDS},
            "claims": final_items,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.trace

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
