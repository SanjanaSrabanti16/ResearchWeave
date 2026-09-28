from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from pydantic import BaseModel

from app.models.insights import InsightFields


class CurrentInsightState(BaseModel):
    paper_id: str
    document_fingerprint: str
    provider_id: str
    model: str
    extraction_version: str
    insights: InsightFields


class InsightCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @staticmethod
    def key(
        document_fingerprint: str,
        model: str,
        extraction_version: str,
        provider_id: str | None = None,
    ) -> str:
        provider_part = f"{provider_id}\0" if provider_id is not None else ""
        payload = f"{document_fingerprint}\0{provider_part}{model}\0{extraction_version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> InsightFields | None:
        try:
            return InsightFields.model_validate_json(
                (self.directory / f"{key}.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None

    def put(self, key: str, insights: InsightFields) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        temporary = self.directory / f".{key}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(insights.model_dump_json(), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _paper_state_key(paper_id: str) -> str:
        return hashlib.sha256(paper_id.encode("utf-8")).hexdigest()

    def _paper_state_path(self, paper_id: str) -> Path:
        return self.directory / f"paper-state-{self._paper_state_key(paper_id)}.json"

    def get_current(
        self,
        paper_id: str,
        document_fingerprint: str,
        extraction_version: str,
    ) -> CurrentInsightState | None:
        try:
            state = CurrentInsightState.model_validate_json(
                self._paper_state_path(paper_id).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        if (
            state.paper_id != paper_id
            or state.document_fingerprint != document_fingerprint
            or state.extraction_version != extraction_version
        ):
            return None
        return state

    def put_current(
        self,
        *,
        paper_id: str,
        document_fingerprint: str,
        provider_id: str,
        model: str,
        extraction_version: str,
        insights: InsightFields,
    ) -> None:
        """Record the latest successful analysis without replacing immutable cache entries."""
        self.directory.mkdir(parents=True, exist_ok=True)
        state = CurrentInsightState(
            paper_id=paper_id,
            document_fingerprint=document_fingerprint,
            provider_id=provider_id,
            model=model,
            extraction_version=extraction_version,
            insights=insights,
        )
        path = self._paper_state_path(paper_id)
        temporary = self.directory / f".{path.stem}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(state.model_dump_json(), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
