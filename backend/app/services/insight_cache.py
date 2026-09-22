from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from app.models.insights import InsightFields


class InsightCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @staticmethod
    def key(document_fingerprint: str, model: str, extraction_version: str) -> str:
        payload = f"{document_fingerprint}\0{model}\0{extraction_version}"
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
