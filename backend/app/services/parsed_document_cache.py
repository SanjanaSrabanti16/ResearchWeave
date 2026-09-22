from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.models.document import ParsedPaper


class ParsedDocumentCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @staticmethod
    def key(pdf_bytes: bytes, parser: str, parser_version: str) -> str:
        digest = hashlib.sha256()
        digest.update(pdf_bytes)
        digest.update(b"\0")
        digest.update(parser.encode())
        digest.update(b"\0")
        digest.update(parser_version.encode())
        return digest.hexdigest()

    def get(self, key: str) -> ParsedPaper | None:
        path = self.directory / f"{key}.json"
        try:
            return ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key: str, document: ParsedPaper) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(document.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)
