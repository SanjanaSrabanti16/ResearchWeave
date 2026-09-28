from __future__ import annotations

import hashlib
import json
import uuid
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

    @staticmethod
    def _paper_state_key(paper_id: str) -> str:
        return hashlib.sha256(paper_id.encode("utf-8")).hexdigest()

    def _paper_state_path(self, paper_id: str) -> Path:
        return self.directory / f"paper-state-{self._paper_state_key(paper_id)}.json"

    def get_paper_state(self, paper_id: str) -> ParsedPaper | None:
        try:
            document = ParsedPaper.model_validate_json(
                self._paper_state_path(paper_id).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        return document if document.paper_id == paper_id else None

    def put_paper_state(self, document: ParsedPaper) -> None:
        """Atomically record the authoritative parsed document for a canonical paper."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._paper_state_path(document.paper_id)
        temporary = self.directory / f".{path.stem}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps(document.model_dump(mode="json"), ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def resolve_paper_states(self, paper_ids: set[str]) -> dict[str, ParsedPaper]:
        """Resolve current snapshots, with a deterministic newest-first legacy fallback."""
        documents = {
            paper_id: document
            for paper_id in paper_ids
            if (document := self.get_paper_state(paper_id)) is not None
        }
        unresolved = paper_ids - documents.keys()
        if not unresolved or not self.directory.exists():
            return documents

        candidates: list[tuple[int, str, Path]] = []
        for path in self.directory.glob("*.json"):
            if path.name.startswith("paper-state-"):
                continue
            try:
                candidates.append((path.stat().st_mtime_ns, path.name, path))
            except OSError:
                continue
        for _mtime, _name, path in sorted(candidates, reverse=True):
            try:
                document = ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if document.paper_id in unresolved:
                documents[document.paper_id] = document
                unresolved.remove(document.paper_id)
                if not unresolved:
                    break
        return documents
