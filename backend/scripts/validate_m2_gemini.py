"""One-shot Milestone 2 Gemini validation using the cached target ParsedPaper."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from app.core.config import Settings
from app.llm import GeminiProvider, LLMProviderRegistry
from app.models.document import ParsedPaper
from app.services.insight_cache import InsightCache
from app.services.insight_service import document_fingerprint
from app.services.paper_understanding_service import (
    PIPELINE_VERSION,
    PaperUnderstandingDiagnostics,
    PaperUnderstandingService,
)

TARGET_TITLE = "Agentic Design Patterns: A System-Theoretic Framework"
PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_PATH = PROJECT_ROOT / "validation" / "gemini_m2_validation.json"


def _load_target() -> ParsedPaper:
    directory = Path(__file__).parents[1] / "data" / "parsed_documents"
    for path in directory.glob("*.json"):
        try:
            document = ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if document.title == TARGET_TITLE:
            return document
    raise RuntimeError("The cached target ParsedPaper is unavailable")


async def main() -> None:
    settings = Settings(_env_file=PROJECT_ROOT / ".env")
    if settings.gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    document = _load_target()
    cache = InsightCache(settings.insight_cache_dir)
    cache_key = cache.key(
        document_fingerprint(document),
        settings.gemini_model,
        PIPELINE_VERSION,
        "gemini",
    )
    if cache.get(cache_key) is not None:
        raise RuntimeError("Fresh validation would reuse an existing Gemini cache entry")
    provider = GeminiProvider(
        api_key=settings.gemini_api_key.get_secret_value(),
        model_id=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
    )
    registry = LLMProviderRegistry([provider], default_provider="gemini")
    diagnostics = PaperUnderstandingDiagnostics()
    try:
        result = await PaperUnderstandingService(
            registry=registry,
            cache=cache,
            batch_chars=settings.ollama_batch_chars,
        ).extract(document, provider_id="gemini", diagnostics=diagnostics)
    finally:
        await registry.aclose()
    if result.cached:
        raise RuntimeError("Fresh validation unexpectedly resolved to a cache hit")
    compact_insights = {
        field_name: [
            {
                "claim": claim.claim,
                "evidence": [
                    {"chunk_id": ref.chunk_id, "quote": ref.quote} for ref in claim.evidence
                ],
            }
            for claim in getattr(result.insights, field_name)
        ]
        for field_name in result.insights.__class__.model_fields
    }
    payload = {
        "provider": "gemini",
        "model": result.model,
        "pipeline_version": result.extraction_version,
        "diagnostics": asdict(diagnostics),
        "insights": compact_insights,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "output_path": str(OUTPUT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
