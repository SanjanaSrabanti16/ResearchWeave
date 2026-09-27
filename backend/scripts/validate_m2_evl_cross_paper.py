"""Run one EVL paper-understanding validation through the production service path."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from app.core.config import Settings
from app.llm import EVLGemmaProvider, LLMProviderRegistry
from app.models.document import ParsedPaper
from app.services.insight_cache import InsightCache
from app.services.insight_service import document_fingerprint
from app.services.paper_understanding_service import (
    PIPELINE_VERSION,
    PaperUnderstandingDiagnostics,
    PaperUnderstandingService,
)

PROJECT_ROOT = Path(__file__).parents[2]


def _load_target(title: str) -> ParsedPaper:
    directory = Path(__file__).parents[1] / "data" / "parsed_documents"
    matches: list[tuple[float, ParsedPaper]] = []
    for path in directory.glob("*.json"):
        try:
            document = ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if document.title == title:
            matches.append((path.stat().st_mtime, document))
    if not matches:
        raise RuntimeError(f"Cached ParsedPaper is unavailable: {title}")
    return max(matches, key=lambda item: item[0])[1]


async def _run(title: str, output_path: Path) -> None:
    settings = Settings(_env_file=PROJECT_ROOT / ".env")
    if settings.evl_gemma_api_key is None:
        raise RuntimeError("EVL_GEMMA_API_KEY is not configured")
    document = _load_target(title)
    cache = InsightCache(settings.insight_cache_dir)
    cache_key = cache.key(
        document_fingerprint(document),
        settings.evl_gemma_model,
        PIPELINE_VERSION,
        "evl_gemma",
    )
    if cache.get(cache_key) is not None:
        raise RuntimeError("Fresh validation would reuse an existing EVL Gemma cache entry")

    provider = EVLGemmaProvider(
        api_key=settings.evl_gemma_api_key.get_secret_value(),
        base_url=settings.evl_gemma_base_url,
        model_id=settings.evl_gemma_model,
        timeout_seconds=settings.evl_gemma_timeout_seconds,
        diagnostic_dir=settings.llm_diagnostic_dir,
    )
    registry = LLMProviderRegistry([provider], default_provider="evl_gemma")
    diagnostics = PaperUnderstandingDiagnostics()
    try:
        result = await PaperUnderstandingService(
            registry=registry,
            cache=cache,
            batch_chars=settings.ollama_batch_chars,
        ).extract(document, provider_id="evl_gemma", diagnostics=diagnostics)
    finally:
        await registry.aclose()

    payload = {
        "paper": {"title": document.title, "paper_id": document.paper_id},
        "provider": "evl_gemma",
        "model": result.model,
        "pipeline_version": result.extraction_version,
        "cached": result.cached,
        "diagnostics": asdict(diagnostics),
        "insights": result.insights.model_dump(mode="json"),
    }
    await asyncio.to_thread(output_path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(
        output_path.write_text,
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "paper": payload["paper"],
                "provider": payload["provider"],
                "model": payload["model"],
                "pipeline_version": payload["pipeline_version"],
                "cached": payload["cached"],
                "diagnostics": payload["diagnostics"],
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(_run(args.title, args.output))


if __name__ == "__main__":
    main()
