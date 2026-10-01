"""One-shot developer validation of cached ParsedPaper -> normalized Markdown insights."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import httpx

from app.core.config import Settings
from app.llm import EVLGemmaProvider, GeminiProvider, LLMProviderRegistry, OllamaProvider
from app.models.document import ParsedPaper
from app.models.insights import InsightFields
from app.services.insight_cache import InsightCache
from app.services.insight_markdown import validate_markdown_insights
from app.services.insight_service import document_fingerprint
from app.services.paper_understanding_service import (
    PIPELINE_VERSION,
    PaperUnderstandingDiagnostics,
    PaperUnderstandingService,
)

PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_TITLE = "A Tale of Two Centers: Visual Exploration of Health Disparities in Cancer Care"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument(
        "--provider",
        choices=("evl_gemma", "gemini", "ollama"),
        default="evl_gemma",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "validation" / "m2_markdown_acceptance.json",
    )
    parser.add_argument(
        "--normalize-existing",
        action="store_true",
        help="Reapply deterministic parsing/grounding to a saved run without calling a provider.",
    )
    return parser.parse_args()


def _load_document(title: str) -> ParsedPaper:
    directory = Path(__file__).parents[1] / "data" / "parsed_documents"
    for path in directory.glob("*.json"):
        try:
            document = ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if document.title == title:
            return document
    raise RuntimeError(f"Cached ParsedPaper not found for title: {title}")


def _provider(settings: Settings, provider_id: str):
    if provider_id == "evl_gemma":
        if settings.evl_gemma_api_key is None:
            raise RuntimeError("EVL_GEMMA_API_KEY is not configured")
        return EVLGemmaProvider(
            api_key=settings.evl_gemma_api_key.get_secret_value(),
            base_url=settings.evl_gemma_base_url,
            model_id=settings.evl_gemma_model,
            timeout_seconds=settings.evl_gemma_timeout_seconds,
        )
    if provider_id == "gemini":
        if settings.gemini_api_key is None:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        return GeminiProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            model_id=settings.gemini_model,
            timeout_seconds=settings.gemini_timeout_seconds,
        )
    client = httpx.AsyncClient(timeout=settings.ollama_timeout_seconds)
    return OllamaProvider(client, settings.ollama_base_url, settings.ollama_model)


async def main() -> None:
    args = _arguments()
    settings = Settings(_env_file=PROJECT_ROOT / ".env")
    document = _load_document(args.title)
    cache = InsightCache(settings.insight_cache_dir)
    if args.normalize_existing:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        insights = validate_markdown_insights(
            InsightFields.model_validate(payload["insights"]), document
        )
        provider_id = str(payload["provider"])
        model = str(payload["model"])
        key = cache.key(document_fingerprint(document), model, PIPELINE_VERSION, provider_id)
        cache.put(key, insights)
        cache.put_current(
            paper_id=document.paper_id,
            document_fingerprint=document_fingerprint(document),
            provider_id=provider_id,
            model=model,
            extraction_version=PIPELINE_VERSION,
            insights=insights,
        )
        payload["insights"] = insights.model_dump(mode="json")
        payload["deterministic_normalization_reapplied"] = True
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "provider_called": False,
                    "normalized_claim_counts": {
                        field: len(getattr(insights, field))
                        for field in insights.__class__.model_fields
                    },
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return
    provider = _provider(settings, args.provider)
    registry = LLMProviderRegistry([provider], default_provider=args.provider)
    key = cache.key(
        document_fingerprint(document),
        provider.model_id,
        PIPELINE_VERSION,
        provider.provider_id,
    )
    if cache.get(key) is not None:
        raise RuntimeError("Acceptance run would reuse an existing cache entry")
    diagnostics = PaperUnderstandingDiagnostics()
    try:
        result = await PaperUnderstandingService(
            registry,
            cache,
            batch_chars=settings.ollama_batch_chars,
        ).extract(document, provider_id=args.provider, diagnostics=diagnostics)
    finally:
        await registry.aclose()
        if isinstance(provider, OllamaProvider):
            await provider.client.aclose()
    payload = {
        "paper": {"paper_id": document.paper_id, "title": document.title},
        "provider": provider.provider_id,
        "model": provider.model_id,
        "pipeline_version": result.extraction_version,
        "cached": result.cached,
        "diagnostics": asdict(diagnostics),
        "insights": result.insights.model_dump(mode="json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "paper": payload["paper"],
                "provider": payload["provider"],
                "model": payload["model"],
                "pipeline_version": payload["pipeline_version"],
                "cached": payload["cached"],
                "diagnostics": payload["diagnostics"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
