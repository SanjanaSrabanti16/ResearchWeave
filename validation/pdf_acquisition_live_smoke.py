# ruff: noqa: E402 -- validation script adds the backend package to sys.path.
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import Settings
from app.models.document import PDFAcquisitionRequest
from app.parsers import GrobidParser
from app.providers import ArxivProvider, OpenAlexProvider, SemanticScholarProvider
from app.providers.base import ProviderError
from app.providers.unpaywall import UnpaywallProvider
from app.services.deduplication_service import DeduplicationService
from app.services.normalization import normalize_title
from app.services.parsed_document_cache import ParsedDocumentCache
from app.services.pdf_service import PDFAcquisitionService, PDFProcessingService
from app.services.pdf_validation import SecurePDFDownloader

TITLE = "Attention Is All You Need"


async def main() -> None:
    settings = Settings()
    timeout = httpx.Timeout(settings.provider_timeout_seconds)
    async with httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": "ResearchWeave/0.1 (open-source research tool)"},
    ) as client:
        arxiv_provider = ArxivProvider(client, retries=settings.provider_retries)
        providers = [
            OpenAlexProvider(
                client,
                retries=settings.provider_retries,
                api_key=settings.openalex_api_key,
            ),
            SemanticScholarProvider(
                client,
                retries=settings.provider_retries,
                api_key=settings.semantic_scholar_api_key,
            ),
            arxiv_provider,
        ]
        provider_counts: dict[str, dict[str, int | str]] = {}
        exact_records = []
        for provider in providers:
            try:
                records = await provider.search(TITLE, 10, 2017, 2017)
            except ProviderError as exc:
                provider_counts[provider.name] = {"status": "error", "error": str(exc)}
                continue
            exact = [
                record
                for record in records
                if normalize_title(record.title) == normalize_title(TITLE)
            ]
            provider_counts[provider.name] = {
                "status": "ok",
                "returned": len(records),
                "exact_title_records": len(exact),
            }
            exact_records.extend(exact)

        papers = DeduplicationService().deduplicate(exact_records)
        if not papers:
            print(
                json.dumps(
                    {"provider_records": provider_counts, "result": "no_exact_record"}, indent=2
                )
            )
            return
        paper = max(papers, key=lambda item: len(item.source_names))
        request = PDFAcquisitionRequest(
            paper_id=paper.id,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            arxiv_ids=paper.arxiv_ids,
            pdf_url=paper.pdf_url,
            alternate_pdf_urls=paper.alternate_pdf_urls,
            url=paper.url,
            alternate_urls=paper.alternate_urls,
            title=paper.title,
            authors=paper.authors,
            publication_year=paper.publication_year,
        )
        with tempfile.TemporaryDirectory(prefix="researchweave-pdf-smoke-") as cache_dir:
            processor = PDFProcessingService(
                parser=GrobidParser(
                    client,
                    base_url="http://localhost:8070",
                    version=settings.grobid_parser_version,
                    timeout_seconds=settings.grobid_timeout_seconds,
                ),
                cache=ParsedDocumentCache(Path(cache_dir)),
                max_bytes=settings.pdf_max_size_mb * 1024 * 1024,
            )
            service = PDFAcquisitionService(
                downloader=SecurePDFDownloader(
                    client,
                    max_bytes=settings.pdf_max_size_mb * 1024 * 1024,
                    timeout_seconds=settings.pdf_download_timeout_seconds,
                    allowed_hosts=settings.pdf_allowed_host_list,
                ),
                unpaywall=UnpaywallProvider(client, settings.unpaywall_email),
                processor=processor,
                arxiv_provider=arxiv_provider,
            )
            preview_attempts = []
            initial = service._initial_candidates(request, preview_attempts)
            candidate_order = [
                {"route": candidate.route, "host": service._safe_identifier(candidate.url)}
                for candidate in initial
            ]
            candidate_order.extend(
                [
                    {"route": "unpaywall", "host": "api.unpaywall.org"},
                    {
                        "route": "title_verified_arxiv_fallback",
                        "host": "export.arxiv.org/arxiv.org",
                    },
                ]
            )
            result = await service.acquire(request)

        output = {
            "query": TITLE,
            "provider_records": provider_counts,
            "deduplicated_count": len(papers),
            "canonical": {
                "id": paper.id,
                "title": paper.title,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "arxiv_ids": paper.arxiv_ids,
                "source_names": paper.source_names,
                "url": paper.url,
                "alternate_urls": paper.alternate_urls,
                "pdf_url": paper.pdf_url,
                "alternate_pdf_urls": paper.alternate_pdf_urls,
            },
            "candidate_order": candidate_order,
            "status": result.status,
            "acquisition_provenance": result.acquisition_provenance,
            "attempts": [attempt.model_dump(mode="json") for attempt in result.attempts],
            "pdf_validated_and_parsed": result.document is not None,
            "pdf_sha256": result.document.source_pdf.sha256 if result.document else None,
            "parser": result.document.parser if result.document else None,
            "manual_upload_available": True,
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
