"""Small live smoke check for provider connectivity; not part of the offline test suite."""

import argparse
import asyncio

import httpx

from app.core.config import get_settings
from app.providers import ArxivProvider, OpenAlexProvider, SemanticScholarProvider
from app.providers.base import ProviderError


async def main(query: str, limit: int) -> int:
    settings = get_settings()
    timeout = httpx.Timeout(settings.provider_timeout_seconds)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "research-landscape-explorer/0.1 (live provider check)"},
    ) as client:
        providers = [
            OpenAlexProvider(client, retries=1, api_key=settings.openalex_api_key),
            SemanticScholarProvider(client, retries=1, api_key=settings.semantic_scholar_api_key),
            ArxivProvider(client, retries=1),
        ]
        succeeded = 0
        for provider in providers:
            try:
                papers = await provider.search(query, limit)
                example = papers[0].title if papers else "no results"
                print(f"{provider.name}: ok ({len(papers)} papers); first: {example}")
                succeeded += 1
            except ProviderError as exc:
                print(f"{provider.name}: error ({exc})")
    return 0 if succeeded else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="AI agents for visualization systems")
    parser.add_argument("--limit", type=int, default=3)
    arguments = parser.parse_args()
    raise SystemExit(asyncio.run(main(arguments.query, arguments.limit)))
