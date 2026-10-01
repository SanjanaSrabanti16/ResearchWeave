from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import urljoin

from app.services.pdf_validation import SecurePDFDownloader


class _PDFMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value for key, value in attrs if value is not None}
        if tag.casefold() == "meta":
            name = (values.get("name") or values.get("property") or "").casefold()
            if name in {
                "citation_pdf_url",
                "dc.identifier.pdf",
                "eprints.document_url",
                "wkhealth_pdf_url",
                "pdf_url",
            } and values.get("content"):
                self.urls.append(values["content"])
        if tag.casefold() == "link":
            rel = values.get("rel", "").casefold().split()
            content_type = values.get("type", "").casefold()
            href = values.get("href")
            if href and "alternate" in rel and content_type == "application/pdf":
                self.urls.append(href)


class LandingPageResolver:
    def __init__(self, downloader: SecurePDFDownloader) -> None:
        self.downloader = downloader

    async def find_pdf_urls(self, url: str) -> tuple[list[str], str, bool]:
        html, final_url = await self.downloader.download_html(url, trusted_discovery=True)
        parser = _PDFMetadataParser()
        parser.feed(html)
        resolved = list(dict.fromkeys(urljoin(final_url, value) for value in parser.urls))
        paywall = any(
            marker in html.casefold()
            for marker in ("purchase access", "institutional login", "sign in to access")
        )
        return resolved, final_url, paywall
