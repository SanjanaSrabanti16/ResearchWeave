from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

import httpx

PDF_MAGIC = b"%PDF-"
DEFAULT_ALLOWED_PDF_HOSTS = (
    "arxiv.org",
    "biorxiv.org",
    "core.ac.uk",
    "europepmc.org",
    "hal.science",
    "medrxiv.org",
    "ncbi.nlm.nih.gov",
    "openreview.net",
    "osf.io",
    "semanticscholar.org",
    "zenodo.org",
)


class PDFValidationError(RuntimeError):
    """A safe validation error for an uploaded or downloaded PDF."""


class PDFTooLargeError(PDFValidationError):
    pass


class PDFDownloadError(PDFValidationError):
    pass


Resolver = Callable[[str], Awaitable[list[str]]]


async def resolve_host(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    results = await loop.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    return list({row[4][0] for row in results})


def validate_pdf_bytes(data: bytes, max_bytes: int) -> None:
    if len(data) > max_bytes:
        raise PDFTooLargeError(f"PDF exceeds the {max_bytes // (1024 * 1024)} MB limit")
    if not data.startswith(PDF_MAGIC):
        raise PDFValidationError("The file is not a valid PDF")


def _host_is_allowed(host: str, allowed_hosts: tuple[str, ...]) -> bool:
    normalized = host.casefold().rstrip(".")
    return any(normalized == item or normalized.endswith(f".{item}") for item in allowed_hosts)


async def validate_remote_url(
    url: str,
    *,
    resolver: Resolver = resolve_host,
    allowed_hosts: tuple[str, ...] | None = None,
) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise PDFDownloadError("PDF URLs must use public HTTPS addresses")
    if parsed.port not in (None, 443):
        raise PDFDownloadError("PDF URLs must use the standard HTTPS port")
    if allowed_hosts is not None and not _host_is_allowed(parsed.hostname, allowed_hosts):
        raise PDFDownloadError("The PDF host is not an approved scholarly repository")
    try:
        addresses = await resolver(parsed.hostname)
    except OSError as exc:
        raise PDFDownloadError("The PDF host could not be resolved") from exc
    if not addresses:
        raise PDFDownloadError("The PDF host could not be resolved")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise PDFDownloadError("PDF URLs cannot target private or local networks")


class SecurePDFDownloader:
    def __init__(
        self,
        client: httpx.AsyncClient,
        max_bytes: int,
        timeout_seconds: float,
        allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_PDF_HOSTS,
        resolver: Resolver = resolve_host,
        max_redirects: int = 5,
    ) -> None:
        self.client = client
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds
        self.allowed_hosts = allowed_hosts
        self.resolver = resolver
        self.max_redirects = max_redirects

    async def download(self, url: str, *, trusted_discovery: bool = False) -> tuple[bytes, str]:
        current_url = url
        for redirect_count in range(self.max_redirects + 1):
            await validate_remote_url(
                current_url,
                resolver=self.resolver,
                allowed_hosts=None if trusted_discovery else self.allowed_hosts,
            )
            try:
                async with self.client.stream(
                    "GET", current_url, follow_redirects=False, timeout=self.timeout_seconds
                ) as response:
                    if response.is_redirect:
                        if redirect_count >= self.max_redirects:
                            raise PDFDownloadError("The PDF URL redirected too many times")
                        location = response.headers.get("location")
                        if not location:
                            raise PDFDownloadError("The PDF URL returned an invalid redirect")
                        current_url = str(response.url.join(location))
                        continue
                    if response.status_code >= 400:
                        raise PDFDownloadError(
                            f"The PDF server rejected the request ({response.status_code})"
                        )
                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > self.max_bytes:
                                raise PDFTooLargeError(
                                    f"PDF exceeds the {self.max_bytes // (1024 * 1024)} MB limit"
                                )
                        except ValueError:
                            pass
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if content_type not in {"application/pdf", "application/octet-stream"}:
                        raise PDFValidationError("The remote resource is not served as a PDF")
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise PDFTooLargeError(
                                f"PDF exceeds the {self.max_bytes // (1024 * 1024)} MB limit"
                            )
                        chunks.append(chunk)
                    data = b"".join(chunks)
                    validate_pdf_bytes(data, self.max_bytes)
                    return data, current_url
            except PDFValidationError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                raise PDFDownloadError("The PDF could not be downloaded") from exc
        raise PDFDownloadError("The PDF URL redirected too many times")
