import asyncio
import json
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from app.models.api import HealthResponse, SearchRequest, SearchResponse
from app.models.document import PDFAcquisitionRequest, PDFProcessingResponse
from app.models.insights import InsightsRequest, InsightsResponse
from app.parsers import PDFParserError, PDFParserUnavailableError
from app.services.insight_service import (
    InsightInputError,
    InsightOutputError,
    InsightService,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from app.services.pdf_service import PDFAcquisitionService
from app.services.pdf_upload import read_pdf_upload
from app.services.pdf_validation import PDFTooLargeError, PDFValidationError
from app.services.ranking_service import RankingUnavailableError
from app.services.search_service import AllProvidersFailedError, SearchService

router = APIRouter()


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def get_pdf_service(request: Request) -> PDFAcquisitionService:
    return request.app.state.pdf_service


def get_insight_service(request: Request) -> InsightService:
    return request.app.state.insight_service


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        application=request.app.title,
        version=request.app.version,
        timestamp=datetime.now(UTC),
    )


@router.post("/api/search", response_model=SearchResponse)
async def search(payload: SearchRequest, request: Request) -> SearchResponse:
    try:
        return await get_search_service(request).search(payload)
    except AllProvidersFailedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except RankingUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@router.post("/api/papers/pdf/acquire", response_model=PDFProcessingResponse)
async def acquire_pdf(payload: PDFAcquisitionRequest, request: Request) -> PDFProcessingResponse:
    try:
        return await get_pdf_service(request).acquire(payload)
    except PDFParserUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except PDFParserError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except PDFValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/api/papers/pdf/upload", response_model=PDFProcessingResponse)
async def upload_pdf(
    request: Request,
    paper_id: Annotated[str, Form(min_length=1, max_length=200)],
    file: Annotated[UploadFile, File()],
) -> PDFProcessingResponse:
    max_bytes = get_pdf_service(request).processor.max_bytes
    try:
        content = await read_pdf_upload(file, max_bytes)
        return await get_pdf_service(request).upload(paper_id, content)
    except PDFTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except PDFParserUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except (PDFValidationError, PDFParserError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc


@router.post("/api/papers/insights", response_model=InsightsResponse)
async def extract_insights(payload: InsightsRequest, request: Request) -> InsightsResponse:
    try:
        return await get_insight_service(request).extract(payload.document)
    except InsightInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except OllamaTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except InsightOutputError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post("/api/papers/insights/stream")
async def stream_insights(payload: InsightsRequest, request: Request) -> StreamingResponse:
    service = get_insight_service(request)

    async def events():
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

        async def progress(stage: str) -> None:
            await queue.put({"type": "stage", "stage": stage})

        async def run() -> None:
            try:
                result = await service.extract(payload.document, on_progress=progress)
                await queue.put({"type": "result", "data": result.model_dump(mode="json")})
            except (
                InsightInputError,
                OllamaUnavailableError,
                OllamaTimeoutError,
                InsightOutputError,
            ) as exc:
                await queue.put({"type": "error", "message": str(exc)})
            except Exception:
                await queue.put(
                    {"type": "error", "message": "Insight extraction failed unexpectedly"}
                )

        worker = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                yield json.dumps(event) + "\n"
                if event["type"] in {"result", "error"}:
                    break
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    return StreamingResponse(
        events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"}
    )
