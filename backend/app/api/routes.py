import asyncio
import json
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from app.llm import (
    LLMAuthenticationError,
    LLMOutputError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
    UnknownLLMProviderError,
)
from app.models.api import (
    CachedPaperAnalysisResponse,
    GraphRequest,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from app.models.document import PDFAcquisitionRequest, PDFProcessingResponse
from app.models.insights import (
    InsightsRequest,
    InsightsResponse,
    LLMProviderStatusResponse,
)
from app.models.relationships import (
    RelationshipProposal,
    RelationshipProposalRequest,
    RelationshipReview,
    RelationshipReviewHistory,
    RelationshipReviewRequest,
)
from app.parsers import PDFParserError, PDFParserUnavailableError
from app.services.graph_semantics import (
    GraphSeed,
    GraphSemanticsInputError,
    GraphSemanticsUnavailableError,
)
from app.services.graph_service import GraphService
from app.services.insight_service import (
    InsightInputError,
    InsightOutputError,
    InsightService,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from app.services.paper_understanding_service import PaperUnderstandingService
from app.services.pdf_service import PDFAcquisitionService
from app.services.pdf_upload import read_pdf_upload
from app.services.pdf_validation import PDFTooLargeError, PDFValidationError
from app.services.ranking_service import RankingUnavailableError
from app.services.relationship_service import RelationshipError, RelationshipService
from app.services.search_service import AllProvidersFailedError, SearchService

router = APIRouter()


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def get_pdf_service(request: Request) -> PDFAcquisitionService:
    return request.app.state.pdf_service


def get_insight_service(request: Request) -> PaperUnderstandingService | InsightService:
    return request.app.state.insight_service


def get_graph_service(request: Request) -> GraphService:
    return request.app.state.graph_service


def get_relationship_service(request: Request) -> RelationshipService:
    return request.app.state.relationship_service


def _relationship_http_error(exc: RelationshipError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": exc.message},
    )


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


@router.post("/api/graph", response_model=GraphSeed)
async def graph(payload: GraphRequest, request: Request) -> GraphSeed:
    try:
        return await get_graph_service(request).build(payload)
    except GraphSemanticsInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except GraphSemanticsUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@router.post("/api/relationships/propose", response_model=RelationshipProposal)
async def propose_relationship(
    payload: RelationshipProposalRequest,
    request: Request,
) -> RelationshipProposal:
    try:
        return await get_relationship_service(request).propose(payload)
    except RelationshipError as exc:
        raise _relationship_http_error(exc) from exc


@router.get("/api/relationships/by-pair", response_model=RelationshipProposal | None)
async def get_relationship_by_pair(
    source_paper_id: str,
    target_paper_id: str,
    request: Request,
    response: Response,
) -> RelationshipProposal | None:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_relationship_service(request).lookup_pair(
            source_paper_id,
            target_paper_id,
        )
    except RelationshipError as exc:
        raise _relationship_http_error(exc) from exc


@router.get("/api/relationships/{proposal_id}", response_model=RelationshipProposal)
async def get_relationship(
    proposal_id: str,
    request: Request,
    response: Response,
) -> RelationshipProposal:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_relationship_service(request).get(proposal_id)
    except RelationshipError as exc:
        raise _relationship_http_error(exc) from exc


@router.post("/api/relationships/{proposal_id}/review", response_model=RelationshipReview)
async def review_relationship(
    proposal_id: str,
    payload: RelationshipReviewRequest,
    request: Request,
) -> RelationshipReview:
    try:
        return await get_relationship_service(request).review(proposal_id, payload)
    except RelationshipError as exc:
        raise _relationship_http_error(exc) from exc


@router.get(
    "/api/relationships/{proposal_id}/history",
    response_model=RelationshipReviewHistory,
)
async def relationship_history(
    proposal_id: str,
    request: Request,
    response: Response,
) -> RelationshipReviewHistory:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_relationship_service(request).history(proposal_id)
    except RelationshipError as exc:
        raise _relationship_http_error(exc) from exc


@router.get(
    "/api/papers/{paper_id}/cached-analysis",
    response_model=CachedPaperAnalysisResponse,
)
async def cached_paper_analysis(
    paper_id: str, request: Request, response: Response
) -> CachedPaperAnalysisResponse:
    response.headers["Cache-Control"] = "no-store"
    return await get_graph_service(request).cached_analysis(paper_id)


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
        service = get_insight_service(request)
        if isinstance(service, PaperUnderstandingService):
            return await service.extract(payload.document, provider_id=payload.provider)
        return await service.extract(payload.document)
    except InsightInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except UnknownLLMProviderError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LLMAuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except LLMRateLimitError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except (OllamaUnavailableError, LLMProviderUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except (OllamaTimeoutError, LLMTimeoutError) as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except (InsightOutputError, LLMOutputError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.get("/api/llm/providers", response_model=LLMProviderStatusResponse)
async def llm_provider_status(request: Request) -> LLMProviderStatusResponse:
    registry = request.app.state.llm_registry
    return LLMProviderStatusResponse(
        default_provider=registry.default_provider,
        providers=registry.statuses(),
    )


@router.post("/api/papers/insights/stream")
async def stream_insights(payload: InsightsRequest, request: Request) -> StreamingResponse:
    service = get_insight_service(request)

    async def events():
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

        async def progress(stage: str) -> None:
            await queue.put({"type": "stage", "stage": stage})

        async def run() -> None:
            try:
                if isinstance(service, PaperUnderstandingService):
                    result = await service.extract(
                        payload.document,
                        provider_id=payload.provider,
                        on_progress=progress,
                    )
                else:
                    result = await service.extract(payload.document, on_progress=progress)
                await queue.put({"type": "result", "data": result.model_dump(mode="json")})
            except (
                InsightInputError,
                OllamaUnavailableError,
                OllamaTimeoutError,
                InsightOutputError,
                UnknownLLMProviderError,
                LLMAuthenticationError,
                LLMRateLimitError,
                LLMProviderUnavailableError,
                LLMTimeoutError,
                LLMOutputError,
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
