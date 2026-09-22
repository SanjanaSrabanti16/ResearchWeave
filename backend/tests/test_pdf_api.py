from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.models.document import ParsedPaper, PDFProcessingResponse, SourcePDF


class MockPDFService:
    processor = SimpleNamespace(max_bytes=1024)

    async def acquire(self, request):
        return PDFProcessingResponse(
            status="upload_required",
            paper_id=request.paper_id,
            message="No validated open-access PDF was found; upload is required.",
        )

    async def upload(self, paper_id: str, pdf_bytes: bytes):
        return PDFProcessingResponse(
            status="upload",
            paper_id=paper_id,
            document=ParsedPaper(
                paper_id=paper_id,
                title="Uploaded paper",
                parser="mock",
                parser_version="1",
                source_pdf=SourcePDF(
                    acquisition_method="upload",
                    sha256="b" * 64,
                    size_bytes=len(pdf_bytes),
                ),
            ),
        )


def test_pdf_acquisition_and_upload_api_serialization(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}",
        parsed_document_cache_dir=str(tmp_path / "parsed"),
    )
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.pdf_service = MockPDFService()
        acquisition = client.post(
            "/api/papers/pdf/acquire",
            json={"paper_id": "paper-1", "doi": "10.1000/example"},
        )
        upload = client.post(
            "/api/papers/pdf/upload",
            data={"paper_id": "paper-1"},
            files={"file": ("paper.pdf", b"%PDF-1.7\ntest", "application/pdf")},
        )

    assert acquisition.status_code == 200
    assert acquisition.json()["status"] == "upload_required"
    assert upload.status_code == 200
    assert upload.json()["document"]["title"] == "Uploaded paper"
    assert upload.json()["document"]["source_pdf"]["acquisition_method"] == "upload"


def test_upload_rejects_non_pdf_magic(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}",
        parsed_document_cache_dir=str(tmp_path / "parsed"),
    )
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post(
            "/api/papers/pdf/upload",
            data={"paper_id": "paper-1"},
            files={"file": ("paper.pdf", b"not pdf", "application/pdf")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "The file is not a valid PDF"
