from __future__ import annotations

from fastapi import UploadFile

from app.services.pdf_validation import PDFTooLargeError, PDFValidationError


async def read_pdf_upload(file: UploadFile, max_bytes: int) -> bytes:
    try:
        if file.filename and not file.filename.casefold().endswith(".pdf"):
            raise PDFValidationError("Uploaded files must use the .pdf extension")
        if file.content_type and file.content_type.casefold() not in {
            "application/pdf",
            "application/octet-stream",
        }:
            raise PDFValidationError("Uploaded files must use a PDF content type")
        content = await file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise PDFTooLargeError(f"PDF exceeds the {max_bytes // (1024 * 1024)} MB limit")
        return content
    finally:
        await file.close()
