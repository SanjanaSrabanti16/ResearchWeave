from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.llm.base import (
    LLMAuthenticationError,
    LLMOutputBudgetExceeded,
    LLMOutputError,
    LLMProvider,
    LLMProviderCapabilities,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMRequestContext,
    LLMTimeoutError,
)
from app.llm.structured import (
    StructuredSchemaError,
    StructuredSerializationError,
    format_correction_prompt,
    parse_structured_output,
)


class EVLGemmaProvider(LLMProvider):
    """OpenAI-compatible adapter for the EVL-hosted Gemma service."""

    provider_id = "evl_gemma"
    capabilities = LLMProviderCapabilities(
        structured_json=True,
        context_strategy="full_document",
        cloud=True,
        max_context_chars=180000,
        max_extraction_packets=1,
        supports_format_correction=True,
    )

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model_id: str,
        timeout_seconds: float = 180.0,
        client: Any | None = None,
        diagnostic_dir: str | Path | None = None,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url
        self.model_id = model_id
        self._timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None
        self._diagnostic_dir = Path(diagnostic_dir) if diagnostic_dir is not None else None
        self._last_generation_call_count = 1
        self._last_format_correction_used = False

    def _record_invalid_output(
        self,
        *,
        context: LLMRequestContext | None,
        raw_content: str | None,
        envelope_type: str,
        finish_reason: object,
        parse_error: str | None = None,
        schema_errors: list[dict[str, Any]] | None = None,
        attempt: str = "original",
        correction_attempted: bool = False,
    ) -> None:
        if self._diagnostic_dir is None:
            return
        timestamp = datetime.now(UTC)
        run_id = context.run_id if context is not None else f"unscoped-{timestamp:%Y%m%dT%H%M%S%fZ}"
        payload = {
            "provider": self.provider_id,
            "model": self.model_id,
            "stage": context.stage if context is not None else "unknown",
            "paper_id": context.paper_id if context is not None else None,
            "paper_title": context.paper_title if context is not None else None,
            "extraction_pass": context.extraction_pass if context is not None else None,
            "raw_assistant_content": raw_content,
            "response_envelope_type": envelope_type,
            "finish_reason": str(finish_reason) if finish_reason is not None else None,
            "json_parse_error": parse_error,
            "structural_schema_errors": schema_errors or [],
            "timestamp": timestamp.isoformat(),
            "run_id": run_id,
            "attempt": attempt,
            "format_correction_attempted": correction_attempted,
        }
        try:
            self._diagnostic_dir.mkdir(parents=True, exist_ok=True)
            pass_suffix = (
                f"-{context.extraction_pass}"
                if context is not None and context.extraction_pass
                else ""
            )
            path = self._diagnostic_dir / (
                f"{run_id}-{payload['stage']}{pass_suffix}-{attempt}.json"
            )
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError:
            # Diagnostics must never replace the original safe provider error.
            return

    @property
    def configured(self) -> bool:
        return bool(self._api_key) or self._client is not None

    @property
    def last_generation_call_count(self) -> int:
        return self._last_generation_call_count

    @property
    def last_format_correction_used(self) -> bool:
        return self._last_format_correction_used

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise LLMProviderUnavailableError(
                "EVL Gemma is not configured; set EVL_GEMMA_API_KEY on the backend"
            )
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self.base_url,
            timeout=self._timeout_seconds,
            max_retries=0,
        )
        return self._client

    async def _request(self, *, messages: list[dict[str, str]], max_tokens: int) -> Any:
        client = self._get_client()
        try:
            return await client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except openai.AuthenticationError as exc:
            raise LLMAuthenticationError("EVL Gemma authentication failed") from exc
        except openai.RateLimitError as exc:
            raise LLMRateLimitError("EVL Gemma rate limit was reached") from exc
        except (openai.APITimeoutError, TimeoutError) as exc:
            raise LLMTimeoutError("EVL Gemma extraction timed out") from exc
        except openai.APIConnectionError as exc:
            raise LLMProviderUnavailableError("EVL Gemma is unavailable") from exc
        except openai.APIStatusError as exc:
            if exc.status_code in {401, 403}:
                raise LLMAuthenticationError("EVL Gemma authentication failed") from exc
            if exc.status_code == 429:
                raise LLMRateLimitError("EVL Gemma rate limit was reached") from exc
            if exc.status_code in {408, 504}:
                raise LLMTimeoutError("EVL Gemma extraction timed out") from exc
            if exc.status_code >= 500:
                raise LLMProviderUnavailableError("EVL Gemma is unavailable") from exc
            raise LLMOutputError("EVL Gemma rejected structured generation") from exc
        except OSError as exc:
            raise LLMProviderUnavailableError("EVL Gemma is unavailable") from exc

    def _content_from_response(
        self,
        response: Any,
        *,
        context: LLMRequestContext | None,
        attempt: str,
    ) -> tuple[str, object]:
        try:
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = getattr(choice, "finish_reason", None)
        except (AttributeError, IndexError, TypeError) as exc:
            self._record_invalid_output(
                context=context,
                raw_content=None,
                envelope_type="missing_response_content",
                finish_reason=None,
                parse_error="Provider response did not contain assistant content",
                attempt=attempt,
                correction_attempted=attempt == "correction",
            )
            raise LLMOutputError("EVL Gemma returned an invalid response") from exc
        if not isinstance(content, str) or not content.strip():
            budget_exceeded = str(finish_reason).lower() == "length"
            self._record_invalid_output(
                context=context,
                raw_content=content if isinstance(content, str) else None,
                envelope_type="empty_or_null_content",
                finish_reason=finish_reason,
                parse_error=(
                    "output_budget_exceeded"
                    if budget_exceeded
                    else "Provider returned empty or null assistant content"
                ),
                attempt=attempt,
                correction_attempted=attempt == "correction",
            )
            if budget_exceeded:
                raise LLMOutputBudgetExceeded("EVL Gemma output budget was exceeded")
            raise LLMOutputError("EVL Gemma returned empty structured output")
        return content, finish_reason

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        max_output_tokens: int,
        context: LLMRequestContext | None = None,
    ) -> str:
        self._last_generation_call_count = 1
        self._last_format_correction_used = False
        schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        json_prompt = (
            f"{user_prompt}\n\n"
            "OUTPUT CONTRACT:\n"
            "Return only one syntactically valid JSON object matching the following JSON Schema "
            "exactly. Use double-quoted keys and strings. Do not emit comments, trailing commas, "
            "Markdown, code fences, a preamble, or a postamble. Obey every required field, enum, "
            "minItems, and maxItems constraint exactly; never exceed an array's maxItems.\n"
            f"{schema_json}"
        )
        response = await self._request(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json_prompt},
            ],
            max_tokens=max_output_tokens,
        )
        content, finish_reason = self._content_from_response(
            response, context=context, attempt="original"
        )
        if str(finish_reason).lower() == "length":
            self._record_invalid_output(
                context=context,
                raw_content=content,
                envelope_type="truncated_output",
                finish_reason=finish_reason,
                parse_error="output_budget_exceeded",
            )
            raise LLMOutputBudgetExceeded("EVL Gemma output budget was exceeded")
        try:
            return parse_structured_output(content, schema).content
        except StructuredSchemaError as exc:
            self._record_invalid_output(
                context=context,
                raw_content=content,
                envelope_type=exc.envelope_type,
                finish_reason=finish_reason,
                schema_errors=exc.validation_error.errors(include_url=False, include_input=False),
            )
            raise LLMOutputError("EVL Gemma returned structurally invalid output") from exc
        except StructuredSerializationError as exc:
            self._record_invalid_output(
                context=context,
                raw_content=content,
                envelope_type=exc.envelope_type,
                finish_reason=finish_reason,
                parse_error=str(exc),
                correction_attempted=True,
            )
        self._last_generation_call_count = 2
        self._last_format_correction_used = True
        correction_system, correction_user = format_correction_prompt(content, schema)
        correction_response = await self._request(
            messages=[
                {"role": "system", "content": correction_system},
                {"role": "user", "content": correction_user},
            ],
            max_tokens=max_output_tokens,
        )
        corrected, corrected_finish_reason = self._content_from_response(
            correction_response,
            context=context,
            attempt="correction",
        )
        try:
            return parse_structured_output(corrected, schema).content
        except StructuredSerializationError as exc:
            self._record_invalid_output(
                context=context,
                raw_content=corrected,
                envelope_type=exc.envelope_type,
                finish_reason=corrected_finish_reason,
                parse_error=str(exc),
                attempt="correction",
                correction_attempted=True,
            )
            raise LLMOutputError("EVL Gemma structured-output correction failed") from exc
        except StructuredSchemaError as exc:
            self._record_invalid_output(
                context=context,
                raw_content=corrected,
                envelope_type=exc.envelope_type,
                finish_reason=corrected_finish_reason,
                schema_errors=exc.validation_error.errors(include_url=False, include_input=False),
                attempt="correction",
                correction_attempted=True,
            )
            raise LLMOutputError("EVL Gemma structured-output correction failed") from exc

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.close()
