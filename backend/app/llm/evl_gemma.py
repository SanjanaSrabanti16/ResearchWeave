from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Awaitable, Callable
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
    LLMProviderAvailability,
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
from app.services.provider_reliability import CircuitBreaker, CircuitOpenError

logger = logging.getLogger(__name__)


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
        circuit_breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] | None = None,
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
        self._availability: LLMProviderAvailability = (
            "configured" if self.configured else "unconfigured"
        )
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._sleep = sleep
        self._jitter = jitter or (lambda: random.uniform(0.0, 0.25))

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
        return bool(self.base_url) and (bool(self._api_key) or self._client is not None)

    @property
    def availability(self) -> LLMProviderAvailability:
        return self._availability

    @property
    def availability_message(self) -> str:
        return {
            "configured": "Configured; availability has not been checked yet",
            "unconfigured": "Not configured",
            "available": "Available",
            "temporarily_unavailable": "Temporarily unavailable",
            "authentication_error": "Authentication failed",
        }[self._availability]

    def _set_availability(self, state: LLMProviderAvailability) -> None:
        self._availability = state
        logger.info(
            "llm_provider_status provider=%s configured=%s model=%s availability=%s",
            self.provider_id,
            self.configured,
            self.model_id,
            state,
        )

    @property
    def last_generation_call_count(self) -> int:
        return self._last_generation_call_count

    @property
    def last_format_correction_used(self) -> bool:
        return self._last_format_correction_used

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.configured:
            self._set_availability("unconfigured")
            raise LLMProviderUnavailableError(
                "EVL Gemma is not configured; set EVL_GEMMA_API_KEY and "
                "EVL_GEMMA_BASE_URL on the backend"
            )
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self.base_url,
            timeout=self._timeout_seconds,
            max_retries=0,
        )
        return self._client

    async def _request(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        json_mode: bool = True,
    ) -> Any:
        client = self._get_client()
        operation_key = (self.provider_id, "generation")
        request: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        for attempt in range(2):
            try:
                self._circuit_breaker.before_call(operation_key)
                response = await client.chat.completions.create(**request)
            except CircuitOpenError as exc:
                self._set_availability("temporarily_unavailable")
                raise LLMProviderUnavailableError("EVL Gemma is temporarily unavailable") from exc
            except openai.AuthenticationError as exc:
                self._circuit_breaker.record_failure(operation_key, "authentication")
                self._set_availability("authentication_error")
                raise LLMAuthenticationError("EVL Gemma authentication failed") from exc
            except openai.APIStatusError as exc:
                status = exc.status_code
                if status in {401, 403}:
                    self._circuit_breaker.record_failure(operation_key, "authentication")
                    self._set_availability("authentication_error")
                    raise LLMAuthenticationError("EVL Gemma authentication failed") from exc
                category = (
                    "rate_limited"
                    if status == 429
                    else "timeout"
                    if status in {408, 504}
                    else "server_error"
                    if status in {500, 502, 503}
                    else "invalid_response"
                )
                if category in {"rate_limited", "timeout", "server_error"} and attempt == 0:
                    await self._sleep(1.0 + self._jitter())
                    continue
                self._circuit_breaker.record_failure(operation_key, category)
                self._set_availability("temporarily_unavailable")
                if category == "rate_limited":
                    raise LLMRateLimitError("EVL Gemma rate limit was reached") from exc
                if category == "timeout":
                    raise LLMTimeoutError("EVL Gemma extraction timed out") from exc
                if category == "server_error":
                    raise LLMProviderUnavailableError(
                        "EVL Gemma is temporarily unavailable"
                    ) from exc
                self._set_availability("available")
                raise LLMOutputError("EVL Gemma rejected structured generation") from exc
            except (openai.APITimeoutError, TimeoutError) as exc:
                if attempt == 0:
                    await self._sleep(1.0 + self._jitter())
                    continue
                self._circuit_breaker.record_failure(operation_key, "timeout")
                self._set_availability("temporarily_unavailable")
                raise LLMTimeoutError("EVL Gemma extraction timed out") from exc
            except (openai.APIConnectionError, OSError) as exc:
                if attempt == 0:
                    await self._sleep(1.0 + self._jitter())
                    continue
                self._circuit_breaker.record_failure(operation_key, "network_error")
                self._set_availability("temporarily_unavailable")
                raise LLMProviderUnavailableError("EVL Gemma is temporarily unavailable") from exc
            self._circuit_breaker.record_success(operation_key)
            self._set_availability("available")
            return response
        raise LLMProviderUnavailableError("EVL Gemma is temporarily unavailable")

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

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        context: LLMRequestContext | None = None,
    ) -> str:
        self._last_generation_call_count = 1
        self._last_format_correction_used = False
        response = await self._request(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_output_tokens,
            json_mode=False,
        )
        content, finish_reason = self._content_from_response(
            response, context=context, attempt="original"
        )
        if str(finish_reason).casefold() == "length":
            self._record_invalid_output(
                context=context,
                raw_content=content,
                envelope_type="truncated_output",
                finish_reason=finish_reason,
                parse_error="output_budget_exceeded",
            )
            raise LLMOutputBudgetExceeded("EVL Gemma output budget was exceeded")
        return content

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.close()
