from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

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
from app.services.provider_reliability import CircuitBreaker, CircuitOpenError


def _gemini_json_schema(value: object) -> object:
    """Remove JSON Schema keywords unsupported by Gemini structured output."""
    if isinstance(value, dict):
        return {
            key: _gemini_json_schema(item)
            for key, item in value.items()
            if key not in {"additionalProperties", "default", "minLength", "maxLength"}
        }
    if isinstance(value, list):
        return [_gemini_json_schema(item) for item in value]
    return value


class GeminiProvider(LLMProvider):
    provider_id = "gemini"
    capabilities = LLMProviderCapabilities(
        structured_json=True,
        context_strategy="full_document",
        cloud=True,
        max_context_chars=180000,
        max_extraction_packets=1,
    )

    def __init__(
        self,
        api_key: str | None,
        model_id: str,
        timeout_seconds: float = 180.0,
        client: Any | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._api_key = api_key
        self.model_id = model_id
        self._client = client
        self._owns_client = client is None
        self._timeout_seconds = timeout_seconds
        self._sleep = sleep
        self._jitter = jitter or (lambda: random.uniform(0, 0.5))
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._availability: LLMProviderAvailability = (
            "configured" if self.configured else "unconfigured"
        )

    @property
    def configured(self) -> bool:
        return bool(self._api_key) or self._client is not None

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

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise LLMProviderUnavailableError(
                "Google Gemini is not configured; set GEMINI_API_KEY on the backend"
            )
        from google import genai
        from google.genai import types

        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(
                timeout=int(self._timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        return self._client

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        max_output_tokens: int,
        context: LLMRequestContext | None = None,
    ) -> str:
        del context
        client = self._get_client()
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_json_schema=_gemini_json_schema(schema.model_json_schema()),
            temperature=0,
            max_output_tokens=max_output_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = None
        operation_key = (self.provider_id, "generation")
        for attempt in range(3):
            try:
                self._circuit_breaker.before_call(operation_key)
                response = await client.aio.models.generate_content(
                    model=self.model_id,
                    contents=user_prompt,
                    config=config,
                )
                self._circuit_breaker.record_success(operation_key)
                self._availability = "available"
                break
            except CircuitOpenError as exc:
                self._availability = "temporarily_unavailable"
                raise LLMProviderUnavailableError("Google Gemini is unavailable") from exc
            except (TimeoutError, httpx.TimeoutException) as exc:
                if attempt < 2:
                    await self._sleep((2**attempt) + self._jitter())
                    continue
                self._circuit_breaker.record_failure(operation_key, "timeout")
                self._availability = "temporarily_unavailable"
                raise LLMTimeoutError("Google Gemini extraction timed out") from exc
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                safe_error_kind = str(getattr(exc, "details", "")).upper()
                invalid_key = (
                    "API_KEY_INVALID" in safe_error_kind or "API KEY NOT VALID" in safe_error_kind
                )
                if code in {401, 403} or (code == 400 and invalid_key):
                    self._circuit_breaker.record_failure(operation_key, "authentication")
                    self._availability = "authentication_error"
                    raise LLMAuthenticationError("Google Gemini authentication failed") from exc
                category = (
                    "rate_limited"
                    if code == 429
                    else "timeout"
                    if code in {408, 504}
                    else "server_error"
                    if code in {500, 502, 503}
                    else "invalid_response"
                )
                if category in {"rate_limited", "timeout", "server_error"} and attempt < 2:
                    await self._sleep((2**attempt) * 2 + self._jitter())
                    continue
                self._circuit_breaker.record_failure(operation_key, category)
                self._availability = "temporarily_unavailable"
                if code == 429:
                    raise LLMRateLimitError("Google Gemini rate limit was reached") from exc
                if code in {408, 504}:
                    raise LLMTimeoutError("Google Gemini extraction timed out") from exc
                if code == 404 or (isinstance(code, int) and code >= 500):
                    raise LLMProviderUnavailableError("Google Gemini is unavailable") from exc
                raise LLMOutputError("Google Gemini rejected structured generation") from exc
            except (httpx.RequestError, OSError) as exc:
                if attempt < 2:
                    await self._sleep((2**attempt) + self._jitter())
                    continue
                self._circuit_breaker.record_failure(operation_key, "network_error")
                self._availability = "temporarily_unavailable"
                raise LLMProviderUnavailableError("Google Gemini is unavailable") from exc
        if (
            response is None
        ):  # Defensive: every loop exit above returns, raises, or sets a response.
            raise LLMProviderUnavailableError("Google Gemini is unavailable")
        content = getattr(response, "text", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMOutputError("Google Gemini returned an invalid structured response")
        try:
            schema.model_validate_json(content)
        except ValidationError as exc:
            raise LLMOutputError("Google Gemini returned invalid structured output") from exc
        return content

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        context: LLMRequestContext | None = None,
    ) -> str:
        del context
        client = self._get_client()
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
            max_output_tokens=max_output_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = None
        operation_key = (self.provider_id, "generation")
        for attempt in range(3):
            try:
                self._circuit_breaker.before_call(operation_key)
                response = await client.aio.models.generate_content(
                    model=self.model_id,
                    contents=user_prompt,
                    config=config,
                )
                self._circuit_breaker.record_success(operation_key)
                self._availability = "available"
                break
            except CircuitOpenError as exc:
                self._availability = "temporarily_unavailable"
                raise LLMProviderUnavailableError("Google Gemini is unavailable") from exc
            except (TimeoutError, httpx.TimeoutException) as exc:
                if attempt < 2:
                    await self._sleep((2**attempt) + self._jitter())
                    continue
                self._circuit_breaker.record_failure(operation_key, "timeout")
                self._availability = "temporarily_unavailable"
                raise LLMTimeoutError("Google Gemini extraction timed out") from exc
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                safe_error_kind = str(getattr(exc, "details", "")).upper()
                invalid_key = (
                    "API_KEY_INVALID" in safe_error_kind or "API KEY NOT VALID" in safe_error_kind
                )
                if code in {401, 403} or (code == 400 and invalid_key):
                    self._circuit_breaker.record_failure(operation_key, "authentication")
                    self._availability = "authentication_error"
                    raise LLMAuthenticationError("Google Gemini authentication failed") from exc
                category = (
                    "rate_limited"
                    if code == 429
                    else "timeout"
                    if code in {408, 504}
                    else "server_error"
                    if code in {500, 502, 503}
                    else "invalid_response"
                )
                if category in {"rate_limited", "timeout", "server_error"} and attempt < 2:
                    await self._sleep((2**attempt) * 2 + self._jitter())
                    continue
                self._circuit_breaker.record_failure(operation_key, category)
                self._availability = "temporarily_unavailable"
                if code == 429:
                    raise LLMRateLimitError("Google Gemini rate limit was reached") from exc
                if code in {408, 504}:
                    raise LLMTimeoutError("Google Gemini extraction timed out") from exc
                if code == 404 or (isinstance(code, int) and code >= 500):
                    raise LLMProviderUnavailableError("Google Gemini is unavailable") from exc
                raise LLMOutputError("Google Gemini rejected text generation") from exc
            except (httpx.RequestError, OSError) as exc:
                if attempt < 2:
                    await self._sleep((2**attempt) + self._jitter())
                    continue
                self._circuit_breaker.record_failure(operation_key, "network_error")
                self._availability = "temporarily_unavailable"
                raise LLMProviderUnavailableError("Google Gemini is unavailable") from exc
        if response is None:
            raise LLMProviderUnavailableError("Google Gemini is unavailable")
        finish_reason = ""
        try:
            finish_reason = str(response.candidates[0].finish_reason).upper()
        except (AttributeError, IndexError, TypeError):
            pass
        if "MAX_TOKENS" in finish_reason or "LENGTH" in finish_reason:
            raise LLMOutputBudgetExceeded("Google Gemini output budget was exceeded")
        content = getattr(response, "text", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMOutputError("Google Gemini returned empty text output")
        return content

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            close = getattr(self._client.aio, "aclose", None)
            if close is not None:
                await close()
