from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.llm.base import (
    LLMAuthenticationError,
    LLMOutputError,
    LLMProvider,
    LLMProviderCapabilities,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMRequestContext,
    LLMTimeoutError,
)


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
    ) -> None:
        self._api_key = api_key
        self.model_id = model_id
        self._client = client
        self._owns_client = client is None
        self._timeout_seconds = timeout_seconds
        self._sleep = sleep
        self._jitter = jitter or (lambda: random.uniform(0, 0.5))

    @property
    def configured(self) -> bool:
        return bool(self._api_key) or self._client is not None

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
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=self.model_id,
                    contents=user_prompt,
                    config=config,
                )
                break
            except (TimeoutError, httpx.TimeoutException) as exc:
                raise LLMTimeoutError("Google Gemini extraction timed out") from exc
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                safe_error_kind = str(getattr(exc, "details", "")).upper()
                invalid_key = (
                    "API_KEY_INVALID" in safe_error_kind or "API KEY NOT VALID" in safe_error_kind
                )
                if code in {401, 403} or (code == 400 and invalid_key):
                    raise LLMAuthenticationError("Google Gemini authentication failed") from exc
                if code in {429, 503} and attempt < 2:
                    await self._sleep((2**attempt) * 2 + self._jitter())
                    continue
                if code == 429:
                    raise LLMRateLimitError("Google Gemini rate limit was reached") from exc
                if code in {408, 504}:
                    raise LLMTimeoutError("Google Gemini extraction timed out") from exc
                if code == 404 or (isinstance(code, int) and code >= 500):
                    raise LLMProviderUnavailableError("Google Gemini is unavailable") from exc
                raise LLMOutputError("Google Gemini rejected structured generation") from exc
            except (httpx.RequestError, OSError) as exc:
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

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            close = getattr(self._client.aio, "aclose", None)
            if close is not None:
                await close()
