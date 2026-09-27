from __future__ import annotations

import httpx
from pydantic import BaseModel

from app.llm.base import (
    LLMOutputError,
    LLMProvider,
    LLMProviderCapabilities,
    LLMProviderUnavailableError,
    LLMRequestContext,
    LLMTimeoutError,
)


class OllamaProvider(LLMProvider):
    provider_id = "ollama"
    capabilities = LLMProviderCapabilities(
        structured_json=True,
        context_strategy="hierarchical_sections",
        cloud=False,
        max_context_chars=26000,
        max_extraction_packets=12,
        supports_thinking_control=True,
    )

    def __init__(self, client: httpx.AsyncClient, base_url: str, model_id: str) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id

    @property
    def configured(self) -> bool:
        return True

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
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": schema.model_json_schema(),
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": max_output_tokens,
                "num_ctx": 8192,
            },
        }
        try:
            response = await self.client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Local Ollama extraction timed out") from exc
        except httpx.RequestError as exc:
            raise LLMProviderUnavailableError("Local Ollama is unavailable") from exc
        if response.status_code >= 400:
            if response.status_code == 404 or response.status_code >= 500:
                raise LLMProviderUnavailableError(
                    f"Local Ollama or model {self.model_id} is unavailable"
                )
            raise LLMOutputError("Local Ollama rejected structured generation")
        try:
            content = response.json()["message"]["content"]
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMOutputError("Local Ollama returned an invalid response") from exc
        if not isinstance(content, str):
            raise LLMOutputError("Local Ollama returned an invalid response")
        return content
