from __future__ import annotations

from collections.abc import Iterable

from app.llm.base import LLMProvider, LLMProviderUnavailableError


class UnknownLLMProviderError(ValueError):
    pass


class LLMProviderRegistry:
    def __init__(self, providers: Iterable[LLMProvider], default_provider: str = "ollama") -> None:
        self._providers = {provider.provider_id: provider for provider in providers}
        if default_provider not in self._providers:
            raise UnknownLLMProviderError(f"Unknown default LLM provider: {default_provider}")
        self.default_provider = default_provider

    def get(self, provider_id: str | None = None) -> LLMProvider:
        selected = provider_id or self.default_provider
        provider = self._providers.get(selected)
        if provider is None:
            raise UnknownLLMProviderError(f"Unknown LLM provider: {selected}")
        if not provider.configured:
            raise LLMProviderUnavailableError(f"{selected} is not configured on the backend")
        return provider

    def statuses(self) -> list[dict[str, object]]:
        return [
            {
                "provider_id": provider.provider_id,
                "model": provider.model_id,
                "configured": provider.configured,
                "cloud": provider.capabilities.cloud,
                "availability": provider.availability,
                "message": provider.availability_message,
            }
            for provider in self._providers.values()
        ]

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
