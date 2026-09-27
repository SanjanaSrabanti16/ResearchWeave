from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

ContextStrategy = Literal["full_document", "hierarchical_sections"]


@dataclass(frozen=True)
class LLMProviderCapabilities:
    structured_json: bool
    context_strategy: ContextStrategy
    cloud: bool
    max_context_chars: int
    max_extraction_packets: int
    supports_thinking_control: bool = False
    supports_format_correction: bool = False


@dataclass(frozen=True)
class LLMRequestContext:
    stage: Literal["evidence_extraction", "synthesis"]
    paper_id: str
    paper_title: str
    run_id: str
    extraction_pass: str | None = None


class LLMProviderError(RuntimeError):
    """Base class for safe, provider-neutral generation failures."""


class LLMProviderUnavailableError(LLMProviderError):
    pass


class LLMTimeoutError(LLMProviderError):
    pass


class LLMRateLimitError(LLMProviderError):
    pass


class LLMAuthenticationError(LLMProviderError):
    pass


class LLMOutputError(LLMProviderError):
    pass


class LLMOutputBudgetExceeded(LLMOutputError):
    """The provider stopped because the requested output exceeded its token budget."""


class LLMProvider(ABC):
    provider_id: str
    model_id: str
    capabilities: LLMProviderCapabilities

    @property
    def last_generation_call_count(self) -> int:
        return 1

    @property
    def last_format_correction_used(self) -> bool:
        return False

    @property
    @abstractmethod
    def configured(self) -> bool: ...

    @abstractmethod
    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        max_output_tokens: int,
        context: LLMRequestContext | None = None,
    ) -> str: ...

    async def aclose(self) -> None:
        return None
