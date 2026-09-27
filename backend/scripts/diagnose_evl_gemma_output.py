"""One-request EVL output diagnostic for the cached Milestone 2 paper."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.core.config import Settings
from app.llm import EVLGemmaProvider, LLMOutputError
from app.services.insight_context import build_paper_context_packets
from app.services.paper_understanding_service import (
    _EVIDENCE_SYSTEM_PROMPT,
    _evidence_prompt,
    _EvidenceExtraction,
)
from scripts.validate_m2_evl_gemma import PROJECT_ROOT, _load_target

OUTPUT_PATH = PROJECT_ROOT / "validation" / "evl_gemma_output_diagnostic.json"


class _RecordingCompletions:
    def __init__(self, completions: Any) -> None:
        self._completions = completions
        self.raw_content: str | None = None

    async def create(self, **kwargs: Any) -> Any:
        response = await self._completions.create(**kwargs)
        self.raw_content = response.choices[0].message.content
        return response


def _diagnostic_candidate(content: str) -> tuple[str, str]:
    stripped = content.strip()
    lines = stripped.splitlines()
    if (
        len(lines) >= 3
        and lines[0].strip().casefold() in {"```json", "```"}
        and lines[-1].strip() == "```"
        and all("```" not in line for line in lines[1:-1])
    ):
        fence = "json" if lines[0].strip().casefold() == "```json" else "generic"
        return "\n".join(lines[1:-1]).strip(), f"single_{fence}_fence"
    return stripped, "plain_or_unsupported_envelope"


async def main() -> None:
    settings = Settings(_env_file=PROJECT_ROOT / ".env")
    if settings.evl_gemma_api_key is None:
        raise RuntimeError("EVL_GEMMA_API_KEY is not configured")
    document = _load_target()
    real_client = AsyncOpenAI(
        api_key=settings.evl_gemma_api_key.get_secret_value(),
        base_url=settings.evl_gemma_base_url,
        timeout=settings.evl_gemma_timeout_seconds,
        max_retries=0,
    )
    recorder = _RecordingCompletions(real_client.chat.completions)
    provider = EVLGemmaProvider(
        api_key="configured",
        base_url=settings.evl_gemma_base_url,
        model_id=settings.evl_gemma_model,
        timeout_seconds=settings.evl_gemma_timeout_seconds,
        client=SimpleNamespace(chat=SimpleNamespace(completions=recorder)),
    )
    packets = build_paper_context_packets(
        document,
        provider.capabilities,
        settings.ollama_batch_chars,
    )
    if len(packets) != 1:
        raise RuntimeError(f"Expected one EVL evidence packet, found {len(packets)}")

    adapter_result = "accepted"
    adapter_error = None
    try:
        await provider.generate_structured(
            system_prompt=_EVIDENCE_SYSTEM_PROMPT,
            user_prompt=_evidence_prompt(document, packets[0]),
            schema=_EvidenceExtraction,
            max_output_tokens=5000,
        )
    except LLMOutputError as exc:
        adapter_result = "rejected"
        adapter_error = str(exc)
    finally:
        await real_client.close()

    if not isinstance(recorder.raw_content, str):
        raise RuntimeError("EVL returned no assistant text to diagnose")
    candidate, envelope = _diagnostic_candidate(recorder.raw_content)
    syntax_error = None
    schema_errors: list[dict[str, Any]] = []
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        syntax_error = {
            "message": exc.msg,
            "line": exc.lineno,
            "column": exc.colno,
            "position": exc.pos,
        }
    else:
        try:
            _EvidenceExtraction.model_validate(parsed)
        except ValidationError as exc:
            schema_errors = [
                {
                    "location": list(error["loc"]),
                    "type": error["type"],
                    "message": error["msg"],
                }
                for error in exc.errors(include_url=False, include_input=False)
            ]

    payload = {
        "paper": document.title,
        "provider": provider.provider_id,
        "model": provider.model_id,
        "adapter_result": adapter_result,
        "adapter_error": adapter_error,
        "envelope": envelope,
        "syntax_error": syntax_error,
        "schema_errors": schema_errors,
        "raw_assistant_content": recorder.raw_content,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "raw_assistant_content"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Diagnostic artifact: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
