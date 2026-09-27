from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class ParsedStructuredOutput:
    content: str
    envelope_type: str
    value: Any


class StructuredSerializationError(ValueError):
    def __init__(self, message: str, *, envelope_type: str) -> None:
        super().__init__(message)
        self.envelope_type = envelope_type


class StructuredSchemaError(ValueError):
    def __init__(self, validation_error: ValidationError, *, envelope_type: str) -> None:
        super().__init__(str(validation_error))
        self.validation_error = validation_error
        self.envelope_type = envelope_type


def normalize_json_envelope(content: str) -> tuple[str, str]:
    """Normalize only a single harmless JSON/generic Markdown fence."""
    stripped = content.strip()
    lines = stripped.splitlines()
    if (
        len(lines) >= 3
        and lines[0].strip().casefold() in {"```json", "```"}
        and lines[-1].strip() == "```"
        and all("```" not in line for line in lines[1:-1])
    ):
        envelope_type = (
            "json_fence" if lines[0].strip().casefold() == "```json" else "generic_fence"
        )
        return "\n".join(lines[1:-1]).strip(), envelope_type
    if "```" in stripped:
        return stripped, "mixed_or_embedded_fence"
    return stripped, "plain"


def parse_structured_output(
    content: str,
    schema: type[BaseModel],
) -> ParsedStructuredOutput:
    normalized, envelope_type = normalize_json_envelope(content)
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise StructuredSerializationError(str(exc), envelope_type=envelope_type) from exc
    try:
        schema.model_validate(value)
    except ValidationError as exc:
        raise StructuredSchemaError(exc, envelope_type=envelope_type) from exc
    return ParsedStructuredOutput(content=normalized, envelope_type=envelope_type, value=value)


def format_correction_prompt(raw_content: str, schema: type[BaseModel]) -> tuple[str, str]:
    schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    system_prompt = (
        "You correct JSON serialization only. Preserve the semantic content, claim text, "
        "categories, and evidence IDs. Only re-emit the same response as syntactically valid JSON "
        "matching the provided structural contract. Do not add, remove, summarize, rewrite, or "
        "reclassify scientific content or citations. Return one JSON object and nothing else."
    )
    user_prompt = (
        "Re-emit the response below as valid JSON. Preserve every claim and citation exactly.\n\n"
        f"STRUCTURAL CONTRACT:\n{schema_json}\n\n"
        f"ORIGINAL RESPONSE:\n{raw_content}"
    )
    return system_prompt, user_prompt
