"""One-request probe for schema-constrained tool arguments on the EVL endpoint."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import openai
from openai import AsyncOpenAI

from app.core.config import Settings

PROJECT_ROOT = Path(__file__).parents[2]


async def main() -> None:
    settings = Settings(_env_file=PROJECT_ROOT / ".env")
    if settings.evl_gemma_api_key is None:
        raise RuntimeError("EVL_GEMMA_API_KEY is not configured")
    client = AsyncOpenAI(
        api_key=settings.evl_gemma_api_key.get_secret_value(),
        base_url=settings.evl_gemma_base_url,
        timeout=settings.evl_gemma_timeout_seconds,
        max_retries=0,
    )
    try:
        response = await client.chat.completions.create(
            model=settings.evl_gemma_model,
            messages=[{"role": "user", "content": "Call emit_result with ok=true."}],
            temperature=0,
            max_tokens=40,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "emit_result",
                        "description": "Return the structured result.",
                        "parameters": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "emit_result"}},
        )
    except openai.APIStatusError as exc:
        print(json.dumps({"supported": False, "status_code": exc.status_code}))
        return
    finally:
        await client.close()
    try:
        call = response.choices[0].message.tool_calls[0]
        arguments = json.loads(call.function.arguments)
        valid = call.function.name == "emit_result" and arguments == {"ok": True}
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError):
        valid = False
    print(json.dumps({"supported": bool(valid), "valid_arguments": bool(valid)}))


if __name__ == "__main__":
    asyncio.run(main())
