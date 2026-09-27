"""One-request capability probe for EVL OpenAI-compatible JSON mode."""

from __future__ import annotations

import asyncio
import json

import openai
from openai import AsyncOpenAI

from app.core.config import Settings
from scripts.validate_m2_evl_gemma import PROJECT_ROOT


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
            messages=[
                {
                    "role": "user",
                    "content": 'Return exactly this JSON object: {"ok": true}',
                }
            ],
            temperature=0,
            max_tokens=20,
            response_format={"type": "json_object"},
        )
    except openai.APIStatusError as exc:
        print(json.dumps({"supported": False, "status_code": exc.status_code}))
        return
    finally:
        await client.close()
    content = response.choices[0].message.content
    valid = False
    if isinstance(content, str):
        try:
            valid = json.loads(content) == {"ok": True}
        except json.JSONDecodeError:
            pass
    print(json.dumps({"supported": True, "valid_json": valid, "content": content}))


if __name__ == "__main__":
    asyncio.run(main())
