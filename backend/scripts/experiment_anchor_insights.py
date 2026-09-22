"""Bounded local experiment: compact excerpt IDs instead of generated quote copies."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.document import ParsedPaper  # noqa: E402
from app.models.insights import INSIGHT_FIELDS, InsightClaim, InsightFields  # noqa: E402
from app.services.insight_selector import EvidenceSelection, select_evidence  # noqa: E402
from app.services.insight_service import InsightService, validate_insights  # noqa: E402

PAPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/parsed_documents"
    / "5824172788d0e6e1026dc00f4fb49f472bb6d87c65165a6aa2e1515ba8bcfd06.json"
)


class AnchorClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1, max_length=180)
    evidence_id: str = Field(min_length=1)


class Overview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    research_problem: list[AnchorClaim] = Field(max_length=4)
    key_contributions: list[AnchorClaim] = Field(max_length=8)
    main_findings: list[AnchorClaim] = Field(max_length=10)
    why_it_matters: list[AnchorClaim] = Field(max_length=5)
    target_audience: list[AnchorClaim] = Field(max_length=5)


class Technical(BaseModel):
    model_config = ConfigDict(extra="forbid")

    methods: list[AnchorClaim] = Field(max_length=8)
    limitations: list[AnchorClaim] = Field(max_length=8)
    future_work: list[AnchorClaim] = Field(max_length=8)


def excerpt_catalog(selection: EvidenceSelection) -> dict[str, tuple[str, str, str]]:
    catalog: dict[str, tuple[str, str, str]] = {}
    for heading, chunk_id, text in selection.chunks:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)]
        sentences = [part for part in sentences if len(part) >= 20]
        if not sentences:
            sentences = [text.strip()]
        ranked = sorted(
            enumerate(sentences),
            key=lambda pair: (
                -(
                    2 * bool(re.search(r"\d", pair[1]))
                    + bool(
                        re.search(r"\b(?:we|model|method|result|attention|train)\b", pair[1], re.I)
                    )
                ),
                pair[0],
            ),
        )
        picked = sorted({0, *(index for index, _ in ranked[:2])})[:3]
        for index in picked:
            quote = sentences[index][:180].strip()
            if len(quote) < 20:
                continue
            evidence_id = f"E{len(catalog) + 1:02d}"
            catalog[evidence_id] = (chunk_id, quote, heading)
    return catalog


async def run_call(
    client: httpx.AsyncClient,
    model: str,
    title: str,
    selection: EvidenceSelection,
    schema: type[Overview] | type[Technical],
) -> tuple[InsightFields, dict[str, object], int, int]:
    catalog = excerpt_catalog(selection)
    excerpts = "\n".join(
        f"[{evidence_id}] {heading}: {quote}"
        for evidence_id, (_, quote, heading) in catalog.items()
    )
    prompt = (
        f"Paper: {title}\nExact excerpts:\n{excerpts}\n\n"
        "Return JSON only. Extract distinct atomic claims supported directly by the excerpts. "
        "For each claim, cite one exact evidence_id from the list. Do not restate the same point, "
        "guess absent details, or fill quotas. Methods are architecture, algorithms, data, "
        "training, or evaluation procedures; never put measured scores or performance comparisons "
        "under methods. Contributions are distinct ideas or capabilities the paper claims to "
        "introduce. Findings are measured results or supported comparisons. Only return a "
        "limitation when its cited excerpt explicitly states a limitation, inability, or "
        "unresolved problem. Only return target audience when an excerpt explicitly names that "
        "audience. Use [] for unsupported fields. Inspect the full excerpt list before answering "
        "and do not stop after the first supported claim. When present, keep distinct architecture "
        "components, "
        "representations or encodings, data, optimization, and regularization as separate methods. "
        "Keep distinct tasks, metrics, measured outcomes, and comparisons as separate findings."
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Use only supplied scholarly excerpts. Claims must be specific, concise, "
                    "and supported by the cited excerpt. No outside knowledge."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "format": schema.model_json_schema(),
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 1400, "num_ctx": 8192},
    }
    started = time.perf_counter()
    response = await client.post("http://localhost:11434/api/chat", json=payload)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    parsed = schema.model_validate_json(data["message"]["content"])
    valid: dict[str, list[InsightClaim]] = {field: [] for field in INSIGHT_FIELDS}
    generated = rejected = 0
    for field in schema.model_fields:
        for item in getattr(parsed, field):
            generated += 1
            if item.evidence_id not in catalog:
                rejected += 1
                continue
            chunk_id, quote, _ = catalog[item.evidence_id]
            valid[field].append(
                InsightClaim(claim=item.claim, evidence=[{"chunk_id": chunk_id, "quote": quote}])
            )
    metrics = {
        "elapsed_s": round(elapsed, 3),
        "prompt_chars": sum(len(message["content"]) for message in payload["messages"]),
        "excerpt_count": len(catalog),
        "load_s": round(data.get("load_duration", 0) / 1e9, 3),
        "prompt_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
        "output_tokens_per_second": round(
            data.get("eval_count", 0) * 1e9 / max(data.get("eval_duration", 1), 1), 2
        ),
        "stop_reason": data.get("done_reason"),
        "generated_claims": generated,
        "bad_evidence_ids": rejected,
    }
    return InsightFields(**valid), metrics, generated, rejected


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    document = ParsedPaper.model_validate_json(
        await asyncio.to_thread(PAPER_PATH.read_text, encoding="utf-8")
    )
    selection = select_evidence(document)
    overview = InsightService._focus_selection(
        selection, (("overview", 5), ("findings", 7), ("gaps", 2)), 14
    )
    technical = InsightService._focus_selection(selection, (("methods", 14), ("gaps", 4)), 18)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=300.0) as client:
        first, first_metrics, _, first_bad = await run_call(
            client, args.model, document.title or "Untitled paper", overview, Overview
        )
        print(json.dumps({"call": 1, **first_metrics}), flush=True)
        second, second_metrics, _, second_bad = await run_call(
            client, args.model, document.title or "Untitled paper", technical, Technical
        )
        print(json.dumps({"call": 2, **second_metrics}), flush=True)
    combined = InsightFields(
        **{field: [*getattr(first, field), *getattr(second, field)] for field in INSIGHT_FIELDS}
    )
    grounded = validate_insights(combined, document)
    counts = {field: len(getattr(grounded, field)) for field in INSIGHT_FIELDS}
    print(
        json.dumps(
            {
                "total_s": round(time.perf_counter() - started, 3),
                "selected_chunks": len(selection.chunks),
                "counts": counts,
                "valid_references": sum(counts.values()),
                "bad_evidence_ids": first_bad + second_bad,
                "claims": {
                    field: [item.claim for item in getattr(grounded, field)]
                    for field in INSIGHT_FIELDS
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
