"""Profile one real parsed paper with cold/warm uncached and cache-hit extraction.

Run from backend: python scripts/profile_insights.py --model qwen3:4b --runs 3
The script uses an in-memory cache so existing application artifacts are untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.document import ParsedPaper  # noqa: E402
from app.models.insights import INSIGHT_FIELDS, InsightFields  # noqa: E402
from app.services.insight_selector import evidence_catalog, select_evidence  # noqa: E402
from app.services.insight_service import (  # noqa: E402
    InsightService,
    _OverviewInsights,
    restore_selected_chunk_prefixes,
    validate_insights,
)

PAPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/parsed_documents"
    / "5824172788d0e6e1026dc00f4fb49f472bb6d87c65165a6aa2e1515ba8bcfd06.json"
)


class MemoryCache:
    def __init__(self) -> None:
        self.data: dict[str, InsightFields] = {}

    @staticmethod
    def key(fingerprint: str, model: str, version: str) -> str:
        return f"{fingerprint}:{model}:{version}"

    def get(self, key: str) -> InsightFields | None:
        return self.data.get(key)

    def put(self, key: str, insights: InsightFields) -> None:
        self.data[key] = insights


def count_generated_evidence(messages: list[str], selection) -> tuple[int, int]:
    """Count model references accepted/rejected before final claim deduplication."""
    focused = (
        InsightService._focus_selection(
            selection, (("overview", 5), ("findings", 7), ("gaps", 2)), 14
        ),
        InsightService._focus_selection(selection, (("methods", 14), ("gaps", 4)), 18),
    )
    generated = accepted = 0
    for content, evidence in zip(messages, focused, strict=True):
        payload = json.loads(content)
        catalog = evidence_catalog(evidence)
        items = [item for field in INSIGHT_FIELDS for item in payload.get(field, [])]
        generated += len(items)
        accepted += sum(item.get("evidence_id") in catalog for item in items)
    return accepted, generated - accepted


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--overview-only", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument(
        "--paper",
        type=Path,
        default=PAPER_PATH,
    )
    args = parser.parse_args()
    document = ParsedPaper.model_validate_json(
        await asyncio.to_thread(args.paper.read_text, encoding="utf-8")
    )
    selection = select_evidence(document)
    print(
        json.dumps(
            {
                "paper": document.title,
                "selected_chunks": len(selection.chunks),
                "selected_chars": sum(len(text) for _, _, text in selection.chunks),
            }
        ),
        flush=True,
    )
    if args.select_only:
        technical = InsightService._focus_selection(selection, (("methods", 14), ("gaps", 4)), 18)
        print(
            json.dumps(
                {
                    "all_selected": [
                        {"heading": heading, "id": chunk_id}
                        for heading, chunk_id, _ in selection.chunks
                    ],
                    "method_pool": selection.pools["methods"],
                    "technical": [
                        {"heading": heading, "id": chunk_id}
                        for heading, chunk_id, _ in technical.chunks
                    ],
                }
            ),
            flush=True,
        )
        return
    latest_metrics: dict[str, object] = {}
    all_metrics: list[dict[str, object]] = []
    raw_messages: list[str] = []
    calls = 0

    async def record(response: httpx.Response) -> None:
        nonlocal calls, latest_metrics
        calls += 1
        await response.aread()
        try:
            data = response.json()
            latest_metrics = {
                key: data.get(key)
                for key in (
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                    "total_duration",
                    "done_reason",
                )
            }
            latest_metrics["status"] = response.status_code
            latest_metrics["thinking_returned"] = bool(data.get("message", {}).get("thinking"))
            content = data.get("message", {}).get("content", "")
            raw_messages.append(content)
            latest_metrics["response_chars"] = len(content)
            latest_metrics["response_tail"] = content[-120:]
            request = json.loads(response.request.content)
            latest_metrics["prompt_chars"] = sum(
                len(message["content"]) for message in request["messages"]
            )
            duration = data.get("eval_duration")
            count = data.get("eval_count")
            if isinstance(duration, int) and duration and isinstance(count, int):
                latest_metrics["tokens_per_second"] = round(count * 1e9 / duration, 2)
        except ValueError:
            latest_metrics = {"status": response.status_code}
        all_metrics.append(latest_metrics)
        print(json.dumps({"completed_call": calls, "ollama": latest_metrics}), flush=True)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(600.0), event_hooks={"response": [record]}
    ) as client:
        if args.overview_only:
            service = InsightService(client, args.base_url, args.model, MemoryCache())
            overview = service._focus_selection(
                selection, (("overview", 5), ("findings", 7), ("gaps", 2)), 14
            )
            raw = await service._extract_batch(
                document,
                overview,
                _OverviewInsights,
                "research_problem, key_contributions, main_findings, why_it_matters, "
                "and target_audience",
            )
            selected_ids = {chunk_id for _, chunk_id, _ in overview.chunks}
            valid = validate_insights(
                restore_selected_chunk_prefixes(raw, selected_ids), document, selected_ids
            )
            print(
                json.dumps(
                    {
                        "selected": [chunk_id for _, chunk_id, _ in overview.chunks],
                        "raw": {
                            field: [item.model_dump() for item in getattr(raw, field)]
                            for field in _OverviewInsights.model_fields
                        },
                        "validated_counts": {
                            field: len(getattr(valid, field))
                            for field in _OverviewInsights.model_fields
                        },
                    }
                ),
                flush=True,
            )
            return
        for run in range(1, args.runs + 1):
            first_call = len(all_metrics)
            cache = MemoryCache()
            service = InsightService(client, args.base_url, args.model, cache)
            stages: list[tuple[str, float]] = []
            start = time.perf_counter()

            async def progress(stage: str, *, recorded=stages, started: float = start) -> None:
                recorded.append((stage, time.perf_counter() - started))

            try:
                result = await service.extract(document, on_progress=progress)
                elapsed = time.perf_counter() - start
                counts = {field: len(getattr(result.insights, field)) for field in INSIGHT_FIELDS}
                refs = sum(
                    len(item.evidence)
                    for field in INSIGHT_FIELDS
                    for item in getattr(result.insights, field)
                )
                accepted, rejected = count_generated_evidence(raw_messages[first_call:], selection)
                print(
                    json.dumps(
                        {
                            "run": run,
                            "success": True,
                            "elapsed_s": round(elapsed, 3),
                            "calls": calls,
                            "stages": stages,
                            "counts": counts,
                            "validated_references": refs,
                            "accepted_before_dedup": accepted,
                            "rejected_references": rejected,
                            "ollama": all_metrics[first_call:],
                            "sample_claims": {
                                field: [item.claim for item in getattr(result.insights, field)[:3]]
                                for field in ("methods", "key_contributions", "main_findings")
                            },
                            "all_claims": {
                                field: [item.claim for item in getattr(result.insights, field)]
                                for field in INSIGHT_FIELDS
                            },
                        }
                    ),
                    flush=True,
                )
                cache_start = time.perf_counter()
                hit = await service.extract(document)
                print(
                    json.dumps(
                        {
                            "run": run,
                            "cache_hit": hit.cached,
                            "cache_elapsed_s": round(time.perf_counter() - cache_start, 3),
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "run": run,
                            "success": False,
                            "elapsed_s": round(time.perf_counter() - start, 3),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "calls": calls,
                            "stages": stages,
                            "ollama": all_metrics[first_call:],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    asyncio.run(main())
