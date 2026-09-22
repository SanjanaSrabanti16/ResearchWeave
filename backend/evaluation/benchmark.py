from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import hashlib
import itertools
import json
import math
import statistics
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.core.config import Settings
from app.models.paper import Paper
from app.providers import ArxivProvider, OpenAlexProvider, SemanticScholarProvider
from app.providers.base import ProviderError, SearchProvider
from app.services.deduplication_service import DeduplicationService
from app.services.search_service import SearchService
from evaluation.metrics import ndcg_at_k, precision_at_k, reciprocal_rank, top_k_overlap
from evaluation.pooling import JUDGMENT_FIELDS, pool_top_results, validate_relevance_labels
from evaluation.profiles import EvaluationProfile, resolve_profiles
from evaluation.ranking import EvaluationRanker

EVALUATION_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_PATH = EVALUATION_DIRECTORY / "benchmark_queries.json"
DEFAULT_ARTIFACT_DIRECTORY = EVALUATION_DIRECTORY / "artifacts"
CANDIDATE_SETS_FILE = "candidate_sets.json"
JUDGMENTS_FILE = "relevance_judgments.csv"
SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS = 1.05


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_benchmark(path: Path = DEFAULT_BENCHMARK_PATH) -> dict[str, Any]:
    benchmark = load_json(path)
    required = {"start_year", "end_year", "top_k", "queries"}
    if not required.issubset(benchmark):
        raise ValueError(f"Benchmark file must contain: {', '.join(sorted(required))}")
    query_ids = [row.get("id") for row in benchmark["queries"]]
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("Benchmark query IDs must be present and unique")
    return benchmark


async def _collect_from_provider(
    provider: SearchProvider,
    query: str,
    limit: int,
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    try:
        papers = await provider.search(query, limit, start_year, end_year)
        return {"name": provider.name, "status": "ok", "warning": None, "papers": papers}
    except ProviderError as exc:
        return {
            "name": provider.name,
            "status": "error",
            "warning": str(exc),
            "papers": [],
        }
    except Exception:
        return {
            "name": provider.name,
            "status": "error",
            "warning": f"{provider.name} failed unexpectedly",
            "papers": [],
        }


async def collect_candidate_sets(
    benchmark: dict[str, Any],
    providers: Sequence[SearchProvider],
    candidate_target: int,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if not providers:
        raise ValueError("At least one provider is required")
    provider_limit = min(100, max(20, math.ceil(candidate_target / len(providers))))
    deduplicator = DeduplicationService()
    query_outputs: list[dict[str, Any]] = []
    has_semantic_scholar = any(provider.name == "semantic_scholar" for provider in providers)
    last_semantic_scholar_request_at: float | None = None

    for query_spec in benchmark["queries"]:
        if has_semantic_scholar and last_semantic_scholar_request_at is not None:
            elapsed = monotonic() - last_semantic_scholar_request_at
            await sleep(max(0.0, SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS - elapsed))
        if has_semantic_scholar:
            last_semantic_scholar_request_at = monotonic()
        results = await asyncio.gather(
            *(
                _collect_from_provider(
                    provider,
                    query_spec["query"],
                    provider_limit,
                    benchmark["start_year"],
                    benchmark["end_year"],
                )
                for provider in providers
            )
        )
        if all(result["status"] == "error" for result in results):
            raise RuntimeError(f"All providers failed for benchmark query {query_spec['id']}")

        candidates = [paper for result in results for paper in result["papers"]]
        filtered = [
            paper
            for paper in candidates
            if SearchService._within_years(
                paper,
                benchmark["start_year"],
                benchmark["end_year"],
            )
        ]
        deduplicated = deduplicator.deduplicate(filtered)
        query_outputs.append(
            {
                "query_id": query_spec["id"],
                "query": query_spec["query"],
                "start_year": benchmark["start_year"],
                "end_year": benchmark["end_year"],
                "provider_status": {result["name"]: result["status"] for result in results},
                "warnings": [result["warning"] for result in results if result["warning"]],
                "original_candidate_count": len(candidates),
                "filtered_candidate_count": len(filtered),
                "deduplicated_count": len(deduplicated),
                "final_saved_candidate_count": len(deduplicated),
                "papers": [paper.model_dump(mode="json") for paper in deduplicated],
            }
        )

    return {
        "schema_version": 1,
        "collected_at": datetime.now(UTC).isoformat(),
        "benchmark_file": DEFAULT_BENCHMARK_PATH.name,
        "start_year": benchmark["start_year"],
        "end_year": benchmark["end_year"],
        "top_k": benchmark["top_k"],
        "candidate_target": candidate_target,
        "effective_per_provider_limit": provider_limit,
        "providers": [provider.name for provider in providers],
        "queries": query_outputs,
    }


async def collect_command(artifact_directory: Path) -> None:
    benchmark = load_benchmark()
    settings = Settings()
    timeout = httpx.Timeout(settings.provider_timeout_seconds)
    async with httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": "research-landscape-explorer-evaluation/0.1"},
        follow_redirects=True,
    ) as client:
        providers: list[SearchProvider] = [
            OpenAlexProvider(
                client,
                retries=settings.provider_retries,
                api_key=settings.openalex_api_key,
            ),
            SemanticScholarProvider(
                client,
                retries=settings.provider_retries,
                api_key=settings.semantic_scholar_api_key,
            ),
            ArxivProvider(client, retries=settings.provider_retries),
        ]
        candidate_sets = await collect_candidate_sets(
            benchmark,
            providers,
            settings.candidate_target,
        )
    output_path = artifact_directory / CANDIDATE_SETS_FILE
    write_json(output_path, candidate_sets)
    print(f"Saved fixed candidate sets to {output_path}")
    for query in candidate_sets["queries"]:
        print(
            f"{query['query_id']}: original={query['original_candidate_count']} "
            f"deduplicated={query['deduplicated_count']} "
            f"saved={query['final_saved_candidate_count']} "
            f"providers={query['provider_status']}"
        )


def candidate_set_fingerprint(papers: list[dict[str, Any]]) -> str:
    payload = json.dumps(papers, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


RankerFactory = Callable[[EvaluationProfile], Any]


def rank_candidate_sets(
    candidate_sets: dict[str, Any],
    profile_names: list[str],
    ranker_factory: RankerFactory = EvaluationRanker,
) -> dict[str, dict[str, Any]]:
    profiles = resolve_profiles(profile_names)
    documents: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        ranker = ranker_factory(profile)
        initialization_seconds = float(ranker.initialize())
        query_outputs: list[dict[str, Any]] = []
        durations: list[float] = []
        for query_data in candidate_sets["queries"]:
            source_papers = query_data["papers"]
            fingerprint = candidate_set_fingerprint(source_papers)
            papers = [Paper.model_validate(row) for row in source_papers]
            ranked, duration = ranker.rank(
                query_data["query"],
                papers,
                candidate_sets["top_k"],
            )
            durations.append(duration)
            query_outputs.append(
                {
                    "query_id": query_data["query_id"],
                    "query": query_data["query"],
                    "candidate_set_fingerprint": fingerprint,
                    "candidate_count": len(source_papers),
                    "ranking_duration_seconds": duration,
                    "results": [
                        {
                            "query_id": query_data["query_id"],
                            "paper_id": paper.id,
                            "rank": index,
                            "title": paper.title,
                            "year": paper.publication_year,
                            "venue": paper.venue,
                            "doi": paper.doi,
                            "arxiv_id": paper.arxiv_id,
                            "semantic_score": paper.semantic_score,
                            "reranker_score": paper.reranker_score,
                            "ranking_profile": profile.name,
                            "ranking_duration_seconds": duration,
                        }
                        for index, paper in enumerate(ranked, 1)
                    ],
                }
            )
        resources = ranker.resource_metadata()
        resources["model_initialization_seconds"] = initialization_seconds
        documents[profile.name] = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "ranking_profile": profile.name,
            "embedding_model": profile.embedding_model,
            "reranker_model": profile.reranker_model,
            "shortlist_size": profile.shortlist_size,
            "query_prefix": profile.query_prefix,
            "source_candidate_file": CANDIDATE_SETS_FILE,
            "mean_ranking_duration_seconds": statistics.fmean(durations),
            "resources": resources,
            "queries": query_outputs,
        }
        del ranker
        gc.collect()
    return documents


def rank_command(artifact_directory: Path, profile_names: list[str]) -> None:
    candidate_path = artifact_directory / CANDIDATE_SETS_FILE
    if not candidate_path.exists():
        raise FileNotFoundError(f"Run collect first; missing {candidate_path}")
    documents = rank_candidate_sets(load_json(candidate_path), profile_names)
    ranking_directory = artifact_directory / "rankings"
    for profile, document in documents.items():
        output_path = ranking_directory / f"{profile}.json"
        write_json(output_path, document)
        size = document["resources"]["approximate_combined_model_weight_bytes"]
        size_text = f"{size / 1_000_000:.1f} MB" if size is not None else "unknown"
        print(
            f"{profile}: mean ranking={document['mean_ranking_duration_seconds']:.3f}s, "
            f"initialization={document['resources']['model_initialization_seconds']:.3f}s, "
            f"weights={size_text}; saved {output_path}"
        )


def load_ranking_documents(artifact_directory: Path) -> dict[str, dict[str, Any]]:
    ranking_directory = artifact_directory / "rankings"
    documents = {path.stem: load_json(path) for path in sorted(ranking_directory.glob("*.json"))}
    if not documents:
        raise FileNotFoundError(f"No ranking files found in {ranking_directory}; run rank first")
    return documents


def prepare_labels_command(artifact_directory: Path, force: bool = False) -> Path:
    candidate_path = artifact_directory / CANDIDATE_SETS_FILE
    if not candidate_path.exists():
        raise FileNotFoundError(f"Run collect first; missing {candidate_path}")
    output_path = artifact_directory / JUDGMENTS_FILE
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists; refusing to overwrite possible human labels. "
            "Use --force only if replacement is intentional."
        )
    rows = pool_top_results(
        load_json(candidate_path),
        load_ranking_documents(artifact_directory),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=JUDGMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} unlabeled pooled judgments to {output_path}")
    print("Fill every relevance cell with 0, 1, 2, or 3 before running evaluate.")
    return output_path


def read_judgment_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_evaluation_results(
    candidate_sets: dict[str, Any],
    ranking_documents: dict[str, dict[str, Any]],
    judgment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = validate_relevance_labels(judgment_rows)
    labels_by_query: dict[str, list[int]] = {}
    for (query_id, _), relevance in labels.items():
        labels_by_query.setdefault(query_id, []).append(relevance)

    per_query: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, float]] = {}
    for profile, document in ranking_documents.items():
        profile_rows: list[dict[str, Any]] = []
        for query_result in document["queries"]:
            query_id = query_result["query_id"]
            relevances = [
                labels[(query_id, result["paper_id"])] for result in query_result["results"]
            ]
            ideal = labels_by_query[query_id]
            row = {
                "profile": profile,
                "query_id": query_id,
                "query": query_result["query"],
                "precision_at_5": precision_at_k(relevances, 5),
                "precision_at_10": precision_at_k(relevances, 10),
                "ndcg_at_10": ndcg_at_k(relevances, 10, ideal),
                "ndcg_at_20": ndcg_at_k(relevances, 20, ideal),
                "mrr": reciprocal_rank(relevances),
                "ranking_duration_seconds": query_result["ranking_duration_seconds"],
            }
            profile_rows.append(row)
            per_query.append(row)
        metric_names = [
            "precision_at_5",
            "precision_at_10",
            "ndcg_at_10",
            "ndcg_at_20",
            "mrr",
            "ranking_duration_seconds",
        ]
        aggregate[profile] = {
            f"mean_{name}": statistics.fmean(row[name] for row in profile_rows)
            for name in metric_names
        }

    overlaps: list[dict[str, Any]] = []
    queries = {row["query_id"]: row["query"] for row in candidate_sets["queries"]}
    profile_query_results = {
        profile: {row["query_id"]: row["results"] for row in document["queries"]}
        for profile, document in ranking_documents.items()
    }
    for left, right in itertools.combinations(ranking_documents, 2):
        for query_id, query in queries.items():
            left_ids = [row["paper_id"] for row in profile_query_results[left][query_id]]
            right_ids = [row["paper_id"] for row in profile_query_results[right][query_id]]
            overlaps.append(
                {
                    "left_profile": left,
                    "right_profile": right,
                    "query_id": query_id,
                    "query": query,
                    **top_k_overlap(left_ids, right_ids, 20),
                }
            )

    aggregate_overlap: list[dict[str, Any]] = []
    for left, right in itertools.combinations(ranking_documents, 2):
        pair_rows = [
            row for row in overlaps if row["left_profile"] == left and row["right_profile"] == right
        ]
        aggregate_overlap.append(
            {
                "left_profile": left,
                "right_profile": right,
                "mean_intersection_count": statistics.fmean(
                    row["intersection_count"] for row in pair_rows
                ),
                "mean_overlap_fraction": statistics.fmean(
                    row["overlap_fraction"] for row in pair_rows
                ),
                "mean_jaccard": statistics.fmean(row["jaccard"] for row in pair_rows),
            }
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "relevance_threshold_for_binary_metrics": 2,
        "ndcg_gain": "2^relevance - 1",
        "profiles": list(ranking_documents),
        "per_query": per_query,
        "aggregate": aggregate,
        "top_20_overlap": overlaps,
        "aggregate_top_20_overlap": aggregate_overlap,
        "resources": {
            profile: document["resources"] for profile, document in ranking_documents.items()
        },
        "notes": [
            "No true Recall@K is reported because the literature-wide relevant set is unknown.",
            "Raw reranker scores are not compared across model families.",
            "No winner is selected automatically.",
        ],
    }


def render_evaluation_report(results: dict[str, Any]) -> str:
    lines = [
        "# Ranking evaluation report",
        "",
        "Human labels use 0-3 graded relevance. Precision and MRR treat relevance >= 2 as "
        "relevant. nDCG uses gain `2^relevance - 1`.",
        "",
        "| Profile | P@5 | P@10 | nDCG@10 | nDCG@20 | MRR | Ranking time | Model weights |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in results["profiles"]:
        metrics = results["aggregate"][profile]
        resources = results["resources"][profile]
        size = resources["approximate_combined_model_weight_bytes"]
        size_text = f"{size / 1_000_000:.1f} MB" if size is not None else "unknown"
        lines.append(
            f"| {profile} | {metrics['mean_precision_at_5']:.3f} | "
            f"{metrics['mean_precision_at_10']:.3f} | {metrics['mean_ndcg_at_10']:.3f} | "
            f"{metrics['mean_ndcg_at_20']:.3f} | {metrics['mean_mrr']:.3f} | "
            f"{metrics['mean_ranking_duration_seconds']:.3f}s | {size_text} |"
        )
    lines.extend(
        [
            "",
            "No winner is declared automatically. A production QUALITY decision must balance "
            "relevance, download size, CPU latency, and memory requirements.",
            "",
            "True Recall@K is not reported because the benchmark does not establish every "
            "relevant paper in the literature. See `evaluation_results.json` for per-query "
            "metrics, resource details, and pairwise top-20 overlap.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_outputs(artifact_directory: Path, results: dict[str, Any]) -> None:
    write_json(artifact_directory / "evaluation_results.json", results)
    csv_path = artifact_directory / "evaluation_results.csv"
    fields = [
        "profile",
        "query_id",
        "query",
        "precision_at_5",
        "precision_at_10",
        "ndcg_at_10",
        "ndcg_at_20",
        "mrr",
        "ranking_duration_seconds",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results["per_query"])
        for profile in results["profiles"]:
            metrics = results["aggregate"][profile]
            writer.writerow(
                {
                    "profile": profile,
                    "query_id": "MEAN",
                    "query": "All benchmark queries",
                    **{name.removeprefix("mean_"): value for name, value in metrics.items()},
                }
            )
    report_path = artifact_directory / "evaluation_report.md"
    report_path.write_text(render_evaluation_report(results), encoding="utf-8")


def evaluate_command(artifact_directory: Path) -> None:
    candidate_path = artifact_directory / CANDIDATE_SETS_FILE
    judgment_path = artifact_directory / JUDGMENTS_FILE
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing {candidate_path}; run collect first")
    if not judgment_path.exists():
        raise FileNotFoundError(f"Missing {judgment_path}; run prepare-labels first")
    results = build_evaluation_results(
        load_json(candidate_path),
        load_ranking_documents(artifact_directory),
        read_judgment_rows(judgment_path),
    )
    write_evaluation_outputs(artifact_directory, results)
    print(f"Saved evaluation results to {artifact_directory}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Milestone 1.5 ranking evaluation")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
        help="Generated artifact directory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("collect", help="Collect one fixed candidate set per query")
    rank_parser = subparsers.add_parser("rank", help="Rank saved candidates")
    rank_parser.add_argument("--profiles", nargs="+", required=True)
    labels_parser = subparsers.add_parser("prepare-labels", help="Create unlabeled CSV pool")
    labels_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("evaluate", help="Calculate metrics from completed human labels")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            asyncio.run(collect_command(args.artifacts))
        elif args.command == "rank":
            rank_command(args.artifacts, args.profiles)
        elif args.command == "prepare-labels":
            prepare_labels_command(args.artifacts, args.force)
        elif args.command == "evaluate":
            evaluate_command(args.artifacts)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
