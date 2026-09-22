from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.models.paper import Paper
from app.services.ranking_service import RankingService
from evaluation.profiles import EvaluationProfile

WEIGHT_SUFFIXES = {".bin", ".pt", ".safetensors"}


class Encoder(Protocol):
    def encode(self, sentences: str | list[str], **kwargs: Any) -> Any: ...


class CrossEncoderLike(Protocol):
    def predict(self, sentences: list[list[str]], **kwargs: Any) -> Any: ...


def huggingface_hub_directory() -> Path:
    if value := os.getenv("HF_HUB_CACHE"):
        return Path(value)
    if value := os.getenv("HUGGINGFACE_HUB_CACHE"):
        return Path(value)
    if value := os.getenv("HF_HOME"):
        return Path(value) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_repository_directory(model_name: str) -> Path:
    return huggingface_hub_directory() / f"models--{model_name.replace('/', '--')}"


def model_is_cached(model_name: str) -> bool:
    snapshots = model_repository_directory(model_name) / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.casefold() in WEIGHT_SUFFIXES
        for path in snapshots.rglob("*")
    )


def approximate_model_weight_bytes(model_name: str) -> int | None:
    snapshots = model_repository_directory(model_name) / "snapshots"
    if not snapshots.is_dir():
        return None
    snapshot_directories = sorted(
        (path for path in snapshots.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not snapshot_directories:
        return None
    files: dict[str, int] = {}
    for path in snapshot_directories[0].rglob("*"):
        if path.is_file() and path.suffix.casefold() in WEIGHT_SUFFIXES:
            resolved = str(path.resolve())
            files[resolved] = path.stat().st_size
    return sum(files.values()) if files else None


class EvaluationRanker:
    def __init__(
        self,
        profile: EvaluationProfile,
        encoder: Encoder | None = None,
        reranker: CrossEncoderLike | None = None,
    ) -> None:
        self.profile = profile
        self.encoder = encoder
        self.reranker = reranker
        self.initialization_seconds = 0.0

    def initialize(self) -> float:
        if self.encoder is not None and (
            self.profile.reranker_model is None or self.reranker is not None
        ):
            return self.initialization_seconds
        if self.profile.evaluation_only:
            model_names = [self.profile.embedding_model, self.profile.reranker_model]
            missing = [name for name in model_names if name and not model_is_cached(name)]
            if missing:
                raise RuntimeError(
                    "full_bge_reference will not auto-download. Pre-cache these models first: "
                    + ", ".join(missing)
                )

        started = time.perf_counter()
        from sentence_transformers import CrossEncoder, SentenceTransformer

        local_only = self.profile.evaluation_only
        self.encoder = SentenceTransformer(
            self.profile.embedding_model,
            local_files_only=local_only,
        )
        if self.profile.reranker_model:
            self.reranker = CrossEncoder(
                self.profile.reranker_model,
                local_files_only=local_only,
            )
        self.initialization_seconds = time.perf_counter() - started
        return self.initialization_seconds

    def rank(self, query: str, papers: list[Paper], limit: int) -> tuple[list[Paper], float]:
        self.initialize()
        assert self.encoder is not None
        started = time.perf_counter()
        query_text = f"{self.profile.query_prefix}{query}"
        texts = [RankingService.paper_text(paper) for paper in papers]
        query_embedding = np.asarray(
            self.encoder.encode(query_text, normalize_embeddings=True, convert_to_numpy=True)
        ).reshape(-1)
        paper_embeddings = np.asarray(
            self.encoder.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=16,
            )
        )
        semantic_scores = paper_embeddings @ query_embedding
        for paper, score in zip(papers, semantic_scores, strict=True):
            paper.semantic_score = float(score)

        semantic_order = sorted(
            papers,
            key=lambda paper: paper.semantic_score if paper.semantic_score is not None else -1e9,
            reverse=True,
        )
        if self.profile.reranker_model is None:
            duration = time.perf_counter() - started
            return semantic_order[:limit], duration

        assert self.reranker is not None
        shortlist = semantic_order[: self.profile.shortlist_size]
        pairs = [[query, RankingService.paper_text(paper)] for paper in shortlist]
        scores = np.asarray(self.reranker.predict(pairs, batch_size=8)).reshape(-1)
        for paper, score in zip(shortlist, scores, strict=True):
            paper.reranker_score = float(score)
        ranked = sorted(
            shortlist,
            key=lambda paper: (
                paper.reranker_score if paper.reranker_score is not None else -1e9,
                paper.semantic_score if paper.semantic_score is not None else -1e9,
            ),
            reverse=True,
        )[:limit]
        duration = time.perf_counter() - started
        return ranked, duration

    def resource_metadata(self) -> dict[str, Any]:
        model_names = [self.profile.embedding_model]
        if self.profile.reranker_model:
            model_names.append(self.profile.reranker_model)
        sizes = {name: approximate_model_weight_bytes(name) for name in model_names}
        device = str(getattr(self.encoder, "device", "unknown"))
        if self.reranker is not None and device == "unknown":
            device = str(getattr(getattr(self.reranker, "model", None), "device", "unknown"))
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            gpu = torch.cuda.get_device_name(0) if cuda_available else None
        except ImportError:
            cuda_available = False
            gpu = None
        return {
            "embedding_model": self.profile.embedding_model,
            "reranker_model": self.profile.reranker_model,
            "approximate_model_weight_bytes": sizes,
            "approximate_combined_model_weight_bytes": (
                sum(size for size in sizes.values() if size is not None)
                if all(size is not None for size in sizes.values())
                else None
            ),
            "model_initialization_seconds": self.initialization_seconds,
            "cuda_required": self.profile.cuda_required,
            "device_used": device,
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "cuda_available": cuda_available,
            "gpu": gpu,
            "peak_process_memory_bytes": None,
            "peak_process_memory_note": (
                "Not measured; no reliable lightweight cross-platform probe."
            ),
        }
