from __future__ import annotations

import threading
import unicodedata
from typing import Any, Literal, Protocol

import numpy as np

from app.models.paper import Paper


class RankingUnavailableError(RuntimeError):
    pass


class BiEncoder(Protocol):
    def encode(self, sentences: str | list[str], **kwargs: Any) -> Any: ...


class Reranker(Protocol):
    def predict(self, sentences: list[list[str]], **kwargs: Any) -> Any: ...


def rrf_score(semantic_rank: int, reranker_rank: int, k: int = 60) -> float:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    return 1 / (k + semantic_rank) + 1 / (k + reranker_rank)


def normalize_title_match_text(value: str) -> str:
    """Normalize case, punctuation, and whitespace for conservative title matching."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    without_punctuation = "".join(
        character if character.isalnum() else " " for character in normalized
    )
    return " ".join(without_punctuation.split())


def is_exact_title_match(query: str, title: str) -> bool:
    normalized_query = normalize_title_match_text(query)
    return bool(normalized_query) and normalized_query == normalize_title_match_text(title)


def prioritize_exact_title_matches(query: str, papers: list[Paper]) -> list[Paper]:
    """Move exact normalized title matches first while preserving all other ordering."""
    return sorted(papers, key=lambda paper: not is_exact_title_match(query, paper.title))


def reciprocal_rank_fusion(
    semantic_order: list[Paper], reranker_order: list[Paper], k: int = 60
) -> list[Paper]:
    """Fuse ordinal ranks without comparing incompatible model score scales."""
    semantic_positions = {id(paper): rank for rank, paper in enumerate(semantic_order, 1)}
    reranker_positions = {id(paper): rank for rank, paper in enumerate(reranker_order, 1)}
    return sorted(
        reranker_order,
        key=lambda paper: (
            -rrf_score(semantic_positions[id(paper)], reranker_positions[id(paper)], k),
            semantic_positions[id(paper)],
            reranker_positions[id(paper)],
            paper.title.casefold(),
            paper.id,
        ),
    )


class RankingService:
    def __init__(
        self,
        embedding_model_name: str,
        reranker_model_name: str,
        shortlist_size: int = 100,
        bi_encoder: BiEncoder | None = None,
        reranker: Reranker | None = None,
        fusion_mode: Literal["reranker", "rrf"] = "reranker",
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.shortlist_size = shortlist_size
        self._bi_encoder = bi_encoder
        self._reranker = reranker
        self.fusion_mode = fusion_mode
        self._load_lock = threading.Lock()

    def _load_models(self) -> None:
        if self._bi_encoder is not None and self._reranker is not None:
            return
        with self._load_lock:
            if self._bi_encoder is not None and self._reranker is not None:
                return
            try:
                from sentence_transformers import CrossEncoder, SentenceTransformer

                self._bi_encoder = self._bi_encoder or SentenceTransformer(
                    self.embedding_model_name
                )
                self._reranker = self._reranker or CrossEncoder(self.reranker_model_name)
            except Exception as exc:
                raise RankingUnavailableError(
                    "Local ranking models could not be initialized. Check disk space and network "
                    "access for the first download, plus EMBEDDING_MODEL/RERANKER_MODEL settings."
                ) from exc

    @staticmethod
    def paper_text(paper: Paper) -> str:
        text = f"Title: {paper.title}"
        if paper.abstract:
            text += f"\nAbstract: {paper.abstract}"
        return text

    def rank(self, query: str, papers: list[Paper], limit: int) -> list[Paper]:
        if not papers:
            return []
        self._load_models()
        assert self._bi_encoder is not None
        assert self._reranker is not None

        texts = [self.paper_text(paper) for paper in papers]
        try:
            query_embedding = np.asarray(
                self._bi_encoder.encode(query, normalize_embeddings=True, convert_to_numpy=True)
            ).reshape(-1)
            paper_embeddings = np.asarray(
                self._bi_encoder.encode(
                    texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=16
                )
            )
            semantic_scores = paper_embeddings @ query_embedding
            for paper, score in zip(papers, semantic_scores, strict=True):
                paper.semantic_score = float(score)

            semantic_order = sorted(
                papers,
                key=lambda paper: (
                    -(paper.semantic_score if paper.semantic_score is not None else -1e9),
                    paper.title.casefold(),
                    paper.id,
                ),
            )
            shortlist = semantic_order[: self.shortlist_size]
            shortlisted_ids = {id(paper) for paper in shortlist}
            shortlist.extend(
                paper
                for paper in semantic_order[self.shortlist_size :]
                if id(paper) not in shortlisted_ids and is_exact_title_match(query, paper.title)
            )
            pairs = [[query, self.paper_text(paper)] for paper in shortlist]
            reranker_scores = np.asarray(self._reranker.predict(pairs, batch_size=8)).reshape(-1)
            for paper, score in zip(shortlist, reranker_scores, strict=True):
                paper.reranker_score = float(score)
        except Exception as exc:
            raise RankingUnavailableError("Local ranking failed while scoring the papers.") from exc

        reranker_order = sorted(
            shortlist,
            key=lambda paper: (
                -(paper.reranker_score if paper.reranker_score is not None else -1e9),
                -(paper.semantic_score if paper.semantic_score is not None else -1e9),
                paper.title.casefold(),
                paper.id,
            ),
        )
        if self.fusion_mode == "rrf":
            final_order = reciprocal_rank_fusion(shortlist, reranker_order)
        else:
            final_order = reranker_order
        return prioritize_exact_title_matches(query, final_order)[:limit]
