from __future__ import annotations

import math

import numpy as np

from app.models.paper import Paper
from app.services.graph_semantics import (
    MAX_RELATED_CONCEPTS,
    GraphSemanticsService,
    extract_concept_candidates,
)


class ConceptEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, sentences: str | list[str], **kwargs: object) -> np.ndarray:
        assert isinstance(sentences, list)
        self.calls.append((sentences, kwargs))
        vectors: list[list[float]] = []
        for index, text in enumerate(sentences):
            if index == 0:
                similarity = 1.0
            elif "street visualization" in text.casefold():
                similarity = 0.96
            elif "urban network" in text.casefold():
                similarity = 0.82
            else:
                similarity = 0.25
            vectors.append([similarity, math.sqrt(max(0.0, 1 - similarity**2))])
        return np.asarray(vectors)


def _papers() -> list[Paper]:
    return [
        Paper(
            id="one",
            title="Street Visualization for Urban Networks",
            abstract=(
                "This paper presents a study of street networks and trajectory visualization. "
                "The method supports multivariate urban exploration."
            ),
        ),
        Paper(
            id="two",
            title="Street Networks and Declarative Grammars",
            abstract="Research results connect urban networks with visual grammar systems.",
        ),
    ]


def test_candidates_come_from_result_papers_and_filter_generic_noise() -> None:
    candidates = extract_concept_candidates(_papers())
    lowered = {candidate.casefold() for candidate in candidates}
    generic = {"paper", "study", "method", "results", "data", "analysis", "research"}

    assert "street visualization" in lowered
    assert "urban networks" in lowered
    assert not (generic & lowered)
    assert all(
        phrase.split()[0] not in generic and phrase.split()[-1] not in generic for phrase in lowered
    )


def test_normalized_singular_plural_variants_are_deduplicated() -> None:
    candidates = extract_concept_candidates(_papers())
    network_variants = [
        candidate
        for candidate in candidates
        if candidate.casefold() in {"street network", "street networks"}
    ]

    assert len(network_variants) == 1


def test_concepts_use_one_batched_encoder_call_and_original_query_ranking() -> None:
    encoder = ConceptEncoder()
    service = GraphSemanticsService("existing-m1-model", encoder)

    concepts = service.related_concepts("street visualization", _papers())

    assert concepts[0].text.casefold() == "street visualization"
    assert concepts[0].query_similarity == 0.96
    assert all(
        left.query_similarity >= right.query_similarity
        for left, right in zip(concepts, concepts[1:], strict=False)
    )
    assert len(concepts) <= MAX_RELATED_CONCEPTS
    assert len(encoder.calls) == 1
    inputs, kwargs = encoder.calls[0]
    assert inputs[0] == "street visualization"
    assert len(inputs) > 2
    assert kwargs == {
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "batch_size": 64,
    }


def test_requested_limit_and_global_maximum_are_respected() -> None:
    service = GraphSemanticsService("existing-m1-model", ConceptEncoder())

    assert len(service.related_concepts("street visualization", _papers(), limit=3)) == 3
    assert len(service.related_concepts("street visualization", _papers(), limit=100)) <= 7
