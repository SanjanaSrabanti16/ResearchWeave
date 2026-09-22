from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationProfile:
    name: str
    embedding_model: str
    reranker_model: str | None
    shortlist_size: int | None
    query_prefix: str = ""
    evaluation_only: bool = False
    cuda_required: bool = False


PROFILES = {
    "semantic_only": EvaluationProfile(
        name="semantic_only",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model=None,
        shortlist_size=None,
    ),
    "fast": EvaluationProfile(
        name="fast",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        shortlist_size=50,
    ),
    "quality_local": EvaluationProfile(
        name="quality_local",
        embedding_model="BAAI/bge-base-en-v1.5",
        reranker_model="cross-encoder/ms-marco-MiniLM-L12-v2",
        shortlist_size=100,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "full_bge_reference": EvaluationProfile(
        name="full_bge_reference",
        embedding_model="BAAI/bge-m3",
        reranker_model="BAAI/bge-reranker-v2-m3",
        shortlist_size=100,
        evaluation_only=True,
    ),
}


def resolve_profiles(names: list[str]) -> list[EvaluationProfile]:
    unknown = [name for name in names if name not in PROFILES]
    if unknown:
        choices = ", ".join(PROFILES)
        raise ValueError(f"Unknown profile(s): {', '.join(unknown)}. Choose from: {choices}")
    return [PROFILES[name] for name in dict.fromkeys(names)]
