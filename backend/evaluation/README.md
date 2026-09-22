# Milestone 1.5 ranking evaluation

This framework compares ranking profiles on identical scholarly candidates. Collection runs once per benchmark query using the production provider adapters, normalization, strict year filtering, and deduplication. Ranking never calls a provider; it reads the saved `artifacts/candidate_sets.json` file.

Run commands from `backend/` with the backend virtual environment activated.

## Profiles

| Profile | Embedding model | Reranker | Shortlist | Purpose |
|---|---|---|---:|---|
| `semantic_only` | `sentence-transformers/all-MiniLM-L6-v2` | none | all candidates | Semantic baseline |
| `fast` | `sentence-transformers/all-MiniLM-L6-v2` | `cross-encoder/ms-marco-MiniLM-L6-v2` | 50 | Current FAST behavior |
| `quality_local` | `BAAI/bge-base-en-v1.5` | `cross-encoder/ms-marco-MiniLM-L12-v2` | 100 | Laptop-friendly QUALITY candidate |
| `full_bge_reference` | `BAAI/bge-m3` | `BAAI/bge-reranker-v2-m3` | 100 | Large evaluation-only reference |

`quality_local` uses the BGE v1.5 retrieval query instruction. None of these experimental definitions changes the production ranking profiles.

The full-BGE command uses local-only model loading. If either model is absent from the Hugging Face cache, it exits with a clear error instead of starting a multi-GB download. Pre-cache both models deliberately before running it.

## Workflow

```bash
python -m evaluation.benchmark collect

python -m evaluation.benchmark rank \
  --profiles semantic_only fast quality_local

python -m evaluation.benchmark prepare-labels
```

Collection uses the five queries in `benchmark_queries.json`, years 2018-2026, top-k 20, and the configured production candidate target. It writes provider statuses and candidate counts alongside each fixed paper set.

Open `evaluation/artifacts/relevance_judgments.csv` and assign every row one human label:

- `3`: highly relevant; directly addresses the topic
- `2`: relevant
- `1`: tangentially related
- `0`: irrelevant

Do not leave labels blank. Notes are optional. The CSV is the deduplicated union of top-20 results from every profile actually run. `prepare-labels` refuses to overwrite an existing file unless `--force` is supplied.

After labeling:

```bash
python -m evaluation.benchmark evaluate
```

The command rejects missing or invalid labels. It then creates `evaluation_results.json`, `evaluation_results.csv`, and `evaluation_report.md` under `evaluation/artifacts/`.

To run the large reference later, after deliberately pre-caching it:

```bash
python -m evaluation.benchmark rank --profiles full_bge_reference
```

Rerun `prepare-labels --force` only if you intentionally want to replace the judgment pool and any existing human labels.

## Metrics and interpretation

Per query and mean metrics are Precision@5, Precision@10, nDCG@10, nDCG@20, MRR, and ranking latency. Relevance >= 2 is binary-relevant for Precision and MRR. nDCG uses all grades with gain `2^relevance - 1`. Pairwise top-20 overlap includes intersection count, overlap fraction, and Jaccard similarity.

True Recall@K is not reported because the benchmark does not identify every relevant work in the literature. Raw reranker scores are saved for inspection but must not be compared across model families. The report deliberately does not select a winner.

Resource records include model names, cached weight-size estimates, initialization time, ranking latency, CUDA requirement, actual device, and basic CPU/GPU information. Peak process memory is left unmeasured because there is no reliable lightweight cross-platform probe in the current dependencies.

Generated artifacts are gitignored. Preserve `candidate_sets.json` with a particular experiment so every profile uses the exact same papers.
