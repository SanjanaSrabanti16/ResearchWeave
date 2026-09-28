# Graph semantics: Milestone 3.1–3.6

This specification freezes node relevance, size, information opacity, paper similarity, and sparse
edge selection. It does not define layout, a graph API, or graph UI behavior.

## Existing M1 ranking audit

- `semantic_score` is the normalized bi-encoder query/document dot product, which is cosine
  similarity. Its theoretical range is `[-1, 1]`. The document text is title plus abstract when
  available.
- `reranker_score` is the cross-encoder's raw, model-dependent output. It is not normalized and has
  no fixed universal range.
- FAST ranking may use Reciprocal Rank Fusion (RRF):
  `1 / (60 + semantic_rank) + 1 / (60 + reranker_rank)`. RRF compares ordinal positions, is used
  only for ordering, and is not stored on the paper or exposed by the API.
- Exact-title priority normalizes Unicode, case, punctuation, and whitespace, then moves normalized
  exact matches ahead of the model-produced order. It does not alter either model score.
- The search API and frontend currently expose/display `semantic_score` and `reranker_score`.
- These ranking values are query-specific. Provider identity, citation count, year, venue, and PDF
  availability do not contribute ranking scores.
- Query variants retrieve candidates, but final M1 ranking is anchored to the original request
  query. M3 graph scoring independently receives that same original query.

M3 does not change M1 candidate selection, shortlist construction, scores, fusion, exact-title
priority, or final ordering.

## Frozen node semantics

- **Node size:** relevance of the paper to the original user query.
- **Node opacity:** completeness of usable information ResearchWeave currently has for the paper.
- **Edge weight:** paper-to-paper semantic similarity, independent of the user query.
- **Edge existence:** deterministic thresholded top-K semantic-neighborhood union.

Node size never represents citations, year, venue, provider, popularity, PDF availability, or
paper-insight completeness.

## Canonical query relevance

`query_relevance` is provider-independent and lies in `[0, 1]`:

1. Normalize the query and paper title with the existing exact-title normalization. Equality gives
   `query_relevance = 1.0` deterministically.
2. Otherwise, encode one batch containing the original query, every paper title, and every paper's
   stable text: `Title: <title>` plus `Abstract: <abstract>` when available.
3. Request normalized embeddings from the configured M1 bi-encoder model and compute both title
   cosine similarity and title-plus-abstract cosine similarity with the original-query vector.
4. A conservative strong partial-title match supplies a `0.90` floor only when the query has at
   least two meaningful title tokens, all occur in the title, and at least one is distinctive
   rather than a generic scholarly/title term.
5. Use `max(title_similarity, document_similarity, applicable_title_floor)`, clamped to `[0, 1]`.
   There is no result-set min-max normalization.

`node_weight` equals `query_relevance`.

## Node radius

Constants:

- `MIN_RADIUS = 8`
- `MAX_RADIUS = 28`

Radius is area-aware and monotonic:

`node_radius = sqrt(MIN_RADIUS² + node_weight × (MAX_RADIUS² - MIN_RADIUS²))`

Thus `0.0 → 8`, `1.0 → 28`, and greater relevance can never produce a smaller radius. The same
query and paper input produce the same seed. Missing abstracts use title-only scoring.

## Information completeness and node opacity

`information_completeness` and `node_opacity` are identical discrete values. The highest explicitly
satisfied state wins:

- metadata/title only: `0.40`
- non-empty abstract available: `0.55`
- full PDF successfully parsed: `0.78`
- structured research insights available: `1.00`

Parsed-document and insight availability must be supplied explicitly by canonical paper ID. The
graph layer never infers those states from URLs, filenames, providers, citations, year, relevance,
or model confidence.

## Paper-to-paper semantic similarity

All selected papers are encoded in one batch with the same configured M1 sentence-transformer
bi-encoder used for semantic retrieval and query relevance. Each stable representation is
`Title: <title>` plus `Abstract: <abstract>` when available; missing abstracts fall back to title.

For every distinct pair:

`paper_similarity(A, B) = clamp(cosine(embedding(A), embedding(B)), 0, 1)`

The score is symmetric, provider-independent, and query-independent. It has no result-set min-max
normalization and never includes query relevance, reranker scores, citations, year, venue, PDF
state, or insight state. `edge_weight` equals `paper_similarity`.

## Sparse edge selection

For each paper, order all other papers by descending similarity with paper ID as the deterministic
tie breaker. Consider only its first `K_NEIGHBORS = 3`, discard candidates below
`MIN_EDGE_SIMILARITY = 0.35`, then form the undirected union. Mutual selection is not required and
isolated nodes are valid. Each edge is stored once in canonical paper-ID order and records whether
the canonical source selected the target (`source_top_k`), the target selected the source
(`target_top_k`), or both selected each other (`both_top_k`).
