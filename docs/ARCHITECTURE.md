# ResearchWeave architecture

This document describes the implemented `v0.1.0-alpha` research-prototype architecture. Detailed graph rules remain frozen in [`backend/GRAPH_SEMANTICS.md`](../backend/GRAPH_SEMANTICS.md).

## System boundaries

The React/TypeScript frontend owns interaction and visualization state. FastAPI owns authoritative scholarly records, graph seeds, parsed documents, provider availability, and cached insights. External scholarly and AI systems are accessed only through backend adapters; credentials are never returned to the browser.

## M1 search flow

1. `SearchService` creates the original query plus at most three deterministic lexical variants.
2. OpenAlex, Semantic Scholar, and arXiv adapters retrieve independently. A failed provider is reported through provider health while successful results continue.
3. Provider records are normalized into `PaperCandidate` values and year bounds are enforced locally.
4. Deduplication resolves one canonical paper by normalized DOI, then version-free arXiv ID, then exact normalized title. A merged paper retains its provider identities and best available metadata.
5. A bi-encoder scores the original query against canonical papers. A cross-encoder reranks the configured shortlist; the Fast profile uses reciprocal-rank fusion. Exact normalized title matches receive deterministic priority without altering model scores.
6. Normalized provider responses are cached in SQLite by provider, query, years, and retrieval limit.

Provider health is explicit: each provider reports successful and failed variant requests plus an `ok`, `cached`, or `error` state. Search fails only when no provider can supply results or required local ranking cannot initialize.

## M3 semantic graph

The graph is built from the already ranked canonical papers; it does not retrieve or deduplicate again. The configured local bi-encoder computes provider-independent query relevance and pairwise paper similarity from titles and abstracts. Deterministic thresholded top-K neighbor selection produces a sparse undirected edge set.

Node size represents query relevance. Node opacity represents the highest backend-confirmed information state: metadata, abstract, parsed PDF, or validated insights. Edge weight represents paper-to-paper semantic similarity only. Citations, provider identity, PDF availability, and LLM confidence do not alter those meanings.

## PDF acquisition and `ParsedPaper`

For one canonical paper, the acquisition service tries all known approved PDF candidates, then arXiv, then Unpaywall when a DOI and email are configured. Each HTTPS response and redirect is checked against the public-network and scholarly-host policy, streamed through a size limit, checked for PDF-compatible content type, and verified by PDF signature. Invalid candidates fall through; upload remains the final user-controlled route.

Validated bytes are sent to local GROBID. The TEI response becomes a `ParsedPaper` containing bibliographic metadata, abstract, hierarchical sections, references, stable chunks, text offsets, PDF SHA-256, and parser provenance. Page numbers remain null when reliable mapping is unavailable. Parsed documents are cached by PDF fingerprint and parser version.

## M2 evidence-grounded insights

Ollama, Gemini, and EVL Gemma implement one provider interface with capability-aware context planning. The model first proposes categorized evidence claims. Backend validators require canonical chunk IDs, exact normalized quotes, current-paper attribution, and field-specific semantic support. Invalid claims are rejected individually. Only the resulting validated evidence ledger is provided to final synthesis, and final citations are mapped back to canonical chunks.

Insight cache identity includes document fingerprint, provider, model, and extraction version. Changing any of them prevents stale analysis from masquerading as current output.

## Backend-authoritative state synchronization

The graph response marks information completeness from backend cache state. Selecting a paper asks the backend for its current parsed document and validated insights. After parsing or analysis, the frontend refreshes that authoritative state and rebuilds graph display state rather than treating browser memory as canonical. This prevents navigation, aborted requests, or reloads from silently losing completed analysis.

## Deployment shape

Docker Compose builds a Python 3.11 FastAPI image and a Node 20-built static frontend served by nginx. GROBID is an opt-in `pdf` profile. Ollama is an optional separately managed local service; Gemini and EVL Gemma are optional remote services. The supported shape is currently one application process with local filesystem/SQLite caches, not a horizontally scaled production deployment.
