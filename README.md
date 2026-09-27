# Research Landscape Explorer

Research Landscape Explorer is an open-source, local-first foundation for finding the scholarly literature most relevant to a research question. It searches multiple free scholarly indexes, converts their records to one schema, removes duplicates, and applies a two-stage open-source relevance ranker.

This repository implements Milestone 1 retrieval/ranking, Milestone 2A PDF acquisition/parsing, and **Milestone 2B provider-independent, evidence-grounded paper insights**. It does not synthesize across papers, cluster topics, or draw research graphs.

## Milestone 1 features

- Bounded, deterministic query variants searched across OpenAlex, Semantic Scholar, and arXiv
- Provider-specific parsing behind one `SearchProvider` interface
- Normalized DOI, arXiv ID, title, author, date, venue, link, and citation metadata
- Deterministic deduplication by DOI, then arXiv ID, then exact normalized title
- Optional inclusive year filters and 5–50 final results
- Laptop-friendly Fast ranking by default, with an optional BGE Quality profile
- Semantic/reranker rank fusion in Fast; both raw scores retained for inspection
- 24-hour SQLite cache for normalized provider responses
- FastAPI API and a responsive React search/results interface
- Independent provider failure handling and user-visible warnings
- Offline unit tests with mocked providers and ranking models

## Milestone 2A features

- Open-access PDF acquisition from an existing approved repository link, arXiv ID, or DOI through the free Unpaywall API
- Explicit upload fallback for a PDF the user is authorized to use
- HTTPS, public-network, redirect, content-type, PDF-signature, timeout, and 50 MB size checks
- Local GROBID 0.9.0 CRF parsing into title, authors, abstract, sections, references, and stable text chunks
- Parser provenance and SHA-256 source fingerprints on every parsed document
- A parser-versioned parsed-document cache independent of the scholarly search cache
- Per-result PDF controls and a compact structured-document preview

## Milestone 2B features

- A shared paper-understanding pipeline with local Ollama, Google Gemini, and EVL Gemma adapters
- Evidence-first extraction, deterministic validation, and synthesis from a validated evidence ledger
- Ten structured insight fields, including paper overview and evaluation, with chunk-ID/quote evidence
- Programmatic chunk-ID and normalized exact-quote validation; unsupported claims are removed
- Provider-capability-aware context planning plus provider/model/version-separated caching
- Minimal provider selection and expandable evidence in the existing PDF panel

## Architecture

```text
React UI
   │ POST /api/search
FastAPI route
   │
SearchService
   ├── SQLite cache ── provider adapters ── OpenAlex / S2 / arXiv
   ├── local year filter
   ├── deterministic deduplication
   ├── profile-selected bi-encoder cosine-similarity shortlist
   └── profile-selected cross-encoder ── top papers + both scores
```

The backend separates HTTP routes (`app/api`), API/domain models (`app/models`), external adapters (`app/providers`), persistence (`app/db`), and pipeline logic (`app/services`). Adding Crossref later means implementing the existing provider interface and registering it in `app/main.py`.

## Tech stack

- Python 3.11+, FastAPI, Pydantic, HTTPX, SQLAlchemy, SQLite
- sentence-transformers, PyTorch, Hugging Face models, NumPy
- React 18, TypeScript, Vite
- pytest, Ruff, Vitest, ESLint

Search, ranking, PDF parsing, and local Ollama extraction require no paid service. Gemini and EVL Gemma are optional remote providers configured with server-side access keys.

Milestone 2B defaults to a separately installed local Ollama server and `qwen3:1.7b`. Run `ollama pull qwen3:1.7b`, then start Ollama if needed. The backend defaults to `http://localhost:11434`; Docker Desktop Compose reaches the host at `http://host.docker.internal:11434`. Local Ollama keeps selected paper text on the local machine.

To enable Gemini, create an API key in Google AI Studio and set `GEMINI_API_KEY` only in the private backend `.env`; never put it in `VITE_*` variables or browser storage. `GEMINI_MODEL` defaults to `gemini-3.5-flash`. Gemini sends the selected paper context and validated evidence ledger to Google's API for cloud analysis. The provider selector marks Gemini unavailable when no server-side key is configured.

To enable the EVL-hosted Gemma service, set `EVL_GEMMA_API_KEY` only in the private backend `.env`. `EVL_GEMMA_BASE_URL` defaults to `https://sage200.evl.uic.edu`, and `EVL_GEMMA_MODEL` defaults to `gemma4`. Selected paper context is sent to the EVL inference service. The provider selector marks EVL Gemma unavailable when no server-side key is configured.

## Local setup

Requirements: Python 3.11+, Node.js 20+, and npm. A GPU is optional; CPU inference works but can be slow. Clone the project and optionally copy `.env.example` to `.env`.

### Backend

From the repository root:

```bash
cd backend
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux), then:

```bash
python -m pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/health` for health information or `http://localhost:8000/docs` for the generated API UI.

The first real search downloads the selected embedding and reranker models from Hugging Face. They are cached locally and reused across requests. The default Fast models are suitable for normal laptops. The optional Quality profile downloads several GB of model files and is intended for powerful hardware.

### Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The normal development command uses Vite's local-only binding. To deliberately expose the development server to other machines on the network, run `npm run dev:host` instead and review your firewall/network settings first.

### Docker

Optionally create `.env`, then run:

```bash
docker compose up --build
```

The Compose volume persists Hugging Face downloads outside the image. The frontend is built as static production assets and served by nginx; it does not run the Vite development server. The UI is at port 5173 and the API at port 8000. `VITE_API_URL` is supplied as a frontend image build argument and defaults to `http://localhost:8000`. Local development does not require Docker.

Normal Compose startup leaves the optional PDF parser off, so search remains lightweight and works when GROBID is unavailable. To start the full Milestone 2A stack, including the pinned CPU-only GROBID image:

```bash
docker compose --profile pdf up --build
```

GROBID is then available locally on port 8070. Its first image pull is large and can take several minutes depending on the connection. The repository does not vendor GROBID models or PDF data. If you run the backend outside Compose, the equivalent standalone parser command is:

```bash
docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.1-crf
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEMANTIC_SCHOLAR_API_KEY` | empty | Optional S2 key for higher rate limits |
| `OPENALEX_API_KEY` | empty | Optional OpenAlex key, sent as `api_key`; unauthenticated use remains supported |
| `UNPAYWALL_EMAIL` | empty | Email required by Unpaywall for DOI-based OA PDF lookup |
| `RANKING_PROFILE` | `fast` | Ranking defaults: `fast` or `quality` |
| `EMBEDDING_MODEL` | profile default | Optional explicit bi-encoder override |
| `RERANKER_MODEL` | profile default | Optional explicit cross-encoder override |
| `CACHE_TTL_HOURS` | `24` | Search cache lifetime |
| `CANDIDATE_TARGET` | `240` | Total desired pre-deduplication retrieval pool |
| `RERANK_SHORTLIST_SIZE` | profile default | Optional explicit shortlist-size override |
| `RANKING_FUSION_MODE` | profile default | `rrf` for rank fusion or `reranker` to restore raw reranker ordering |
| `VITE_API_URL` | `http://localhost:8000` | Browser-visible backend origin |
| `PDF_MAX_SIZE_MB` | `50` | Maximum downloaded or uploaded PDF size |
| `PDF_DOWNLOAD_TIMEOUT_SECONDS` | `30` | Remote PDF request timeout |
| `PDF_ALLOWED_HOSTS` | approved repository list | Comma-separated hosts accepted for metadata PDF links |
| `GROBID_URL` | `http://localhost:8070` | Local GROBID REST base URL |
| `GROBID_TIMEOUT_SECONDS` | `120` | Full-document parser timeout |
| `GROBID_PARSER_VERSION` | `0.9.1-crf+frontmatter-v1` | Parser/cache provenance version |
| `PARSED_DOCUMENT_CACHE_DIR` | `backend/data/parsed_documents` | Structured parsed-document cache |
| `LLM_PROVIDER` | `ollama` | Default insight provider: `ollama`, `gemini`, or `evl_gemma` |
| `OLLAMA_MODEL` | `qwen3:1.7b` | Local model used for structured paper insights |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local-only Ollama address for a host-run backend |
| `OLLAMA_DOCKER_BASE_URL` | `http://host.docker.internal:11434` | Local Ollama address passed into the Compose backend |
| `OLLAMA_TIMEOUT_SECONDS` | `180` | Per-call Ollama timeout; CPU-only extraction can still be slow |
| `OLLAMA_BATCH_CHARS` | `13000` | Maximum selected evidence-chunk text per extraction |
| `GEMINI_API_KEY` | empty | Server-side Google AI Studio key; never returned to the browser |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Gemini model used for paper insights |
| `GEMINI_TIMEOUT_SECONDS` | `180` | Per-call Gemini timeout |
| `EVL_GEMMA_API_KEY` | empty | Server-side EVL access key; never returned to the browser |
| `EVL_GEMMA_BASE_URL` | `https://sage200.evl.uic.edu` | OpenAI-compatible EVL service base URL |
| `EVL_GEMMA_MODEL` | `gemma4` | EVL-hosted model used for paper insights |
| `EVL_GEMMA_TIMEOUT_SECONDS` | `180` | Per-call EVL Gemma timeout |
| `INSIGHT_CACHE_DIR` | `backend/data/insights` | Validated insight cache |

Ranking profiles resolve centrally in backend settings:

| Profile | Embedding model | Reranker | Shortlist | Final order |
| --- | --- | --- | ---: | --- |
| `fast` (default) | `sentence-transformers/all-MiniLM-L6-v2` | `cross-encoder/ms-marco-MiniLM-L6-v2` | 100 | RRF |
| `quality` | `BAAI/bge-m3` | `BAAI/bge-reranker-v2-m3` | 100 | Reranker |

Fast is recommended for ordinary laptops. Quality is optional for powerful hardware and downloads several GB. Ranking quality will be scientifically compared on an evaluation set; the project does not assume that the larger Quality profile is always necessary. The model, shortlist, and fusion settings can override their selected profile defaults explicitly. Hardware auto-detection is not implemented.

## Scholarly retrieval

Each provider receives the original query (up to 100 results) and up to three deterministic lexical variants (up to 40 results each). Variants normalize plurals, remove function words, compress an intervening modifier, and expand common acronyms; they do not use an LLM or topic-specific rules. Providers run concurrently, while each provider's variants run in order. Live Semantic Scholar requests are spaced at least one second apart and arXiv searches at least three seconds apart within the backend process. Calls retain bounded retries for network, rate-limit, and server errors. OpenAlex uses Works search and date filters; Semantic Scholar uses paper relevance search and its `year` filter; arXiv uses its Atom API and is filtered locally by year. Every external field is treated as optional and validated before entering the pipeline.

Normalized results are cached by provider, whitespace/case-normalized query, start year, end year, and provider retrieval limit. A failed provider produces a warning while successful sources continue. The service returns HTTP 503 only when every provider fails or local ranking cannot initialize. Because this project is pre-release and the cache is disposable, startup drops and recreates the `search_cache` table once when it detects the older schema without `retrieval_limit`.

References: [OpenAlex API documentation](https://docs.openalex.org/), [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/api-docs/), and [arXiv API documentation](https://info.arxiv.org/help/api/index.html).

## PDF acquisition and parsing

For each canonical search result, **Get PDF** tries sources in this order:

1. The result's existing PDF link, but only when its host is in the configured scholarly-repository allowlist.
2. The official arXiv PDF URL when an arXiv ID is present.
3. Unpaywall's best open-access location when a DOI and `UNPAYWALL_EMAIL` are configured.
4. A browser upload prompt if no source produces a validated PDF.

The server does not scrape publisher pages, resolve paywalls, or expose a general-purpose URL fetcher. Every download and redirect must remain HTTPS and resolve only to public IP addresses. Responses are streamed through a configured byte limit, restricted to PDF-compatible content types, and finally checked for the `%PDF-` signature. Upload filenames and MIME types are secondary checks; server-side size and PDF-signature validation are authoritative. Uploaded PDF bytes are held only for validation and parsing and are not permanently retained.

Validated PDF bytes are sent to the local GROBID `processFulltextDocument` endpoint. The resulting TEI is parsed defensively into `ParsedPaper`: bibliographic metadata, abstract, hierarchical sections, structured references, and paragraph-sized evidence chunks. Chunk IDs are deterministic for the source PDF and text location; text spans are offsets within each section. Page numbers stay null because this integration does not yet derive them reliably from PDF coordinates. No claim or evidence extraction occurs in Milestone 2A.

Parsed JSON is cached by SHA-256 of the exact PDF bytes plus parser name/version. The source URL when applicable, acquisition method, byte size, PDF SHA-256, parser name, and parser version are preserved as provenance. Search caching, ranking, and Milestone 1.5 evaluation artifacts are unchanged.

The API endpoints are:

- `POST /api/papers/pdf/acquire` with `paper_id` and optional canonical `pdf_url`, `arxiv_id`, and `doi`
- `POST /api/papers/pdf/upload` as multipart form data with `paper_id` and `file`

Acquisition reports `existing_pdf`, `arxiv`, `unpaywall`, or `upload_required` only after a source has been downloaded and validated. Upload responses report `upload`. A missing or busy GROBID instance returns a clear parser-unavailable error without affecting `/api/search`.

## Evidence-grounded insights

After a PDF is parsed, the paper panel sends its `ParsedPaper` and optional provider to `POST /api/papers/insights/stream`. Omitting `provider` remains backward-compatible and uses `LLM_PROVIDER`, which defaults to Ollama. `GET /api/llm/providers` returns only safe provider IDs, models, configuration state, and local/cloud flags; it never returns keys. The JSON endpoint remains available. The ten insight arrays are `paper_overview`, `research_problem`, `methods`, `key_contributions`, `evaluation`, `main_findings`, `why_it_matters`, `target_audience`, `limitations`, and `future_work`.

The shared context builder gives large-context providers one bounded rich paper packet and small local providers at most two deterministic, section-aware packets. Providers return categorized evidence claims using compact excerpt IDs. The backend maps those IDs to canonical chunks and quotes, runs the frozen v16 attribution, finding, qualifier, limitation, future-work, and semantic safeguards, and creates a ledger containing only validated evidence. A final provider call receives that ledger—not rejected model output—and may cite only ledger claim IDs. ResearchWeave maps those IDs back to existing evidence references and validates final claims again. Cache identity includes document content, pipeline version, provider ID, and model ID.

Provider-specific SDK behavior lives behind `LLMProvider`. Adding another provider normally requires one adapter, server-side configuration/registration, and provider-focused tests; context construction, evidence models, validation, caching, API responses, and rendering remain shared.

References: [GROBID REST API](https://grobid.readthedocs.io/en/latest/Grobid-service/), [GROBID Docker setup](https://github.com/grobidOrg/grobid/blob/master/doc/getting_started.md), [Unpaywall REST API](https://unpaywall.org/api), and [Unpaywall data format](https://unpaywall.org/data-format).

## Ranking

1. The bi-encoder embeds the original user query once and each deduplicated candidate's `Title + Abstract` once. Normalized embeddings produce cosine similarities, and the profile-configured shortlist proceeds: 100 candidates for Fast or Quality unless explicitly overridden.
2. The cross-encoder scores each `(original query, paper text)` pair in that shortlist. Fast uses reciprocal rank fusion (`k=60`) of semantic and cross-encoder positions; it never adds or compares their incompatible raw score scales. Quality retains raw reranker ordering by default. `RANKING_FUSION_MODE=reranker` switches either profile back to raw reranker ordering. Both model scores remain unchanged in the response for inspection.

Citation counts are displayed only as metadata. Models load lazily once per backend process and remain in memory. Tests substitute deterministic in-memory models.

## Milestone 1.5 ranking evaluation

The optional framework in `backend/evaluation/` compares four ranking conditions without changing production behavior: semantic-only MiniLM, the current Fast pipeline, a laptop-oriented `quality_local` candidate, and the large full-BGE research reference. Scholarly candidates are collected once per benchmark query, normalized, strictly year-filtered, deduplicated, and saved; every ranking condition reads that exact fixed candidate set.

Human judgments use a 0-3 relevance scale: 3 highly relevant, 2 relevant, 1 tangential, and 0 irrelevant. Reported measures are Precision@5, Precision@10, nDCG@10, nDCG@20, MRR, ranking latency, model-size estimates, and pairwise top-20 overlap. Relevance >= 2 is used for binary metrics. True Recall@K is not reported because the benchmark does not establish every relevant paper in the literature.

The intended production QUALITY model should preferably stay within a combined 750 MB weight budget, run on CPU, and remain practical for ordinary researchers. `full_bge_reference` is exempt from that budget and exists only to provide an experimental ceiling; the evaluator will not automatically download it when it is absent. No profile is promoted automatically. See `backend/evaluation/README.md` for the reproducible workflow and labeling instructions.

## Deduplication

Identifiers are normalized without changing display titles. Matching uses exact normalized DOI, exact version-free arXiv ID, or exact Unicode/punctuation/whitespace-normalized title. It intentionally avoids fuzzy title matching. Merged records keep all source IDs, the longest abstract, the maximum citation count, the most complete author list, and non-missing metadata.

## API

```http
GET /health
POST /api/search
Content-Type: application/json

{
  "query": "AI agents for visualization systems",
  "start_year": 2020,
  "end_year": 2026,
  "limit": 20
}
```

The response includes candidate, unique, and ranked counts; papers; each provider's `ok`, `cached`, or `error` state; and warnings.

To smoke-check only the live provider adapters without downloading ranking models:

```bash
cd backend
python scripts/check_providers.py "AI agents for visualization systems" --limit 3
```

## Tests and linting

Backend (no internet or model download required):

```bash
cd backend
pytest
ruff check .
ruff format --check .
```

Frontend:

```bash
cd frontend
npm test
npm run lint
npm run build
```

## Current limitations

- The Quality profile downloads several GB and its first-search CPU latency is substantial.
- Public APIs can be incomplete, rate-limited, or temporarily unavailable.
- Records with an unknown publication year are retained only when no year filter is supplied; any year bound excludes them.
- Exact title matching favors precision and may leave duplicates whose titles differ materially across indexes.
- The SQLite cache stores normalized metadata, not embeddings; embeddings are reused within a request but not persisted across requests yet.
- arXiv exposes no citation count, and metadata completeness varies by provider.
- GROBID must be started separately (or through the Compose `pdf` profile) before parsing; search
  remains available without it.
- PDF extraction quality depends on source layout, scans may contain little usable text, and page
  numbers are intentionally omitted until they can be derived reliably.
- Direct metadata PDF links outside the configured scholarly-host allowlist fall back to arXiv,
  Unpaywall, or upload.
- A single application process is the intended deployment shape.

## Roadmap (not implemented)

- **Milestone 3:** paper embeddings, citation and semantic links, topic clustering, interactive research landscape
- **Milestone 4:** evidence-grounded synthesis, themes, historical evolution, open problems, future directions, and research gaps

## Next recommended step

Complete the existing Milestone 1.5 human relevance judgments and review Milestone 2B extracted claims against their displayed evidence before any later milestone.

## License

MIT
