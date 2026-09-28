# Search Latency Fix Report

## 1. Exact Root Cause

The multi-minute delay was provider-side queue and retry amplification, not model loading or ranking.

Search expands the user query to three or four deterministic variants. Provider families run concurrently, but each provider executes its variants sequentially. Before this fix, a provider failure did not stop the remaining variants.

The worst path was arXiv:

- the maintained `arxiv` package used a synchronous `requests.Session` with no explicit HTTP timeout;
- `num_retries=5` meant six total HTTP attempts per variant;
- the package enforced at least three seconds between attempts/requests;
- every failed variant was followed by the remaining variants;
- one provider-wide lock serialized arXiv operations, including work from overlapping search requests.

Semantic Scholar was a secondary contributor. Its one-request-per-second pacing and bounded HTTP retries are intentional, but overlapping searches could queue indefinitely behind the same provider lock. A failed variant also did not stop later variants.

Thus one slow provider cycle could be multiplied by variants and again by overlapping browser requests. OpenAlex/Semantic Scholar results could already be ready while `/api/search` still waited for arXiv.

## 2. Measured Before Timing

Exactly one fresh real diagnostic search was started for:

- Query: `Agentic Design Patterns: A System-Theoretic Framework`
- Years: 2025–2026
- Limit: 20
- Generated variants: 3

The timing harness completed the search but encountered a Windows SQLite file-lock error while deleting its temporary cache before printing its in-memory JSON. The search was not repeated, preserving the one-real-search limit. Timing was reconstructed from that retained cache, and ranking was replayed once from the cache with all network providers disabled.

Fresh diagnostic cache completion evidence:

| Component | Measured/reconstructed time |
|---|---:|
| Query preparation | Below timer resolution for practical purposes; Pydantic request construction only |
| Query expansion | 0.000218 s |
| OpenAlex successful-write span, 3 variants | 1.702 s |
| Semantic Scholar successful-write span, 2 successful variants | 2.216 s; third variant failed and was not cached |
| arXiv successful-write span, 3 variants | 12.130 s |
| First provider completion → final provider completion | 14.665 s |
| Cache deserialize/write CPU in offline replay | 0.067387 s |
| Normalization | Instrumented inside provider time; exact counter was lost with the in-memory harness output |
| Deduplication/canonical merge | 0.075953 s |
| Bi-encoder | 7.774773 s |
| Cross-encoder | 5.614987 s |
| RRF | 0.000311 s |
| Ranking total | 13.399133 s |
| Response filtering/assembly residual | Less than approximately 0.04 s |
| Cached replay total | 13.512255 s |
| Reconstructed warmed fresh-search total | Approximately 28–32 s for this non-pathological run |

The offline replay ranked the same retained candidate set: 350 candidates, 217 deduplicated papers, and 20 returned papers. It made zero provider requests.

This fresh run did not reproduce the intermittent multi-minute failure. The existing Docker cache for the affected query did preserve the pathological history: OpenAlex/Semantic Scholar wrote initial results around `03:44:48`, a later Semantic Scholar variant was not written until `03:56:24`, the first arXiv result appeared at `03:59:55`, and later arXiv variant writes appeared around `04:08:42–04:08:59`. Those timestamps may include overlapping/cancelled browser requests, but they directly demonstrate provider-lock queue amplification rather than ranking work.

## 3. Provider Timing and Request Breakdown

| Provider | Variant calls in fresh run | Outcome | Relevant pre-fix retry behavior |
|---|---:|---|---|
| OpenAlex | 3 | 3 successful cache writes | Up to 3 HTTP attempts per variant; 20 s HTTP timeout |
| Semantic Scholar | 3 | 2 successful writes; 1 failed variant | Up to 3 HTTP attempts per variant; 20 s timeout; ≥1 s retry/pacing wait |
| arXiv | 3 | 3 successful writes | Up to 6 HTTP attempts per variant; ≥3 s spacing; no explicit HTTP timeout |

Successful cache writes imply a final HTTP 200 for those calls. The failed Semantic Scholar call's exact final status and the raw retry counters were held only in memory and were lost during the post-search cleanup failure. They are not guessed here. Static bounds before the fix were 9 provider-level variant calls, up to 9 OpenAlex HTTP attempts, up to 9 Semantic Scholar HTTP attempts, and up to 18 arXiv HTTP attempts for this three-variant query.

No timeout, retry, or provider error is hidden by the production response: provider failures still generate provider status/warnings while other providers can supply results.

## 4. Code Files Changed

Production:

- `backend/app/providers/arxiv.py`
- `backend/app/services/search_service.py`

Focused tests:

- `backend/tests/test_arxiv_provider.py`
- `backend/tests/test_search_service.py`

Validation-only instrumentation:

- `validation/diagnose_search_latency.py`
- `validation/replay_search_latency_cache.py`

This report:

- `search_latency_fix_report.md`

Pre-existing PDF acquisition/canonical merge worktree changes were preserved and not altered for this latency fix.

## 5. Exact Fix

1. arXiv's package-owned requests session now receives an explicit 10-second request timeout.
2. arXiv now uses the existing configured provider retry count (default 2 retries/3 attempts) instead of 5 retries/6 attempts.
3. Waiting for the arXiv provider lock is bounded at 10 seconds. A queued request then degrades gracefully to other providers.
4. The shared pacing lock for Semantic Scholar/arXiv is bounded at 10 seconds, preventing overlapping searches from accumulating an unbounded request queue.
5. Once a provider returns an error for a query variant, later variants for that provider are skipped. Other providers continue normally.

## 6. Why the Fix Is Safe

- Provider set is unchanged: OpenAlex, Semantic Scholar, and arXiv remain active.
- Healthy providers still run every query variant and use the normal cache.
- Only the already-failed provider stops doing additional expansion calls.
- Partial-provider failure remains explicit and graceful.
- No unbounded tasks or new background work were introduced.
- DOI → arXiv ID → normalized-title deduplication is unchanged.
- Canonical merge metadata behavior is unchanged.
- Exact-title ranking is unchanged.
- Bi-encoder, cross-encoder, shortlist, RRF, and ranking model choices are unchanged.
- Search request and response schemas are unchanged.

## 7. Ranking Semantics

Ranking quality/algorithm changed: **NO**.

The only possible result-set difference occurs during an actual provider failure: remaining variants for that failed provider are no longer attempted. Available results from that provider's earlier successful variants and all successful providers are still normalized, deduplicated, and ranked identically.

## 8. Tests

New focused coverage verifies:

- arXiv request timeout is applied and explicit overrides are preserved;
- arXiv retry count is bounded;
- invalid timeout configuration is rejected;
- a busy arXiv lock fails quickly without starting a client request;
- an unavailable provider is called only once rather than once per variant;
- an available provider still completes and returns papers when another provider fails;
- paced cross-request queue waiting is bounded;
- provider status/warnings and result structure remain intact.

Results:

- Most-focused latency/retry tests: **33 passed**.
- Relevant search/ranking/provider/dedupe/API tests: **87 passed**, 2 dependency deprecation warnings.
- Full backend suite: **301 passed**, 2 dependency deprecation warnings.
- Ruff check: passed.
- Ruff format check: passed.
- `git diff --check`: passed.

## 9. Expected Search Latency After Fix

- Warm, healthy fresh search for the diagnostic-sized candidate set: approximately 30 seconds on this machine, based on the measured provider window plus 13.4-second ranking time.
- Cached search: approximately 13.5 seconds, dominated by the unchanged local bi-encoder/cross-encoder ranking.
- Degraded provider: one bounded failure cycle rather than one cycle per query variant.
- Concurrent request waiting for a paced/busy provider: at most 10 seconds before that provider is marked unavailable for the waiting search and other provider results continue.

The first active arXiv failure can still consume its bounded HTTP attempts, and third-party servers can vary. The fix removes the 3–5+ minute variant/queue multiplication; it does not promise constant network latency.

## 10. External Limits

- Provider response time, rate limiting, DNS behavior, and remote availability remain outside ResearchWeave's control.
- Semantic Scholar can still return 429 and arXiv can still return transient errors.
- Local ranking remains the largest non-network cost for a warm cached query (13.4 seconds in this diagnostic).
- No post-fix real search was run because the task allowed only one real diagnostic search.

## 11. Explicit Answers

- Did ranking models reload per request? **NO**. The application keeps one `RankingService`; the separate validation process loaded once before timing and excluded that load.
- Did PDF acquisition/canonical merge cause the slowdown? **NO**. It performs no provider network calls during search, and measured deduplication/merge took 0.075953 seconds.
- Was arXiv involved? **YES — primary bottleneck/pathological queue source**.
- Was Semantic Scholar involved? **YES — secondary rate-limit/retry and cross-request queue contributor**.
- Were provider calls unnecessarily sequential? **YES, after a provider had already failed**. Normal within-provider pacing remains intentionally sequential.
- Were retries/backoff multiplying latency? **YES**.
- Did the fix change ranking quality/algorithm? **NO**.
