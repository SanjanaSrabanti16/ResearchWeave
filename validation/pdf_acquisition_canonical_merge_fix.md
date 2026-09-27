# PDF Acquisition and Canonical Merge Fix

## Root Cause

Provider normalization produced useful DOI, arXiv, landing-page, and PDF metadata, but `DeduplicationService._merge` reduced each merged work to one `url`, one `pdf_url`, and one `arxiv_id`. Only `source_names` was unioned. The frontend then sent only the surviving DOI/arXiv/PDF fields to `/api/papers/pdf/acquire`, whose acquisition path tried one direct URL, one arXiv route, and one Unpaywall lookup. Alternate acquisition identities were therefore unavailable after canonicalization.

## Previous Flow

`provider records → normalize → deduplicate/select base → discard alternate URLs/identities → rank → frontend sends canonical fields → direct PDF → arXiv ID → Unpaywall → upload required`

The secure downloader already enforced HTTPS, approved-host policy for client-supplied URLs, public-IP DNS resolution, redirect revalidation, content type, PDF magic, size, timeout, and redirect limits. The defect was metadata loss and single-route acquisition, not PDF validation.

## New Merge Behavior

- Existing DOI → arXiv ID → exact normalized-title dedupe identity remains unchanged.
- The best bibliographic base favors a DOI-bearing, non-arXiv venue record. A later sparse duplicate cannot replace its title, venue, year, or authors.
- DOI values normalize raw, `doi:`, DOI-URL, whitespace, and case variants.
- arXiv values normalize raw IDs, `arXiv:` prefixes, `/abs/`, `/pdf/`, URLs, `.pdf`, query strings, and version suffixes. Versions resolve to the same work identity.
- The canonical `Paper` remains backward compatible and now also carries `arxiv_ids`, `alternate_urls`, and `alternate_pdf_urls`.
- Canonical and alternate URLs are unioned and deduplicated without replacing the best canonical URL.
- `source_names` remains a union of every provider that supplied the work.
- OpenAlex normalization now preserves distinct landing and PDF URLs from primary, best-OA, and other locations.
- Ranking inputs and algorithms are unchanged; the extra fields are acquisition metadata only.

## Acquisition Candidate Order

The acquisition service creates at most 12 unique candidates and tries them sequentially:

1. canonical known `pdf_url`
2. preserved alternate PDF URLs
3. canonical and alternate normalized arXiv IDs, including IDs recovered from preserved arXiv landing URLs
4. DOI-based Unpaywall OA discovery/result
5. conservative title-verified arXiv fallback
6. manual upload remains available when all automatic routes fail

Equivalent URLs and arXiv version variants are attempted once. Acquisition stops after the first downloaded document passes the existing PDF validation and parser path.

Success records `known_pdf_url`, `alternate_pdf_url`, `arxiv_id`, `unpaywall`, `title_verified_arxiv_fallback`, or `upload` provenance. Each failed attempt records only its route, candidate type, safe hostname/identifier, and outcome category; full URLs/query strings and secrets are excluded.

## arXiv Fallback Rules

- A preserved valid arXiv ID is always used even when the canonical landing record is publisher-, OpenAlex-, or Semantic-Scholar-based.
- The final discovery fallback reuses the existing maintained `ArxivProvider`; no new provider or scraper was added.
- The normalized title must match exactly after Unicode/case/punctuation/hyphen/whitespace normalization.
- Without a matching DOI, at least one author identity must overlap and known years must be within one year.
- A matching normalized DOI strongly confirms an exact-title result.
- Near titles, incompatible authors, incompatible years, weak similarity alone, and candidates without a valid arXiv ID are rejected.
- The fallback performs one bounded search for at most five results and downloads at most one verified match.

## Security Guarantees

Security behavior is unchanged and applies to every attempted URL:

- Client-supplied canonical and alternate URLs retain the approved scholarly-host allowlist.
- Only the server-returned Unpaywall URL uses the existing trusted-discovery mode; it still requires HTTPS, standard port, public DNS/IP, and redirect revalidation.
- Official arXiv routes use the existing allowlisted secure downloader path.
- Every redirect destination is revalidated.
- Localhost, private/non-global IPs, credentials in URLs, nonstandard ports, HTTP URLs, unapproved hosts, excessive redirects, timeouts, oversized bodies, non-PDF content types, HTML/error bodies, and invalid PDF magic remain rejected.
- The 50 MB configured default is unchanged.
- No API key or secret is placed in acquisition diagnostics.

## Cache Behavior

There is no negative PDF-acquisition cache, so no acquisition cache version bump was needed. A previous `upload_required` response cannot suppress a later request with newly preserved routes. Successful documents continue to use the existing content-hash + parser-name/version parsed-document cache.

The parsed-PDF version remains `0.9.1-crf+frontmatter-v1`. The insight cache/version remains `m2b-v16-trust-hardening`; M2B code and caches were not changed.

## Files Changed

Production/backend:

- `backend/app/models/paper.py`
- `backend/app/models/document.py`
- `backend/app/services/normalization.py`
- `backend/app/services/deduplication_service.py`
- `backend/app/services/pdf_service.py`
- `backend/app/providers/openalex.py`
- `backend/app/main.py`

Small frontend compatibility update:

- `frontend/src/api/client.ts`
- `frontend/src/types/paper.ts`

Tests:

- `backend/tests/test_api.py`
- `backend/tests/test_deduplication.py`
- `backend/tests/test_normalization.py`
- `backend/tests/test_pdf_acquisition.py`
- `backend/tests/test_pdf_processing.py`
- `backend/tests/test_providers.py`
- `frontend/src/api/client.test.ts`

Validation-only artifacts:

- `validation/pdf_acquisition_live_smoke.py`
- `validation/pdf_acquisition_canonical_merge_fix.md`

## Synthetic Tests

- Added 34 backend test cases and one frontend compatibility test; two existing backend tests were updated for the backward-compatible response/schema fields.
- Focused merge/normalization/provider/acquisition/security/API suite: **70 passed** after the final added cases.
- Full backend regression: **295 passed**, with two existing dependency deprecation warnings.
- Frontend: **13 passed**; ESLint passed; TypeScript/Vite production build passed.
- Tests cover canonical bibliographic preservation; DOI/arXiv normalization; source, landing, and PDF URL union/deduplication; candidate order/fallthrough/bounds; conservative title fallback; success/failure provenance; secret redaction; SSRF/redirect/HTTP/content/size/magic defenses; negative-result behavior; parsed cache; manual upload; normal direct acquisition; and API compatibility.
- Ruff check and format check passed. `git diff --check` passed; only existing Git LF→CRLF notices were emitted.

## Live Smoke Test

- Query/title: `Attention Is All You Need`, year 2017.
- Provider results from the single bounded run:
  - OpenAlex: OK, 10 returned, 0 exact normalized-title records.
  - Semantic Scholar: OK, 10 returned, 1 exact normalized-title record.
  - arXiv: OK, 0 returned for the bounded year-filtered query.
- Deduplicated canonical count: 1.
- Canonical ID: `a1a0262a54cd0bbc3e7baed2`.
- Canonical title: `Attention is All you Need`.
- Preserved DOI: none supplied by the exact provider record.
- Preserved arXiv ID: `1706.03762`.
- Preserved provider set: `semantic_scholar`.
- Candidate order: preserved arXiv ID → Unpaywall if a DOI existed → title-verified arXiv fallback.
- Route used: `arxiv_id`.
- Attempt result: `arxiv.org`, success on the first candidate.
- PDF validation and GROBID parsing: successful.
- PDF SHA-256: `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`.
- Acquisition provenance: `arxiv_id`.
- Manual upload would remain available on automatic failure: yes.
- Ollama/model calls: **0**.

This live result proves that a canonical non-arXiv provider landing record can acquire through its preserved arXiv identity. The cross-provider DOI/publisher + arXiv union was exercised deterministically in mocked merge/acquisition tests because only one provider returned an exact record during the bounded live query.

## Remaining Limitations

- Conservative fallback deliberately rejects approximate titles or exact titles without DOI/author corroboration; this favors avoiding the wrong paper over maximizing recall.
- Client-forwarded alternate PDF URLs remain subject to the configured scholarly-host allowlist. Unknown repositories may fail safely and proceed to arXiv/Unpaywall/upload rather than bypass SSRF policy.
- Provider metadata quality still limits what can be preserved; acquisition cannot infer a DOI or arXiv identity that no provider supplies unless the final conservative arXiv match succeeds.
- Attempt diagnostics intentionally expose only safe host/route categories, not raw URLs or provider query details.

The required canonical metadata preservation, bounded safe alternates, preserved-arXiv acquisition, conservative fallback, provenance, cache behavior, and manual upload path are validated. Milestone 2 PDF acquisition is stable enough to freeze; no additional provider-edge tuning is recommended before Milestone 3.
