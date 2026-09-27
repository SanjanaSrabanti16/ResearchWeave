# PDF Front-Matter Robustness

## Root Cause

The failing PDF has a publisher/citation cover before the article and a malformed machine-readable text layer on the first article page. GROBID full-text parsing returned HTTP success but produced an empty title and abstract, omitted section 1, and began the body mid-sentence. Later sections remained readable and structurally useful.

## Implementation

- `backend/app/parsers/grobid.py`: added conservative front-matter health assessment, a bounded GROBID header-only fallback, guarded title/abstract merging, warning logs, and parser provenance.
- `backend/app/core/config.py`, `docker-compose.yml`, and `README.md`: bumped parser/cache provenance to `0.9.1-crf+frontmatter-v1` and aligned the documented GROBID image with 0.9.1.
- `backend/tests/test_pdf_front_matter.py`: added 14 generic synthetic parser tests.

Full-text GROBID remains authoritative. The fallback never replaces or appends body sections and never synthesizes missing content.

## Health Detection

The assessment combines missing/degraded title or abstract, malformed/truncated opening text, a body section sequence beginning after section 1, and the contrast between front-matter problems and a readable multi-section body. One signal alone does not trigger fallback. Results are classified internally as `healthy`, `front_matter_degraded`, or `severely_degraded`; non-healthy status and recovered fields are retained in the existing parser provenance string and logged.

## Fallback Order

1. Parse the complete PDF with GROBID full-text extraction.
2. If multiple critical front-matter signals agree, call GROBID's local header-only endpoint for pages 1–3.
3. Patch only a missing/degraded title or a readable explicit Abstract node.
4. Reject publisher boilerplate and noisy candidates; preserve the full-text body unchanged.
5. Keep and report degraded status when missing material cannot be safely recovered.

No new dependency or OCR capability was added.

## Shamma Parser-Only Smoke Test

| Metric | Before | After |
|---|---:|---:|
| Title | missing | `Bayesian reasoning machine on a magnetotunneling junction network` |
| Abstract | missing | missing; not fabricated |
| Quality status | not assessed | `front_matter_degraded` |
| Sections | 9 | 9 |
| Chunks | 22 | 22 |
| Parsed text characters | 17,331 | 17,331 |

Fallback provenance is `grobid_header_fallback(title)`. The later GROBID body and 33 references remained intact. The corrupted abstract and missing beginning of section 1 remain unrecoverable without OCR or a better source text layer.

## Clean-Paper Regression Check

StreetWeave remained on the normal GROBID path with no fallback. Its title, 756-character abstract, 31 sections, 58 chunks, 59,736 parsed text characters, and 59 references were unchanged.

## Tests

- New focused tests: 14 passed.
- Relevant parser/PDF tests: 37 passed, with two existing dependency deprecation warnings.
- Full backend regression suite: 198 passed, with the same two warnings.
- Ruff check, Ruff format check, Compose configuration validation, and `git diff --check`: passed.

## Remaining Limitations

The fallback intentionally does not OCR or reconstruct damaged article text. If GROBID's bounded header extraction cannot produce readable structured metadata, missing fields remain missing and the parser is marked degraded. The Shamma abstract and missing opening section are such cases.
