# M2B v16 Trust Hardening

## Version

`m2b-v16-trust-hardening`

The extraction version is part of the existing insight-cache key, so v15 insight results cannot
mask v16 behavior. The parser/cache version remains `0.9.1-crf+frontmatter-v1`.

## Problems Addressed

- Prior/background work being retained as the current paper's contribution, finding, or limitation.
- Contribution claims supported only by a need, gap, background statement, or bare list header.
- Comparison setup, method descriptions, capabilities, and use-case procedures becoming Findings.
- Loss of concrete limitation, future-work, and declared-contribution sentences during selection.
- Rejection of explicit negative limitations and current-author future plans.
- Why It Matters accepting one-source capabilities or rejecting grounded multi-source significance.
- Loss of material exceptions, scope, conditions, or uncertainty.
- Target Audience inferred from domain relevance rather than explicit user-role evidence.

## Generic Changes

- Added clause-aware current/prior ownership checks and claim-aligned contribution support checks.
- Reserved up to six deduplicated declared-contribution items and four diverse concrete passages
  for each boundary field without retaining bare headers as evidence.
- Required outcome-bearing quantitative or qualitative evidence for Findings and added named
  rejection reasons for setup-only comparisons and capability-only candidates.
- Recognized explicit current-work negative limitations and expanded owned future-plan forms while
  preserving modal strength and substantive conditions.
- Required Why It Matters to connect at least two distinct validated Problem, Contribution, or
  Finding components using the smallest supporting multi-evidence union.
- Added material qualifier/exception checks and explicit-role/intended-user audience checks.
- Kept diagnostics optional and reused their existing zero-call validation recording path.
- Did not add optional evidence rebinding: the current evidence-ID flow has no small rescue hook
  that can safely rebind while reapplying every ownership, field-type, semantic, and qualifier rule.

## Files Changed

- `backend/app/services/insight_selector.py`
- `backend/app/services/insight_service.py`
- `backend/tests/test_insights.py`
- `backend/tests/test_insights_v16.py`
- `validation/m2b_v16_trust_hardening_implementation.md`

## Test Coverage

Added 63 generic synthetic v16 tests covering attribution, substantive contribution support,
declared-item selection, outcome-bearing Findings, boundary recall, negative limitations,
author-owned Future Work, multi-component significance, qualifier preservation, conservative
audience validation, diagnostics defaults, API shape, call count, and parser metadata stability.
Two existing assertions were updated for the bounded 4+4 boundary reservation and the more precise
`prior_work_attribution` diagnostic reason.

## Test Results

- Focused v16 suite: 63 passed.
- Complete insight suite: 146 passed; 2 existing Starlette/httpx deprecation warnings.
- Full backend regression suite: 261 passed; the same 2 deprecation warnings.
- Ruff check: passed.
- Ruff format check: passed; 69 files already formatted.
- `git diff --check`: passed (Git reported only normal LF-to-CRLF working-copy notices).

## Deferred Issues

Optional evidence rebinding was deferred because implementing it safely would require a new
candidate-to-stage-evidence rescue path rather than a small reuse of an existing mechanism.

## Validation Status

The code and synthetic regression gate are ready for the separate final real-paper validation.

Real-paper validation was NOT run in this task.
