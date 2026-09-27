# Milestone 2B v15 implementation

Version: `m2b-v15-generalized-insights`

## Generic fixes

- Reserves a bounded, deduplicated set of concrete limitation and future-work passages before general evidence packing.
- Recognizes conservative author-declared future directions while continuing to reject participant suggestions, completed work, current capabilities, invented actions, and guaranteed outcomes.
- Rejects prior-work-only support for contributions, findings, and limitations, while retaining claims that are attributed to the current paper.
- Requires substantive evidence beyond a bare contribution heading.
- Requires current-paper result or observation evidence for findings, including quantitative results, qualitative observations, case studies, surveys, and expert feedback.
- Truncates catalog excerpts at punctuation or whitespace boundaries so evidence quotes do not end mid-word.
- Treats `associated with` as an ordinary relationship rather than an uncertainty marker; genuine modal and observational qualifiers remain protected.
- Builds Why It Matters from already validated problem, contribution, and finding claims, supports up to three evidence references, and retains the smallest evidence union that supports the synthesis without adding a model call.

## Files changed

- `backend/app/services/insight_selector.py`
- `backend/app/services/insight_service.py`
- `backend/app/services/insight_diagnostics.py`
- `backend/tests/test_insights.py`
- `validation/m2b_v15_implementation.md`

## Verification

- Added 29 synthetic focused tests; 31 focused cases passed, including two existing related cases.
- Full insight suite: 83 passed with two dependency deprecation warnings.
- Ruff check: passed.
- Ruff format check: passed.
- `git diff --check`: passed.

## Scope and remaining limitations

Diagnostics remain optional and off by default. Public API schemas and the three-call extraction budget are unchanged. This implementation did not run Ollama, invoke any model, or process a real paper. Semantic quality on real papers therefore remains to be validated in a later, explicitly authorized extraction task.
