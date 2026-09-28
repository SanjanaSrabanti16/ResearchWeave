# Milestone 2 Final Provider Architecture Report

## Status

The final closeout in **Section 22** supersedes all earlier provisional decisions. The bounded
output architecture, contribution completeness, open-challenge presentation, full Overview
visibility, provider degradation, and regression checks now satisfy the Milestone 2 acceptance
contract.

**MILESTONE 2: FROZEN**

## 22. Final Milestone 2 closeout (authoritative)

This section supersedes the remaining acceptance gaps and freeze decision in Section 21. Pipeline
version: `m2-final-v7-closeout`.

### Exact closeout fixes

1. **Explicit contribution completeness.** Final synthesis now tracks which validated ledger claims
   it actually used. Grounded contribution items identified by the generic current-paper
   contribution-list detector are restored when synthesis omits one; an Overview mention does not
   suppress the item from `key_contributions`. The restored claim retains its already-validated
   source evidence and still passes the existing grounding/current-paper/v16 path. Related-work
   contribution lists remain excluded.
2. **Limitations versus open challenges.** Explicit self-limitations remain normal limitations. If
   none survive but a validated future-work ledger claim is backed by explicit open-challenge
   language, the final result labels it honestly as an unresolved challenge that the paper does not
   frame as a study limitation. Plain future directions are not recast as challenges. If neither
   exists, the UI states: “The paper does not explicitly state study limitations.”
3. **Complete Overview visibility.** The overview paragraph has an explicit unclamped style. It is
   displayed in full by default with no line clamp, fixed height, hidden overflow, ellipsis, or
   “Show more” interaction. Evidence remains collapsible.

The cache/pipeline version was incremented so older incomplete synthesis results cannot mask the
closeout behavior.

### Exact-paper deterministic acceptance

The cached parsed document for **Intellicise Wireless Networks Meet Agentic AI: A Security and
Privacy Perspective** (arXiv 2602.15290) was replayed through the deterministic final resolver; no
new model call was required.

- Grounded source-declared contributions: 5
- Contributions initially represented by synthesis: 4
- Contributions restored generically: 1
- Final `key_contributions`: **5/5**
- Validated future-work claims: 3
- Honestly labeled open challenges restored: 3
- Final future-work claims retained: 3

### Semantic Scholar diagnostic

One live Semantic Scholar provider request was made with retries disabled and without exposing the
configured key.

- API key configured: yes
- Exception: `httpx.HTTPStatusError`
- HTTP status: **429**
- Response: “Too Many Requests. Please wait and try again or apply for a key for higher rate
  limits.”
- Classification: external rate limiting, not authentication failure or a local request/parsing bug

No search code was changed. Existing behavior is correct: `SearchService` records a non-fatal
Semantic Scholar warning, keeps OpenAlex/arXiv results, and fails the overall search only when every
provider fails.

### Final verification

- Focused backend insight/v16 tests: 136 passed
- Focused frontend PDF/insight tests: 9 passed
- Full backend suite: 374 passed
- Full frontend suite: 17 passed
- Frontend ESLint: passed
- Frontend production build: passed (Vite 6.4.3)
- Ruff check: passed
- Ruff format check: passed after mechanical formatting
- `git diff --check`: passed; existing Git CRLF conversion warnings remain
- Known test warning: upstream Starlette/AnyIO `BlockingPortal` deprecation
- No live Gemini call
- No Ollama run or model tuning
- No additional EVL run
- No Milestone 3 work

### Final freeze decision

All bounded closeout acceptance conditions are met. Unsupported content remains rejected by the
existing validators, full Overview text is visible, explicit current-paper contribution lists are
complete when grounded, and open challenges are presented without fabricating self-limitations.

**MILESTONE 2: FROZEN**

## 21. Final output-budget architecture and acceptance (authoritative continuation)

This section supersedes Section 20. Pipeline version:
`m2-final-v6-output-batches`.

### Implemented output architecture

- EVL Gemma and Gemini continue to receive the complete `full_document` packet with the same 142
  evidence spans, stable IDs, reading order, and bibliography exclusion on every evidence pass.
- Full-document evidence extraction is split only on output semantics:
  - Pass A: problem, approach/method, contribution
  - Pass B: evaluation, finding, significance
  - Pass C: limitation, future work, audience
- Category-specific schemas and ceilings bound each response without reducing input context.
- Pass outputs are merged, deterministically deduplicated, structurally validated item by item, and
  passed through the unchanged grounding/current-paper/v16 validation before one ledger-only
  synthesis call.
- A failed pass is diagnosed and isolated; valid sibling passes remain usable. A
  `finish_reason=length` response is diagnosed as `output_budget_exceeded` and is not sent through
  format correction.
- Ollama remains `hierarchical_sections`; Gemini remains `full_document`. No live Gemini or Ollama
  call was made.

### Single authorized EVL acceptance run

Paper: **Intellicise Wireless Networks Meet Agentic AI: A Security and Privacy Perspective**
(arXiv 2602.15290). Provider/model: `evl_gemma` / `gemma4`.

- Fresh cache: yes
- Model calls: 4 (three evidence passes plus one synthesis)
- Pass A: 24.15 s; 17 generated; 16 validation-retained; 6 rejected after five deterministic
  source contribution claims were preserved
- Pass B: 11.16 s; 6 generated; 3 retained; 3 rejected
- Pass C: 14.64 s; 6 generated; 3 retained; 3 rejected
- Evidence extraction: 50.01 s
- Synthesis: 29.21 s
- Total: 79.24 s
- Merged validated ledger: 22 claims
- Failed passes: none
- Partial result: no
- Format corrections: 0
- Output truncation: no
- Synthesis coverage-loss flags: none
- Final claims: overview 1, problem 4, methods 7, contributions 4, evaluation 1, findings 1,
  significance 1, audience 0, limitations/open challenges 0, future work 3
- Provenance audit: 53/53 exact chunk/quote references; production validation unchanged; no
  internal evidence-claim markers

Saved acceptance artifact:
`validation/evl_m2_final_output_budget_acceptance.json`.

The final overview, problem analysis, seven-part methods explanation, case-study evaluation,
qualitative finding, grounded significance, and three future directions are coherent and useful.
No unsupported final claim survived validation.

### Remaining acceptance gaps

The strict synthesis-completeness contract is not fully met:

1. Four of the five author-declared contributions appear in `key_contributions`. The omitted first
   contribution concept remains present in the overview, but the category-specific contribution
   list is incomplete.
2. The paper's unresolved challenges are represented through future-work items, but the final
   `limitations` / open-challenges field is empty instead of explicitly distinguishing “no stated
   self-limitations” from grounded unresolved challenges.

The output-budget failure is fixed and the result is researcher-useful, but these are explicit
acceptance requirements, so they are not hidden or treated as passed.

### Regression results

- Focused context/provider/semantic/v16 suite: 133 passed
- Full backend suite: 371 passed
- Ruff check: passed
- Ruff format check: passed
- `git diff --check`: passed (existing Git CRLF conversion warnings only)
- Known warning: upstream Starlette/AnyIO `BlockingPortal` deprecation warning
- No frontend code changed; frontend tests were not run.

### Final freeze decision

**MILESTONE 2: NOT FROZEN**

## 1. Architecture before

`InsightService` owned the Ollama HTTP request format, three-stage prompts, selection,
validation, synthesis, and caching. The API and frontend assumed one local model. Cache identity
included the document fingerprint, model, and extraction version, but not a provider ID. The v16
trust rules already provided exact chunk/quote checks plus conservative attribution, comparison,
qualifier, finding, limitation, future-work, audience, and significance safeguards.

## 2. Architecture after

```text
ParsedPaper
  -> provider-independent context planning
  -> LLMProvider evidence extraction (Ollama, Gemini, or EVL Gemma)
  -> frozen v16 deterministic validation
  -> ValidatedEvidenceLedger
  -> provider-independent synthesis prompt using ledger claim IDs only
  -> deterministic claim-ID-to-EvidenceRef mapping and final validation
  -> Structured Research Insights
  -> API / frontend
```

Production uses `PaperUnderstandingService`. The original v16 `InsightService` remains intact as a
compatibility/test layer, and its validators are reused by the new pipeline.

## 3. Files changed for this task

- `.env.example`
- `README.md`
- `docker-compose.yml`
- `backend/pyproject.toml`
- `backend/app/api/routes.py`
- `backend/app/core/config.py`
- `backend/app/main.py`
- `backend/app/models/insights.py`
- `backend/app/llm/__init__.py`
- `backend/app/llm/base.py`
- `backend/app/llm/evl_gemma.py`
- `backend/app/llm/gemini.py`
- `backend/app/llm/ollama.py`
- `backend/app/llm/registry.py`
- `backend/app/services/insight_cache.py`
- `backend/app/services/insight_context.py`
- `backend/app/services/insight_service.py` (coverage limits for the two new fields only)
- `backend/app/services/paper_understanding_service.py`
- `backend/scripts/validate_m2_gemini.py`
- `backend/scripts/validate_m2_evl_gemma.py`
- `backend/tests/test_llm_architecture.py`
- `frontend/src/api/client.ts`
- `frontend/src/api/client.test.ts`
- `frontend/src/components/PaperInsights.tsx`
- `frontend/src/components/PaperPdfPanel.tsx`
- `frontend/src/components/PaperPdfPanel.test.tsx`
- `frontend/src/types/paper.ts`
- `m2_final_provider_architecture_report.md`

The worktree contained earlier uncommitted search, canonical-merge, and PDF changes before this
task. They were preserved and were not modified as part of this architecture work.

## 4. Provider interface

`LLMProvider` exposes `provider_id`, `model_id`, `configured`, `capabilities`, and
`generate_structured`. Capabilities describe structured JSON, bounded versus large context,
local/cloud operation, context budget, packet count, and thinking controls. `LLMProviderRegistry`
owns selection and safe provider status. Provider-specific SDK/HTTP behavior exists only in the
three provider adapters.

Gemini uses the official `google-genai` SDK (`2.25.0` installed; dependency constrained to
`>=2.25,<3`), async generation, structured JSON, a bounded timeout, and one SDK attempt. Search,
URL context, code execution, tools, and persistent file upload are not enabled. The default model
is configurable through `GEMINI_MODEL` and defaults to `gemini-3.5-flash`.

EVL Gemma uses the official `openai` Python client (`3.19.2` tested; dependency constrained to
`>=3.19.2,<4`) with the configured base URL passed unchanged, async Chat Completions, a bounded
timeout, and SDK retries disabled. The server accepts OpenAI-compatible JSON-object response mode.
The adapter requests that mode, states strict JSON/schema constraints explicitly, accepts plain JSON
or exactly one `json`/generic code-fence envelope, and Pydantic-validates the result. Preambles,
postambles, extra fences, malformed JSON, and schema-invalid JSON are rejected; no JSON repair or
schema weakening is used.

## 5. Context and evidence pipeline

The common context builder derives packets only from `ParsedPaper`. Large-context providers receive
one rich, bounded packet containing the abstract and meaningful non-bibliography body chunks.
Bounded local providers receive at most two section-aware packets: problem/approach/contributions,
then evaluation/results/discussion/boundaries.

Evidence extraction returns categorized atomic claims using compact excerpt IDs. Supported
categories are problem, approach/method, contribution, evaluation, finding, significance,
audience, limitation, and future work. IDs are resolved server-side to canonical chunks and exact
quotes; provider output cannot supply arbitrary quotes or chunk IDs.

## 6. Validation reuse

The new orchestration calls the v16 quote/chunk, current-work attribution, comparison direction,
uncertainty, qualifier, finding, audience, limitation, future-work, and semantic-support rules.
Unknown evidence IDs, quote mismatches, related-work contributions, method-as-finding claims,
fabricated future work, unsupported synthesis, and wrong-category ledger references are rejected.
Only retained claims enter `ValidatedEvidenceLedger`.

## 7. Synthesis design

Synthesis receives only ledger claim IDs, validated claims, and their validated evidence. It cannot
create a new evidence reference. Final claim IDs are deterministically mapped back to the ledger's
existing `EvidenceReference` values and validated again. Missing categories are allowed. The final
schema preserves all eight prior arrays and adds `paper_overview` and `evaluation`.

## 8. Cache changes

Pipeline version: `m2-final-v1-evidence-ledger`.

New cache identity includes document fingerprint (which covers PDF/document/parser content),
provider ID, model ID, and pipeline version. Ollama, Gemini, and EVL Gemma entries, different models,
and later pipeline versions cannot collide. Existing cache files are left in place.

## 9. API and frontend

- Existing insight requests without `provider` remain valid and use `LLM_PROVIDER` (default
  `ollama`).
- Requests may select the registered `ollama`, `gemini`, or `evl_gemma` provider; arbitrary URLs
  are impossible.
- `GET /api/llm/providers` returns provider ID, configured state, model, and local/cloud status only.
- Provider errors map to clear authentication, rate-limit, unavailable, timeout, or invalid-output
  responses. Streaming remains newline-delimited JSON with compatible stage/result/error events.
- The PDF panel has a compact Local Ollama / Google Gemini / EVL Gemma selector. Unconfigured remote
  providers are disabled, and the UI identifies when selected paper text leaves the local machine.
- A distinct-label regression test asserts the exact three option labels and the running frontend
  container was rebuilt; its served bundle contains all three names.
- Existing evidence rendering remains, with new Paper Overview and Evaluation sections.

## 10. Security and key handling

`GEMINI_API_KEY` and `EVL_GEMMA_API_KEY` are loaded only by backend settings as `SecretStr`, passed
to their adapters, and wired through Docker server-side environment variables. They are never
returned by provider status, API
responses, diagnostics, prompts, frontend code, or tests. Provider exceptions use fixed safe
messages and do not interpolate SDK response text or credentials. Tests verify secret redaction.

## 11. Verification

- Backend: **323 passed**; one upstream Starlette/AnyIO deprecation warning.
- Focused provider/v16/API tests: **171 passed**.
- Ruff check: passed after final formatting.
- Ruff format check: passed after final formatting.
- Frontend: **17 passed** across four test files.
- ESLint: passed.
- TypeScript/Vite build: passed with Vite 6.4.3.
- `docker compose config --quiet`: passed.
- `git diff --check`: passed; Git reports only existing LF-to-CRLF checkout notices.

## 12. Real Gemini validation

Target cached paper: **Agentic Design Patterns: A System-Theoretic Framework**, arXiv 2601.19752.
Provider/model: `gemini` / `gemini-3.5-flash`.

The first API attempt exposed a genuine adapter schema bug: Gemini rejected Pydantic's
`additionalProperties` keyword with HTTP 400 before generation. The adapter now recursively removes
that unsupported keyword and sends `response_json_schema`; a focused regression test covers this.
The same post-fix validation was then accepted structurally but returned HTTP 503 `UNAVAILABLE`:
the model was experiencing high demand. That failure was safely translated to
`LLMProviderUnavailableError`. No additional model requests were made.

## 13. Runtime

- Pre-generation schema rejection: approximately **2.95 seconds**.
- Post-fix high-demand response: approximately **5.94 seconds**.
- Evidence extraction runtime: unavailable (generation did not start).
- Synthesis runtime: unavailable.
- Successful total runtime: unavailable.

## 13A. Real EVL Gemma validation

Target cached paper: **Agentic Design Patterns: A System-Theoretic Framework**, arXiv 2601.19752.
Provider/model: `evl_gemma` / `gemma4`. The configured base URL was passed to the official client
unchanged, with no inferred `/v1` suffix.

The one-request diagnostic captured an exact `json` fence and found category C: syntactically invalid
JSON. At generated line 72, column 7, an `evidence_ids` array contained a trailing comma immediately
before its closing bracket. The JSON parser reported `Expecting value`. The raw generated assistant
content and credential-free parse summary are saved in
`validation/evl_gemma_output_diagnostic.json`.

The generic compatibility fix requests `response_format={"type":"json_object"}`, forbids trailing
commas/comments/preambles/postambles explicitly, accepts plain JSON or one exact `json`/generic
fence, and retains strict Pydantic validation. A single tiny capability probe was accepted by the
endpoint, although its 20-token response had null content and therefore did not independently prove
semantic JSON enforcement.

The one permitted post-fix full-paper run reached EVL and returned syntactically valid JSON after
approximately 30 seconds. Strict Pydantic validation then found a second concrete schema mismatch:
`claims[3].evidence_ids` and `claims[7].evidence_ids` each contained five IDs, while the frozen common
schema allows at most four. The schema was not widened and the citations were not truncated or
repaired. The generic prompt now emphasizes every `required`, `enum`, `minItems`, and `maxItems`
constraint, but no third paper run was made.

- Evidence request wall time before rejection: approximately **30 seconds**
- Parsed raw claim count: unavailable because adapter schema validation did not complete
- Validated ledger claims: **0**; ledger creation was not reached
- Deterministically rejected claims: unavailable; v16 validation was not reached
- Synthesis runtime: **not started**
- Total runtime before failure: approximately **30 seconds**
- Final category counts: no final insights were produced
- EVL cache entry: not written

No researcher-quality claims were available to audit. The overview, systems-theoretic approach,
five subsystems, 12 design-pattern categories, ReAct case study, contributions, findings,
significance, audience, limitations, future work, attribution, and evidence-ledger mapping therefore
remain unassessed for EVL Gemma. No quality claim is inferred from mocked tests.

## 13B. Final per-claim isolation fix and live result

The shared evidence transport now validates malformed JSON, the required top-level `claims` array,
and the top-level claim-count bound as whole-response constraints. It then validates every raw claim
independently with the unchanged `_GeneratedEvidenceClaim` model. A five-ID claim is rejected as
`too_many_evidence_ids`; an invalid enum is rejected as `invalid_category`; other claim-local schema
failures use `invalid_evidence_claim_schema`. Valid siblings continue to frozen v16 validation.
Diagnostics separately count raw, structurally valid, structurally rejected, evidence-validation
rejected, and ledger-retained claims. If no structurally valid claim survives across the packets, the
extraction fails clearly. The provider generation schema remains fully strict for Ollama, Gemini,
EVL Gemma, and future adapters.

The exactly one authorized EVL run on the cached arXiv 2601.19752 paper reached
`_resolve_generated_evidence`, proving the provider response passed the new top-level and per-claim
parsing stage. It then encountered a canonical context excerpt longer than the frozen
`EvidenceReference.quote` maximum of 500 characters. Creating that evidence reference raised a
Pydantic `string_too_long` error before the ledger was completed. A generic guard now rejects only
such a claim as `evidence_quote_too_long` and continues with other claims; it does not truncate the
quote or relax the schema. No second EVL run was made.

- Live runtime before the failure: approximately **30 seconds**
- Raw claim count: unavailable because the diagnostic script terminated before serialization
- Structurally rejected count/reasons: unavailable from the terminated process
- v16 retained/rejected counts: unavailable; v16 processing did not complete
- Evidence runtime: unavailable as a completed stage
- Synthesis runtime: **not started**
- Total runtime before failure: approximately **30 seconds**
- Final insights: none
- Cache entry: none

Because there is no final validated ledger or synthesis, the five functional subsystems, 12 design
patterns/categories, ReAct case study, attribution correctness, fabricated findings/future work, and
researcher usefulness cannot be audited from this run.

## 14. Quality assessment

No live claims were generated, so raw/validated/rejected counts, final category counts, and the
requested manual assessment of the overview, five subsystems, 12 patterns, ReAct case study,
contributions, findings, significance, audience, limitations, and future work cannot be reported.
Mocked end-to-end tests prove the structural guarantee that rejected evidence is absent from the
synthesis prompt and that synthesis cannot manufacture evidence, but they do not substitute for the
required real-paper quality audit.

## 15. Known limitations

- Gemini service capacity can block a live validation even when configuration and schemas are valid.
- Deterministic lexical/semantic guards are conservative and may reject supported paraphrases.
- The local 1.7B model receives smaller packets and may be shallower than a large-context provider.
- Provider status reports configured state, not a live network health probe.
- Per-claim schema isolation and oversized-reference rejection have complete offline coverage but
  still need one successful real EVL run before semantic quality can be accepted.
- The UI does not accept browser-supplied API keys or arbitrary provider endpoints.

## 16. Adding another OpenAI-compatible provider

1. Implement one `LLMProvider` adapter with its safe transport and structured-output mapping.
2. Declare its capabilities and server-side configuration.
3. Register it in `LLMProviderRegistry` during application startup.
4. Add adapter error/structured-output tests and one registry/status entry.
5. Add a selector label only when the provider is intentionally exposed in the UI.

The context builder, evidence models, ledger validation, synthesis, cache, API response, and insight
renderer do not need to be rewritten.

## 17. Freeze decision

- Is Ollama still supported? **YES**
- Is Gemini supported? **YES** (implemented and mocked; live quality validation pending after 503)
- Is EVL Gemma supported? **PARTIALLY** (adapter, configuration, API/UI selection, cache separation,
  per-claim isolation, and error handling are implemented and tested; strict real full-paper
  validation still has not completed)
- Is the core pipeline provider-independent? **YES**
- Does synthesis use only validated evidence? **YES**
- Were v16 trust safeguards weakened? **NO**
- Did search/ranking change? **NO**
- Did PDF acquisition/parsing change? **NO**
- Can another OpenAI-compatible provider be added without rewriting the pipeline? **YES**
- Are provider secrets kept server-side? **YES**
- Is Milestone 2 ready to freeze? **NO** — the required successful real-provider quality validation
  is still incomplete.

## 18. Final hardening validation (authoritative update)

### Shared evidence and structured-output architecture

The context builder now divides every source chunk into deterministic citeable spans before any
provider sees it. Each span is at most the frozen 500-character `EvidenceReference.quote` limit,
is an exact source substring, preserves chunk/section/heading provenance, has stable ordering and a
stable compact ID, and prefers paragraph, sentence, and whitespace boundaries. One long chunk may
produce several spans. EVL Gemma, Gemini, and Ollama all use this common abstraction.

Evidence generation keeps strict top-level JSON and the required `claims` array, then validates
claims independently with the unchanged common schema. Claim-local errors drop only that claim and
record a reason; malformed top-level JSON, an invalid/missing claims array, and zero surviving
claims still fail. Synthesis now uses the same provider-independent isolation: the provider receives
the strict synthesis schema, but one invalid synthesis item no longer aborts valid siblings. No
claim, citation, category, or evidence is repaired, truncated, or invented. Only v16-retained
claims enter `ValidatedEvidenceLedger`, and synthesis receives only ledger claim IDs.

### EVL Gemma live result

The second and final permitted full-paper run succeeded end-to-end on **Agentic Design Patterns: A
System-Theoretic Framework** (arXiv 2601.19752) with `evl_gemma` / `gemma4`:

- Context: 1 packet, 85 bounded evidence spans
- Model calls: 2
- Evidence runtime: 29.76 seconds
- Synthesis runtime: 24.17 seconds
- Total runtime: 53.94 seconds
- Raw evidence claims: 20
- Structurally rejected evidence claims: 0
- v16 rejected evidence claims: 10
- Retained ledger claims: 10
- Raw synthesis items: 13
- Structurally rejected synthesis items: 1 (`synthesis_claim_too_long`)

Evidence/v16 rejection distribution: `finding_without_outcome` 4,
`uncertainty_mismatch` 2, `contribution_support_mismatch` 1,
`audience_not_explicit` 1, `prior_work_attribution` 1, and
`future_work_strength_or_condition_mismatch` 1. Post-ledger synthesis rejected five items through
`v16_semantic_guard`, two through `category_mismatch`, and one through
`why_requires_multiple_evidence_categories`.

Final counts: Paper Overview 0; Research Problem 3; Methods 2; Key Contributions 1; Evaluation 1;
Main Findings 0; Why It Matters 0; Target Audience 0; Limitations 1; Future Work 1.

The retained insights capture:

- unreliable/brittle agentic systems arising from foundation-model weaknesses and ad-hoc design;
- the missing rigorous systems-theoretic basis and implementation difficulty of existing pattern
  taxonomies;
- the five interacting subsystems: Reasoning & World Model, Perception & Grounding, Action
  Execution, Learning & Adaptation, and Inter-Agent Communication;
- the 12 agentic patterns and their four organizational categories;
- the system-theoretic framework as the retained contribution;
- the ReAct qualitative case-study method using deconstruct, diagnose, and prescribe;
- the explicitly conceptual nature of the framework as a limitation; and
- quantitative benchmarking against baselines as future work.

The result did not retain the generated ReAct findings about observation validation, context
management, planning, tool-error recovery, and absent learning/communication subsystems. Frozen
v16 rejected them as `finding_without_outcome`. No related-work claim, capability-as-finding,
unsupported comparison, fabricated limitation/future work, or invented evidence ID survived.
The live artifact contains 17 displayed references; all 17 resolve to exact source substrings, none
exceeds 500 characters, and the maximum is 486.

### Gemini compatibility, retry, and live status

The configured model is `gemini-3.8-flash`, while `GEMINI_MODEL` remains configurable. The current
common Pydantic schemas contain `minLength`, `maxLength`, and `default`, which are outside Gemini's
documented structured-output subset. `GeminiProvider` now removes those unsupported keywords only
at its adapter boundary (along with its existing `additionalProperties` compatibility handling).
The common schemas and server-side validation remain unchanged.

Gemini retries only HTTP 429 and 503, for at most three total attempts, with approximately 2- and
4-second exponential delays plus jitter. SDK retries remain disabled. Authentication errors, HTTP
400/schema errors, invalid output, timeouts, and deterministic validation failures are not retried.
Mocked tests cover both retry statuses, backoff, the attempt cap, no retry for 400/authentication,
current-schema sanitization, and secret redaction.

One tiny capability probe and the single bounded full-pipeline validation both exhausted the three
attempts with HTTP 503 `UNAVAILABLE`; Google reported high model demand. Neither returned HTTP 400,
so the prior schema-rejection path was not reproduced after sanitization. No further Gemini request
was made. Google's supported-schema reference is
<https://ai.google.dev/gemini-api/docs/generate-content/structured-output>.

### Ollama regression

The installed configured `qwen3:1.7b` completed a tiny live call through `OllamaProvider` and
returned `{"status":"ok"}`. No model was downloaded or tuned. The one-line diagnostic harness then
raised an event-loop-close error while closing its HTTP client in a second event loop; generation
had already succeeded and the process cleaned up on exit.

### Final verification

- Focused common/provider/v16 tests: **98 passed**
- Full backend suite: **336 passed**
- Backend warning: one upstream Starlette/AnyIO `BlockingPortal` deprecation warning
- Ruff check: passed
- Ruff format check: passed (82 files already formatted)
- Frontend: **17 passed** across 4 files
- ESLint: passed
- TypeScript/Vite build: passed with Vite 6.4.3
- Docker Compose configuration: passed
- Exact-source artifact audit: 17 references, 0 invalid, maximum length 486
- `git diff --check`: run after this report update

Files changed by this final hardening task:

- `backend/app/models/insights.py`
- `backend/app/services/insight_context.py`
- `backend/app/services/paper_understanding_service.py`
- `backend/app/llm/gemini.py`
- `backend/scripts/validate_m2_gemini.py`
- `backend/tests/test_llm_architecture.py`
- `validation/evl_gemma_m2_validation.json`
- `m2_final_provider_architecture_report.md`

The existing dirty worktree includes earlier search, provider, PDF, API, frontend, and validation
changes. They were preserved and not modified by this final hardening task.

### Remaining limitations and freeze decision

- Frozen v16 conservatively removed substantively plausible ReAct diagnostic findings.
- The successful EVL output has no final overview, main findings, why-it-matters, or audience.
- Some accepted generated claim text contains compact ledger markers such as `[EC001]`; grounding
  is correct, but these markers are presentation noise.
- Gemini semantic quality remains untested because Google returned bounded HTTP 503 responses.
- The local 1.7B Ollama model remains a lower-capacity fallback.

Explicit final answers:

- Is EVL Gemma working end-to-end? **YES**
- Are EVL insights researcher-useful? **YES, but incomplete**
- Is Gemini adapter structurally compatible with the common pipeline? **YES**
- Does Gemini use bounded retry for 429/503? **YES**
- Is Ollama still supported? **YES**
- Is the pipeline provider-independent? **YES**
- Does synthesis use only validated evidence? **YES**
- Were v16 safeguards weakened? **NO**
- Did search/ranking change? **NO**
- Did PDF acquisition/parsing change? **NO**
- Can future providers be added without rewriting paper understanding? **YES**
- Is Milestone 2 ready to freeze? **NO**

The blocking acceptance gap is final semantic coverage, not transport, grounding, provider
architecture, or provenance. Because the substantive ReAct findings and integrated research story
were not retained, the stated depth criterion is not fully met.

**Historical decision: NOT FROZEN (superseded by Section 19).**

## 19. Final reliability and semantic-depth acceptance

### Exact production failure diagnosis

The most recently parsed distinct UI paper was **Toward Agentic AI: Generative Information
Retrieval Inspired Intelligent Communications and Networking**. A diagnostic run through the same
`PaperUnderstandingService` path used by the UI reproduced the failure before the reliability fix.

The response was plain, syntactically valid JSON and generation finished normally (`stop`). The
failure occurred during **synthesis** because `research_problem` contained five items while the
contract permits four. This was category **H: cardinality violation**, not malformed JSON. The
relaxed transport envelope still enforced the whole field's list maximum before per-item isolation,
so one overflow item caused the complete paper to fail.

Credential-free diagnostics now record provider, model, stage, paper ID/title, raw assistant
content, envelope type, finish reason, JSON parse error, structural schema errors, timestamp, run ID,
and whether a correction was attempted. They never include API keys, authorization headers, or
environment secrets.

### Generic reliability fix

The common boundary now performs these steps:

1. Normalize plain JSON or one harmless JSON/generic Markdown fence.
2. Strictly parse JSON.
3. If parsing fails, permit exactly one serialization-only correction generation.
4. The correction prompt requires preservation of semantic content, claim text, categories, and
   evidence IDs and forbids additions, removals, summaries, reclassification, and invented content.
5. Parse and validate the corrected response once; a second failure stops with
   `structured-output correction failed`.
6. After a valid top-level object exists, validate evidence and synthesis items independently.
7. Reject field-overflow synthesis items as `too_many_synthesis_items`; valid siblings continue.

Structural schema failures do not trigger format correction. Evidence IDs are never truncated and
scientific content is never repaired. A one-request EVL function/tool-argument probe returned HTTP
400, so the endpoint does not support that stronger native mechanism. EVL therefore keeps its
supported `json_object` mode plus the bounded shared correction path.

Provider/UI errors now distinguish malformed output, failed format correction, structural output,
no evidence surviving validation (with a rejection distribution), no usable synthesis, provider
unavailability, and timeout without exposing raw provider internals or secrets.

### Semantic-depth fixes

- Qualitative findings may be supported by current-paper Results, Findings, Evaluation, Case Study,
  Analysis, Discussion, or explicit diagnostic/observation language. Architectural weaknesses,
  failure modes, absent subsystems, workflow problems, thematic findings, and expert observations
  no longer require a numeric metric.
- Method descriptions, proposed capabilities, related work, speculation, future work, and
  unsupported interpretations remain rejected.
- Paper Overview permits one grounded synthesis item up to 1,200 characters. Evidence quotes remain
  capped at 500 characters.
- Why It Matters accepts either one explicit validated significance statement or a synthesis across
  at least two compatible ledger claims. Unsupported hype and unsupported effects remain rejected.
- Internal `[EC###]` ledger markers are stripped deterministically from final claim text after ID
  resolution; evidence mappings are unchanged.
- Cache/pipeline version is `m2-final-v2-reliability-depth`, preventing stale results from masking
  these changes.

### EVL paper A: Agentic Design Patterns

Paper: **Agentic Design Patterns: A System-Theoretic Framework**, arXiv 2601.19752.

- Runtime: 56.85 seconds
- Provider/model: `evl_gemma` / `gemma4`
- Evidence spans: 85
- Provider calls: 2
- Format correction used: no
- Raw evidence claims: 20
- Structural evidence rejections: 0
- Semantic/v16 rejections: 7
- Retained ledger claims: 13
- Raw synthesis items: 15
- Structural synthesis rejections: 0
- Final counts: overview 1, problem 3, methods 2, contributions 1, evaluation 1, findings 3,
  significance 1, audience 0, limitations 1, future work 1

Rejections: `uncertainty_mismatch` 2, `contribution_support_mismatch` 1,
`finding_without_outcome` 1, `audience_not_explicit` 1, `prior_work_attribution` 1,
`future_work_strength_or_condition_mismatch` 1, and one synthesis `category_mismatch`.

Actual final insights:

- **Overview:** The paper addresses brittle agentic systems and weak existing taxonomies with a
  system-theoretic decomposition into five subsystems and 12 design patterns, demonstrated through
  a ReAct case study using deconstruct, diagnose, and prescribe.
- **Problem:** Foundation-model weaknesses and ad-hoc design cause brittleness; existing pattern
  taxonomies lack a rigorous systems basis, are difficult to implement, and connect poorly to
  established software patterns.
- **Methods:** Five interacting subsystems—Reasoning & World Model, Perception & Grounding, Action
  Execution, Learning & Adaptation, and Inter-Agent Communication—and 12 patterns organized into
  Foundational, Cognitive & Decisional, Execution & Interaction, and Adaptive & Learning groups.
- **Contribution:** A novel system-theoretic framework decomposing agents into five interacting
  functional subsystems.
- **Evaluation:** A qualitative ReAct case study using deconstruct, diagnose, and prescribe.
- **Findings:** ReAct exhibits world-model fragility from missing observation validation and weak
  context management; its unstructured Thought process causes suboptimal planning; and it lacks
  robust tool-error recovery and adaptation, preventing learning from failures.
- **Why It Matters:** The framework supplies a shared language and structured methodology for more
  modular, understandable, and reliable agentic-system design.
- **Limitation:** The framework is primarily conceptual.
- **Future Work:** Quantitative benchmarking of reliability and efficiency improvements against
  baselines.

Quality audit: a researcher can understand the paper's problem, framework, contribution,
demonstration, qualitative findings, significance, limitation, and next step without reading the
entire PDF. One absent-subsystem claim was conservatively rejected, and the 12-pattern contribution
appears under Methods and Overview rather than as a second Contribution bullet. These are minor
coverage imperfections, not trust failures.

### EVL paper B: Toward Agentic AI

Paper: **Toward Agentic AI: Generative Information Retrieval Inspired Intelligent Communications
and Networking**.

- Runtime: 68.08 seconds
- Provider/model: `evl_gemma` / `gemma4`
- Evidence spans: 152
- Provider calls: 2
- Format correction used: no
- Raw evidence claims: 27
- Structural evidence rejections: 0
- Semantic/v16 rejections: 5
- Retained ledger claims: 22
- Raw synthesis items: 24
- Structural synthesis rejections: 1 (`too_many_synthesis_items`)
- Final counts: overview 1, problem 4, methods 5, contributions 1, evaluation 4, findings 3,
  significance 1, audience 0, limitations 1, future work 2

Evidence rejections: `contribution_support_mismatch` 1, `entity_mismatch` 1,
`uncertainty_mismatch` 1, and `future_work_guard` 2. Synthesis rejected the fifth problem as
`too_many_synthesis_items` and one contribution as `category_mismatch`.

Actual final insights:

- **Overview:** The paper proposes an LLM-based agentic contextual-retrieval framework integrating
  multi-source retrieval, structured reasoning, and self-reflective validation, evaluated with
  Qwen2.5-Max and TeleQnA, with improvements over traditional and semantic retrieval.
- **Problem:** Telecom scale and complexity require automation; decisions require standards-aware
  multi-hop retrieval; current agents suffer inconsistency, drift, and hallucination; conventional
  RAG faces fragmented standards, retrieval inefficiency, and ambiguous intent.
- **Methods:** Context-aware document vectorization and query reformulation; multi-source vector,
  structured, and online retrieval; LLM evidence aggregation; chain-of-thought decision-making; and
  self-reflective iterative validation.
- **Contribution:** A categorized review of communications/networking retrieval methods from 2023
  through late 2024.
- **Evaluation:** Qwen2.5-Max, 50 TeleQnA/3GPP R18 QA pairs, FAISS and Mpnet-base-V2, three retrieval
  baselines, and four answer/explanation metrics.
- **Findings:** The framework outperformed all baselines; achieved 84% answer accuracy and 90.37%
  answer F1; and achieved 90.95% Explanation BERT Score and 80.83% cosine similarity.
- **Why It Matters:** It improves retrieval efficiency, answer accuracy, and explanation consistency
  relative to traditional and semantic retrieval.
- **Limitation:** General LLMs lack telecommunications-specific standards, control logic, and intent
  templates.
- **Future Work:** Privacy-preserving retrieval and network-aware adaptive retrieval driven by
  topology and live traffic.

Quality audit: the output provides a coherent problem-method-evaluation-results story with specific
architecture and quantitative results. The framework contribution was conservatively rejected from
the Contribution field because of entity/category checks, but it remains explicit in Overview and
Methods. Related work stays separated and unsupported significance/future claims were rejected.

### Provenance and presentation audit

- Paper A: 14 final claims, 34 evidence references, 0 invalid mappings, maximum quote length 495.
- Paper B: 22 final claims, 36 evidence references, 0 invalid mappings, maximum quote length 500.
- Every displayed quote is an exact substring of its declared `ParsedPaper` chunk.
- No final claim contains an internal `[EC###]` marker.
- No unsupported comparison, invented evidence ID, fabricated limitation, or fabricated future work
  survived.

### Regression results

- Focused insight/provider/API suite: 200 passed
- Full backend: 352 passed
- Ruff check: passed
- Ruff format check: 85 files already formatted
- Frontend: 17 tests passed across 4 files
- ESLint: passed
- TypeScript/Vite production build: passed with Vite 6.4.3
- Docker Compose configuration: passed
- Known warning: upstream Starlette/AnyIO `BlockingPortal` deprecation warning

Gemini retains current-schema sanitization and bounded 429/503 retries under mocked regression
coverage; no repeated Gemini live request was made. Ollama remains supported by the common pipeline;
no model was downloaded, tuned, or subjected to a long run. Search, ranking, PDF acquisition,
GROBID parsing, graph functionality, and Milestone 3 were not changed.

### Final freeze decision

- EVL completed two different real papers: **YES**
- Recurring invalid-output failure resolved in acceptance runs: **YES**
- Useful grounded overview when supported: **YES**
- Genuine qualitative and quantitative findings retained: **YES**
- Useful grounded Why It Matters: **YES**
- Exact provenance preserved: **YES**
- Unsupported claims still rejected: **YES**
- Gemini structurally compatible: **YES**
- Ollama supported: **YES**
- All tests/build checks pass: **YES**

**MILESTONE 2: FROZEN**

## 20. Final full-document context delivery (authoritative continuation)

This section supersedes the freeze decision above. Pipeline version:
`m2-final-v5-full-document`.

### Context strategy

- EVL Gemma and Gemini advertise `full_document`.
- Ollama `qwen3:1.7b` advertises `hierarchical_sections`.
- Full-document mode sends one copy of the parsed paper in source reading order, grouped under
  preserved section headings. Abstract, Introduction, split contribution lists, methods,
  evaluation/results, discussion, limitations/open issues/outlook, and conclusion are retained.
- Bibliography/reference sections and small affiliation/email-only chunks are excluded.
- Hierarchical mode reads every non-reference section through ordered Introduction/Problem,
  Methods/Approach, Results/Evaluation, Discussion, Limitations/Open Issues/Future Work, and
  Conclusion packets. It no longer relies on arbitrary top-k selection.
- Both strategies retain the same evidence IDs, <=500-character exact source spans, ledger,
  validation, cache semantics, final insight schema, API response, and frontend representation.

### Exact-paper context audit

Paper: **Intellicise Wireless Networks Meet Agentic AI: A Security and Privacy Perspective**
(arXiv 2602.15290).

- Strategy: `full_document`
- Packets: 1
- Evidence spans: 142 across 110 source chunks
- Source quote characters: 38,152
- Maximum quote length: 498
- Invalid exact chunk/quote mappings: 0
- Duplicate evidence IDs: 0
- First content: Abstract
- Last content: VII. Conclusions
- Explicit contribution bullets delivered: 5
- Bibliography/reference sections delivered: no

### Single authorized EVL acceptance run

- Provider/model: `evl_gemma` / `gemma4`
- Fresh cache: yes (`m2-final-v5-full-document`)
- Approximate wall time to failure: 81 seconds
- Stage: evidence extraction
- Original finish reason: `length`
- Original response: substantively rich but truncated inside the final JSON string
- JSON error: `Unterminated string starting at: line 257 column 16 (char 10035)`
- Serialization-only correction attempted: exactly once
- Correction finish reason: `length`
- Correction content: empty/null
- Final provider error: `EVL Gemma returned empty structured output`
- Structural claims parsed: none, because malformed top-level JSON correctly remains a whole-response
  failure
- v16 rejection distribution: not reached
- Validated ledger: not created
- Final synthesis/insight text: not created

Before truncation, the raw response had recovered all five author-declared contributions and a much
deeper set of approach/method claims, evaluation evidence, findings, significance, and future work.
This confirms the full-document delivery solved the prior context-coverage failure, but the run does
not satisfy acceptance because the response never became valid structured input.

Diagnostic artifacts:

- `backend/data/llm_diagnostics/80d65e415fd447f08741a5ed128bd396-evidence_extraction-original.json`
- `backend/data/llm_diagnostics/80d65e415fd447f08741a5ed128bd396-evidence_extraction-correction.json`

### Regression results

- Focused context/semantic/provider suite: 212 passed
- Full backend suite: 367 passed
- Final architecture/version regression before the acceptance run: 58 passed
- Known warning: upstream Starlette/AnyIO `BlockingPortal` deprecation warning
- No frontend code changed; frontend tests were not rerun.
- No Gemini live call was made.
- No long Ollama extraction was run.
- Exactly one EVL full-paper run was made for this continuation; it was not repeated.

### Final freeze decision

The context architecture is implemented and locally validated, but the required live acceptance run
did not produce final researcher-facing insights or a validated ledger. Therefore the acceptance
standard is not met.

**MILESTONE 2: NOT FROZEN**

> **Authoritative status:** The historical Section 20 result above is superseded by the successful
> output-budget run in Section 21 and the completed closeout in Section 22.
>
> **MILESTONE 2: FROZEN**
