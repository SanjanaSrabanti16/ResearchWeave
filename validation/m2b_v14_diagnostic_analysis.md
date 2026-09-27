# M2B v14 Diagnostic Analysis

## Instrumentation

- Extraction baseline: `m2b-v14-conditional-future`
- Model: `qwen3:1.7b`
- Changed/added implementation files:
  - `backend/app/services/insight_diagnostics.py` (new opt-in trace recorder)
  - `backend/app/services/insight_service.py` (optional recorder hooks and reason-returning wrappers around existing boolean decisions)
  - `backend/tests/test_insights.py` (focused observability and behavior-preservation tests)
- Diagnostic artifacts:
  - `validation/diagnostics/gs_cache_v14_trace.json`
  - `validation/diagnostics/streetweave_v14_trace.json`
  - this report
- Mechanism: callers may explicitly pass an `InsightDiagnostics` recorder to `InsightService.extract`. Without that argument, diagnostics are off. The normal API does not create diagnostic files or expose a changed response schema.
- Each trace records stage identity, selected excerpts and routing categories, exact raw model content, parsed candidates, evidence-ID resolution, named validation decisions, final retention/disposition, and final claim provenance.
- No additional LLM call is made. Both diagnostic runs used the normal three calls.
- Prompt strings, evidence selection, evidence budgets/order, model and generation parameters, extraction version, cache key semantics, acceptance thresholds, field definitions, and final response schema were not intentionally changed.
- Where a validator previously returned a boolean, the diagnostic path now returns the first failing rule name and the production boolean wrapper tests whether that reason is absent. The order and predicates of the existing checks are preserved.
- Temporary parsed/insight caches used to guarantee fresh runs were removed after the two traces were saved, preventing full parsed-paper text from being retained in diagnostics.
- Verification before model runs: 54 insight tests passed; focused diagnostic subset was 3 passed; Ruff check and format check passed for changed Python files.

## Diagnostic Run 1: GSCache

- Paper: `GSCache: Real-Time Radiance Caching for Volume Path Tracing using 3D Gaussian Splatting`
- Runtime: **170.085 seconds**
- Fresh cache: yes (`cached: false`)
- Model calls: 3
- Selected evidence counts: core story 79; evidence/boundaries 85; synthesis 52
- Candidate counts: core story 9; evidence/boundaries 7; synthesis 6
- Final field counts:

| Field | Count |
|---|---:|
| Research Problem | 2 |
| Methods | 3 |
| Key Contributions | 3 |
| Main Findings | 3 |
| Why It Matters | 0 |
| Target Audience | 1 |
| Limitations | 1 |
| Future Work | 0 |

### Findings

| Issue | Evidence Reached Model? | Candidate Generated? | Validator Result | Failure Stage | Explanation |
|---|---|---|---|---|---|
| Explicit bias/no-error-bound limitation missing | Yes: E73-E77 contained the bias limitation; E74 stated that bias has no guaranteed error bound. | Yes: “GSCache introduces bias in rendering results, with no guaranteed error bounds for bias.” | Rejected as `uncertainty_mismatch`. | VALIDATOR_FALSE_NEGATIVE | The claim accurately combines adjacent bias statements. The uncertainty regex interpreted “associated with bias” in E74 as hedging, although it describes the error's relation to bias rather than uncertainty. |
| Other concrete limitations missing: multi-pass inefficiency, GPU memory, application-specific scheduler, morphology reinitialization, short-term memory | No. Only the generic Limitations/Future header and the Bias subsection reached the compact stage catalog. | No. | Not reached. | SELECTOR_LOSS | The relevant parsed sections existed, but their concrete passages were absent from stage-2 selected excerpts. |
| Retained limitation blurs rendering versus cache timing | Yes: E68 explicitly said lights/transfer function can affect overall rendering but not caching time. | Yes: model stated “GSCache's performance is affected...” | Retained. | MODEL_GENERATION_ERROR + VALIDATOR_FALSE_POSITIVE | The model dropped the cache-timing exception, and the semantic checks did not preserve this contrast/qualifier. |
| Runtime nuance missing | Yes: E65 said timings were comparable to NRC and cache-rendering overhead was offset by early path termination; E67 gave the C=0.5 tradeoff. | No runtime candidate. | Not applicable. | MODEL_OMISSION | Appropriate current-paper result evidence reached stage 2 but qwen3:1.7b generated only image-quality findings. |
| Concrete future directions missing | No: specific joint rasterization, compression, refitting, viewport optimization, self-training, and world-space-cache directions did not enter the compact catalog. | No valid candidate for those directions. | Not applicable. | SELECTOR_LOSS | Stage 2 received only a generic future-work heading plus unrelated/recommendation passages, not the concrete action sentences. |
| Invalid future-work candidates | The cited evidence reached stage 2, but neither excerpt stated future work: E60 was an observed bias/variance tradeoff and E78 recommended alternative methods to users requiring unbiased results. | Yes: two candidates recast these as future research/development. | Both rejected by `future_work_guard`. | MODEL_GENERATION_ERROR (correctly rejected) | The guard prevented current findings/user recommendations from becoming Future Work. |
| Why It Matters absent | Yes: validated problem/contribution/finding claims plus result and discussion evidence reached synthesis. | Yes: 3 candidates. | All rejected by `semantic_concept_coverage`. | MODEL_GENERATION_ERROR; validator rejection supported | The candidates added concepts not present in their single cited excerpts (for example real-time/noise or adaptive performance). No fully supported synthesis candidate was generated. |

## Diagnostic Run 2: StreetWeave

- Paper: `StreetWeave: A Declarative Grammar for Street-Overlaid Visualization of Multivariate Data`
- Runtime: **191.263 seconds**
- Fresh cache: yes (`cached: false`)
- Model calls: 3
- Selected evidence counts: core story 77; evidence/boundaries 67; synthesis 39
- Candidate counts: core story 10; evidence/boundaries 11; synthesis 8
- Final field counts:

| Field | Count |
|---|---:|
| Research Problem | 2 |
| Methods | 4 |
| Key Contributions | 3 |
| Main Findings | 5 |
| Why It Matters | 1 |
| Target Audience | 0 |
| Limitations | 1 |
| Future Work | 0 |

### Findings

| Issue | Evidence Reached Model? | Candidate Generated? | Validator Result | Failure Stage | Explanation |
|---|---|---|---|---|---|
| Prior work became a paper contribution | Yes: E32 said “Their work provides a comprehensive taxonomy...” | Yes: “Provide a comprehensive taxonomy for map-like visualizations.” | Retained. | MODEL_GENERATION_ERROR + VALIDATOR_FALSE_POSITIVE | Contribution validation has no current-paper/prior-work restriction for this case, so lexical overlap was enough to pass. |
| Bare contribution header supports a detailed claim | Only a header: E25 was exactly “Our key contributions are”. | Yes: the model attached the grammar contribution to that header. | Retained through the explicit-contribution shortcut. | MODEL_GENERATION_ERROR + VALIDATOR_FALSE_POSITIVE | The shortcut treats contribution markers as sufficient without requiring the cited excerpt to contain the candidate's substance. |
| General/capability statements became Findings | Yes: E10-E12 were introduction claims about lower barriers, adaptation, reuse, and extensibility. | Yes: three candidates were generated directly as `main_findings`. | All retained. | FIELD_CLASSIFICATION_ERROR + VALIDATOR_FALSE_POSITIVE | `supports_finding` rejects method-like text but does not require these capability statements to be observed evaluation results. |
| A generic prior-work statement became a StreetWeave finding | Yes: E19 referred to “these visualizations” generally. | Yes: model attributed the capability specifically to StreetWeave. | Retained. | MODEL_GENERATION_ERROR + VALIDATOR_FALSE_POSITIVE | The evidence does not name StreetWeave, and the current-paper attribution checks did not reject it. |
| Important evaluation findings omitted | Yes: E48-E55 included workflow simplification, rapid iteration, reproducibility, uncertainty-analysis possibilities, and additional-computation constraints. | Only one finding candidate, for ease/workflow simplification. | That candidate was retained. | MODEL_OMISSION | The stage had substantial expert-feedback evidence, but qwen3:1.7b stopped after one evaluation theme and instead generated introduction capabilities as findings. |
| False programming-expertise limitation retained | Yes: E30 said “their use” demands programming expertise and referred to earlier tools, while StreetWeave claims to lower barriers. | Yes: the model attributed it to StreetWeave. | Retained. | MODEL_GENERATION_ERROR + VALIDATOR_FALSE_POSITIVE | Limitation validation does not perform a prior-work/current-system antecedent check. |
| Explicit limitations missing | Mostly no. E59 was only the generic “has a few limitations” sentence; detailed no-user-study, coverage, plugin, SVG-scaling, default-choice, and workflow-integration limitations were not represented in the compact excerpts. | No candidates for those concrete limitations. | Not applicable. | SELECTOR_LOSS | The combined section reached stage 2, but the excerpt construction favored its header, conclusion, and future-action passages rather than the detailed limitation sentences. |
| Target Audience missing | Yes: E13 explicitly listed urban planners, transportation engineers, public-health analysts, climate researchers, and other domain experts. | Yes: 5 correct audience candidates. | All rejected as `quote_mismatch`. | VALIDATOR_FALSE_NEGATIVE | The 280-character selected excerpt ended mid-word (`experien`); normalized whole-word matching could not find that truncated token in the source chunk (`experience`). |
| Why It Matters narrow | Yes: validated upstream story claims and E10-E15 reached synthesis. | Yes: 3 candidates. | Two rejected by `semantic_concept_coverage`; the narrow adaptation claim survived. | MODEL_GENERATION_ERROR; validator rejection supported | Two candidates added unsupported concepts (“interdisciplinary collaboration” and “urban domain experts”) relative to their cited single excerpts. The model did not produce a richer fully supported synthesis. |

## Future Work Diagnosis

### GSCache

- Did concrete future-work evidence reach stage 2? **No.** The stage contained E72's generic section introduction and bias/user-recommendation excerpts, but not the concrete joint-pipeline, compression, refitting, viewport, self-training, or world-space directions.
- Did qwen3:1.7b generate future-work candidates? **Yes, two, but neither represented genuine future-work evidence.**
- Candidate 1 recast the observed C bias/variance behavior (E60) as future optimization; candidate 2 recast a recommendation to use ReSTIR/Volume ReSTIR when unbiased output is required (E78) as future development.
- Both were rejected by `future_work_guard` before semantic validation.
- Dominant cause for the real future-work loss: **selector/excerpt loss**, followed by model generation errors on unrelated evidence. No GSCache Future Work validator false negative was observed.

### StreetWeave

- Did concrete future-work evidence reach stage 2? **Yes.** E63 (formal user studies), E65 (simulation/predictive integration), E66 (WebGL/WebGPU), and E67 (UTK/Curio integration) were present.
- Did qwen3:1.7b generate future-work candidates? **Yes, three.**
- The valid formal-user-study candidate citing E63 was rejected by `future_work_guard` because the guard does not recognize the explicit construction “we will conduct”. This is a proven **VALIDATOR_FALSE_NEGATIVE**.
- The simulation/predictive-model candidate citing E65 was also rejected by the guard. Its generated wording (“will enhance”) may be stronger than the source's forward requirement, so the trace does not prove that the final semantic result should have been retained.
- The feature candidate citing E57 converted expert suggestions into “is planned”; rejection by `future_work_guard` was appropriate.
- The model omitted the explicitly selected WebGL/WebGPU and UTK/Curio plans (E66-E67).
- Dominant cause: **one proven narrow-guard false negative plus model omission/generation error**.

## Findings Diagnosis

- StreetWeave capability statements survived as Findings because qwen3:1.7b generated them in `main_findings` from E10-E12, and the existing finding guard only excludes recognized method descriptions, prior-work headings, and certain semantic conflicts. Plain capability claims are not excluded.
- The model itself therefore made the initial field-classification error; validation then produced false positives by retaining all three.
- Actual expert-evaluation evidence reached the model in E48-E55. The model generated one correct evaluation finding and omitted the other important themes.
- A fourth invalid finding came from E19, a generic statement about street-overlaid visualizations. The model made it StreetWeave-specific, and the validator did not detect the attribution change.
- GSCache's selected runtime result E65 produced no candidate, making that loss a model omission rather than selector or validator loss.

## Attribution Diagnosis

- The false StreetWeave taxonomy contribution originated from core-stage E32: “Their work provides a comprehensive taxonomy for maplike visualizations...”.
- The false StreetWeave programming-expertise limitation originated from boundary-stage E30: “However, their use demands programming expertise...”.
- The false StreetWeave-specific finding originated from E19's generic “these visualizations” statement.
- The model generated all three misattributions before validation.
- Existing current-paper/prior-work checking only operates on `main_findings` when every cited section heading matches a limited related-work heading pattern. It does not protect contributions or limitations, does not resolve antecedents such as “their”, and the subsection heading did not trigger the existing heading rule.
- All three survived, proving validator false positives rather than downstream/postprocessing errors.

## Limitations Diagnosis

### GSCache

- Explicit bias evidence reached stage 2 and produced an accurate candidate.
- That valid candidate was rejected as `uncertainty_mismatch` because “associated with bias” was treated as an uncertainty marker. This is a proven validator false negative.
- Other major limitations never reached the stage catalog, so their absence is selector loss.
- The retained light/transfer-function claim was generated without the source's “no meaningful impact on caching times” qualification and then incorrectly passed validation.

### StreetWeave

- The combined section reached stage 2, but only its generic limitation header was retained; the concrete limitation list did not. This is selector loss.
- The model filled the field using prior-tool evidence E29/E30 and a future technical passage E66.
- E29's invalid prior-tool claim was rejected, although the recorded rule was `uncertainty_mismatch` rather than attribution.
- E30's false StreetWeave attribution passed, a validator false positive.
- The E66 candidate failed `quote_mismatch` because its selected excerpt was cut mid-word, but the candidate was also substantively unsupported by that excerpt; it is not counted as a proven false negative.

## Why It Matters Diagnosis

### GSCache

- Validated upstream claims were available: two problems, three contributions, and three findings.
- Synthesis received selected evaluation/discussion excerpts plus the validated story evidence.
- The model generated three candidates; none was sufficiently supported by its cited single excerpt.
- All were rejected by `semantic_concept_coverage`. The trace supports model generation error/omission, not a proven validator false negative.

### StreetWeave

- Synthesis received validated problem/contribution/finding claims and relevant introduction/discussion/evaluation evidence.
- Three candidates were generated. Two added concepts absent from their cited evidence and were rejected; one narrow adaptation claim passed.
- The weak final field is therefore principally model-generation quality/coverage, compounded by false upstream contributions/findings. No postprocessing loss was observed.

## Cross-Paper Root Causes

### Selector

- GSCache's specific future directions and most explicit limitations never reached stage 2.
- StreetWeave's detailed limitation list was lost even though the combined section was selected.
- Long single chunks and excerpt prioritization can preserve a section header/future tail while dropping the concrete limitation body.

### Model

- Both papers show omissions despite appropriate evidence: GSCache runtime nuance and StreetWeave evaluation themes.
- GSCache generated future work from a current result and a user recommendation.
- StreetWeave generated prior-work/current-paper misattributions, capabilities as findings, and expert suggestions as a firm plan.
- Why It Matters candidates often added concepts beyond their cited single evidence item.

### Validator

- Proven false negatives:
  - GSCache's valid bias/no-error-bound limitation rejected because `associated with` triggered uncertainty handling.
  - StreetWeave's valid formal-user-study plan rejected because “we will conduct” is absent from the forward-action guard.
  - Five valid StreetWeave audience candidates rejected because excerpt truncation created a non-matching partial final token.
- Proven false positives:
  - GSCache's retained performance limitation lost the explicit cache-timing exception.
  - StreetWeave's bare-header contribution, prior-work taxonomy contribution, prior-work programming limitation, generic-to-StreetWeave finding, and three capability-as-finding claims all passed.

### Field Classification

- StreetWeave's three introduction capability statements were generated and retained as Findings.
- Expert suggestions were generated as Future Work with stronger planned-action wording.

### Postprocessing

- No important loss was attributed to merge/deduplication or coverage limits in either trace.
- Every observed missing final claim was already absent, rejected during evidence/semantic validation, or never selected/generated.

## Candidate Generic Fixes

No fixes were implemented.

| Priority | Generic candidate fix | Observed failure addressed | Seen in | Likely component | Precision risk |
|---|---|---|---|---|---|
| P0 | Add current-paper/prior-work/antecedent attribution checks across contributions, findings, and limitations; do not treat a bare contribution header as substantive support. | Prior taxonomy, prior-tool limitation, generic visualization statement, and header-only contribution survived as current-paper claims. | StreetWeave | Validator/evidence attribution | Low if conservative; possible false rejections when authors summarize prior work and their extension in one excerpt. |
| P0 | Make excerpt truncation word-boundary-safe or validate against the exact source span represented by the excerpt. | Five correct audience candidates were rejected because a 280-character excerpt ended at `experien` rather than `experience`. | StreetWeave | Evidence catalog/grounding | Very low; this repairs provenance representation rather than broadening semantic acceptance. |
| P0 | Require Findings evidence to express a current-paper observation/result, not merely a capability or generic statement. | Three capabilities and one generic statement survived as StreetWeave findings. | StreetWeave; GSCache showed the desired result style | Field validation | Moderate: qualitative findings can be phrased without standard result verbs. |
| P1 | Preserve concrete limitation/future sentences from combined or nested sections, then expand the future-action guard to explicit generic constructions such as “we will conduct” while retaining strength/condition checks. | GSCache future/limitations selector loss; StreetWeave limitation loss and formal-user-study false negative. | Both | Selector/catalog + future validator | Moderate: broader forward-language recognition must not admit completed work, suggestions, or guaranteed outcomes. |
| P1 | Narrow uncertainty detection so relational phrases such as “error associated with bias” do not automatically mark evidence as epistemically hedged. | Valid GSCache bias/no-error-bound limitation rejected. | GSCache | Semantic validator | Low-to-moderate; true association claims still need uncertainty preservation in causal restatements. |
| P2 | Improve synthesis generation/citation assembly so Why It Matters can cite the minimum set of validated upstream evidence needed for all claim concepts. | GSCache produced only under-supported synthesis; StreetWeave retained one narrow claim. | Both | Prompt/context mapping or synthesis postprocessing | Moderate: combining more evidence can encourage over-generalization unless causal checks remain strict. |

## Recommended v15 Scope

A small future v15 should be limited to:

1. P0 attribution and Findings-type precision checks, including removal of header-only contribution acceptance.
2. P0 word-boundary-safe excerpt provenance so exact valid evidence is not rejected mechanically.
3. P1 deterministic preservation of concrete limitation/future passages plus carefully expanded explicit-future phrasing.
4. P1 correction of the `associated with` uncertainty false positive.
5. Focused regression tests from synthetic examples before any new real-paper evaluation.

This task did not implement any of those semantic changes.
