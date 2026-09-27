# M2B v16 Final Three-Paper Validation

## Environment

- Extraction version: `m2b-v16-trust-hardening`
- Model: `qwen3:1.7b`
- Parser/cache version: `0.9.1-crf+frontmatter-v1`
- Diagnostics: existing optional zero-call insight diagnostics were enabled for every run.
- Papers, in required order: `test papers/Hamid_Paper2.pdf`, `test papers/Paper1_StreetWeave_A_Declarative_Grammar_for_Street-Overlaid_Visualization_of_Multivariate_Data(5).pdf`, and `test papers/Shamma_Paper1.pdf`.
- Fresh extraction count: **3**.
- Model-call count: **9** (3 per paper).
- Cache status: all three extraction results reported `cached: false`; each used an isolated fresh validation cache.
- Mechanical retries: **0**.
- Production code was unchanged. The only new task artifacts are validation outputs and this report.

## Paper 1: GSCache

### Parsing Health

- Title: `GSCache: Real-Time Radiance Caching for Volume Path Tracing using 3D Gaussian Splatting`
- Abstract present: no.
- Parser health: `healthy`.
- Parser provenance: `grobid`.
- Sections: 39.
- Chunks: 71.
- Parse time: 14.210 seconds.

### Frozen v16 Extraction

- Runtime: **166.284 seconds**.
- Model calls: 3; cache hit: no; retries: 0.
- Counts: Problem 1, Methods 4, Contributions 1, Findings 2, Why It Matters 0, Audience 0, Limitations 0, Future Work 0.

| Field | Frozen claim | Evidence reference |
|---|---|---|
| Research Problem | Real-time rendering of scientific volume datasets faces challenges with high Monte Carlo noise due to sampling variance. | `chunk-a51f7fd74c8aaff3c0a6` |
| Methods | Use 3D Gaussian Splatting to represent path-space radiance with multi-level hierarchy. | `chunk-c50dd7226d9c1c5c7c71` |
| Methods | Implement throughput importance sampling to adjust cache sampling probability and account for early terminations. | `chunk-f5d0ebe414b35a741e93` |
| Methods | Optimize cache with sub-sampled levels to prioritize paths with higher image contribution. | `chunk-036e732bd5a4429b9ba6` |
| Methods | Use N = 300k initial samples with exponentially sub-sampled levels for cache initialization. | `chunk-7ea48fa51317acf8724e` |
| Key Contributions | Introduce real-time path-space cache using 3D Gaussian Splatting to store attenuated radiance at different path lengths. | `chunk-036e732bd5a4429b9ba6` |
| Main Findings | Our path-space radiance cache achieves higher image quality at low sample rates. | `chunk-191ffad8151c69943246` |
| Main Findings | The method achieves comparable runtime performance to state-of-the-art neural radiance caching (NRC). | `chunk-315f3a7cef3a5abad1fc` |

### Claim Audit

| Field | Claim | Rating | Evidence Assessment | Reason |
|---|---|---|---|---|
| Research Problem | Real-time rendering of scientific volume datasets faces challenges with high Monte Carlo noise due to sampling variance. | GOOD | The quote explicitly identifies real-time volume rendering, Monte Carlo noise, and high sampling variance. | Accurate paper-level problem framing. |
| Methods | Use 3D Gaussian Splatting to represent path-space radiance with multi-level hierarchy. | UNSUPPORTED | The cited quote says only that a Gaussian cache represents path-space radiance; it does not establish 3D Gaussian Splatting or a multi-level hierarchy. | The method is present elsewhere in the paper, but this retained evidence does not ground the full claim. |
| Methods | Implement throughput importance sampling to adjust cache sampling probability and account for early terminations. | PARTIAL | The quote supports adjustment for early cache termination to preserve unbiased path sampling, but not the named sampling method or probability-adjustment detail. | Core action is supported; important method detail is not grounded by the cited excerpt. |
| Methods | Optimize cache with sub-sampled levels to prioritize paths with higher image contribution. | UNSUPPORTED | The quote explains that shorter paths generally contribute more, but does not state sub-sampling or cache optimization. | The evidence supplies a rationale without grounding the claimed operation. |
| Methods | Use N = 300k initial samples with exponentially sub-sampled levels for cache initialization. | GOOD | The quote states the sample count and exponential sub-sampling directly. | Exact method detail. |
| Key Contributions | Introduce real-time path-space cache using 3D Gaussian Splatting to store attenuated radiance at different path lengths. | GOOD | A current-paper sentence explicitly says “we introduce” the cache and describes its stored quantity. | Correct ownership and substantive contribution support. |
| Main Findings | Our path-space radiance cache achieves higher image quality at low sample rates. | GOOD | The evidence explicitly reports significantly higher image quality at low sample rates. | Outcome-bearing result evidence. |
| Main Findings | The method achieves comparable runtime performance to state-of-the-art neural radiance caching (NRC). | GOOD | The evidence reports comparable timings and explains the overhead/early-termination trade-off. | Correct comparison direction and material qualifier retained in the evidence. |

### Missing Core Concepts

| Missing Concept | Failure Stage | Explanation |
|---|---|---|
| Full declared contribution set: adaptation to lighting/transfer-function/slicing changes, training from noisy Monte Carlo data, and non-invasive integration | SELECTOR_LOSS | The substantive enumerated contribution items were not preserved in the core evidence catalog; the model saw only a broader introduction statement for the retained cache contribution. |
| Evaluation design and quantitative ablations for cache size, sampling coefficient, regularization, quality, runtime, and memory | MODEL_OMISSION | Relevant evaluation/results evidence reached the stages, but the model retained only two high-level outcomes and no evaluation-method claim. |
| Concrete limitations: multi-pass inefficiency, GPU-memory cost, application-specific scheduling, morphology-change reinitialization, short-term memory, bias, and lack of guaranteed error bounds | SELECTOR_LOSS | The boundary pack contained a generic limitations heading but omitted the concrete limitation subsections/passages. |
| Explicit future directions: joint rasterization/optimization, compression, refitting, viewport-focused optimization, self-training, and world-space caching | SELECTOR_LOSS | The concrete forward-looking passages did not reach the boundary-stage evidence catalog. |
| Two retained method claims have narrower citations than their wording | VALIDATOR_FALSE_POSITIVE | Evidence validation proved quote occurrence but allowed claims whose main method attributes were not supported by the selected quote. |

### v16 Trust Checks

- Attribution: no prior-work/background claim was retained as GSCache's contribution, finding, or limitation.
- Findings: both retained Findings cite actual outcomes; comparison setup was not converted into an outcome.
- Boundary fields: Limitations and Future Work are conservatively empty, but this is a significant selector-driven recall gap.
- Why It Matters: empty; unsafe single-component candidates were rejected rather than presented as significance.
- Qualifiers: the retained runtime comparison includes the relevant overhead/termination explanation in its evidence; no serious reversal remains.
- Audience: empty; no domain-inferred audience was retained.

### Paper Assessment

Core-story coverage: **Moderate**. The extraction communicates the central problem, cache idea, initialization detail, contribution, and two principal outcomes. A researcher would still miss most declared contributions, the evaluation design, and the paper's concrete limitation/future-work story.

## Paper 2: StreetWeave

### Parsing Health

- Title: `StreetWeave: A Declarative Grammar for Street-Overlaid Visualization of Multivariate Data`.
- Abstract present: yes.
- Parser health: `healthy`.
- Parser provenance: `grobid`.
- Sections: 31.
- Chunks: 58.
- Parse time: 6.004 seconds.

### Frozen v16 Extraction

- Runtime: **188.166 seconds**.
- Model calls: 3; cache hit: no; retries: 0.
- Counts: Problem 2, Methods 3, Contributions 1, Findings 1, Why It Matters 0, Audience 1, Limitations 3, Future Work 3.

| Field | Frozen claim | Evidence reference |
|---|---|---|
| Research Problem | The need for a structured and dedicated approach to visualize multivariate data on streets. | `chunk-18845cfd778f8bb06f7c` |
| Research Problem | Challenges in visualizing street and pedestrian networks due to complex structures and diverse analytical requirements. | `chunk-18845cfd778f8bb06f7c` |
| Methods | System implementation using JavaScript, React.js, Leaflet.js, and D3.js. | `chunk-90029e37120bfe8eb2ad` |
| Methods | Data coding process for 45 existing street-overlaid visualizations. | `chunk-02f3d120e0c23ecaf001` |
| Methods | Evaluation through semi-structured interviews with urban domain experts. | `chunk-ed877f6360cd167a1817` |
| Key Contributions | Declarative grammar for street-overlaid visualization of multivariate data. | `chunk-0c4a43d3cd42dbfbed77` |
| Main Findings | Expert feedback highlights the ease of use and workflow simplification of StreetWeave compared to existing tools like D3 and GeoPandas. | `chunk-0b1b29c3ea47284c15c1` |
| Target Audience | Urban planners, transportation engineers, and public health analysts. | `chunk-0c4a43d3cd42dbfbed77` |
| Limitations | StreetWeave lacks a modular plugin system for custom visualization components outside of Vega-Lite. | `chunk-b6d1235185a5e9f53e91` |
| Limitations | StreetWeave relies on SVG-based rendering, which can become slow for large street networks or multiple data layers. | `chunk-b6d1235185a5e9f53e91` |
| Limitations | StreetWeave has not yet been evaluated for usability and effectiveness across diverse user groups. | `chunk-b6d1235185a5e9f53e91` |
| Future Work | Formal user studies will be conducted across diverse domains and user expertise levels to evaluate the effectiveness of different street-overlaid visualizations. | `chunk-b6d1235185a5e9f53e91` |
| Future Work | StreetWeave will be ported to WebGL and WebGPU for better performance in large-scale street network visualization. | `chunk-b6d1235185a5e9f53e91` |
| Future Work | Integration with UTK and Curio frameworks will enable modular component usage in urban data analysis workflows. | `chunk-b6d1235185a5e9f53e91` |

### Claim Audit

| Field | Claim | Rating | Evidence Assessment | Reason |
|---|---|---|---|---|
| Research Problem | The need for a structured and dedicated approach to visualize multivariate data on streets. | PARTIAL | The excerpt is truncated after “tailored” and does not preserve the full object of that tailoring. | The paper supports the need, but the retained evidence loses scope. |
| Research Problem | Challenges in visualizing street and pedestrian networks due to complex structures and diverse analytical requirements. | GOOD | The quote explicitly states the complex structures, large datasets, and diverse requirements. | Accurate problem statement. |
| Methods | System implementation using JavaScript, React.js, Leaflet.js, and D3.js. | GOOD | The implementation stack is stated directly. | Correct method/implementation classification. |
| Methods | Data coding process for 45 existing street-overlaid visualizations. | GOOD | The evidence explicitly describes the 45-paper corpus and structured thematic coding. | Correct design-space construction method. |
| Methods | Evaluation through semi-structured interviews with urban domain experts. | GOOD | The quote states five experts and semi-structured interviews. | Correct evaluation procedure. |
| Key Contributions | Declarative grammar for street-overlaid visualization of multivariate data. | GOOD | A current-paper sentence explicitly says “we introduce StreetWeave” and identifies it as a declarative grammar. | The v15 prior-work ownership failure is eliminated for the retained contribution. |
| Main Findings | Expert feedback highlights the ease of use and workflow simplification of StreetWeave compared to existing tools like D3 and GeoPandas. | GOOD | The expert-feedback passage reports this qualitative theme directly. | Legitimate qualitative outcome, not merely a feature or use case. |
| Target Audience | Urban planners, transportation engineers, and public health analysts. | GOOD | The passage explicitly calls these roles primary target users. | No domain-based audience inference. |
| Limitations | StreetWeave lacks a modular plugin system for custom visualization components outside of Vega-Lite. | GOOD | The explicit third limitation states this restriction directly. | Correct current-paper limitation. |
| Limitations | StreetWeave relies on SVG-based rendering, which can become slow for large street networks or multiple data layers. | GOOD | The explicit fourth limitation states the rendering and scale constraint. | Correct negative limitation and scope. |
| Limitations | StreetWeave has not yet been evaluated for usability and effectiveness across diverse user groups. | UNSUPPORTED | The cited excerpt instead concerns quantitative design-space coverage and expressiveness. | The paper states a no-formal-user-study limitation elsewhere, but this retained citation does not support this claim. |
| Future Work | Formal user studies will be conducted across diverse domains and user expertise levels to evaluate the effectiveness of different street-overlaid visualizations. | GOOD | The passage explicitly says “we will conduct” those studies. | Correct author-owned future action. |
| Future Work | StreetWeave will be ported to WebGL and WebGPU for better performance in large-scale street network visualization. | PARTIAL | The evidence says “we plan to port” and frames scalability as the intended benefit. | The direction is correct, but “will be ported” is stronger than the source's plan. |
| Future Work | Integration with UTK and Curio frameworks will enable modular component usage in urban data analysis workflows. | PARTIAL | The passage says the authors plan to integrate the frameworks, enabling modular use. | The action and purpose are supported, but the claim compresses a plan into a definite future construction. |

### Missing Core Concepts

| Missing Concept | Failure Stage | Explanation |
|---|---|---|
| Declared design-space contribution based on 45 papers and declared use-case/ease/flexibility/reproducibility contribution | SELECTOR_LOSS | The core pack preserved the bare “Our key contributions are” header but not the substantive enumerated items. The grammar was recovered from a separate current-paper introduction sentence. |
| Rich expert-feedback findings on reproducibility, iteration, uncertainty analysis, and required additional computations | MODEL_OMISSION | The expert-feedback evidence reached the boundary stage, but only ease/workflow simplification became a retained finding. |
| Distinction between no formal user study and no quantitative design-space coverage assessment | MODEL_GENERATION_ERROR | Both source limitations reached the stage, but the model paired the user-study wording with the coverage-assessment excerpt. |
| Empirically ungrounded defaults and manual GIS pipeline-integration constraints | SELECTOR_LOSS | These concrete limitation sentences did not reach the selected boundary evidence. |
| Additional author plans for quantitative coverage, what-if analysis, broader planning grammars, and a gallery | MODEL_OMISSION | Forward-looking evidence reached the boundary stage, but no distinct candidates survived for these directions. |
| Safe multi-component scientific-significance synthesis | MODEL_GENERATION_ERROR | No candidate combined grounded Problem, Contribution, Finding, and Discussion evidence; v16 correctly left Why It Matters empty. |
| The user-study limitation retained against the wrong excerpt | VALIDATOR_FALSE_POSITIVE | Quote-presence validation did not catch the semantic evidence mismatch. |

### v16 Trust Checks

- Attribution: the generic prior-work design-space candidate that blocked v15 was generated but rejected as `contribution_support_mismatch`; no prior-work contribution, finding, or limitation survives.
- Contribution support: the sole retained contribution uses explicit current-paper “we introduce” evidence. Declared-list recall remains incomplete because substantive bullet items were lost before generation.
- Findings: the single retained item is outcome-bearing expert feedback. Use-case and capability candidates were rejected.
- Boundary fields: two explicit limitations are strong; one retained limitation has mismatched evidence. Three genuine author plans survive, with two modest modal-strength compressions.
- Why It Matters: empty and conservative; the v15 plain-capability significance error is gone.
- Qualifiers: no serious reversal, although “plan to” became “will” in two Future Work summaries.
- Audience: explicit primary-user evidence supports the retained roles.

### Paper Assessment

Core-story coverage: **High**. The extraction gives the problem, review/coding and implementation methods, expert evaluation, core grammar contribution, one central qualitative finding, explicit audience, limitations, and future plans. It still misses two declared contributions, several expert observations, and safe significance synthesis.

## Paper 3: Bayesian Reasoning Machine

### Parsing Health

- Title: `Bayesian reasoning machine on a magnetotunneling junction network`.
- Abstract present: no.
- Parser health: `front_matter_degraded`.
- Parser provenance: `grobid+grobid_header_fallback(title)[front_matter_degraded]`.
- Degradation reasons: `missing_abstract`; `body_section_sequence_starts_late`.
- Sections: 9.
- Chunks: 22.
- Parse time: 3.993 seconds.

### Frozen v16 Extraction

- Runtime: **198.753 seconds**.
- Model calls: 3; cache hit: no; retries: 0.
- Counts: Problem 1, Methods 0, Contributions 0, Findings 0, Why It Matters 0, Audience 0, Limitations 2, Future Work 0.

| Field | Frozen claim | Evidence reference |
|---|---|---|
| Research Problem | The challenge of generating sub-nanosecond probability samples using traditional Bayesian networks. | `chunk-b97391e307f2ed1080a2` |
| Limitations | Non-idealities in parent node MTJs affect inference accuracy. | `chunk-ccb5a503447ac1f7d749` |
| Limitations | Process variability sources impact encoded probability parameters. | `chunk-24f4ea646d26635e6f00` |

### Claim Audit

| Field | Claim | Rating | Evidence Assessment | Reason |
|---|---|---|---|---|
| Research Problem | The challenge of generating sub-nanosecond probability samples using traditional Bayesian networks. | PARTIAL | The evidence states the proposed MTJ network's sub-nanosecond capability, not the complete problem or a limitation of “traditional Bayesian networks.” | Fast sampling is central, but missing front matter and the candidate's framing leave the problem incomplete. |
| Limitations | Non-idealities in parent node MTJs affect inference accuracy. | UNSUPPORTED | The cited quote enumerates three non-idealities but does not state their effect on inference accuracy. | The paper analyzes that effect elsewhere; the retained evidence does not ground the claimed consequence. |
| Limitations | Process variability sources impact encoded probability parameters. | UNSUPPORTED | The excerpt says variability sources are lumped into a distribution; it does not report that they change encoded parameters. | Adjacent source text supports an impact, but the retained quote is semantically insufficient. |

### Missing Core Concepts

| Missing Concept | Failure Stage | Explanation |
|---|---|---|
| Complete abstract/introduction-level problem and contribution framing | PARSING_LIMITATION | The abstract and beginning of section 1 are genuinely absent from `ParsedPaper`; v16 cannot recover content that was not parsed. |
| Grid-based MTJ architecture as a Method | VALIDATOR_FALSE_NEGATIVE | A valid current-paper architecture candidate cited “our grid-based architecture” but was rejected as `entity_mismatch`. |
| Voltage-controlled probabilistic nodes, magnetostrictive conditional coupling, mapping algorithms, and stochastic LLG simulation | MODEL_GENERATION_ERROR | Readable technical evidence reached the core stage, but generated candidates used narrow or mismatched excerpts and were correctly rejected. |
| Process-variability/error trends and graph-parameter sensitivity findings | MODEL_OMISSION | Strong result passages reached the boundary stage, but the model produced no adequately evidenced outcome candidates for them. |
| Sub-nanosecond sampling result | VALIDATOR_FALSE_NEGATIVE | A current-paper candidate cited the explicit “Our approach allows sub-ns” result but was rejected by the method/capability-as-finding restriction. |
| Planned up-sizing/post-fabrication calibration for process variability | VALIDATOR_FALSE_NEGATIVE | The candidate cited the explicit “we plan to address in a future work” passage but was rejected as `entity_mismatch`. |
| Two retained limitations have consequence wording not grounded by their selected excerpts | VALIDATOR_FALSE_POSITIVE | Exact-quote validation passed, but semantic claim-to-evidence alignment did not. |

### v16 Trust Checks

- Attribution: no prior-work/background statement was retained as a current-paper contribution, finding, or limitation.
- Findings: conservatively empty. No setup or capability statement survived as a finding, but valid readable results were also lost.
- Boundary fields: two limitation concepts survive with inadequate evidence association; explicit calibration future work is a proven validator false negative.
- Why It Matters: empty rather than inferred from incomplete source framing.
- Qualifiers: no retained serious reversal; the conditional future action was rejected rather than strengthened.
- Audience: empty; no domain-inferred audience was retained.

### Paper Assessment

Core-story coverage: **Low**. Parser damage legitimately removes the abstract and opening framing, but the surviving body still contains substantial architecture, simulation, results, and future-work evidence that v16 does not retain. That latter loss is a semantic-pipeline recall problem, not a parsing limitation.

## Cross-Paper Claim Results

| Paper | GOOD | PARTIAL | UNSUPPORTED | MISCLASSIFIED | Core Story Coverage |
|---|---:|---:|---:|---:|---|
| GSCache | 5 | 1 | 2 | 0 | Moderate |
| StreetWeave | 10 | 3 | 1 | 0 | High |
| Bayesian Reasoning Machine | 0 | 1 | 2 | 0 | Low; parsing-constrained and semantically sparse |
| **Total** | **15** | **5** | **5** | **0** | Mixed |

The unsupported ratings are evidence-association failures: the claims are generally discussed elsewhere in the same papers, but the cited excerpts do not substantively support the retained wording. No retained Finding has unsupported comparison or result wording, and no retained claim attributes prior work to the current paper.

## v15 → v16 Comparison

| Area | v15 | v16 | Assessment |
|---|---|---|---|
| Attribution | StreetWeave retained prior-work material as a current contribution. | The prior-work candidate is explicitly rejected; no current/prior ownership error survives. | **Improved; trust-critical blocker fixed** |
| Contribution support | StreetWeave retained unsupported generic/design-space and grammar claims; other contribution evidence was uneven. | Every retained contribution uses explicit current-paper introduction language. | **Improved** |
| Contribution recall | 7 contribution claims across the papers, including unsupported/partial items. | 2 precise claims survive; substantive declared-list bullets still suffer selector loss. | **Regressed in recall, improved in precision** |
| Findings precision | Two GSCache comparison-setup claims and a StreetWeave use-case capability survived as Findings. | All three retained Findings are outcome-bearing: two GSCache results and one expert-feedback result. | **Improved** |
| Findings recall | GSCache had five, StreetWeave two, and Shamma none. | GSCache has two, StreetWeave one, and Shamma none; several valid result passages are omitted. | **Regressed** |
| Limitations | GSCache had three mostly strong items, StreetWeave two, and Shamma one strong item. | GSCache is empty; StreetWeave has two strong plus one evidence-mismatched item; Shamma has two evidence-mismatched items. | **Regressed in coverage/evidence association** |
| Future Work | Valid StreetWeave and Shamma plans survived; GSCache was empty. | Three StreetWeave plans survive, but GSCache and Shamma are empty despite explicit source directions. | **Mixed: broader StreetWeave coverage, Shamma regression** |
| Why It Matters | One StreetWeave capability was misclassified as significance; the other papers were empty. | All are conservatively empty; no unsupported significance survives. | **Improved precision; recall remains weak** |
| Qualifier preservation | A GSCache exception was lost; some overstatement remained. | No serious reversal survives; two StreetWeave “plan to” statements are compressed to “will.” | **Improved, still imperfect** |
| Target Audience | StreetWeave was explicit/good; GSCache was inferred/partial; Shamma empty. | StreetWeave remains explicit/good; GSCache and Shamma are safely empty. | **Improved precision** |
| Overall precision | 24 GOOD, 8 PARTIAL, 4 UNSUPPORTED, 2 MISCLASSIFIED; included trust-critical attribution/result errors. | 15 GOOD, 5 PARTIAL, 5 UNSUPPORTED, 0 MISCLASSIFIED; unsupported items are evidence-association issues, not false ownership or result claims. | **Improved on trust-critical error types despite lower yield** |
| Core-story coverage | GSCache Moderate; StreetWeave Partial/precision-compromised; Shamma Partial/parsing-constrained. | GSCache Moderate; StreetWeave High; Shamma Low/parsing-constrained and semantically sparse. | **Mixed** |

The lower v16 GOOD count reflects a much smaller, more conservative output (25 claims versus 38), not a simple aggregate quality regression. The remaining evidence-association errors are real, but the v15 blocking classes—prior-work attribution, unsupported outcome comparisons, capability-as-Finding, and capability-as-significance—do not recur.

## Trust-Critical Error Review

| Question | Answer | Evidence |
|---|---|---|
| Prior-work attribution errors remaining? | **NO** | StreetWeave's generic prior-work contribution candidate was rejected; all retained contributions cite explicit current-paper language. |
| Unsupported result/comparison claims remaining? | **NO** | Every retained Finding uses outcome-bearing evidence; no comparison setup becomes a claimed win. |
| Capability-as-Finding errors remaining? | **NO** | StreetWeave use cases/capabilities and Shamma capability candidates did not survive as Findings. |
| Fabricated Future Work remaining? | **NO** | All retained Future Work is tied to explicit StreetWeave author plans. |
| Unsupported Why It Matters remaining? | **NO** | Why It Matters is empty for all three papers. |
| Serious qualifier reversals remaining? | **NO** | Two StreetWeave plans are mildly strengthened from “plan to” to “will,” rated PARTIAL, but no condition, exception, direction, or causal conclusion is seriously reversed. |
| Audience inference errors remaining? | **NO** | The sole audience claim uses explicit “primary target users” evidence; inferred GSCache/Shamma audiences are absent. |

### Proven Validator False Positives Remaining

- GSCache: two Method claims survive with citations that do not state their principal method attributes; one further Method is PARTIAL because its named sampling detail is absent from the excerpt.
- StreetWeave: a no-user-study limitation survives against evidence about quantitative design-space coverage.
- Shamma: two limitation claims survive with evidence that names variability/non-idealities but not the claimed effects.

These are deferred claim-to-evidence rebinding/semantic-alignment weaknesses. They are not prior-work attribution, fabricated science, or unsupported performance/result comparisons.

### Proven Validator False Negatives Remaining

- Shamma: valid grid-architecture Method rejected as `entity_mismatch`.
- Shamma: explicit sub-nanosecond result rejected by the method/capability-as-finding restriction.
- Shamma: explicit planned up-sizing/calibration Future Work rejected as `entity_mismatch`.
- StreetWeave: some valid current-system/contribution capability wording is rejected by the conservative contribution-support guard, although the principal grammar contribution survives.

No important loss was observed in merge, deduplication, or final assembly; no failure is classified as `POSTPROCESSING_LOSS`.

## Remaining Generic Issues

### TRUST/PRECISION issues

1. Exact quote occurrence does not guarantee semantic claim-to-evidence alignment. Five retained claims use excerpts too narrow to ground their substantive wording.
2. Future-plan compression can strengthen “we plan to” into “will.” This did not create a false direction or guaranteed outcome here, but it loses modal precision.

### RECALL/COVERAGE issues

1. Enumerated contribution items are still lost when parsed list structure leaves only a bare contribution header in the evidence catalog.
2. Concrete boundary selection is uneven: GSCache loses most limitations/future directions, while StreetWeave receives much stronger boundary coverage.
3. Conservative Findings validation removes invalid capabilities but also rejects at least one legitimate Shamma result.
4. Why It Matters remains safely empty rather than useful; precision is acceptable, synthesis recall is not.
5. Small-model evidence association remains a bottleneck on the degraded Shamma input, even where readable technical evidence exists.

## Shamma Parsing Impact

`front_matter_degraded` is material: the abstract is absent and the beginning of section 1 is incomplete. Missing abstract-level framing and any claims confined to that lost text are `PARSING_LIMITATION`, not v16 semantic failures. The title fallback is correctly disclosed in provenance.

The parser does preserve readable sections on MTJ nodes, conditional coupling, grid architecture, mapping, simulations/results, process variability, conclusions, and explicit calibration future work. Losses involving those preserved passages are semantic pipeline failures: model omission/generation errors or validator false negatives. The two retained limitation evidence mismatches are validator false positives, not parser-caused errors.

## M2B Freeze Decision

**READY TO FREEZE**

The severe v15 blocker is fixed: no prior-work statement is retained as a current-paper contribution, and no unsupported comparison outcome, capability-as-Finding, fabricated future direction, unsupported significance claim, audience inference, or serious qualifier reversal survives. Retained contributions are explicitly current-paper and retained Findings are outcome-bearing, including a qualitative expert result. The remaining weaknesses are conservative recall loss and deferred evidence association/rebinding—important known limitations, but not the severe recurring trust patterns defined as freeze blockers. Shamma's genuinely absent front matter is separated from failures on its readable body.

No further M2B semantic tuning is recommended before moving to the PDF acquisition fix and Milestone 3.
