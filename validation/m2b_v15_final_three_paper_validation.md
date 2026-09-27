# M2B v15 Final Three-Paper Validation

## Environment

- Extraction version: `m2b-v15-generalized-insights`
- Model: `qwen3:1.7b`
- Parser/cache version: `0.9.1-crf+frontmatter-v1`
- Diagnostics: existing optional `InsightDiagnostics`, enabled for every run
- Papers, in order: `Hamid_Paper2.pdf`, `Paper1_StreetWeave_A_Declarative_Grammar_for_Street-Overlaid_Visualization_of_Multivariate_Data(5).pdf`, and `Shamma_Paper1.pdf`
- Fresh extractions: 3; all reported `cached: false`
- Model calls: exactly 9 (3 per paper)
- Mechanical retries: 0
- No production code, model settings, prompts, selectors, validators, or parser behavior changed.

## Paper 1: GSCache

### Parsing Health

- Title present; abstract field absent.
- Quality: `healthy`; provenance: `grobid`.
- 39 sections and 71 chunks.
- Extraction runtime: **181.139 seconds**.
- Retained counts: Problem 2, Methods 3, Contributions 3, Findings 5, Why It Matters 0, Audience 1, Limitations 3, Future Work 0.

### Final v15 Extraction and Claim Audit

| Field | Claim | Rating | Reason |
|---|---|---|---|
| Research Problem | Real-time rendering of scientific volume datasets is challenging due to high Monte Carlo noise and the need for efficient sampling strategies. | GOOD | Directly captures the paper's high-variance real-time volume-rendering problem. |
| Research Problem | Volume rendering requires low sample counts and advanced post-processing to maintain acceptable output quality. | PARTIAL | The evidence describes real-time rendering practice, but the claim overstates it as a general requirement. |
| Methods | Use 3D Gaussian splatting to represent path-space radiance with multi-level hierarchy. | GOOD | Directly supported and central to the method. |
| Methods | Optimize cache sizes with sub-sampling from higher to lower levels. | GOOD | Accurately represents the multi-level cache allocation strategy. |
| Methods | Evaluate method using runtime and image quality metrics. | GOOD | Directly supported by the evaluation design. |
| Key Contributions | Introduce a real-time path-space cache for volume rendering that stores attenuated radiance at different path lengths. | GOOD | Direct current-paper contribution. |
| Key Contributions | Optimize cache with noisy samples from the renderer and support cache hit handling. | PARTIAL | Noisy-sample optimization is supported; “cache hit handling” is not supported by the cited evidence. |
| Key Contributions | Provide an effective, fast, and easy-to-integrate alternative to world- and image-space-based radiance caches. | GOOD | Matches the paper's stated comparative contribution. |
| Main Findings | GSCache achieves higher image quality at low sample rates compared to baseline path tracers. | UNSUPPORTED | The cited quote says only that baselines were used; it gives no quality result. |
| Main Findings | GSCache reduces image noise and improves overall quality at equivalent sampling rates. | GOOD | Directly supported by PSNR and comparative visual-quality evidence. |
| Main Findings | GSCache demonstrates significant improvements in rendering speed with optimal cache sampling coefficients. | PARTIAL | The evidence supports a bias/quality/speed trade-off, not a “significant” speed improvement. |
| Main Findings | GSCache provides a robust, easy-to-integrate solution for scientific volume rendering. | GOOD | Supported by the paper's result/conclusion language. |
| Main Findings | GSCache maintains comparable rendering quality to state-of-the-art neural radiance caching methods. | UNSUPPORTED | The cited quote states that NRC was compared, not the comparison outcome. |
| Target Audience | Scientific visualization researchers and developers seeking efficient, real-time volume rendering solutions. | PARTIAL | Scientific-visualization relevance is explicit; the specific researcher/developer audience is inferred. |
| Limitations | GSCache introduces bias in rendering results, with no guaranteed error bounds for bias. | GOOD | Explicit limitation; the v14 false uncertainty rejection is fixed. |
| Limitations | GSCache is not suitable for applications requiring completely unbiased rendering results. | GOOD | Explicit and correctly qualified. |
| Limitations | GSCache's performance is affected by the number of lights and transfer function choices. | PARTIAL | It omits the important exception that caching times are not meaningfully affected. |

### Missing Core Concepts

| Missing Concept | Failure Stage | Evidence |
|---|---|---|
| Full declared contribution set, especially rapid adaptation to lighting/transfer-function changes | MODEL_OMISSION | Introduction/contribution evidence reached the core stage, but no distinct retained item covered it. |
| Quantitative runtime behavior, cache-overhead amortization, and important ablation findings | MODEL_OMISSION | Evaluation/result evidence reached stage 2, but no accurate candidates captured these results. |
| Multi-pass inefficiency, GPU-memory cost, application-specific scheduling, morphology reinitialization, and short-term memory | SELECTOR_LOSS | These parsed limitation passages were absent from the boundary evidence catalog. |
| Joint rasterization/optimization, compression, refitting, viewport-focused optimization, self-training, and world-space-cache future directions | SELECTOR_LOSS | Only a generic future-work header and unrelated passages reached stage 2. |
| Fast/easy integration as scientific significance | VALIDATOR_FALSE_NEGATIVE | A directly supported candidate reused a validated contribution quote but was rejected as `semantic_concept_coverage`. |

### v15 Assessment

Core-story coverage is **moderate**. Limitations improved substantially, but concrete future work and Why It Matters remain absent, and two retained Findings lack result-bearing evidence.

## Paper 2: StreetWeave

### Parsing Health

- Title and abstract present.
- Quality: `healthy`; provenance: `grobid`.
- 31 sections and 58 chunks.
- Extraction runtime: **177.327 seconds**.
- Retained counts: Problem 2, Methods 2, Contributions 3, Findings 2, Why It Matters 1, Audience 1, Limitations 2, Future Work 1.

### Final v15 Extraction and Claim Audit

| Field | Claim | Rating | Reason |
|---|---|---|---|
| Research Problem | The need for a structured design space for street and pedestrian network visualizations. | PARTIAL | Correct paper-level need, but the cited challenge passage does not itself establish the design-space conclusion. |
| Research Problem | Challenges in visualizing complex street networks with diverse data types. | GOOD | Directly supported by overplotting, limited-space, and spatial-detail challenges. |
| Methods | System implementation using JavaScript, React.js, and D3.js. | GOOD | The implementation evidence identifies the React/D3 stack; the concise claim is substantively accurate. |
| Methods | Evaluation through semi-structured interviews with domain experts. | GOOD | Directly supported. |
| Key Contributions | Design spaces for geospatial visualizations. | UNSUPPORTED | The evidence is a generic opening statement in a prior-work subsection, not StreetWeave's contribution. This is a current-paper/prior-work attribution failure. |
| Key Contributions | Declarative grammar for street-overlaid visualizations. | UNSUPPORTED | The cited passage establishes a need for structured design spaces, not the claimed grammar contribution. |
| Key Contributions | Support for integrating multiple thematic and spatial data layers. | GOOD | Direct current-system capability/contribution. |
| Main Findings | Expert feedback highlights ease of use and workflow simplification compared with traditional tools. | GOOD | Correct qualitative evaluation finding. |
| Main Findings | StreetWeave facilitates detailed sidewalk accessibility analysis through sub-segment granularity and enhanced visualization. | MISCLASSIFIED | The evidence describes a use-case procedure, not an observed evaluation finding. |
| Why It Matters | StreetWeave supports the integration of diverse spatial and thematic data layers. | MISCLASSIFIED | Supported as a contribution/capability, but it is not a synthesis of scientific significance. |
| Target Audience | Urban planners, transportation engineers, and public health analysts. | GOOD | Explicit subset of the stated primary target users. |
| Limitations | Quantitative assessment of visualization reproduction within the broader design space is limited. | GOOD | Explicit current-paper limitation. |
| Limitations | StreetWeave lacks a modular plugin system for custom visualization components outside Vega-Lite. | GOOD | Explicit current-paper limitation. |
| Future Work | Formal user studies across diverse domains and expertise levels will evaluate visualization effectiveness. | GOOD | Explicit author plan with correct future strength. |

### Missing Core Concepts

| Missing Concept | Failure Stage | Evidence |
|---|---|---|
| Declared design-space contribution based on 45 papers and declared use-case contribution | SELECTOR_LOSS | The core catalog retained the bare contribution header and a general grammar sentence, but not the substantive bullet claims. |
| Systematic-review/coding method and design-space construction | MODEL_OMISSION | Method evidence was parsed/routed, but only implementation and interview evaluation survived. |
| Expert findings on reproducibility, rapid iteration, uncertainty analysis, and added-computation constraints | MODEL_OMISSION | Rich expert-feedback evidence reached stage 2; only ease/workflow simplification was generated accurately. |
| No formal user study as a limitation | VALIDATOR_FALSE_NEGATIVE | A valid candidate was generated from the explicit limitation and rejected as `unsupported_attribute_or_detail`. |
| SVG scaling, empirical-default, and manual pipeline-integration limitations | SELECTOR_LOSS | The combined section was selected, but these concrete sentences were absent from the catalog. |
| Planned UTK/Curio integration | VALIDATOR_FALSE_NEGATIVE | The explicit “we plan to integrate” candidate was rejected by `future_work_guard`. |
| Coverage assessment, what-if analysis, broader planning grammars, WebGL/WebGPU, and gallery plans | MODEL_OMISSION | Several forward passages reached stage 2; no valid retained candidates represented them. The WebGL candidate correctly failed because it strengthened a plan into a guaranteed outcome. |
| A genuine Problem + Contribution + Finding significance synthesis | FIELD_CLASSIFICATION_ERROR | The retained Why It Matters item is only a current-system capability. |

### v15 Assessment

Core-story coverage is **partial and still precision-compromised**. Boundary fields, audience, and capability filtering improved materially, but a prior-work contribution still survives and the explicit contribution story is mostly absent.

## Paper 3: Bayesian Reasoning Machine

### Parsing Health

- Title recovered as `Bayesian reasoning machine on a magnetotunneling junction network`.
- Abstract absent; opening section remains truncated.
- Quality: `front_matter_degraded`.
- Provenance: `grobid+grobid_header_fallback(title)[front_matter_degraded]`.
- 9 sections and 22 chunks.
- Extraction runtime: **196.262 seconds**.
- Retained counts: Problem 2, Methods 2, Contributions 1, Findings 0, Why It Matters 0, Audience 0, Limitations 1, Future Work 1.

### Final v15 Extraction and Claim Audit

| Field | Claim | Rating | Reason |
|---|---|---|---|
| Research Problem | The challenge of generating sub-nanosecond probability samples using traditional Bayesian networks. | PARTIAL | Fast sampling is central, but the evidence states the proposed MTJ capability rather than the full problem; missing front matter limits framing. |
| Research Problem | The inefficiency of independent sampling methods in handling evidence on child nodes. | GOOD | Directly supported background problem. |
| Methods | Implementation of independent sampling on an MTJ grid. | GOOD | Directly supported. |
| Methods | Use of stochastic Landau-Lifshitz-Gilbert simulations. | GOOD | Directly supported. |
| Key Contributions | Development of 2D grid-based mapping for general Bayesian graphs. | PARTIAL | The paper presents a mapping strategy, but the cited sentence emphasizes that new mapping/restructuring algorithms are still needed. |
| Limitations | Non-idealities in parent-node MTJs and dipole coupling introduce inference errors. | GOOD | The simulation section identifies these non-idealities and evaluates their accuracy impact. |
| Future Work | Up-sizing components and post-fabrication calibration are planned to improve process-variability tolerance. | GOOD | Supported by the explicit future-work statement. |

### Missing Core Concepts

| Missing Concept | Failure Stage | Evidence |
|---|---|---|
| Complete problem framing and abstract-level contribution story | PARSING_LIMITATION | Abstract and beginning of section 1 are genuinely unavailable in `ParsedPaper`. |
| Voltage-controlled node generation, magnetostrictive conditional coupling, and full grid architecture as contributions/methods | MODEL_GENERATION_ERROR | Candidates were generated from overly narrow or mismatched evidence snippets and correctly rejected. |
| Quantitative process-variability findings and graph-parameter sensitivity | MODEL_GENERATION_ERROR | A substantively correct variability candidate cited only a non-substantive figure-description excerpt; stronger “feasible” wording also lost the source's “potential” qualifier. |
| Sub-nanosecond sampling enabling larger-graph inference under performance constraints as significance | VALIDATOR_FALSE_NEGATIVE | The claim was directly supported by current-paper evidence but rejected as `semantic_concept_coverage`. |
| Additional framing/findings that may have appeared in the damaged article opening | PARSING_LIMITATION | These cannot be assessed or attributed to v15 because the source text is absent. |

### v15 Assessment

Core-story coverage is **partial and source-constrained**. Precision improved and genuine Future Work now survives, but no Findings or Why It Matters remain even though readable body evidence supports both.

## Cross-Paper Results

| Paper | GOOD | PARTIAL | UNSUPPORTED | MISCLASSIFIED | Core Story Coverage |
|---|---:|---:|---:|---:|---|
| GSCache | 10 | 5 | 2 | 0 | Moderate |
| StreetWeave | 9 | 1 | 2 | 2 | Partial; precision-compromised |
| Bayesian Reasoning Machine | 5 | 2 | 0 | 0 | Partial; parsing-constrained |
| **Total** | **24** | **8** | **4** | **2** | Mixed |

No evidence quote was rejected because an excerpt ended mid-word. No relational `associated with` phrase was misread as epistemic uncertainty. Genuine uncertainty/conditional-strength protections remained active.

## v14 vs v15 Comparison

| Area | v14 Pattern | v15 Result | Assessment |
|---|---|---|---|
| Attribution | StreetWeave retained prior taxonomy, prior-tool limitation, and generic statements as current work. | Prior-tool limitation and most generic findings are gone, but one prior-work contribution remains. | **Improved, not fixed** |
| Bare contribution headers | A detailed grammar contribution survived using only “Our key contributions are.” | The bare header did not directly validate a detailed claim. | **Improved** |
| Findings | StreetWeave retained three capabilities and a generic statement; Shamma retained a method as a finding. | StreetWeave retains one use-case capability as a finding; Shamma is conservatively empty. GSCache gained two evidence-insufficient findings. | **Mixed improvement** |
| Limitations | GSCache retained one qualified-imprecise item; StreetWeave retained a reversed prior-work limitation; Shamma had two partial items. | GSCache retains two strong limitations plus one partial; StreetWeave retains two explicit limitations; Shamma retains one strong limitation. | **Improved** |
| Future Work | Empty for all three papers. | Valid author plans survive for StreetWeave and Shamma; GSCache remains empty because concrete passages still suffer selector loss. | **Improved** |
| Why It Matters | Empty for GSCache/Shamma; one narrow StreetWeave capability. | Same broad pattern: empty GSCache/Shamma and one StreetWeave capability misclassified as significance. | **Unchanged / still weak** |
| Audience | StreetWeave's explicit audience was lost to mid-word truncation; Shamma retained an unsupported inferred audience. | StreetWeave audience is recovered; Shamma is conservatively empty; GSCache remains inferred/partial. | **Improved** |
| Qualifier preservation | `associated with` caused a false uncertainty rejection; cache-timing exception was lost. | Bias limitation now survives, but the cache-timing exception is still lost. | **Improved, incomplete** |

The aggregate ratings moved from v14's **20 GOOD / 10 PARTIAL / 5 UNSUPPORTED / 4 MISCLASSIFIED** to v15's **24 GOOD / 8 PARTIAL / 4 UNSUPPORTED / 2 MISCLASSIFIED**. The numerical direction is favorable, but it does not erase the remaining trust-critical attribution error.

## Remaining Generic Issues

1. **Current-paper/prior-work attribution is still not reliable enough for contributions.** StreetWeave retained generic prior-work evidence as a current-paper contribution.
2. **Evidence-bearing Findings remain inconsistent.** Capability filtering improved, but GSCache retained two result claims whose cited excerpts merely state that comparisons were performed.
3. **Concrete boundary coverage remains uneven.** GSCache's limitation/future passages and several StreetWeave limitation sentences still disappear at selection.
4. **Why It Matters remains ineffective.** Valid significance candidates can be rejected, while a plain capability can survive as significance.
5. **Small-model evidence association remains a bottleneck.** Shamma generated several scientifically reasonable concepts against snippets too narrow to support them.

### Proven Validator False Positives

- GSCache: two Findings retained from comparison-setup evidence without comparison outcomes; one speed claim overstates a trade-off; one limitation loses the explicit cache-timing exception.
- StreetWeave: one prior-work contribution, one unsupported grammar contribution, one use-case capability retained as a Finding, and one capability retained as Why It Matters.
- Shamma: no proven retained validator false positive.

### Proven Validator False Negatives

- GSCache: a directly supported fast/easy-integration significance candidate rejected by `semantic_concept_coverage`.
- StreetWeave: explicit no-formal-user-study limitation rejected; explicit UTK/Curio author plan rejected by `future_work_guard`.
- Shamma: directly supported sub-nanosecond/larger-graph significance candidate rejected by `semantic_concept_coverage`.

No important loss was attributed to merge/deduplication or coverage-limit postprocessing.

## Shamma Parsing Impact

The new parser recovered the title and exposed degradation honestly, but the abstract and beginning of section 1 remain unavailable. Missing framing attributable to those absent passages is classified as `PARSING_LIMITATION`, not a v15 semantic failure. The readable technical body still contains architecture, simulation, process-variability, and future-calibration evidence; omissions or validation losses involving those passages are genuine semantic-pipeline issues.

## Final M2B Freeze Assessment

**BLOCKED BY SEVERE TRUST ISSUE**

StreetWeave still retains prior-work evidence as a current-paper contribution, proving that the trust-critical attribution safeguard is not yet reliable enough to freeze M2B v15.
