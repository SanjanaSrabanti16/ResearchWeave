# M2B v14 Cross-Paper Generalization Validation

## Environment

- Project root: `<repository-root>`
- Extraction version: `m2b-v14-conditional-future`
- Production model: `qwen3:1.7b`
- Services: local Ollama and GROBID were reachable during preflight.
- Pipeline: unchanged three-stage production insight extraction; three model calls per completed paper extraction.
- Cache control: each paper used a new, paper-specific insight-cache directory; all three responses reported `cached: false`.
- Raw generated-claim counts, rejected-claim counts, and rejection reasons were not exposed by the existing pipeline. No instrumentation was added merely to obtain them.
- Paper files, in processing order:
  1. `test papers\Hamid_Paper2.pdf`
  2. `test papers\Paper1_StreetWeave_A_Declarative_Grammar_for_Street-Overlaid_Visualization_of_Multivariate_Data(5).pdf`
  3. `test papers\Shamma_Paper1.pdf`
- Completed extraction runs: 3. Paper 2 had one permitted mechanical retry before parsing/model generation because the validation harness initially supplied an unsupported acquisition label; the retry used the production-supported `upload` label. No paper was rerun for semantic quality.

## Paper 1

`GSCache: Real-Time Radiance Caching for Volume Path Tracing using 3D Gaussian Splatting`

### Parsing Health

- GROBID succeeded in 45.047 seconds.
- 4 authors, 39 sections, 71 chunks, 53 references, and 51,676 parsed text characters.
- Title and useful section structure were present.
- The body contained visible character-encoding artifacts such as `â€¢`, but the scientific prose remained readable and there was no severe content corruption.
- Fresh extraction completed in 169.396 seconds with 3 model calls and retained 13 claims.

### Final ResearchWeave Extraction

| Field | Retained claims |
|---|---:|
| Research Problem | 2 |
| Methods | 3 |
| Key Contributions | 3 |
| Main Findings | 3 |
| Why It Matters | 0 |
| Target Audience | 1 |
| Limitations | 1 |
| Future Work | 0 |

### Claim Audit

| Field | Extracted Claim | Rating | Audit Reason |
|---|---|---|---|
| Research Problem | Real-time rendering of scientific volume datasets is challenging due to high Monte Carlo noise and the need for efficient sampling strategies. | GOOD | Correctly states the central rendering problem and is directly supported by the cited introduction passage. |
| Research Problem | Volume rendering requires low sample counts and advanced post-processing to maintain acceptable output quality. | PARTIAL | The cited passage describes real-time rendering practice, but the claim presents it as a requirement and omits the paper's more specific high-variance/path-sampling problem. |
| Methods | Use 3D Gaussian splatting to represent path-space radiance in a multi-level hierarchy. | GOOD | Correctly captures the central cache representation. |
| Methods | Optimize the cache with sub-sampled levels to reduce computational resources. | PARTIAL | The hierarchy is sub-sampled, but the cited quote does not state the claimed resource-reduction purpose; the paper instead emphasizes allocating representational power by path contribution. |
| Methods | Evaluate the method using a baseline path tracer with uniform sampling and next-event estimation. | PARTIAL | The paper does use these baselines, but the cited quote only says “baseline volume path tracer” and does not itself support the added uniform-sampling/NEE detail. |
| Key Contributions | Introduce a real-time path-space cache for volume rendering that stores attenuated radiance at different path lengths. | GOOD | Directly supported and central to the paper. |
| Key Contributions | Optimize the cache in real time with noisy samples from the renderer. | GOOD | Directly supported by the author’s method/contribution statement. |
| Key Contributions | Provide an effective, fast, and easy-to-integrate alternative to world- and image-space-based radiance caches. | GOOD | Accurately reflects the paper's stated comparative contribution. |
| Main Findings | GSCache achieves higher image quality at low sample rates compared to baseline path tracers. | GOOD | Directly supported by the visual-quality results. |
| Main Findings | GSCache reduces noise and improves image quality at equivalent sampling rates. | GOOD | Correctly preserves the comparison with baseline and NRC at equivalent rates. |
| Main Findings | GSCache demonstrates significant improvements in rendering quality for volumetric visualization applications. | GOOD | The conclusion directly supports faster convergence and higher quality in less time. |
| Target Audience | Scientific visualization researchers. | PARTIAL | This is a reasonable inferred audience, but the cited sentence only establishes field relevance and does not explicitly identify intended users/readers. |
| Limitations | GSCache's performance is affected by the number of lights and transfer function choices. | PARTIAL | Overall rendering performance can be affected, but the evidence explicitly says caching times are not meaningfully affected; the claim loses that important distinction. |

Rating totals: **GOOD 8 / PARTIAL 5 / UNSUPPORTED 0 / MISCLASSIFIED 0**.

### Missing Important Concepts

| Missing Concept | Likely Failure Stage | Reason |
|---|---|---|
| The full four-part author-declared contribution story, including rapid adaptation to transfer-function/lighting changes and non-invasive integration | MODEL_OMISSION | The introduction and its contribution list were parsed and routed to the core stage, but the retained list only partially represents the declared set. |
| Quantitative adaptation behavior: the cache surpasses the baseline after fewer than 16 samples | MODEL_OMISSION | The result was present in the parsed Visual Quality section and the evaluation material was available to the boundary stage. |
| Runtime finding: comparable NRC timing and cache overhead amortization through early path termination | MODEL_OMISSION | Runtime-result content was parsed, but no retained finding represented it. |
| Multi-pass rasterization/optimization inefficiency, high GPU-memory demand, use-case-specific learning-rate scheduling, reinitialization after morphology changes, short-term memory, and biased output without guaranteed error bounds | MODEL_OMISSION | The explicit Limitations and Future Directions material was parsed and included in boundary evidence, yet only one imprecise limitation survived. |
| Concrete future directions: joint rasterization/optimization, cache compression, refitting, viewport-focused optimization, self-training, and world-space cache designs | SELECTOR | The future-work pool found the section, but the compact boundary catalog primarily preserved generic future-work headers/conclusion language rather than the specific action sentences. |
| Why the method matters scientifically: better interactive path-traced scientific-volume exploration without extra rendering cost and with easy renderer integration | MODEL_OMISSION | Relevant validated problem/contribution/finding material existed, but grounded synthesis returned no claim. |

### Paper Assessment

Core-story coverage is **moderate**. The output gives a reader the central problem, the Gaussian path-space cache idea, and the headline image-quality result with good precision. It is not researcher-complete because it omits most explicit limitations, quantitative/runtime nuance, concrete future work, and all Why It Matters synthesis.

## Paper 2

`StreetWeave: A Declarative Grammar for Street-Overlaid Visualization of Multivariate Data`

### Parsing Health

- GROBID succeeded in 8.791 seconds.
- 3 authors, 31 sections, 58 chunks, 59 references, and 59,767 parsed text characters.
- Title, abstract text, section structure, contribution list, expert-feedback section, and combined limitations/conclusions/future-work section were present.
- Minor encoding artifacts occurred, but there was no severe text corruption.
- Fresh extraction completed in 202.743 seconds with 3 model calls and retained 16 claims.
- One mechanical pre-extraction attempt failed before parsing/model generation because the harness used an invalid acquisition-method label. The single allowed retry succeeded; this was not a semantic rerun.

### Final ResearchWeave Extraction

| Field | Retained claims |
|---|---:|
| Research Problem | 2 |
| Methods | 4 |
| Key Contributions | 3 |
| Main Findings | 5 |
| Why It Matters | 1 |
| Target Audience | 0 |
| Limitations | 1 |
| Future Work | 0 |

### Claim Audit

| Field | Extracted Claim | Rating | Audit Reason |
|---|---|---|---|
| Research Problem | The need for a structured design space for street and pedestrian network visualizations. | PARTIAL | This is the correct paper-level need, but the cited quote only states general complexity/data/analysis challenges and does not substantiate the design-space conclusion. |
| Research Problem | Challenges in visualizing complex street networks with diverse data types. | GOOD | Correctly captures the overplotting, limited-space, and spatial-detail challenge. |
| Methods | Conduct a systematic review of 45 existing street-overlaid visualizations. | GOOD | The paper describes the 45-work qualitative/systematic analysis and selection process. |
| Methods | Perform thematic coding of visualizations using a framework from geospatial literature. | GOOD | Directly supported by the data-coding procedure. |
| Methods | Implement StreetWeave as a grammar-based toolkit using JavaScript. | GOOD | Directly supported. |
| Methods | Develop a JSON-based specification for visualization authoring. | GOOD | Directly supported. |
| Key Contributions | Introduce a declarative grammar for structured visualization of multivariate data on streets. | UNSUPPORTED | The paper does make this contribution, but the cited evidence is only “Our key contributions are” and does not substantiate the claim; the grounded record is invalid as retained. |
| Key Contributions | Provide a comprehensive taxonomy for map-like visualizations. | UNSUPPORTED | The evidence says “Their work” and describes prior literature, not StreetWeave's contribution. This is a prior-work attribution error. |
| Key Contributions | Support integration of thematic and spatial data layers. | GOOD | This is a supported system-level contribution/capability of StreetWeave. |
| Main Findings | StreetWeave enables the creation of diverse street and pedestrian network visualizations through a declarative grammar. | UNSUPPORTED | The cited text refers generally to “these visualizations” enabling pattern discovery; it does not establish the StreetWeave-specific claim. |
| Main Findings | StreetWeave simplifies the creation of multivariate street-overlaid visualizations by reducing technical complexity. | MISCLASSIFIED | The statement is supported as a design claim/capability in the introduction, but the citation is not an evaluation result. |
| Main Findings | StreetWeave supports flexible adaptation across various applications and geographic contexts. | MISCLASSIFIED | This is a stated design capability, not an observed evaluation finding in the cited evidence. |
| Main Findings | StreetWeave provides reusable and extensible visualization capabilities for urban data analysis. | MISCLASSIFIED | This is a system contribution/capability rather than an empirical or analytical finding. |
| Main Findings | Expert feedback highlights the ease of use and workflow simplification of StreetWeave compared to existing tools. | GOOD | Correctly grounded in the five-expert feedback section and properly classified as an evaluation finding. |
| Why It Matters | StreetWeave supports flexible adaptation across diverse applications. | GOOD | The evidence supports this scientific/practical capability, although it is narrower than the paper's full significance. |
| Limitations | StreetWeave requires programming expertise, limiting accessibility to non-technical users. | UNSUPPORTED | The quote concerns prior GIS/tooling; the paper argues the opposite about StreetWeave. This is an antecedent/prior-work attribution error. |

Rating totals: **GOOD 8 / PARTIAL 1 / UNSUPPORTED 4 / MISCLASSIFIED 3**.

### Missing Important Concepts

| Missing Concept | Likely Failure Stage | Reason |
|---|---|---|
| The declared contribution of a design space organized by tasks, visual techniques, and data types | MODEL_OMISSION | The complete three-item contribution list was parsed and its introduction chunk reached the core stage, but this item was replaced by a prior-work taxonomy claim. |
| The declared contribution of use cases demonstrating ease of use, flexibility, and reproducibility | MODEL_OMISSION | The declared contribution list reached the core stage, but this item was not retained. |
| Primary target users: urban planners, transportation engineers, public-health analysts, climate researchers, and other domain experts without extensive programming experience | SELECTOR | This explicit audience statement was in the introduction, while the synthesis audience selection emphasized expert-feedback/evaluation excerpts and returned no audience. |
| Usage-scenario observations, including clearer trend tracing after changing orientation and adjacent-street service-request differences | MODEL_OMISSION | Usage scenarios were parsed, but Findings retained mostly introductory capability claims rather than scenario observations. |
| Expert feedback on reproducibility, rapid iteration, uncertainty analysis, GIS interoperability, and desired usability extensions | MODEL_OMISSION | The Expert Feedback section was selected, but only ease/workflow simplification survived. |
| Explicit limitations: no formal user study, unmeasured design-space coverage, no modular plugin system, SVG scaling limits, empirically unvalidated defaults, and manual GIS/pipeline coordination | MODEL_OMISSION | The explicit combined limitations section reached boundary evidence, but a false prior-work limitation was retained instead. |
| Explicit future work: formal studies, empirical defaults, coverage assessment, what-if analysis, simulation integration, broader planning grammars, WebGL/WebGPU, framework integration, and a gallery | UNKNOWN | Multiple future-work passages reached the boundary catalog; some satisfy the deterministic future-action guard. Raw pre-validation output is unavailable, so model omission cannot be distinguished from later validation loss. |

### Paper Assessment

Core-story coverage is **partial and precision-compromised**. Methods and the broad grammar idea are visible, and one expert-feedback finding is good. However, the declared contribution set is incomplete, most “findings” are actually introductory capabilities, the only limitation reverses the paper's accessibility argument, target audience is absent, and all explicit future work is lost.

## Paper 3

`Shamma_Paper1.pdf` (GROBID did not recover the title)

### Parsing Health

- GROBID succeeded in 4.637 seconds but returned a null title and an empty abstract field.
- 4 authors, 9 sections, 22 chunks, 33 references, and 17,340 parsed text characters.
- The first untitled section appears to contain only the latter part of the abstract; the opening problem context is missing.
- Background, device design, architecture, mapping, simulation results, and conclusion sections remained readable and useful.
- Fresh extraction completed in 196.105 seconds with 3 model calls and retained 10 claims.

### Final ResearchWeave Extraction

| Field | Retained claims |
|---|---:|
| Research Problem | 2 |
| Methods | 3 |
| Key Contributions | 1 |
| Main Findings | 1 |
| Why It Matters | 0 |
| Target Audience | 1 |
| Limitations | 2 |
| Future Work | 0 |

### Claim Audit

| Field | Extracted Claim | Rating | Audit Reason |
|---|---|---|---|
| Research Problem | The challenge of generating sub-nanosecond probability samples in probabilistic reasoning systems. | PARTIAL | Fast probability sampling is central, but the quote states the proposed capability rather than the challenge; the broader scaling problem for real-time inference on large Bayesian graphs is omitted. |
| Research Problem | The need for stochastic interaction mechanisms in Bayesian networks. | PARTIAL | Conditional stochastic interaction is central, but the cited text is a truncated abstract fragment and does not clearly establish the need or context. |
| Methods | Design of grid-based execution with topological order. | GOOD | Correctly represents the proposed Bayesian-graph execution architecture. |
| Methods | Implementation of independent sampling on MTJ-grid. | GOOD | Directly supported. |
| Methods | Use of stochastic Landau-Lifshitz-Gilbert simulations. | GOOD | Directly supported by the simulation method. |
| Key Contributions | Creation of a novel adaptation of MTJs for Bayesian inference-based algorithms. | GOOD | Correctly grounded in the conclusion and central to the paper. |
| Main Findings | MTJs with in-plane anisotropy allow controlled probability generation. | MISCLASSIFIED | The cited sentence is a device-design possibility/method, not an evaluation or simulation result. |
| Target Audience | Researchers interested in neuromorphic and image processing applications. | UNSUPPORTED | The cited text names examples of variability-tolerant non-Boolean applications; it does not identify the paper's intended audience. |
| Limitations | Non-idealities in parent node MTJs affect inference accuracy. | PARTIAL | The paper later demonstrates increased error, but the cited quote only lists non-ideality sources and does not state their measured effect. |
| Limitations | Dipole coupling imprecision introduces variability in conditional probabilities. | PARTIAL | The passage supports probabilistic failure to anti-correlate because of the energy barrier, but the claim recasts this as coupling imprecision and loses the mechanism/conditions. |

Rating totals: **GOOD 4 / PARTIAL 4 / UNSUPPORTED 1 / MISCLASSIFIED 1**.

### Missing Important Concepts

| Missing Concept | Likely Failure Stage | Reason |
|---|---|---|
| Full paper title and complete abstract/problem framing | PARSING | GROBID returned a null title, empty abstract field, and a truncated untitled opening section. |
| The central scalability motivation: software sampling does not meet real-time constraints as Bayesian graphs and required sample counts grow | MODEL_OMISSION | This argument is present in the parsed Simulation Results section but was not retained as the main problem/significance. |
| Voltage-controlled node probability generation and stress/dipole-controlled conditional correlation | MODEL_OMISSION | Both device mechanisms were parsed, but the Methods list emphasizes execution/mapping and LLG simulation instead. |
| Grid architecture details and the need for graph partitioning/restructuring to map arbitrary graphs with binary nearest-neighbor coupling | MODEL_OMISSION | The architecture and mapping sections were parsed, but these important constraints/components were not retained. |
| Results: correlation tunable from 0 to -1, sub-nanosecond sampling, prediction error rising with process variability, and sensitivity depending on graph parameters | MODEL_OMISSION | Simulation Results content was available to boundary selection, but the only retained “finding” was a method statement. |
| Why fast stochastic hardware matters for large-graph Bayesian inference under real-time constraints | MODEL_OMISSION | Supporting problem/contribution/result content was parsed, but synthesis returned no Why It Matters claim. |
| Planned post-fabrication calibration for process variability | UNKNOWN | The exact sentence reached the boundary catalog and passed the future-action guard. With no raw pre-validation claims, model omission cannot be separated from later semantic-validation rejection. |

### Paper Assessment

Core-story coverage is **partial-to-low**. The output identifies MTJs, grid execution, independent sampling, and stochastic LLG simulation, but it does not convey the decisive simulation findings or the full hardware mechanism/scalability argument. The title/abstract parse loss materially weakens problem framing.

## Cross-Paper Summary

| Paper | GOOD | PARTIAL | UNSUPPORTED | MISCLASSIFIED | Core Story Coverage |
|---|---:|---:|---:|---:|---|
| GSCache | 8 | 5 | 0 | 0 | Moderate: central method and headline quality results are clear; limitations, future work, runtime nuance, and significance are incomplete. |
| StreetWeave | 8 | 1 | 4 | 3 | Partial: implementation and one expert finding are recovered, but prior-work attribution and field confusion materially reduce trust. |
| Shamma_Paper1 | 4 | 4 | 1 | 1 | Partial-to-low: several methods survive, but core results, significance, and full framing are missing. |

Across all retained claims: **GOOD 20 / PARTIAL 10 / UNSUPPORTED 5 / MISCLASSIFIED 4**.

### Reliable Fields

- **Methods** generalized best: all papers retained multiple real implementation, analysis, or simulation methods. Some evidence/wording qualifiers were lost, but no method claim was wholly unsupported.
- **Research Problem** usually recovered the topic and broad technical difficulty, although it often converted solution statements into problem claims or omitted the decisive motivation.
- **Key Contributions** worked well for GSCache and the MTJ paper, but StreetWeave demonstrates that contribution evidence can be contaminated by a bare contribution-list header and prior-work passages.

### Recurring Weak Fields

- **Main Findings remains a recurring weak field.** GSCache performed well, but StreetWeave placed three system capabilities under Findings and the MTJ paper retained a device-design statement while omitting its simulation results.
- **Future Work failed to retain any claim on all three papers**, despite genuine future directions in each paper.
- **Limitations had poor recall on all papers** and produced a severe prior-work/reversal error for StreetWeave.
- **Why It Matters was absent for two papers and narrow for StreetWeave.** It did not consistently synthesize the full research value from validated problem, contribution, and result claims.
- **Target Audience was absent once and weakly inferred twice**, showing that relevance to a field or application is being treated as audience evidence.

### Recurring Failure Patterns

1. **Model omissions after relevant evidence selection:** important evaluation results, explicit limitations, and declared contributions were parsed/selected but not retained on at least two papers.
2. **Findings classification drift:** introductory capabilities or method descriptions became findings on StreetWeave and the MTJ paper.
3. **Weak evidence-to-entity attribution:** StreetWeave retained prior work as this paper's contribution and limitation; the MTJ audience claim inferred a readership from cited application examples.
4. **Qualifier/scope loss:** several otherwise valid claims dropped distinctions such as overall rendering versus cache timing, or listed non-idealities versus demonstrated accuracy effects.
5. **Sparse synthesis:** Why It Matters did not integrate the retained research story on GSCache or the MTJ paper.

### Validator False Negatives

- No validator false negative can be proven from the available artifacts because the unchanged pipeline does not expose raw generated claims or rejection reasons.
- There is, however, a repeated **post-selection Future Work loss**. StreetWeave and Shamma_Paper1 had concrete future-action evidence in the model catalog that passed the deterministic future-action guard, yet retained no Future Work. The loss is therefore either model omission or a later semantic-support false negative; the current evidence cannot distinguish them.
- GSCache differs: the selector found the future-work section, but the compact catalog emphasized generic future headers/conclusion language and did not reliably preserve the concrete action passages, making selector loss the more likely cause there.
- The v14 conditional-future change did not introduce observed future-work false positives, but this validation cannot establish that it improved recall because all three final Future Work arrays were empty.

### Parsing Issues

- GSCache and StreetWeave were structurally complete enough for extraction but contained minor PDF/GROBID character-encoding artifacts.
- Shamma_Paper1 had the only material parsing failure: null title, empty abstract field, and a truncated untitled opening section. The technical body and results remained usable.
- No paper was reparsed or replaced with an alternative PDF.

### Paper-Specific Issues

- **GSCache:** future-action sentences were not well represented in the compact boundary catalog; its extracted limitation blurred overall rendering and cache-specific timing.
- **StreetWeave:** antecedent/prior-work attribution caused the most serious precision failures; this paper also contains unusually rich combined limitation/future-work prose that the model largely ignored.
- **Shamma_Paper1:** title/abstract loss harmed framing, and the short technical paper lacked explicit audience language, making audience inference particularly unsafe.

### Generalization Assessment

v14 does **not yet maintain uniformly high precision or researcher-level core-story coverage across these three structurally different papers**. It generalizes reasonably for concrete methods and can produce strong headline findings when result prose is direct, as in GSCache. It is less reliable when it must distinguish prior work from current work, capabilities from findings, or extract dense limitation/future-work sections. The qwen3:1.7b staged pipeline also shows repeated omissions even when relevant evidence reaches a stage.

### Candidate Generic Issues

These are observations only; no fixes were implemented.

- Strengthen current-paper versus prior-work/antecedent attribution before retaining contributions, findings, limitations, and audience claims.
- Require Findings to cite actual evaluation/result observations rather than introductory capability statements.
- Preserve explicit limitation and future-action sentences in compact catalogs, especially from combined Discussion/Conclusion/Limitation/Future sections.
- Expose stage-level raw/rejection diagnostics in a future validation-capable design so model omissions can be separated from validator false negatives without changing extraction behavior during an experiment.
- Improve Why It Matters synthesis coverage while retaining conservative causal language.
- Treat field/domain mentions as insufficient audience evidence unless intended users/readers are explicit.
