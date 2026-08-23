# Phase 13 Manuscript Claim–Evidence Plan

Locked claims: 24

## Abstract

### A01 — problem_and_scope

The study evaluates a TinyML-oriented IoT intrusion-detection pipeline under device-generalization, compression, deployment, statistical, selection, and robustness constraints.

**Evidence:** figure_01_methodology_overview, table_01_dataset_protocol

**Drafting rule:** Use as scope statement; do not add unsupported benchmark claims.

### A02 — generalization_result

Strict leave-one-device-out evaluation covered nine devices, three model families, and 27 scientific runs without severe LODO cells under the locked gate.

**Evidence:** figure_02_lodo_generalization

**Drafting rule:** Use exact locked counts and model means only.

### A03 — selection_result

The balanced and relaxed multi-objective profiles selected the TinyML-MLP P50-QAT family; the strict sensitivity profile selected TinyML-MLP P50-FP32-FT.

**Evidence:** figure_04_pareto_selection

**Drafting rule:** Do not describe the strict sensitivity result as the final family.

### A04 — robustness_result

Across 27 locked robustness evaluations, five severe-fragility evaluations occurred and all were associated with HGB; the locked final model was unchanged.

**Evidence:** figure_05_input_robustness, table_06_robustness_results

**Drafting rule:** Do not imply retraining or post-test reselection.

## I. Introduction

### I01 — contribution

Contribution 1: a leakage-aware strict device-generalization evaluation is performed before compression/model-family selection.

**Evidence:** figure_01_methodology_overview, figure_02_lodo_generalization

**Drafting rule:** Present as a study-design contribution, not a universal novelty claim.

### I02 — contribution

Contribution 2: compression accuracy and deployment cost are evaluated jointly across a locked multi-seed architecture–variant matrix.

**Evidence:** figure_03_compression_tradeoff, table_03_compression_results, table_04_deployment_benchmarks

**Drafting rule:** Avoid claiming superiority before the selection subsection.

### I03 — contribution

Contribution 3: model-family choice follows a predeclared eligibility, Pareto, and deployment-ordering process rather than a weighted score.

**Evidence:** figure_04_pareto_selection

**Drafting rule:** State that pairwise p-values were not used as a selection gate.

### I04 — contribution

Contribution 4: post-selection robustness and deterministic release integrity are checked without changing the selected model.

**Evidence:** table_06_robustness_results, table_07_reproducibility_release

**Drafting rule:** Do not imply that the release contains raw or processed datasets.

## III. Methodology and Evaluation Protocol

### III01 — workflow

The locked workflow proceeds from strict LODO generalization through compression, CPU benchmarking, multi-objective selection, artifact locking, deterministic release, and robustness evaluation.

**Evidence:** figure_01_methodology_overview

**Drafting rule:** Preserve experimental ordering exactly.

### III02 — protocol

The protocol uses nine held-out devices, three LODO model families, four classical baselines, two neural architectures, eleven compression variants, and five locked seeds for the compression matrix.

**Evidence:** table_01_dataset_protocol

**Drafting rule:** Use exact locked values only.

## IV-A. Strict Leave-One-Device-Out Generalization

### IVA01 — result

The complete strict LODO matrix contains 27 device–model evaluations, with no severe LODO cells under the locked fragility thresholds.

**Evidence:** figure_02_lodo_generalization

**Drafting rule:** Mention the two-class held-out-device caveat where relevant.

## IV-B. Classical Baseline Performance

### IVB01 — result

HistGradientBoosting has the highest descriptive mean Macro-F1 among the four classical baselines.

**Evidence:** table_02_classical_baselines

**Drafting rule:** Explicitly state that this ranking is not final model selection.

## IV-C. Compression Accuracy Trade-offs

### IVC01 — result

The compression study contains 22 five-seed architecture–variant groups and exposes distinct accuracy–size trade-offs.

**Evidence:** figure_03_compression_tradeoff, table_03_compression_results

**Drafting rule:** Keep Phase 5 descriptive; do not present a final selection here.

### IVC02 — fragility

Three Phase 5 groups triggered fragility in at least one run: TinyML-MLP P50-noFT, TinyML-MLP DQ, and Compact-DNN P50-noFT.

**Evidence:** table_03_compression_results

**Drafting rule:** Do not generalize this fragility label beyond the locked gate.

## IV-D. CPU Deployment Benchmarks

### IVD01 — deployment

Deployment measurements use isolated single-thread CPU runs and report batch-1 latency, batch-32 throughput, state size, and peak inference RSS.

**Evidence:** table_04_deployment_benchmarks

**Drafting rule:** State that validation and test data were not accessed by the benchmark.

## IV-E. Statistical Analysis

### IVE01 — inference

All 16 omnibus tests reject at the nominal level, while none of the 160 exact paired post-hoc comparisons reject after Holm correction.

**Evidence:** table_05_statistical_analysis

**Drafting rule:** Do not interpret zero post-hoc rejections as equivalence.

### IVE02 — limitation

With five paired seeds, the minimum attainable two-sided exact p-value is 0.0625, limiting post-hoc inferential resolution.

**Evidence:** table_05_statistical_analysis

**Drafting rule:** Frame as a small-sample limitation.

## IV-F. Multi-objective Model-Family Selection

### IVF01 — selection

The balanced profile evaluates 18 compressed candidates, retains 11 eligible candidates and five Pareto candidates, and selects TinyML-MLP P50-QAT.

**Evidence:** figure_04_pareto_selection

**Drafting rule:** Preserve the predeclared ordering rule and profile distinction.

## IV-G. Input Robustness

### IVG01 — robustness

The robustness stage contains 27 evaluations across three models and nine input conditions; five severe-fragility evaluations all occur for HGB.

**Evidence:** figure_05_input_robustness, table_06_robustness_results

**Drafting rule:** Report model-specific worst conditions exactly from the locked table.

## V. Reproducibility and Release Integrity

### V01 — reproducibility

The locked deployment family is TinyML-MLP P50-QAT with static-int8 representation and canonical seed 2026.

**Evidence:** table_07_reproducibility_release

**Drafting rule:** Use exact model-family label and seed.

### V02 — reproducibility

Source/bundle outputs and serialization round-trip are exact, and the release rebuild is deterministic with locked SHA-256 identifiers.

**Evidence:** table_07_reproducibility_release

**Drafting rule:** Copy hashes exactly from the locked table; never retype from memory.

## VI. Discussion

### VI01 — interpretation

The combined evidence supports a trade-off interpretation rather than a single-metric winner: generalization, accuracy, model size, latency, throughput, inferential uncertainty, and perturbation sensitivity must be interpreted jointly.

**Evidence:** figure_02_lodo_generalization, figure_03_compression_tradeoff, table_04_deployment_benchmarks, table_05_statistical_analysis, figure_04_pareto_selection, table_06_robustness_results

**Drafting rule:** Do not introduce new quantitative claims in Discussion.

## VII. Limitations and Threats to Validity

### VII01 — limitation

Primary limitations include N-BaIoT-only primary evidence, five-seed inferential resolution, hardware-specific CPU benchmarking, two-class held-out devices, and deferred optional external-dataset validation.

**Evidence:** table_01_dataset_protocol, table_04_deployment_benchmarks, table_05_statistical_analysis

**Drafting rule:** State clearly that Phase 12 was not completed.

## VIII. Conclusion

### VIII01 — conclusion

The conclusion may summarize the locked device-generalization, compression/deployment, selection, robustness, and reproducibility findings without adding any new metric or experiment.

**Evidence:** figure_02_lodo_generalization, figure_04_pareto_selection, table_06_robustness_results, table_07_reproducibility_release

**Drafting rule:** No new numerical claim is allowed.

## Global drafting rules

- Every quantitative manuscript statement must trace to a locked Phase 13 asset.
- Do not recompute metrics during drafting.
- Do not convert descriptive ranking into inferential superiority.
- Do not interpret non-significant Holm post-hoc results as equivalence.
- Do not use robustness outcomes to retroactively change model-family selection.
- Do not claim external-dataset validation was completed.
- Use exact locked hashes only in the reproducibility section.
- External literature claims in Introduction/Related Work require separate citations and are outside this locked internal-evidence plan.
