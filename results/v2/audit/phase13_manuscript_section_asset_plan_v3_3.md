# Phase 13 Manuscript Section and Asset Plan

## Main sections

### I. Introduction

Define the IoT intrusion-detection problem, the TinyML deployment constraint, the research gap, and the paper contributions.

**Primary assets:** None

**Writing rule:** No experimental result should be introduced here beyond high-level motivation and contribution statements.

### II. Related Work

Position the work against IoT IDS, N-BaIoT evaluation, TinyML, compression/quantization, device-generalization, and robustness studies.

**Primary assets:** None

**Writing rule:** External literature belongs here; locked project results must not be used as substitutes for citations.

### III. Methodology and Evaluation Protocol

Describe the locked end-to-end workflow, N-BaIoT study protocol, strict LODO design, compression matrix, and test-isolation rules.

**Primary assets:** figure_01_methodology_overview, table_01_dataset_protocol

**Writing rule:** Report only protocol facts supported by the locked Phase 3–6 evidence. Do not claim final model superiority in this section.

### IV. Experimental Results

Present generalization, baseline, compression, deployment, statistical, selection, and robustness results in the locked experimental order.

**Primary assets:** figure_02_lodo_generalization, table_02_classical_baselines, figure_03_compression_tradeoff, table_03_compression_results, table_04_deployment_benchmarks, table_05_statistical_analysis, figure_04_pareto_selection, figure_05_input_robustness, table_06_robustness_results

**Writing rule:** Preserve the distinction between descriptive rankings, inferential results, predeclared selection, and post-selection robustness.

### V. Reproducibility and Release Integrity

Document the selected family, canonical seed, artifact checks, deterministic release hashes, and release-content exclusions.

**Primary assets:** table_07_reproducibility_release

**Writing rule:** Use exact locked hashes and integrity statements. Do not present the release bundle as containing raw or processed datasets.

### VI. Discussion

Interpret the combined evidence: strong strict-LODO generalization, compression/deployment trade-offs, statistical limitations, selected P50-QAT family, and perturbation sensitivity.

**Primary assets:** None

**Writing rule:** Interpret rather than repeat tables. State that zero Holm post-hoc rejections do not establish equivalence.

### VII. Limitations and Threats to Validity

State dataset scope, five-seed inferential resolution, hardware-specific CPU benchmarking, two-class held-out devices, and the deferred optional external-dataset validation.

**Primary assets:** None

**Writing rule:** Phase 12 must be described as deferred/optional and must not be implied to have been completed.

### VIII. Conclusion

Summarize the supported contribution without adding new metrics, experiments, or claims.

**Primary assets:** None

**Writing rule:** No new numerical claim may appear here unless already established in the locked evidence.

## Experimental Results subsections

### IV-A — Strict Leave-One-Device-Out Generalization

**Assets:** figure_02_lodo_generalization

Report the 27-run 9-device × 3-model LODO matrix and the absence of severe LODO cells under the locked thresholds.

### IV-B — Classical Baseline Performance

**Assets:** table_02_classical_baselines

Present the four classical baselines as descriptive comparisons only; the ranking is not final model selection.

### IV-C — Compression Accuracy Trade-offs

**Assets:** figure_03_compression_tradeoff, table_03_compression_results

Present all 22 five-seed architecture–variant groups and identify the three Phase 5 fragility groups without treating Phase 5 as selection.

### IV-D — CPU Deployment Benchmarks

**Assets:** table_04_deployment_benchmarks

Report isolated one-thread CPU latency, throughput, state size, and memory measurements without validation/test-data access.

### IV-E — Statistical Analysis

**Assets:** table_05_statistical_analysis

Report 16 nominal omnibus rejections, zero Holm-corrected post-hoc rejections across 160 comparisons, and the n=5 exact-p limitation.

### IV-F — Multi-objective Model-Family Selection

**Assets:** figure_04_pareto_selection

Explain the predeclared eligibility → Pareto → deployment ordering. Balanced/relaxed select TinyML-MLP P50-QAT; strict sensitivity selects TinyML-MLP P50-FP32-FT.

### IV-G — Input Robustness

**Assets:** figure_05_input_robustness, table_06_robustness_results

Report 27 locked evaluations, five severe-fragility evaluations all for HGB, and nine material-sensitivity evaluations; no retraining or reselection occurred.

## Locked writing guardrails

- Do not use test results for retroactive model selection.
- Do not describe Phase 4 classical-baseline ranking as final model selection.
- Do not describe Phase 5 compression results as final model selection.
- Do not interpret zero Holm-corrected post-hoc rejections as model equivalence.
- Do not claim external-dataset validation was completed; Phase 12 is deferred/optional.
- Do not claim robustness experiments retrained or changed the selected model.
- Do not state that the release archive contains raw data, processed data, or nonselected checkpoints.
- Use the exact locked model-family label tinyml_mlp::P50-QAT and canonical seed 2026 when reproducibility details are required.
- Use locked tables/figures as the source of manuscript numbers rather than recomputing metrics.
