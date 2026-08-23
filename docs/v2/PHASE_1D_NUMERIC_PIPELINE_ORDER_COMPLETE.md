# Phase 1D — Numeric Pipeline Order Audit Completed

- Completion time: 2026-08-05T16:56:39.341403+00:00
- Source-precision rows: 2,482,676
- Feature count: 115
- Different-row rate: 96.912726%
- Different-element rate: 13.458617%
- Mean absolute difference: 1.08103991133e-08
- Maximum absolute difference: 7.62939453125e-06
- Training pilot required by protocol: true
- All validation checks passed: true

## Compared pipelines

### Pipeline A

Source float64 → float32 → train-only StandardScaler with
float64 parameters → final float32 model input.

### Pipeline B

Source float64 → train-only StandardScaler in float64 →
final float32 model input.

Both pipelines were evaluated on the same source-precision
primary train, validation, and test splits.

## Split-level results

| Split | Rows | Different rows % | A unique | B unique | Delta |
|---|---:|---:|---:|---:|---:|
| train | 1,738,133 | 96.903804 | 1,594,075 | 1,594,082 | 7 |
| validation | 371,884 | 96.904411 | 341,080 | 341,090 | 10 |
| test | 372,659 | 96.962639 | 342,040 | 342,048 | 8 |

## Final float32 overlap results

| Split pair | Pipeline A | Pipeline B | Difference |
|---|---:|---:|---:|
| train–validation | 240 | 246 | 6 |
| train–test | 232 | 238 | 6 |
| validation–test | 92 | 98 | 6 |

## Interpretation

The two processing orders are not bitwise equivalent.

Most rows contain at least one changed float32 value, but the
numerical magnitude of these changes is small. The observed mean
and maximum absolute differences remain below the predetermined
numerical-effect thresholds.

The number of unique final model inputs changes only slightly and
remains below the predetermined relative unique-count threshold.

However, final float32 overlap counts differ for all three split
pairs. Therefore, the predetermined protocol requires the
eight-condition training pilot before selecting the final numeric
processing order.

## Pilot decision reasons

- train-validation: splitler arası final float32 örtüşme sayısı iki hat arasında farklıdır.
- train-test: splitler arası final float32 örtüşme sayısı iki hat arasında farklıdır.
- validation-test: splitler arası final float32 örtüşme sayısı iki hat arasında farklıdır.

## Important limitation

This audit used the source-precision primary splits as a common
comparison universe. Historical main experiments used the separate
float32-canonical split. Therefore, this audit diagnoses processing
order and does not replace the historical split analysis.

## Artifacts

- `results\v2\audit\numeric_pipeline_order_summary_v2.json`
- `results\v2\audit\numeric_pipeline_order_split_metrics_v2.csv`
- `results\v2\audit\numeric_pipeline_order_overlap_v2.csv`
- `results\v2\audit\numeric_pipeline_order_feature_stats_v2.csv`
- `results\v2\audit\numeric_pipeline_order_release_manifest_v2.csv`
