# Early LODO Global Completion

- Generated: `2026-08-05T22:50:58.840382+00:00`
- Devices: `9/9`
- Locked scientific runs: `27/27`
- Gate status: `PASS_NO_SEVERE_FRAGILITY`
- Local severe cells: `0`

## Model ranking

Ranking uses the unweighted mean test fingerprint Macro-F1 across the nine held-out devices.

| Rank | Model | Mean Macro-F1 | Minimum Macro-F1 | Max attack FNR | Mean seconds | Device wins |
|---:|---|---:|---:|---:|---:|---:|
| 1 | compact_dnn_b0 | 0.986932236 | 0.962962680 | 0.095541864 | 75.034 | 5 |
| 2 | tinyml_mlp_b0 | 0.986283143 | 0.962982630 | 0.065431319 | 34.797 | 1 |
| 3 | hist_gradient_boosting_b0 | 0.967404248 | 0.903349652 | 0.319813714 | 37.929 | 3 |

## Interpretation note

Ennio and Samsung held-out tests contain no Mirai. Their Macro-F1 values are computed over the classes present. The global ranking must be read with this coverage limitation.

Test data were not used for model selection.
