| Endpoint | Architecture | Metric | Blocks | Variants | Friedman statistic | Kendall W | Monte Carlo p | Omnibus reject (nominal) | Post-hoc comparisons vs B0 | Min exact post-hoc p | Min Holm-adjusted p | Holm rejections |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| confirmatory | tinyml_mlp | test_fingerprint_macro_f1 | 5 | 11 | 46.327273 | 0.926545 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| confirmatory | tinyml_mlp | batch_1_median_ms | 5 | 11 | 39.600000 | 0.792000 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| confirmatory | tinyml_mlp | batch_32_samples_per_second | 5 | 11 | 45.745455 | 0.914909 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| confirmatory | tinyml_mlp | serialized_state_dict_bytes | 5 | 11 | 49.090909 | 0.981818 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| exploratory | tinyml_mlp | test_raw_weighted_macro_f1 | 5 | 11 | 45.527273 | 0.910545 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| exploratory | tinyml_mlp | batch_1_p95_ms | 5 | 11 | 39.636364 | 0.792727 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| exploratory | tinyml_mlp | batch_32_p95_ms | 5 | 11 | 43.781818 | 0.875636 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| exploratory | tinyml_mlp | peak_inference_RSS_delta_bytes | 5 | 11 | 46.481818 | 0.929636 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| confirmatory | compact_dnn | test_fingerprint_macro_f1 | 5 | 11 | 47.490909 | 0.949818 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| confirmatory | compact_dnn | batch_1_median_ms | 5 | 11 | 40.836364 | 0.816727 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| confirmatory | compact_dnn | batch_32_samples_per_second | 5 | 11 | 45.454545 | 0.909091 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| confirmatory | compact_dnn | serialized_state_dict_bytes | 5 | 11 | 49.090909 | 0.981818 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| exploratory | compact_dnn | test_raw_weighted_macro_f1 | 5 | 11 | 47.781818 | 0.955636 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| exploratory | compact_dnn | batch_1_p95_ms | 5 | 11 | 43.236364 | 0.864727 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| exploratory | compact_dnn | batch_32_p95_ms | 5 | 11 | 42.545455 | 0.850909 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
| exploratory | compact_dnn | peak_inference_RSS_delta_bytes | 5 | 11 | 46.136364 | 0.922727 | 0.000010 | Yes | 10 | 0.0625 | 0.6250 | 0 |
