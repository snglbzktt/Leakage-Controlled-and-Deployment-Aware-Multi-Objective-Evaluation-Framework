| Section | Reproducibility item | Locked value | Integrity / interpretation |
|---|---|---|---|
| Model-family selection | Primary selection profile | balanced | Predeclared balanced profile. |
| Model-family selection | Selected model family | tinyml_mlp::P50-QAT | static_int8 representation. |
| Model-family selection | Sensitivity-profile selections | strict=tinyml_mlp::P50-FP32-FT; balanced=tinyml_mlp::P50-QAT; relaxed=tinyml_mlp::P50-QAT | Stable across profiles: False. |
| Model-family selection | Weighted score used | No | Selection used eligibility, Pareto filtering, and deployment ordering. |
| Model-family selection | Pairwise p-value used as gate | No | Inferential p-values were not used as a model-selection gate. |
| Final artifact | Canonical checkpoint seed | 2026 | Concrete deployment checkpoint chosen after family lock. |
| Final artifact | Output class order | benign, gafgyt, mirai | Locked inference-contract output index order. |
| Final artifact | Checkpoint SHA-256 | 9a2d2c0dc469777e82a2daba5bd1bbad1dc5bd77fa46e0304f94752eefbfef8a | Source and bundled checkpoint hashes are identical. |
| Final artifact | Source/bundle output equality | Exact | Locked verification reports exact source/bundle outputs. |
| Final artifact | Serialization round-trip | Exact | Serialization/deserialization round-trip reproduced exactly. |
| Release archive | Release name | TinyML_IDS_IEEE_release_v3_3 | 15 payload files; 16 archive files including manifest. |
| Release archive | Release tree SHA-256 | 6348e0329c6f926863fadf5ce6be2d8a7f27cad927b0dfe8165811e538faa255 | Hash of the deterministic release-directory tree. |
| Release archive | ZIP SHA-256 | b3decdf91a16b0716aa04dce8f406edd0f7fad117216d7822e00f554a02950b5 | ZIP size: 33787 bytes. |
| Release archive | Deterministic rebuild | Exact | Independent rebuild reproduced the locked archive exactly. |
| Release archive | Datasets in release | Raw: No; Processed: No | Release excludes raw and processed datasets. |
| Release archive | Nonselected checkpoints in release | No | Release contains no nonselected checkpoints. |
| Post-selection robustness | Model retraining performed | No | Robustness analysis did not retrain any model. |
| Post-selection robustness | Test used for model selection | No | Robustness test results were not fed back into selection. |
| Post-selection robustness | Final model changed | No | The locked deployment model remained unchanged. |
