# TinyML IDS IEEE Release v3.3

## Locked deployment artifact

- Model family: `tinyml_mlp::P50-QAT`
- Representation: `static_int8`
- Canonical deployment checkpoint seed: `2026`
- Input shape: `[batch, 115]`
- Input dtype: `float32`
- Preprocessing: locked train-only StandardScaler
- Output shape: `[batch, 3]`
- Output semantics: logits
- Class order: `0=benign`, `1=gafgyt`, `2=mirai`
- Runtime target: CPU
- Quantization backend recorded by the artifact: `onednn`

The model family was selected from aggregate five-seed evidence. The concrete deployment checkpoint was fixed to the canonical project seed 2026 and was not selected by ranking test, validation, or deployment metrics.

## Directory layout

- `deployment_artifact/`: locked model checkpoint, source modules, scaler, inference contract, artifact manifest, and artifact lock.
- `audit/`: minimum locked evidence for model-family and deployment-artifact provenance.
- `release_manifest.json`: SHA-256 and byte size for every payload file except the manifest itself.
- `DATA_NOT_INCLUDED.md`: dataset redistribution statement.

## Integrity

Verify each payload file against `release_manifest.json`. The final ZIP SHA-256 and release-directory tree hash are stored in the external Phase 10 release lock after independent verification.

## Scientific scope

This bundle contains the locked deployment artifact only. It does not replace the full experimental repository, raw data, processed split arrays, or nonselected checkpoints.
