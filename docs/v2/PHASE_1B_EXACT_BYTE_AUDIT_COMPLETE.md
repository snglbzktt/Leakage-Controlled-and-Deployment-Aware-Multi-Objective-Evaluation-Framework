# Phase 1B — Exact Byte Audit Completed

- Completion time: 2026-08-05T14:53:53.192783+00:00
- Raw records: 7,062,606
- Source-precision exact vectors: 2,482,676
- Feature count: 115
- Canonical bytes per vector: 920
- Byte-conflict groups: 0
- SHA-256 conflict groups: 0
- Maximum occurrence count: 36
- All validation checks passed: true

## Interpretation

The original pair of 64-bit vector hashes was independently
validated against canonical little-endian float64 byte sequences
and SHA-256 digests.

No case was found where records assigned to the same original
hash pair contained different canonical bytes.

This result validates source-precision exact-vector grouping.
It does not replace the separate float32 model-input grouping
analysis.

## Artifacts

- `results\v2\audit\nbaiot_exact_byte_audit_v2.sqlite`
- `results\v2\audit\nbaiot_exact_byte_audit_summary_v2.json`
- `results\v2\audit\nbaiot_exact_byte_conflicts_v2.csv`
- `results\v2\audit\exact_byte_audit_release_manifest_v2.csv`
