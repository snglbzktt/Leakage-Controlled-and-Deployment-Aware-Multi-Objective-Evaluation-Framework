# Phase 1C — Task-Aware and Device-Overlap Audit Completed

- Completion time: 2026-08-05T15:08:23.280160+00:00
- Source-precision exact groups: 2,482,676
- Raw records: 7,062,606
- Devices: 9
- Single-device exact groups: 611,623 (24.635635%)
- Multi-device exact groups: 1,871,053 (75.364365%)
- All validation checks passed: true

## Task-aware source-precision conflicts

- binary: 0 conflict groups
- family_3: 0 conflict groups
- multiclass_11: 0 conflict groups

## Device-family coverage

- Devices missing Benign: none
- Devices missing Gafgyt: none
- Devices missing Mirai: Ennio_Doorbell, Samsung_SNH_1011_N_Webcam

For held-out devices without Mirai support, Mirai FNR must be
reported as N/A rather than zero. Macro metrics must include an
explicit class-coverage field.

## LODO feasibility

| ID | Device | Novel FP % | Novel raw % | Train-exposed FP % | Missing family |
|---|---|---:|---:|---:|---|
| D01 | Danmini_Doorbell | 8.518884 | 9.236294 | 91.481116 | none |
| D02 | Ecobee_Thermostat | 2.686804 | 2.646086 | 97.313196 | none |
| D03 | Ennio_Doorbell | 10.575046 | 11.114205 | 89.424954 | mirai |
| D04 | Philips_B120N10_Baby_Monitor | 15.551947 | 16.149241 | 84.448053 | none |
| D05 | Provision_PT_737E_Security_Camera | 8.700157 | 9.328955 | 91.299843 | none |
| D06 | Provision_PT_838_Security_Camera | 12.347416 | 12.638802 | 87.652584 | none |
| D07 | Samsung_SNH_1011_N_Webcam | 14.305948 | 14.417865 | 85.694052 | mirai |
| D08 | SimpleHome_XCS7_1002_WHT_Security_Camera | 5.454479 | 5.793714 | 94.545521 | none |
| D09 | SimpleHome_XCS7_1003_WHT_Security_Camera | 3.865359 | 3.988242 | 96.134641 | none |

## Required LODO reporting

Every held-out-device evaluation must report:

1. Record-weighted performance on all held-out records.
2. Fingerprint-balanced performance.
3. Performance on the novel-fingerprint subset.
4. Class coverage and N/A metrics for absent classes.
5. Train-exposed fingerprint and occurrence rates.

## Interpretation

Most source-precision exact groups occur on more than one device.
Therefore, holding out a device does not by itself guarantee that
all test feature vectors are unseen. The separate novel-fingerprint
evaluation is mandatory.

## Artifacts

- `results\v2\audit\device_lodo_compact_summary_v2.csv`
- `results\v2\audit\device_family_support_v2.csv`
- `results\v2\audit\top_device_pair_overlap_v2.csv`
- `results\v2\audit\task_aware_device_audit_summary_v2.json`
- `results\v2\audit\task_aware_device_audit_release_manifest_v2.csv`
