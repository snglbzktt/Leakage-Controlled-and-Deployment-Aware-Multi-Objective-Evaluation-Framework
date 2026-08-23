from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()

AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

DOCS_DIRECTORY = (
    PROJECT_ROOT
    / "docs"
    / "v2"
)

SUMMARY_JSON = (
    AUDIT_DIRECTORY
    / "numeric_pipeline_order_summary_v2.json"
)

SPLIT_METRICS_CSV = (
    AUDIT_DIRECTORY
    / "numeric_pipeline_order_split_metrics_v2.csv"
)

OVERLAP_CSV = (
    AUDIT_DIRECTORY
    / "numeric_pipeline_order_overlap_v2.csv"
)

FEATURE_STATS_CSV = (
    AUDIT_DIRECTORY
    / "numeric_pipeline_order_feature_stats_v2.csv"
)

RUN_LOG = (
    AUDIT_DIRECTORY
    / "numeric_pipeline_order_run_v2.log"
)

MANIFEST_CSV = (
    AUDIT_DIRECTORY
    / "numeric_pipeline_order_release_manifest_v2.csv"
)

COMPLETION_MD = (
    DOCS_DIRECTORY
    / "PHASE_1D_NUMERIC_PIPELINE_ORDER_COMPLETE.md"
)


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                chunk_size
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


required_files = [
    SUMMARY_JSON,
    SPLIT_METRICS_CSV,
    OVERLAP_CSV,
    FEATURE_STATS_CSV,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Eksik Faz 1D dosyaları:")

    for path in missing_files:
        print(f"- {path}")

    sys.exit(1)


summary = json.loads(
    SUMMARY_JSON.read_text(
        encoding="utf-8",
    )
)

split_rows = read_csv(
    SPLIT_METRICS_CSV
)

overlap_rows = read_csv(
    OVERLAP_CSV
)

feature_rows = read_csv(
    FEATURE_STATS_CSV
)


expected_splits = {
    "train": 1_738_133,
    "validation": 371_884,
    "test": 372_659,
}

observed_splits = {
    row["split"]:
        int(row["row_count"])
    for row in split_rows
}


validation_checks = {
    "status_completed":
        summary.get("status")
        == "completed",

    "audit_checks_passed":
        summary.get(
            "all_checks_passed"
        )
        is True,

    "training_pilot_required":
        summary.get(
            "requires_training_pilot"
        )
        is True,

    "global_row_count":
        int(
            summary.get(
                "global_row_count",
                -1,
            )
        )
        == 2_482_676,

    "split_rows":
        observed_splits
        == expected_splits,

    "feature_count":
        len(feature_rows) == 115,

    "split_metric_count":
        len(split_rows) == 3,

    "overlap_comparison_count":
        len(overlap_rows) == 3,

    "all_overlap_counts_changed":
        all(
            int(
                row[
                    "pipeline_a_overlap_count"
                ]
            )
            !=
            int(
                row[
                    "pipeline_b_overlap_count"
                ]
            )
            for row in overlap_rows
        ),

    "mean_difference_below_numeric_threshold":
        float(
            summary[
                "global_mean_absolute_difference"
            ]
        )
        <
        float(
            summary[
                "thresholds"
            ][
                "mean_absolute_difference_threshold"
            ]
        ),

    "maximum_difference_below_numeric_threshold":
        float(
            summary[
                "global_maximum_absolute_difference"
            ]
        )
        <
        float(
            summary[
                "thresholds"
            ][
                "maximum_absolute_difference_threshold"
            ]
        ),
}


failed_checks = [
    name
    for name, passed
    in validation_checks.items()
    if not passed
]

if failed_checks:
    print("Faz 1D kilitlenemedi.")
    print("Başarısız kontroller:")

    for check in failed_checks:
        print(f"- {check}")

    sys.exit(1)


split_table_lines = [
    "| Split | Rows | Different rows % | A unique | B unique | Delta |",
    "|---|---:|---:|---:|---:|---:|",
]

for row in split_rows:
    split_table_lines.append(
        f"| {row['split']} "
        f"| {int(row['row_count']):,} "
        f"| {100 * float(row['different_row_rate']):.6f} "
        f"| {int(row['pipeline_a_unique_input_count']):,} "
        f"| {int(row['pipeline_b_unique_input_count']):,} "
        f"| {int(row['unique_count_difference_b_minus_a']):,} |"
    )


overlap_table_lines = [
    "| Split pair | Pipeline A | Pipeline B | Difference |",
    "|---|---:|---:|---:|",
]

for row in overlap_rows:
    overlap_table_lines.append(
        f"| {row['split_a']}–{row['split_b']} "
        f"| {int(row['pipeline_a_overlap_count']):,} "
        f"| {int(row['pipeline_b_overlap_count']):,} "
        f"| {int(row['overlap_difference_b_minus_a']):,} |"
    )


decision_reasons = summary.get(
    "decision_reasons",
    []
)

decision_lines = [
    f"- {reason}"
    for reason in decision_reasons
]


completion_text = f"""# Phase 1D — Numeric Pipeline Order Audit Completed

- Completion time: {datetime.now(timezone.utc).isoformat()}
- Source-precision rows: {int(summary["global_row_count"]):,}
- Feature count: 115
- Different-row rate: {100 * float(summary["global_different_row_rate"]):.6f}%
- Different-element rate: {100 * float(summary["global_different_element_rate"]):.6f}%
- Mean absolute difference: {float(summary["global_mean_absolute_difference"]):.12g}
- Maximum absolute difference: {float(summary["global_maximum_absolute_difference"]):.12g}
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

{chr(10).join(split_table_lines)}

## Final float32 overlap results

{chr(10).join(overlap_table_lines)}

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

{chr(10).join(decision_lines)}

## Important limitation

This audit used the source-precision primary splits as a common
comparison universe. Historical main experiments used the separate
float32-canonical split. Therefore, this audit diagnoses processing
order and does not replace the historical split analysis.

## Artifacts

- `{SUMMARY_JSON.relative_to(PROJECT_ROOT)}`
- `{SPLIT_METRICS_CSV.relative_to(PROJECT_ROOT)}`
- `{OVERLAP_CSV.relative_to(PROJECT_ROOT)}`
- `{FEATURE_STATS_CSV.relative_to(PROJECT_ROOT)}`
- `{MANIFEST_CSV.relative_to(PROJECT_ROOT)}`
"""


DOCS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

COMPLETION_MD.write_text(
    completion_text,
    encoding="utf-8",
)


manifest_files = [
    SUMMARY_JSON,
    SPLIT_METRICS_CSV,
    OVERLAP_CSV,
    FEATURE_STATS_CSV,
    COMPLETION_MD,
]

if RUN_LOG.exists():
    manifest_files.append(
        RUN_LOG
    )


manifest_rows = []

print("SHA-256 manifesti oluşturuluyor...")

for path in manifest_files:
    print(f"  hash: {path.name}")

    manifest_rows.append(
        {
            "relative_path":
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "size_bytes":
                path.stat().st_size,

            "sha256":
                sha256_file(path),
        }
    )


write_csv(
    MANIFEST_CSV,
    manifest_rows,
    [
        "relative_path",
        "size_bytes",
        "sha256",
    ],
)


print()
print("=" * 78)
print("FAZ 1D KİLİT ÖZETİ")
print("=" * 78)

print(
    "Farklı satır oranı       : "
    f"{100 * float(summary['global_different_row_rate']):.6f}%"
)

print(
    "Ortalama mutlak fark     : "
    f"{float(summary['global_mean_absolute_difference']):.12g}"
)

print(
    "Maksimum mutlak fark     : "
    f"{float(summary['global_maximum_absolute_difference']):.12g}"
)

print(
    "8 koşuluk pilot gerekli  : "
    f"{summary['requires_training_pilot']}"
)

print()
print("SPLIT UNIQUE DELTALARI")

for row in split_rows:
    print(
        f"{row['split']}: "
        f"{int(row['unique_count_difference_b_minus_a']):+,}"
    )

print()
print("SPLIT ÖRTÜŞME DELTALARI")

for row in overlap_rows:
    print(
        f"{row['split_a']}-"
        f"{row['split_b']}: "
        f"{int(row['overlap_difference_b_minus_a']):+,}"
    )

print()
print("VALIDATION CHECKS")

for name, passed in validation_checks.items():
    print(f"{name}: {passed}")

print()
print(f"Manifest  : {MANIFEST_CSV}")
print(f"Tamamlama : {COMPLETION_MD}")
print()
print("FAZ 1D KİLİTLENDİ")
