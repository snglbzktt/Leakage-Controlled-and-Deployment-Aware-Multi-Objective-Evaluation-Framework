from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()

AUDIT_DIR = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

DOCS_DIR = (
    PROJECT_ROOT
    / "docs"
    / "v2"
)

SUMMARY_JSON = (
    AUDIT_DIR
    / "task_aware_device_audit_summary_v2.json"
)

TASK_SUMMARY_CSV = (
    AUDIT_DIR
    / "task_aware_conflict_summary_v2.csv"
)

TASK_DETAILS_CSV = (
    AUDIT_DIR
    / "task_aware_conflict_details_v2.csv"
)

LODO_CSV = (
    AUDIT_DIR
    / "device_fingerprint_lodo_feasibility_v2.csv"
)

CLASS_SUPPORT_CSV = (
    AUDIT_DIR
    / "device_class_support_v2.csv"
)

PAIR_CSV = (
    AUDIT_DIR
    / "device_pair_fingerprint_overlap_v2.csv"
)

DEVICE_COUNT_CSV = (
    AUDIT_DIR
    / "device_count_distribution_v2.csv"
)

LOG_FILE = (
    AUDIT_DIR
    / "task_aware_device_audit_run_v2.log"
)

COMPACT_LODO_CSV = (
    AUDIT_DIR
    / "device_lodo_compact_summary_v2.csv"
)

FAMILY_SUPPORT_CSV = (
    AUDIT_DIR
    / "device_family_support_v2.csv"
)

TOP_PAIR_CSV = (
    AUDIT_DIR
    / "top_device_pair_overlap_v2.csv"
)

MANIFEST_CSV = (
    AUDIT_DIR
    / "task_aware_device_audit_release_manifest_v2.csv"
)

COMPLETION_MD = (
    DOCS_DIR
    / "PHASE_1C_TASK_DEVICE_AUDIT_COMPLETE.md"
)


def read_csv_rows(
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


def write_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
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
            block = handle.read(chunk_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def as_int(value: str | int) -> int:
    return int(value)


def as_float(value: str | float) -> float:
    return float(value)


def percent(value: float) -> float:
    return round(
        100.0 * value,
        6,
    )


def family_from_label(
    label: str,
) -> str:
    normalized = (
        label
        .strip()
        .casefold()
    )

    if normalized == "benign":
        return "benign"

    if normalized.startswith("gafgyt_"):
        return "gafgyt"

    if normalized.startswith("mirai_"):
        return "mirai"

    raise ValueError(
        f"Unknown class label: {label}"
    )


required_files = [
    SUMMARY_JSON,
    TASK_SUMMARY_CSV,
    TASK_DETAILS_CSV,
    LODO_CSV,
    CLASS_SUPPORT_CSV,
    PAIR_CSV,
    DEVICE_COUNT_CSV,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Missing required files:")

    for path in missing_files:
        print(f"- {path}")

    sys.exit(1)

summary = json.loads(
    SUMMARY_JSON.read_text(
        encoding="utf-8",
    )
)

task_rows = read_csv_rows(
    TASK_SUMMARY_CSV
)

lodo_rows = read_csv_rows(
    LODO_CSV
)

class_rows = read_csv_rows(
    CLASS_SUPPORT_CSV
)

pair_rows = read_csv_rows(
    PAIR_CSV
)

validation_checks: dict[str, bool] = {
    "phase_1c_all_checks_passed":
        summary.get("all_checks_passed") is True,

    "source_group_count":
        summary.get("source_group_count")
        == 2_482_676,

    "source_occurrence_count":
        summary.get("source_occurrence_count")
        == 7_062_606,

    "device_count":
        summary.get("distinct_device_count")
        == 9,

    "task_count":
        len(task_rows) == 3,

    "all_source_precision_conflicts_zero":
        all(
            as_int(
                row["conflict_group_count"]
            ) == 0
            for row in task_rows
        ),

    "lodo_device_count":
        len(lodo_rows) == 9,
}

compact_rows: list[dict[str, Any]] = []

sorted_lodo_rows = sorted(
    lodo_rows,
    key=lambda row: row["device"],
)

device_aliases = {
    row["device"]: f"D{index:02d}"
    for index, row in enumerate(
        sorted_lodo_rows,
        start=1,
    )
}

for row in sorted_lodo_rows:
    fingerprint_count = as_int(
        row["fingerprint_count"]
    )

    raw_count = as_int(
        row["raw_occurrence_count"]
    )

    novel_fingerprint_count = as_int(
        row["novel_fingerprint_count"]
    )

    novel_raw_count = as_int(
        row["novel_occurrence_count"]
    )

    exposed_fingerprint_count = as_int(
        row[
            "train_exposed_fingerprint_count"
        ]
    )

    exposed_raw_count = as_int(
        row[
            "train_exposed_occurrence_count"
        ]
    )

    validation_checks[
        f"{device_aliases[row['device']]}_fingerprint_partition"
    ] = (
        fingerprint_count
        == novel_fingerprint_count
        + exposed_fingerprint_count
    )

    validation_checks[
        f"{device_aliases[row['device']]}_raw_partition"
    ] = (
        raw_count
        == novel_raw_count
        + exposed_raw_count
    )

    compact_rows.append(
        {
            "device_id":
                device_aliases[
                    row["device"]
                ],

            "device":
                row["device"],

            "fingerprint_count":
                fingerprint_count,

            "raw_occurrence_count":
                raw_count,

            "novel_fingerprint_count":
                novel_fingerprint_count,

            "novel_fingerprint_pct":
                percent(
                    as_float(
                        row[
                            "novel_fingerprint_rate"
                        ]
                    )
                ),

            "novel_occurrence_count":
                novel_raw_count,

            "novel_occurrence_pct":
                percent(
                    as_float(
                        row[
                            "novel_occurrence_rate"
                        ]
                    )
                ),

            "train_exposed_fingerprint_count":
                exposed_fingerprint_count,

            "train_exposed_fingerprint_pct":
                percent(
                    as_float(
                        row[
                            "train_exposed_fingerprint_rate"
                        ]
                    )
                ),

            "train_exposed_occurrence_count":
                exposed_raw_count,

            "train_exposed_occurrence_pct":
                percent(
                    as_float(
                        row[
                            "train_exposed_occurrence_rate"
                        ]
                    )
                ),
        }
    )

write_csv_rows(
    COMPACT_LODO_CSV,
    compact_rows,
    [
        "device_id",
        "device",
        "fingerprint_count",
        "raw_occurrence_count",
        "novel_fingerprint_count",
        "novel_fingerprint_pct",
        "novel_occurrence_count",
        "novel_occurrence_pct",
        "train_exposed_fingerprint_count",
        "train_exposed_fingerprint_pct",
        "train_exposed_occurrence_count",
        "train_exposed_occurrence_pct",
    ],
)

family_counts: dict[
    str,
    dict[str, dict[str, int]],
] = defaultdict(
    lambda: defaultdict(
        lambda: {
            "fingerprint_count": 0,
            "raw_occurrence_count": 0,
        }
    )
)

for row in class_rows:
    device = row["device"]

    family = family_from_label(
        row["class_label"]
    )

    family_counts[
        device
    ][
        family
    ][
        "fingerprint_count"
    ] += as_int(
        row["fingerprint_count"]
    )

    family_counts[
        device
    ][
        family
    ][
        "raw_occurrence_count"
    ] += as_int(
        row["raw_occurrence_count"]
    )

family_support_rows: list[
    dict[str, Any]
] = []

devices_missing_mirai: list[str] = []
devices_missing_gafgyt: list[str] = []
devices_missing_benign: list[str] = []

for device in sorted(device_aliases):
    benign_fp = family_counts[
        device
    ][
        "benign"
    ][
        "fingerprint_count"
    ]

    gafgyt_fp = family_counts[
        device
    ][
        "gafgyt"
    ][
        "fingerprint_count"
    ]

    mirai_fp = family_counts[
        device
    ][
        "mirai"
    ][
        "fingerprint_count"
    ]

    if benign_fp == 0:
        devices_missing_benign.append(
            device
        )

    if gafgyt_fp == 0:
        devices_missing_gafgyt.append(
            device
        )

    if mirai_fp == 0:
        devices_missing_mirai.append(
            device
        )

    present_families = [
        family
        for family, count
        in [
            ("benign", benign_fp),
            ("gafgyt", gafgyt_fp),
            ("mirai", mirai_fp),
        ]
        if count > 0
    ]

    missing_families = [
        family
        for family, count
        in [
            ("benign", benign_fp),
            ("gafgyt", gafgyt_fp),
            ("mirai", mirai_fp),
        ]
        if count == 0
    ]

    family_support_rows.append(
        {
            "device_id":
                device_aliases[device],

            "device":
                device,

            "benign_fingerprint_count":
                benign_fp,

            "benign_raw_occurrence_count":
                family_counts[
                    device
                ][
                    "benign"
                ][
                    "raw_occurrence_count"
                ],

            "gafgyt_fingerprint_count":
                gafgyt_fp,

            "gafgyt_raw_occurrence_count":
                family_counts[
                    device
                ][
                    "gafgyt"
                ][
                    "raw_occurrence_count"
                ],

            "mirai_fingerprint_count":
                mirai_fp,

            "mirai_raw_occurrence_count":
                family_counts[
                    device
                ][
                    "mirai"
                ][
                    "raw_occurrence_count"
                ],

            "present_family_count":
                len(present_families),

            "present_families":
                "|".join(
                    present_families
                ),

            "missing_families":
                "|".join(
                    missing_families
                ),

            "family3_macro_f1_coverage":
                (
                    "3_of_3"
                    if len(present_families) == 3
                    else
                    f"{len(present_families)}_of_3"
                ),
        }
    )

write_csv_rows(
    FAMILY_SUPPORT_CSV,
    family_support_rows,
    [
        "device_id",
        "device",
        "benign_fingerprint_count",
        "benign_raw_occurrence_count",
        "gafgyt_fingerprint_count",
        "gafgyt_raw_occurrence_count",
        "mirai_fingerprint_count",
        "mirai_raw_occurrence_count",
        "present_family_count",
        "present_families",
        "missing_families",
        "family3_macro_f1_coverage",
    ],
)

sorted_pair_rows = sorted(
    pair_rows,
    key=lambda row: as_int(
        row[
            "shared_fingerprint_count"
        ]
    ),
    reverse=True,
)

top_pair_rows: list[
    dict[str, Any]
] = []

for rank, row in enumerate(
    sorted_pair_rows[:20],
    start=1,
):
    top_pair_rows.append(
        {
            "rank": rank,

            "device_a_id":
                device_aliases[
                    row["device_a"]
                ],

            "device_a":
                row["device_a"],

            "device_b_id":
                device_aliases[
                    row["device_b"]
                ],

            "device_b":
                row["device_b"],

            "shared_fingerprint_count":
                as_int(
                    row[
                        "shared_fingerprint_count"
                    ]
                ),

            "device_a_occurrences_on_shared":
                as_int(
                    row[
                        "device_a_occurrences_on_shared"
                    ]
                ),

            "device_b_occurrences_on_shared":
                as_int(
                    row[
                        "device_b_occurrences_on_shared"
                    ]
                ),
        }
    )

write_csv_rows(
    TOP_PAIR_CSV,
    top_pair_rows,
    [
        "rank",
        "device_a_id",
        "device_a",
        "device_b_id",
        "device_b",
        "shared_fingerprint_count",
        "device_a_occurrences_on_shared",
        "device_b_occurrences_on_shared",
    ],
)

source_group_count = as_int(
    summary["source_group_count"]
)

single_device_group_count = as_int(
    summary["single_device_group_count"]
)

multi_device_group_count = as_int(
    summary["multi_device_group_count"]
)

validation_checks[
    "single_plus_multi_equals_total"
] = (
    single_device_group_count
    + multi_device_group_count
    == source_group_count
)

validation_checks[
    "no_device_missing_benign"
] = len(
    devices_missing_benign
) == 0

validation_checks[
    "no_device_missing_gafgyt"
] = len(
    devices_missing_gafgyt
) == 0

all_checks_passed = all(
    validation_checks.values()
)

single_device_pct = percent(
    single_device_group_count
    / source_group_count
)

multi_device_pct = percent(
    multi_device_group_count
    / source_group_count
)

task_conflict_lines = []

for row in task_rows:
    task_conflict_lines.append(
        f"- {row['task']}: "
        f"{as_int(row['conflict_group_count']):,} "
        "conflict groups"
    )

device_table_lines = [
    "| ID | Device | Novel FP % | Novel raw % | Train-exposed FP % | Missing family |",
    "|---|---|---:|---:|---:|---|",
]

family_by_device = {
    row["device"]: row
    for row in family_support_rows
}

for row in compact_rows:
    family_row = family_by_device[
        row["device"]
    ]

    missing = (
        family_row["missing_families"]
        or "none"
    )

    device_table_lines.append(
        f"| {row['device_id']} "
        f"| {row['device']} "
        f"| {row['novel_fingerprint_pct']:.6f} "
        f"| {row['novel_occurrence_pct']:.6f} "
        f"| {row['train_exposed_fingerprint_pct']:.6f} "
        f"| {missing} |"
    )

completion_text = f"""# Phase 1C — Task-Aware and Device-Overlap Audit Completed

- Completion time: {datetime.now(timezone.utc).isoformat()}
- Source-precision exact groups: {source_group_count:,}
- Raw records: {as_int(summary["source_occurrence_count"]):,}
- Devices: {as_int(summary["distinct_device_count"])}
- Single-device exact groups: {single_device_group_count:,} ({single_device_pct:.6f}%)
- Multi-device exact groups: {multi_device_group_count:,} ({multi_device_pct:.6f}%)
- All validation checks passed: {str(all_checks_passed).lower()}

## Task-aware source-precision conflicts

{chr(10).join(task_conflict_lines)}

## Device-family coverage

- Devices missing Benign: {", ".join(devices_missing_benign) or "none"}
- Devices missing Gafgyt: {", ".join(devices_missing_gafgyt) or "none"}
- Devices missing Mirai: {", ".join(devices_missing_mirai) or "none"}

For held-out devices without Mirai support, Mirai FNR must be
reported as N/A rather than zero. Macro metrics must include an
explicit class-coverage field.

## LODO feasibility

{chr(10).join(device_table_lines)}

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

- `{COMPACT_LODO_CSV.relative_to(PROJECT_ROOT)}`
- `{FAMILY_SUPPORT_CSV.relative_to(PROJECT_ROOT)}`
- `{TOP_PAIR_CSV.relative_to(PROJECT_ROOT)}`
- `{SUMMARY_JSON.relative_to(PROJECT_ROOT)}`
- `{MANIFEST_CSV.relative_to(PROJECT_ROOT)}`
"""

DOCS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMPLETION_MD.write_text(
    completion_text,
    encoding="utf-8",
)

manifest_inputs = [
    SUMMARY_JSON,
    TASK_SUMMARY_CSV,
    TASK_DETAILS_CSV,
    LODO_CSV,
    CLASS_SUPPORT_CSV,
    PAIR_CSV,
    DEVICE_COUNT_CSV,
    COMPACT_LODO_CSV,
    FAMILY_SUPPORT_CSV,
    TOP_PAIR_CSV,
    COMPLETION_MD,
]

if LOG_FILE.exists():
    manifest_inputs.append(
        LOG_FILE
    )

manifest_rows = []

print("Creating SHA-256 manifest...")

for path in manifest_inputs:
    print(f"  hashing: {path.name}")

    manifest_rows.append(
        {
            "relative_path": str(
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

write_csv_rows(
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
print("PHASE 1C COMPACT RESULTS")
print("=" * 78)
print(
    f"Source exact groups : "
    f"{source_group_count:,}"
)
print(
    f"Single-device groups: "
    f"{single_device_group_count:,} "
    f"({single_device_pct:.6f}%)"
)
print(
    f"Multi-device groups : "
    f"{multi_device_group_count:,} "
    f"({multi_device_pct:.6f}%)"
)
print()

print("DEVICE LEGEND")

for device, device_id in sorted(
    device_aliases.items(),
    key=lambda item: item[1],
):
    print(f"{device_id}={device}")

print()
print(
    "ID | NOVEL_FP_PCT | NOVEL_RAW_PCT "
    "| EXPOSED_FP_PCT | MISSING"
)

for row in compact_rows:
    missing = (
        family_by_device[
            row["device"]
        ][
            "missing_families"
        ]
        or "none"
    )

    print(
        f"{row['device_id']} | "
        f"{row['novel_fingerprint_pct']:.6f} | "
        f"{row['novel_occurrence_pct']:.6f} | "
        f"{row['train_exposed_fingerprint_pct']:.6f} | "
        f"{missing}"
    )

print()
print("TOP 10 DEVICE PAIRS")

for row in top_pair_rows[:10]:
    print(
        f"{row['rank']:02d} | "
        f"{row['device_a_id']}-"
        f"{row['device_b_id']} | "
        f"shared_fp="
        f"{row['shared_fingerprint_count']:,}"
    )

print()
print("VALIDATION CHECKS")

for name, passed in validation_checks.items():
    print(f"{name}: {passed}")

print()
print(f"Compact LODO : {COMPACT_LODO_CSV}")
print(f"Family support: {FAMILY_SUPPORT_CSV}")
print(f"Top pairs    : {TOP_PAIR_CSV}")
print(f"Manifest     : {MANIFEST_CSV}")
print(f"Completion   : {COMPLETION_MD}")

if not all_checks_passed:
    print()
    print("PHASE 1C LOCK FAILED")
    sys.exit(1)

print()
print("PHASE 1C LOCKED")
