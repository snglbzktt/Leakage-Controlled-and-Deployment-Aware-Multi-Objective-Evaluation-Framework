from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path.cwd()

SUMMARY_JSON = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "vs2_canonical_rehash_alignment_summary_v3.json"
)

PROTOCOL_DATABASE = (
    ROOT
    / "data"
    / "splits"
    / "vs2_float32_leakage_ablation_scaled_repaired_seed2026_v3.sqlite"
)

PRIMARY_METADATA = (
    ROOT
    / "data"
    / "processed"
    / "nbaiot_primary_seed2026"
    / "metadata.parquet"
)

CANONICAL_METADATA = (
    ROOT
    / "data"
    / "processed"
    / "nbaiot_float32_canonical_seed2026"
    / "metadata.parquet"
)

OUTPUT_JSON = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "vs2_protocol_only_source_audit_v3.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_path(value: str) -> str:
    return str(value).replace("\\", "/").strip().lower()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


for required_path in (
    SUMMARY_JSON,
    PROTOCOL_DATABASE,
    PRIMARY_METADATA,
    CANONICAL_METADATA,
):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

started = time.perf_counter()

summary = json.loads(
    SUMMARY_JSON.read_text(encoding="utf-8")
)

protocol_only_rows = summary.get(
    "protocol_only_rows",
    [],
)

if not protocol_only_rows:
    raise RuntimeError(
        "The rehash summary contains no protocol-only groups."
    )

protocol_hashes = {
    (
        int(row["hash_forward"]),
        int(row["hash_reverse"]),
    )
    for row in protocol_only_rows
}

protocol_file_ids = sorted(
    {
        int(row["representative_file_id"])
        for row in protocol_only_rows
    }
)

protocol_paths = sorted(
    {
        str(row["representative_file_path"])
        for row in protocol_only_rows
    }
)

protocol_labels = sorted(
    {
        str(row["representative_raw_label"])
        for row in protocol_only_rows
    }
)

print("=" * 92)
print("VS2 PROTOCOL-ONLY SOURCE AUDIT")
print("=" * 92)
print(f"Protocol-only group count : {len(protocol_hashes):,}")
print(f"Representative file IDs   : {protocol_file_ids}")
print(f"Representative raw labels : {protocol_labels}")

database_uri = (
    PROTOCOL_DATABASE.resolve().as_uri()
    + "?mode=ro"
)

source_files: list[dict[str, Any]] = []
file_group_composition: list[dict[str, Any]] = []

with sqlite3.connect(
    database_uri,
    uri=True,
    timeout=120.0,
) as connection:
    connection.row_factory = sqlite3.Row

    quick_check = str(
        connection.execute(
            "PRAGMA quick_check"
        ).fetchone()[0]
    )

    if quick_check.lower() != "ok":
        raise RuntimeError(
            f"Protocol database quick_check failed: {quick_check}"
        )

    placeholders = ",".join(
        "?"
        for _ in protocol_file_ids
    )

    source_rows = connection.execute(
        f"""
        SELECT
            file_id,
            file_path,
            class_label,
            expected_row_count,
            file_size_bytes,
            file_mtime_ns,
            sha256
        FROM source_file_integrity
        WHERE file_id IN ({placeholders})
        ORDER BY file_id
        """,
        protocol_file_ids,
    ).fetchall()

    source_files = [
        dict(row)
        for row in source_rows
    ]

    for source_file in source_files:
        file_id = int(source_file["file_id"])

        composition_rows = connection.execute(
            """
            SELECT
                a.hash_forward,
                a.hash_reverse,
                COUNT(*) AS assignment_row_count,
                MIN(a.family_label) AS family_label_min,
                MAX(a.family_label) AS family_label_max,
                f.raw_row_count,
                f.grouped_split
            FROM assignments a
            JOIN fingerprint_stats f
              ON f.hash_forward = a.hash_forward
             AND f.hash_reverse = a.hash_reverse
            WHERE a.file_id = ?
            GROUP BY
                a.hash_forward,
                a.hash_reverse
            ORDER BY
                assignment_row_count DESC,
                a.hash_forward,
                a.hash_reverse
            """,
            (file_id,),
        ).fetchall()

        for row in composition_rows:
            record = dict(row)
            record["file_id"] = file_id
            record["is_protocol_only_group"] = (
                (
                    int(record["hash_forward"]),
                    int(record["hash_reverse"]),
                )
                in protocol_hashes
            )
            file_group_composition.append(record)

print()
print("PROTOCOL SOURCE FILES")

for row in source_files:
    print(
        "  "
        f"file_id={row['file_id']} | "
        f"expected_rows={int(row['expected_row_count']):,} | "
        f"label={row['class_label']} | "
        f"path={row['file_path']}"
    )

print()
print("SOURCE FILE FINGERPRINT COMPOSITION")

for row in file_group_composition:
    print(
        "  "
        f"file_id={row['file_id']} | "
        f"hash=({row['hash_forward']},{row['hash_reverse']}) | "
        f"assignment_rows={int(row['assignment_row_count']):,} | "
        f"global_raw_rows={int(row['raw_row_count']):,} | "
        f"grouped_split={row['grouped_split']} | "
        f"protocol_only={row['is_protocol_only_group']}"
    )

# Use the locked protocol path to derive the relative source tail.
target_relative_tails: set[str] = set()

for path_text in protocol_paths:
    normalized = normalize_path(path_text)
    marker = "/nbaiot_unpacked/"

    if marker in normalized:
        target_relative_tails.add(
            normalized.split(marker, 1)[1]
        )
    else:
        target_relative_tails.add(
            normalized
        )

target_devices = {
    tail.split("/", 1)[0]
    for tail in target_relative_tails
    if "/" in tail
}

target_raw_labels = {
    label.strip().lower()
    for label in protocol_labels
}

primary_exact_path_representatives = 0
primary_exact_path_occurrences = 0
primary_device_label_representatives = 0
primary_device_label_occurrences = 0
primary_raw_label_representatives = 0
primary_raw_label_occurrences = 0

matched_primary_paths: Counter[str] = Counter()
matched_primary_classes: Counter[str] = Counter()

primary_file = pq.ParquetFile(
    PRIMARY_METADATA
)

for batch in primary_file.iter_batches(
    batch_size=100_000,
    columns=[
        "source_relative_path",
        "source_device",
        "class_label",
        "original_occurrence_count",
    ],
):
    paths = batch.column(0).to_pylist()
    devices = batch.column(1).to_pylist()
    labels = batch.column(2).to_pylist()
    occurrences = batch.column(3).to_pylist()

    for path_value, device, label, occurrence in zip(
        paths,
        devices,
        labels,
        occurrences,
    ):
        normalized_path = normalize_path(path_value)
        normalized_device = str(device).strip().lower()
        normalized_label = str(label).strip().lower()
        occurrence_count = int(occurrence)

        exact_path_match = any(
            normalized_path.endswith(tail)
            for tail in target_relative_tails
        )

        device_label_match = (
            normalized_device in target_devices
            and normalized_label in target_raw_labels
        )

        raw_label_match = (
            normalized_label in target_raw_labels
        )

        if exact_path_match:
            primary_exact_path_representatives += 1
            primary_exact_path_occurrences += occurrence_count
            matched_primary_paths[normalized_path] += 1
            matched_primary_classes[normalized_label] += 1

        if device_label_match:
            primary_device_label_representatives += 1
            primary_device_label_occurrences += occurrence_count

        if raw_label_match:
            primary_raw_label_representatives += 1
            primary_raw_label_occurrences += occurrence_count

canonical_file = pq.ParquetFile(
    CANONICAL_METADATA
)

canonical_total_occurrences = 0

for batch in canonical_file.iter_batches(
    batch_size=200_000,
    columns=["raw_occurrence_count"],
):
    canonical_total_occurrences += sum(
        int(value)
        for value in batch.column(0).to_pylist()
    )

protocol_total_raw_rows = int(
    summary.get(
        "common_protocol_raw_row_count",
        0,
    )
) + sum(
    int(row["raw_row_count"])
    for row in protocol_only_rows
)

protocol_only_raw_rows = sum(
    int(row["raw_row_count"])
    for row in protocol_only_rows
)

source_expected_rows = sum(
    int(row["expected_row_count"])
    for row in source_files
)

source_assignment_rows = sum(
    int(row["assignment_row_count"])
    for row in file_group_composition
)

all_source_groups_protocol_only = bool(
    file_group_composition
    and all(
        bool(row["is_protocol_only_group"])
        for row in file_group_composition
    )
)

if (
    primary_exact_path_representatives == 0
    and primary_device_label_representatives == 0
):
    omission_stage = (
        "The source file is absent from the primary unique-fingerprint "
        "metadata, so the omission occurred before the float32 canonical "
        "dataset was built."
    )
elif primary_exact_path_representatives > 0:
    omission_stage = (
        "The source file is represented in primary metadata; investigate "
        "the float32 canonical builder for the omission."
    )
else:
    omission_stage = (
        "Primary metadata contains related device/label rows but no exact "
        "path match; inspect source-path normalization and upstream indexing."
    )

checks = {
    "protocol_database_quick_check_ok": (
        quick_check.lower() == "ok"
    ),
    "protocol_only_groups_share_one_file": (
        len(protocol_file_ids) == 1
        and len(protocol_paths) == 1
    ),
    "source_expected_rows_equal_protocol_only_raw_rows": (
        source_expected_rows
        == protocol_only_raw_rows
    ),
    "source_assignment_rows_equal_expected_rows": (
        source_assignment_rows
        == source_expected_rows
    ),
    "all_source_groups_are_protocol_only": (
        all_source_groups_protocol_only
    ),
    "canonical_plus_protocol_only_equals_full_protocol_rows": (
        canonical_total_occurrences
        + protocol_only_raw_rows
        == protocol_total_raw_rows
    ),
}

all_checks_passed = all(checks.values())

elapsed = time.perf_counter() - started

result = {
    "status": "completed",
    "generated_at_utc": utc_now(),
    "protocol_database": str(PROTOCOL_DATABASE),
    "primary_metadata": str(PRIMARY_METADATA),
    "canonical_metadata": str(CANONICAL_METADATA),
    "protocol_only_group_count": len(protocol_hashes),
    "protocol_only_raw_rows": protocol_only_raw_rows,
    "protocol_total_raw_rows": protocol_total_raw_rows,
    "protocol_only_raw_row_percentage": (
        100.0
        * protocol_only_raw_rows
        / protocol_total_raw_rows
    ),
    "source_files": source_files,
    "source_file_fingerprint_composition": (
        file_group_composition
    ),
    "primary_exact_path_representative_count": (
        primary_exact_path_representatives
    ),
    "primary_exact_path_occurrence_sum": (
        primary_exact_path_occurrences
    ),
    "primary_device_label_representative_count": (
        primary_device_label_representatives
    ),
    "primary_device_label_occurrence_sum": (
        primary_device_label_occurrences
    ),
    "primary_raw_label_representative_count_all_devices": (
        primary_raw_label_representatives
    ),
    "primary_raw_label_occurrence_sum_all_devices": (
        primary_raw_label_occurrences
    ),
    "matched_primary_paths": dict(
        matched_primary_paths
    ),
    "matched_primary_classes": dict(
        matched_primary_classes
    ),
    "canonical_total_raw_occurrence_sum": (
        canonical_total_occurrences
    ),
    "omission_stage_inference": omission_stage,
    "checks": checks,
    "all_checks_passed": all_checks_passed,
    "elapsed_seconds": elapsed,
}

atomic_json(
    OUTPUT_JSON,
    result,
)

print()
print("=" * 92)
print("SOURCE AUDIT SUMMARY")
print("=" * 92)
print(
    "Protocol-only groups             : "
    f"{len(protocol_hashes):,}"
)
print(
    "Protocol-only raw rows           : "
    f"{protocol_only_raw_rows:,}"
)
print(
    "Protocol-only raw-row percentage : "
    f"{result['protocol_only_raw_row_percentage']:.6f}%"
)
print(
    "Source expected rows             : "
    f"{source_expected_rows:,}"
)
print(
    "Source assignment rows           : "
    f"{source_assignment_rows:,}"
)
print(
    "Canonical represented raw rows   : "
    f"{canonical_total_occurrences:,}"
)
print(
    "Full protocol raw rows           : "
    f"{protocol_total_raw_rows:,}"
)
print(
    "Primary exact-path representatives: "
    f"{primary_exact_path_representatives:,}"
)
print(
    "Primary exact-path occurrences    : "
    f"{primary_exact_path_occurrences:,}"
)
print(
    "Primary device+label representatives: "
    f"{primary_device_label_representatives:,}"
)
print(
    "Primary device+label occurrences    : "
    f"{primary_device_label_occurrences:,}"
)
print()
print(f"Omission-stage inference: {omission_stage}")
print()
print("CHECKS")

for name, passed in checks.items():
    print(f"  {name}: {passed}")

print()
print(f"All checks passed : {all_checks_passed}")
print(f"Report            : {OUTPUT_JSON}")
print(f"Elapsed seconds   : {elapsed:.1f}")
print()
print("VS2 PROTOCOL-ONLY SOURCE AUDIT COMPLETED")
