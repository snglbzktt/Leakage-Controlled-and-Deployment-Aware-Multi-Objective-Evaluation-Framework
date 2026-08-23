from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path.cwd()

EXPECTED_ROW_COUNT = 2_482_676
EXPECTED_FEATURE_COUNT = 115

EXPECTED_SPLIT_COUNTS = {
    "train": 1_738_133,
    "validation": 371_884,
    "test": 372_659,
}

SPLIT_TO_CODE = {
    "train": 0,
    "validation": 1,
    "test": 2,
}

LABELS = [
    "benign",
    "gafgyt_combo",
    "gafgyt_junk",
    "gafgyt_scan",
    "gafgyt_tcp",
    "gafgyt_udp",
    "mirai_ack",
    "mirai_scan",
    "mirai_syn",
    "mirai_udp",
    "mirai_udpplain",
]

LABEL_TO_ID = {
    label: index
    for index, label in enumerate(LABELS)
}

PRIMARY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_primary_seed2026"
)

METADATA_FILE = (
    PRIMARY_ROOT
    / "metadata.parquet"
)

FEATURE_FILES = {
    "train":
        PRIMARY_ROOT
        / "train.parquet",

    "validation":
        PRIMARY_ROOT
        / "validation.parquet",

    "test":
        PRIMARY_ROOT
        / "test.parquet",
}

MEMBERSHIP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_early_lodo_membership_v2"
)

LOCKED_SPLIT_CODE_FILE = (
    MEMBERSHIP_ROOT
    / "split_code.npy"
)

LOCKED_FAMILY_FILE = (
    MEMBERSHIP_ROOT
    / "family3.npy"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_early_lodo_feature_alignment_v2"
)

AUDIT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

SUMMARY_FILE = (
    AUDIT_ROOT
    / "early_lodo_feature_alignment_summary_v2.json"
)

REPORT_FILE = (
    AUDIT_ROOT
    / "early_lodo_feature_alignment_v2.txt"
)

SPLIT_SUMMARY_FILE = (
    AUDIT_ROOT
    / "early_lodo_feature_split_validation_v2.csv"
)

MANIFEST_FILE = (
    AUDIT_ROOT
    / "early_lodo_feature_alignment_manifest_v2.csv"
)

ALIGNMENT_FILE = (
    OUTPUT_ROOT
    / "alignment.json"
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
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

    os.replace(
        temporary,
        path,
    )


def save_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def normalize_text(
    value: Any,
) -> str:
    return (
        str(value)
        .strip()
        .casefold()
    )


def encode_labels(
    values: list[Any],
) -> np.ndarray:
    encoded = np.empty(
        len(values),
        dtype=np.uint8,
    )

    for index, value in enumerate(values):
        label = normalize_text(value)

        if label not in LABEL_TO_ID:
            raise ValueError(
                f"Unknown class label: {value!r}"
            )

        encoded[index] = (
            LABEL_TO_ID[label]
        )

    return encoded


def encode_splits(
    values: list[Any],
) -> np.ndarray:
    encoded = np.empty(
        len(values),
        dtype=np.uint8,
    )

    for index, value in enumerate(values):
        split = normalize_text(value)

        if split not in SPLIT_TO_CODE:
            raise ValueError(
                f"Unknown split value: {value!r}"
            )

        encoded[index] = (
            SPLIT_TO_CODE[split]
        )

    return encoded


def labels_to_family3(
    label_ids: np.ndarray,
) -> np.ndarray:
    family = np.empty(
        label_ids.shape,
        dtype=np.int8,
    )

    family[
        label_ids == LABEL_TO_ID["benign"]
    ] = 0

    gafgyt_ids = np.asarray(
        [
            LABEL_TO_ID[label]
            for label in LABELS
            if label.startswith(
                "gafgyt_"
            )
        ],
        dtype=np.uint8,
    )

    mirai_ids = np.asarray(
        [
            LABEL_TO_ID[label]
            for label in LABELS
            if label.startswith(
                "mirai_"
            )
        ],
        dtype=np.uint8,
    )

    family[
        np.isin(
            label_ids,
            gafgyt_ids,
        )
    ] = 1

    family[
        np.isin(
            label_ids,
            mirai_ids,
        )
    ] = 2

    return family


def is_numeric_type(
    data_type: pa.DataType,
) -> bool:
    return bool(
        pa.types.is_integer(data_type)
        or pa.types.is_floating(
            data_type
        )
        or pa.types.is_decimal(
            data_type
        )
    )


required_files = [
    METADATA_FILE,
    LOCKED_SPLIT_CODE_FILE,
    LOCKED_FAMILY_FILE,
    *FEATURE_FILES.values(),
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


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


started_at = time.perf_counter()


# -------------------------------------------------------------------------
# Metadata row order
# -------------------------------------------------------------------------

metadata_parquet = pq.ParquetFile(
    METADATA_FILE
)

metadata_row_count = int(
    metadata_parquet
    .metadata
    .num_rows
)

if metadata_row_count != EXPECTED_ROW_COUNT:
    raise RuntimeError(
        "Metadata row count mismatch."
    )


global_index_chunks: dict[
    str,
    list[np.ndarray]
] = {
    split: []
    for split in SPLIT_TO_CODE
}

metadata_label_chunks: dict[
    str,
    list[np.ndarray]
] = {
    split: []
    for split in SPLIT_TO_CODE
}

metadata_split_code = np.empty(
    EXPECTED_ROW_COUNT,
    dtype=np.uint8,
)

metadata_label_id = np.empty(
    EXPECTED_ROW_COUNT,
    dtype=np.uint8,
)


processed_rows = 0

print("=" * 82)
print("PHASE 2E-2 EARLY LODO FEATURE ALIGNMENT")
print("=" * 82)
print(f"Metadata rows: {metadata_row_count:,}")
print()
print("Reading metadata row order...")


for batch_number, batch in enumerate(
    metadata_parquet.iter_batches(
        batch_size=100_000,
        columns=[
            "split",
            "class_label",
        ],
    ),
    start=1,
):
    table = pa.Table.from_batches(
        [batch]
    )

    split_values = (
        table["split"]
        .to_pylist()
    )

    label_values = (
        table["class_label"]
        .to_pylist()
    )

    split_ids = encode_splits(
        split_values
    )

    label_ids = encode_labels(
        label_values
    )

    batch_size = len(split_ids)

    start = processed_rows
    end = start + batch_size

    metadata_split_code[
        start:end
    ] = split_ids

    metadata_label_id[
        start:end
    ] = label_ids

    global_indices = np.arange(
        start,
        end,
        dtype=np.int64,
    )

    for split, split_code in (
        SPLIT_TO_CODE.items()
    ):
        mask = (
            split_ids
            == split_code
        )

        if not np.any(mask):
            continue

        global_index_chunks[
            split
        ].append(
            global_indices[mask]
        )

        metadata_label_chunks[
            split
        ].append(
            label_ids[mask]
        )

    processed_rows = end

    if (
        batch_number == 1
        or batch_number % 5 == 0
        or processed_rows
        == EXPECTED_ROW_COUNT
    ):
        print(
            f"Metadata processed: "
            f"{processed_rows:,}/"
            f"{EXPECTED_ROW_COUNT:,}"
        )


global_indices_by_split = {
    split:
        np.concatenate(
            global_index_chunks[split]
        )
    for split in SPLIT_TO_CODE
}

metadata_labels_by_split = {
    split:
        np.concatenate(
            metadata_label_chunks[split]
        )
    for split in SPLIT_TO_CODE
}


# -------------------------------------------------------------------------
# Compare locked membership arrays
# -------------------------------------------------------------------------

locked_split_code = np.load(
    LOCKED_SPLIT_CODE_FILE,
    mmap_mode="r",
    allow_pickle=False,
)

locked_family3 = np.load(
    LOCKED_FAMILY_FILE,
    mmap_mode="r",
    allow_pickle=False,
)

metadata_family3 = labels_to_family3(
    metadata_label_id
)

split_code_mismatch_count = int(
    np.count_nonzero(
        metadata_split_code
        != locked_split_code
    )
)

family3_mismatch_count = int(
    np.count_nonzero(
        metadata_family3
        != locked_family3
    )
)


# -------------------------------------------------------------------------
# Validate feature partitions and label sequence
# -------------------------------------------------------------------------

split_validation_rows = []
feature_columns_reference: list[str] | None = None

all_schema_matches = True
all_split_counts_match = True
all_label_sequences_match = True
all_numeric_feature_counts_match = True


for split in (
    "train",
    "validation",
    "test",
):
    feature_file = (
        FEATURE_FILES[split]
    )

    parquet_file = pq.ParquetFile(
        feature_file
    )

    row_count = int(
        parquet_file
        .metadata
        .num_rows
    )

    schema = (
        parquet_file
        .schema_arrow
    )

    columns = [
        field.name
        for field in schema
    ]

    numeric_columns = [
        field.name
        for field in schema
        if is_numeric_type(
            field.type
        )
    ]

    non_numeric_columns = [
        field.name
        for field in schema
        if not is_numeric_type(
            field.type
        )
    ]

    feature_columns = [
        column
        for column in numeric_columns
        if column != "class_label"
    ]

    row_count_matches = (
        row_count
        == EXPECTED_SPLIT_COUNTS[
            split
        ]
        == len(
            global_indices_by_split[
                split
            ]
        )
    )

    numeric_feature_count_matches = (
        len(feature_columns)
        == EXPECTED_FEATURE_COUNT
    )

    class_label_available = (
        "class_label"
        in columns
    )

    schema_matches_reference = True

    if feature_columns_reference is None:
        feature_columns_reference = (
            feature_columns
        )
    else:
        schema_matches_reference = (
            feature_columns
            == feature_columns_reference
        )

    expected_labels = (
        metadata_labels_by_split[
            split
        ]
    )

    compared_rows = 0
    label_mismatch_count = 0
    first_label_mismatch_index = None

    for batch in parquet_file.iter_batches(
        batch_size=100_000,
        columns=[
            "class_label",
        ],
    ):
        table = pa.Table.from_batches(
            [batch]
        )

        observed_labels = encode_labels(
            table["class_label"]
            .to_pylist()
        )

        start = compared_rows
        end = (
            start
            + len(observed_labels)
        )

        expected_chunk = (
            expected_labels[
                start:end
            ]
        )

        mismatch_mask = (
            observed_labels
            != expected_chunk
        )

        mismatch_count = int(
            np.count_nonzero(
                mismatch_mask
            )
        )

        if (
            mismatch_count > 0
            and first_label_mismatch_index
            is None
        ):
            first_label_mismatch_index = int(
                start
                + np.flatnonzero(
                    mismatch_mask
                )[0]
            )

        label_mismatch_count += (
            mismatch_count
        )

        compared_rows = end

    label_sequence_matches = (
        compared_rows
        == len(expected_labels)
        and label_mismatch_count == 0
    )

    all_schema_matches &= (
        schema_matches_reference
        and class_label_available
    )

    all_split_counts_match &= (
        row_count_matches
    )

    all_numeric_feature_counts_match &= (
        numeric_feature_count_matches
    )

    all_label_sequences_match &= (
        label_sequence_matches
    )

    split_validation_rows.append(
        {
            "split":
                split,

            "feature_file":
                str(
                    feature_file
                    .relative_to(
                        PROJECT_ROOT
                    )
                ),

            "expected_row_count":
                EXPECTED_SPLIT_COUNTS[
                    split
                ],

            "feature_row_count":
                row_count,

            "metadata_filtered_row_count":
                len(
                    global_indices_by_split[
                        split
                    ]
                ),

            "numeric_feature_count":
                len(feature_columns),

            "non_numeric_columns":
                "|".join(
                    non_numeric_columns
                ),

            "row_count_matches":
                row_count_matches,

            "schema_matches_reference":
                schema_matches_reference,

            "label_rows_compared":
                compared_rows,

            "label_mismatch_count":
                label_mismatch_count,

            "first_label_mismatch_index":
                first_label_mismatch_index,

            "label_sequence_matches":
                label_sequence_matches,
        }
    )


# -------------------------------------------------------------------------
# Global index coverage
# -------------------------------------------------------------------------

coverage = np.zeros(
    EXPECTED_ROW_COUNT,
    dtype=np.uint8,
)

duplicate_global_index_count = 0

for split in (
    "train",
    "validation",
    "test",
):
    indexes = (
        global_indices_by_split[
            split
        ]
    )

    duplicate_global_index_count += int(
        np.count_nonzero(
            coverage[indexes] != 0
        )
    )

    coverage[indexes] += 1

uncovered_global_index_count = int(
    np.count_nonzero(
        coverage == 0
    )
)

multiply_covered_index_count = int(
    np.count_nonzero(
        coverage > 1
    )
)


# -------------------------------------------------------------------------
# Save split-local to global mappings
# -------------------------------------------------------------------------

index_files: dict[str, Path] = {}

for split in (
    "train",
    "validation",
    "test",
):
    output_file = (
        OUTPUT_ROOT
        / f"{split}_global_indices.npy"
    )

    temporary_file = (
        OUTPUT_ROOT
        / f"{split}_global_indices.partial.npy"
    )

    if temporary_file.exists():
        temporary_file.unlink()

    np.save(
        temporary_file,
        global_indices_by_split[
            split
        ],
        allow_pickle=False,
    )

    os.replace(
        temporary_file,
        output_file,
    )

    index_files[split] = (
        output_file
    )


validation_checks = {
    "metadata_row_count":
        metadata_row_count
        == EXPECTED_ROW_COUNT,

    "metadata_processed_completely":
        processed_rows
        == EXPECTED_ROW_COUNT,

    "all_split_counts_match":
        all_split_counts_match,

    "all_feature_schemas_match":
        all_schema_matches,

    "all_numeric_feature_counts_115":
        all_numeric_feature_counts_match,

    "all_label_sequences_match":
        all_label_sequences_match,

    "locked_split_code_exact_match":
        split_code_mismatch_count
        == 0,

    "locked_family3_exact_match":
        family3_mismatch_count
        == 0,

    "all_global_rows_covered":
        uncovered_global_index_count
        == 0,

    "no_duplicate_global_indices":
        duplicate_global_index_count
        == 0
        and multiply_covered_index_count
        == 0,

    "total_split_rows":
        sum(
            len(
                global_indices_by_split[
                    split
                ]
            )
            for split in SPLIT_TO_CODE
        )
        == EXPECTED_ROW_COUNT,

    "index_files_created":
        all(
            path.exists()
            for path
            in index_files.values()
        ),
}


all_checks_passed = all(
    validation_checks.values()
)


alignment = {
    "protocol_version":
        "early_lodo_feature_alignment_v2_1",

    "status":
        (
            "validated"
            if all_checks_passed
            else "failed"
        ),

    "completed_at":
        utc_now(),

    "metadata_file":
        str(METADATA_FILE),

    "row_alignment": (
        "Each split parquet row maps to "
        "metadata.parquet through the corresponding "
        "*_global_indices.npy array. The local row "
        "position in the split file is the array "
        "position; the array value is the global "
        "membership row index."
    ),

    "feature_source_policy": (
        "Use raw primary split parquet features. "
        "Do not use numeric_order_pilot pipeline_a "
        "NPY features because LODO requires a new "
        "fold-train scaler."
    ),

    "feature_count":
        EXPECTED_FEATURE_COUNT,

    "feature_columns":
        feature_columns_reference,

    "class_label_column":
        "class_label",

    "splits": {
        split: {
            "feature_file":
                str(
                    FEATURE_FILES[
                        split
                    ]
                ),

            "global_index_file":
                str(
                    index_files[
                        split
                    ]
                ),

            "row_count":
                len(
                    global_indices_by_split[
                        split
                    ]
                ),
        }
        for split in SPLIT_TO_CODE
    },

    "locked_membership_arrays": {
        "split_code":
            str(
                LOCKED_SPLIT_CODE_FILE
            ),

        "family3":
            str(
                LOCKED_FAMILY_FILE
            ),
    },
}


save_json(
    ALIGNMENT_FILE,
    alignment,
)


elapsed_seconds = (
    time.perf_counter()
    - started_at
)


summary = {
    "protocol_version":
        "early_lodo_feature_alignment_v2_1",

    "status":
        (
            "passed"
            if all_checks_passed
            else "failed"
        ),

    "completed_at":
        utc_now(),

    "elapsed_seconds":
        elapsed_seconds,

    "metadata_row_count":
        metadata_row_count,

    "feature_count":
        EXPECTED_FEATURE_COUNT,

    "split_code_mismatch_count":
        split_code_mismatch_count,

    "family3_mismatch_count":
        family3_mismatch_count,

    "uncovered_global_index_count":
        uncovered_global_index_count,

    "duplicate_global_index_count":
        duplicate_global_index_count,

    "multiply_covered_index_count":
        multiply_covered_index_count,

    "split_validation":
        split_validation_rows,

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,

    "alignment_file":
        str(ALIGNMENT_FILE),
}


save_json(
    SUMMARY_FILE,
    summary,
)


write_csv(
    SPLIT_SUMMARY_FILE,
    split_validation_rows,
    list(
        split_validation_rows[0]
        .keys()
    ),
)


manifest_paths = [
    METADATA_FILE,
    *FEATURE_FILES.values(),
    LOCKED_SPLIT_CODE_FILE,
    LOCKED_FAMILY_FILE,
    *index_files.values(),
    ALIGNMENT_FILE,
    SUMMARY_FILE,
    SPLIT_SUMMARY_FILE,
]

manifest_rows = [
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
    for path in manifest_paths
]


write_csv(
    MANIFEST_FILE,
    manifest_rows,
    [
        "relative_path",
        "size_bytes",
        "sha256",
    ],
)


report_lines = [
    "=" * 82,
    "PHASE 2E-2 EARLY LODO FEATURE ALIGNMENT",
    "=" * 82,
    "",
    (
        f"Metadata rows          : "
        f"{metadata_row_count:,}"
    ),
    (
        f"Feature count          : "
        f"{EXPECTED_FEATURE_COUNT}"
    ),
    (
        f"Split-code mismatches  : "
        f"{split_code_mismatch_count:,}"
    ),
    (
        f"Family mismatches      : "
        f"{family3_mismatch_count:,}"
    ),
    (
        f"Uncovered global rows  : "
        f"{uncovered_global_index_count:,}"
    ),
    (
        f"Duplicate global rows  : "
        f"{duplicate_global_index_count:,}"
    ),
    "",
    "SPLIT VALIDATION",
]

for row in split_validation_rows:
    report_lines.append(
        (
            f"- {row['split']} | "
            f"feature_rows="
            f"{row['feature_row_count']:,} | "
            f"metadata_rows="
            f"{row['metadata_filtered_row_count']:,} | "
            f"features="
            f"{row['numeric_feature_count']} | "
            f"label_mismatches="
            f"{row['label_mismatch_count']:,} | "
            f"aligned="
            f"{row['label_sequence_matches']}"
        )
    )

report_lines.extend(
    [
        "",
        "VALIDATION CHECKS",
    ]
)

for name, passed in (
    validation_checks.items()
):
    report_lines.append(
        f"{name}: {passed}"
    )


REPORT_FILE.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


print()
print("=" * 82)
print("PHASE 2E-2 FEATURE ALIGNMENT SUMMARY")
print("=" * 82)

for row in split_validation_rows:
    print(
        f"{row['split']} | "
        f"feature_rows="
        f"{row['feature_row_count']:,} | "
        f"metadata_rows="
        f"{row['metadata_filtered_row_count']:,} | "
        f"features="
        f"{row['numeric_feature_count']} | "
        f"label_mismatches="
        f"{row['label_mismatch_count']:,} | "
        f"aligned="
        f"{row['label_sequence_matches']}"
    )

print()
print(
    f"Split-code mismatches : "
    f"{split_code_mismatch_count:,}"
)

print(
    f"Family mismatches     : "
    f"{family3_mismatch_count:,}"
)

print(
    f"Uncovered rows        : "
    f"{uncovered_global_index_count:,}"
)

print(
    f"Duplicate rows        : "
    f"{duplicate_global_index_count:,}"
)

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"Alignment: {ALIGNMENT_FILE}")
print(f"Summary  : {SUMMARY_FILE}")
print(f"Splits   : {SPLIT_SUMMARY_FILE}")
print(f"Manifest : {MANIFEST_FILE}")
print(f"Report   : {REPORT_FILE}")

if not all_checks_passed:
    print()
    print(
        "PHASE 2E-2 FEATURE "
        "ALIGNMENT FAILED"
    )
    sys.exit(1)

print()
print(
    "PHASE 2E-2 FEATURE "
    "ALIGNMENT PASSED"
)
