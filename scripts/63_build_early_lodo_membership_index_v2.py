from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path.cwd()

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_primary_seed2026"
    / "metadata.parquet"
)

MEMBERSHIP_DATABASE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "nbaiot_duplicate_audit.sqlite"
)

DEVICE_SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "device_lodo_compact_summary_v2.csv"
)

DEVICE_FAMILY_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "device_family_support_v2.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_early_lodo_membership_v2"
)

AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

SUMMARY_FILE = (
    AUDIT_DIRECTORY
    / "early_lodo_membership_index_summary_v2.json"
)

DEVICE_VALIDATION_FILE = (
    AUDIT_DIRECTORY
    / "early_lodo_membership_device_validation_v2.csv"
)

FOLD_FEASIBILITY_FILE = (
    AUDIT_DIRECTORY
    / "early_lodo_fold_feasibility_v2.csv"
)

REPORT_FILE = (
    AUDIT_DIRECTORY
    / "early_lodo_membership_index_v2.txt"
)


EXPECTED_GROUP_COUNT = 2_482_676
EXPECTED_MEMBERSHIP_COUNT = 6_904_827
EXPECTED_RAW_RECORD_COUNT = 7_062_606
EXPECTED_DEVICE_COUNT = 9

BATCH_SIZE = 50_000

CLASS_NAMES = [
    "benign",
    "gafgyt",
    "mirai",
]

SPLIT_CODES = {
    "train": 0,
    "validation": 1,
    "test": 2,
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_csv(
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


def encode_family(
    label: str,
) -> int:
    normalized = (
        str(label)
        .strip()
        .casefold()
    )

    if normalized == "benign":
        return 0

    if (
        normalized == "gafgyt"
        or normalized.startswith(
            "gafgyt_"
        )
    ):
        return 1

    if (
        normalized == "mirai"
        or normalized.startswith(
            "mirai_"
        )
    ):
        return 2

    raise ValueError(
        f"Unknown class label: {label!r}"
    )


def encode_split(
    value: str,
) -> int:
    normalized = (
        str(value)
        .strip()
        .casefold()
    )

    if normalized not in SPLIT_CODES:
        raise ValueError(
            f"Unknown split: {value!r}"
        )

    return SPLIT_CODES[
        normalized
    ]


def validate_npy(
    path: Path,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> bool:
    if not path.exists():
        return False

    try:
        array = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        return (
            array.shape == shape
            and array.dtype
            == np.dtype(dtype)
        )

    except Exception:
        return False


required_files = [
    METADATA_FILE,
    MEMBERSHIP_DATABASE,
    DEVICE_SUMMARY_FILE,
    DEVICE_FAMILY_FILE,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Missing input files:")

    for path in missing_files:
        print(f"- {path}")

    sys.exit(1)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


device_summary_rows = load_csv(
    DEVICE_SUMMARY_FILE
)

device_family_rows = load_csv(
    DEVICE_FAMILY_FILE
)

expected_device_summary = {
    row["device"]: row
    for row in device_summary_rows
}

expected_device_family = {
    row["device"]: row
    for row in device_family_rows
}


database_uri = (
    MEMBERSHIP_DATABASE
    .resolve()
    .as_uri()
    + "?mode=ro"
)

connection = sqlite3.connect(
    database_uri,
    uri=True,
)

connection.row_factory = sqlite3.Row

connection.execute(
    "PRAGMA temp_store = MEMORY"
)

connection.execute(
    "PRAGMA cache_size = -131072"
)


devices = [
    str(row[0])
    for row in connection.execute(
        """
        SELECT DISTINCT device
        FROM vector_device
        ORDER BY device
        """
    ).fetchall()
]

if len(devices) != EXPECTED_DEVICE_COUNT:
    raise RuntimeError(
        "Expected nine devices, found "
        f"{len(devices)}."
    )

device_to_index = {
    device: index
    for index, device
    in enumerate(devices)
}


vector_device_row_count = int(
    connection.execute(
        """
        SELECT COUNT(*)
        FROM vector_device
        """
    ).fetchone()[0]
)

vector_group_count = int(
    connection.execute(
        """
        SELECT COUNT(*)
        FROM vector_groups
        """
    ).fetchone()[0]
)

if (
    vector_device_row_count
    != EXPECTED_MEMBERSHIP_COUNT
):
    raise RuntimeError(
        "vector_device row count mismatch."
    )

if vector_group_count != EXPECTED_GROUP_COUNT:
    raise RuntimeError(
        "vector_groups row count mismatch."
    )


parquet_file = pq.ParquetFile(
    METADATA_FILE
)

metadata_row_count = int(
    parquet_file.metadata.num_rows
)

if metadata_row_count != EXPECTED_GROUP_COUNT:
    raise RuntimeError(
        "Metadata row count mismatch."
    )

required_columns = [
    "hash_forward",
    "hash_reverse",
    "class_label",
    "split",
    "source_device_count",
]

metadata_columns = [
    field.name
    for field
    in parquet_file.schema_arrow
]

missing_columns = [
    column
    for column in required_columns
    if column not in metadata_columns
]

if missing_columns:
    raise RuntimeError(
        "Missing metadata columns: "
        + ", ".join(
            missing_columns
        )
    )


device_mask_file = (
    OUTPUT_ROOT
    / "device_mask.npy"
)

device_occurrence_file = (
    OUTPUT_ROOT
    / "device_occurrence_counts.npy"
)

family_file = (
    OUTPUT_ROOT
    / "family3.npy"
)

split_file = (
    OUTPUT_ROOT
    / "split_code.npy"
)

source_device_count_file = (
    OUTPUT_ROOT
    / "source_device_count.npy"
)

alignment_file = (
    OUTPUT_ROOT
    / "alignment.json"
)


output_specs = [
    (
        device_mask_file,
        (
            EXPECTED_GROUP_COUNT,
        ),
        np.uint16,
    ),
    (
        device_occurrence_file,
        (
            EXPECTED_GROUP_COUNT,
            EXPECTED_DEVICE_COUNT,
        ),
        np.uint32,
    ),
    (
        family_file,
        (
            EXPECTED_GROUP_COUNT,
        ),
        np.int8,
    ),
    (
        split_file,
        (
            EXPECTED_GROUP_COUNT,
        ),
        np.uint8,
    ),
    (
        source_device_count_file,
        (
            EXPECTED_GROUP_COUNT,
        ),
        np.uint8,
    ),
]


existing_complete = (
    SUMMARY_FILE.exists()
    and alignment_file.exists()
    and all(
        validate_npy(
            path,
            shape,
            dtype,
        )
        for path, shape, dtype
        in output_specs
    )
)

if existing_complete:
    existing_summary = json.loads(
        SUMMARY_FILE.read_text(
            encoding="utf-8",
        )
    )

    if existing_summary.get(
        "all_checks_passed"
    ) is True:
        print(
            "Existing validated membership "
            "index found."
        )
        print(
            "PHASE 2C MEMBERSHIP INDEX PASSED"
        )
        sys.exit(0)


partial_paths = {
    path:
        path.with_name(
            path.stem + ".partial.npy"
        )
    for path, _, _
    in output_specs
}

for partial in partial_paths.values():
    if partial.exists():
        partial.unlink()


device_mask = np.lib.format.open_memmap(
    partial_paths[
        device_mask_file
    ],
    mode="w+",
    dtype=np.uint16,
    shape=(
        EXPECTED_GROUP_COUNT,
    ),
)

device_occurrence = (
    np.lib.format.open_memmap(
        partial_paths[
            device_occurrence_file
        ],
        mode="w+",
        dtype=np.uint32,
        shape=(
            EXPECTED_GROUP_COUNT,
            EXPECTED_DEVICE_COUNT,
        ),
    )
)

family3 = np.lib.format.open_memmap(
    partial_paths[
        family_file
    ],
    mode="w+",
    dtype=np.int8,
    shape=(
        EXPECTED_GROUP_COUNT,
    ),
)

split_code = np.lib.format.open_memmap(
    partial_paths[
        split_file
    ],
    mode="w+",
    dtype=np.uint8,
    shape=(
        EXPECTED_GROUP_COUNT,
    ),
)

source_device_count = (
    np.lib.format.open_memmap(
        partial_paths[
            source_device_count_file
        ],
        mode="w+",
        dtype=np.uint8,
        shape=(
            EXPECTED_GROUP_COUNT,
        ),
    )
)


connection.execute(
    """
    CREATE TEMP TABLE batch_groups (
        row_index INTEGER PRIMARY KEY,
        hash_forward INTEGER NOT NULL,
        hash_reverse INTEGER NOT NULL
    )
    """
)


processed_rows = 0
returned_memberships = 0
label_conflict_rows = 0
family_mismatch_rows = 0
unknown_device_rows = 0

started_at = time.perf_counter()


print("=" * 78)
print("PHASE 2C EARLY LODO MEMBERSHIP INDEX")
print("=" * 78)
print(f"Metadata rows      : {metadata_row_count:,}")
print(f"Membership rows    : {vector_device_row_count:,}")
print(f"Devices            : {len(devices)}")
print(f"Output directory   : {OUTPUT_ROOT}")
print()


for batch_number, batch in enumerate(
    parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=required_columns,
    ),
    start=1,
):
    frame = batch.to_pandas()

    batch_rows = len(frame)

    start_index = processed_rows
    end_index = (
        start_index + batch_rows
    )

    hashes_forward = np.asarray(
        frame["hash_forward"],
        dtype=np.int64,
    )

    hashes_reverse = np.asarray(
        frame["hash_reverse"],
        dtype=np.int64,
    )

    encoded_family = np.fromiter(
        (
            encode_family(value)
            for value
            in frame["class_label"]
        ),
        dtype=np.int8,
        count=batch_rows,
    )

    encoded_split = np.fromiter(
        (
            encode_split(value)
            for value
            in frame["split"]
        ),
        dtype=np.uint8,
        count=batch_rows,
    )

    expected_membership_count = (
        np.asarray(
            frame[
                "source_device_count"
            ],
            dtype=np.uint8,
        )
    )

    family3[
        start_index:end_index
    ] = encoded_family

    split_code[
        start_index:end_index
    ] = encoded_split

    source_device_count[
        start_index:end_index
    ] = expected_membership_count

    connection.execute(
        "DELETE FROM batch_groups"
    )

    connection.executemany(
        """
        INSERT INTO batch_groups (
            row_index,
            hash_forward,
            hash_reverse
        )
        VALUES (?, ?, ?)
        """,
        (
            (
                int(start_index + offset),
                int(hashes_forward[offset]),
                int(hashes_reverse[offset]),
            )
            for offset in range(
                batch_rows
            )
        ),
    )

    membership_rows = (
        connection.execute(
            """
            SELECT
                batch_groups.row_index
                    AS row_index,

                vector_device.device
                    AS device,

                SUM(
                    vector_device.occurrence_count
                ) AS occurrence_count,

                MIN(
                    vector_device.class_label
                ) AS class_label,

                COUNT(
                    DISTINCT
                    vector_device.class_label
                ) AS label_count

            FROM batch_groups

            INNER JOIN vector_device
                ON
                    vector_device.hash_forward
                        = batch_groups.hash_forward
                    AND
                    vector_device.hash_reverse
                        = batch_groups.hash_reverse

            GROUP BY
                batch_groups.row_index,
                vector_device.device

            ORDER BY
                batch_groups.row_index,
                vector_device.device
            """
        )
    )

    for row in membership_rows:
        row_index = int(
            row["row_index"]
        )

        device = str(
            row["device"]
        )

        device_index = (
            device_to_index.get(
                device
            )
        )

        if device_index is None:
            unknown_device_rows += 1
            continue

        occurrence_count = int(
            row["occurrence_count"]
        )

        if occurrence_count < 1:
            raise RuntimeError(
                "Invalid occurrence count."
            )

        if occurrence_count > np.iinfo(
            np.uint32
        ).max:
            raise RuntimeError(
                "Occurrence count exceeds uint32."
            )

        device_mask[
            row_index
        ] |= np.uint16(
            1 << device_index
        )

        device_occurrence[
            row_index,
            device_index,
        ] = np.uint32(
            occurrence_count
        )

        returned_memberships += 1

        if int(row["label_count"]) != 1:
            label_conflict_rows += 1

        database_family = encode_family(
            row["class_label"]
        )

        if (
            database_family
            != int(
                family3[row_index]
            )
        ):
            family_mismatch_rows += 1

    processed_rows = end_index

    if (
        batch_number == 1
        or batch_number % 5 == 0
        or processed_rows
        == EXPECTED_GROUP_COUNT
    ):
        print(
            f"Processed: "
            f"{processed_rows:,}/"
            f"{EXPECTED_GROUP_COUNT:,}"
        )


device_mask.flush()
device_occurrence.flush()
family3.flush()
split_code.flush()
source_device_count.flush()

del device_mask
del device_occurrence
del family3
del split_code
del source_device_count


for final_path, _, _ in output_specs:
    os.replace(
        partial_paths[
            final_path
        ],
        final_path,
    )


device_mask = np.load(
    device_mask_file,
    mmap_mode="r",
    allow_pickle=False,
)

device_occurrence = np.load(
    device_occurrence_file,
    mmap_mode="r",
    allow_pickle=False,
)

family3 = np.load(
    family_file,
    mmap_mode="r",
    allow_pickle=False,
)

split_code = np.load(
    split_file,
    mmap_mode="r",
    allow_pickle=False,
)

source_device_count = np.load(
    source_device_count_file,
    mmap_mode="r",
    allow_pickle=False,
)


bit_count_lookup = np.asarray(
    [
        value.bit_count()
        for value in range(
            1 << EXPECTED_DEVICE_COUNT
        )
    ],
    dtype=np.uint8,
)

derived_device_count = (
    bit_count_lookup[
        device_mask
    ]
)

membership_count_from_index = int(
    derived_device_count.sum(
        dtype=np.int64
    )
)

raw_record_count_from_index = int(
    device_occurrence.sum(
        dtype=np.uint64
    )
)

missing_membership_rows = int(
    np.sum(
        derived_device_count == 0
    )
)

membership_count_mismatch_rows = int(
    np.sum(
        derived_device_count
        != source_device_count
    )
)


device_validation_rows = []

all_device_summary_matches = True
all_device_family_matches = True


for device_index, device in enumerate(
    devices
):
    occurrence_column = (
        device_occurrence[
            :,
            device_index
        ]
    )

    membership = (
        occurrence_column > 0
    )

    observed_fingerprint_count = int(
        membership.sum()
    )

    observed_raw_count = int(
        occurrence_column.sum(
            dtype=np.uint64
        )
    )

    expected_summary = (
        expected_device_summary[
            device
        ]
    )

    expected_fingerprint_count = int(
        expected_summary[
            "fingerprint_count"
        ]
    )

    expected_raw_count = int(
        expected_summary[
            "raw_occurrence_count"
        ]
    )

    summary_match = (
        observed_fingerprint_count
        == expected_fingerprint_count
        and observed_raw_count
        == expected_raw_count
    )

    all_device_summary_matches &= (
        summary_match
    )

    expected_family = (
        expected_device_family[
            device
        ]
    )

    family_values = (
        family3[membership]
    )

    family_fingerprint_counts = (
        np.bincount(
            family_values.astype(
                np.int64
            ),
            minlength=3,
        )
    )

    family_raw_counts = np.bincount(
        family3.astype(
            np.int64
        ),
        weights=occurrence_column,
        minlength=3,
    ).astype(
        np.int64
    )

    family_matches = True

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        expected_family_fingerprint = int(
            expected_family[
                f"{class_name}_fingerprint_count"
            ]
        )

        expected_family_raw = int(
            expected_family[
                f"{class_name}_raw_occurrence_count"
            ]
        )

        family_matches &= (
            int(
                family_fingerprint_counts[
                    class_index
                ]
            )
            == expected_family_fingerprint
        )

        family_matches &= (
            int(
                family_raw_counts[
                    class_index
                ]
            )
            == expected_family_raw
        )

    all_device_family_matches &= (
        family_matches
    )

    device_validation_rows.append(
        {
            "device_id":
                device_index,

            "device":
                device,

            "expected_fingerprint_count":
                expected_fingerprint_count,

            "observed_fingerprint_count":
                observed_fingerprint_count,

            "expected_raw_occurrence_count":
                expected_raw_count,

            "observed_raw_occurrence_count":
                observed_raw_count,

            "benign_fingerprint_count":
                int(
                    family_fingerprint_counts[
                        0
                    ]
                ),

            "gafgyt_fingerprint_count":
                int(
                    family_fingerprint_counts[
                        1
                    ]
                ),

            "mirai_fingerprint_count":
                int(
                    family_fingerprint_counts[
                        2
                    ]
                ),

            "benign_raw_occurrence_count":
                int(
                    family_raw_counts[0]
                ),

            "gafgyt_raw_occurrence_count":
                int(
                    family_raw_counts[1]
                ),

            "mirai_raw_occurrence_count":
                int(
                    family_raw_counts[2]
                ),

            "summary_match":
                summary_match,

            "family_match":
                family_matches,
        }
    )


global_occurrence = (
    device_occurrence
    .astype(
        np.uint64
    )
    .sum(
        axis=1
    )
)


fold_rows = []

all_train_classes_present = True
all_validation_classes_present = True
all_test_nonempty = True
all_fold_overlap_zero = True


for device_index, device in enumerate(
    devices
):
    held_out_bit = np.uint16(
        1 << device_index
    )

    test_mask = (
        device_mask
        & held_out_bit
    ) != 0

    nonheldout_mask = ~test_mask

    train_mask = (
        nonheldout_mask
        & (
            split_code
            == SPLIT_CODES["train"]
        )
    )

    validation_mask = (
        nonheldout_mask
        & (
            split_code
            == SPLIT_CODES[
                "validation"
            ]
        )
    )

    unused_mask = (
        nonheldout_mask
        & (
            split_code
            == SPLIT_CODES["test"]
        )
    )

    train_class_counts = np.bincount(
        family3[
            train_mask
        ].astype(
            np.int64
        ),
        minlength=3,
    )

    validation_class_counts = (
        np.bincount(
            family3[
                validation_mask
            ].astype(
                np.int64
            ),
            minlength=3,
        )
    )

    test_class_counts = np.bincount(
        family3[
            test_mask
        ].astype(
            np.int64
        ),
        minlength=3,
    )

    train_all_classes = bool(
        np.all(
            train_class_counts > 0
        )
    )

    validation_all_classes = bool(
        np.all(
            validation_class_counts > 0
        )
    )

    test_present_family_count = int(
        np.sum(
            test_class_counts > 0
        )
    )

    overlap_count = int(
        np.sum(
            train_mask
            & test_mask
        )
        +
        np.sum(
            validation_mask
            & test_mask
        )
    )

    all_train_classes_present &= (
        train_all_classes
    )

    all_validation_classes_present &= (
        validation_all_classes
    )

    all_test_nonempty &= bool(
        np.any(test_mask)
    )

    all_fold_overlap_zero &= (
        overlap_count == 0
    )

    heldout_occurrence = (
        device_occurrence[
            :,
            device_index
        ]
    )

    fold_rows.append(
        {
            "device_id":
                device_index,

            "held_out_device":
                device,

            "train_fingerprint_count":
                int(
                    train_mask.sum()
                ),

            "validation_fingerprint_count":
                int(
                    validation_mask.sum()
                ),

            "test_fingerprint_count":
                int(
                    test_mask.sum()
                ),

            "unused_nonheldout_test_fingerprint_count":
                int(
                    unused_mask.sum()
                ),

            "train_raw_occurrence_count":
                int(
                    global_occurrence[
                        train_mask
                    ].sum(
                        dtype=np.uint64
                    )
                ),

            "validation_raw_occurrence_count":
                int(
                    global_occurrence[
                        validation_mask
                    ].sum(
                        dtype=np.uint64
                    )
                ),

            "heldout_test_raw_occurrence_count":
                int(
                    heldout_occurrence.sum(
                        dtype=np.uint64
                    )
                ),

            "train_benign_count":
                int(
                    train_class_counts[0]
                ),

            "train_gafgyt_count":
                int(
                    train_class_counts[1]
                ),

            "train_mirai_count":
                int(
                    train_class_counts[2]
                ),

            "validation_benign_count":
                int(
                    validation_class_counts[
                        0
                    ]
                ),

            "validation_gafgyt_count":
                int(
                    validation_class_counts[
                        1
                    ]
                ),

            "validation_mirai_count":
                int(
                    validation_class_counts[
                        2
                    ]
                ),

            "test_benign_count":
                int(
                    test_class_counts[0]
                ),

            "test_gafgyt_count":
                int(
                    test_class_counts[1]
                ),

            "test_mirai_count":
                int(
                    test_class_counts[2]
                ),

            "test_present_family_count":
                test_present_family_count,

            "train_all_classes_present":
                train_all_classes,

            "validation_all_classes_present":
                validation_all_classes,

            "fingerprint_overlap_count":
                overlap_count,
        }
    )


validation_checks = {
    "metadata_row_count":
        metadata_row_count
        == EXPECTED_GROUP_COUNT,

    "database_group_count":
        vector_group_count
        == EXPECTED_GROUP_COUNT,

    "database_membership_row_count":
        vector_device_row_count
        == EXPECTED_MEMBERSHIP_COUNT,

    "returned_membership_count":
        returned_memberships
        == EXPECTED_MEMBERSHIP_COUNT,

    "membership_count_from_index":
        membership_count_from_index
        == EXPECTED_MEMBERSHIP_COUNT,

    "raw_record_count_from_index":
        raw_record_count_from_index
        == EXPECTED_RAW_RECORD_COUNT,

    "no_missing_membership_rows":
        missing_membership_rows == 0,

    "source_device_count_matches":
        membership_count_mismatch_rows
        == 0,

    "no_label_conflicts":
        label_conflict_rows == 0,

    "no_family_mismatches":
        family_mismatch_rows == 0,

    "no_unknown_devices":
        unknown_device_rows == 0,

    "all_device_summary_counts_match":
        all_device_summary_matches,

    "all_device_family_counts_match":
        all_device_family_matches,

    "all_train_classes_present":
        all_train_classes_present,

    "all_validation_classes_present":
        all_validation_classes_present,

    "all_tests_nonempty":
        all_test_nonempty,

    "all_fold_fingerprint_overlap_zero":
        all_fold_overlap_zero,
}


all_checks_passed = all(
    validation_checks.values()
)


write_csv(
    DEVICE_VALIDATION_FILE,
    device_validation_rows,
    [
        "device_id",
        "device",
        "expected_fingerprint_count",
        "observed_fingerprint_count",
        "expected_raw_occurrence_count",
        "observed_raw_occurrence_count",
        "benign_fingerprint_count",
        "gafgyt_fingerprint_count",
        "mirai_fingerprint_count",
        "benign_raw_occurrence_count",
        "gafgyt_raw_occurrence_count",
        "mirai_raw_occurrence_count",
        "summary_match",
        "family_match",
    ],
)


write_csv(
    FOLD_FEASIBILITY_FILE,
    fold_rows,
    list(
        fold_rows[0].keys()
    ),
)


alignment = {
    "protocol_version":
        "early_lodo_membership_index_v2_1",

    "metadata_file":
        str(METADATA_FILE),

    "metadata_row_count":
        metadata_row_count,

    "database_file":
        str(MEMBERSHIP_DATABASE),

    "row_alignment":
        (
            "All arrays use the exact row order "
            "of nbaiot_primary_seed2026/"
            "metadata.parquet."
        ),

    "devices": [
        {
            "device_id":
                index,

            "device":
                device,

            "bit_value":
                1 << index,
        }
        for index, device
        in enumerate(devices)
    ],

    "family_mapping": {
        "benign": 0,
        "gafgyt_*": 1,
        "mirai_*": 2,
    },

    "split_mapping":
        SPLIT_CODES,

    "files": {
        "device_mask":
            str(device_mask_file),

        "device_occurrence_counts":
            str(
                device_occurrence_file
            ),

        "family3":
            str(family_file),

        "split_code":
            str(split_file),

        "source_device_count":
            str(
                source_device_count_file
            ),
    },
}


save_json(
    alignment_file,
    alignment,
)


elapsed_seconds = (
    time.perf_counter()
    - started_at
)


summary = {
    "protocol_version":
        "early_lodo_membership_index_v2_1",

    "status":
        (
            "completed"
            if all_checks_passed
            else "failed"
        ),

    "completed_at":
        utc_now(),

    "elapsed_seconds":
        elapsed_seconds,

    "metadata_row_count":
        metadata_row_count,

    "vector_group_count":
        vector_group_count,

    "vector_device_row_count":
        vector_device_row_count,

    "returned_membership_count":
        returned_memberships,

    "membership_count_from_index":
        membership_count_from_index,

    "raw_record_count_from_index":
        raw_record_count_from_index,

    "device_count":
        len(devices),

    "devices":
        devices,

    "missing_membership_rows":
        missing_membership_rows,

    "membership_count_mismatch_rows":
        membership_count_mismatch_rows,

    "label_conflict_rows":
        label_conflict_rows,

    "family_mismatch_rows":
        family_mismatch_rows,

    "unknown_device_rows":
        unknown_device_rows,

    "fold_policy_candidate": {
        "test":
            (
                "Every fingerprint containing "
                "the held-out device."
            ),

        "train":
            (
                "Existing locked train rows after "
                "removing every held-out-device "
                "fingerprint."
            ),

        "validation":
            (
                "Existing locked validation rows "
                "after removing every held-out-device "
                "fingerprint."
            ),

        "unused":
            (
                "Existing non-held-out test rows are "
                "not used by the early LODO pilot."
            ),

        "fingerprint_overlap":
            "zero by construction",
    },

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,

    "outputs": {
        "alignment":
            str(alignment_file),

        "device_validation":
            str(
                DEVICE_VALIDATION_FILE
            ),

        "fold_feasibility":
            str(
                FOLD_FEASIBILITY_FILE
            ),
    },
}


save_json(
    SUMMARY_FILE,
    summary,
)


report_lines = [
    "=" * 78,
    "PHASE 2C EARLY LODO MEMBERSHIP INDEX",
    "=" * 78,
    "",
    (
        f"Metadata groups       : "
        f"{metadata_row_count:,}"
    ),
    (
        f"Membership rows       : "
        f"{membership_count_from_index:,}"
    ),
    (
        f"Raw records           : "
        f"{raw_record_count_from_index:,}"
    ),
    (
        f"Devices               : "
        f"{len(devices)}"
    ),
    (
        f"Elapsed seconds       : "
        f"{elapsed_seconds:.3f}"
    ),
    "",
    "FOLD FEASIBILITY",
]

for row in fold_rows:
    report_lines.append(
        (
            f"- {row['held_out_device']} | "
            f"train="
            f"{row['train_fingerprint_count']:,} | "
            f"validation="
            f"{row['validation_fingerprint_count']:,} | "
            f"test="
            f"{row['test_fingerprint_count']:,} | "
            f"test_families="
            f"{row['test_present_family_count']} | "
            f"overlap="
            f"{row['fingerprint_overlap_count']}"
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
print("=" * 78)
print("PHASE 2C MEMBERSHIP INDEX SUMMARY")
print("=" * 78)
print(
    f"Metadata groups       : "
    f"{metadata_row_count:,}"
)
print(
    f"Membership rows       : "
    f"{membership_count_from_index:,}"
)
print(
    f"Raw records           : "
    f"{raw_record_count_from_index:,}"
)
print(
    f"Missing memberships   : "
    f"{missing_membership_rows:,}"
)
print(
    f"Membership mismatches : "
    f"{membership_count_mismatch_rows:,}"
)
print(
    f"Family mismatches     : "
    f"{family_mismatch_rows:,}"
)

print()
print("FOLD FEASIBILITY")

for row in fold_rows:
    print(
        f"{row['held_out_device']} | "
        f"train="
        f"{row['train_fingerprint_count']:,} | "
        f"val="
        f"{row['validation_fingerprint_count']:,} | "
        f"test="
        f"{row['test_fingerprint_count']:,} | "
        f"families="
        f"{row['test_present_family_count']} | "
        f"overlap="
        f"{row['fingerprint_overlap_count']}"
    )

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"Index root : {OUTPUT_ROOT}")
print(f"Summary    : {SUMMARY_FILE}")
print(
    f"Device CSV : "
    f"{DEVICE_VALIDATION_FILE}"
)
print(
    f"Fold CSV   : "
    f"{FOLD_FEASIBILITY_FILE}"
)
print(f"Report     : {REPORT_FILE}")

connection.close()

if not all_checks_passed:
    print()
    print(
        "PHASE 2C MEMBERSHIP INDEX FAILED"
    )
    sys.exit(1)

print()
print(
    "PHASE 2C MEMBERSHIP INDEX PASSED"
)
