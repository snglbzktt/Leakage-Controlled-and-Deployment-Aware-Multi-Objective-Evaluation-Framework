from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path.cwd()

EXPECTED_FEATURE_COUNT = 115
EXPECTED_GLOBAL_ROWS = 2_482_676

FEATURE_ALIGNMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_early_lodo_feature_alignment_v2"
    / "alignment.json"
)

MEMBERSHIP_ALIGNMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_early_lodo_membership_v2"
    / "alignment.json"
)

MEMBERSHIP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_early_lodo_membership_v2"
)

DEVICE_MASK_FILE = (
    MEMBERSHIP_ROOT
    / "device_mask.npy"
)

FAMILY_FILE = (
    MEMBERSHIP_ROOT
    / "family3.npy"
)

DEVICE_OCCURRENCE_FILE = (
    MEMBERSHIP_ROOT
    / "device_occurrence_counts.npy"
)

FOLD_FEASIBILITY_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_fold_feasibility_v2.csv"
)

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_protocol_locked_v2.json"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "early_lodo_fold_cache_v2"
)

AUDIT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

CLASS_NAMES = [
    "benign",
    "gafgyt",
    "mirai",
]

SPLIT_SOURCE_CODES = {
    "train": 0,
    "validation": 1,
    "test": 2,
}

PARQUET_BATCH_SIZE = 50_000
SCALER_BATCH_SIZE = 100_000


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def resolve_path(
    value: str,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def safe_name(
    value: str,
) -> str:
    result = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    return result.strip("_")


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig",
        )
    )


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


def open_output_memmap(
    path: Path,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> tuple[Path, np.memmap]:
    partial = path.with_name(
        path.stem
        + ".partial"
        + path.suffix
    )

    if partial.exists():
        partial.unlink()

    array = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )

    return partial, array


def validate_npy(
    path: Path,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype,
) -> bool:
    if not path.exists():
        return False

    try:
        value = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        return (
            tuple(value.shape)
            == tuple(expected_shape)
            and value.dtype
            == np.dtype(expected_dtype)
        )

    except Exception:
        return False


parser = argparse.ArgumentParser()

parser.add_argument(
    "--device",
    default="Danmini_Doorbell",
)

parser.add_argument(
    "--rebuild",
    action="store_true",
)

args = parser.parse_args()

held_out_device = str(
    args.device
)

cache_device_name = safe_name(
    held_out_device
)

cache_directory = (
    CACHE_ROOT
    / cache_device_name
)

summary_file = (
    cache_directory
    / "cache_summary.json"
)

report_file = (
    AUDIT_ROOT
    / (
        "early_lodo_fold_cache_"
        + cache_device_name
        + "_v2.txt"
    )
)


required_files = [
    FEATURE_ALIGNMENT_FILE,
    MEMBERSHIP_ALIGNMENT_FILE,
    DEVICE_MASK_FILE,
    FAMILY_FILE,
    DEVICE_OCCURRENCE_FILE,
    FOLD_FEASIBILITY_FILE,
    PROTOCOL_FILE,
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


feature_alignment = load_json(
    FEATURE_ALIGNMENT_FILE
)

membership_alignment = load_json(
    MEMBERSHIP_ALIGNMENT_FILE
)

protocol = load_json(
    PROTOCOL_FILE
)

fold_rows = load_csv(
    FOLD_FEASIBILITY_FILE
)


if protocol.get("status") != "locked":
    raise RuntimeError(
        "Early LODO protocol is not locked."
    )

if protocol.get(
    "all_checks_passed"
) is not True:
    raise RuntimeError(
        "Early LODO protocol checks failed."
    )


device_entries = [
    row
    for row
    in membership_alignment[
        "devices"
    ]
    if row["device"]
    == held_out_device
]

if len(device_entries) != 1:
    raise RuntimeError(
        "Held-out device was not found "
        "exactly once."
    )

device_entry = device_entries[0]

device_id = int(
    device_entry["device_id"]
)

held_out_bit = np.uint16(
    int(device_entry["bit_value"])
)


matching_fold_rows = [
    row
    for row in fold_rows
    if row["held_out_device"]
    == held_out_device
]

if len(matching_fold_rows) != 1:
    raise RuntimeError(
        "Fold feasibility row was not "
        "found exactly once."
    )

fold_row = matching_fold_rows[0]


expected_counts = {
    "train":
        int(
            fold_row[
                "train_fingerprint_count"
            ]
        ),

    "validation":
        int(
            fold_row[
                "validation_fingerprint_count"
            ]
        ),

    "test":
        int(
            fold_row[
                "test_fingerprint_count"
            ]
        ),
}

expected_class_counts = {
    "train": [
        int(
            fold_row[
                "train_benign_count"
            ]
        ),
        int(
            fold_row[
                "train_gafgyt_count"
            ]
        ),
        int(
            fold_row[
                "train_mirai_count"
            ]
        ),
    ],

    "validation": [
        int(
            fold_row[
                "validation_benign_count"
            ]
        ),
        int(
            fold_row[
                "validation_gafgyt_count"
            ]
        ),
        int(
            fold_row[
                "validation_mirai_count"
            ]
        ),
    ],

    "test": [
        int(
            fold_row[
                "test_benign_count"
            ]
        ),
        int(
            fold_row[
                "test_gafgyt_count"
            ]
        ),
        int(
            fold_row[
                "test_mirai_count"
            ]
        ),
    ],
}

expected_test_occurrence_count = int(
    fold_row[
        "heldout_test_raw_occurrence_count"
    ]
)


feature_columns = list(
    feature_alignment[
        "feature_columns"
    ]
)

if len(feature_columns) != EXPECTED_FEATURE_COUNT:
    raise RuntimeError(
        "Expected 115 feature columns."
    )


cache_files = {
    "x_train":
        cache_directory
        / "x_train.npy",

    "y_train":
        cache_directory
        / "y_train.npy",

    "train_global_indices":
        cache_directory
        / "train_global_indices.npy",

    "x_validation":
        cache_directory
        / "x_validation.npy",

    "y_validation":
        cache_directory
        / "y_validation.npy",

    "validation_global_indices":
        cache_directory
        / "validation_global_indices.npy",

    "x_test":
        cache_directory
        / "x_test.npy",

    "y_test":
        cache_directory
        / "y_test.npy",

    "test_global_indices":
        cache_directory
        / "test_global_indices.npy",

    "test_occurrence_count":
        cache_directory
        / "test_occurrence_count.npy",

    "test_source_split":
        cache_directory
        / "test_source_split.npy",

    "scaler":
        cache_directory
        / "scaler.npz",
}


cache_specs = {
    "x_train": (
        (
            expected_counts["train"],
            EXPECTED_FEATURE_COUNT,
        ),
        np.float32,
    ),

    "y_train": (
        (
            expected_counts["train"],
        ),
        np.int8,
    ),

    "train_global_indices": (
        (
            expected_counts["train"],
        ),
        np.int64,
    ),

    "x_validation": (
        (
            expected_counts[
                "validation"
            ],
            EXPECTED_FEATURE_COUNT,
        ),
        np.float32,
    ),

    "y_validation": (
        (
            expected_counts[
                "validation"
            ],
        ),
        np.int8,
    ),

    "validation_global_indices": (
        (
            expected_counts[
                "validation"
            ],
        ),
        np.int64,
    ),

    "x_test": (
        (
            expected_counts["test"],
            EXPECTED_FEATURE_COUNT,
        ),
        np.float32,
    ),

    "y_test": (
        (
            expected_counts["test"],
        ),
        np.int8,
    ),

    "test_global_indices": (
        (
            expected_counts["test"],
        ),
        np.int64,
    ),

    "test_occurrence_count": (
        (
            expected_counts["test"],
        ),
        np.uint32,
    ),

    "test_source_split": (
        (
            expected_counts["test"],
        ),
        np.uint8,
    ),
}


existing_cache_valid = (
    not args.rebuild
    and summary_file.exists()
    and cache_files["scaler"].exists()
    and all(
        validate_npy(
            cache_files[name],
            shape,
            dtype,
        )
        for name, (
            shape,
            dtype,
        ) in cache_specs.items()
    )
)

if existing_cache_valid:
    existing_summary = load_json(
        summary_file
    )

    if existing_summary.get(
        "all_checks_passed"
    ) is True:
        print(
            "Existing validated fold cache "
            "was found."
        )
        print(
            "PHASE 2G-0 FOLD CACHE PASSED"
        )
        sys.exit(0)


estimated_array_bytes = (
    expected_counts["train"]
    * EXPECTED_FEATURE_COUNT
    * 4
    + expected_counts["validation"]
    * EXPECTED_FEATURE_COUNT
    * 4
    + expected_counts["test"]
    * EXPECTED_FEATURE_COUNT
    * 4
    + (
        expected_counts["train"]
        + expected_counts["validation"]
        + expected_counts["test"]
    )
    * (
        1
        + 8
    )
    + expected_counts["test"]
    * (
        4
        + 1
    )
)

free_disk_bytes = shutil.disk_usage(
    PROJECT_ROOT
).free

minimum_required_bytes = int(
    estimated_array_bytes
    * 1.20
    + 512
    * 1024
    * 1024
)

if free_disk_bytes < minimum_required_bytes:
    raise RuntimeError(
        "Not enough free disk space. "
        f"Required approximately "
        f"{minimum_required_bytes / (1024 ** 3):.3f} GB, "
        f"available "
        f"{free_disk_bytes / (1024 ** 3):.3f} GB."
    )


cache_directory.mkdir(
    parents=True,
    exist_ok=True,
)

partial_files: dict[str, Path] = {}
output_arrays: dict[str, np.memmap] = {}

for name, (
    shape,
    dtype,
) in cache_specs.items():
    partial, array = open_output_memmap(
        cache_files[name],
        dtype,
        shape,
    )

    partial_files[name] = partial
    output_arrays[name] = array


device_mask = np.load(
    DEVICE_MASK_FILE,
    mmap_mode="r",
    allow_pickle=False,
)

family3 = np.load(
    FAMILY_FILE,
    mmap_mode="r",
    allow_pickle=False,
)

device_occurrence = np.load(
    DEVICE_OCCURRENCE_FILE,
    mmap_mode="r",
    allow_pickle=False,
)

if device_mask.shape != (
    EXPECTED_GLOBAL_ROWS,
):
    raise RuntimeError(
        "device_mask shape mismatch."
    )

if family3.shape != (
    EXPECTED_GLOBAL_ROWS,
):
    raise RuntimeError(
        "family3 shape mismatch."
    )

if device_occurrence.shape != (
    EXPECTED_GLOBAL_ROWS,
    9,
):
    raise RuntimeError(
        "device_occurrence shape mismatch."
    )


write_positions = {
    "train": 0,
    "validation": 0,
    "test": 0,
}

observed_class_counts = {
    "train":
        np.zeros(
            3,
            dtype=np.int64,
        ),

    "validation":
        np.zeros(
            3,
            dtype=np.int64,
        ),

    "test":
        np.zeros(
            3,
            dtype=np.int64,
        ),
}

test_source_distribution = Counter()

all_features_finite = True

started_at = time.perf_counter()


print("=" * 82)
print("PHASE 2G-0 BUILD EARLY LODO FOLD CACHE")
print("=" * 82)
print(
    f"Held-out device : "
    f"{held_out_device}"
)
print(
    f"Device id       : "
    f"{device_id}"
)
print(
    f"Train rows      : "
    f"{expected_counts['train']:,}"
)
print(
    f"Validation rows : "
    f"{expected_counts['validation']:,}"
)
print(
    f"Test rows       : "
    f"{expected_counts['test']:,}"
)
print(
    f"Estimated cache : "
    f"{estimated_array_bytes / (1024 ** 3):.3f} GB"
)
print(
    f"Free disk       : "
    f"{free_disk_bytes / (1024 ** 3):.3f} GB"
)
print()


for source_split in (
    "train",
    "validation",
    "test",
):
    split_entry = (
        feature_alignment[
            "splits"
        ][source_split]
    )

    feature_file = resolve_path(
        split_entry["feature_file"]
    )

    global_index_file = resolve_path(
        split_entry[
            "global_index_file"
        ]
    )

    global_indices = np.load(
        global_index_file,
        mmap_mode="r",
        allow_pickle=False,
    )

    parquet_file = pq.ParquetFile(
        feature_file
    )

    local_offset = 0

    print(
        f"Scanning {source_split}: "
        f"{len(global_indices):,} rows"
    )

    for batch in parquet_file.iter_batches(
        batch_size=PARQUET_BATCH_SIZE,
        columns=feature_columns,
    ):
        batch_rows = int(
            batch.num_rows
        )

        batch_global_indices = (
            global_indices[
                local_offset:
                local_offset
                + batch_rows
            ]
        )

        batch_device_mask = (
            device_mask[
                batch_global_indices
            ]
        )

        held_out_membership = (
            (
                batch_device_mask
                & held_out_bit
            )
            != 0
        )

        features = (
            batch
            .to_pandas()
            .to_numpy(
                dtype=np.float32,
                copy=False,
            )
        )

        if not np.isfinite(
            features
        ).all():
            all_features_finite = False

        labels = family3[
            batch_global_indices
        ].astype(
            np.int8,
            copy=False,
        )

        if source_split == "train":
            train_positions = np.flatnonzero(
                ~held_out_membership
            )

            count = len(
                train_positions
            )

            start = write_positions[
                "train"
            ]

            end = start + count

            output_arrays[
                "x_train"
            ][start:end] = features[
                train_positions
            ]

            output_arrays[
                "y_train"
            ][start:end] = labels[
                train_positions
            ]

            output_arrays[
                "train_global_indices"
            ][start:end] = (
                batch_global_indices[
                    train_positions
                ]
            )

            observed_class_counts[
                "train"
            ] += np.bincount(
                labels[
                    train_positions
                ].astype(
                    np.int64
                ),
                minlength=3,
            )

            write_positions[
                "train"
            ] = end

        elif source_split == "validation":
            validation_positions = (
                np.flatnonzero(
                    ~held_out_membership
                )
            )

            count = len(
                validation_positions
            )

            start = write_positions[
                "validation"
            ]

            end = start + count

            output_arrays[
                "x_validation"
            ][start:end] = features[
                validation_positions
            ]

            output_arrays[
                "y_validation"
            ][start:end] = labels[
                validation_positions
            ]

            output_arrays[
                "validation_global_indices"
            ][start:end] = (
                batch_global_indices[
                    validation_positions
                ]
            )

            observed_class_counts[
                "validation"
            ] += np.bincount(
                labels[
                    validation_positions
                ].astype(
                    np.int64
                ),
                minlength=3,
            )

            write_positions[
                "validation"
            ] = end

        test_positions = np.flatnonzero(
            held_out_membership
        )

        test_count = len(
            test_positions
        )

        if test_count > 0:
            start = write_positions[
                "test"
            ]

            end = start + test_count

            test_global_indices = (
                batch_global_indices[
                    test_positions
                ]
            )

            output_arrays[
                "x_test"
            ][start:end] = features[
                test_positions
            ]

            output_arrays[
                "y_test"
            ][start:end] = labels[
                test_positions
            ]

            output_arrays[
                "test_global_indices"
            ][start:end] = (
                test_global_indices
            )

            output_arrays[
                "test_occurrence_count"
            ][start:end] = (
                device_occurrence[
                    test_global_indices,
                    device_id,
                ]
            )

            output_arrays[
                "test_source_split"
            ][start:end] = np.uint8(
                SPLIT_SOURCE_CODES[
                    source_split
                ]
            )

            observed_class_counts[
                "test"
            ] += np.bincount(
                labels[
                    test_positions
                ].astype(
                    np.int64
                ),
                minlength=3,
            )

            test_source_distribution[
                source_split
            ] += test_count

            write_positions[
                "test"
            ] = end

        local_offset += batch_rows

    if local_offset != len(
        global_indices
    ):
        raise RuntimeError(
            f"{source_split} alignment failed."
        )

    print(
        f"Completed {source_split}: "
        f"{local_offset:,} rows"
    )


for array in output_arrays.values():
    array.flush()

    mmap_handle = getattr(
        array,
        "_mmap",
        None,
    )

    if mmap_handle is not None:
        mmap_handle.close()

del array
output_arrays.clear()
del output_arrays

gc.collect()


for name, final_path in (
    cache_files.items()
):
    if name == "scaler":
        continue

    os.replace(
        partial_files[name],
        final_path,
    )


x_train = np.load(
    cache_files["x_train"],
    mmap_mode="r",
    allow_pickle=False,
)

y_train = np.load(
    cache_files["y_train"],
    mmap_mode="r",
    allow_pickle=False,
)

train_global_indices = np.load(
    cache_files[
        "train_global_indices"
    ],
    mmap_mode="r",
    allow_pickle=False,
)

x_validation = np.load(
    cache_files["x_validation"],
    mmap_mode="r",
    allow_pickle=False,
)

y_validation = np.load(
    cache_files["y_validation"],
    mmap_mode="r",
    allow_pickle=False,
)

validation_global_indices = np.load(
    cache_files[
        "validation_global_indices"
    ],
    mmap_mode="r",
    allow_pickle=False,
)

x_test = np.load(
    cache_files["x_test"],
    mmap_mode="r",
    allow_pickle=False,
)

y_test = np.load(
    cache_files["y_test"],
    mmap_mode="r",
    allow_pickle=False,
)

test_global_indices = np.load(
    cache_files[
        "test_global_indices"
    ],
    mmap_mode="r",
    allow_pickle=False,
)

test_occurrence_count = np.load(
    cache_files[
        "test_occurrence_count"
    ],
    mmap_mode="r",
    allow_pickle=False,
)

test_source_split = np.load(
    cache_files[
        "test_source_split"
    ],
    mmap_mode="r",
    allow_pickle=False,
)


print()
print("Fitting fold-only StandardScaler...")

scaler = StandardScaler(
    copy=True
)

for start in range(
    0,
    len(x_train),
    SCALER_BATCH_SIZE,
):
    end = min(
        len(x_train),
        start
        + SCALER_BATCH_SIZE,
    )

    scaler.partial_fit(
        x_train[
            start:end
        ].astype(
            np.float64,
            copy=False,
        )
    )

    if (
        start == 0
        or end == len(x_train)
        or (
            start
            // SCALER_BATCH_SIZE
        )
        % 5
        == 0
    ):
        print(
            f"Scaler rows: "
            f"{end:,}/"
            f"{len(x_train):,}"
        )


scaler_partial = (
    cache_directory
    / "scaler.partial.npz"
)

if scaler_partial.exists():
    scaler_partial.unlink()

np.savez(
    scaler_partial,
    mean=scaler.mean_,
    scale=scaler.scale_,
    var=scaler.var_,
    n_features_in=np.asarray(
        scaler.n_features_in_,
        dtype=np.int64,
    ),
    n_samples_seen=np.asarray(
        scaler.n_samples_seen_,
    ),
)

os.replace(
    scaler_partial,
    cache_files["scaler"],
)


coverage = np.zeros(
    EXPECTED_GLOBAL_ROWS,
    dtype=np.uint8,
)

train_duplicate_count = int(
    np.count_nonzero(
        coverage[
            train_global_indices
        ]
        != 0
    )
)

coverage[
    train_global_indices
] += 1

validation_overlap_count = int(
    np.count_nonzero(
        coverage[
            validation_global_indices
        ]
        != 0
    )
)

coverage[
    validation_global_indices
] += 1

test_overlap_count = int(
    np.count_nonzero(
        coverage[
            test_global_indices
        ]
        != 0
    )
)

coverage[
    test_global_indices
] += 1


observed_test_occurrence_count = int(
    test_occurrence_count.sum(
        dtype=np.uint64
    )
)

observed_class_counts_final = {
    "train":
        np.bincount(
            y_train.astype(
                np.int64
            ),
            minlength=3,
        ).tolist(),

    "validation":
        np.bincount(
            y_validation.astype(
                np.int64
            ),
            minlength=3,
        ).tolist(),

    "test":
        np.bincount(
            y_test.astype(
                np.int64
            ),
            minlength=3,
        ).tolist(),
}


validation_checks = {
    "write_counts_match":
        write_positions
        == expected_counts,

    "train_shape_matches":
        x_train.shape
        == (
            expected_counts["train"],
            EXPECTED_FEATURE_COUNT,
        ),

    "validation_shape_matches":
        x_validation.shape
        == (
            expected_counts[
                "validation"
            ],
            EXPECTED_FEATURE_COUNT,
        ),

    "test_shape_matches":
        x_test.shape
        == (
            expected_counts["test"],
            EXPECTED_FEATURE_COUNT,
        ),

    "all_features_finite":
        all_features_finite,

    "class_counts_match":
        all(
            observed_class_counts_final[
                split
            ]
            == expected_class_counts[
                split
            ]
            for split in (
                "train",
                "validation",
                "test",
            )
        ),

    "stream_class_counts_match":
        all(
            observed_class_counts[
                split
            ].tolist()
            == expected_class_counts[
                split
            ]
            for split in (
                "train",
                "validation",
                "test",
            )
        ),

    "train_excludes_heldout":
        bool(
            np.all(
                (
                    device_mask[
                        train_global_indices
                    ]
                    & held_out_bit
                )
                == 0
            )
        ),

    "validation_excludes_heldout":
        bool(
            np.all(
                (
                    device_mask[
                        validation_global_indices
                    ]
                    & held_out_bit
                )
                == 0
            )
        ),

    "test_contains_heldout":
        bool(
            np.all(
                (
                    device_mask[
                        test_global_indices
                    ]
                    & held_out_bit
                )
                != 0
            )
        ),

    "train_internal_duplicates_zero":
        train_duplicate_count
        == 0,

    "train_validation_overlap_zero":
        validation_overlap_count
        == 0,

    "train_validation_test_overlap_zero":
        test_overlap_count
        == 0,

    "test_occurrences_positive":
        bool(
            np.all(
                test_occurrence_count > 0
            )
        ),

    "test_occurrence_total_matches":
        observed_test_occurrence_count
        == expected_test_occurrence_count,

    "scaler_feature_count_115":
        int(
            scaler.n_features_in_
        )
        == EXPECTED_FEATURE_COUNT,

    "scaler_sample_count_matches":
        int(
            np.asarray(
                scaler.n_samples_seen_
            ).max()
        )
        == expected_counts["train"],

    "scaler_values_finite":
        bool(
            np.isfinite(
                scaler.mean_
            ).all()
            and np.isfinite(
                scaler.scale_
            ).all()
            and np.isfinite(
                scaler.var_
            ).all()
        ),

    "scaler_scales_positive":
        bool(
            np.all(
                scaler.scale_ > 0
            )
        ),

    "all_cache_files_exist":
        all(
            path.exists()
            for path
            in cache_files.values()
        ),
}


all_checks_passed = all(
    validation_checks.values()
)

elapsed_seconds = (
    time.perf_counter()
    - started_at
)


summary = {
    "protocol_version":
        "early_lodo_fold_cache_v2_1",

    "status":
        (
            "validated"
            if all_checks_passed
            else "failed"
        ),

    "completed_at":
        utc_now(),

    "elapsed_seconds":
        elapsed_seconds,

    "held_out_device":
        held_out_device,

    "device_id":
        device_id,

    "held_out_bit":
        int(held_out_bit),

    "pipeline":
        "pipeline_a",

    "feature_policy": (
        "Primary float64 parquet features "
        "are converted to canonical float32 "
        "before fold-only scaling."
    ),

    "expected_counts":
        expected_counts,

    "observed_counts":
        write_positions,

    "expected_class_counts":
        expected_class_counts,

    "observed_class_counts":
        observed_class_counts_final,

    "test_source_distribution":
        dict(
            test_source_distribution
        ),

    "expected_test_occurrence_count":
        expected_test_occurrence_count,

    "observed_test_occurrence_count":
        observed_test_occurrence_count,

    "scaler": {
        "fit_rows":
            expected_counts["train"],

        "feature_count":
            int(
                scaler.n_features_in_
            ),

        "maximum_absolute_mean":
            float(
                np.max(
                    np.abs(
                        scaler.mean_
                    )
                )
            ),

        "minimum_scale":
            float(
                np.min(
                    scaler.scale_
                )
            ),

        "maximum_scale":
            float(
                np.max(
                    scaler.scale_
                )
            ),
    },

    "estimated_cache_bytes":
        estimated_array_bytes,

    "cache_files": {
        name:
            str(path)
        for name, path
        in cache_files.items()
    },

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,
}


save_json(
    summary_file,
    summary,
)


report_lines = [
    "=" * 82,
    "PHASE 2G-0 EARLY LODO FOLD CACHE",
    "=" * 82,
    "",
    (
        f"Held-out device : "
        f"{held_out_device}"
    ),
    (
        f"Elapsed seconds : "
        f"{elapsed_seconds:.3f}"
    ),
    "",
    "CACHE COUNTS",
    (
        f"Train           : "
        f"{write_positions['train']:,}"
    ),
    (
        f"Validation      : "
        f"{write_positions['validation']:,}"
    ),
    (
        f"Test            : "
        f"{write_positions['test']:,}"
    ),
    (
        f"Test occurrence : "
        f"{observed_test_occurrence_count:,}"
    ),
    "",
    "CLASS COUNTS",
]

for split in (
    "train",
    "validation",
    "test",
):
    report_lines.append(
        (
            f"{split}: "
            f"{observed_class_counts_final[split]}"
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


report_file.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


print()
print("=" * 82)
print("PHASE 2G-0 FOLD CACHE SUMMARY")
print("=" * 82)
print(
    f"Held-out device : "
    f"{held_out_device}"
)
print(
    f"Train rows      : "
    f"{write_positions['train']:,}"
)
print(
    f"Validation rows : "
    f"{write_positions['validation']:,}"
)
print(
    f"Test rows       : "
    f"{write_positions['test']:,}"
)
print(
    f"Test occurrence : "
    f"{observed_test_occurrence_count:,}"
)
print(
    f"Elapsed seconds : "
    f"{elapsed_seconds:.3f}"
)

print()
print("CLASS COUNTS")

for split in (
    "train",
    "validation",
    "test",
):
    print(
        f"{split}: "
        f"{observed_class_counts_final[split]}"
    )

print()
print("TEST SOURCE DISTRIBUTION")

for split in (
    "train",
    "validation",
    "test",
):
    print(
        f"{split}: "
        f"{test_source_distribution[split]}"
    )

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"Cache   : {cache_directory}")
print(f"Summary : {summary_file}")
print(f"Report  : {report_file}")

if not all_checks_passed:
    print()
    print(
        "PHASE 2G-0 FOLD CACHE FAILED"
    )
    sys.exit(1)

print()
print(
    "PHASE 2G-0 FOLD CACHE PASSED"
)
