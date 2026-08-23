from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path.cwd()

EXPECTED_PRIMARY_ROWS = 2_482_676
EXPECTED_CANONICAL_ROWS = 2_278_174
EXPECTED_FEATURE_COUNT = 115

SEARCH_ROOTS = [
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "models" / "preprocessing",
    PROJECT_ROOT / "results" / "v2",
]

TARGET_SCRIPTS = [
    PROJECT_ROOT
    / "scripts"
    / "08_materialize_primary_dataset.py",

    PROJECT_ROOT
    / "scripts"
    / "12_materialize_float32_canonical_dataset.py",

    PROJECT_ROOT
    / "scripts"
    / "14_fit_training_standardizer.py",

    PROJECT_ROOT
    / "scripts"
    / "37_build_float32_leakage_ablation_splits.py",

    PROJECT_ROOT
    / "scripts"
    / "38_run_b0_matched_leakage_ablation.py",

    PROJECT_ROOT
    / "scripts"
    / "52_build_numeric_order_pilot_cache_v2.py",

    PROJECT_ROOT
    / "scripts"
    / "53_run_numeric_order_training_pilot_v2.py",
]

OUTPUT_JSON = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_feature_storage_trace_v2.json"
)

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_feature_storage_trace_v2.txt"
)


ALLOWED_SUFFIXES = {
    ".npy",
    ".npz",
    ".parquet",
    ".arrow",
    ".feather",
    ".dat",
    ".bin",
    ".memmap",
    ".pkl",
    ".pickle",
    ".joblib",
    ".csv",
    ".json",
}

RELEVANT_PATH_PATTERN = re.compile(
    r"("
    r"nbaiot|"
    r"primary|"
    r"canonical|"
    r"float32|"
    r"feature|"
    r"numeric|"
    r"pilot|"
    r"leakage"
    r")",
    flags=re.IGNORECASE,
)

CODE_PATTERN = re.compile(
    r"("
    r"FEATURE|"
    r"features|"
    r"feature_file|"
    r"feature_path|"
    r"X_train|"
    r"X_validation|"
    r"X_val|"
    r"X_test|"
    r"open_memmap|"
    r"np\.save|"
    r"np\.load|"
    r"ParquetWriter|"
    r"write_table|"
    r"to_parquet|"
    r"read_parquet|"
    r"metadata_file|"
    r"metadata\.parquet|"
    r"source_relative_path|"
    r"source_row_number|"
    r"primary_seed2026|"
    r"float32_canonical"
    r")",
    flags=re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def is_numeric_arrow_type(
    data_type: pa.DataType,
) -> bool:
    return bool(
        pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_decimal(data_type)
        or pa.types.is_boolean(data_type)
    )


def inspect_npy(
    path: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shape": None,
        "dtype": None,
        "dimension_count": None,
        "row_count": None,
        "column_count": None,
        "is_115_feature_matrix": False,
        "error": None,
    }

    try:
        array = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        shape = tuple(
            int(value)
            for value in array.shape
        )

        result["shape"] = list(shape)
        result["dtype"] = str(array.dtype)
        result["dimension_count"] = len(shape)

        if len(shape) >= 1:
            result["row_count"] = shape[0]

        if len(shape) == 2:
            result["column_count"] = shape[1]

            result[
                "is_115_feature_matrix"
            ] = (
                shape[1]
                == EXPECTED_FEATURE_COUNT
                and np.issubdtype(
                    array.dtype,
                    np.number,
                )
            )

    except Exception as error:
        result["error"] = repr(error)

    return result


def inspect_parquet(
    path: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "row_count": None,
        "column_count": None,
        "numeric_column_count": None,
        "columns": None,
        "numeric_columns": None,
        "is_possible_115_feature_table": False,
        "error": None,
    }

    try:
        parquet_file = pq.ParquetFile(
            path
        )

        schema = (
            parquet_file.schema_arrow
        )

        columns = [
            field.name
            for field in schema
        ]

        numeric_columns = [
            field.name
            for field in schema
            if is_numeric_arrow_type(
                field.type
            )
        ]

        result["row_count"] = int(
            parquet_file
            .metadata
            .num_rows
        )

        result["column_count"] = len(
            columns
        )

        result[
            "numeric_column_count"
        ] = len(
            numeric_columns
        )

        result["columns"] = columns

        result["numeric_columns"] = (
            numeric_columns
        )

        result[
            "is_possible_115_feature_table"
        ] = (
            len(numeric_columns)
            >= EXPECTED_FEATURE_COUNT
        )

    except Exception as error:
        result["error"] = repr(error)

    return result


file_inventory: list[
    dict[str, Any]
] = []


for root in SEARCH_ROOTS:
    if not root.exists():
        continue

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.casefold()

        if suffix not in ALLOWED_SUFFIXES:
            continue

        relative_path = str(
            path.relative_to(
                PROJECT_ROOT
            )
        )

        if not RELEVANT_PATH_PATTERN.search(
            relative_path
        ):
            continue

        item: dict[str, Any] = {
            "relative_path":
                relative_path,

            "parent":
                str(
                    path.parent.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "name":
                path.name,

            "suffix":
                suffix,

            "size_bytes":
                path.stat().st_size,

            "inspection":
                None,
        }

        if suffix == ".npy":
            item["inspection"] = (
                inspect_npy(path)
            )

        elif suffix == ".parquet":
            item["inspection"] = (
                inspect_parquet(path)
            )

        elif suffix == ".npz":
            try:
                with np.load(
                    path,
                    allow_pickle=False,
                ) as archive:
                    item["inspection"] = {
                        "keys":
                            list(
                                archive.files
                            ),

                        "key_count":
                            len(
                                archive.files
                            ),

                        "note":
                            (
                                "Arrays not loaded "
                                "during inventory."
                            ),

                        "error":
                            None,
                    }

            except Exception as error:
                item["inspection"] = {
                    "error":
                        repr(error)
                }

        file_inventory.append(item)


file_inventory.sort(
    key=lambda row: (
        row["parent"],
        row["name"],
    )
)


feature_candidates = []

for item in file_inventory:
    inspection = (
        item.get("inspection")
        or {}
    )

    is_candidate = bool(
        inspection.get(
            "is_115_feature_matrix"
        )
        or inspection.get(
            "is_possible_115_feature_table"
        )
    )

    if is_candidate:
        feature_candidates.append(item)


partition_groups: list[
    dict[str, Any]
] = []

groups: dict[
    str,
    list[dict[str, Any]]
] = defaultdict(list)

for item in feature_candidates:
    groups[
        item["parent"]
    ].append(item)


for parent, items in sorted(
    groups.items()
):
    row_counts = []

    for item in items:
        inspection = (
            item.get("inspection")
            or {}
        )

        row_count = inspection.get(
            "row_count"
        )

        if row_count is not None:
            row_counts.append(
                int(row_count)
            )

    total_rows = sum(row_counts)

    names = [
        item["name"]
        for item in items
    ]

    lower_names = " ".join(
        names
    ).casefold()

    has_train_name = (
        "train"
        in lower_names
    )

    has_validation_name = (
        "validation"
        in lower_names
        or "val"
        in lower_names
    )

    has_test_name = (
        "test"
        in lower_names
    )

    partition_groups.append(
        {
            "parent":
                parent,

            "candidate_file_count":
                len(items),

            "candidate_files":
                names,

            "row_counts":
                row_counts,

            "total_rows":
                total_rows,

            "matches_primary_total":
                total_rows
                == EXPECTED_PRIMARY_ROWS,

            "matches_canonical_total":
                total_rows
                == EXPECTED_CANONICAL_ROWS,

            "looks_like_split_set":
                (
                    has_train_name
                    and has_validation_name
                    and has_test_name
                ),
        }
    )


code_hints = []

for script_path in TARGET_SCRIPTS:
    if not script_path.exists():
        code_hints.append(
            {
                "relative_path":
                    str(
                        script_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "exists":
                    False,

                "match_count":
                    0,

                "context":
                    [],
            }
        )

        continue

    lines = script_path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    ).splitlines()

    matching_indexes = [
        index
        for index, line in enumerate(
            lines
        )
        if CODE_PATTERN.search(line)
    ]

    selected_indexes: set[int] = set()

    for index in matching_indexes:
        selected_indexes.update(
            range(
                max(0, index - 3),
                min(
                    len(lines),
                    index + 4,
                ),
            )
        )

    context = []
    previous_index: int | None = None

    for index in sorted(
        selected_indexes
    ):
        if (
            previous_index is not None
            and index
            > previous_index + 1
        ):
            context.append(
                "... omitted ..."
            )

        context.append(
            f"{index + 1:05d}: "
            f"{lines[index]}"
        )

        previous_index = index

    code_hints.append(
        {
            "relative_path":
                str(
                    script_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "exists":
                True,

            "match_count":
                len(matching_indexes),

            "context":
                context[:300],
        }
    )


primary_directory = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_primary_seed2026"
)

primary_directory_files = []

if primary_directory.exists():
    for path in sorted(
        primary_directory.rglob("*")
    ):
        if not path.is_file():
            continue

        primary_directory_files.append(
            {
                "relative_path":
                    str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "size_bytes":
                    path.stat().st_size,

                "suffix":
                    path.suffix.casefold(),
            }
        )


summary = {
    "protocol_version":
        "early_lodo_feature_storage_trace_v2_1",

    "status":
        "completed",

    "completed_at":
        utc_now(),

    "expected": {
        "primary_rows":
            EXPECTED_PRIMARY_ROWS,

        "canonical_rows":
            EXPECTED_CANONICAL_ROWS,

        "feature_count":
            EXPECTED_FEATURE_COUNT,
    },

    "primary_directory_files":
        primary_directory_files,

    "file_inventory":
        file_inventory,

    "feature_candidates":
        feature_candidates,

    "partition_groups":
        partition_groups,

    "code_hints":
        code_hints,
}


OUTPUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


report = [
    "=" * 86,
    "PHASE 2E-1 EARLY LODO FEATURE STORAGE TRACE",
    "=" * 86,
    "",
    "PRIMARY DATASET DIRECTORY",
]

if primary_directory_files:
    for item in primary_directory_files:
        report.append(
            (
                f"- {item['relative_path']} | "
                f"{item['size_bytes']} bytes"
            )
        )
else:
    report.append(
        "- Directory not found or empty."
    )


report.extend(
    [
        "",
        "FEATURE CANDIDATES",
    ]
)

if feature_candidates:
    for item in feature_candidates:
        inspection = (
            item.get("inspection")
            or {}
        )

        report.append(
            (
                f"- {item['relative_path']} | "
                f"size="
                f"{item['size_bytes']} | "
                f"rows="
                f"{inspection.get('row_count')} | "
                f"columns="
                f"{inspection.get('column_count')} | "
                f"numeric_columns="
                f"{inspection.get('numeric_column_count')} | "
                f"shape="
                f"{inspection.get('shape')} | "
                f"dtype="
                f"{inspection.get('dtype')}"
            )
        )
else:
    report.append(
        "- No 115-feature candidate detected."
    )


report.extend(
    [
        "",
        "PARTITION GROUPS",
    ]
)

if partition_groups:
    for group in partition_groups:
        report.append(
            (
                f"- {group['parent']} | "
                f"files="
                f"{group['candidate_file_count']} | "
                f"rows="
                f"{group['row_counts']} | "
                f"total="
                f"{group['total_rows']} | "
                f"primary_match="
                f"{group['matches_primary_total']} | "
                f"canonical_match="
                f"{group['matches_canonical_total']} | "
                f"split_set="
                f"{group['looks_like_split_set']}"
            )
        )

        for filename in group[
            "candidate_files"
        ]:
            report.append(
                f"  - {filename}"
            )

else:
    report.append(
        "- No candidate partition groups."
    )


report.extend(
    [
        "",
        "CODE HINTS",
    ]
)

for script in code_hints:
    report.append(
        (
            f"--- {script['relative_path']} | "
            f"exists={script['exists']} | "
            f"matches={script['match_count']} ---"
        )
    )

    report.extend(
        script["context"]
    )


OUTPUT_REPORT.write_text(
    "\n".join(report),
    encoding="utf-8",
)


print("=" * 86)
print("PHASE 2E-1 EARLY LODO FEATURE STORAGE TRACE")
print("=" * 86)

print()
print("PRIMARY DATASET DIRECTORY")

if primary_directory_files:
    for item in primary_directory_files:
        print(
            f"{item['relative_path']} | "
            f"size_gb="
            f"{item['size_bytes'] / (1024 ** 3):.3f}"
        )
else:
    print(
        "Directory not found or empty."
    )

print()
print("FEATURE CANDIDATES")

if feature_candidates:
    for item in feature_candidates:
        inspection = (
            item.get("inspection")
            or {}
        )

        print(
            f"{item['relative_path']} | "
            f"rows="
            f"{inspection.get('row_count')} | "
            f"columns="
            f"{inspection.get('column_count')} | "
            f"numeric="
            f"{inspection.get('numeric_column_count')} | "
            f"shape="
            f"{inspection.get('shape')} | "
            f"dtype="
            f"{inspection.get('dtype')}"
        )
else:
    print(
        "No 115-feature candidate detected."
    )

print()
print("PARTITION GROUPS")

if partition_groups:
    for group in partition_groups:
        print(
            f"{group['parent']} | "
            f"files="
            f"{group['candidate_file_count']} | "
            f"total_rows="
            f"{group['total_rows']} | "
            f"primary_match="
            f"{group['matches_primary_total']} | "
            f"canonical_match="
            f"{group['matches_canonical_total']} | "
            f"split_set="
            f"{group['looks_like_split_set']}"
        )
else:
    print(
        "No candidate partition groups."
    )

print()
print("CODE HINTS")

for script in code_hints:
    print(
        f"{script['relative_path']} | "
        f"exists={script['exists']} | "
        f"matches={script['match_count']}"
    )

print()
print(f"JSON   : {OUTPUT_JSON}")
print(f"Report : {OUTPUT_REPORT}")
print()
print(
    "PHASE 2E-1 FEATURE STORAGE "
    "TRACE COMPLETE"
)
