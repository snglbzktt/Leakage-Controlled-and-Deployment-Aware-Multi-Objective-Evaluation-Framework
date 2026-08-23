from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import sklearn
import torch

try:
    import psutil
except ImportError:
    psutil = None


PROJECT_ROOT = Path.cwd()

EXPECTED_ROW_COUNT = 2_482_676
EXPECTED_FEATURE_COUNT = 115
EXPECTED_DEVICE_COUNT = 9
EXPECTED_RUN_COUNT = 27
EXPECTED_SEED = 2026

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_protocol_locked_v2.json"
)

RUN_MATRIX_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_run_matrix_v2.csv"
)

ALIGNMENT_FILE = (
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

PRIMARY_METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_primary_seed2026"
    / "metadata.parquet"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_training_preflight_v2.json"
)

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_training_preflight_v2.txt"
)


MEMBERSHIP_ARRAY_SPECS = {
    "device_mask.npy": {
        "shape":
            (
                EXPECTED_ROW_COUNT,
            ),

        "dtype":
            "uint16",
    },

    "device_occurrence_counts.npy": {
        "shape":
            (
                EXPECTED_ROW_COUNT,
                EXPECTED_DEVICE_COUNT,
            ),

        "dtype":
            "uint32",
    },

    "family3.npy": {
        "shape":
            (
                EXPECTED_ROW_COUNT,
            ),

        "dtype":
            "int8",
    },

    "split_code.npy": {
        "shape":
            (
                EXPECTED_ROW_COUNT,
            ),

        "dtype":
            "uint8",
    },

    "source_device_count.npy": {
        "shape":
            (
                EXPECTED_ROW_COUNT,
            ),

        "dtype":
            "uint8",
    },
}


CODE_SEARCH_TERMS = [
    "tinyml_mlp",
    "compact_dnn",
    "build_model",
    "model_registry",
    "StandardScaler",
    "np.load",
    "open_memmap",
    "features.npy",
    "X_train",
    "metadata.parquet",
    "pipeline_a",
]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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


def format_shape(
    shape: tuple[int, ...],
) -> str:
    return (
        "("
        + ", ".join(
            str(value)
            for value in shape
        )
        + ")"
    )


required_files = [
    PROTOCOL_FILE,
    RUN_MATRIX_FILE,
    ALIGNMENT_FILE,
    PRIMARY_METADATA_FILE,
]

missing_required_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_required_files:
    print("Missing required files:")

    for path in missing_required_files:
        print(f"- {path}")

    sys.exit(1)


protocol = load_json(
    PROTOCOL_FILE
)

alignment = load_json(
    ALIGNMENT_FILE
)

run_rows = load_csv(
    RUN_MATRIX_FILE
)


# -------------------------------------------------------------------------
# Membership arrays
# -------------------------------------------------------------------------

membership_inventory = []
membership_arrays_valid = True

for filename, expected in (
    MEMBERSHIP_ARRAY_SPECS.items()
):
    path = (
        MEMBERSHIP_ROOT
        / filename
    )

    result: dict[str, Any] = {
        "filename":
            filename,

        "relative_path":
            str(
                path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "exists":
            path.exists(),

        "expected_shape":
            list(
                expected["shape"]
            ),

        "expected_dtype":
            expected["dtype"],

        "shape":
            None,

        "dtype":
            None,

        "size_bytes":
            (
                path.stat().st_size
                if path.exists()
                else None
            ),

        "valid":
            False,

        "error":
            None,
    }

    if path.exists():
        try:
            array = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )

            result["shape"] = list(
                array.shape
            )

            result["dtype"] = str(
                array.dtype
            )

            result["valid"] = (
                tuple(array.shape)
                == tuple(
                    expected["shape"]
                )
                and str(array.dtype)
                == expected["dtype"]
            )

        except Exception as error:
            result["error"] = repr(
                error
            )

    membership_arrays_valid &= bool(
        result["valid"]
    )

    membership_inventory.append(
        result
    )


# -------------------------------------------------------------------------
# Primary metadata
# -------------------------------------------------------------------------

primary_metadata = pq.ParquetFile(
    PRIMARY_METADATA_FILE
)

primary_metadata_rows = int(
    primary_metadata.metadata.num_rows
)

primary_metadata_columns = [
    field.name
    for field
    in primary_metadata.schema_arrow
]


# -------------------------------------------------------------------------
# NPY file inventory
# -------------------------------------------------------------------------

npy_inventory = []

npy_search_roots = [
    PROJECT_ROOT
    / "data"
    / "processed",

    PROJECT_ROOT
    / "models"
    / "preprocessing",

    PROJECT_ROOT
    / "results"
    / "v2",
]

seen_npy_paths: set[Path] = set()

for root in npy_search_roots:
    if not root.exists():
        continue

    for path in root.rglob("*.npy"):
        if path in seen_npy_paths:
            continue

        seen_npy_paths.add(path)

        result: dict[str, Any] = {
            "relative_path":
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "size_bytes":
                path.stat().st_size,

            "shape":
                None,

            "dtype":
                None,

            "row_count_match":
                False,

            "exact_feature_matrix_candidate":
                False,

            "error":
                None,
        }

        try:
            array = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )

            result["shape"] = list(
                array.shape
            )

            result["dtype"] = str(
                array.dtype
            )

            if len(
                array.shape
            ) >= 1:
                result[
                    "row_count_match"
                ] = (
                    int(array.shape[0])
                    == EXPECTED_ROW_COUNT
                )

            result[
                "exact_feature_matrix_candidate"
            ] = (
                len(array.shape) == 2
                and int(array.shape[0])
                == EXPECTED_ROW_COUNT
                and int(array.shape[1])
                == EXPECTED_FEATURE_COUNT
                and np.issubdtype(
                    array.dtype,
                    np.number,
                )
            )

        except Exception as error:
            result["error"] = repr(
                error
            )

        npy_inventory.append(
            result
        )


npy_inventory.sort(
    key=lambda row: (
        not row[
            "exact_feature_matrix_candidate"
        ],
        not row[
            "row_count_match"
        ],
        row["relative_path"],
    )
)


feature_matrix_candidates = [
    row
    for row in npy_inventory
    if row[
        "exact_feature_matrix_candidate"
    ]
]


# -------------------------------------------------------------------------
# Parquet inventory
# -------------------------------------------------------------------------

parquet_inventory = []

parquet_root = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

if parquet_root.exists():
    for path in parquet_root.rglob(
        "*.parquet"
    ):
        result: dict[str, Any] = {
            "relative_path":
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "size_bytes":
                path.stat().st_size,

            "row_count":
                None,

            "column_count":
                None,

            "columns":
                None,

            "row_count_match":
                False,

            "possible_feature_table":
                False,

            "error":
                None,
        }

        try:
            parquet_file = pq.ParquetFile(
                path
            )

            row_count = int(
                parquet_file
                .metadata
                .num_rows
            )

            columns = [
                field.name
                for field
                in parquet_file
                .schema_arrow
            ]

            result["row_count"] = (
                row_count
            )

            result["column_count"] = len(
                columns
            )

            result["columns"] = columns

            result["row_count_match"] = (
                row_count
                == EXPECTED_ROW_COUNT
            )

            result[
                "possible_feature_table"
            ] = (
                row_count
                == EXPECTED_ROW_COUNT
                and len(columns)
                >= EXPECTED_FEATURE_COUNT
            )

        except Exception as error:
            result["error"] = repr(
                error
            )

        parquet_inventory.append(
            result
        )


parquet_inventory.sort(
    key=lambda row: (
        not row[
            "possible_feature_table"
        ],
        not row[
            "row_count_match"
        ],
        row["relative_path"],
    )
)


parquet_feature_candidates = [
    row
    for row in parquet_inventory
    if row[
        "possible_feature_table"
    ]
]


# -------------------------------------------------------------------------
# Search model and pipeline code
# -------------------------------------------------------------------------

code_search_roots = [
    PROJECT_ROOT
    / "src",

    PROJECT_ROOT
    / "scripts",
]

code_matches = []

for root in code_search_roots:
    if not root.exists():
        continue

    for path in root.rglob("*.py"):
        try:
            lines = path.read_text(
                encoding="utf-8-sig",
                errors="replace",
            ).splitlines()

        except Exception:
            continue

        matches = []

        for line_number, line in enumerate(
            lines,
            start=1,
        ):
            matched_terms = [
                term
                for term
                in CODE_SEARCH_TERMS
                if term.casefold()
                in line.casefold()
            ]

            if not matched_terms:
                continue

            matches.append(
                {
                    "line_number":
                        line_number,

                    "text":
                        line.strip()[:300],

                    "terms":
                        matched_terms,
                }
            )

        if matches:
            code_matches.append(
                {
                    "relative_path":
                        str(
                            path.relative_to(
                                PROJECT_ROOT
                            )
                        ),

                    "match_count":
                        len(matches),

                    "matches":
                        matches[:40],
                }
            )


code_matches.sort(
    key=lambda row: (
        -row["match_count"],
        row["relative_path"],
    )
)


all_code_text = "\n".join(
    (
        match["text"]
        for file_result
        in code_matches
        for match
        in file_result["matches"]
    )
).casefold()

tinyml_code_found = (
    "tinyml_mlp"
    in all_code_text
)

compact_code_found = (
    "compact_dnn"
    in all_code_text
)

standard_scaler_code_found = (
    "standardscaler"
    in all_code_text
)


# -------------------------------------------------------------------------
# Run matrix
# -------------------------------------------------------------------------

run_model_counts = Counter(
    row["model_id"]
    for row in run_rows
)

run_device_counts = Counter(
    row["held_out_device"]
    for row in run_rows
)

run_seeds = sorted(
    {
        int(row["seed"])
        for row in run_rows
    }
)

run_status_counts = Counter(
    row["status"]
    for row in run_rows
)

run_ids = [
    row["run_id"]
    for row in run_rows
]


# -------------------------------------------------------------------------
# System resources
# -------------------------------------------------------------------------

disk_usage = shutil.disk_usage(
    PROJECT_ROOT
)

free_disk_gb = (
    disk_usage.free
    / (1024 ** 3)
)

total_disk_gb = (
    disk_usage.total
    / (1024 ** 3)
)

if psutil is not None:
    virtual_memory = (
        psutil.virtual_memory()
    )

    available_ram_gb = (
        virtual_memory.available
        / (1024 ** 3)
    )

    total_ram_gb = (
        virtual_memory.total
        / (1024 ** 3)
    )

else:
    available_ram_gb = None
    total_ram_gb = None


estimated_feature_matrix_gb = (
    EXPECTED_ROW_COUNT
    * EXPECTED_FEATURE_COUNT
    * np.dtype(
        np.float32
    ).itemsize
    / (1024 ** 3)
)


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------

protocol_checks = protocol.get(
    "validation_checks",
    {},
)

validation_checks = {
    "protocol_status_locked":
        protocol.get("status")
        == "locked",

    "protocol_all_checks_passed":
        protocol.get(
            "all_checks_passed"
        )
        is True,

    "protocol_internal_checks_true":
        bool(protocol_checks)
        and all(
            value is True
            for value
            in protocol_checks.values()
        ),

    "run_matrix_has_27_rows":
        len(run_rows)
        == EXPECTED_RUN_COUNT,

    "run_ids_are_unique":
        len(set(run_ids))
        == EXPECTED_RUN_COUNT,

    "three_models_in_run_matrix":
        len(run_model_counts)
        == 3
        and all(
            count == EXPECTED_DEVICE_COUNT
            for count
            in run_model_counts.values()
        ),

    "nine_devices_in_run_matrix":
        len(run_device_counts)
        == EXPECTED_DEVICE_COUNT
        and all(
            count == 3
            for count
            in run_device_counts.values()
        ),

    "single_seed_2026":
        run_seeds
        == [
            EXPECTED_SEED
        ],

    "all_runs_pending":
        run_status_counts
        == Counter(
            {
                "pending":
                    EXPECTED_RUN_COUNT
            }
        ),

    "membership_arrays_valid":
        membership_arrays_valid,

    "alignment_row_count_matches":
        int(
            alignment.get(
                "metadata_row_count",
                -1,
            )
        )
        == EXPECTED_ROW_COUNT,

    "primary_metadata_row_count":
        primary_metadata_rows
        == EXPECTED_ROW_COUNT,

    "family_column_available":
        "class_label"
        in primary_metadata_columns,

    "split_column_available":
        "split"
        in primary_metadata_columns,

    "tinyml_model_code_found":
        tinyml_code_found,

    "compact_model_code_found":
        compact_code_found,

    "standard_scaler_code_found":
        standard_scaler_code_found,

    "feature_matrix_candidate_found":
        (
            len(
                feature_matrix_candidates
            )
            + len(
                parquet_feature_candidates
            )
        )
        >= 1,

    "minimum_free_disk_8gb":
        free_disk_gb
        >= 8.0,

    "minimum_available_ram_4gb":
        (
            available_ram_gb
            is None
            or available_ram_gb
            >= 4.0
        ),
}


all_checks_passed = all(
    validation_checks.values()
)


summary = {
    "protocol_version":
        "early_lodo_training_preflight_v2_1",

    "status":
        (
            "passed"
            if all_checks_passed
            else "failed"
        ),

    "completed_at":
        utc_now(),

    "project_root":
        str(PROJECT_ROOT),

    "software": {
        "python":
            sys.version,

        "platform":
            platform.platform(),

        "numpy":
            np.__version__,

        "pyarrow":
            pq.__version__
            if hasattr(
                pq,
                "__version__",
            )
            else None,

        "torch":
            torch.__version__,

        "scikit_learn":
            sklearn.__version__,

        "cuda_available":
            torch.cuda.is_available(),
    },

    "resources": {
        "total_ram_gb":
            total_ram_gb,

        "available_ram_gb":
            available_ram_gb,

        "total_disk_gb":
            total_disk_gb,

        "free_disk_gb":
            free_disk_gb,

        "estimated_float32_feature_matrix_gb":
            estimated_feature_matrix_gb,
    },

    "run_matrix": {
        "row_count":
            len(run_rows),

        "model_counts":
            dict(
                run_model_counts
            ),

        "device_counts":
            dict(
                run_device_counts
            ),

        "seeds":
            run_seeds,

        "status_counts":
            dict(
                run_status_counts
            ),
    },

    "membership_inventory":
        membership_inventory,

    "primary_metadata": {
        "relative_path":
            str(
                PRIMARY_METADATA_FILE
                .relative_to(
                    PROJECT_ROOT
                )
            ),

        "row_count":
            primary_metadata_rows,

        "column_count":
            len(
                primary_metadata_columns
            ),

        "columns":
            primary_metadata_columns,
    },

    "npy_inventory":
        npy_inventory,

    "feature_matrix_candidates":
        feature_matrix_candidates,

    "parquet_inventory":
        parquet_inventory,

    "parquet_feature_candidates":
        parquet_feature_candidates,

    "code_matches":
        code_matches,

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,
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
    "=" * 82,
    "PHASE 2E EARLY LODO TRAINING PREFLIGHT",
    "=" * 82,
    "",
    "SYSTEM RESOURCES",
    (
        f"Available RAM : "
        f"{available_ram_gb:.3f} GB"
        if available_ram_gb
        is not None
        else "Available RAM : unknown"
    ),
    (
        f"Free disk     : "
        f"{free_disk_gb:.3f} GB"
    ),
    (
        f"Feature matrix estimate: "
        f"{estimated_feature_matrix_gb:.3f} GB"
    ),
    "",
    "RUN MATRIX",
    (
        f"Runs    : {len(run_rows)}"
    ),
    (
        "Models  : "
        + " | ".join(
            (
                f"{name}={count}"
                for name, count
                in sorted(
                    run_model_counts.items()
                )
            )
        )
    ),
    (
        f"Devices : "
        f"{len(run_device_counts)}"
    ),
    (
        "Seeds   : "
        + " | ".join(
            str(value)
            for value in run_seeds
        )
    ),
    "",
    "MEMBERSHIP ARRAYS",
]

for row in membership_inventory:
    report.append(
        (
            f"- {row['filename']} | "
            f"shape={row['shape']} | "
            f"dtype={row['dtype']} | "
            f"valid={row['valid']}"
        )
    )


report.extend(
    [
        "",
        "FEATURE MATRIX CANDIDATES",
    ]
)

if feature_matrix_candidates:
    for row in feature_matrix_candidates:
        report.append(
            (
                f"- NPY | "
                f"{row['relative_path']} | "
                f"shape={row['shape']} | "
                f"dtype={row['dtype']} | "
                f"size={row['size_bytes']}"
            )
        )

if parquet_feature_candidates:
    for row in parquet_feature_candidates:
        report.append(
            (
                f"- PARQUET | "
                f"{row['relative_path']} | "
                f"rows={row['row_count']} | "
                f"columns={row['column_count']} | "
                f"size={row['size_bytes']}"
            )
        )

if (
    not feature_matrix_candidates
    and not parquet_feature_candidates
):
    report.append(
        "- No exact candidate detected."
    )


report.extend(
    [
        "",
        "TOP CODE MATCHES",
    ]
)

for row in code_matches[:15]:
    report.append(
        (
            f"- {row['relative_path']} | "
            f"matches={row['match_count']}"
        )
    )


report.extend(
    [
        "",
        "VALIDATION CHECKS",
    ]
)

for name, passed in (
    validation_checks.items()
):
    report.append(
        f"{name}: {passed}"
    )


OUTPUT_REPORT.write_text(
    "\n".join(report),
    encoding="utf-8",
)


print("=" * 82)
print("PHASE 2E EARLY LODO TRAINING PREFLIGHT")
print("=" * 82)

print()
print("SYSTEM RESOURCES")

if available_ram_gb is None:
    print("Available RAM : unknown")
else:
    print(
        f"Available RAM : "
        f"{available_ram_gb:.3f} GB"
    )

print(
    f"Free disk     : "
    f"{free_disk_gb:.3f} GB"
)

print(
    f"Feature matrix estimate: "
    f"{estimated_feature_matrix_gb:.3f} GB"
)

print()
print("RUN MATRIX")
print(f"Runs    : {len(run_rows)}")

for name, count in sorted(
    run_model_counts.items()
):
    print(
        f"- {name}: {count}"
    )

print(
    f"Devices : "
    f"{len(run_device_counts)}"
)

print(f"Seeds   : {run_seeds}")

print()
print("MEMBERSHIP ARRAYS")

for row in membership_inventory:
    print(
        f"{row['filename']} | "
        f"shape={row['shape']} | "
        f"dtype={row['dtype']} | "
        f"valid={row['valid']}"
    )

print()
print("FEATURE MATRIX CANDIDATES")

for row in feature_matrix_candidates:
    print(
        f"NPY | "
        f"{row['relative_path']} | "
        f"shape={row['shape']} | "
        f"dtype={row['dtype']} | "
        f"size_gb="
        f"{row['size_bytes'] / (1024 ** 3):.3f}"
    )

for row in parquet_feature_candidates:
    print(
        f"PARQUET | "
        f"{row['relative_path']} | "
        f"rows={row['row_count']} | "
        f"columns={row['column_count']} | "
        f"size_gb="
        f"{row['size_bytes'] / (1024 ** 3):.3f}"
    )

if (
    not feature_matrix_candidates
    and not parquet_feature_candidates
):
    print(
        "No exact feature matrix "
        "candidate detected."
    )

print()
print("TOP CODE MATCHES")

for row in code_matches[:15]:
    print(
        f"{row['relative_path']} | "
        f"matches={row['match_count']}"
    )

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"JSON   : {OUTPUT_JSON}")
print(f"Report : {OUTPUT_REPORT}")

if not all_checks_passed:
    print()
    print(
        "PHASE 2E EARLY LODO "
        "TRAINING PREFLIGHT FAILED"
    )
    sys.exit(1)

print()
print(
    "PHASE 2E EARLY LODO "
    "TRAINING PREFLIGHT PASSED"
)
