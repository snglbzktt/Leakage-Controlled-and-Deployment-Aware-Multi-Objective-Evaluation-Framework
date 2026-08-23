from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn


ROOT = Path.cwd()

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

CACHE_MANIFEST = (
    FINAL_CACHE
    / "manifest.json"
)

FINAL_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_iteration2_summary_v3_2.json"
)

BASE_PROTOCOL = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_locked_v3.json"
)

BASE_LOCK_MANIFEST = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_lock_manifest_v3.json"
)

ADDENDUM = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_scaled_collision_addendum_v3_2.json"
)

ADDENDUM_MANIFEST = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_scaled_collision_addendum_manifest_v3_2.json"
)

OUTPUT_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_final_cache_verification_v3_2.json"
)

EXPECTED_PROTOCOL_VERSION = "tabular_baseline_protocol_v3_2"
EXPECTED_BASE_VERSION = "tabular_baseline_protocol_v3_1"
EXPECTED_FEATURE_COUNT = 115
EXPECTED_FINGERPRINTS = 2_278_176
EXPECTED_RAW_ROWS = 7_062_606
EXPECTED_RUN_COUNT = 12

EXPECTED_SPLITS = {
    "train": {
        "fingerprints": 1_534_583,
        "raw_rows": 4_943_823,
    },
    "validation": {
        "fingerprints": 371_797,
        "raw_rows": 1_059_390,
    },
    "test": {
        "fingerprints": 371_796,
        "raw_rows": 1_059_393,
    },
}

EXPECTED_MODEL_RUN_COUNTS = {
    "logistic_regression_b0": 1,
    "decision_tree_b0": 5,
    "random_forest_b0": 5,
    "hist_gradient_boosting_b0": 1,
}

LEGACY_CACHE_DIRECTORIES = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_1.building",
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2.building",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)

    return digest.hexdigest()


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def atomic_json(
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
            ensure_ascii=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


required_paths = [
    FINAL_CACHE,
    CACHE_MANIFEST,
    FINAL_REPORT,
    BASE_PROTOCOL,
    BASE_LOCK_MANIFEST,
    ADDENDUM,
    ADDENDUM_MANIFEST,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

if OUTPUT_REPORT.exists():
    raise FileExistsError(
        "Verification report already exists; "
        f"refusing to overwrite: {OUTPUT_REPORT}"
    )

manifest = read_json(
    CACHE_MANIFEST
)

final_report = read_json(
    FINAL_REPORT
)

base_protocol = read_json(
    BASE_PROTOCOL
)

base_lock_manifest = read_json(
    BASE_LOCK_MANIFEST
)

addendum = read_json(
    ADDENDUM
)

addendum_manifest = read_json(
    ADDENDUM_MANIFEST
)

run_matrix_path = Path(
    str(
        base_protocol[
            "run_policy"
        ][
            "run_matrix_file"
        ]
    )
)

if not run_matrix_path.exists():
    raise FileNotFoundError(
        run_matrix_path
    )

with run_matrix_path.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as handle:
    run_rows = list(
        csv.DictReader(handle)
    )

if not run_rows:
    raise RuntimeError(
        "Run matrix is empty."
    )

fieldnames = set(
    run_rows[0]
)

model_column = next(
    (
        name
        for name in (
            "model_name",
            "model",
            "baseline_model",
        )
        if name in fieldnames
    ),
    None,
)

run_id_column = next(
    (
        name
        for name in (
            "run_id",
            "id",
        )
        if name in fieldnames
    ),
    None,
)

if model_column is None:
    raise RuntimeError(
        "Run matrix model column was not found: "
        f"{sorted(fieldnames)}"
    )

model_counts: dict[str, int] = {}

for row in run_rows:
    model_name = str(
        row[model_column]
    )

    model_counts[model_name] = (
        model_counts.get(
            model_name,
            0,
        )
        + 1
    )

run_ids_unique = True

if run_id_column is not None:
    run_ids = [
        str(row[run_id_column])
        for row in run_rows
    ]

    run_ids_unique = (
        len(run_ids)
        == len(set(run_ids))
    )

split_records: dict[
    str,
    dict[str, Any],
] = {}

total_fingerprints = 0
total_raw_rows = 0

for split_name, expected in EXPECTED_SPLITS.items():
    x_path = (
        FINAL_CACHE
        / f"X_{split_name}.npy"
    )

    y_path = (
        FINAL_CACHE
        / f"y_{split_name}.npy"
    )

    raw_path = (
        FINAL_CACHE
        / f"raw_row_count_{split_name}.npy"
    )

    forward_path = (
        FINAL_CACHE
        / f"hash_forward_{split_name}.npy"
    )

    reverse_path = (
        FINAL_CACHE
        / f"hash_reverse_{split_name}.npy"
    )

    for path in (
        x_path,
        y_path,
        raw_path,
        forward_path,
        reverse_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    x = np.load(
        x_path,
        mmap_mode="r",
    )

    y = np.load(
        y_path,
        mmap_mode="r",
    )

    raw_counts = np.load(
        raw_path,
        mmap_mode="r",
    )

    forward = np.load(
        forward_path,
        mmap_mode="r",
    )

    reverse = np.load(
        reverse_path,
        mmap_mode="r",
    )

    expected_rows = int(
        expected["fingerprints"]
    )

    shape_dtype_checks = {
        "X": (
            x.shape
            == (
                expected_rows,
                EXPECTED_FEATURE_COUNT,
            )
            and x.dtype == np.float32
        ),
        "y": (
            y.shape == (expected_rows,)
            and y.dtype == np.int8
        ),
        "raw_row_count": (
            raw_counts.shape
            == (expected_rows,)
            and raw_counts.dtype
            == np.int64
        ),
        "hash_forward": (
            forward.shape
            == (expected_rows,)
            and forward.dtype
            == np.int64
        ),
        "hash_reverse": (
            reverse.shape
            == (expected_rows,)
            and reverse.dtype
            == np.int64
        ),
    }

    observed_raw_rows = int(
        np.asarray(
            raw_counts,
            dtype=np.int64,
        ).sum()
    )

    class_fingerprint_counts = np.bincount(
        np.asarray(
            y,
            dtype=np.int64,
        ),
        minlength=3,
    ).astype(np.int64).tolist()

    class_raw_counts = np.bincount(
        np.asarray(
            y,
            dtype=np.int64,
        ),
        weights=np.asarray(
            raw_counts,
            dtype=np.float64,
        ),
        minlength=3,
    ).astype(np.int64).tolist()

    split_records[split_name] = {
        "fingerprint_count": int(
            len(x)
        ),
        "raw_row_count": (
            observed_raw_rows
        ),
        "class_fingerprint_counts": (
            class_fingerprint_counts
        ),
        "class_raw_row_counts": (
            class_raw_counts
        ),
        "shape_dtype_checks": (
            shape_dtype_checks
        ),
        "all_shape_dtype_checks_passed": all(
            shape_dtype_checks.values()
        ),
        "all_raw_row_counts_positive": bool(
            np.all(
                np.asarray(
                    raw_counts,
                    dtype=np.int64,
                )
                > 0
            )
        ),
    }

    total_fingerprints += int(
        len(x)
    )

    total_raw_rows += (
        observed_raw_rows
    )

artifact_records = manifest.get(
    "artifacts"
)

if not isinstance(
    artifact_records,
    list,
):
    raise RuntimeError(
        "Manifest artifact inventory is missing."
    )

artifact_checks: list[
    dict[str, Any]
] = []

for item in artifact_records:
    relative_path = str(
        item["relative_path"]
    )

    path = (
        FINAL_CACHE
        / relative_path
    )

    exists = path.exists()

    observed_size = (
        int(path.stat().st_size)
        if exists
        else None
    )

    observed_sha256 = (
        sha256_file(path)
        if exists
        else None
    )

    expected_size = int(
        item["size_bytes"]
    )

    expected_sha256 = str(
        item["sha256"]
    )

    artifact_checks.append(
        {
            "relative_path": (
                relative_path
            ),
            "exists": exists,
            "expected_size_bytes": (
                expected_size
            ),
            "observed_size_bytes": (
                observed_size
            ),
            "size_matches": (
                observed_size
                == expected_size
            ),
            "expected_sha256": (
                expected_sha256
            ),
            "observed_sha256": (
                observed_sha256
            ),
            "sha256_matches": (
                observed_sha256
                == expected_sha256
            ),
        }
    )

legacy_cache_inventory = []

for path in LEGACY_CACHE_DIRECTORIES:
    legacy_cache_inventory.append(
        {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": (
                directory_size_bytes(path)
            ),
        }
    )

free_disk_bytes = shutil.disk_usage(
    ROOT
).free

checks = {
    "manifest_completed": (
        manifest.get("status")
        == "completed"
    ),
    "manifest_protocol_version_matches": (
        manifest.get(
            "protocol_version"
        )
        == EXPECTED_PROTOCOL_VERSION
    ),
    "manifest_all_checks_passed": (
        manifest.get(
            "all_checks_passed"
        )
        is True
    ),
    "manifest_repair_iterations_is_2": (
        int(
            manifest[
                "repair_iterations_completed"
            ]
        )
        == 2
    ),
    "manifest_cumulative_moves_is_356": (
        int(
            manifest[
                "cumulative_move_count"
            ]
        )
        == 356
    ),
    "final_report_completed": (
        final_report.get("status")
        == "completed"
    ),
    "final_report_all_checks_passed": (
        final_report.get(
            "all_checks_passed"
        )
        is True
    ),
    "final_report_zero_scaled_overlap": (
        int(
            final_report[
                "scaled_cross_split_overlap_count"
            ]
        )
        == 0
    ),
    "final_report_zero_family_conflicts": (
        int(
            final_report[
                "scaled_global_family_conflict_count"
            ]
        )
        == 0
    ),
    "base_protocol_locked": (
        base_protocol.get("status")
        == "locked"
    ),
    "base_protocol_version_matches": (
        base_protocol.get(
            "protocol_version"
        )
        == EXPECTED_BASE_VERSION
    ),
    "base_protocol_hash_matches_lock_manifest": (
        base_lock_manifest.get(
            "protocol_file_sha256"
        )
        == sha256_file(
            BASE_PROTOCOL
        )
    ),
    "addendum_locked": (
        addendum.get("status")
        == "locked"
    ),
    "addendum_version_matches": (
        addendum.get(
            "protocol_version"
        )
        == EXPECTED_PROTOCOL_VERSION
    ),
    "addendum_hash_matches_manifest": (
        addendum_manifest.get(
            "addendum_sha256"
        )
        == sha256_file(
            ADDENDUM
        )
    ),
    "total_fingerprint_count_matches": (
        total_fingerprints
        == EXPECTED_FINGERPRINTS
    ),
    "total_raw_row_count_matches": (
        total_raw_rows
        == EXPECTED_RAW_ROWS
    ),
    "all_split_shape_dtype_checks_pass": all(
        row[
            "all_shape_dtype_checks_passed"
        ]
        for row in split_records.values()
    ),
    "all_raw_row_counts_positive": all(
        row[
            "all_raw_row_counts_positive"
        ]
        for row in split_records.values()
    ),
    "all_split_counts_match": all(
        split_records[
            split_name
        ][
            "fingerprint_count"
        ]
        == expected[
            "fingerprints"
        ]
        and split_records[
            split_name
        ][
            "raw_row_count"
        ]
        == expected[
            "raw_rows"
        ]
        for split_name, expected
        in EXPECTED_SPLITS.items()
    ),
    "all_manifest_artifacts_exist": all(
        row["exists"]
        for row in artifact_checks
    ),
    "all_manifest_artifact_sizes_match": all(
        row["size_matches"]
        for row in artifact_checks
    ),
    "all_manifest_artifact_hashes_match": all(
        row["sha256_matches"]
        for row in artifact_checks
    ),
    "run_matrix_count_is_12": (
        len(run_rows)
        == EXPECTED_RUN_COUNT
    ),
    "run_matrix_model_counts_match": (
        model_counts
        == EXPECTED_MODEL_RUN_COUNTS
    ),
    "run_ids_unique": (
        run_ids_unique
    ),
    "sklearn_version_is_1_9_0": (
        sklearn.__version__
        == "1.9.0"
    ),
}

failed = [
    name
    for name, passed in checks.items()
    if not passed
]

report = {
    "status": (
        "passed"
        if not failed
        else "failed"
    ),
    "generated_at_utc": utc_now(),
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "final_cache": str(
        FINAL_CACHE
    ),
    "cache_manifest": str(
        CACHE_MANIFEST
    ),
    "cache_manifest_sha256": (
        sha256_file(
            CACHE_MANIFEST
        )
    ),
    "final_report": str(
        FINAL_REPORT
    ),
    "final_report_sha256": (
        sha256_file(
            FINAL_REPORT
        )
    ),
    "run_matrix": str(
        run_matrix_path
    ),
    "run_matrix_sha256": (
        sha256_file(
            run_matrix_path
        )
    ),
    "run_count": len(
        run_rows
    ),
    "model_run_counts": (
        model_counts
    ),
    "run_ids_unique": (
        run_ids_unique
    ),
    "split_inventory": (
        split_records
    ),
    "total_fingerprint_count": (
        total_fingerprints
    ),
    "total_raw_row_count": (
        total_raw_rows
    ),
    "manifest_artifact_checks": (
        artifact_checks
    ),
    "legacy_cache_inventory": (
        legacy_cache_inventory
    ),
    "final_cache_size_bytes": (
        directory_size_bytes(
            FINAL_CACHE
        )
    ),
    "free_disk_bytes": (
        free_disk_bytes
    ),
    "free_disk_gib": (
        free_disk_bytes
        / (1024**3)
    ),
    "python_environment": {
        "numpy_version": (
            np.__version__
        ),
        "sklearn_version": (
            sklearn.__version__
        ),
    },
    "checks": checks,
    "failed_checks": failed,
    "all_checks_passed": (
        not failed
    ),
}

atomic_json(
    OUTPUT_REPORT,
    report,
)

print("=" * 92)
print("FINAL TABULAR CACHE VERIFICATION")
print("=" * 92)
print(
    "Protocol version      : "
    f"{EXPECTED_PROTOCOL_VERSION}"
)
print(
    "Fingerprint rows      : "
    f"{total_fingerprints:,}"
)
print(
    "Represented raw rows  : "
    f"{total_raw_rows:,}"
)
print(
    "Run matrix rows       : "
    f"{len(run_rows):,}"
)
print(
    "Model run counts      : "
    f"{model_counts}"
)
print(
    "Manifest artifacts    : "
    f"{len(artifact_checks):,}"
)
print(
    "Final cache size      : "
    f"{directory_size_bytes(FINAL_CACHE) / (1024**3):.3f} GiB"
)
print(
    "Free disk             : "
    f"{free_disk_bytes / (1024**3):.3f} GiB"
)

print()
print("SPLIT INVENTORY")

for split_name in SPLITS:
    row = split_records[
        split_name
    ]

    print(
        f"  {split_name:<10} | "
        f"fingerprints={row['fingerprint_count']:,} | "
        f"raw_rows={row['raw_row_count']:,}"
    )

print()
print("LEGACY CACHE INVENTORY")

for row in legacy_cache_inventory:
    print(
        f"  exists={str(row['exists']):<5} | "
        f"size={row['size_bytes'] / (1024**3):.3f} GiB | "
        f"{row['path']}"
    )

print()
print("VALIDATION CHECKS")

for name, passed in checks.items():
    print(
        f"  {name}: {passed}"
    )

print()
print(
    "Verification report : "
    f"{OUTPUT_REPORT}"
)
print(
    "All checks passed   : "
    f"{not failed}"
)

if failed:
    print(
        "Failed checks       : "
        f"{failed}"
    )

    raise RuntimeError(
        "Final tabular cache verification failed."
    )

print(
    "FINAL TABULAR CACHE VERIFICATION COMPLETED"
)
