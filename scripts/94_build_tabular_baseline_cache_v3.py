from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path.cwd()

PROTOCOL_DATABASE = (
    ROOT
    / "data"
    / "splits"
    / "vs2_float32_leakage_ablation_scaled_repaired_seed2026_v3.sqlite"
)

LOCKED_PROTOCOL = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_locked_v3.json"
)

LOCK_MANIFEST = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_lock_manifest_v3.json"
)

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_1"
)

BUILD_CACHE = FINAL_CACHE.with_name(
    FINAL_CACHE.name + ".building"
)

OUTPUT_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_cache_build_summary_v3.json"
)

PROTOCOL_VERSION = "tabular_baseline_protocol_v3_1"
EXPECTED_SPLIT_PROTOCOL_VERSION = "2.3"

FEATURE_COUNT = 115
EXPECTED_FILE_COUNT = 89
EXPECTED_GROUP_COUNT = 2_278_176
EXPECTED_RAW_ROW_COUNT = 7_062_606

CHUNK_SIZE = 100_000
SCALER_AUDIT_TOLERANCE = 1e-4
MINIMUM_FREE_DISK_GIB = 2.5

SPLITS = (
    "train",
    "validation",
    "test",
)

SPLIT_CODES = {
    "train": 0,
    "validation": 1,
    "test": 2,
}

SPLIT_NAMES = {
    0: "train",
    1: "validation",
    2: "test",
}

FAMILY_NAMES = {
    0: "benign",
    1: "gafgyt",
    2: "mirai",
}

EXPECTED_SPLIT_COUNTS = {
    "train": {
        "fingerprints": 1_534_583,
        "raw_rows": 4_943_823,
    },
    "validation": {
        "fingerprints": 371_796,
        "raw_rows": 1_059_389,
    },
    "test": {
        "fingerprints": 371_797,
        "raw_rows": 1_059_394,
    },
}

EXPECTED_FAMILY_SPLIT_COUNTS = {
    ("train", "benign"): (359_439, 389_152),
    ("train", "gafgyt"): (27_221, 1_986_790),
    ("train", "mirai"): (1_147_923, 2_567_881),
    ("validation", "benign"): (77_021, 83_389),
    ("validation", "gafgyt"): (48_793, 425_740),
    ("validation", "mirai"): (245_982, 550_260),
    ("test", "benign"): (77_022, 83_391),
    ("test", "gafgyt"): (48_793, 425_742),
    ("test", "mirai"): (245_982, 550_261),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)

    return digest.hexdigest()


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


def family_code(
    raw_label: str,
) -> int:
    normalized = str(raw_label).strip().lower()

    if normalized == "benign":
        return 0

    if normalized.startswith("gafgyt_"):
        return 1

    if normalized.startswith("mirai_"):
        return 2

    raise RuntimeError(
        f"Unknown raw class label: {raw_label!r}"
    )


def to_signed_int64(
    values: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        values,
        dtype=np.uint64,
    ).view(np.int64)


def float32_fingerprints(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.array(
        values,
        dtype="<f4",
        order="C",
        copy=True,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[1] != FEATURE_COUNT
    ):
        raise RuntimeError(
            "Expected a (*, 115) feature matrix; "
            f"found {matrix.shape}"
        )

    if not np.isfinite(matrix).all():
        raise RuntimeError(
            "Non-finite model input found."
        )

    matrix[matrix == 0.0] = 0.0

    words = matrix.view("<u4").reshape(
        matrix.shape
    )

    forward = np.full(
        matrix.shape[0],
        np.uint64(0xCBF29CE484222325),
        dtype=np.uint64,
    )

    reverse = np.full(
        matrix.shape[0],
        np.uint64(0x84222325CBF29CE4),
        dtype=np.uint64,
    )

    prime_forward = np.uint64(
        0x100000001B3
    )

    prime_reverse = np.uint64(
        0x9E3779B185EBCA87
    )

    with np.errstate(over="ignore"):
        for column in range(
            matrix.shape[1]
        ):
            word_forward = words[
                :,
                column,
            ].astype(
                np.uint64,
                copy=False,
            )

            word_reverse = words[
                :,
                matrix.shape[1] - 1 - column,
            ].astype(
                np.uint64,
                copy=False,
            )

            forward = (
                forward
                ^ (
                    word_forward
                    + np.uint64(column + 1)
                )
            ) * prime_forward

            reverse = (
                reverse
                ^ (
                    word_reverse
                    + np.uint64(column + 1)
                )
            ) * prime_reverse

            forward ^= (
                forward
                >> np.uint64(32)
            )

            reverse ^= (
                reverse
                >> np.uint64(29)
            )

    return (
        to_signed_int64(forward),
        to_signed_int64(reverse),
    )


def database_metadata(
    connection: sqlite3.Connection,
) -> dict[str, str]:
    rows = connection.execute(
        "SELECT key, value FROM metadata"
    ).fetchall()

    return {
        str(key): str(value)
        for key, value in rows
    }


def scalar(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[Any, ...] = (),
) -> int:
    return int(
        connection.execute(
            query,
            parameters,
        ).fetchone()[0]
    )


def validate_locked_inputs() -> dict[str, Any]:
    for path in (
        PROTOCOL_DATABASE,
        LOCKED_PROTOCOL,
        LOCK_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if FINAL_CACHE.exists():
        raise FileExistsError(
            "Final cache already exists; refusing to overwrite: "
            f"{FINAL_CACHE}"
        )

    if BUILD_CACHE.exists():
        raise FileExistsError(
            "Incomplete build directory exists; inspect it before retrying: "
            f"{BUILD_CACHE}"
        )

    protocol = json.loads(
        LOCKED_PROTOCOL.read_text(
            encoding="utf-8"
        )
    )

    lock_manifest = json.loads(
        LOCK_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        protocol.get("status") != "locked"
        or protocol.get("protocol_version")
        != PROTOCOL_VERSION
    ):
        raise RuntimeError(
            "Locked tabular baseline protocol is invalid."
        )

    protocol_sha256 = sha256_file(
        LOCKED_PROTOCOL
    )

    run_matrix_path = Path(
        str(
            protocol[
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

    checks = {
        "lock_manifest_status_locked": (
            lock_manifest.get("status")
            == "locked"
        ),
        "lock_manifest_protocol_version_matches": (
            lock_manifest.get(
                "protocol_version"
            )
            == PROTOCOL_VERSION
        ),
        "locked_protocol_sha256_matches_manifest": (
            lock_manifest.get(
                "protocol_file_sha256"
            )
            == protocol_sha256
        ),
        "run_matrix_sha256_matches_manifest": (
            lock_manifest.get(
                "run_matrix_file_sha256"
            )
            == sha256_file(
                run_matrix_path
            )
        ),
        "protocol_database_path_matches": (
            Path(
                str(
                    protocol[
                        "data_policy"
                    ][
                        "split_database"
                    ]
                )
            ).resolve()
            == PROTOCOL_DATABASE.resolve()
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "Locked input validation failed: "
            + ", ".join(failed)
        )

    return {
        "protocol": protocol,
        "lock_manifest": lock_manifest,
        "protocol_sha256": protocol_sha256,
        "run_matrix_path": run_matrix_path,
        "checks": checks,
    }


def validate_database(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    quick_check = str(
        connection.execute(
            "PRAGMA quick_check"
        ).fetchone()[0]
    )

    if quick_check.lower() != "ok":
        raise RuntimeError(
            "Protocol database quick_check failed: "
            f"{quick_check}"
        )

    metadata = database_metadata(
        connection
    )

    expected_metadata = {
        "protocol": "vs2_float32_leakage_ablation",
        "protocol_version": (
            EXPECTED_SPLIT_PROTOCOL_VERSION
        ),
        "feature_dtype": "float32",
        "feature_count": str(
            FEATURE_COUNT
        ),
        "task": "family_3",
        "validation_status": "passed",
        "source_file_count": str(
            EXPECTED_FILE_COUNT
        ),
    }

    metadata_checks = {
        key: metadata.get(key) == value
        for key, value
        in expected_metadata.items()
    }

    group_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprint_stats
        """,
    )

    raw_row_count = scalar(
        connection,
        """
        SELECT SUM(raw_row_count)
        FROM fingerprint_stats
        """,
    )

    assignment_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM assignments
        """,
    )

    cross_family_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprint_stats
        WHERE family_label_min
              != family_label_max
        """,
    )

    invalid_split_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprint_stats
        WHERE grouped_split IS NULL
           OR grouped_split NOT IN (0, 1, 2)
        """,
    )

    observed_split_counts = {
        SPLIT_NAMES[int(code)]: {
            "fingerprints": int(groups),
            "raw_rows": int(raw_rows),
        }
        for code, groups, raw_rows
        in connection.execute(
            """
            SELECT
                grouped_split,
                COUNT(*),
                SUM(raw_row_count)
            FROM fingerprint_stats
            GROUP BY grouped_split
            ORDER BY grouped_split
            """
        ).fetchall()
    }

    observed_family_counts = {
        (
            SPLIT_NAMES[int(split_code)],
            FAMILY_NAMES[int(family)],
        ): (
            int(groups),
            int(raw_rows),
        )
        for (
            split_code,
            family,
            groups,
            raw_rows,
        ) in connection.execute(
            """
            SELECT
                grouped_split,
                family_label_min,
                COUNT(*),
                SUM(raw_row_count)
            FROM fingerprint_stats
            GROUP BY
                grouped_split,
                family_label_min
            ORDER BY
                grouped_split,
                family_label_min
            """
        ).fetchall()
    }

    checks = {
        **{
            f"metadata_{key}_matches": passed
            for key, passed
            in metadata_checks.items()
        },
        "assignment_count_matches": (
            assignment_count
            == EXPECTED_RAW_ROW_COUNT
        ),
        "group_count_matches": (
            group_count
            == EXPECTED_GROUP_COUNT
        ),
        "raw_row_count_matches": (
            raw_row_count
            == EXPECTED_RAW_ROW_COUNT
        ),
        "cross_family_count_zero": (
            cross_family_count == 0
        ),
        "invalid_split_count_zero": (
            invalid_split_count == 0
        ),
        "split_counts_match": (
            observed_split_counts
            == EXPECTED_SPLIT_COUNTS
        ),
        "family_split_counts_match": (
            observed_family_counts
            == EXPECTED_FAMILY_SPLIT_COUNTS
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "Protocol database validation failed: "
            + ", ".join(failed)
        )

    return {
        "quick_check": quick_check,
        "metadata": metadata,
        "assignment_count": assignment_count,
        "group_count": group_count,
        "raw_row_count": raw_row_count,
        "cross_family_count": (
            cross_family_count
        ),
        "invalid_split_count": (
            invalid_split_count
        ),
        "split_counts": (
            observed_split_counts
        ),
        "family_split_counts": [
            {
                "split": split_name,
                "family": family_name,
                "fingerprints": values[0],
                "raw_rows": values[1],
            }
            for (
                split_name,
                family_name,
            ), values
            in sorted(
                observed_family_counts.items()
            )
        ],
        "checks": checks,
    }


def validate_source_files(
    connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            file_id,
            file_path,
            class_label,
            expected_row_count,
            file_size_bytes,
            sha256
        FROM source_file_integrity
        ORDER BY file_id
        """
    ).fetchall()

    if len(rows) != EXPECTED_FILE_COUNT:
        raise RuntimeError(
            "Source file count mismatch: "
            f"{len(rows)} != {EXPECTED_FILE_COUNT}"
        )

    validated: list[dict[str, Any]] = []

    for position, row in enumerate(
        rows,
        start=1,
    ):
        (
            file_id,
            file_path,
            class_label,
            expected_rows,
            expected_size,
            expected_sha256,
        ) = row

        path = Path(str(file_path))

        if not path.exists():
            raise FileNotFoundError(path)

        observed_size = int(
            path.stat().st_size
        )

        if observed_size != int(
            expected_size
        ):
            raise RuntimeError(
                "Source file size changed: "
                f"{path}"
            )

        observed_sha256 = sha256_file(
            path
        )

        if observed_sha256 != str(
            expected_sha256
        ):
            raise RuntimeError(
                "Source file SHA-256 changed: "
                f"{path}"
            )

        validated.append(
            {
                "file_id": int(file_id),
                "file_path": str(
                    path.resolve()
                ),
                "class_label": str(
                    class_label
                ),
                "family_code": family_code(
                    str(class_label)
                ),
                "expected_row_count": int(
                    expected_rows
                ),
                "file_size_bytes": (
                    observed_size
                ),
                "sha256": observed_sha256,
            }
        )

        print(
            f"[source integrity] "
            f"{position}/{len(rows)} | "
            f"{path.name}",
            flush=True,
        )

    return validated


def create_representative_table(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    max_file_id, max_row_number = (
        connection.execute(
            """
            SELECT
                MAX(file_id),
                MAX(row_number)
            FROM assignments
            """
        ).fetchone()
    )

    if (
        int(max_file_id) >= 2**31
        or int(max_row_number) >= 2**32
        or int(max_row_number) < 0
    ):
        raise RuntimeError(
            "Representative location encoding limits exceeded."
        )

    print(
        "Creating deterministic representative table...",
        flush=True,
    )

    connection.execute(
        "PRAGMA temp_store=FILE"
    )

    connection.execute(
        "PRAGMA cache_size=-250000"
    )

    connection.execute(
        """
        CREATE TEMP TABLE representatives AS

        WITH locations AS (
            SELECT
                hash_forward,
                hash_reverse,
                MIN(
                    (
                        CAST(file_id AS INTEGER)
                        << 32
                    )
                    |
                    CAST(row_number AS INTEGER)
                ) AS encoded_location
            FROM assignments
            GROUP BY
                hash_forward,
                hash_reverse
        )

        SELECT
            locations.hash_forward
                AS hash_forward,
            locations.hash_reverse
                AS hash_reverse,
            (
                locations.encoded_location
                >> 32
            ) AS file_id,
            (
                locations.encoded_location
                & 4294967295
            ) AS row_number,
            fingerprint_stats.grouped_split
                AS grouped_split,
            fingerprint_stats.family_label_min
                AS family_label,
            fingerprint_stats.raw_row_count
                AS raw_row_count

        FROM locations

        JOIN fingerprint_stats
          ON fingerprint_stats.hash_forward
             = locations.hash_forward
         AND fingerprint_stats.hash_reverse
             = locations.hash_reverse
        """
    )

    connection.execute(
        """
        CREATE UNIQUE INDEX
        temp.idx_representatives_hash
        ON representatives (
            hash_forward,
            hash_reverse
        )
        """
    )

    connection.execute(
        """
        CREATE UNIQUE INDEX
        temp.idx_representatives_file_row
        ON representatives (
            file_id,
            row_number
        )
        """
    )

    representative_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM representatives
        """,
    )

    representative_raw_rows = scalar(
        connection,
        """
        SELECT SUM(raw_row_count)
        FROM representatives
        """,
    )

    file_count = scalar(
        connection,
        """
        SELECT COUNT(DISTINCT file_id)
        FROM representatives
        """,
    )

    duplicate_location_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                file_id,
                row_number,
                COUNT(*) AS amount
            FROM representatives
            GROUP BY
                file_id,
                row_number
            HAVING amount > 1
        )
        """,
    )

    checks = {
        "representative_count_matches": (
            representative_count
            == EXPECTED_GROUP_COUNT
        ),
        "represented_raw_rows_match": (
            representative_raw_rows
            == EXPECTED_RAW_ROW_COUNT
        ),
        "representative_file_count_positive": (
            file_count > 0
        ),
        "duplicate_location_count_zero": (
            duplicate_location_count == 0
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "Representative table validation failed: "
            + ", ".join(failed)
        )

    print(
        "Representative table ready: "
        f"{representative_count:,} rows",
        flush=True,
    )

    return {
        "representative_count": (
            representative_count
        ),
        "represented_raw_rows": (
            representative_raw_rows
        ),
        "representative_file_count": (
            file_count
        ),
        "duplicate_location_count": (
            duplicate_location_count
        ),
        "checks": checks,
    }


def open_cache_arrays() -> dict[
    str,
    dict[str, np.memmap],
]:
    arrays: dict[
        str,
        dict[str, np.memmap],
    ] = {}

    for split_name in SPLITS:
        row_count = EXPECTED_SPLIT_COUNTS[
            split_name
        ]["fingerprints"]

        arrays[split_name] = {
            "X": np.lib.format.open_memmap(
                BUILD_CACHE
                / f"X_{split_name}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(
                    row_count,
                    FEATURE_COUNT,
                ),
            ),
            "y": np.lib.format.open_memmap(
                BUILD_CACHE
                / f"y_{split_name}.npy",
                mode="w+",
                dtype=np.int8,
                shape=(row_count,),
            ),
            "raw_row_count": (
                np.lib.format.open_memmap(
                    BUILD_CACHE
                    / (
                        "raw_row_count_"
                        f"{split_name}.npy"
                    ),
                    mode="w+",
                    dtype=np.int64,
                    shape=(row_count,),
                )
            ),
            "hash_forward": (
                np.lib.format.open_memmap(
                    BUILD_CACHE
                    / (
                        "hash_forward_"
                        f"{split_name}.npy"
                    ),
                    mode="w+",
                    dtype=np.int64,
                    shape=(row_count,),
                )
            ),
            "hash_reverse": (
                np.lib.format.open_memmap(
                    BUILD_CACHE
                    / (
                        "hash_reverse_"
                        f"{split_name}.npy"
                    ),
                    mode="w+",
                    dtype=np.int64,
                    shape=(row_count,),
                )
            ),
        }

    return arrays


def materialize_cache(
    connection: sqlite3.Connection,
    source_files: list[dict[str, Any]],
) -> dict[str, Any]:
    arrays = open_cache_arrays()

    offsets = {
        split_name: 0
        for split_name in SPLITS
    }

    per_file_records: list[
        dict[str, Any]
    ] = []

    try:
        for file_position, source in enumerate(
            source_files,
            start=1,
        ):
            rows = connection.execute(
                """
                SELECT
                    row_number,
                    hash_forward,
                    hash_reverse,
                    grouped_split,
                    family_label,
                    raw_row_count
                FROM representatives
                WHERE file_id = ?
                ORDER BY row_number
                """,
                (
                    int(source["file_id"]),
                ),
            ).fetchall()

            if rows:
                representative_rows = np.fromiter(
                    (
                        int(row[0])
                        for row in rows
                    ),
                    dtype=np.int64,
                    count=len(rows),
                )

                expected_forward = np.fromiter(
                    (
                        int(row[1])
                        for row in rows
                    ),
                    dtype=np.int64,
                    count=len(rows),
                )

                expected_reverse = np.fromiter(
                    (
                        int(row[2])
                        for row in rows
                    ),
                    dtype=np.int64,
                    count=len(rows),
                )

                split_codes = np.fromiter(
                    (
                        int(row[3])
                        for row in rows
                    ),
                    dtype=np.int8,
                    count=len(rows),
                )

                families = np.fromiter(
                    (
                        int(row[4])
                        for row in rows
                    ),
                    dtype=np.int8,
                    count=len(rows),
                )

                occurrence_counts = np.fromiter(
                    (
                        int(row[5])
                        for row in rows
                    ),
                    dtype=np.int64,
                    count=len(rows),
                )

                if np.any(
                    families
                    != int(
                        source[
                            "family_code"
                        ]
                    )
                ):
                    raise RuntimeError(
                        "Representative family/source label mismatch: "
                        f"file_id={source['file_id']}"
                    )
            else:
                representative_rows = np.empty(
                    0,
                    dtype=np.int64,
                )
                expected_forward = np.empty(
                    0,
                    dtype=np.int64,
                )
                expected_reverse = np.empty(
                    0,
                    dtype=np.int64,
                )
                split_codes = np.empty(
                    0,
                    dtype=np.int8,
                )
                families = np.empty(
                    0,
                    dtype=np.int8,
                )
                occurrence_counts = np.empty(
                    0,
                    dtype=np.int64,
                )

            selected_count = 0
            base = 0
            path = Path(
                source["file_path"]
            )
            last_progress = time.perf_counter()

            for frame in pd.read_csv(
                path,
                chunksize=CHUNK_SIZE,
            ):
                values = frame.to_numpy(
                    dtype=np.float32,
                    copy=False,
                )

                if (
                    values.ndim != 2
                    or values.shape[1]
                    != FEATURE_COUNT
                ):
                    raise RuntimeError(
                        "Invalid source feature shape: "
                        f"{path} -> {values.shape}"
                    )

                if not np.isfinite(
                    values
                ).all():
                    raise RuntimeError(
                        "Non-finite source value: "
                        f"{path}"
                    )

                end = base + len(values)

                left = int(
                    np.searchsorted(
                        representative_rows,
                        base,
                        side="left",
                    )
                )

                right = int(
                    np.searchsorted(
                        representative_rows,
                        end,
                        side="left",
                    )
                )

                if right > left:
                    local_rows = (
                        representative_rows[
                            left:right
                        ]
                        - base
                    )

                    selected_values = np.array(
                        values[local_rows],
                        dtype=np.float32,
                        order="C",
                        copy=True,
                    )

                    selected_values[
                        selected_values == 0.0
                    ] = 0.0

                    actual_forward, actual_reverse = (
                        float32_fingerprints(
                            selected_values
                        )
                    )

                    expected_forward_chunk = (
                        expected_forward[
                            left:right
                        ]
                    )

                    expected_reverse_chunk = (
                        expected_reverse[
                            left:right
                        ]
                    )

                    if not np.array_equal(
                        actual_forward,
                        expected_forward_chunk,
                    ):
                        mismatch_count = int(
                            np.count_nonzero(
                                actual_forward
                                != expected_forward_chunk
                            )
                        )

                        raise RuntimeError(
                            "Representative forward hash mismatch: "
                            f"file_id={source['file_id']}, "
                            f"count={mismatch_count}"
                        )

                    if not np.array_equal(
                        actual_reverse,
                        expected_reverse_chunk,
                    ):
                        mismatch_count = int(
                            np.count_nonzero(
                                actual_reverse
                                != expected_reverse_chunk
                            )
                        )

                        raise RuntimeError(
                            "Representative reverse hash mismatch: "
                            f"file_id={source['file_id']}, "
                            f"count={mismatch_count}"
                        )

                    for split_code in range(3):
                        mask = (
                            split_codes[
                                left:right
                            ]
                            == split_code
                        )

                        amount = int(
                            np.count_nonzero(
                                mask
                            )
                        )

                        if amount == 0:
                            continue

                        split_name = SPLIT_NAMES[
                            split_code
                        ]

                        output_start = offsets[
                            split_name
                        ]

                        output_end = (
                            output_start
                            + amount
                        )

                        expected_limit = (
                            EXPECTED_SPLIT_COUNTS[
                                split_name
                            ][
                                "fingerprints"
                            ]
                        )

                        if output_end > expected_limit:
                            raise RuntimeError(
                                "Cache split overflow: "
                                f"{split_name}"
                            )

                        target = arrays[
                            split_name
                        ]

                        target[
                            "X"
                        ][
                            output_start:output_end
                        ] = selected_values[
                            mask
                        ]

                        target[
                            "y"
                        ][
                            output_start:output_end
                        ] = families[
                            left:right
                        ][
                            mask
                        ]

                        target[
                            "raw_row_count"
                        ][
                            output_start:output_end
                        ] = occurrence_counts[
                            left:right
                        ][
                            mask
                        ]

                        target[
                            "hash_forward"
                        ][
                            output_start:output_end
                        ] = expected_forward_chunk[
                            mask
                        ]

                        target[
                            "hash_reverse"
                        ][
                            output_start:output_end
                        ] = expected_reverse_chunk[
                            mask
                        ]

                        offsets[
                            split_name
                        ] = output_end

                    selected_count += (
                        right - left
                    )

                base = end

                now = time.perf_counter()

                if (
                    now - last_progress
                    >= 60.0
                ):
                    print(
                        f"[cache] "
                        f"file={file_position}/"
                        f"{len(source_files)} | "
                        f"{path.name} | "
                        f"{base:,}/"
                        f"{source['expected_row_count']:,}",
                        flush=True,
                    )

                    last_progress = now

            if base != int(
                source[
                    "expected_row_count"
                ]
            ):
                raise RuntimeError(
                    "Source row count mismatch: "
                    f"{path.name} | "
                    f"{base:,} != "
                    f"{source['expected_row_count']:,}"
                )

            if selected_count != len(
                representative_rows
            ):
                raise RuntimeError(
                    "Representative selection count mismatch: "
                    f"file_id={source['file_id']} | "
                    f"{selected_count:,} != "
                    f"{len(representative_rows):,}"
                )

            per_file_records.append(
                {
                    "file_id": int(
                        source["file_id"]
                    ),
                    "file_path": str(path),
                    "raw_label": str(
                        source["class_label"]
                    ),
                    "source_rows": int(base),
                    "representative_rows": int(
                        selected_count
                    ),
                }
            )

            print(
                f"[cache] "
                f"file={file_position}/"
                f"{len(source_files)} complete | "
                f"source_rows={base:,} | "
                f"representatives="
                f"{selected_count:,} | "
                f"{path.name}",
                flush=True,
            )

        for split_name in SPLITS:
            expected_count = (
                EXPECTED_SPLIT_COUNTS[
                    split_name
                ][
                    "fingerprints"
                ]
            )

            if offsets[
                split_name
            ] != expected_count:
                raise RuntimeError(
                    "Final split offset mismatch: "
                    f"{split_name} | "
                    f"{offsets[split_name]:,} "
                    f"!= {expected_count:,}"
                )

            for array in arrays[
                split_name
            ].values():
                array.flush()

    finally:
        arrays.clear()

    return {
        "offsets": offsets,
        "per_file": per_file_records,
    }


def validate_unscaled_cache() -> dict[str, Any]:
    split_records: list[
        dict[str, Any]
    ] = []

    total_rows = 0
    total_raw_rows = 0

    for split_name in SPLITS:
        expected = EXPECTED_SPLIT_COUNTS[
            split_name
        ]

        x = np.load(
            BUILD_CACHE
            / f"X_{split_name}.npy",
            mmap_mode="r",
        )

        y = np.load(
            BUILD_CACHE
            / f"y_{split_name}.npy",
            mmap_mode="r",
        )

        raw_counts = np.load(
            BUILD_CACHE
            / (
                "raw_row_count_"
                f"{split_name}.npy"
            ),
            mmap_mode="r",
        )

        hash_forward = np.load(
            BUILD_CACHE
            / (
                "hash_forward_"
                f"{split_name}.npy"
            ),
            mmap_mode="r",
        )

        hash_reverse = np.load(
            BUILD_CACHE
            / (
                "hash_reverse_"
                f"{split_name}.npy"
            ),
            mmap_mode="r",
        )

        row_count = int(
            expected["fingerprints"]
        )

        shape_checks = {
            "X": (
                x.shape
                == (
                    row_count,
                    FEATURE_COUNT,
                )
                and x.dtype
                == np.float32
            ),
            "y": (
                y.shape == (row_count,)
                and y.dtype == np.int8
            ),
            "raw_row_count": (
                raw_counts.shape
                == (row_count,)
                and raw_counts.dtype
                == np.int64
            ),
            "hash_forward": (
                hash_forward.shape
                == (row_count,)
                and hash_forward.dtype
                == np.int64
            ),
            "hash_reverse": (
                hash_reverse.shape
                == (row_count,)
                and hash_reverse.dtype
                == np.int64
            ),
        }

        if not all(
            shape_checks.values()
        ):
            raise RuntimeError(
                "Cache shape/dtype validation failed: "
                f"{split_name} -> {shape_checks}"
            )

        observed_class_groups = (
            np.bincount(
                np.asarray(
                    y,
                    dtype=np.int64,
                ),
                minlength=3,
            )
            .astype(np.int64)
        )

        observed_class_raw_rows = np.bincount(
            np.asarray(
                y,
                dtype=np.int64,
            ),
            weights=np.asarray(
                raw_counts,
                dtype=np.float64,
            ),
            minlength=3,
        ).astype(np.int64)

        expected_group_counts = np.asarray(
            [
                EXPECTED_FAMILY_SPLIT_COUNTS[
                    (
                        split_name,
                        family_name,
                    )
                ][0]
                for family_name in (
                    "benign",
                    "gafgyt",
                    "mirai",
                )
            ],
            dtype=np.int64,
        )

        expected_class_raw_rows = np.asarray(
            [
                EXPECTED_FAMILY_SPLIT_COUNTS[
                    (
                        split_name,
                        family_name,
                    )
                ][1]
                for family_name in (
                    "benign",
                    "gafgyt",
                    "mirai",
                )
            ],
            dtype=np.int64,
        )

        if not np.array_equal(
            observed_class_groups,
            expected_group_counts,
        ):
            raise RuntimeError(
                "Fingerprint class counts mismatch: "
                f"{split_name}"
            )

        if not np.array_equal(
            observed_class_raw_rows,
            expected_class_raw_rows,
        ):
            raise RuntimeError(
                "Raw-row class counts mismatch: "
                f"{split_name}"
            )

        observed_raw_rows = int(
            np.asarray(
                raw_counts,
                dtype=np.int64,
            ).sum()
        )

        if observed_raw_rows != int(
            expected["raw_rows"]
        ):
            raise RuntimeError(
                "Split raw-row sum mismatch: "
                f"{split_name}"
            )

        if np.any(
            np.asarray(
                raw_counts,
                dtype=np.int64,
            )
            <= 0
        ):
            raise RuntimeError(
                "Non-positive raw_row_count found: "
                f"{split_name}"
            )

        finite = True
        hash_mismatch_count = 0
        minimum_value = math.inf
        maximum_value = -math.inf

        for start in range(
            0,
            row_count,
            CHUNK_SIZE,
        ):
            end = min(
                start + CHUNK_SIZE,
                row_count,
            )

            values = np.asarray(
                x[start:end],
                dtype=np.float32,
            )

            finite = (
                finite
                and bool(
                    np.isfinite(
                        values
                    ).all()
                )
            )

            minimum_value = min(
                minimum_value,
                float(
                    np.min(values)
                ),
            )

            maximum_value = max(
                maximum_value,
                float(
                    np.max(values)
                ),
            )

            actual_forward, actual_reverse = (
                float32_fingerprints(
                    values
                )
            )

            hash_mismatch_count += int(
                np.count_nonzero(
                    actual_forward
                    != np.asarray(
                        hash_forward[
                            start:end
                        ],
                        dtype=np.int64,
                    )
                )
            )

            hash_mismatch_count += int(
                np.count_nonzero(
                    actual_reverse
                    != np.asarray(
                        hash_reverse[
                            start:end
                        ],
                        dtype=np.int64,
                    )
                )
            )

        if not finite:
            raise RuntimeError(
                "Non-finite cache value found: "
                f"{split_name}"
            )

        if hash_mismatch_count != 0:
            raise RuntimeError(
                "Cache fingerprint mismatch: "
                f"{split_name} -> "
                f"{hash_mismatch_count:,}"
            )

        split_records.append(
            {
                "split": split_name,
                "fingerprint_count": (
                    row_count
                ),
                "raw_row_count": (
                    observed_raw_rows
                ),
                "class_fingerprint_counts": (
                    observed_class_groups.tolist()
                ),
                "class_raw_row_counts": (
                    observed_class_raw_rows.tolist()
                ),
                "all_values_finite": (
                    finite
                ),
                "minimum_value": (
                    minimum_value
                ),
                "maximum_value": (
                    maximum_value
                ),
                "hash_mismatch_count": (
                    hash_mismatch_count
                ),
                "shape_dtype_checks": (
                    shape_checks
                ),
            }
        )

        total_rows += row_count
        total_raw_rows += observed_raw_rows

    checks = {
        "total_fingerprint_count_matches": (
            total_rows
            == EXPECTED_GROUP_COUNT
        ),
        "total_raw_row_count_matches": (
            total_raw_rows
            == EXPECTED_RAW_ROW_COUNT
        ),
        "all_split_hashes_match": all(
            row["hash_mismatch_count"]
            == 0
            for row in split_records
        ),
        "all_values_finite": all(
            row["all_values_finite"]
            for row in split_records
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "Unscaled cache validation failed: "
            + ", ".join(failed)
        )

    return {
        "splits": split_records,
        "total_fingerprint_count": (
            total_rows
        ),
        "total_raw_row_count": (
            total_raw_rows
        ),
        "checks": checks,
    }


def fit_and_audit_scaler() -> dict[str, Any]:
    x_train = np.load(
        BUILD_CACHE / "X_train.npy",
        mmap_mode="r",
    )

    scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )

    for start in range(
        0,
        len(x_train),
        CHUNK_SIZE,
    ):
        scaler.partial_fit(
            np.asarray(
                x_train[
                    start:start + CHUNK_SIZE
                ],
                dtype=np.float32,
            )
        )

    mean = np.asarray(
        scaler.mean_,
        dtype=np.float64,
    )

    scale = np.asarray(
        scaler.scale_,
        dtype=np.float64,
    )

    if (
        mean.shape != (FEATURE_COUNT,)
        or scale.shape
        != (FEATURE_COUNT,)
    ):
        raise RuntimeError(
            "Scaler parameter shape mismatch."
        )

    if (
        not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise RuntimeError(
            "Invalid scaler parameter values."
        )

    np.save(
        BUILD_CACHE
        / "scaler_mean.npy",
        mean,
    )

    np.save(
        BUILD_CACHE
        / "scaler_scale.npy",
        scale,
    )

    scaled_sum = np.zeros(
        FEATURE_COUNT,
        dtype=np.float64,
    )

    scaled_square = np.zeros(
        FEATURE_COUNT,
        dtype=np.float64,
    )

    all_scaled_values_finite = True

    for start in range(
        0,
        len(x_train),
        CHUNK_SIZE,
    ):
        raw = np.asarray(
            x_train[
                start:start + CHUNK_SIZE
            ],
            dtype=np.float64,
        )

        scaled = (
            raw - mean
        ) / scale

        all_scaled_values_finite = (
            all_scaled_values_finite
            and bool(
                np.isfinite(
                    scaled
                ).all()
            )
        )

        scaled_sum += scaled.sum(
            axis=0
        )

        scaled_square += np.square(
            scaled
        ).sum(
            axis=0
        )

    scaled_mean = (
        scaled_sum / len(x_train)
    )

    scaled_variance = np.maximum(
        scaled_square / len(x_train)
        - np.square(scaled_mean),
        0.0,
    )

    scaled_std = np.sqrt(
        scaled_variance
    )

    maximum_absolute_mean = float(
        np.max(
            np.abs(
                scaled_mean
            )
        )
    )

    maximum_std_error = float(
        np.max(
            np.abs(
                scaled_std - 1.0
            )
        )
    )

    passed = bool(
        all_scaled_values_finite
        and maximum_absolute_mean
        <= SCALER_AUDIT_TOLERANCE
        and maximum_std_error
        <= SCALER_AUDIT_TOLERANCE
    )

    if not passed:
        raise RuntimeError(
            "Train-only scaler audit failed: "
            f"mean={maximum_absolute_mean:.3e}, "
            f"std_error={maximum_std_error:.3e}"
        )

    return {
        "fit_split": "train",
        "fit_unit": (
            "one representative per "
            "VS2 float32 fingerprint"
        ),
        "sample_weight": None,
        "method": (
            "sklearn.preprocessing."
            "StandardScaler.partial_fit"
        ),
        "parameter_dtype": "float64",
        "scaled_model_input_dtype": (
            "float32"
        ),
        "train_sample_count": int(
            len(x_train)
        ),
        "all_scaled_values_finite": (
            all_scaled_values_finite
        ),
        "maximum_absolute_scaled_mean": (
            maximum_absolute_mean
        ),
        "maximum_scaled_standard_deviation_error": (
            maximum_std_error
        ),
        "tolerance": (
            SCALER_AUDIT_TOLERANCE
        ),
        "passed": passed,
    }


def audit_scaled_overlap() -> dict[str, Any]:
    audit_database = (
        BUILD_CACHE
        / "scaled_model_input_overlap_audit.sqlite"
    )

    mean = np.load(
        BUILD_CACHE
        / "scaler_mean.npy"
    )

    scale = np.load(
        BUILD_CACHE
        / "scaler_scale.npy"
    )

    if audit_database.exists():
        audit_database.unlink()

    connection = sqlite3.connect(
        audit_database,
        timeout=120.0,
    )

    started = time.perf_counter()

    try:
        connection.execute(
            "PRAGMA journal_mode=OFF"
        )

        connection.execute(
            "PRAGMA synchronous=OFF"
        )

        connection.execute(
            "PRAGMA temp_store=FILE"
        )

        connection.execute(
            """
            CREATE TABLE scaled_fingerprints (
                hash_forward INTEGER NOT NULL,
                hash_reverse INTEGER NOT NULL,
                split_code INTEGER NOT NULL,
                family_min INTEGER NOT NULL,
                family_max INTEGER NOT NULL,
                row_count INTEGER NOT NULL,

                PRIMARY KEY (
                    hash_forward,
                    hash_reverse,
                    split_code
                )
            ) WITHOUT ROWID
            """
        )

        insert_sql = """
            INSERT INTO scaled_fingerprints (
                hash_forward,
                hash_reverse,
                split_code,
                family_min,
                family_max,
                row_count
            )
            VALUES (?, ?, ?, ?, ?, 1)

            ON CONFLICT (
                hash_forward,
                hash_reverse,
                split_code
            )
            DO UPDATE SET
                family_min = MIN(
                    family_min,
                    excluded.family_min
                ),
                family_max = MAX(
                    family_max,
                    excluded.family_max
                ),
                row_count = row_count + 1
        """

        processed_rows = 0
        unique_by_split: dict[
            str,
            int
        ] = {}

        for split_code, split_name in enumerate(
            SPLITS
        ):
            x = np.load(
                BUILD_CACHE
                / f"X_{split_name}.npy",
                mmap_mode="r",
            )

            y = np.load(
                BUILD_CACHE
                / f"y_{split_name}.npy",
                mmap_mode="r",
            )

            for start in range(
                0,
                len(x),
                CHUNK_SIZE,
            ):
                end = min(
                    start + CHUNK_SIZE,
                    len(x),
                )

                raw = np.asarray(
                    x[start:end],
                    dtype=np.float64,
                )

                scaled = (
                    (
                        raw - mean
                    )
                    / scale
                ).astype(
                    np.float32
                )

                if not np.isfinite(
                    scaled
                ).all():
                    raise RuntimeError(
                        "Non-finite scaled input: "
                        f"{split_name}"
                    )

                forward, reverse = (
                    float32_fingerprints(
                        scaled
                    )
                )

                families = np.asarray(
                    y[start:end],
                    dtype=np.int64,
                )

                connection.executemany(
                    insert_sql,
                    zip(
                        forward.tolist(),
                        reverse.tolist(),
                        [split_code]
                        * len(scaled),
                        families.tolist(),
                        families.tolist(),
                    ),
                )

                processed_rows += len(
                    scaled
                )

            connection.commit()

            unique_by_split[
                split_name
            ] = scalar(
                connection,
                """
                SELECT COUNT(*)
                FROM scaled_fingerprints
                WHERE split_code = ?
                """,
                (split_code,),
            )

            print(
                "[scaled overlap audit] "
                f"{split_name} complete | "
                f"rows={len(x):,} | "
                f"unique_scaled="
                f"{unique_by_split[split_name]:,}",
                flush=True,
            )

        cross_split_overlap = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT
                    hash_forward,
                    hash_reverse
                FROM scaled_fingerprints
                GROUP BY
                    hash_forward,
                    hash_reverse
                HAVING COUNT(*) > 1
            )
            """,
        )

        within_split_family_conflicts = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM scaled_fingerprints
            WHERE family_min
                  != family_max
            """,
        )

        global_family_conflicts = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT
                    hash_forward,
                    hash_reverse,
                    MIN(family_min)
                        AS minimum_family,
                    MAX(family_max)
                        AS maximum_family
                FROM scaled_fingerprints
                GROUP BY
                    hash_forward,
                    hash_reverse
                HAVING minimum_family
                       != maximum_family
            )
            """,
        )

        unique_total = scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM scaled_fingerprints
            """,
        )

        collapsed_within_split_rows = (
            processed_rows - unique_total
        )

        quick_check = str(
            connection.execute(
                "PRAGMA quick_check"
            ).fetchone()[0]
        )

        if quick_check.lower() != "ok":
            raise RuntimeError(
                "Scaled overlap audit database "
                f"quick_check failed: {quick_check}"
            )

    finally:
        connection.close()

    checks = {
        "processed_rows_match": (
            processed_rows
            == EXPECTED_GROUP_COUNT
        ),
        "all_splits_present": (
            set(unique_by_split)
            == set(SPLITS)
        ),
        "cross_split_overlap_zero": (
            cross_split_overlap == 0
        ),
        "within_split_family_conflicts_zero": (
            within_split_family_conflicts
            == 0
        ),
        "global_family_conflicts_zero": (
            global_family_conflicts == 0
        ),
        "audit_database_quick_check_ok": (
            quick_check.lower() == "ok"
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    result = {
        "feature_space": (
            "train-only StandardScaler "
            "float64 parameters then "
            "final float32 model input"
        ),
        "processed_fingerprint_rows": (
            processed_rows
        ),
        "unique_scaled_fingerprints_by_split": (
            unique_by_split
        ),
        "collapsed_within_split_rows": (
            collapsed_within_split_rows
        ),
        "cross_split_model_input_fingerprint_count": (
            cross_split_overlap
        ),
        "within_split_family_conflict_count": (
            within_split_family_conflicts
        ),
        "global_family_conflict_count": (
            global_family_conflicts
        ),
        "audit_database_quick_check": (
            quick_check
        ),
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
        "checks": checks,
        "passed": not failed,
    }

    if failed:
        atomic_json(
            BUILD_CACHE
            / "scaled_overlap_failure.json",
            result,
        )

        raise RuntimeError(
            "Scaled model-input overlap audit failed: "
            + ", ".join(failed)
        )

    return result


def artifact_inventory() -> list[
    dict[str, Any]
]:
    paths = sorted(
        path
        for path in BUILD_CACHE.iterdir()
        if path.is_file()
        and path.name not in {
            "manifest.json",
            "build_status.json",
        }
    )

    records: list[
        dict[str, Any]
    ] = []

    for position, path in enumerate(
        paths,
        start=1,
    ):
        print(
            f"[artifact hash] "
            f"{position}/{len(paths)} | "
            f"{path.name}",
            flush=True,
        )

        records.append(
            {
                "relative_path": (
                    path.name
                ),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "sha256": sha256_file(
                    path
                ),
            }
        )

    return records


def main() -> None:
    started = time.perf_counter()

    print("=" * 92)
    print("FAMILY-3 UNIQUE-FINGERPRINT TABULAR CACHE BUILD")
    print("=" * 92)
    print(f"Protocol database : {PROTOCOL_DATABASE}")
    print(f"Locked protocol   : {LOCKED_PROTOCOL}")
    print(f"Final cache       : {FINAL_CACHE}")
    print(f"Build directory   : {BUILD_CACHE}")
    print(f"Chunk size        : {CHUNK_SIZE:,}")

    locked_inputs = (
        validate_locked_inputs()
    )

    free_disk_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        / (1024**3)
    )

    if free_disk_gib < MINIMUM_FREE_DISK_GIB:
        raise RuntimeError(
            "Insufficient free disk space: "
            f"{free_disk_gib:.3f} GiB < "
            f"{MINIMUM_FREE_DISK_GIB:.3f} GiB"
        )

    protocol_database_sha256_before = (
        sha256_file(
            PROTOCOL_DATABASE
        )
    )

    expected_database_sha256 = (
        locked_inputs[
            "protocol"
        ][
            "data_policy"
        ][
            "split_database_sha256"
        ]
    )

    if (
        protocol_database_sha256_before
        != expected_database_sha256
    ):
        raise RuntimeError(
            "Protocol database SHA-256 no longer "
            "matches the locked protocol."
        )

    BUILD_CACHE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    BUILD_CACHE.mkdir()

    building_marker = {
        "status": "building",
        "started_at_utc": utc_now(),
        "builder_script": str(
            Path(__file__).resolve()
        ),
        "builder_script_sha256": (
            sha256_file(
                Path(__file__).resolve()
            )
        ),
        "locked_protocol_sha256": (
            locked_inputs[
                "protocol_sha256"
            ]
        ),
        "protocol_database_sha256": (
            protocol_database_sha256_before
        ),
    }

    atomic_json(
        BUILD_CACHE
        / "build_status.json",
        building_marker,
    )

    database_uri = (
        PROTOCOL_DATABASE.resolve().as_uri()
        + "?mode=ro"
    )

    with sqlite3.connect(
        database_uri,
        uri=True,
        timeout=120.0,
    ) as connection:
        database_validation = (
            validate_database(
                connection
            )
        )

        print(
            "Protocol database validated.",
            flush=True,
        )

        source_files = (
            validate_source_files(
                connection
            )
        )

        print(
            "All source CSV SHA-256 values validated.",
            flush=True,
        )

        representative_validation = (
            create_representative_table(
                connection
            )
        )

        materialization = (
            materialize_cache(
                connection=connection,
                source_files=source_files,
            )
        )

    protocol_database_sha256_after = (
        sha256_file(
            PROTOCOL_DATABASE
        )
    )

    if (
        protocol_database_sha256_before
        != protocol_database_sha256_after
    ):
        raise RuntimeError(
            "Protocol database changed during "
            "the read-only cache build."
        )

    print(
        "Validating unscaled cache...",
        flush=True,
    )

    unscaled_validation = (
        validate_unscaled_cache()
    )

    print(
        "Fitting train-only unique-fingerprint scaler...",
        flush=True,
    )

    scaler_audit = (
        fit_and_audit_scaler()
    )

    print(
        "Auditing final scaled float32 overlap...",
        flush=True,
    )

    scaled_overlap_audit = (
        audit_scaled_overlap()
    )

    artifacts = artifact_inventory()

    completed_at = utc_now()

    manifest = {
        "status": "completed",
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "created_at_utc": completed_at,
        "cache_role": (
            "family_3 classical tabular "
            "baseline cache"
        ),
        "cache_directory": str(
            FINAL_CACHE
        ),
        "assignment_unit": (
            "one representative row per "
            "paired VS2 float32 fingerprint"
        ),
        "feature_count": FEATURE_COUNT,
        "feature_dtype": "float32",
        "label_dtype": "int8",
        "raw_row_count_dtype": "int64",
        "hash_dtype": "int64",
        "classes": [
            "benign",
            "gafgyt",
            "mirai",
        ],
        "training_sample_weight": None,
        "primary_evaluation_view": (
            "fingerprint_level"
        ),
        "secondary_evaluation_view": (
            "raw-record weighted by "
            "raw_row_count"
        ),
        "locked_protocol": str(
            LOCKED_PROTOCOL
        ),
        "locked_protocol_sha256": (
            locked_inputs[
                "protocol_sha256"
            ]
        ),
        "lock_manifest": str(
            LOCK_MANIFEST
        ),
        "lock_manifest_sha256": (
            sha256_file(
                LOCK_MANIFEST
            )
        ),
        "run_matrix": str(
            locked_inputs[
                "run_matrix_path"
            ]
        ),
        "run_matrix_sha256": (
            sha256_file(
                locked_inputs[
                    "run_matrix_path"
                ]
            )
        ),
        "protocol_database": str(
            PROTOCOL_DATABASE
        ),
        "protocol_database_sha256_before": (
            protocol_database_sha256_before
        ),
        "protocol_database_sha256_after": (
            protocol_database_sha256_after
        ),
        "protocol_database_unchanged": True,
        "free_disk_before_build_gib": (
            free_disk_gib
        ),
        "database_validation": (
            database_validation
        ),
        "source_integrity": {
            "source_file_count": len(
                source_files
            ),
            "all_source_sha256_values_match": (
                True
            ),
            "files": source_files,
        },
        "representative_validation": (
            representative_validation
        ),
        "materialization": (
            materialization
        ),
        "unscaled_cache_validation": (
            unscaled_validation
        ),
        "scaler_audit": scaler_audit,
        "scaled_model_input_overlap_audit": (
            scaled_overlap_audit
        ),
        "artifacts": artifacts,
        "builder_script": str(
            Path(__file__).resolve()
        ),
        "builder_script_sha256": (
            sha256_file(
                Path(__file__).resolve()
            )
        ),
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
        "all_checks_passed": True,
    }

    atomic_json(
        BUILD_CACHE
        / "manifest.json",
        manifest,
    )

    (BUILD_CACHE / "build_status.json").unlink()

    BUILD_CACHE.rename(
        FINAL_CACHE
    )

    report = {
        "status": "completed",
        "generated_at_utc": (
            completed_at
        ),
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "cache_directory": str(
            FINAL_CACHE
        ),
        "cache_manifest": str(
            FINAL_CACHE
            / "manifest.json"
        ),
        "cache_manifest_sha256": (
            sha256_file(
                FINAL_CACHE
                / "manifest.json"
            )
        ),
        "fingerprint_count": (
            EXPECTED_GROUP_COUNT
        ),
        "represented_raw_row_count": (
            EXPECTED_RAW_ROW_COUNT
        ),
        "scaled_cross_split_overlap_count": (
            scaled_overlap_audit[
                "cross_split_model_input_fingerprint_count"
            ]
        ),
        "scaled_within_split_family_conflict_count": (
            scaled_overlap_audit[
                "within_split_family_conflict_count"
            ]
        ),
        "scaled_global_family_conflict_count": (
            scaled_overlap_audit[
                "global_family_conflict_count"
            ]
        ),
        "all_checks_passed": True,
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
    }

    atomic_json(
        OUTPUT_REPORT,
        report,
    )

    cache_size_bytes = sum(
        path.stat().st_size
        for path in FINAL_CACHE.iterdir()
        if path.is_file()
    )

    print()
    print("=" * 92)
    print("TABULAR CACHE BUILD SUMMARY")
    print("=" * 92)
    print(
        "Fingerprint rows              : "
        f"{EXPECTED_GROUP_COUNT:,}"
    )
    print(
        "Represented raw rows          : "
        f"{EXPECTED_RAW_ROW_COUNT:,}"
    )

    for split_name in SPLITS:
        print(
            f"{split_name:<10} fingerprints       : "
            f"{EXPECTED_SPLIT_COUNTS[split_name]['fingerprints']:,}"
        )

    print(
        "Scaled cross-split overlap    : "
        f"{scaled_overlap_audit['cross_split_model_input_fingerprint_count']:,}"
    )
    print(
        "Scaled within-split conflicts : "
        f"{scaled_overlap_audit['within_split_family_conflict_count']:,}"
    )
    print(
        "Scaled global conflicts       : "
        f"{scaled_overlap_audit['global_family_conflict_count']:,}"
    )
    print(
        "Cache size                    : "
        f"{cache_size_bytes / (1024**3):.3f} GiB"
    )
    print(
        "Cache directory               : "
        f"{FINAL_CACHE}"
    )
    print(
        "Manifest                      : "
        f"{FINAL_CACHE / 'manifest.json'}"
    )
    print(
        "Report                        : "
        f"{OUTPUT_REPORT}"
    )
    print(
        "Elapsed                       : "
        f"{(time.perf_counter() - started) / 60.0:.2f} minutes"
    )
    print(
        "All checks passed             : True"
    )
    print("TABULAR BASELINE CACHE BUILD COMPLETED")


if __name__ == "__main__":
    main()
