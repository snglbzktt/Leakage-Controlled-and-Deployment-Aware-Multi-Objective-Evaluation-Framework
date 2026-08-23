from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path.cwd()

CANONICAL_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "nbaiot_float32_canonical_seed2026"
)

METADATA_PARQUET = CANONICAL_ROOT / "metadata.parquet"

PROTOCOL_DATABASE = (
    ROOT
    / "data"
    / "splits"
    / "vs2_float32_leakage_ablation_scaled_repaired_seed2026_v3.sqlite"
)

AUDIT_DATABASE = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "vs2_canonical_rehash_alignment_work_v3.sqlite"
)

OUTPUT_JSON = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "vs2_canonical_rehash_alignment_summary_v3.json"
)

SPLITS = ("train", "validation", "test")

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

EXPECTED_FEATURE_COUNT = 115
BATCH_SIZE = 50_000


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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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

    os.replace(temporary, path)


def family_code_from_class_name(class_name: str) -> int:
    normalized = str(class_name).strip().lower()

    if normalized == "benign":
        return 0

    if normalized.startswith("gafgyt_"):
        return 1

    if normalized.startswith("mirai_"):
        return 2

    raise RuntimeError(
        f"Unknown canonical class: {class_name!r}"
    )


def float32_fingerprints(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the exact VS2 paired float32 fingerprint."""
    matrix = np.array(
        values,
        dtype="<f4",
        order="C",
        copy=True,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[1] != EXPECTED_FEATURE_COUNT
    ):
        raise RuntimeError(
            "Expected a (*, 115) feature matrix; "
            f"found {matrix.shape}"
        )

    if not np.isfinite(matrix).all():
        raise RuntimeError(
            "Non-finite value found in canonical float32 features."
        )

    matrix[matrix == 0.0] = 0.0

    words = matrix.view("<u4").reshape(
        matrix.shape
    )

    row_count = matrix.shape[0]

    forward = np.full(
        row_count,
        np.uint64(0xCBF29CE484222325),
        dtype=np.uint64,
    )

    reverse = np.full(
        row_count,
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
        forward.view(np.int64),
        reverse.view(np.int64),
    )


def scalar(
    connection: sqlite3.Connection,
    query: str,
) -> int:
    return int(
        connection.execute(query).fetchone()[0]
    )


def rows_as_dicts(
    connection: sqlite3.Connection,
    query: str,
) -> list[dict[str, Any]]:
    cursor = connection.execute(query)
    column_names = [
        item[0]
        for item in cursor.description
    ]

    return [
        dict(zip(column_names, row))
        for row in cursor.fetchall()
    ]


required_paths = [
    METADATA_PARQUET,
    PROTOCOL_DATABASE,
]

for split_name in SPLITS:
    required_paths.append(
        CANONICAL_ROOT
        / f"{split_name}.parquet"
    )

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

AUDIT_DATABASE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if AUDIT_DATABASE.exists():
    AUDIT_DATABASE.unlink()

protocol_sha256_before = sha256_file(
    PROTOCOL_DATABASE
)

protocol_uri = (
    PROTOCOL_DATABASE.resolve().as_uri()
    + "?mode=ro"
)

started = time.perf_counter()

print("=" * 92)
print("VS2 REHASHED CANONICAL ALIGNMENT AUDIT")
print("=" * 92)
print(f"Canonical root    : {CANONICAL_ROOT}")
print(f"Protocol database : {PROTOCOL_DATABASE}")
print(f"Audit database    : {AUDIT_DATABASE}")
print(f"Batch size        : {BATCH_SIZE:,}")

with sqlite3.connect(
    protocol_uri,
    uri=True,
    timeout=120.0,
) as connection:
    connection.execute(
        "PRAGMA temp_store=FILE"
    )

    connection.execute(
        "PRAGMA cache_size=-250000"
    )

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

    connection.execute(
        "ATTACH DATABASE ? AS audit",
        (str(AUDIT_DATABASE),),
    )

    connection.execute(
        "PRAGMA audit.journal_mode=OFF"
    )

    connection.execute(
        "PRAGMA audit.synchronous=OFF"
    )

    connection.execute(
        """
        CREATE TABLE audit.canonical_rehashed (
            hash_forward INTEGER NOT NULL,
            hash_reverse INTEGER NOT NULL,
            canonical_split INTEGER NOT NULL,
            family_label INTEGER NOT NULL,
            raw_occurrence_count INTEGER NOT NULL,
            source_split TEXT NOT NULL,
            source_row INTEGER NOT NULL
        )
        """
    )

    insert_sql = """
        INSERT INTO audit.canonical_rehashed (
            hash_forward,
            hash_reverse,
            canonical_split,
            family_label,
            raw_occurrence_count,
            source_split,
            source_row
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    inserted_total = 0

    for split_name in SPLITS:
        split_file = (
            CANONICAL_ROOT
            / f"{split_name}.parquet"
        )

        parquet_file = pq.ParquetFile(
            split_file
        )

        feature_columns = [
            field.name
            for field in parquet_file.schema_arrow
            if field.name != "class_label"
        ]

        if (
            len(feature_columns)
            != EXPECTED_FEATURE_COUNT
        ):
            raise RuntimeError(
                f"{split_name}: expected "
                f"{EXPECTED_FEATURE_COUNT} features; "
                f"found {len(feature_columns)}"
            )

        metadata_table = pq.read_table(
            METADATA_PARQUET,
            columns=[
                "raw_occurrence_count",
            ],
            filters=[
                ("split", "=", split_name),
            ],
        )

        occurrence_counts = (
            metadata_table[
                "raw_occurrence_count"
            ]
            .combine_chunks()
            .to_numpy(
                zero_copy_only=False
            )
            .astype(
                np.int64,
                copy=False,
            )
        )

        expected_rows = int(
            parquet_file.metadata.num_rows
        )

        if (
            len(occurrence_counts)
            != expected_rows
        ):
            raise RuntimeError(
                f"{split_name}: metadata/parquet "
                f"row mismatch: "
                f"{len(occurrence_counts):,} != "
                f"{expected_rows:,}"
            )

        print()
        print(
            f"[{split_name}] "
            f"rows={expected_rows:,}"
        )

        offset = 0

        for batch_number, batch in enumerate(
            parquet_file.iter_batches(
                batch_size=BATCH_SIZE,
                columns=(
                    feature_columns
                    + ["class_label"]
                ),
            ),
            start=1,
        ):
            frame = batch.to_pandas()

            values = frame[
                feature_columns
            ].to_numpy(
                dtype=np.float32,
                copy=False,
            )

            labels = frame[
                "class_label"
            ].astype(str).to_numpy()

            forward, reverse = (
                float32_fingerprints(values)
            )

            batch_rows = len(frame)
            end = offset + batch_rows

            batch_occurrences = (
                occurrence_counts[offset:end]
            )

            family_codes = np.fromiter(
                (
                    family_code_from_class_name(
                        label
                    )
                    for label in labels
                ),
                dtype=np.int8,
                count=batch_rows,
            )

            source_rows = np.arange(
                offset,
                end,
                dtype=np.int64,
            )

            records = zip(
                forward.tolist(),
                reverse.tolist(),
                [SPLIT_CODES[split_name]]
                * batch_rows,
                family_codes.astype(
                    np.int64,
                    copy=False,
                ).tolist(),
                batch_occurrences.tolist(),
                [split_name] * batch_rows,
                source_rows.tolist(),
            )

            connection.executemany(
                insert_sql,
                records,
            )

            offset = end
            inserted_total += batch_rows

            print(
                f"  batch={batch_number:02d} "
                f"split_rows={offset:,}/"
                f"{expected_rows:,} "
                f"total={inserted_total:,}",
                flush=True,
            )

        if offset != expected_rows:
            raise RuntimeError(
                f"{split_name}: processed rows "
                f"{offset:,} != {expected_rows:,}"
            )

        connection.commit()

    print()
    print("Building rehashed fingerprint summary...")

    connection.execute(
        """
        CREATE INDEX audit.idx_canonical_rehashed_hash
        ON canonical_rehashed (
            hash_forward,
            hash_reverse
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE audit.rehashed_stats AS
        SELECT
            hash_forward,
            hash_reverse,
            COUNT(*) AS canonical_row_count,
            MIN(canonical_split) AS canonical_split_min,
            MAX(canonical_split) AS canonical_split_max,
            MIN(family_label) AS family_label_min,
            MAX(family_label) AS family_label_max,
            SUM(raw_occurrence_count) AS raw_occurrence_count
        FROM audit.canonical_rehashed
        GROUP BY
            hash_forward,
            hash_reverse
        """
    )

    connection.execute(
        """
        CREATE UNIQUE INDEX
        audit.idx_rehashed_stats_hash
        ON rehashed_stats (
            hash_forward,
            hash_reverse
        )
        """
    )

    connection.commit()

    canonical_row_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM audit.canonical_rehashed
        """,
    )

    rehashed_group_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM audit.rehashed_stats
        """,
    )

    duplicate_rehash_group_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM audit.rehashed_stats
        WHERE canonical_row_count > 1
        """,
    )

    duplicate_rehash_excess_rows = scalar(
        connection,
        """
        SELECT COALESCE(
            SUM(canonical_row_count - 1),
            0
        )
        FROM audit.rehashed_stats
        WHERE canonical_row_count > 1
        """,
    )

    rehash_cross_split_group_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM audit.rehashed_stats
        WHERE canonical_split_min
              != canonical_split_max
        """,
    )

    rehash_cross_family_group_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM audit.rehashed_stats
        WHERE family_label_min
              != family_label_max
        """,
    )

    protocol_group_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprint_stats
        """,
    )

    common_group_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprint_stats f
        JOIN audit.rehashed_stats c
          ON c.hash_forward = f.hash_forward
         AND c.hash_reverse = f.hash_reverse
        """,
    )

    protocol_only_group_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprint_stats f
        LEFT JOIN audit.rehashed_stats c
          ON c.hash_forward = f.hash_forward
         AND c.hash_reverse = f.hash_reverse
        WHERE c.hash_forward IS NULL
        """,
    )

    canonical_only_group_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM audit.rehashed_stats c
        LEFT JOIN fingerprint_stats f
          ON f.hash_forward = c.hash_forward
         AND f.hash_reverse = c.hash_reverse
        WHERE f.hash_forward IS NULL
        """,
    )

    family_mismatch_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprint_stats f
        JOIN audit.rehashed_stats c
          ON c.hash_forward = f.hash_forward
         AND c.hash_reverse = f.hash_reverse
        WHERE c.family_label_min
              != f.family_label_min
           OR c.family_label_max
              != f.family_label_max
        """,
    )

    occurrence_mismatch_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprint_stats f
        JOIN audit.rehashed_stats c
          ON c.hash_forward = f.hash_forward
         AND c.hash_reverse = f.hash_reverse
        WHERE c.raw_occurrence_count
              != f.raw_row_count
        """,
    )

    canonical_split_vs_protocol_mismatch = (
        scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM fingerprint_stats f
            JOIN audit.rehashed_stats c
              ON c.hash_forward = f.hash_forward
             AND c.hash_reverse = f.hash_reverse
            WHERE c.canonical_split_min
                  != f.grouped_split
               OR c.canonical_split_max
                  != f.grouped_split
            """,
        )
    )

    common_raw_rows = scalar(
        connection,
        """
        SELECT COALESCE(
            SUM(f.raw_row_count),
            0
        )
        FROM fingerprint_stats f
        JOIN audit.rehashed_stats c
          ON c.hash_forward = f.hash_forward
         AND c.hash_reverse = f.hash_reverse
        """,
    )

    protocol_only_rows = rows_as_dicts(
        connection,
        """
        SELECT
            f.hash_forward,
            f.hash_reverse,
            f.raw_row_count,
            f.grouped_split,
            f.family_label_min,
            f.family_label_max,

            (
                SELECT a.file_id
                FROM assignments a
                WHERE
                    a.hash_forward
                    = f.hash_forward
                    AND a.hash_reverse
                    = f.hash_reverse
                ORDER BY
                    a.file_id,
                    a.row_number
                LIMIT 1
            ) AS representative_file_id,

            (
                SELECT a.row_number
                FROM assignments a
                WHERE
                    a.hash_forward
                    = f.hash_forward
                    AND a.hash_reverse
                    = f.hash_reverse
                ORDER BY
                    a.file_id,
                    a.row_number
                LIMIT 1
            ) AS representative_row_number,

            (
                SELECT s.file_path
                FROM assignments a
                JOIN source_file_integrity s
                  ON s.file_id = a.file_id
                WHERE
                    a.hash_forward
                    = f.hash_forward
                    AND a.hash_reverse
                    = f.hash_reverse
                ORDER BY
                    a.file_id,
                    a.row_number
                LIMIT 1
            ) AS representative_file_path,

            (
                SELECT s.class_label
                FROM assignments a
                JOIN source_file_integrity s
                  ON s.file_id = a.file_id
                WHERE
                    a.hash_forward
                    = f.hash_forward
                    AND a.hash_reverse
                    = f.hash_reverse
                ORDER BY
                    a.file_id,
                    a.row_number
                LIMIT 1
            ) AS representative_raw_label

        FROM fingerprint_stats f

        LEFT JOIN audit.rehashed_stats c
          ON c.hash_forward = f.hash_forward
         AND c.hash_reverse = f.hash_reverse

        WHERE c.hash_forward IS NULL

        ORDER BY
            f.family_label_min,
            f.grouped_split,
            f.hash_forward,
            f.hash_reverse

        LIMIT 50
        """,
    )

    canonical_only_rows = rows_as_dicts(
        connection,
        """
        SELECT
            c.hash_forward,
            c.hash_reverse,
            c.canonical_row_count,
            c.raw_occurrence_count,
            c.canonical_split_min,
            c.canonical_split_max,
            c.family_label_min,
            c.family_label_max
        FROM audit.rehashed_stats c

        LEFT JOIN fingerprint_stats f
          ON f.hash_forward = c.hash_forward
         AND f.hash_reverse = c.hash_reverse

        WHERE f.hash_forward IS NULL

        ORDER BY
            c.family_label_min,
            c.canonical_split_min,
            c.hash_forward,
            c.hash_reverse

        LIMIT 50
        """,
    )

    duplicate_rehash_rows = rows_as_dicts(
        connection,
        """
        SELECT
            hash_forward,
            hash_reverse,
            canonical_row_count,
            canonical_split_min,
            canonical_split_max,
            family_label_min,
            family_label_max,
            raw_occurrence_count
        FROM audit.rehashed_stats
        WHERE canonical_row_count > 1
        ORDER BY
            canonical_row_count DESC,
            hash_forward,
            hash_reverse
        LIMIT 50
        """,
    )

    transition_matrix = rows_as_dicts(
        connection,
        """
        SELECT
            c.canonical_split_min
                AS canonical_split,
            f.grouped_split
                AS protocol_split,
            f.family_label_min
                AS family_label,
            COUNT(*)
                AS fingerprint_count,
            SUM(f.raw_row_count)
                AS raw_row_count
        FROM fingerprint_stats f

        JOIN audit.rehashed_stats c
          ON c.hash_forward = f.hash_forward
         AND c.hash_reverse = f.hash_reverse

        GROUP BY
            c.canonical_split_min,
            f.grouped_split,
            f.family_label_min

        ORDER BY
            c.canonical_split_min,
            f.grouped_split,
            f.family_label_min
        """,
    )

    connection.execute(
        "DETACH DATABASE audit"
    )

protocol_sha256_after = sha256_file(
    PROTOCOL_DATABASE
)

if (
    protocol_sha256_before
    != protocol_sha256_after
):
    raise RuntimeError(
        "Protocol database SHA-256 changed during "
        "the read-only audit."
    )

elapsed = time.perf_counter() - started

all_protocol_groups_recoverable = bool(
    protocol_only_group_count == 0
    and canonical_only_group_count == 0
    and duplicate_rehash_group_count == 0
    and family_mismatch_count == 0
    and occurrence_mismatch_count == 0
)

summary = {
    "status": "completed",
    "generated_at_utc": utc_now(),
    "canonical_root": str(CANONICAL_ROOT),
    "protocol_database": str(
        PROTOCOL_DATABASE
    ),
    "audit_database": str(AUDIT_DATABASE),
    "protocol_database_quick_check": (
        quick_check
    ),
    "protocol_database_sha256_before": (
        protocol_sha256_before
    ),
    "protocol_database_sha256_after": (
        protocol_sha256_after
    ),
    "protocol_database_unchanged": True,
    "canonical_row_count": (
        canonical_row_count
    ),
    "canonical_rehashed_group_count": (
        rehashed_group_count
    ),
    "protocol_group_count": (
        protocol_group_count
    ),
    "common_group_count": (
        common_group_count
    ),
    "protocol_only_group_count": (
        protocol_only_group_count
    ),
    "canonical_only_group_count": (
        canonical_only_group_count
    ),
    "duplicate_rehash_group_count": (
        duplicate_rehash_group_count
    ),
    "duplicate_rehash_excess_rows": (
        duplicate_rehash_excess_rows
    ),
    "rehash_cross_split_group_count": (
        rehash_cross_split_group_count
    ),
    "rehash_cross_family_group_count": (
        rehash_cross_family_group_count
    ),
    "family_mismatch_count": (
        family_mismatch_count
    ),
    "raw_occurrence_mismatch_count": (
        occurrence_mismatch_count
    ),
    "canonical_split_vs_protocol_grouped_split_mismatch_count": (
        canonical_split_vs_protocol_mismatch
    ),
    "common_protocol_raw_row_count": (
        common_raw_rows
    ),
    "all_protocol_groups_recoverable_from_canonical_parquet": (
        all_protocol_groups_recoverable
    ),
    "protocol_only_rows": (
        protocol_only_rows
    ),
    "canonical_only_rows": (
        canonical_only_rows
    ),
    "duplicate_rehash_rows": (
        duplicate_rehash_rows
    ),
    "split_transition_matrix": (
        transition_matrix
    ),
    "elapsed_seconds": elapsed,
}

atomic_json(
    OUTPUT_JSON,
    summary,
)

print()
print("=" * 92)
print("REHASH ALIGNMENT SUMMARY")
print("=" * 92)
print(
    "Canonical rows                  : "
    f"{canonical_row_count:,}"
)
print(
    "Canonical VS2 rehashed groups   : "
    f"{rehashed_group_count:,}"
)
print(
    "Protocol 2.3 groups             : "
    f"{protocol_group_count:,}"
)
print(
    "Common groups                   : "
    f"{common_group_count:,}"
)
print(
    "Protocol-only groups            : "
    f"{protocol_only_group_count:,}"
)
print(
    "Canonical-only groups           : "
    f"{canonical_only_group_count:,}"
)
print(
    "Duplicate VS2 rehash groups     : "
    f"{duplicate_rehash_group_count:,}"
)
print(
    "Duplicate rehash excess rows    : "
    f"{duplicate_rehash_excess_rows:,}"
)
print(
    "Rehash cross-split groups       : "
    f"{rehash_cross_split_group_count:,}"
)
print(
    "Rehash cross-family groups      : "
    f"{rehash_cross_family_group_count:,}"
)
print(
    "Family-label mismatches         : "
    f"{family_mismatch_count:,}"
)
print(
    "Raw-occurrence mismatches       : "
    f"{occurrence_mismatch_count:,}"
)
print(
    "Canonical/protocol split changes: "
    f"{canonical_split_vs_protocol_mismatch:,}"
)
print(
    "Common represented raw rows     : "
    f"{common_raw_rows:,}"
)
print(
    "All protocol groups recoverable : "
    f"{all_protocol_groups_recoverable}"
)

print()
print("PROTOCOL-ONLY GROUPS")

if not protocol_only_rows:
    print("  none")

for row in protocol_only_rows:
    split_name = SPLIT_NAMES.get(
        int(row["grouped_split"]),
        str(row["grouped_split"]),
    )

    family_name = FAMILY_NAMES.get(
        int(row["family_label_min"]),
        str(row["family_label_min"]),
    )

    print(
        "  "
        f"hash=({row['hash_forward']},"
        f"{row['hash_reverse']}) | "
        f"raw_rows={row['raw_row_count']} | "
        f"split={split_name} | "
        f"family={family_name} | "
        f"file_id={row['representative_file_id']} | "
        f"row={row['representative_row_number']} | "
        f"raw_label={row['representative_raw_label']} | "
        f"path={row['representative_file_path']}"
    )

print()
print("CANONICAL-ONLY GROUPS")

if not canonical_only_rows:
    print("  none")

for row in canonical_only_rows:
    split_name = SPLIT_NAMES.get(
        int(row["canonical_split_min"]),
        str(row["canonical_split_min"]),
    )

    family_name = FAMILY_NAMES.get(
        int(row["family_label_min"]),
        str(row["family_label_min"]),
    )

    print(
        "  "
        f"hash=({row['hash_forward']},"
        f"{row['hash_reverse']}) | "
        f"rows={row['canonical_row_count']} | "
        f"raw_rows={row['raw_occurrence_count']} | "
        f"split={split_name} | "
        f"family={family_name}"
    )

print()
print("DUPLICATE VS2 REHASH GROUPS")

if not duplicate_rehash_rows:
    print("  none")

for row in duplicate_rehash_rows:
    print(
        "  "
        f"hash=({row['hash_forward']},"
        f"{row['hash_reverse']}) | "
        f"canonical_rows="
        f"{row['canonical_row_count']} | "
        f"raw_rows="
        f"{row['raw_occurrence_count']}"
    )

print()
print("SPLIT TRANSITION MATRIX")

for row in transition_matrix:
    canonical_split = SPLIT_NAMES[
        int(row["canonical_split"])
    ]

    protocol_split = SPLIT_NAMES[
        int(row["protocol_split"])
    ]

    family_name = FAMILY_NAMES[
        int(row["family_label"])
    ]

    print(
        "  "
        f"{family_name}: "
        f"{canonical_split} -> "
        f"{protocol_split} | "
        f"fingerprints="
        f"{row['fingerprint_count']:,} | "
        f"raw_rows="
        f"{row['raw_row_count']:,}"
    )

print()
print(f"Summary JSON : {OUTPUT_JSON}")
print(f"Audit DB     : {AUDIT_DATABASE}")
print(f"Elapsed      : {elapsed:.1f} seconds")
print()
print("VS2 REHASHED CANONICAL ALIGNMENT AUDIT COMPLETED")
