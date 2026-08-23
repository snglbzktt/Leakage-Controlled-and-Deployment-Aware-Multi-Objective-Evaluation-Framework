from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path.cwd()

METADATA_PARQUET = (
    ROOT
    / "data"
    / "processed"
    / "nbaiot_float32_canonical_seed2026"
    / "metadata.parquet"
)

SPLIT_DATABASE = (
    ROOT
    / "data"
    / "splits"
    / "vs2_float32_leakage_ablation_scaled_repaired_seed2026_v3.sqlite"
)

OUTPUT_JSON = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "vs2_canonical_alignment_summary_v3.json"
)

SPLIT_CODES = {
    "train": 0,
    "validation": 1,
    "test": 2,
}

FAMILY_CODES = {
    "benign": 0,
    "gafgyt": 1,
    "mirai": 2,
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scalar(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def rows_as_dicts(
    connection: sqlite3.Connection,
    query: str,
) -> list[dict]:
    cursor = connection.execute(query)
    columns = [item[0] for item in cursor.description]
    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]


for required_path in (
    METADATA_PARQUET,
    SPLIT_DATABASE,
):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

OUTPUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 88)
print("VS2 CANONICAL / PROTOCOL 2.3 ALIGNMENT AUDIT")
print("=" * 88)
print(f"Canonical metadata : {METADATA_PARQUET}")
print(f"Protocol database  : {SPLIT_DATABASE}")

database_uri = (
    SPLIT_DATABASE.resolve().as_uri()
    + "?mode=ro"
)

started = time.perf_counter()

with sqlite3.connect(
    database_uri,
    uri=True,
    timeout=120.0,
) as connection:
    # The main database is protected by mode=ro.
    # Do not enable PRAGMA query_only because that also blocks TEMP-table writes.
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-200000")

    quick_check = str(
        connection.execute(
            "PRAGMA quick_check"
        ).fetchone()[0]
    )

    if quick_check.lower() != "ok":
        raise RuntimeError(
            f"Protocol database quick_check failed: {quick_check}"
        )

    connection.execute(
        """
        CREATE TEMP TABLE canonical_metadata (
            hash_forward INTEGER NOT NULL,
            hash_reverse INTEGER NOT NULL,
            canonical_split INTEGER NOT NULL,
            family_label INTEGER NOT NULL,
            raw_occurrence_count INTEGER NOT NULL,
            PRIMARY KEY (
                hash_forward,
                hash_reverse
            )
        ) WITHOUT ROWID
        """
    )

    parquet_file = pq.ParquetFile(
        METADATA_PARQUET
    )

    inserted_rows = 0

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=100_000,
            columns=[
                "hash_forward",
                "hash_reverse",
                "split",
                "class_label",
                "raw_occurrence_count",
            ],
        ),
        start=1,
    ):
        hash_forward = batch.column(0).to_pylist()
        hash_reverse = batch.column(1).to_pylist()
        split_values = batch.column(2).to_pylist()
        class_values = batch.column(3).to_pylist()
        occurrence_values = batch.column(4).to_pylist()

        records = []

        for (
            forward,
            reverse,
            split_name,
            class_name,
            occurrence_count,
        ) in zip(
            hash_forward,
            hash_reverse,
            split_values,
            class_values,
            occurrence_values,
        ):
            if split_name not in SPLIT_CODES:
                raise RuntimeError(
                    f"Unknown canonical split: {split_name!r}"
                )

            if class_name not in FAMILY_CODES:
                raise RuntimeError(
                    f"Unknown canonical class: {class_name!r}"
                )

            records.append(
                (
                    int(forward),
                    int(reverse),
                    SPLIT_CODES[split_name],
                    FAMILY_CODES[class_name],
                    int(occurrence_count),
                )
            )

        connection.executemany(
            """
            INSERT INTO canonical_metadata (
                hash_forward,
                hash_reverse,
                canonical_split,
                family_label,
                raw_occurrence_count
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )

        inserted_rows += len(records)

        print(
            f"[canonical load] "
            f"batch={batch_number} "
            f"rows={inserted_rows:,}",
            flush=True,
        )

    canonical_count = scalar(
        connection,
        "SELECT COUNT(*) FROM canonical_metadata",
    )

    database_count = scalar(
        connection,
        "SELECT COUNT(*) FROM fingerprint_stats",
    )

    common_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprint_stats f
        JOIN canonical_metadata c
          ON c.hash_forward = f.hash_forward
         AND c.hash_reverse = f.hash_reverse
        """,
    )

    database_only_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprint_stats f
        LEFT JOIN canonical_metadata c
          ON c.hash_forward = f.hash_forward
         AND c.hash_reverse = f.hash_reverse
        WHERE c.hash_forward IS NULL
        """,
    )

    canonical_only_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM canonical_metadata c
        LEFT JOIN fingerprint_stats f
          ON f.hash_forward = c.hash_forward
         AND f.hash_reverse = c.hash_reverse
        WHERE f.hash_forward IS NULL
        """,
    )

    split_mismatch_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprint_stats f
        JOIN canonical_metadata c
          ON c.hash_forward = f.hash_forward
         AND c.hash_reverse = f.hash_reverse
        WHERE c.canonical_split != f.grouped_split
        """,
    )

    family_mismatch_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprint_stats f
        JOIN canonical_metadata c
          ON c.hash_forward = f.hash_forward
         AND c.hash_reverse = f.hash_reverse
        WHERE c.family_label != f.family_label_min
           OR c.family_label != f.family_label_max
        """,
    )

    occurrence_mismatch_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprint_stats f
        JOIN canonical_metadata c
          ON c.hash_forward = f.hash_forward
         AND c.hash_reverse = f.hash_reverse
        WHERE c.raw_occurrence_count != f.raw_row_count
        """,
    )

    database_only_rows = rows_as_dicts(
        connection,
        """
        SELECT
            f.hash_forward,
            f.hash_reverse,
            f.raw_row_count,
            f.grouped_split,
            f.family_label_min,
            f.family_label_max
        FROM fingerprint_stats f
        LEFT JOIN canonical_metadata c
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
            c.raw_occurrence_count,
            c.canonical_split,
            c.family_label
        FROM canonical_metadata c
        LEFT JOIN fingerprint_stats f
          ON f.hash_forward = c.hash_forward
         AND f.hash_reverse = c.hash_reverse
        WHERE f.hash_forward IS NULL
        ORDER BY
            c.family_label,
            c.canonical_split,
            c.hash_forward,
            c.hash_reverse
        LIMIT 50
        """,
    )

    split_transition_rows = rows_as_dicts(
        connection,
        """
        SELECT
            c.canonical_split,
            f.grouped_split,
            c.family_label,
            COUNT(*) AS fingerprint_count,
            SUM(f.raw_row_count) AS raw_row_count
        FROM fingerprint_stats f
        JOIN canonical_metadata c
          ON c.hash_forward = f.hash_forward
         AND c.hash_reverse = f.hash_reverse
        GROUP BY
            c.canonical_split,
            f.grouped_split,
            c.family_label
        ORDER BY
            c.canonical_split,
            f.grouped_split,
            c.family_label
        """,
    )

    canonical_split_rows = rows_as_dicts(
        connection,
        """
        SELECT
            canonical_split,
            family_label,
            COUNT(*) AS fingerprint_count,
            SUM(raw_occurrence_count) AS raw_row_count
        FROM canonical_metadata
        GROUP BY
            canonical_split,
            family_label
        ORDER BY
            canonical_split,
            family_label
        """,
    )

    database_split_rows = rows_as_dicts(
        connection,
        """
        SELECT
            grouped_split,
            family_label_min AS family_label,
            COUNT(*) AS fingerprint_count,
            SUM(raw_row_count) AS raw_row_count
        FROM fingerprint_stats
        GROUP BY
            grouped_split,
            family_label_min
        ORDER BY
            grouped_split,
            family_label_min
        """,
    )

elapsed = time.perf_counter() - started

summary = {
    "status": "completed",
    "generated_at_utc": utc_now(),
    "canonical_metadata": str(METADATA_PARQUET),
    "protocol_database": str(SPLIT_DATABASE),
    "database_quick_check": quick_check,
    "canonical_fingerprint_count": canonical_count,
    "protocol_fingerprint_count": database_count,
    "common_fingerprint_count": common_count,
    "database_only_fingerprint_count": database_only_count,
    "canonical_only_fingerprint_count": canonical_only_count,
    "canonical_vs_protocol_split_mismatch_count": (
        split_mismatch_count
    ),
    "family_label_mismatch_count": family_mismatch_count,
    "raw_occurrence_count_mismatch_count": (
        occurrence_mismatch_count
    ),
    "database_only_rows": database_only_rows,
    "canonical_only_rows": canonical_only_rows,
    "canonical_split_counts": canonical_split_rows,
    "protocol_grouped_split_counts": database_split_rows,
    "split_transition_matrix": split_transition_rows,
    "elapsed_seconds": elapsed,
}

temporary_output = OUTPUT_JSON.with_suffix(
    OUTPUT_JSON.suffix + ".tmp"
)

temporary_output.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=True,
    ),
    encoding="utf-8",
)

temporary_output.replace(OUTPUT_JSON)

print()
print("=" * 88)
print("ALIGNMENT SUMMARY")
print("=" * 88)
print(f"Canonical fingerprints       : {canonical_count:,}")
print(f"Protocol 2.3 fingerprints    : {database_count:,}")
print(f"Common fingerprints          : {common_count:,}")
print(f"Database-only fingerprints   : {database_only_count:,}")
print(f"Canonical-only fingerprints  : {canonical_only_count:,}")
print(f"Split mismatches              : {split_mismatch_count:,}")
print(f"Family-label mismatches       : {family_mismatch_count:,}")
print(
    "Raw-occurrence mismatches    : "
    f"{occurrence_mismatch_count:,}"
)

print()
print("DATABASE-ONLY FINGERPRINTS")

for row in database_only_rows:
    split_name = SPLIT_NAMES.get(
        int(row["grouped_split"]),
        str(row["grouped_split"]),
    )
    family_name = FAMILY_NAMES.get(
        int(row["family_label_min"]),
        str(row["family_label_min"]),
    )

    print(
        f"  hash=({row['hash_forward']},"
        f"{row['hash_reverse']}) "
        f"raw_rows={row['raw_row_count']} "
        f"split={split_name} "
        f"family={family_name}"
    )

print()
print("CANONICAL-ONLY FINGERPRINTS")

for row in canonical_only_rows:
    split_name = SPLIT_NAMES.get(
        int(row["canonical_split"]),
        str(row["canonical_split"]),
    )
    family_name = FAMILY_NAMES.get(
        int(row["family_label"]),
        str(row["family_label"]),
    )

    print(
        f"  hash=({row['hash_forward']},"
        f"{row['hash_reverse']}) "
        f"raw_rows={row['raw_occurrence_count']} "
        f"split={split_name} "
        f"family={family_name}"
    )

print()
print("SPLIT TRANSITION MATRIX")

for row in split_transition_rows:
    old_split = SPLIT_NAMES[int(row["canonical_split"])]
    new_split = SPLIT_NAMES[int(row["grouped_split"])]
    family = FAMILY_NAMES[int(row["family_label"])]

    print(
        f"  {family}: "
        f"{old_split} -> {new_split} | "
        f"fingerprints={row['fingerprint_count']:,} | "
        f"raw_rows={row['raw_row_count']:,}"
    )

print()
print(f"Report: {OUTPUT_JSON}")
print(f"Elapsed seconds: {elapsed:.1f}")
print()
print("VS2 CANONICAL ALIGNMENT AUDIT COMPLETED")
