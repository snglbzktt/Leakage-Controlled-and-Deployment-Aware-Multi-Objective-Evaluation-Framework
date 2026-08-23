from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "exact_byte_fingerprint_v2_1"

EXPECTED_FILE_COUNT = 89
EXPECTED_TOTAL_ROWS = 7_062_606
EXPECTED_FEATURE_COUNT = 115
EXPECTED_CANONICAL_BYTE_LENGTH = EXPECTED_FEATURE_COUNT * 8

CHUNK_SIZE = 5_000
MINIMUM_FREE_DISK_GB = 12.0


PROJECT_ROOT = Path.cwd()

RAW_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "nbaiot_unpacked"
)

SOURCE_DATABASE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "nbaiot_duplicate_audit.sqlite"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DATABASE = (
    OUTPUT_DIRECTORY
    / "nbaiot_exact_byte_audit_v2.sqlite"
)

SUMMARY_FILE = (
    OUTPUT_DIRECTORY
    / "nbaiot_exact_byte_audit_summary_v2.json"
)

CONFLICT_FILE = (
    OUTPUT_DIRECTORY
    / "nbaiot_exact_byte_conflicts_v2.csv"
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def normalize_path(value: str) -> str:
    return (
        value
        .replace("\\", "/")
        .strip()
        .lstrip("./")
        .casefold()
    )


def format_duration(seconds: float) -> str:
    seconds_int = max(
        0,
        int(round(seconds)),
    )

    hours, remainder = divmod(
        seconds_int,
        3600,
    )

    minutes, seconds_value = divmod(
        remainder,
        60,
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds_value:02d}"
    )


def canonicalize_matrix(
    frame: pd.DataFrame,
) -> np.ndarray:
    if frame.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Beklenen "
            f"{EXPECTED_FEATURE_COUNT} sütun, "
            f"bulunan {frame.shape[1]}"
        )

    numeric_frame = frame.apply(
        pd.to_numeric,
        errors="raise",
    )

    matrix = numeric_frame.to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "NaN veya sonsuz değer bulundu."
        )

    matrix[matrix == 0.0] = 0.0

    matrix = np.asarray(
        matrix,
        dtype=np.dtype("<f8"),
        order="C",
    )

    if not matrix.flags.c_contiguous:
        matrix = np.ascontiguousarray(
            matrix,
            dtype=np.dtype("<f8"),
        )

    return matrix


def build_raw_file_map() -> dict[str, Path]:
    csv_files = sorted(
        path.resolve()
        for path in RAW_DATA_ROOT.rglob("*.csv")
        if path.is_file()
    )

    return {
        normalize_path(
            path.relative_to(
                RAW_DATA_ROOT
            ).as_posix()
        ): path
        for path in csv_files
    }


def initialize_output_database(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(
        "PRAGMA journal_mode = TRUNCATE"
    )

    connection.execute(
        "PRAGMA synchronous = NORMAL"
    )

    connection.execute(
        "PRAGMA temp_store = MEMORY"
    )

    connection.execute(
        "PRAGMA cache_size = -262144"
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS file_state (
            file_id INTEGER PRIMARY KEY,
            relative_path TEXT NOT NULL,
            expected_row_count INTEGER NOT NULL,
            processed_row_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            elapsed_seconds REAL,
            error TEXT
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS group_exact (
            hash_forward INTEGER NOT NULL,
            hash_reverse INTEGER NOT NULL,

            canonical_bytes BLOB NOT NULL,
            sha256_digest BLOB NOT NULL,

            occurrence_count INTEGER NOT NULL,

            byte_conflict_count INTEGER NOT NULL
                DEFAULT 0,

            sha256_conflict_count INTEGER NOT NULL
                DEFAULT 0,

            first_file_id INTEGER NOT NULL,
            first_row_number INTEGER NOT NULL,

            first_conflict_file_id INTEGER,
            first_conflict_row_number INTEGER,

            PRIMARY KEY (
                hash_forward,
                hash_reverse
            )
        ) WITHOUT ROWID
        """
    )

    connection.execute(
        """
        INSERT INTO run_metadata (
            key,
            value
        )
        VALUES (
            'protocol_version',
            ?
        )
        ON CONFLICT(key)
        DO NOTHING
        """,
        (PROTOCOL_VERSION,),
    )

    stored_protocol_row = connection.execute(
        """
        SELECT value
        FROM run_metadata
        WHERE key = 'protocol_version'
        """
    ).fetchone()

    stored_protocol = (
        str(stored_protocol_row[0])
        if stored_protocol_row
        else ""
    )

    if stored_protocol != PROTOCOL_VERSION:
        raise RuntimeError(
            "Mevcut çıktı veritabanının "
            "protokol sürümü uyuşmuyor. "
            f"Beklenen={PROTOCOL_VERSION}, "
            f"bulunan={stored_protocol}"
        )

    metadata_values = {
        "source_database": str(
            SOURCE_DATABASE
        ),
        "raw_data_root": str(
            RAW_DATA_ROOT
        ),
        "expected_total_rows": str(
            EXPECTED_TOTAL_ROWS
        ),
        "expected_feature_count": str(
            EXPECTED_FEATURE_COUNT
        ),
        "canonical_byte_length": str(
            EXPECTED_CANONICAL_BYTE_LENGTH
        ),
        "chunk_size": str(
            CHUNK_SIZE
        ),
    }

    for key, value in metadata_values.items():
        connection.execute(
            """
            INSERT INTO run_metadata (
                key,
                value
            )
            VALUES (?, ?)
            ON CONFLICT(key)
            DO UPDATE SET
                value = excluded.value
            """,
            (key, value),
        )

    connection.commit()


UPSERT_SQL = """
INSERT INTO group_exact (
    hash_forward,
    hash_reverse,
    canonical_bytes,
    sha256_digest,
    occurrence_count,
    byte_conflict_count,
    sha256_conflict_count,
    first_file_id,
    first_row_number,
    first_conflict_file_id,
    first_conflict_row_number
)
VALUES (
    ?, ?, ?, ?, 1, 0, 0, ?, ?, NULL, NULL
)
ON CONFLICT (
    hash_forward,
    hash_reverse
)
DO UPDATE SET
    occurrence_count =
        group_exact.occurrence_count + 1,

    byte_conflict_count =
        group_exact.byte_conflict_count
        +
        CASE
            WHEN
                group_exact.canonical_bytes
                <> excluded.canonical_bytes
            THEN 1
            ELSE 0
        END,

    sha256_conflict_count =
        group_exact.sha256_conflict_count
        +
        CASE
            WHEN
                group_exact.sha256_digest
                <> excluded.sha256_digest
            THEN 1
            ELSE 0
        END,

    first_conflict_file_id =
        CASE
            WHEN
                group_exact.canonical_bytes
                <> excluded.canonical_bytes
                AND
                group_exact.first_conflict_file_id
                IS NULL
            THEN excluded.first_file_id
            ELSE
                group_exact.first_conflict_file_id
        END,

    first_conflict_row_number =
        CASE
            WHEN
                group_exact.canonical_bytes
                <> excluded.canonical_bytes
                AND
                group_exact.first_conflict_row_number
                IS NULL
            THEN excluded.first_row_number
            ELSE
                group_exact.first_conflict_row_number
        END
"""


if not RAW_DATA_ROOT.exists():
    print(
        "HATA: Ham veri kökü bulunamadı:"
    )
    print(RAW_DATA_ROOT)
    sys.exit(1)

if not SOURCE_DATABASE.exists():
    print(
        "HATA: Kaynak fingerprint "
        "veritabanı bulunamadı:"
    )
    print(SOURCE_DATABASE)
    sys.exit(1)

disk_usage = shutil.disk_usage(
    PROJECT_ROOT
)

free_disk_gb = (
    disk_usage.free
    / (1024 ** 3)
)

existing_output_size_gb = (
    OUTPUT_DATABASE.stat().st_size
    / (1024 ** 3)
    if OUTPUT_DATABASE.exists()
    else 0.0
)

if (
    not OUTPUT_DATABASE.exists()
    and free_disk_gb
        < MINIMUM_FREE_DISK_GB
):
    print(
        "HATA: Yeni exact audit için "
        f"en az {MINIMUM_FREE_DISK_GB:.1f} GB "
        "boş alan gerekiyor."
    )
    print(
        f"Mevcut boş alan: "
        f"{free_disk_gb:.3f} GB"
    )
    sys.exit(1)

print("=" * 78)
print("FAZ 1B — TAM BYTE-DOĞRULAMALI EXACT FINGERPRINT DENETİMİ")
print("=" * 78)
print(f"Proje kökü        : {PROJECT_ROOT}")
print(f"Ham veri kökü     : {RAW_DATA_ROOT}")
print(f"Kaynak DB         : {SOURCE_DATABASE}")
print(f"Çıktı DB          : {OUTPUT_DATABASE}")
print(f"Çıktı DB mevcut   : {OUTPUT_DATABASE.exists()}")
print(f"Çıktı DB boyutu   : {existing_output_size_gb:.3f} GB")
print(f"Boş disk alanı    : {free_disk_gb:.3f} GB")
print(f"Chunk büyüklüğü   : {CHUNK_SIZE}")
print(f"Kanonik byte/vektör: {EXPECTED_CANONICAL_BYTE_LENGTH}")

raw_file_map = build_raw_file_map()

if len(raw_file_map) != EXPECTED_FILE_COUNT:
    raise RuntimeError(
        f"Beklenen {EXPECTED_FILE_COUNT} ham CSV, "
        f"bulunan {len(raw_file_map)}"
    )

source_connection = sqlite3.connect(
    str(SOURCE_DATABASE)
)

source_connection.row_factory = sqlite3.Row

source_connection.execute(
    "PRAGMA query_only = ON"
)

output_connection = sqlite3.connect(
    str(OUTPUT_DATABASE)
)

initialize_output_database(
    output_connection
)

file_rows = source_connection.execute(
    """
    SELECT
        file_id,
        relative_path,
        expected_row_count,
        inserted_row_count,
        status
    FROM files
    ORDER BY file_id
    """
).fetchall()

if len(file_rows) != EXPECTED_FILE_COUNT:
    raise RuntimeError(
        f"Kaynak DB'de beklenen "
        f"{EXPECTED_FILE_COUNT} dosya, "
        f"bulunan {len(file_rows)}"
    )

source_total_rows = sum(
    int(row["expected_row_count"])
    for row in file_rows
)

if source_total_rows != EXPECTED_TOTAL_ROWS:
    raise RuntimeError(
        f"Beklenen toplam {EXPECTED_TOTAL_ROWS}, "
        f"bulunan {source_total_rows}"
    )

completed_file_ids = {
    int(row[0])
    for row in output_connection.execute(
        """
        SELECT file_id
        FROM file_state
        WHERE status = 'done'
        """
    ).fetchall()
}

already_completed_rows = (
    output_connection.execute(
        """
        SELECT
            COALESCE(
                SUM(processed_row_count),
                0
            )
        FROM file_state
        WHERE status = 'done'
        """
    ).fetchone()[0]
)

print(
    f"Daha önce tamamlanan dosya: "
    f"{len(completed_file_ids)}"
)

print(
    f"Daha önce tamamlanan satır: "
    f"{int(already_completed_rows):,}"
)

overall_start_time = time.perf_counter()
session_processed_rows = 0

for file_position, file_row in enumerate(
    file_rows,
    start=1,
):
    file_id = int(
        file_row["file_id"]
    )

    stored_relative_path = str(
        file_row["relative_path"]
    )

    expected_row_count = int(
        file_row["expected_row_count"]
    )

    inserted_row_count = int(
        file_row["inserted_row_count"]
    )

    if file_id in completed_file_ids:
        print(
            f"[{file_position:02d}/{len(file_rows)}] "
            f"SKIP file_id={file_id} "
            f"({expected_row_count:,} satır)"
        )
        continue

    if inserted_row_count != expected_row_count:
        raise RuntimeError(
            f"file_id={file_id} için "
            "inserted_row_count uyuşmuyor."
        )

    normalized_relative_path = normalize_path(
        stored_relative_path
    )

    csv_path = raw_file_map.get(
        normalized_relative_path
    )

    if csv_path is None:
        raise RuntimeError(
            f"Kaynak CSV çözümlenemedi: "
            f"{stored_relative_path}"
        )

    range_row = source_connection.execute(
        """
        SELECT
            MIN(row_number),
            MAX(row_number),
            COUNT(*)
        FROM fingerprints
        WHERE file_id = ?
        """,
        (file_id,),
    ).fetchone()

    minimum_row_number = int(
        range_row[0]
    )

    maximum_row_number = int(
        range_row[1]
    )

    fingerprint_row_count = int(
        range_row[2]
    )

    if (
        minimum_row_number == 0
        and
        maximum_row_number
            == expected_row_count - 1
    ):
        row_number_base = 0
    elif (
        minimum_row_number == 1
        and
        maximum_row_number
            == expected_row_count
    ):
        row_number_base = 1
    else:
        raise RuntimeError(
            f"file_id={file_id} için "
            "beklenmeyen row_number aralığı: "
            f"{minimum_row_number}.."
            f"{maximum_row_number}"
        )

    if (
        fingerprint_row_count
        != expected_row_count
    ):
        raise RuntimeError(
            f"file_id={file_id} için "
            "fingerprint satır sayısı uyuşmuyor."
        )

    print()
    print(
        f"[{file_position:02d}/{len(file_rows)}] "
        f"BAŞLIYOR file_id={file_id}"
    )

    print(
        f"Dosya : {stored_relative_path}"
    )

    print(
        f"Satır : {expected_row_count:,}"
    )

    file_start_time = time.perf_counter()
    processed_in_file = 0

    output_connection.execute(
        """
        INSERT INTO file_state (
            file_id,
            relative_path,
            expected_row_count,
            processed_row_count,
            status,
            started_at,
            completed_at,
            elapsed_seconds,
            error
        )
        VALUES (?, ?, ?, 0, 'running', ?, NULL, NULL, NULL)
        ON CONFLICT(file_id)
        DO UPDATE SET
            relative_path =
                excluded.relative_path,
            expected_row_count =
                excluded.expected_row_count,
            processed_row_count = 0,
            status = 'running',
            started_at =
                excluded.started_at,
            completed_at = NULL,
            elapsed_seconds = NULL,
            error = NULL
        """,
        (
            file_id,
            stored_relative_path,
            expected_row_count,
            utc_now(),
        ),
    )

    output_connection.commit()

    try:
        output_connection.execute(
            "BEGIN IMMEDIATE"
        )

        fingerprint_cursor = (
            source_connection.execute(
                """
                SELECT
                    row_number,
                    hash_forward,
                    hash_reverse
                FROM fingerprints
                WHERE file_id = ?
                ORDER BY row_number
                """,
                (file_id,),
            )
        )

        csv_iterator = pd.read_csv(
            csv_path,
            header=0,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        next_progress_threshold = 100_000

        for frame in csv_iterator:
            matrix = canonicalize_matrix(
                frame
            )

            chunk_row_count = int(
                matrix.shape[0]
            )

            fingerprint_rows = (
                fingerprint_cursor.fetchmany(
                    chunk_row_count
                )
            )

            if (
                len(fingerprint_rows)
                != chunk_row_count
            ):
                raise RuntimeError(
                    "CSV chunk satır sayısı ile "
                    "fingerprint satır sayısı "
                    "uyuşmuyor."
                )

            payload = []

            for local_index in range(
                chunk_row_count
            ):
                fingerprint_row = (
                    fingerprint_rows[
                        local_index
                    ]
                )

                row_number = int(
                    fingerprint_row[
                        "row_number"
                    ]
                )

                expected_row_number = (
                    row_number_base
                    + processed_in_file
                    + local_index
                )

                if (
                    row_number
                    != expected_row_number
                ):
                    raise RuntimeError(
                        f"Satır hizalama hatası: "
                        f"beklenen="
                        f"{expected_row_number}, "
                        f"bulunan={row_number}"
                    )

                canonical_bytes = (
                    matrix[
                        local_index
                    ].tobytes(
                        order="C"
                    )
                )

                if (
                    len(canonical_bytes)
                    != EXPECTED_CANONICAL_BYTE_LENGTH
                ):
                    raise RuntimeError(
                        "Kanonik byte uzunluğu "
                        "beklenenden farklı."
                    )

                digest = hashlib.sha256(
                    canonical_bytes
                ).digest()

                payload.append(
                    (
                        int(
                            fingerprint_row[
                                "hash_forward"
                            ]
                        ),
                        int(
                            fingerprint_row[
                                "hash_reverse"
                            ]
                        ),
                        sqlite3.Binary(
                            canonical_bytes
                        ),
                        sqlite3.Binary(
                            digest
                        ),
                        file_id,
                        row_number,
                    )
                )

            output_connection.executemany(
                UPSERT_SQL,
                payload,
            )

            processed_in_file += (
                chunk_row_count
            )

            session_processed_rows += (
                chunk_row_count
            )

            if (
                processed_in_file
                >= next_progress_threshold
                or
                processed_in_file
                == expected_row_count
            ):
                elapsed = (
                    time.perf_counter()
                    - file_start_time
                )

                speed = (
                    processed_in_file
                    / elapsed
                    if elapsed > 0
                    else 0.0
                )

                remaining_rows = (
                    expected_row_count
                    - processed_in_file
                )

                eta_seconds = (
                    remaining_rows / speed
                    if speed > 0
                    else 0.0
                )

                print(
                    f"  {processed_in_file:,}"
                    f"/{expected_row_count:,} "
                    f"({speed:,.0f} satır/sn, "
                    f"dosya ETA "
                    f"{format_duration(eta_seconds)})"
                )

                while (
                    next_progress_threshold
                    <= processed_in_file
                ):
                    next_progress_threshold += (
                        100_000
                    )

        extra_fingerprint_row = (
            fingerprint_cursor.fetchone()
        )

        if extra_fingerprint_row is not None:
            raise RuntimeError(
                "CSV bittikten sonra "
                "fingerprints tablosunda "
                "fazladan satır kaldı."
            )

        if (
            processed_in_file
            != expected_row_count
        ):
            raise RuntimeError(
                f"Beklenen "
                f"{expected_row_count:,} satır, "
                f"işlenen "
                f"{processed_in_file:,} satır."
            )

        file_elapsed_seconds = (
            time.perf_counter()
            - file_start_time
        )

        output_connection.execute(
            """
            UPDATE file_state
            SET
                processed_row_count = ?,
                status = 'done',
                completed_at = ?,
                elapsed_seconds = ?,
                error = NULL
            WHERE file_id = ?
            """,
            (
                processed_in_file,
                utc_now(),
                file_elapsed_seconds,
                file_id,
            ),
        )

        output_connection.commit()

        output_size_gb = (
            OUTPUT_DATABASE.stat().st_size
            / (1024 ** 3)
        )

        current_free_gb = (
            shutil.disk_usage(
                PROJECT_ROOT
            ).free
            / (1024 ** 3)
        )

        print(
            f"TAMAMLANDI file_id={file_id} | "
            f"süre="
            f"{format_duration(file_elapsed_seconds)} | "
            f"çıktı DB="
            f"{output_size_gb:.3f} GB | "
            f"boş alan="
            f"{current_free_gb:.3f} GB"
        )

        if current_free_gb < 3.0:
            raise RuntimeError(
                "Boş disk alanı 3 GB altına "
                "düştü. Güvenlik nedeniyle "
                "denetim durduruldu."
            )

    except Exception as error:
        output_connection.rollback()

        output_connection.execute(
            """
            UPDATE file_state
            SET
                processed_row_count = 0,
                status = 'error',
                completed_at = ?,
                elapsed_seconds = ?,
                error = ?
            WHERE file_id = ?
            """,
            (
                utc_now(),
                (
                    time.perf_counter()
                    - file_start_time
                ),
                str(error),
                file_id,
            ),
        )

        output_connection.commit()

        print(
            f"HATA file_id={file_id}: {error}"
        )

        raise

print()
print("=" * 78)
print("SON DOĞRULAMALAR")
print("=" * 78)

completed_summary = (
    output_connection.execute(
        """
        SELECT
            COUNT(*),
            COALESCE(
                SUM(processed_row_count),
                0
            )
        FROM file_state
        WHERE status = 'done'
        """
    ).fetchone()
)

completed_file_count = int(
    completed_summary[0]
)

completed_row_count = int(
    completed_summary[1]
)

group_summary = (
    output_connection.execute(
        """
        SELECT
            COUNT(*) AS group_count,
            COALESCE(
                SUM(occurrence_count),
                0
            ) AS occurrence_count,
            COALESCE(
                SUM(
                    CASE
                        WHEN byte_conflict_count > 0
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS byte_conflict_groups,
            COALESCE(
                SUM(
                    CASE
                        WHEN sha256_conflict_count > 0
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS sha256_conflict_groups,
            MIN(
                LENGTH(canonical_bytes)
            ) AS minimum_byte_length,
            MAX(
                LENGTH(canonical_bytes)
            ) AS maximum_byte_length,
            MAX(
                occurrence_count
            ) AS maximum_occurrence_count
        FROM group_exact
        """
    ).fetchone()
)

exact_group_count = int(
    group_summary[0]
)

exact_occurrence_count = int(
    group_summary[1]
)

byte_conflict_group_count = int(
    group_summary[2]
)

sha256_conflict_group_count = int(
    group_summary[3]
)

minimum_byte_length = int(
    group_summary[4]
)

maximum_byte_length = int(
    group_summary[5]
)

maximum_occurrence_count = int(
    group_summary[6]
)

source_group_count = int(
    source_connection.execute(
        """
        SELECT COUNT(*)
        FROM vector_groups
        """
    ).fetchone()[0]
)

source_occurrence_count = int(
    source_connection.execute(
        """
        SELECT
            COALESCE(
                SUM(occurrence_count),
                0
            )
        FROM vector_groups
        """
    ).fetchone()[0]
)

output_connection.execute(
    "ATTACH DATABASE ? AS source_db",
    (str(SOURCE_DATABASE),),
)

source_to_exact_mismatch_count = int(
    output_connection.execute(
        """
        SELECT COUNT(*)
        FROM source_db.vector_groups AS source_group
        LEFT JOIN group_exact AS exact_group
            ON
                source_group.hash_forward
                    = exact_group.hash_forward
                AND
                source_group.hash_reverse
                    = exact_group.hash_reverse
        WHERE
            exact_group.hash_forward IS NULL
            OR
            source_group.occurrence_count
                <> exact_group.occurrence_count
        """
    ).fetchone()[0]
)

exact_to_source_missing_count = int(
    output_connection.execute(
        """
        SELECT COUNT(*)
        FROM group_exact AS exact_group
        LEFT JOIN source_db.vector_groups AS source_group
            ON
                source_group.hash_forward
                    = exact_group.hash_forward
                AND
                source_group.hash_reverse
                    = exact_group.hash_reverse
        WHERE
            source_group.hash_forward IS NULL
        """
    ).fetchone()[0]
)

conflict_rows = (
    output_connection.execute(
        """
        SELECT
            hash_forward,
            hash_reverse,
            occurrence_count,
            byte_conflict_count,
            sha256_conflict_count,
            first_file_id,
            first_row_number,
            first_conflict_file_id,
            first_conflict_row_number,
            hex(sha256_digest)
        FROM group_exact
        WHERE
            byte_conflict_count > 0
            OR
            sha256_conflict_count > 0
        ORDER BY
            byte_conflict_count DESC,
            sha256_conflict_count DESC
        """
    ).fetchall()
)

with CONFLICT_FILE.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as conflict_handle:
    writer = csv.writer(
        conflict_handle
    )

    writer.writerow(
        [
            "hash_forward",
            "hash_reverse",
            "occurrence_count",
            "byte_conflict_count",
            "sha256_conflict_count",
            "first_file_id",
            "first_row_number",
            "first_conflict_file_id",
            "first_conflict_row_number",
            "first_sha256",
        ]
    )

    writer.writerows(
        conflict_rows
    )

output_connection.execute(
    "DETACH DATABASE source_db"
)

overall_elapsed_seconds = (
    time.perf_counter()
    - overall_start_time
)

final_output_size_gb = (
    OUTPUT_DATABASE.stat().st_size
    / (1024 ** 3)
)

final_free_disk_gb = (
    shutil.disk_usage(
        PROJECT_ROOT
    ).free
    / (1024 ** 3)
)

summary = {
    "protocol_version": PROTOCOL_VERSION,
    "status": "completed",
    "started_from_existing_files": (
        len(completed_file_ids)
    ),
    "completed_file_count": (
        completed_file_count
    ),
    "completed_row_count": (
        completed_row_count
    ),
    "expected_file_count": (
        EXPECTED_FILE_COUNT
    ),
    "expected_total_rows": (
        EXPECTED_TOTAL_ROWS
    ),
    "source_group_count": (
        source_group_count
    ),
    "exact_group_count": (
        exact_group_count
    ),
    "source_occurrence_count": (
        source_occurrence_count
    ),
    "exact_occurrence_count": (
        exact_occurrence_count
    ),
    "byte_conflict_group_count": (
        byte_conflict_group_count
    ),
    "sha256_conflict_group_count": (
        sha256_conflict_group_count
    ),
    "source_to_exact_mismatch_count": (
        source_to_exact_mismatch_count
    ),
    "exact_to_source_missing_count": (
        exact_to_source_missing_count
    ),
    "minimum_canonical_byte_length": (
        minimum_byte_length
    ),
    "maximum_canonical_byte_length": (
        maximum_byte_length
    ),
    "expected_canonical_byte_length": (
        EXPECTED_CANONICAL_BYTE_LENGTH
    ),
    "maximum_occurrence_count": (
        maximum_occurrence_count
    ),
    "output_database": str(
        OUTPUT_DATABASE
    ),
    "output_database_size_gb": round(
        final_output_size_gb,
        6,
    ),
    "conflict_report": str(
        CONFLICT_FILE
    ),
    "elapsed_seconds": round(
        overall_elapsed_seconds,
        3,
    ),
    "elapsed_formatted": (
        format_duration(
            overall_elapsed_seconds
        )
    ),
    "final_free_disk_gb": round(
        final_free_disk_gb,
        3,
    ),
    "completed_at": utc_now(),
}

fatal_checks = {
    "completed_file_count": (
        completed_file_count
        == EXPECTED_FILE_COUNT
    ),
    "completed_row_count": (
        completed_row_count
        == EXPECTED_TOTAL_ROWS
    ),
    "exact_occurrence_count": (
        exact_occurrence_count
        == EXPECTED_TOTAL_ROWS
    ),
    "source_occurrence_count": (
        source_occurrence_count
        == EXPECTED_TOTAL_ROWS
    ),
    "group_count_match": (
        exact_group_count
        == source_group_count
    ),
    "byte_conflict_free": (
        byte_conflict_group_count == 0
    ),
    "sha256_conflict_free": (
        sha256_conflict_group_count == 0
    ),
    "source_to_exact_match": (
        source_to_exact_mismatch_count == 0
    ),
    "exact_to_source_match": (
        exact_to_source_missing_count == 0
    ),
    "canonical_byte_length_match": (
        minimum_byte_length
        == EXPECTED_CANONICAL_BYTE_LENGTH
        and
        maximum_byte_length
        == EXPECTED_CANONICAL_BYTE_LENGTH
    ),
}

summary["validation_checks"] = (
    fatal_checks
)

summary["all_checks_passed"] = all(
    fatal_checks.values()
)

SUMMARY_FILE.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("=" * 78)
print("FAZ 1B EXACT BYTE AUDIT ÖZETİ")
print("=" * 78)

for key, value in summary.items():
    if key != "validation_checks":
        print(f"{key}: {value}")

print()
print("VALIDATION CHECKS")

for key, value in fatal_checks.items():
    print(f"{key}: {value}")

print()
print(f"Özet raporu    : {SUMMARY_FILE}")
print(f"Conflict raporu: {CONFLICT_FILE}")
print(f"Exact audit DB : {OUTPUT_DATABASE}")

source_connection.close()
output_connection.close()

if not summary["all_checks_passed"]:
    print()
    print("FAZ 1B BAŞARISIZ")
    sys.exit(1)

print()
print("FAZ 1B BAŞARILI")
print(
    "Eski çift-hash grupları kanonik byte "
    "düzeyinde doğrulandı."
)
