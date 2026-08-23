"""
N-BaIoT tekrar kayıt ve veri sızıntısı adayı denetimi.

Bu betik:

- Her özellik vektörü için iki adet 64-bit parmak izi üretir.
- Birleşik 128-bit parmak izlerini SQLite veritabanında saklar.
- Aynı dosya içindeki tekrarları belirler.
- Farklı dosyalardaki tekrarları belirler.
- Farklı cihazlardaki tekrarları belirler.
- Aynı özellik vektörünün farklı sınıflarda bulunma adaylarını belirler.
- Büyük veri setini parça parça okuyarak RAM kullanımını sınırlar.

Önemli:
Sınıflar arası parmak izi eşleşmeleri doğrudan etiket hatası kabul edilmez.
Bu adaylar daha sonra gerçek satır değerleri üzerinden doğrulanacaktır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from itertools import repeat
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "nbaiot_unpacked"
)

DEFAULT_MANIFEST_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_manifest.csv"
)

DEFAULT_DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "nbaiot_duplicate_audit.sqlite"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT tekrar kayıt ve veri "
            "sızıntısı adaylarını denetler."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Çıkarılmış N-BaIoT veri klasörü.",
    )

    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=DEFAULT_MANIFEST_FILE,
        help="N-BaIoT manifest CSV dosyası.",
    )

    parser.add_argument(
        "--database-file",
        type=Path,
        default=DEFAULT_DATABASE_FILE,
        help="SQLite denetim veritabanı.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Raporların kaydedileceği klasör.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Her seferinde okunacak satır sayısı.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Önceki denetim veritabanını silerek "
            "işlemi baştan başlatır."
        ),
    )

    return parser.parse_args()


# ==========================================================
# GENERAL HELPERS
# ==========================================================

def calculate_sha256(file_path: Path) -> str:
    """Bir dosyanın SHA-256 özetini hesaplar."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(
                8 * 1024 * 1024
            )

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def json_default(value: object) -> object:
    """NumPy değerlerini JSON uyumlu hâle getirir."""

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        numeric_value = float(value)

        if not np.isfinite(numeric_value):
            return None

        return numeric_value

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"{type(value).__name__} JSON ile uyumlu değil."
    )


def write_json_atomic(
    data: dict[str, Any],
    output_file: Path,
) -> None:
    """JSON raporunu geçici dosya üzerinden güvenli biçimde yazar."""

    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            data,
            file_handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=json_default,
        )

    temporary_file.replace(output_file)


def validate_manifest(
    manifest: pd.DataFrame,
) -> None:
    """Manifest dosyasını doğrular."""

    required_columns = {
        "device",
        "class_label",
        "relative_path",
        "row_count",
        "status",
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Manifest dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing_columns))
        )

    failed_files = manifest[
        manifest["status"] != "success"
    ]

    if not failed_files.empty:
        raise ValueError(
            "Manifest içinde başarısız dosya kayıtları var."
        )

    if manifest["relative_path"].duplicated().any():
        raise ValueError(
            "Manifestte tekrarlanan dosya yolları bulunuyor."
        )


# ==========================================================
# DATABASE HELPERS
# ==========================================================

def configure_database(
    connection: sqlite3.Connection,
) -> None:
    """SQLite performans ayarlarını uygular."""

    connection.execute(
        "PRAGMA journal_mode=WAL"
    )

    connection.execute(
        "PRAGMA synchronous=NORMAL"
    )

    connection.execute(
        "PRAGMA temp_store=FILE"
    )

    connection.execute(
        "PRAGMA cache_size=-200000"
    )

    connection.execute(
        "PRAGMA foreign_keys=OFF"
    )


def create_database_schema(
    connection: sqlite3.Connection,
) -> None:
    """Denetim tablolarını oluşturur."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            file_id INTEGER PRIMARY KEY,
            relative_path TEXT NOT NULL UNIQUE,
            device TEXT NOT NULL,
            class_label TEXT NOT NULL,
            expected_row_count INTEGER NOT NULL,
            inserted_row_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS fingerprints (
            hash_forward INTEGER NOT NULL,
            hash_reverse INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            row_number INTEGER NOT NULL
        );
        """
    )

    connection.commit()


def validate_database_manifest(
    connection: sqlite3.Connection,
    manifest_hash: str,
) -> None:
    """
    Mevcut veritabanının aynı manifest ile oluşturulduğunu doğrular.
    """

    existing = connection.execute(
        """
        SELECT value
        FROM metadata
        WHERE key = 'manifest_sha256'
        """
    ).fetchone()

    if existing is None:
        connection.execute(
            """
            INSERT INTO metadata (key, value)
            VALUES ('manifest_sha256', ?)
            """,
            (manifest_hash,),
        )

        connection.commit()
        return

    if str(existing[0]) != manifest_hash:
        raise RuntimeError(
            "Manifest dosyası değişmiş. "
            "Betiği --reset parametresiyle yeniden çalıştır."
        )


def initialize_file_table(
    connection: sqlite3.Connection,
    manifest: pd.DataFrame,
) -> None:
    """Manifestteki dosyaları SQLite tablosuna ekler."""

    records: list[
        tuple[int, str, str, str, int]
    ] = []

    for file_id, row in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):
        records.append(
            (
                int(file_id),
                str(row.relative_path),
                str(row.device),
                str(row.class_label),
                int(row.row_count),
            )
        )

    connection.executemany(
        """
        INSERT OR IGNORE INTO files (
            file_id,
            relative_path,
            device,
            class_label,
            expected_row_count
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        records,
    )

    connection.commit()


def scalar_query(
    connection: sqlite3.Connection,
    query: str,
) -> int:
    """Tek sayısal değer döndüren SQL sorgusunu çalıştırır."""

    result = connection.execute(
        query
    ).fetchone()

    if result is None or result[0] is None:
        return 0

    return int(result[0])


# ==========================================================
# FINGERPRINTING
# ==========================================================

def create_row_fingerprints(
    chunk: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Her satır için iki bağımsız 64-bit özet üretir.

    İkinci özet, sütunların ters sırasından hesaplanır.
    """

    forward_hash = pd.util.hash_pandas_object(
        chunk,
        index=False,
        categorize=False,
    ).to_numpy(
        dtype=np.uint64
    )

    reverse_hash = pd.util.hash_pandas_object(
        chunk.iloc[:, ::-1],
        index=False,
        categorize=False,
    ).to_numpy(
        dtype=np.uint64
    )

    return (
        forward_hash.view(np.int64),
        reverse_hash.view(np.int64),
    )


def process_file(
    connection: sqlite3.Connection,
    file_id: int,
    file_path: Path,
    expected_row_count: int,
    chunk_size: int,
) -> int:
    """
    Bir CSV dosyasının bütün satır parmak izlerini veritabanına ekler.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV dosyası bulunamadı: {file_path}"
        )

    # Önceki yarım işlem kayıtlarını kaldır.
    connection.execute(
        """
        DELETE FROM fingerprints
        WHERE file_id = ?
        """,
        (file_id,),
    )

    connection.execute(
        """
        UPDATE files
        SET
            status = 'processing',
            inserted_row_count = 0,
            error = NULL
        WHERE file_id = ?
        """,
        (file_id,),
    )

    connection.commit()

    inserted_row_count = 0

    reader = pd.read_csv(
        file_path,
        chunksize=chunk_size,
        dtype=np.float64,
        low_memory=False,
        on_bad_lines="error",
    )

    for chunk in reader:
        hash_forward, hash_reverse = (
            create_row_fingerprints(chunk)
        )

        current_chunk_size = int(
            len(chunk)
        )

        first_row_number = (
            inserted_row_count + 1
        )

        records = zip(
            map(int, hash_forward),
            map(int, hash_reverse),
            repeat(
                int(file_id),
                current_chunk_size,
            ),
            range(
                first_row_number,
                first_row_number
                + current_chunk_size,
            ),
        )

        connection.executemany(
            """
            INSERT INTO fingerprints (
                hash_forward,
                hash_reverse,
                file_id,
                row_number
            )
            VALUES (?, ?, ?, ?)
            """,
            records,
        )

        inserted_row_count += (
            current_chunk_size
        )

        connection.execute(
            """
            UPDATE files
            SET inserted_row_count = ?
            WHERE file_id = ?
            """,
            (
                inserted_row_count,
                file_id,
            ),
        )

        connection.commit()

    if inserted_row_count != expected_row_count:
        raise ValueError(
            "Satır sayısı uyuşmuyor: "
            f"beklenen={expected_row_count}, "
            f"okunan={inserted_row_count}"
        )

    connection.execute(
        """
        UPDATE files
        SET
            status = 'done',
            inserted_row_count = ?,
            error = NULL
        WHERE file_id = ?
        """,
        (
            inserted_row_count,
            file_id,
        ),
    )

    connection.commit()

    return inserted_row_count


def create_indexes(
    connection: sqlite3.Connection,
) -> None:
    """Analiz sorguları için SQLite indekslerini oluşturur."""

    print("Veritabanı indeksleri oluşturuluyor...")

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_fingerprints_hash
        ON fingerprints (
            hash_forward,
            hash_reverse
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_fingerprints_file
        ON fingerprints (file_id)
        """
    )

    connection.commit()


# ==========================================================
# REPORT GENERATION
# ==========================================================

def generate_reports(
    connection: sqlite3.Connection,
    output_dir: Path,
) -> dict[str, Any]:
    """Duplicate denetim raporlarını üretir."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_rows = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprints
        """,
    )

    unique_vectors = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                hash_forward,
                hash_reverse
            FROM fingerprints
            GROUP BY
                hash_forward,
                hash_reverse
        )
        """,
    )

    duplicate_group_count = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                hash_forward,
                hash_reverse
            FROM fingerprints
            GROUP BY
                hash_forward,
                hash_reverse
            HAVING COUNT(*) > 1
        )
        """,
    )

    cross_file_group_count = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                hash_forward,
                hash_reverse
            FROM fingerprints
            GROUP BY
                hash_forward,
                hash_reverse
            HAVING COUNT(DISTINCT file_id) > 1
        )
        """,
    )

    cross_device_group_count = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                p.hash_forward,
                p.hash_reverse
            FROM fingerprints AS p
            JOIN files AS f
                ON f.file_id = p.file_id
            GROUP BY
                p.hash_forward,
                p.hash_reverse
            HAVING COUNT(DISTINCT f.device) > 1
        )
        """,
    )

    cross_class_group_count = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                p.hash_forward,
                p.hash_reverse
            FROM fingerprints AS p
            JOIN files AS f
                ON f.file_id = p.file_id
            GROUP BY
                p.hash_forward,
                p.hash_reverse
            HAVING COUNT(DISTINCT f.class_label) > 1
        )
        """,
    )

    duplicate_rows_beyond_first = int(
        total_rows - unique_vectors
    )

    duplicate_percentage = (
        duplicate_rows_beyond_first
        / total_rows
        * 100
        if total_rows > 0
        else 0.0
    )

    # ------------------------------------------------------
    # File-level internal duplicate summary
    # ------------------------------------------------------

    duplicate_by_file = pd.read_sql_query(
        """
        WITH internal_groups AS (
            SELECT
                file_id,
                hash_forward,
                hash_reverse,
                COUNT(*) AS occurrence_count
            FROM fingerprints
            GROUP BY
                file_id,
                hash_forward,
                hash_reverse
            HAVING COUNT(*) > 1
        )
        SELECT
            f.relative_path,
            f.device,
            f.class_label,
            SUM(
                g.occurrence_count - 1
            ) AS duplicate_rows_beyond_first,
            COUNT(*) AS duplicate_group_count,
            MAX(
                g.occurrence_count
            ) AS maximum_multiplicity
        FROM internal_groups AS g
        JOIN files AS f
            ON f.file_id = g.file_id
        GROUP BY
            f.file_id,
            f.relative_path,
            f.device,
            f.class_label
        ORDER BY
            duplicate_rows_beyond_first DESC
        """,
        connection,
    )

    # ------------------------------------------------------
    # Largest duplicate groups
    # ------------------------------------------------------

    top_duplicate_groups = pd.read_sql_query(
        """
        SELECT
            p.hash_forward,
            p.hash_reverse,
            COUNT(*) AS occurrence_count,
            COUNT(
                DISTINCT p.file_id
            ) AS file_count,
            COUNT(
                DISTINCT f.device
            ) AS device_count,
            COUNT(
                DISTINCT f.class_label
            ) AS class_count,
            GROUP_CONCAT(
                DISTINCT f.class_label
            ) AS class_labels,
            GROUP_CONCAT(
                DISTINCT f.device
            ) AS devices
        FROM fingerprints AS p
        JOIN files AS f
            ON f.file_id = p.file_id
        GROUP BY
            p.hash_forward,
            p.hash_reverse
        HAVING COUNT(*) > 1
        ORDER BY
            occurrence_count DESC
        LIMIT 500
        """,
        connection,
    )

    # ------------------------------------------------------
    # Cross-class conflict candidates
    # ------------------------------------------------------

    cross_class_candidates = pd.read_sql_query(
        """
        SELECT
            p.hash_forward,
            p.hash_reverse,
            COUNT(*) AS occurrence_count,
            COUNT(
                DISTINCT p.file_id
            ) AS file_count,
            COUNT(
                DISTINCT f.device
            ) AS device_count,
            COUNT(
                DISTINCT f.class_label
            ) AS class_count,
            GROUP_CONCAT(
                DISTINCT f.class_label
            ) AS class_labels,
            GROUP_CONCAT(
                DISTINCT f.device
            ) AS devices,
            GROUP_CONCAT(
                DISTINCT f.relative_path
            ) AS relative_paths
        FROM fingerprints AS p
        JOIN files AS f
            ON f.file_id = p.file_id
        GROUP BY
            p.hash_forward,
            p.hash_reverse
        HAVING COUNT(
            DISTINCT f.class_label
        ) > 1
        ORDER BY
            occurrence_count DESC
        LIMIT 1000
        """,
        connection,
    )

    # ------------------------------------------------------
    # Save reports
    # ------------------------------------------------------

    duplicate_by_file_path = (
        output_dir
        / "nbaiot_duplicate_summary_by_file.csv"
    )

    top_groups_path = (
        output_dir
        / "nbaiot_top_duplicate_groups.csv"
    )

    cross_class_path = (
        output_dir
        / "nbaiot_cross_class_duplicate_candidates.csv"
    )

    summary_path = (
        output_dir
        / "nbaiot_duplicate_audit_summary.json"
    )

    duplicate_by_file.to_csv(
        duplicate_by_file_path,
        index=False,
        encoding="utf-8",
    )

    top_duplicate_groups.to_csv(
        top_groups_path,
        index=False,
        encoding="utf-8",
    )

    cross_class_candidates.to_csv(
        cross_class_path,
        index=False,
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "fingerprint_method": (
            "Composite 128-bit candidate fingerprint "
            "using normal and reversed column order"
        ),
        "total_rows": int(
            total_rows
        ),
        "unique_feature_vectors": int(
            unique_vectors
        ),
        "duplicate_rows_beyond_first": int(
            duplicate_rows_beyond_first
        ),
        "duplicate_percentage": float(
            duplicate_percentage
        ),
        "duplicate_group_count": int(
            duplicate_group_count
        ),
        "cross_file_duplicate_group_count": int(
            cross_file_group_count
        ),
        "cross_device_duplicate_group_count": int(
            cross_device_group_count
        ),
        "cross_class_duplicate_candidate_group_count": int(
            cross_class_group_count
        ),
        "files_with_internal_duplicates": int(
            len(duplicate_by_file)
        ),
        "report_files": {
            "duplicate_by_file": str(
                duplicate_by_file_path
            ),
            "top_duplicate_groups": str(
                top_groups_path
            ),
            "cross_class_candidates": str(
                cross_class_path
            ),
        },
    }

    write_json_atomic(
        data=summary,
        output_file=summary_path,
    )

    return summary


# ==========================================================
# RESET HELPERS
# ==========================================================

def remove_database_files(
    database_file: Path,
) -> None:
    """SQLite ana dosyasını ve yardımcı WAL dosyalarını siler."""

    related_files = [
        database_file,
        Path(str(database_file) + "-wal"),
        Path(str(database_file) + "-shm"),
    ]

    for file_path in related_files:
        if file_path.exists():
            file_path.unlink()


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    data_root = args.data_root.resolve()
    manifest_file = args.manifest_file.resolve()
    database_file = args.database_file.resolve()
    output_dir = args.output_dir.resolve()

    if args.chunk_size <= 0:
        raise ValueError(
            "Chunk size sıfırdan büyük olmalıdır."
        )

    if not data_root.exists():
        raise FileNotFoundError(
            f"Veri klasörü bulunamadı: {data_root}"
        )

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest dosyası bulunamadı: {manifest_file}"
        )

    manifest = pd.read_csv(
        manifest_file
    )

    validate_manifest(manifest)

    database_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.reset:
        remove_database_files(
            database_file
        )

    manifest_hash = calculate_sha256(
        manifest_file
    )

    connection = sqlite3.connect(
        database_file,
        timeout=60,
    )

    try:
        configure_database(
            connection
        )

        create_database_schema(
            connection
        )

        validate_database_manifest(
            connection=connection,
            manifest_hash=manifest_hash,
        )

        initialize_file_table(
            connection=connection,
            manifest=manifest,
        )

        file_rows = connection.execute(
            """
            SELECT
                file_id,
                relative_path,
                expected_row_count,
                status
            FROM files
            ORDER BY file_id
            """
        ).fetchall()

        print("=" * 78)
        print("N-BaIoT Tekrar ve Veri Sızıntısı Adayı Denetimi")
        print("=" * 78)
        print(f"Veri klasörü : {data_root}")
        print(f"Dosya sayısı : {len(file_rows)}")
        print(f"Parça boyutu : {args.chunk_size:,}")
        print(f"Veritabanı   : {database_file}")
        print("=" * 78)

        for (
            file_id,
            relative_path,
            expected_row_count,
            status,
        ) in tqdm(
            file_rows,
            desc="Dosyalar parmak izleniyor",
            unit="dosya",
        ):
            # Daha önce tamamlanan dosyayı tekrar okuma.
            if status == "done":
                continue

            file_path = (
                data_root
                / Path(str(relative_path))
            )

            try:
                process_file(
                    connection=connection,
                    file_id=int(file_id),
                    file_path=file_path,
                    expected_row_count=int(
                        expected_row_count
                    ),
                    chunk_size=int(
                        args.chunk_size
                    ),
                )

            except Exception as error:  # noqa: BLE001
                connection.execute(
                    """
                    UPDATE files
                    SET
                        status = 'error',
                        error = ?
                    WHERE file_id = ?
                    """,
                    (
                        f"{type(error).__name__}: {error}",
                        int(file_id),
                    ),
                )

                connection.commit()

        error_count = scalar_query(
            connection,
            """
            SELECT COUNT(*)
            FROM files
            WHERE status = 'error'
            """,
        )

        incomplete_count = scalar_query(
            connection,
            """
            SELECT COUNT(*)
            FROM files
            WHERE status != 'done'
            """,
        )

        if error_count > 0 or incomplete_count > 0:
            errors = pd.read_sql_query(
                """
                SELECT
                    relative_path,
                    status,
                    error
                FROM files
                WHERE status != 'done'
                """,
                connection,
            )

            error_report = (
                output_dir
                / "nbaiot_duplicate_audit_errors.csv"
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            errors.to_csv(
                error_report,
                index=False,
                encoding="utf-8",
            )

            raise RuntimeError(
                "Bazı dosyalar tamamlanamadı. "
                f"Hatalı={error_count}, "
                f"tamamlanmamış={incomplete_count}. "
                f"Rapor={error_report}"
            )

        create_indexes(
            connection
        )

        print(
            "Tekrar grupları analiz ediliyor..."
        )

        summary = generate_reports(
            connection=connection,
            output_dir=output_dir,
        )

        connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        )

        print()
        print("=" * 78)
        print("Duplicate denetimi tamamlandı")
        print("=" * 78)
        print(
            "Toplam kayıt              : "
            f"{summary['total_rows']:,}"
        )
        print(
            "Benzersiz özellik vektörü : "
            f"{summary['unique_feature_vectors']:,}"
        )
        print(
            "Tekrar eden fazla kayıt   : "
            f"{summary['duplicate_rows_beyond_first']:,}"
        )
        print(
            "Tekrar oranı              : "
            f"{summary['duplicate_percentage']:.4f}%"
        )
        print(
            "Tekrar grubu              : "
            f"{summary['duplicate_group_count']:,}"
        )
        print(
            "Dosyalar arası grup       : "
            f"{summary['cross_file_duplicate_group_count']:,}"
        )
        print(
            "Cihazlar arası grup       : "
            f"{summary['cross_device_duplicate_group_count']:,}"
        )
        print(
            "Sınıflar arası aday grup  : "
            f"{summary['cross_class_duplicate_candidate_group_count']:,}"
        )
        print("=" * 78)

    finally:
        connection.close()


if __name__ == "__main__":
    main()