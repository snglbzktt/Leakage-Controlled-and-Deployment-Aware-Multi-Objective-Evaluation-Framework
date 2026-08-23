"""
N-BaIoT ana deney veri setini Parquet biçiminde oluşturur.

Bu betik:

- Duplicate SQLite veritabanından her benzersiz özellik vektörü için
  deterministik bir temsilci satır seçer.
- Ham CSV dosyalarını parça parça okur.
- Her benzersiz özellik vektörünü yalnızca bir kez kaydeder.
- Seed 2026 ile oluşturulan train/validation/test atamasını yeniden hesaplar.
- Özellikleri float64 olarak korur.
- Split bazlı Parquet dosyaları ve ayrı metadata dosyası üretir.
- Oluşturulan kayıt sayılarını 07 numaralı split raporuyla doğrular.

Çıktı:
data/processed/nbaiot_primary_seed2026/
    train.parquet
    validation.parquet
    test.parquet
    metadata.parquet
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "nbaiot_unpacked"
)

DEFAULT_DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "nbaiot_duplicate_audit.sqlite"
)

DEFAULT_SCHEMA_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_reference_schema.csv"
)

DEFAULT_SPLIT_SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_primary_split_summary.json"
)

DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ==========================================================
# SPLIT CONSTANTS
# ==========================================================

UINT64_RANGE = 1 << 64
UINT64_MASK = UINT64_RANGE - 1

SPLIT_NAMES = {
    0: "train",
    1: "validation",
    2: "test",
}


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT ana deney veri setini "
            "deduplicate edilmiş Parquet dosyaları olarak üretir."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Çıkarılmış N-BaIoT CSV klasörü.",
    )

    parser.add_argument(
        "--database-file",
        type=Path,
        default=DEFAULT_DATABASE_FILE,
        help="Duplicate denetim SQLite veritabanı.",
    )

    parser.add_argument(
        "--schema-file",
        type=Path,
        default=DEFAULT_SCHEMA_FILE,
        help="Referans özellik şeması.",
    )

    parser.add_argument(
        "--split-summary-file",
        type=Path,
        default=DEFAULT_SPLIT_SUMMARY_FILE,
        help="07 numaralı betiğin split özet raporu.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="İşlenmiş veri ana klasörü.",
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Raporların kaydedileceği klasör.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Split tohumu.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Her seferinde okunacak CSV satırı.",
    )

    parser.add_argument(
        "--row-group-size",
        type=int,
        default=50_000,
        help="Parquet satır grubu büyüklüğü.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Mevcut çıktı klasörünü silip yeniden oluşturur.",
    )

    parser.add_argument(
        "--rebuild-representatives",
        action="store_true",
        help=(
            "SQLite içindeki temsilci satır tablosunu "
            "silerek yeniden oluşturur."
        ),
    )

    return parser.parse_args()


# ==========================================================
# JSON HELPERS
# ==========================================================

def json_default(value: object) -> object:
    """NumPy ve Path türlerini JSON uyumlu hâle getirir."""

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

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


# ==========================================================
# SPLIT ASSIGNMENT
# ==========================================================

def splitmix64(
    values: np.ndarray,
) -> np.ndarray:
    """Deterministik uint64 karıştırma fonksiyonu."""

    values = values.astype(
        np.uint64,
        copy=False,
    )

    with np.errstate(over="ignore"):
        values = (
            values
            + np.uint64(0x9E3779B97F4A7C15)
        )

        values = (
            values
            ^ (values >> np.uint64(30))
        ) * np.uint64(0xBF58476D1CE4E5B9)

        values = (
            values
            ^ (values >> np.uint64(27))
        ) * np.uint64(0x94D049BB133111EB)

        values = (
            values
            ^ (values >> np.uint64(31))
        )

    return values


def calculate_split_codes(
    hash_forward: np.ndarray,
    hash_reverse: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Parmak izlerini train, validation ve test kodlarına dönüştürür.

    Kodlar:
        0: train
        1: validation
        2: test
    """

    forward_unsigned = (
        hash_forward
        .astype(np.int64, copy=False)
        .view(np.uint64)
    )

    reverse_unsigned = (
        hash_reverse
        .astype(np.int64, copy=False)
        .view(np.uint64)
    )

    rotated_reverse = (
        reverse_unsigned << np.uint64(32)
    ) | (
        reverse_unsigned >> np.uint64(32)
    )

    seed_unsigned = np.uint64(
        seed & UINT64_MASK
    )

    mixed_input = (
        forward_unsigned
        ^ rotated_reverse
        ^ seed_unsigned
    )

    random_values = splitmix64(
        mixed_input
    )

    train_threshold = np.uint64(
        (UINT64_RANGE * 70) // 100
    )

    validation_threshold = np.uint64(
        (UINT64_RANGE * 85) // 100
    )

    split_codes = np.full(
        len(random_values),
        2,
        dtype=np.uint8,
    )

    split_codes[
        random_values < validation_threshold
    ] = 1

    split_codes[
        random_values < train_threshold
    ] = 0

    return split_codes


# ==========================================================
# INPUT VALIDATION
# ==========================================================

def load_reference_columns(
    schema_file: Path,
) -> list[str]:
    """Referans özellik sütunlarını yükler."""

    if not schema_file.exists():
        raise FileNotFoundError(
            f"Referans şema bulunamadı: {schema_file}"
        )

    schema = pd.read_csv(
        schema_file
    )

    if "column_name" not in schema.columns:
        raise ValueError(
            "Şema dosyasında 'column_name' sütunu yok."
        )

    columns = (
        schema["column_name"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if not columns:
        raise ValueError(
            "Referans özellik listesi boş."
        )

    if len(columns) != len(set(columns)):
        raise ValueError(
            "Referans şemada tekrarlanan sütun adı var."
        )

    return columns


def load_expected_split_counts(
    split_summary_file: Path,
    expected_seed: int,
) -> dict[str, int]:
    """07 numaralı split raporundaki beklenen sayıları yükler."""

    if not split_summary_file.exists():
        raise FileNotFoundError(
            "Split özet raporu bulunamadı: "
            f"{split_summary_file}"
        )

    with split_summary_file.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        summary = json.load(
            file_handle
        )

    actual_seed = int(
        summary["seed"]
    )

    if actual_seed != expected_seed:
        raise ValueError(
            "Split raporu seed değeri uyuşmuyor: "
            f"rapor={actual_seed}, istenen={expected_seed}"
        )

    counts: dict[str, int] = {}

    for record in summary[
        "overall_split_distribution"
    ]:
        counts[
            str(record["split"])
        ] = int(
            record["unique_vector_count"]
        )

    required_splits = {
        "train",
        "validation",
        "test",
    }

    if set(counts) != required_splits:
        raise ValueError(
            "Split raporunda train/validation/test "
            "kayıtlarının tamamı bulunmuyor."
        )

    return counts


# ==========================================================
# SQLITE HELPERS
# ==========================================================

def table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    """SQLite tablosunun varlığını kontrol eder."""

    result = connection.execute(
        """
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    ).fetchone()

    return bool(
        result is not None
        and int(result[0]) > 0
    )


def scalar_query(
    connection: sqlite3.Connection,
    query: str,
) -> int:
    """Tek sayısal değer döndüren sorguyu çalıştırır."""

    result = connection.execute(
        query
    ).fetchone()

    if result is None or result[0] is None:
        return 0

    return int(result[0])


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


def validate_database(
    connection: sqlite3.Connection,
) -> int:
    """Duplicate SQLite veritabanını doğrular."""

    required_tables = {
        "files",
        "fingerprints",
        "vector_groups",
    }

    missing_tables = [
        table_name
        for table_name in sorted(required_tables)
        if not table_exists(
            connection,
            table_name,
        )
    ]

    if missing_tables:
        raise RuntimeError(
            "SQLite veritabanında eksik tablolar var: "
            + ", ".join(missing_tables)
        )

    incomplete_files = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM files
        WHERE status != 'done'
        """,
    )

    if incomplete_files > 0:
        raise RuntimeError(
            "Tamamlanmamış dosya kayıtları var: "
            f"{incomplete_files}"
        )

    cross_class_groups = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM vector_groups
        WHERE class_count > 1
        """,
    )

    if cross_class_groups > 0:
        raise RuntimeError(
            "Sınıflar arası çelişkili vektör bulundu: "
            f"{cross_class_groups:,}"
        )

    return scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM vector_groups
        """,
    )


# ==========================================================
# REPRESENTATIVE ROWS
# ==========================================================

def drop_representative_table(
    connection: sqlite3.Connection,
) -> None:
    """Eski temsilci tablosunu siler."""

    connection.execute(
        """
        DROP TABLE IF EXISTS primary_representatives
        """
    )

    connection.commit()


def create_representative_table(
    connection: sqlite3.Connection,
    expected_unique_count: int,
    rebuild: bool,
) -> None:
    """
    Her benzersiz özellik vektörü için bir kaynak satır seçer.

    Seçim sırası:
        1. En küçük file_id
        2. O dosyadaki en küçük row_number
    """

    if rebuild:
        drop_representative_table(
            connection
        )

    if table_exists(
        connection,
        "primary_representatives",
    ):
        existing_count = scalar_query(
            connection,
            """
            SELECT COUNT(*)
            FROM primary_representatives
            """,
        )

        if existing_count == expected_unique_count:
            print(
                "Temsilci satır tablosu mevcut; "
                "yeniden kullanılacak."
            )
            return

        print(
            "Temsilci satır tablosunun sayısı uyuşmuyor; "
            "yeniden oluşturulacak."
        )

        drop_representative_table(
            connection
        )

    print(
        "Her benzersiz vektör için temsilci satır seçiliyor..."
    )

    try:
        connection.execute(
            "BEGIN IMMEDIATE"
        )

        connection.execute(
            """
            CREATE TABLE primary_representatives AS

            SELECT
                grouped.hash_forward AS hash_forward,
                grouped.hash_reverse AS hash_reverse,

                (
                    grouped.packed_location >> 32
                ) AS file_id,

                (
                    grouped.packed_location
                    & 4294967295
                ) AS row_number

            FROM (
                SELECT
                    hash_forward,
                    hash_reverse,

                    MIN(
                        (
                            CAST(file_id AS INTEGER) << 32
                        )
                        |
                        CAST(row_number AS INTEGER)
                    ) AS packed_location

                FROM fingerprints

                GROUP BY
                    hash_forward,
                    hash_reverse
            ) AS grouped
            """
        )

        connection.execute(
            """
            CREATE UNIQUE INDEX
            idx_primary_representatives_hash
            ON primary_representatives (
                hash_forward,
                hash_reverse
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX
            idx_primary_representatives_file_row
            ON primary_representatives (
                file_id,
                row_number
            )
            """
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    created_count = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM primary_representatives
        """,
    )

    if created_count != expected_unique_count:
        raise RuntimeError(
            "Temsilci satır sayısı uyuşmuyor: "
            f"beklenen={expected_unique_count:,}, "
            f"oluşturulan={created_count:,}"
        )


# ==========================================================
# PARQUET HELPERS
# ==========================================================

def create_feature_schema(
    feature_columns: list[str],
) -> pa.Schema:
    """Model veri dosyası için Arrow şeması oluşturur."""

    fields = [
        pa.field(
            column_name,
            pa.float64(),
        )
        for column_name in feature_columns
    ]

    fields.append(
        pa.field(
            "class_label",
            pa.string(),
        )
    )

    return pa.schema(
        fields
    )


def create_metadata_schema() -> pa.Schema:
    """Metadata Parquet dosyası için şema oluşturur."""

    return pa.schema(
        [
            pa.field(
                "hash_forward",
                pa.int64(),
            ),
            pa.field(
                "hash_reverse",
                pa.int64(),
            ),
            pa.field(
                "split",
                pa.string(),
            ),
            pa.field(
                "class_label",
                pa.string(),
            ),
            pa.field(
                "source_device",
                pa.string(),
            ),
            pa.field(
                "source_relative_path",
                pa.string(),
            ),
            pa.field(
                "source_row_number",
                pa.int64(),
            ),
            pa.field(
                "original_occurrence_count",
                pa.int64(),
            ),
            pa.field(
                "source_device_count",
                pa.int64(),
            ),
        ]
    )


def open_parquet_writers(
    output_directory: Path,
    feature_schema: pa.Schema,
    metadata_schema: pa.Schema,
) -> tuple[
    dict[str, pq.ParquetWriter],
    pq.ParquetWriter,
]:
    """Split ve metadata ParquetWriter nesnelerini oluşturur."""

    split_writers: dict[
        str,
        pq.ParquetWriter,
    ] = {}

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_writers[
            split_name
        ] = pq.ParquetWriter(
            output_directory
            / f"{split_name}.parquet",
            feature_schema,
            compression="zstd",
            use_dictionary=[
                "class_label",
            ],
            write_statistics=True,
        )

    metadata_writer = pq.ParquetWriter(
        output_directory
        / "metadata.parquet",
        metadata_schema,
        compression="zstd",
        use_dictionary=[
            "split",
            "class_label",
            "source_device",
            "source_relative_path",
        ],
        write_statistics=True,
    )

    return (
        split_writers,
        metadata_writer,
    )


def close_writers(
    split_writers: dict[
        str,
        pq.ParquetWriter,
    ],
    metadata_writer: pq.ParquetWriter | None,
) -> None:
    """Açılmış ParquetWriter nesnelerini kapatır."""

    for writer in split_writers.values():
        writer.close()

    if metadata_writer is not None:
        metadata_writer.close()


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    data_root = args.data_root.resolve()
    database_file = args.database_file.resolve()
    schema_file = args.schema_file.resolve()
    split_summary_file = (
        args.split_summary_file.resolve()
    )
    output_root = args.output_root.resolve()
    report_dir = args.report_dir.resolve()

    if args.chunk_size <= 0:
        raise ValueError(
            "Chunk size sıfırdan büyük olmalıdır."
        )

    if args.row_group_size <= 0:
        raise ValueError(
            "Row group size sıfırdan büyük olmalıdır."
        )

    if not data_root.exists():
        raise FileNotFoundError(
            f"Veri klasörü bulunamadı: {data_root}"
        )

    if not database_file.exists():
        raise FileNotFoundError(
            f"SQLite veritabanı bulunamadı: {database_file}"
        )

    feature_columns = load_reference_columns(
        schema_file
    )

    expected_split_counts = (
        load_expected_split_counts(
            split_summary_file=split_summary_file,
            expected_seed=args.seed,
        )
    )

    final_output_directory = (
        output_root
        / f"nbaiot_primary_seed{args.seed}"
    )

    temporary_output_directory = (
        output_root
        / f"nbaiot_primary_seed{args.seed}__tmp"
    )

    if final_output_directory.exists():
        if not args.overwrite:
            raise FileExistsError(
                "Çıktı klasörü zaten mevcut. "
                "--overwrite kullan: "
                f"{final_output_directory}"
            )

        shutil.rmtree(
            final_output_directory
        )

    if temporary_output_directory.exists():
        shutil.rmtree(
            temporary_output_directory
        )

    temporary_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    free_space_gb = (
        shutil.disk_usage(
            output_root
        ).free
        / (1024**3)
    )

    print("=" * 78)
    print("N-BaIoT Ana Deney Parquet Veri Seti")
    print("=" * 78)
    print(f"Veri klasörü       : {data_root}")
    print(f"SQLite veritabanı  : {database_file}")
    print(f"Geçici çıktı       : {temporary_output_directory}")
    print(f"Nihai çıktı        : {final_output_directory}")
    print(f"Seed               : {args.seed}")
    print(f"Özellik sayısı     : {len(feature_columns)}")
    print(f"CSV parça boyutu   : {args.chunk_size:,}")
    print(f"Boş disk alanı     : {free_space_gb:.2f} GB")
    print("=" * 78)

    if free_space_gb < 5:
        print(
            "UYARI: Boş disk alanı 5 GB altında."
        )

    connection = sqlite3.connect(
        database_file,
        timeout=180,
    )

    split_writers: dict[
        str,
        pq.ParquetWriter,
    ] = {}

    metadata_writer: pq.ParquetWriter | None = None

    try:
        configure_database(
            connection
        )

        expected_unique_count = (
            validate_database(
                connection
            )
        )

        create_representative_table(
            connection=connection,
            expected_unique_count=(
                expected_unique_count
            ),
            rebuild=(
                args.rebuild_representatives
            ),
        )

        feature_schema = create_feature_schema(
            feature_columns
        )

        metadata_schema = (
            create_metadata_schema()
        )

        (
            split_writers,
            metadata_writer,
        ) = open_parquet_writers(
            output_directory=(
                temporary_output_directory
            ),
            feature_schema=feature_schema,
            metadata_schema=metadata_schema,
        )

        file_rows = connection.execute(
            """
            SELECT
                file_id,
                relative_path,
                device,
                class_label,
                expected_row_count
            FROM files
            ORDER BY file_id
            """
        ).fetchall()

        split_counts: Counter[str] = Counter()

        class_split_counts: dict[
            str,
            Counter[str],
        ] = defaultdict(Counter)

        processed_representatives = 0

        progress = tqdm(
            file_rows,
            desc="Kaynak CSV dosyaları işleniyor",
            unit="dosya",
        )

        for (
            file_id,
            relative_path,
            device,
            class_label,
            expected_row_count,
        ) in progress:
            file_id = int(
                file_id
            )

            relative_path = str(
                relative_path
            )

            device = str(
                device
            )

            class_label = str(
                class_label
            )

            expected_row_count = int(
                expected_row_count
            )

            source_file = (
                data_root
                / Path(relative_path)
            )

            if not source_file.exists():
                raise FileNotFoundError(
                    f"Kaynak CSV bulunamadı: {source_file}"
                )

            representative_rows = (
                connection.execute(
                    """
                    SELECT
                        r.row_number,
                        r.hash_forward,
                        r.hash_reverse,
                        vg.occurrence_count,
                        vg.device_count

                    FROM primary_representatives AS r

                    JOIN vector_groups AS vg
                      ON vg.hash_forward = r.hash_forward
                     AND vg.hash_reverse = r.hash_reverse

                    WHERE r.file_id = ?

                    ORDER BY r.row_number
                    """,
                    (file_id,),
                ).fetchall()
            )

            if not representative_rows:
                continue

            row_numbers = np.asarray(
                [
                    int(row[0])
                    for row in representative_rows
                ],
                dtype=np.int64,
            )

            hash_forward = np.asarray(
                [
                    int(row[1])
                    for row in representative_rows
                ],
                dtype=np.int64,
            )

            hash_reverse = np.asarray(
                [
                    int(row[2])
                    for row in representative_rows
                ],
                dtype=np.int64,
            )

            occurrence_counts = np.asarray(
                [
                    int(row[3])
                    for row in representative_rows
                ],
                dtype=np.int64,
            )

            device_counts = np.asarray(
                [
                    int(row[4])
                    for row in representative_rows
                ],
                dtype=np.int64,
            )

            file_selected_count = 0
            rows_read = 0

            reader = pd.read_csv(
                source_file,
                chunksize=args.chunk_size,
                dtype=np.float64,
                low_memory=False,
                on_bad_lines="error",
            )

            for chunk in reader:
                chunk.columns = [
                    str(column).strip()
                    for column in chunk.columns
                ]

                if list(chunk.columns) != feature_columns:
                    raise ValueError(
                        "Sütun şeması uyuşmuyor: "
                        f"{relative_path}"
                    )

                chunk_length = int(
                    len(chunk)
                )

                chunk_first_row = (
                    rows_read + 1
                )

                chunk_last_row = (
                    rows_read + chunk_length
                )

                left_index = int(
                    np.searchsorted(
                        row_numbers,
                        chunk_first_row,
                        side="left",
                    )
                )

                right_index = int(
                    np.searchsorted(
                        row_numbers,
                        chunk_last_row,
                        side="right",
                    )
                )

                rows_read += chunk_length

                if left_index == right_index:
                    continue

                selected_global_rows = (
                    row_numbers[
                        left_index:right_index
                    ]
                )

                local_indices = (
                    selected_global_rows
                    - chunk_first_row
                )

                selected_features = chunk.iloc[
                    local_indices
                ].copy()

                selected_forward = (
                    hash_forward[
                        left_index:right_index
                    ]
                )

                selected_reverse = (
                    hash_reverse[
                        left_index:right_index
                    ]
                )

                selected_occurrences = (
                    occurrence_counts[
                        left_index:right_index
                    ]
                )

                selected_device_counts = (
                    device_counts[
                        left_index:right_index
                    ]
                )

                split_codes = calculate_split_codes(
                    hash_forward=selected_forward,
                    hash_reverse=selected_reverse,
                    seed=args.seed,
                )

                for split_code, split_name in (
                    (0, "train"),
                    (1, "validation"),
                    (2, "test"),
                ):
                    positions = np.flatnonzero(
                        split_codes == split_code
                    )

                    if positions.size == 0:
                        continue

                    split_frame = (
                        selected_features.iloc[
                            positions
                        ].copy()
                    )

                    split_frame[
                        "class_label"
                    ] = class_label

                    split_table = (
                        pa.Table.from_pandas(
                            split_frame[
                                feature_columns
                                + ["class_label"]
                            ],
                            schema=feature_schema,
                            preserve_index=False,
                        )
                    )

                    split_writers[
                        split_name
                    ].write_table(
                        split_table,
                        row_group_size=(
                            args.row_group_size
                        ),
                    )

                    selected_count = int(
                        positions.size
                    )

                    split_counts[
                        split_name
                    ] += selected_count

                    class_split_counts[
                        class_label
                    ][split_name] += (
                        selected_count
                    )

                split_names_array = np.asarray(
                    [
                        SPLIT_NAMES[
                            int(code)
                        ]
                        for code in split_codes
                    ],
                    dtype=object,
                )

                metadata_frame = pd.DataFrame(
                    {
                        "hash_forward": (
                            selected_forward
                        ),
                        "hash_reverse": (
                            selected_reverse
                        ),
                        "split": (
                            split_names_array
                        ),
                        "class_label": (
                            class_label
                        ),
                        "source_device": (
                            device
                        ),
                        "source_relative_path": (
                            relative_path
                        ),
                        "source_row_number": (
                            selected_global_rows
                        ),
                        "original_occurrence_count": (
                            selected_occurrences
                        ),
                        "source_device_count": (
                            selected_device_counts
                        ),
                    }
                )

                metadata_table = (
                    pa.Table.from_pandas(
                        metadata_frame,
                        schema=metadata_schema,
                        preserve_index=False,
                    )
                )

                metadata_writer.write_table(
                    metadata_table,
                    row_group_size=(
                        args.row_group_size
                    ),
                )

                selected_in_chunk = int(
                    len(selected_features)
                )

                file_selected_count += (
                    selected_in_chunk
                )

                processed_representatives += (
                    selected_in_chunk
                )

            if rows_read != expected_row_count:
                raise RuntimeError(
                    "Kaynak CSV satır sayısı uyuşmuyor: "
                    f"{relative_path}; "
                    f"beklenen={expected_row_count:,}, "
                    f"okunan={rows_read:,}"
                )

            if file_selected_count != len(
                representative_rows
            ):
                raise RuntimeError(
                    "Temsilci satır sayısı uyuşmuyor: "
                    f"{relative_path}; "
                    f"beklenen={len(representative_rows):,}, "
                    f"seçilen={file_selected_count:,}"
                )

        close_writers(
            split_writers=split_writers,
            metadata_writer=metadata_writer,
        )

        split_writers = {}
        metadata_writer = None

        if (
            processed_representatives
            != expected_unique_count
        ):
            raise RuntimeError(
                "Toplam temsilci sayısı uyuşmuyor: "
                f"beklenen={expected_unique_count:,}, "
                f"işlenen={processed_representatives:,}"
            )

        for split_name, expected_count in (
            expected_split_counts.items()
        ):
            actual_count = int(
                split_counts[split_name]
            )

            if actual_count != expected_count:
                raise RuntimeError(
                    f"{split_name} sayısı uyuşmuyor: "
                    f"beklenen={expected_count:,}, "
                    f"oluşturulan={actual_count:,}"
                )

        # --------------------------------------------------
        # Rename temporary directory
        # --------------------------------------------------

        temporary_output_directory.replace(
            final_output_directory
        )

        # --------------------------------------------------
        # Class distribution report
        # --------------------------------------------------

        class_records: list[
            dict[str, Any]
        ] = []

        for class_name in sorted(
            class_split_counts
        ):
            class_total = int(
                sum(
                    class_split_counts[
                        class_name
                    ].values()
                )
            )

            for split_name in (
                "train",
                "validation",
                "test",
            ):
                count = int(
                    class_split_counts[
                        class_name
                    ][split_name]
                )

                class_records.append(
                    {
                        "class_label": (
                            class_name
                        ),
                        "split": (
                            split_name
                        ),
                        "sample_count": (
                            count
                        ),
                        "percentage_within_class": (
                            count
                            / class_total
                            * 100
                            if class_total > 0
                            else 0.0
                        ),
                    }
                )

        class_distribution = pd.DataFrame(
            class_records
        )

        class_report_file = (
            report_dir
            / "nbaiot_primary_materialized_class_distribution.csv"
        )

        class_distribution.to_csv(
            class_report_file,
            index=False,
            encoding="utf-8",
        )

        # --------------------------------------------------
        # File sizes and summary
        # --------------------------------------------------

        output_files = {
            split_name: (
                final_output_directory
                / f"{split_name}.parquet"
            )
            for split_name in (
                "train",
                "validation",
                "test",
            )
        }

        metadata_file = (
            final_output_directory
            / "metadata.parquet"
        )

        split_summary_records: list[
            dict[str, Any]
        ] = []

        for split_name in (
            "train",
            "validation",
            "test",
        ):
            output_file = output_files[
                split_name
            ]

            split_summary_records.append(
                {
                    "split": split_name,
                    "sample_count": int(
                        split_counts[
                            split_name
                        ]
                    ),
                    "file_path": str(
                        output_file
                    ),
                    "file_size_bytes": int(
                        output_file.stat().st_size
                    ),
                    "file_size_gb": float(
                        output_file.stat().st_size
                        / (1024**3)
                    ),
                }
            )

        summary_file = (
            report_dir
            / "nbaiot_primary_materialization_summary.json"
        )

        summary: dict[str, Any] = {
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "seed": int(
                args.seed
            ),
            "feature_count": int(
                len(feature_columns)
            ),
            "feature_storage_dtype": (
                "float64"
            ),
            "global_unique_sample_count": int(
                processed_representatives
            ),
            "class_count": int(
                len(class_split_counts)
            ),
            "output_directory": str(
                final_output_directory
            ),
            "split_files": (
                split_summary_records
            ),
            "metadata_file": str(
                metadata_file
            ),
            "metadata_file_size_bytes": int(
                metadata_file.stat().st_size
            ),
            "class_distribution_report": str(
                class_report_file
            ),
            "representative_selection": (
                "Lowest file_id, then lowest 1-based row number "
                "for each composite 128-bit fingerprint."
            ),
            "leakage_control": (
                "Every composite feature-vector fingerprint is "
                "materialized exactly once and belongs to one split."
            ),
        }

        write_json_atomic(
            data=summary,
            output_file=summary_file,
        )

        connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        )

        print()
        print("=" * 78)
        print("Ana deney Parquet veri seti tamamlandı")
        print("=" * 78)

        for record in split_summary_records:
            print(
                f"{record['split']:10s}: "
                f"{record['sample_count']:,} örnek, "
                f"{record['file_size_gb']:.4f} GB"
            )

        print()
        print(
            "Toplam benzersiz örnek : "
            f"{processed_representatives:,}"
        )
        print(
            "Özellik sayısı         : "
            f"{len(feature_columns)}"
        )
        print(
            "Sınıf sayısı           : "
            f"{len(class_split_counts)}"
        )
        print()
        print(
            "Veri klasörü : "
            f"{final_output_directory}"
        )
        print(
            "Metadata      : "
            f"{metadata_file}"
        )
        print(
            "JSON özet     : "
            f"{summary_file}"
        )
        print("=" * 78)

    except Exception:
        close_writers(
            split_writers=split_writers,
            metadata_writer=metadata_writer,
        )

        if temporary_output_directory.exists():
            shutil.rmtree(
                temporary_output_directory,
                ignore_errors=True,
            )

        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()