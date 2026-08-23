"""
N-BaIoT float32 kanonik ve sızıntısız veri setini oluşturur.

İşlem adımları:
1. Float32 kanonik SQLite gruplarını doğrular.
2. Birden fazla sınıfa ait iki çelişkili grubu tamamen dışlar.
3. Kalan her float32 vektörü için deterministik train/validation/test
   ataması oluşturur.
4. Temsilci satırları eski float64 Parquet dosyalarından okur.
5. Özellikleri float32 biçiminde kaydeder.
6. Split bazlı Parquet ve metadata dosyaları üretir.
7. Üretilen örnek sayılarını SQLite atamalarıyla karşılaştırır.

Yeni splitler eski float64 splitlerinden bağımsızdır. Atama doğrudan
float32 kanonik parmak izleri üzerinden yapılır.
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

DEFAULT_SOURCE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_primary_seed2026"
)

DEFAULT_DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "nbaiot_float32_canonical.sqlite"
)

DEFAULT_SCHEMA_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_reference_schema.csv"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)


# ==========================================================
# CONSTANTS
# ==========================================================

SPLIT_NAMES = (
    "train",
    "validation",
    "test",
)

SPLIT_TO_CODE = {
    split_name: split_code
    for split_code, split_name in enumerate(
        SPLIT_NAMES
    )
}

CODE_TO_SPLIT = {
    split_code: split_name
    for split_name, split_code
    in SPLIT_TO_CODE.items()
}

UINT64_RANGE = 1 << 64
UINT64_MASK = UINT64_RANGE - 1

REPRESENTATIVE_SHIFT = 40
REPRESENTATIVE_ROW_MASK = (
    (1 << REPRESENTATIVE_SHIFT) - 1
)


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT float32 kanonik ve "
            "sızıntısız Parquet veri setini oluşturur."
        )
    )

    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=DEFAULT_SOURCE_DATA_DIR,
        help="Eski float64 Parquet veri klasörü.",
    )

    parser.add_argument(
        "--database-file",
        type=Path,
        default=DEFAULT_DATABASE_FILE,
        help="Float32 kanonik SQLite veritabanı.",
    )

    parser.add_argument(
        "--schema-file",
        type=Path,
        default=DEFAULT_SCHEMA_FILE,
        help="Referans özellik şeması.",
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
        help="Rapor klasörü.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Yeni float32 split tohumu.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Parquet ve SQLite parça büyüklüğü.",
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
        help="Mevcut çıktı klasörünü silerek yeniden oluşturur.",
    )

    parser.add_argument(
        "--rebuild-assignments",
        action="store_true",
        help=(
            "SQLite içindeki kanonik split atamalarını "
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
# INPUT HELPERS
# ==========================================================

def load_feature_columns(
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
            "Özellik sütun listesi boş."
        )

    if len(columns) != len(set(columns)):
        raise ValueError(
            "Şemada tekrarlanan özellik sütunu var."
        )

    return columns


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
    Float32 parmak izlerinden deterministik split kodu üretir.

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
# FLOAT32 FINGERPRINTS
# ==========================================================

def create_float32_features_and_hashes(
    feature_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Float32 kanonik özellik tablosu ve parmak izlerini üretir."""

    values = feature_frame.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Float32 dönüşümünden sonra sonlu olmayan değer oluştu."
        )

    # -0.0 ve +0.0 tek kanonik gösterime dönüştürülür.
    values[
        values == np.float32(0.0)
    ] = np.float32(0.0)

    float32_frame = pd.DataFrame(
        values,
        columns=feature_frame.columns,
    )

    forward = pd.util.hash_pandas_object(
        float32_frame,
        index=False,
        categorize=False,
    ).to_numpy(
        dtype=np.uint64
    )

    reverse = pd.util.hash_pandas_object(
        float32_frame.iloc[:, ::-1],
        index=False,
        categorize=False,
    ).to_numpy(
        dtype=np.uint64
    )

    return (
        float32_frame,
        forward.view(np.int64),
        reverse.view(np.int64),
    )


# ==========================================================
# SQLITE HELPERS
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
    parameters: tuple[object, ...] = (),
) -> int:
    """Tek sayısal SQL sonucu döndürür."""

    result = connection.execute(
        query,
        parameters,
    ).fetchone()

    if result is None or result[0] is None:
        return 0

    return int(result[0])


def read_run_metadata(
    connection: sqlite3.Connection,
    key: str,
) -> str | None:
    """run_metadata tablosundan bir değer okur."""

    result = connection.execute(
        """
        SELECT value
        FROM run_metadata
        WHERE key = ?
        """,
        (key,),
    ).fetchone()

    if result is None:
        return None

    return str(result[0])


def load_classes(
    connection: sqlite3.Connection,
) -> list[str]:
    """SQLite içinde kayıtlı sınıf sırasını yükler."""

    value = read_run_metadata(
        connection,
        "classes_json",
    )

    if value is None:
        raise RuntimeError(
            "SQLite run_metadata içinde classes_json bulunamadı."
        )

    classes = json.loads(
        value
    )

    if not isinstance(classes, list) or not classes:
        raise RuntimeError(
            "SQLite sınıf listesi geçersiz."
        )

    return [
        str(class_label)
        for class_label in classes
    ]


def validate_canonical_database(
    connection: sqlite3.Connection,
) -> tuple[int, int, int]:
    """
    Kanonik SQLite veritabanını doğrular.

    Dönüş:
        toplam float32 grup,
        çelişkili grup,
        kullanılabilir grup
    """

    required_tables = {
        "run_metadata",
        "float32_groups",
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
            "SQLite içinde gerekli tablolar eksik: "
            + ", ".join(missing_tables)
        )

    total_group_count = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM float32_groups
        """,
    )

    conflict_group_count = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM float32_groups
        WHERE (
            class_mask
            & (class_mask - 1)
        ) != 0
        """,
    )

    usable_group_count = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM float32_groups
        WHERE (
            class_mask
            & (class_mask - 1)
        ) = 0
        """,
    )

    if (
        total_group_count
        != conflict_group_count
        + usable_group_count
    ):
        raise RuntimeError(
            "Float32 grup sayıları kendi içinde uyuşmuyor."
        )

    return (
        total_group_count,
        conflict_group_count,
        usable_group_count,
    )


# ==========================================================
# CANONICAL ASSIGNMENTS
# ==========================================================

def build_canonical_assignments(
    connection: sqlite3.Connection,
    seed: int,
    expected_usable_count: int,
    batch_size: int,
    rebuild: bool,
) -> None:
    """Çelişkisiz gruplar için deterministik split ataması oluşturur."""

    assignment_table_exists = table_exists(
        connection,
        "canonical_assignments",
    )

    stored_seed = read_run_metadata(
        connection,
        "canonical_split_seed",
    )

    if (
        assignment_table_exists
        and not rebuild
        and stored_seed == str(seed)
    ):
        existing_count = scalar_query(
            connection,
            """
            SELECT COUNT(*)
            FROM canonical_assignments
            """,
        )

        if existing_count == expected_usable_count:
            print(
                "Kanonik split atamaları mevcut; "
                "yeniden kullanılacak."
            )
            return

    if assignment_table_exists and not rebuild:
        raise RuntimeError(
            "Mevcut kanonik atama tablosu seed veya kayıt "
            "sayısıyla uyuşmuyor. "
            "--rebuild-assignments kullan."
        )

    print(
        "Float32 kanonik split atamaları oluşturuluyor..."
    )

    connection.execute(
        """
        DROP TABLE IF EXISTS canonical_assignments_tmp
        """
    )

    connection.execute(
        """
        CREATE TABLE canonical_assignments_tmp (
            hash_forward INTEGER NOT NULL,
            hash_reverse INTEGER NOT NULL,
            class_code INTEGER NOT NULL,
            new_split_code INTEGER NOT NULL,
            representative_location INTEGER NOT NULL,
            float64_unique_occurrence_count INTEGER NOT NULL,
            raw_occurrence_count INTEGER NOT NULL,
            old_split_mask INTEGER NOT NULL,

            PRIMARY KEY (
                hash_forward,
                hash_reverse
            )
        ) WITHOUT ROWID
        """
    )

    connection.commit()

    query = """
        SELECT
            hash_forward,
            hash_reverse,
            class_code,
            representative_location,
            float64_unique_occurrence_count,
            raw_occurrence_count,
            old_split_mask

        FROM float32_groups

        WHERE (
            class_mask
            & (class_mask - 1)
        ) = 0

        ORDER BY
            hash_forward,
            hash_reverse
    """

    cursor = connection.execute(
        query
    )

    processed_count = 0

    progress = tqdm(
        total=expected_usable_count,
        desc="Kanonik split atanıyor",
        unit="grup",
    )

    while True:
        rows = cursor.fetchmany(
            batch_size
        )

        if not rows:
            break

        hash_forward = np.asarray(
            [
                int(row[0])
                for row in rows
            ],
            dtype=np.int64,
        )

        hash_reverse = np.asarray(
            [
                int(row[1])
                for row in rows
            ],
            dtype=np.int64,
        )

        split_codes = calculate_split_codes(
            hash_forward=hash_forward,
            hash_reverse=hash_reverse,
            seed=seed,
        )

        records = [
            (
                int(row[0]),
                int(row[1]),
                int(row[2]),
                int(split_code),
                int(row[3]),
                int(row[4]),
                int(row[5]),
                int(row[6]),
            )
            for row, split_code in zip(
                rows,
                split_codes,
                strict=True,
            )
        ]

        connection.executemany(
            """
            INSERT INTO canonical_assignments_tmp (
                hash_forward,
                hash_reverse,
                class_code,
                new_split_code,
                representative_location,
                float64_unique_occurrence_count,
                raw_occurrence_count,
                old_split_mask
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

        connection.commit()

        processed_count += len(rows)

        progress.update(
            len(rows)
        )

    progress.close()

    if processed_count != expected_usable_count:
        raise RuntimeError(
            "Atanan grup sayısı uyuşmuyor: "
            f"beklenen={expected_usable_count:,}, "
            f"işlenen={processed_count:,}"
        )

    temporary_count = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM canonical_assignments_tmp
        """,
    )

    if temporary_count != expected_usable_count:
        raise RuntimeError(
            "Geçici atama tablosu sayısı uyuşmuyor."
        )

    connection.execute(
        """
        DROP TABLE IF EXISTS canonical_assignments
        """
    )

    connection.execute(
        """
        ALTER TABLE canonical_assignments_tmp
        RENAME TO canonical_assignments
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_canonical_assignments_location
        ON canonical_assignments (
            representative_location
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_canonical_assignments_split_class
        ON canonical_assignments (
            new_split_code,
            class_code
        )
        """
    )

    connection.execute(
        """
        INSERT OR REPLACE INTO run_metadata (
            key,
            value
        )
        VALUES (
            'canonical_split_seed',
            ?
        )
        """,
        (str(seed),),
    )

    connection.execute(
        """
        INSERT OR REPLACE INTO run_metadata (
            key,
            value
        )
        VALUES (
            'canonical_split_method',
            'SplitMix64 from float32 composite fingerprint'
        )
        """
    )

    connection.commit()


# ==========================================================
# PARQUET SCHEMAS
# ==========================================================

def create_feature_schema(
    feature_columns: list[str],
) -> pa.Schema:
    """Yeni model veri dosyasının Arrow şemasını oluşturur."""

    fields = [
        pa.field(
            column_name,
            pa.float32(),
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
    """Yeni metadata dosyasının Arrow şemasını oluşturur."""

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
                "representative_source_split",
                pa.string(),
            ),
            pa.field(
                "representative_source_row_number",
                pa.int64(),
            ),
            pa.field(
                "float64_unique_occurrence_count",
                pa.int64(),
            ),
            pa.field(
                "raw_occurrence_count",
                pa.int64(),
            ),
            pa.field(
                "old_split_mask",
                pa.int16(),
            ),
            pa.field(
                "old_split_count",
                pa.int8(),
            ),
        ]
    )


def open_writers(
    output_directory: Path,
    feature_schema: pa.Schema,
    metadata_schema: pa.Schema,
) -> tuple[
    dict[str, pq.ParquetWriter],
    pq.ParquetWriter,
]:
    """ParquetWriter nesnelerini açar."""

    split_writers: dict[
        str,
        pq.ParquetWriter,
    ] = {}

    for split_name in SPLIT_NAMES:
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
            "representative_source_split",
        ],
        write_statistics=True,
    )

    return (
        split_writers,
        metadata_writer,
    )


def close_writers(
    split_writers: dict[str, pq.ParquetWriter],
    metadata_writer: pq.ParquetWriter | None,
) -> None:
    """Açık ParquetWriter nesnelerini kapatır."""

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

    source_data_dir = (
        args.source_data_dir.resolve()
    )

    database_file = (
        args.database_file.resolve()
    )

    schema_file = (
        args.schema_file.resolve()
    )

    output_root = (
        args.output_root.resolve()
    )

    report_dir = (
        args.report_dir.resolve()
    )

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size sıfırdan büyük olmalıdır."
        )

    if args.row_group_size <= 0:
        raise ValueError(
            "Row group size sıfırdan büyük olmalıdır."
        )

    if not source_data_dir.exists():
        raise FileNotFoundError(
            "Kaynak Parquet klasörü bulunamadı: "
            f"{source_data_dir}"
        )

    if not database_file.exists():
        raise FileNotFoundError(
            "Float32 kanonik SQLite bulunamadı: "
            f"{database_file}"
        )

    feature_columns = load_feature_columns(
        schema_file
    )

    final_output_directory = (
        output_root
        / f"nbaiot_float32_canonical_seed{args.seed}"
    )

    temporary_output_directory = (
        output_root
        / (
            "nbaiot_float32_canonical_"
            f"seed{args.seed}__tmp"
        )
    )

    if final_output_directory.exists():
        if not args.overwrite:
            raise FileExistsError(
                "Nihai çıktı klasörü zaten mevcut. "
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
    print("N-BaIoT Float32 Kanonik Parquet Veri Seti")
    print("=" * 78)
    print(f"Kaynak veri      : {source_data_dir}")
    print(f"Kanonik SQLite   : {database_file}")
    print(f"Geçici çıktı     : {temporary_output_directory}")
    print(f"Nihai çıktı      : {final_output_directory}")
    print(f"Seed             : {args.seed}")
    print(f"Özellik sayısı   : {len(feature_columns)}")
    print(f"Parça boyutu     : {args.batch_size:,}")
    print(f"Boş disk alanı   : {free_space_gb:.2f} GB")
    print("=" * 78)

    if free_space_gb < 3:
        print(
            "UYARI: Boş disk alanı 3 GB altında."
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

        (
            total_group_count,
            conflict_group_count,
            usable_group_count,
        ) = validate_canonical_database(
            connection
        )

        classes = load_classes(
            connection
        )

        code_to_label = {
            class_code: class_label
            for class_code, class_label
            in enumerate(classes)
        }

        label_to_code = {
            class_label: class_code
            for class_code, class_label
            in code_to_label.items()
        }

        build_canonical_assignments(
            connection=connection,
            seed=args.seed,
            expected_usable_count=(
                usable_group_count
            ),
            batch_size=args.batch_size,
            rebuild=args.rebuild_assignments,
        )

        expected_split_counts = {
            CODE_TO_SPLIT[int(row[0])]: int(row[1])
            for row in connection.execute(
                """
                SELECT
                    new_split_code,
                    COUNT(*)
                FROM canonical_assignments
                GROUP BY new_split_code
                ORDER BY new_split_code
                """
            ).fetchall()
        }

        if set(expected_split_counts) != set(
            SPLIT_NAMES
        ):
            raise RuntimeError(
                "Atama tablosunda train/validation/test "
                "splitlerinin tamamı bulunmuyor."
            )

        assignment_class_rows = (
            connection.execute(
                """
                SELECT
                    new_split_code,
                    class_code,
                    COUNT(*)
                FROM canonical_assignments
                GROUP BY
                    new_split_code,
                    class_code
                ORDER BY
                    new_split_code,
                    class_code
                """
            ).fetchall()
        )

        expected_class_split_counts = {
            (
                CODE_TO_SPLIT[int(row[0])],
                int(row[1]),
            ): int(row[2])
            for row in assignment_class_rows
        }

        for class_code in code_to_label:
            for split_name in SPLIT_NAMES:
                if (
                    expected_class_split_counts.get(
                        (
                            split_name,
                            class_code,
                        ),
                        0,
                    )
                    == 0
                ):
                    raise RuntimeError(
                        "Bir sınıf yeni splitlerden birinde yok: "
                        f"class={code_to_label[class_code]}, "
                        f"split={split_name}"
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
        ) = open_writers(
            output_directory=(
                temporary_output_directory
            ),
            feature_schema=feature_schema,
            metadata_schema=metadata_schema,
        )

        actual_split_counts: Counter[str] = (
            Counter()
        )

        actual_class_split_counts: dict[
            str,
            Counter[str],
        ] = defaultdict(Counter)

        processed_group_count = 0

        # --------------------------------------------------
        # Scan original float64 Parquet files
        # --------------------------------------------------

        for source_split_code, source_split_name in enumerate(
            SPLIT_NAMES
        ):
            source_file = (
                source_data_dir
                / f"{source_split_name}.parquet"
            )

            if not source_file.exists():
                raise FileNotFoundError(
                    f"Kaynak Parquet bulunamadı: {source_file}"
                )

            parquet_file = pq.ParquetFile(
                source_file
            )

            expected_source_columns = (
                feature_columns
                + ["class_label"]
            )

            if (
                parquet_file.schema_arrow.names
                != expected_source_columns
            ):
                raise ValueError(
                    f"{source_split_name} kaynak şeması uyuşmuyor."
                )

            for column_name in feature_columns:
                arrow_type = (
                    parquet_file.schema_arrow.field(
                        column_name
                    ).type
                )

                if arrow_type != pa.float64():
                    raise TypeError(
                        f"{source_split_name}/{column_name} "
                        f"float64 değil: {arrow_type}"
                    )

            source_row_count = int(
                parquet_file.metadata.num_rows
            )

            source_offset = 0

            progress = tqdm(
                total=source_row_count,
                desc=(
                    f"{source_split_name} "
                    "temsilcileri yazılıyor"
                ),
                unit="satır",
            )

            for batch in parquet_file.iter_batches(
                batch_size=args.batch_size,
                columns=expected_source_columns,
            ):
                frame = batch.to_pandas()

                batch_length = int(
                    len(frame)
                )

                first_source_row = (
                    source_offset + 1
                )

                last_source_row = (
                    source_offset
                    + batch_length
                )

                packed_first = (
                    (
                        source_split_code
                        << REPRESENTATIVE_SHIFT
                    )
                    | first_source_row
                )

                packed_last = (
                    (
                        source_split_code
                        << REPRESENTATIVE_SHIFT
                    )
                    | last_source_row
                )

                assignment_rows = (
                    connection.execute(
                        """
                        SELECT
                            hash_forward,
                            hash_reverse,
                            class_code,
                            new_split_code,
                            representative_location,
                            float64_unique_occurrence_count,
                            raw_occurrence_count,
                            old_split_mask

                        FROM canonical_assignments

                        WHERE representative_location
                              BETWEEN ? AND ?

                        ORDER BY representative_location
                        """,
                        (
                            packed_first,
                            packed_last,
                        ),
                    ).fetchall()
                )

                source_offset += (
                    batch_length
                )

                progress.update(
                    batch_length
                )

                if not assignment_rows:
                    continue

                representative_locations = np.asarray(
                    [
                        int(row[4])
                        for row in assignment_rows
                    ],
                    dtype=np.int64,
                )

                representative_rows = (
                    representative_locations
                    & REPRESENTATIVE_ROW_MASK
                )

                local_indices = (
                    representative_rows
                    - first_source_row
                )

                if (
                    np.any(local_indices < 0)
                    or np.any(
                        local_indices >= batch_length
                    )
                ):
                    raise RuntimeError(
                        "Temsilci satır konumu Parquet "
                        "parçasının dışında."
                    )

                selected = frame.iloc[
                    local_indices
                ].reset_index(
                    drop=True
                )

                (
                    float32_features,
                    actual_forward,
                    actual_reverse,
                ) = create_float32_features_and_hashes(
                    selected[
                        feature_columns
                    ]
                )

                expected_forward = np.asarray(
                    [
                        int(row[0])
                        for row in assignment_rows
                    ],
                    dtype=np.int64,
                )

                expected_reverse = np.asarray(
                    [
                        int(row[1])
                        for row in assignment_rows
                    ],
                    dtype=np.int64,
                )

                if not np.array_equal(
                    actual_forward,
                    expected_forward,
                ):
                    mismatch_count = int(
                        np.count_nonzero(
                            actual_forward
                            != expected_forward
                        )
                    )

                    raise RuntimeError(
                        "Float32 ileri parmak izleri "
                        f"SQLite ile uyuşmuyor: {mismatch_count:,}"
                    )

                if not np.array_equal(
                    actual_reverse,
                    expected_reverse,
                ):
                    mismatch_count = int(
                        np.count_nonzero(
                            actual_reverse
                            != expected_reverse
                        )
                    )

                    raise RuntimeError(
                        "Float32 ters parmak izleri "
                        f"SQLite ile uyuşmuyor: {mismatch_count:,}"
                    )

                expected_class_codes = np.asarray(
                    [
                        int(row[2])
                        for row in assignment_rows
                    ],
                    dtype=np.int16,
                )

                source_labels = (
                    selected["class_label"]
                    .astype(str)
                )

                source_label_codes = source_labels.map(
                    label_to_code
                )

                if source_label_codes.isna().any():
                    unknown_labels = sorted(
                        source_labels[
                            source_label_codes.isna()
                        ].unique().tolist()
                    )

                    raise RuntimeError(
                        "Bilinmeyen kaynak sınıf etiketi var: "
                        + ", ".join(unknown_labels)
                    )

                actual_class_codes = (
                    source_label_codes.to_numpy(
                        dtype=np.int16
                    )
                )

                if not np.array_equal(
                    actual_class_codes,
                    expected_class_codes,
                ):
                    mismatch_count = int(
                        np.count_nonzero(
                            actual_class_codes
                            != expected_class_codes
                        )
                    )

                    raise RuntimeError(
                        "Kaynak sınıf etiketleri SQLite "
                        f"ile uyuşmuyor: {mismatch_count:,}"
                    )

                target_split_codes = np.asarray(
                    [
                        int(row[3])
                        for row in assignment_rows
                    ],
                    dtype=np.uint8,
                )

                calculated_split_codes = (
                    calculate_split_codes(
                        hash_forward=actual_forward,
                        hash_reverse=actual_reverse,
                        seed=args.seed,
                    )
                )

                if not np.array_equal(
                    target_split_codes,
                    calculated_split_codes,
                ):
                    mismatch_count = int(
                        np.count_nonzero(
                            target_split_codes
                            != calculated_split_codes
                        )
                    )

                    raise RuntimeError(
                        "Yeni split ataması parmak iziyle "
                        f"uyuşmuyor: {mismatch_count:,}"
                    )

                # ------------------------------------------
                # Write model data by target split
                # ------------------------------------------

                for target_split_code, target_split_name in enumerate(
                    SPLIT_NAMES
                ):
                    positions = np.flatnonzero(
                        target_split_codes
                        == target_split_code
                    )

                    if positions.size == 0:
                        continue

                    output_frame = (
                        float32_features.iloc[
                            positions
                        ].copy()
                    )

                    output_labels = [
                        code_to_label[
                            int(expected_class_codes[
                                position
                            ])
                        ]
                        for position in positions
                    ]

                    output_frame[
                        "class_label"
                    ] = output_labels

                    output_table = (
                        pa.Table.from_pandas(
                            output_frame[
                                feature_columns
                                + ["class_label"]
                            ],
                            schema=feature_schema,
                            preserve_index=False,
                        )
                    )

                    split_writers[
                        target_split_name
                    ].write_table(
                        output_table,
                        row_group_size=(
                            args.row_group_size
                        ),
                    )

                    selected_count = int(
                        len(positions)
                    )

                    actual_split_counts[
                        target_split_name
                    ] += selected_count

                    for position in positions:
                        class_label = code_to_label[
                            int(
                                expected_class_codes[
                                    position
                                ]
                            )
                        ]

                        actual_class_split_counts[
                            class_label
                        ][target_split_name] += 1

                # ------------------------------------------
                # Write metadata
                # ------------------------------------------

                target_split_names = [
                    CODE_TO_SPLIT[
                        int(split_code)
                    ]
                    for split_code
                    in target_split_codes
                ]

                class_labels = [
                    code_to_label[
                        int(class_code)
                    ]
                    for class_code
                    in expected_class_codes
                ]

                float64_unique_counts = np.asarray(
                    [
                        int(row[5])
                        for row in assignment_rows
                    ],
                    dtype=np.int64,
                )

                raw_occurrence_counts = np.asarray(
                    [
                        int(row[6])
                        for row in assignment_rows
                    ],
                    dtype=np.int64,
                )

                old_split_masks = np.asarray(
                    [
                        int(row[7])
                        for row in assignment_rows
                    ],
                    dtype=np.int16,
                )

                old_split_counts = np.asarray(
                    [
                        int(int(mask).bit_count())
                        for mask in old_split_masks
                    ],
                    dtype=np.int8,
                )

                metadata_frame = pd.DataFrame(
                    {
                        "hash_forward": (
                            expected_forward
                        ),
                        "hash_reverse": (
                            expected_reverse
                        ),
                        "split": (
                            target_split_names
                        ),
                        "class_label": (
                            class_labels
                        ),
                        "representative_source_split": (
                            source_split_name
                        ),
                        "representative_source_row_number": (
                            representative_rows.astype(
                                np.int64
                            )
                        ),
                        "float64_unique_occurrence_count": (
                            float64_unique_counts
                        ),
                        "raw_occurrence_count": (
                            raw_occurrence_counts
                        ),
                        "old_split_mask": (
                            old_split_masks
                        ),
                        "old_split_count": (
                            old_split_counts
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

                processed_group_count += int(
                    len(assignment_rows)
                )

            progress.close()

            if source_offset != source_row_count:
                raise RuntimeError(
                    f"{source_split_name} kaynak satır sayısı "
                    "uyuşmuyor."
                )

        close_writers(
            split_writers=split_writers,
            metadata_writer=metadata_writer,
        )

        split_writers = {}
        metadata_writer = None

        # --------------------------------------------------
        # Final count validation
        # --------------------------------------------------

        if processed_group_count != usable_group_count:
            raise RuntimeError(
                "İşlenen kanonik grup sayısı uyuşmuyor: "
                f"beklenen={usable_group_count:,}, "
                f"işlenen={processed_group_count:,}"
            )

        for split_name in SPLIT_NAMES:
            expected_count = int(
                expected_split_counts[
                    split_name
                ]
            )

            actual_count = int(
                actual_split_counts[
                    split_name
                ]
            )

            if actual_count != expected_count:
                raise RuntimeError(
                    f"{split_name} örnek sayısı uyuşmuyor: "
                    f"beklenen={expected_count:,}, "
                    f"oluşturulan={actual_count:,}"
                )

        for (
            split_name,
            class_code,
        ), expected_count in (
            expected_class_split_counts.items()
        ):
            class_label = code_to_label[
                class_code
            ]

            actual_count = int(
                actual_class_split_counts[
                    class_label
                ][split_name]
            )

            if actual_count != expected_count:
                raise RuntimeError(
                    "Sınıf-split sayısı uyuşmuyor: "
                    f"class={class_label}, "
                    f"split={split_name}, "
                    f"beklenen={expected_count:,}, "
                    f"oluşturulan={actual_count:,}"
                )

        temporary_output_directory.replace(
            final_output_directory
        )

        # --------------------------------------------------
        # Reports
        # --------------------------------------------------

        split_records: list[
            dict[str, Any]
        ] = []

        for split_name in SPLIT_NAMES:
            output_file = (
                final_output_directory
                / f"{split_name}.parquet"
            )

            sample_count = int(
                actual_split_counts[
                    split_name
                ]
            )

            split_records.append(
                {
                    "split": split_name,
                    "sample_count": (
                        sample_count
                    ),
                    "sample_percentage": (
                        sample_count
                        / usable_group_count
                        * 100
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

        class_records: list[
            dict[str, Any]
        ] = []

        for class_label in classes:
            class_total = int(
                sum(
                    actual_class_split_counts[
                        class_label
                    ].values()
                )
            )

            for split_name in SPLIT_NAMES:
                sample_count = int(
                    actual_class_split_counts[
                        class_label
                    ][split_name]
                )

                class_records.append(
                    {
                        "class_label": class_label,
                        "split": split_name,
                        "sample_count": (
                            sample_count
                        ),
                        "percentage_within_class": (
                            sample_count
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
            / "nbaiot_float32_canonical_class_distribution.csv"
        )

        summary_file = (
            report_dir
            / "nbaiot_float32_canonical_materialization_summary.json"
        )

        class_distribution.to_csv(
            class_report_file,
            index=False,
            encoding="utf-8",
        )

        metadata_file = (
            final_output_directory
            / "metadata.parquet"
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
                "float32"
            ),
            "class_count": int(
                len(classes)
            ),
            "classes": classes,
            "original_float32_group_count": int(
                total_group_count
            ),
            "excluded_class_conflict_group_count": int(
                conflict_group_count
            ),
            "materialized_canonical_sample_count": int(
                usable_group_count
            ),
            "split_assignment_method": (
                "Deterministic SplitMix64 assignment from "
                "the float32 composite fingerprint"
            ),
            "split_files": (
                split_records
            ),
            "metadata_file": str(
                metadata_file
            ),
            "metadata_file_size_bytes": int(
                metadata_file.stat().st_size
            ),
            "output_directory": str(
                final_output_directory
            ),
            "class_distribution_report": str(
                class_report_file
            ),
            "leakage_control": (
                "Every conflict-free float32 feature vector "
                "is materialized once and assigned to exactly one split."
            ),
            "conflict_policy": (
                "All float32 groups containing multiple class "
                "labels are excluded in full."
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
        print("Float32 kanonik veri seti tamamlandı")
        print("=" * 78)

        for record in split_records:
            print(
                f"{record['split']:10s}: "
                f"{record['sample_count']:,} örnek "
                f"(%{record['sample_percentage']:.4f}), "
                f"{record['file_size_gb']:.4f} GB"
            )

        print()
        print(
            "Toplam float32 grup      : "
            f"{total_group_count:,}"
        )
        print(
            "Dışlanan çelişkili grup  : "
            f"{conflict_group_count:,}"
        )
        print(
            "Kullanılabilir örnek     : "
            f"{usable_group_count:,}"
        )
        print(
            "Özellik sayısı           : "
            f"{len(feature_columns)}"
        )
        print(
            "Sınıf sayısı             : "
            f"{len(classes)}"
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