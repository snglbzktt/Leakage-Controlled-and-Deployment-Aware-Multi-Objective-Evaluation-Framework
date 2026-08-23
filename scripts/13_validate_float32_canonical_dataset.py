"""
N-BaIoT float32 kanonik veri seti bütünlük doğrulaması.

Kontroller:
- SQLite kanonik grup ve split atamaları geçerli mi?
- Metadata, SQLite canonical_assignments tablosuyla birebir eşleşiyor mu?
- Metadata parmak izleri global olarak benzersiz mi?
- Train, validation ve test arasında float32 vektör örtüşmesi var mı?
- Parquet dosyaları gerçekten float32 mi?
- Parquet satırları metadata ile aynı sırada ve aynı etikette mi?
- Float32 parmak izleri yeniden hesaplandığında metadata ile eşleşiyor mu?
- Split atamaları Seed 2026 ile tekrar üretilebiliyor mu?
- NaN veya sonsuz değer var mı?
- Sınıf ve split örnek sayıları SQLite ile uyuşuyor mu?

Çıktılar:
results/reports/nbaiot_float32_canonical_validation_by_split.csv
results/reports/nbaiot_float32_canonical_validation_summary.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
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

DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_float32_canonical_seed2026"
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
    for split_code, split_name in enumerate(SPLIT_NAMES)
}

CODE_TO_SPLIT = {
    split_code: split_name
    for split_name, split_code in SPLIT_TO_CODE.items()
}

UINT64_RANGE = 1 << 64
UINT64_MASK = UINT64_RANGE - 1

REPRESENTATIVE_SHIFT = 40


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT float32 kanonik Parquet veri setini "
            "SQLite kanonik indeksiyle doğrular."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Float32 kanonik Parquet veri klasörü.",
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
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Doğrulama raporlarının kaydedileceği klasör.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Kanonik split tohumu.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Parquet ve SQLite okuma parça büyüklüğü.",
    )

    return parser.parse_args()


# ==========================================================
# JSON HELPERS
# ==========================================================

def json_default(value: object) -> object:
    """NumPy ve Path değerlerini JSON uyumlu hâle getirir."""

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
    """JSON dosyasını geçici dosya üzerinden güvenli biçimde yazar."""

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

    feature_columns = (
        schema["column_name"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if not feature_columns:
        raise ValueError(
            "Özellik sütun listesi boş."
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(
            "Şema dosyasında tekrarlanan sütun adı var."
        )

    return feature_columns


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
    """Float32 parmak izlerinden split kodlarını tekrar hesaplar."""

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
# FLOAT32 FINGERPRINT
# ==========================================================

def create_float32_fingerprints(
    feature_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Float32 özellik vektörlerinin çift parmak izini hesaplar."""

    values = feature_frame.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Özellikler içinde NaN veya sonsuz değer bulundu."
        )

    # -0.0 ve +0.0 aynı kanonik gösterime dönüştürülür.
    values[
        values == np.float32(0.0)
    ] = np.float32(0.0)

    canonical_frame = pd.DataFrame(
        values,
        columns=feature_frame.columns,
    )

    forward = pd.util.hash_pandas_object(
        canonical_frame,
        index=False,
        categorize=False,
    ).to_numpy(
        dtype=np.uint64
    )

    reverse = pd.util.hash_pandas_object(
        canonical_frame.iloc[:, ::-1],
        index=False,
        categorize=False,
    ).to_numpy(
        dtype=np.uint64
    )

    return (
        forward.view(np.int64),
        reverse.view(np.int64),
    )


# ==========================================================
# SQLITE HELPERS
# ==========================================================

def table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    """SQLite tablosunun mevcut olup olmadığını kontrol eder."""

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
    """Tek sayısal sonuç döndüren SQL sorgusunu çalıştırır."""

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
    """run_metadata tablosundaki değeri okur."""

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


def validate_database(
    connection: sqlite3.Connection,
    expected_seed: int,
) -> tuple[
    list[str],
    int,
    int,
    int,
    dict[str, int],
    dict[tuple[str, int], int],
]:
    """SQLite kanonik indeksini ve split atamalarını doğrular."""

    required_tables = {
        "run_metadata",
        "float32_groups",
        "canonical_assignments",
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
            "SQLite veritabanında gerekli tablolar eksik: "
            + ", ".join(missing_tables)
        )

    classes_json = read_run_metadata(
        connection,
        "classes_json",
    )

    if classes_json is None:
        raise RuntimeError(
            "run_metadata içinde classes_json bulunamadı."
        )

    classes_raw = json.loads(
        classes_json
    )

    if not isinstance(classes_raw, list) or not classes_raw:
        raise RuntimeError(
            "SQLite sınıf listesi geçersiz."
        )

    classes = [
        str(class_label)
        for class_label in classes_raw
    ]

    stored_seed = read_run_metadata(
        connection,
        "canonical_split_seed",
    )

    if stored_seed is None:
        raise RuntimeError(
            "SQLite içinde canonical_split_seed bulunamadı."
        )

    if int(stored_seed) != expected_seed:
        raise RuntimeError(
            "SQLite split seed değeri uyuşmuyor: "
            f"SQLite={stored_seed}, beklenen={expected_seed}"
        )

    total_float32_group_count = scalar_query(
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

    assignment_count = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM canonical_assignments
        """,
    )

    expected_assignment_count = (
        total_float32_group_count
        - conflict_group_count
    )

    if assignment_count != expected_assignment_count:
        raise RuntimeError(
            "Kanonik atama sayısı uyuşmuyor: "
            f"beklenen={expected_assignment_count:,}, "
            f"bulunan={assignment_count:,}"
        )

    split_rows = connection.execute(
        """
        SELECT
            new_split_code,
            COUNT(*)
        FROM canonical_assignments
        GROUP BY new_split_code
        ORDER BY new_split_code
        """
    ).fetchall()

    split_counts: dict[str, int] = {}

    for split_code, count in split_rows:
        split_code = int(split_code)

        if split_code not in CODE_TO_SPLIT:
            raise RuntimeError(
                f"SQLite içinde geçersiz split kodu var: {split_code}"
            )

        split_counts[
            CODE_TO_SPLIT[split_code]
        ] = int(count)

    if set(split_counts) != set(SPLIT_NAMES):
        raise RuntimeError(
            "SQLite atamalarında train, validation ve testin "
            "tamamı bulunmuyor."
        )

    class_split_rows = connection.execute(
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

    class_split_counts: dict[
        tuple[str, int],
        int,
    ] = {}

    for split_code, class_code, count in class_split_rows:
        split_code = int(split_code)
        class_code = int(class_code)

        if split_code not in CODE_TO_SPLIT:
            raise RuntimeError(
                f"Geçersiz split kodu bulundu: {split_code}"
            )

        if class_code < 0 or class_code >= len(classes):
            raise RuntimeError(
                f"Geçersiz sınıf kodu bulundu: {class_code}"
            )

        class_split_counts[
            (
                CODE_TO_SPLIT[split_code],
                class_code,
            )
        ] = int(count)

    for split_name in SPLIT_NAMES:
        for class_code in range(len(classes)):
            if class_split_counts.get(
                (
                    split_name,
                    class_code,
                ),
                0,
            ) == 0:
                raise RuntimeError(
                    "Bir sınıf splitlerden birinde bulunmuyor: "
                    f"class={classes[class_code]}, "
                    f"split={split_name}"
                )

    return (
        classes,
        total_float32_group_count,
        conflict_group_count,
        assignment_count,
        split_counts,
        class_split_counts,
    )


# ==========================================================
# METADATA VALIDATION
# ==========================================================

def load_and_validate_metadata(
    metadata_file: Path,
    connection: sqlite3.Connection,
    classes: list[str],
    expected_total_count: int,
    expected_split_counts: dict[str, int],
    seed: int,
    batch_size: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Metadata dosyasını SQLite ile birebir doğrular."""

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata Parquet bulunamadı: {metadata_file}"
        )

    parquet_file = pq.ParquetFile(
        metadata_file
    )

    expected_columns = [
        "hash_forward",
        "hash_reverse",
        "split",
        "class_label",
        "representative_source_split",
        "representative_source_row_number",
        "float64_unique_occurrence_count",
        "raw_occurrence_count",
        "old_split_mask",
        "old_split_count",
    ]

    if parquet_file.schema_arrow.names != expected_columns:
        raise ValueError(
            "Metadata Parquet sütun şeması uyuşmuyor."
        )

    metadata_row_count = int(
        parquet_file.metadata.num_rows
    )

    if metadata_row_count != expected_total_count:
        raise RuntimeError(
            "Metadata satır sayısı uyuşmuyor: "
            f"beklenen={expected_total_count:,}, "
            f"bulunan={metadata_row_count:,}"
        )

    label_to_code = {
        class_label: class_code
        for class_code, class_label in enumerate(classes)
    }

    hash_forward = np.empty(
        metadata_row_count,
        dtype=np.int64,
    )

    hash_reverse = np.empty(
        metadata_row_count,
        dtype=np.int64,
    )

    split_codes = np.empty(
        metadata_row_count,
        dtype=np.uint8,
    )

    class_codes = np.empty(
        metadata_row_count,
        dtype=np.int16,
    )

    representative_locations = np.empty(
        metadata_row_count,
        dtype=np.int64,
    )

    float64_unique_counts = np.empty(
        metadata_row_count,
        dtype=np.int64,
    )

    raw_occurrence_counts = np.empty(
        metadata_row_count,
        dtype=np.int64,
    )

    old_split_masks = np.empty(
        metadata_row_count,
        dtype=np.int16,
    )

    old_split_counts = np.empty(
        metadata_row_count,
        dtype=np.int8,
    )

    offset = 0

    progress = tqdm(
        total=metadata_row_count,
        desc="Metadata doğrulanıyor",
        unit="satır",
    )

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=expected_columns,
    ):
        frame = batch.to_pandas()

        if frame.isna().any().any():
            raise RuntimeError(
                "Metadata içinde eksik değer bulundu."
            )

        batch_length = int(
            len(frame)
        )

        batch_end = (
            offset + batch_length
        )

        batch_hash_forward = frame[
            "hash_forward"
        ].to_numpy(
            dtype=np.int64,
            copy=False,
        )

        batch_hash_reverse = frame[
            "hash_reverse"
        ].to_numpy(
            dtype=np.int64,
            copy=False,
        )

        split_names = frame[
            "split"
        ].astype(str)

        unknown_splits = (
            set(split_names.unique())
            - set(SPLIT_NAMES)
        )

        if unknown_splits:
            raise RuntimeError(
                "Metadata içinde geçersiz split adı var: "
                + ", ".join(sorted(unknown_splits))
            )

        batch_split_codes = split_names.map(
            SPLIT_TO_CODE
        ).to_numpy(
            dtype=np.uint8
        )

        class_labels = frame[
            "class_label"
        ].astype(str)

        encoded_classes = class_labels.map(
            label_to_code
        )

        if encoded_classes.isna().any():
            unknown_classes = sorted(
                class_labels[
                    encoded_classes.isna()
                ].unique().tolist()
            )

            raise RuntimeError(
                "Metadata içinde bilinmeyen sınıflar var: "
                + ", ".join(unknown_classes)
            )

        batch_class_codes = encoded_classes.to_numpy(
            dtype=np.int16
        )

        source_split_names = frame[
            "representative_source_split"
        ].astype(str)

        unknown_source_splits = (
            set(source_split_names.unique())
            - set(SPLIT_NAMES)
        )

        if unknown_source_splits:
            raise RuntimeError(
                "Metadata içinde geçersiz temsilci split adı var: "
                + ", ".join(sorted(unknown_source_splits))
            )

        source_split_codes = source_split_names.map(
            SPLIT_TO_CODE
        ).to_numpy(
            dtype=np.int64
        )

        source_row_numbers = pd.to_numeric(
            frame[
                "representative_source_row_number"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.int64
        )

        if np.any(source_row_numbers <= 0):
            raise RuntimeError(
                "Temsilci kaynak satır numarası sıfır veya negatif."
            )

        batch_representative_locations = (
            (
                source_split_codes
                << np.int64(REPRESENTATIVE_SHIFT)
            )
            | source_row_numbers
        )

        batch_float64_counts = pd.to_numeric(
            frame[
                "float64_unique_occurrence_count"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.int64
        )

        batch_raw_counts = pd.to_numeric(
            frame[
                "raw_occurrence_count"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.int64
        )

        batch_old_masks = pd.to_numeric(
            frame[
                "old_split_mask"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.int16
        )

        batch_old_counts = pd.to_numeric(
            frame[
                "old_split_count"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.int8
        )

        calculated_old_counts = np.fromiter(
            (
                int(mask).bit_count()
                for mask in batch_old_masks
            ),
            dtype=np.int8,
            count=batch_length,
        )

        if not np.array_equal(
            batch_old_counts,
            calculated_old_counts,
        ):
            mismatch_count = int(
                np.count_nonzero(
                    batch_old_counts
                    != calculated_old_counts
                )
            )

            raise RuntimeError(
                "Metadata old_split_count değeri maskeyle "
                f"uyuşmuyor: {mismatch_count:,}"
            )

        calculated_split_codes = calculate_split_codes(
            hash_forward=batch_hash_forward,
            hash_reverse=batch_hash_reverse,
            seed=seed,
        )

        if not np.array_equal(
            batch_split_codes,
            calculated_split_codes,
        ):
            mismatch_count = int(
                np.count_nonzero(
                    batch_split_codes
                    != calculated_split_codes
                )
            )

            raise RuntimeError(
                "Metadata split atamaları float32 parmak iziyle "
                f"uyuşmuyor: {mismatch_count:,}"
            )

        hash_forward[
            offset:batch_end
        ] = batch_hash_forward

        hash_reverse[
            offset:batch_end
        ] = batch_hash_reverse

        split_codes[
            offset:batch_end
        ] = batch_split_codes

        class_codes[
            offset:batch_end
        ] = batch_class_codes

        representative_locations[
            offset:batch_end
        ] = batch_representative_locations

        float64_unique_counts[
            offset:batch_end
        ] = batch_float64_counts

        raw_occurrence_counts[
            offset:batch_end
        ] = batch_raw_counts

        old_split_masks[
            offset:batch_end
        ] = batch_old_masks

        old_split_counts[
            offset:batch_end
        ] = batch_old_counts

        offset = batch_end

        progress.update(
            batch_length
        )

    progress.close()

    if offset != metadata_row_count:
        raise RuntimeError(
            "İşlenen metadata satır sayısı uyuşmuyor."
        )

    actual_split_counts = Counter(
        CODE_TO_SPLIT[
            int(split_code)
        ]
        for split_code in split_codes
    )

    for split_name in SPLIT_NAMES:
        actual_count = int(
            actual_split_counts[
                split_name
            ]
        )

        expected_count = int(
            expected_split_counts[
                split_name
            ]
        )

        if actual_count != expected_count:
            raise RuntimeError(
                f"Metadata {split_name} sayısı uyuşmuyor: "
                f"beklenen={expected_count:,}, "
                f"bulunan={actual_count:,}"
            )

    # ------------------------------------------------------
    # Global uniqueness
    # ------------------------------------------------------

    sort_order = np.lexsort(
        (
            hash_reverse,
            hash_forward,
        )
    )

    sorted_forward = hash_forward[
        sort_order
    ]

    sorted_reverse = hash_reverse[
        sort_order
    ]

    duplicate_mask = (
        (
            sorted_forward[1:]
            == sorted_forward[:-1]
        )
        &
        (
            sorted_reverse[1:]
            == sorted_reverse[:-1]
        )
    )

    duplicate_fingerprint_count = int(
        np.count_nonzero(
            duplicate_mask
        )
    )

    if duplicate_fingerprint_count > 0:
        raise RuntimeError(
            "Metadata içinde tekrarlanan float32 parmak izi var: "
            f"{duplicate_fingerprint_count:,}"
        )

    # ------------------------------------------------------
    # Read SQLite assignments in fingerprint order
    # ------------------------------------------------------

    database_hash_forward = np.empty(
        expected_total_count,
        dtype=np.int64,
    )

    database_hash_reverse = np.empty(
        expected_total_count,
        dtype=np.int64,
    )

    database_class_codes = np.empty(
        expected_total_count,
        dtype=np.int16,
    )

    database_split_codes = np.empty(
        expected_total_count,
        dtype=np.uint8,
    )

    database_locations = np.empty(
        expected_total_count,
        dtype=np.int64,
    )

    database_float64_counts = np.empty(
        expected_total_count,
        dtype=np.int64,
    )

    database_raw_counts = np.empty(
        expected_total_count,
        dtype=np.int64,
    )

    database_old_masks = np.empty(
        expected_total_count,
        dtype=np.int16,
    )

    cursor = connection.execute(
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
        ORDER BY
            hash_forward,
            hash_reverse
        """
    )

    database_offset = 0

    progress = tqdm(
        total=expected_total_count,
        desc="SQLite atamaları karşılaştırılıyor",
        unit="grup",
    )

    while True:
        rows = cursor.fetchmany(
            batch_size
        )

        if not rows:
            break

        row_count = len(rows)

        row_end = (
            database_offset + row_count
        )

        database_hash_forward[
            database_offset:row_end
        ] = np.asarray(
            [
                int(row[0])
                for row in rows
            ],
            dtype=np.int64,
        )

        database_hash_reverse[
            database_offset:row_end
        ] = np.asarray(
            [
                int(row[1])
                for row in rows
            ],
            dtype=np.int64,
        )

        database_class_codes[
            database_offset:row_end
        ] = np.asarray(
            [
                int(row[2])
                for row in rows
            ],
            dtype=np.int16,
        )

        database_split_codes[
            database_offset:row_end
        ] = np.asarray(
            [
                int(row[3])
                for row in rows
            ],
            dtype=np.uint8,
        )

        database_locations[
            database_offset:row_end
        ] = np.asarray(
            [
                int(row[4])
                for row in rows
            ],
            dtype=np.int64,
        )

        database_float64_counts[
            database_offset:row_end
        ] = np.asarray(
            [
                int(row[5])
                for row in rows
            ],
            dtype=np.int64,
        )

        database_raw_counts[
            database_offset:row_end
        ] = np.asarray(
            [
                int(row[6])
                for row in rows
            ],
            dtype=np.int64,
        )

        database_old_masks[
            database_offset:row_end
        ] = np.asarray(
            [
                int(row[7])
                for row in rows
            ],
            dtype=np.int16,
        )

        database_offset = row_end

        progress.update(
            row_count
        )

    progress.close()

    if database_offset != expected_total_count:
        raise RuntimeError(
            "SQLite atama satır sayısı uyuşmuyor: "
            f"beklenen={expected_total_count:,}, "
            f"okunan={database_offset:,}"
        )

    comparisons = {
        "hash_forward": (
            sorted_forward,
            database_hash_forward,
        ),
        "hash_reverse": (
            sorted_reverse,
            database_hash_reverse,
        ),
        "class_code": (
            class_codes[sort_order],
            database_class_codes,
        ),
        "split_code": (
            split_codes[sort_order],
            database_split_codes,
        ),
        "representative_location": (
            representative_locations[sort_order],
            database_locations,
        ),
        "float64_unique_occurrence_count": (
            float64_unique_counts[sort_order],
            database_float64_counts,
        ),
        "raw_occurrence_count": (
            raw_occurrence_counts[sort_order],
            database_raw_counts,
        ),
        "old_split_mask": (
            old_split_masks[sort_order],
            database_old_masks,
        ),
    }

    for field_name, (
        metadata_values,
        database_values,
    ) in comparisons.items():
        if not np.array_equal(
            metadata_values,
            database_values,
        ):
            mismatch_count = int(
                np.count_nonzero(
                    metadata_values
                    != database_values
                )
            )

            raise RuntimeError(
                "Metadata ile SQLite arasında uyuşmazlık: "
                f"alan={field_name}, "
                f"hatalı={mismatch_count:,}"
            )

    # SQLite karşılaştırma dizileri artık gerekli değil.
    del database_hash_forward
    del database_hash_reverse
    del database_class_codes
    del database_split_codes
    del database_locations
    del database_float64_counts
    del database_raw_counts
    del database_old_masks
    del sort_order
    del sorted_forward
    del sorted_reverse

    expected_hash_forward: dict[
        str,
        np.ndarray,
    ] = {}

    expected_hash_reverse: dict[
        str,
        np.ndarray,
    ] = {}

    expected_class_codes: dict[
        str,
        np.ndarray,
    ] = {}

    for split_name in SPLIT_NAMES:
        split_code = SPLIT_TO_CODE[
            split_name
        ]

        split_indices = np.flatnonzero(
            split_codes == split_code
        )

        expected_hash_forward[
            split_name
        ] = hash_forward[
            split_indices
        ].copy()

        expected_hash_reverse[
            split_name
        ] = hash_reverse[
            split_indices
        ].copy()

        expected_class_codes[
            split_name
        ] = class_codes[
            split_indices
        ].copy()

    metadata_summary = {
        "metadata_row_count": int(
            metadata_row_count
        ),
        "duplicate_fingerprint_count": int(
            duplicate_fingerprint_count
        ),
        "metadata_matches_sqlite": True,
        "split_assignment_valid": True,
        "old_split_masks_valid": True,
        "cross_split_fingerprint_overlap_count": 0,
    }

    return (
        expected_hash_forward,
        expected_hash_reverse,
        expected_class_codes,
        metadata_summary,
    )


# ==========================================================
# SPLIT PARQUET VALIDATION
# ==========================================================

def validate_split_parquet(
    split_name: str,
    parquet_file_path: Path,
    feature_columns: list[str],
    classes: list[str],
    expected_hash_forward: np.ndarray,
    expected_hash_reverse: np.ndarray,
    expected_class_codes: np.ndarray,
    expected_class_split_counts: dict[
        tuple[str, int],
        int,
    ],
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    """Tek bir float32 split Parquet dosyasını doğrular."""

    if not parquet_file_path.exists():
        raise FileNotFoundError(
            f"Parquet dosyası bulunamadı: {parquet_file_path}"
        )

    parquet_file = pq.ParquetFile(
        parquet_file_path
    )

    expected_columns = (
        feature_columns
        + ["class_label"]
    )

    if parquet_file.schema_arrow.names != expected_columns:
        raise ValueError(
            f"{split_name} Parquet sütun şeması uyuşmuyor."
        )

    for feature_name in feature_columns:
        arrow_type = parquet_file.schema_arrow.field(
            feature_name
        ).type

        if arrow_type != pa.float32():
            raise TypeError(
                f"{split_name}/{feature_name} float32 değil: "
                f"{arrow_type}"
            )

    if parquet_file.schema_arrow.field(
        "class_label"
    ).type != pa.string():
        raise TypeError(
            f"{split_name}/class_label string değil."
        )

    expected_row_count = int(
        len(expected_hash_forward)
    )

    parquet_row_count = int(
        parquet_file.metadata.num_rows
    )

    if parquet_row_count != expected_row_count:
        raise RuntimeError(
            f"{split_name} satır sayısı uyuşmuyor: "
            f"beklenen={expected_row_count:,}, "
            f"bulunan={parquet_row_count:,}"
        )

    label_to_code = {
        class_label: class_code
        for class_code, class_label in enumerate(classes)
    }

    class_counts: Counter[int] = Counter()

    missing_value_count = 0
    infinity_value_count = 0

    offset = 0

    progress = tqdm(
        total=parquet_row_count,
        desc=f"{split_name} Parquet doğrulanıyor",
        unit="satır",
    )

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=expected_columns,
    ):
        frame = batch.to_pandas()

        feature_frame = frame[
            feature_columns
        ]

        values = feature_frame.to_numpy(
            dtype=np.float32,
            copy=False,
        )

        batch_missing_count = int(
            np.isnan(values).sum()
        )

        batch_infinity_count = int(
            np.isinf(values).sum()
        )

        missing_value_count += (
            batch_missing_count
        )

        infinity_value_count += (
            batch_infinity_count
        )

        if batch_missing_count > 0:
            raise RuntimeError(
                f"{split_name} içinde NaN değer bulundu."
            )

        if batch_infinity_count > 0:
            raise RuntimeError(
                f"{split_name} içinde sonsuz değer bulundu."
            )

        (
            actual_hash_forward,
            actual_hash_reverse,
        ) = create_float32_fingerprints(
            feature_frame
        )

        batch_length = int(
            len(frame)
        )

        batch_end = (
            offset + batch_length
        )

        expected_forward_batch = (
            expected_hash_forward[
                offset:batch_end
            ]
        )

        expected_reverse_batch = (
            expected_hash_reverse[
                offset:batch_end
            ]
        )

        if not np.array_equal(
            actual_hash_forward,
            expected_forward_batch,
        ):
            mismatch_count = int(
                np.count_nonzero(
                    actual_hash_forward
                    != expected_forward_batch
                )
            )

            raise RuntimeError(
                f"{split_name} ileri parmak izleri metadata "
                f"ile uyuşmuyor: {mismatch_count:,}"
            )

        if not np.array_equal(
            actual_hash_reverse,
            expected_reverse_batch,
        ):
            mismatch_count = int(
                np.count_nonzero(
                    actual_hash_reverse
                    != expected_reverse_batch
                )
            )

            raise RuntimeError(
                f"{split_name} ters parmak izleri metadata "
                f"ile uyuşmuyor: {mismatch_count:,}"
            )

        class_labels = frame[
            "class_label"
        ].astype(str)

        encoded_classes = class_labels.map(
            label_to_code
        )

        if encoded_classes.isna().any():
            unknown_classes = sorted(
                class_labels[
                    encoded_classes.isna()
                ].unique().tolist()
            )

            raise RuntimeError(
                "Parquet içinde bilinmeyen sınıflar bulundu: "
                + ", ".join(unknown_classes)
            )

        actual_class_codes = encoded_classes.to_numpy(
            dtype=np.int16
        )

        expected_class_batch = (
            expected_class_codes[
                offset:batch_end
            ]
        )

        if not np.array_equal(
            actual_class_codes,
            expected_class_batch,
        ):
            mismatch_count = int(
                np.count_nonzero(
                    actual_class_codes
                    != expected_class_batch
                )
            )

            raise RuntimeError(
                f"{split_name} sınıf etiketleri metadata ile "
                f"uyuşmuyor: {mismatch_count:,}"
            )

        calculated_split_codes = calculate_split_codes(
            hash_forward=actual_hash_forward,
            hash_reverse=actual_hash_reverse,
            seed=seed,
        )

        target_split_code = SPLIT_TO_CODE[
            split_name
        ]

        if np.any(
            calculated_split_codes
            != target_split_code
        ):
            mismatch_count = int(
                np.count_nonzero(
                    calculated_split_codes
                    != target_split_code
                )
            )

            raise RuntimeError(
                f"{split_name} dosyasında yanlış split atamalı "
                f"{mismatch_count:,} örnek var."
            )

        unique_codes, unique_counts = np.unique(
            actual_class_codes,
            return_counts=True,
        )

        for class_code, count in zip(
            unique_codes,
            unique_counts,
            strict=True,
        ):
            class_counts[
                int(class_code)
            ] += int(count)

        offset = batch_end

        progress.update(
            batch_length
        )

    progress.close()

    if offset != expected_row_count:
        raise RuntimeError(
            f"{split_name} işlenen satır sayısı uyuşmuyor."
        )

    for class_code, class_label in enumerate(classes):
        expected_count = int(
            expected_class_split_counts[
                (
                    split_name,
                    class_code,
                )
            ]
        )

        actual_count = int(
            class_counts[
                class_code
            ]
        )

        if actual_count != expected_count:
            raise RuntimeError(
                f"{split_name}/{class_label} sayısı uyuşmuyor: "
                f"beklenen={expected_count:,}, "
                f"bulunan={actual_count:,}"
            )

    return {
        "split": split_name,
        "row_count": int(
            offset
        ),
        "row_group_count": int(
            parquet_file.metadata.num_row_groups
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "feature_dtype": "float32",
        "missing_value_count": int(
            missing_value_count
        ),
        "infinity_value_count": int(
            infinity_value_count
        ),
        "fingerprints_match_metadata": True,
        "labels_match_metadata": True,
        "split_assignment_valid": True,
        "class_distribution_matches_sqlite": True,
        "file_size_bytes": int(
            parquet_file_path.stat().st_size
        ),
        "file_size_gb": float(
            parquet_file_path.stat().st_size
            / (1024**3)
        ),
    }


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    data_dir = args.data_dir.resolve()
    database_file = args.database_file.resolve()
    schema_file = args.schema_file.resolve()
    report_dir = args.report_dir.resolve()

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size sıfırdan büyük olmalıdır."
        )

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Float32 veri klasörü bulunamadı: {data_dir}"
        )

    if not database_file.exists():
        raise FileNotFoundError(
            f"Kanonik SQLite bulunamadı: {database_file}"
        )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_columns = load_feature_columns(
        schema_file
    )

    print("=" * 78)
    print("N-BaIoT Float32 Kanonik Veri Seti Doğrulaması")
    print("=" * 78)
    print(f"Veri klasörü      : {data_dir}")
    print(f"Kanonik SQLite    : {database_file}")
    print(f"Özellik sayısı    : {len(feature_columns)}")
    print(f"Seed              : {args.seed}")
    print(f"Batch boyutu      : {args.batch_size:,}")
    print("=" * 78)

    connection = sqlite3.connect(
        database_file,
        timeout=180,
    )

    try:
        (
            classes,
            total_float32_group_count,
            conflict_group_count,
            assignment_count,
            expected_split_counts,
            expected_class_split_counts,
        ) = validate_database(
            connection=connection,
            expected_seed=args.seed,
        )

        metadata_file = (
            data_dir
            / "metadata.parquet"
        )

        (
            expected_hash_forward,
            expected_hash_reverse,
            expected_class_codes,
            metadata_summary,
        ) = load_and_validate_metadata(
            metadata_file=metadata_file,
            connection=connection,
            classes=classes,
            expected_total_count=assignment_count,
            expected_split_counts=expected_split_counts,
            seed=args.seed,
            batch_size=args.batch_size,
        )

        split_records: list[
            dict[str, Any]
        ] = []

        for split_name in SPLIT_NAMES:
            split_records.append(
                validate_split_parquet(
                    split_name=split_name,
                    parquet_file_path=(
                        data_dir
                        / f"{split_name}.parquet"
                    ),
                    feature_columns=feature_columns,
                    classes=classes,
                    expected_hash_forward=(
                        expected_hash_forward[
                            split_name
                        ]
                    ),
                    expected_hash_reverse=(
                        expected_hash_reverse[
                            split_name
                        ]
                    ),
                    expected_class_codes=(
                        expected_class_codes[
                            split_name
                        ]
                    ),
                    expected_class_split_counts=(
                        expected_class_split_counts
                    ),
                    seed=args.seed,
                    batch_size=args.batch_size,
                )
            )

        split_report = pd.DataFrame(
            split_records
        )

        split_report_file = (
            report_dir
            / "nbaiot_float32_canonical_validation_by_split.csv"
        )

        summary_file = (
            report_dir
            / "nbaiot_float32_canonical_validation_summary.json"
        )

        split_report.to_csv(
            split_report_file,
            index=False,
            encoding="utf-8",
        )

        total_validated_rows = int(
            split_report[
                "row_count"
            ].sum()
        )

        total_missing_values = int(
            split_report[
                "missing_value_count"
            ].sum()
        )

        total_infinity_values = int(
            split_report[
                "infinity_value_count"
            ].sum()
        )

        all_fingerprints_match = bool(
            all(
                record[
                    "fingerprints_match_metadata"
                ]
                for record in split_records
            )
        )

        all_labels_match = bool(
            all(
                record[
                    "labels_match_metadata"
                ]
                for record in split_records
            )
        )

        all_split_assignments_valid = bool(
            all(
                record[
                    "split_assignment_valid"
                ]
                for record in split_records
            )
        )

        all_class_distributions_match = bool(
            all(
                record[
                    "class_distribution_matches_sqlite"
                ]
                for record in split_records
            )
        )

        all_values_finite = bool(
            total_missing_values == 0
            and total_infinity_values == 0
        )

        validation_passed = bool(
            total_validated_rows == assignment_count
            and metadata_summary[
                "duplicate_fingerprint_count"
            ] == 0
            and metadata_summary[
                "cross_split_fingerprint_overlap_count"
            ] == 0
            and metadata_summary[
                "metadata_matches_sqlite"
            ]
            and all_fingerprints_match
            and all_labels_match
            and all_split_assignments_valid
            and all_class_distributions_match
            and all_values_finite
        )

        if not validation_passed:
            raise RuntimeError(
                "Float32 kanonik veri seti doğrulaması başarısız."
            )

        summary: dict[str, Any] = {
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "data_directory": str(
                data_dir
            ),
            "database_file": str(
                database_file
            ),
            "seed": int(
                args.seed
            ),
            "feature_count": int(
                len(feature_columns)
            ),
            "feature_dtype": "float32",
            "class_count": int(
                len(classes)
            ),
            "classes": classes,
            "total_float32_group_count": int(
                total_float32_group_count
            ),
            "excluded_class_conflict_group_count": int(
                conflict_group_count
            ),
            "validated_canonical_sample_count": int(
                assignment_count
            ),
            "metadata_validation": (
                metadata_summary
            ),
            "split_validation": (
                split_records
            ),
            "total_missing_value_count": int(
                total_missing_values
            ),
            "total_infinity_value_count": int(
                total_infinity_values
            ),
            "global_duplicate_fingerprint_count": 0,
            "cross_split_fingerprint_overlap_count": 0,
            "metadata_matches_sqlite": True,
            "all_parquet_fingerprints_match_metadata": (
                all_fingerprints_match
            ),
            "all_labels_match_metadata": (
                all_labels_match
            ),
            "all_split_assignments_valid": (
                all_split_assignments_valid
            ),
            "all_class_distributions_match_sqlite": (
                all_class_distributions_match
            ),
            "all_values_finite": (
                all_values_finite
            ),
            "validation_passed": (
                validation_passed
            ),
            "split_report": str(
                split_report_file
            ),
        }

        write_json_atomic(
            data=summary,
            output_file=summary_file,
        )

        print()
        print("=" * 78)
        print("Float32 kanonik veri seti doğrulaması tamamlandı")
        print("=" * 78)

        for record in split_records:
            print(
                f"{record['split']:10s}: "
                f"{record['row_count']:,} satır, "
                f"NaN={record['missing_value_count']:,}, "
                f"Inf={record['infinity_value_count']:,}, "
                "parmak izi=True, etiket=True"
            )

        print()
        print(
            "Toplam float32 grup        : "
            f"{total_float32_group_count:,}"
        )
        print(
            "Dışlanan çelişkili grup    : "
            f"{conflict_group_count:,}"
        )
        print(
            "Doğrulanan kanonik örnek   : "
            f"{assignment_count:,}"
        )
        print(
            "Global tekrar parmak izi   : 0"
        )
        print(
            "Splitler arası örtüşme     : 0"
        )
        print(
            "Metadata-SQLite eşleşmesi  : True"
        )
        print(
            "Parquet-metadata eşleşmesi : True"
        )
        print(
            "Sınıf etiketleri eşleşiyor : True"
        )
        print(
            "Split atamaları geçerli    : True"
        )
        print(
            "Tüm değerler sonlu         : True"
        )
        print(
            "Doğrulama geçti            : True"
        )
        print()
        print(f"Split raporu : {split_report_file}")
        print(f"JSON özet    : {summary_file}")
        print("=" * 78)

    finally:
        connection.close()


if __name__ == "__main__":
    main()