"""
N-BaIoT float32 kanonik grup indeksini oluşturur.

Bu betik:

- Mevcut float64 Parquet dosyalarını parça parça okur.
- Özellikleri float32 biçimine dönüştürür.
- -0.0 ve +0.0 değerlerini aynı gösterime getirir.
- Her float32 özellik vektörü için iki adet 64-bit parmak izi üretir.
- Aynı float32 vektörleri SQLite üzerinde tek grupta toplar.
- Her grubun sınıf, eski split ve oluşum bilgilerini saklar.
- Birden fazla sınıfa ait grupları çelişkili olarak belirler.
- Her çelişkisiz grup için deterministik bir temsilci konumu seçer.
- Önceki float32 denetim raporuyla sonuçları karşılaştırır.

Bu aşamada yeni train/validation/test Parquet dosyaları oluşturulmaz.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from collections import defaultdict
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
    / "nbaiot_primary_seed2026"
)

DEFAULT_SCHEMA_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_reference_schema.csv"
)

DEFAULT_DISTRIBUTION_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_primary_split_class_distribution.csv"
)

DEFAULT_AUDIT_SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_float32_collision_summary.json"
)

DEFAULT_DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "nbaiot_float32_canonical.sqlite"
)

DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)


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

REPRESENTATIVE_SHIFT = 40


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT float32 kanonik grup "
            "SQLite indeksini oluşturur."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Float64 Parquet veri klasörü.",
    )

    parser.add_argument(
        "--schema-file",
        type=Path,
        default=DEFAULT_SCHEMA_FILE,
        help="Referans özellik şeması.",
    )

    parser.add_argument(
        "--distribution-file",
        type=Path,
        default=DEFAULT_DISTRIBUTION_FILE,
        help="Sınıf ve split dağılım raporu.",
    )

    parser.add_argument(
        "--audit-summary-file",
        type=Path,
        default=DEFAULT_AUDIT_SUMMARY_FILE,
        help="10 numaralı float32 denetim özeti.",
    )

    parser.add_argument(
        "--database-file",
        type=Path,
        default=DEFAULT_DATABASE_FILE,
        help="Oluşturulacak SQLite veritabanı.",
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Rapor klasörü.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Parquet okuma parça büyüklüğü.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Mevcut veritabanını silerek yeniden oluşturur.",
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
    """JSON raporunu yarım dosya bırakmadan yazar."""

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
# INPUT LOADERS
# ==========================================================

def load_feature_columns(
    schema_file: Path,
) -> list[str]:
    """Referans özellik sütunlarını yükler."""

    if not schema_file.exists():
        raise FileNotFoundError(
            f"Şema dosyası bulunamadı: {schema_file}"
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
            "Şemada tekrarlanan sütun adı var."
        )

    return columns


def load_classes_and_split_counts(
    distribution_file: Path,
) -> tuple[list[str], dict[str, int]]:
    """Sınıf isimlerini ve split büyüklüklerini yükler."""

    if not distribution_file.exists():
        raise FileNotFoundError(
            "Dağılım dosyası bulunamadı: "
            f"{distribution_file}"
        )

    distribution = pd.read_csv(
        distribution_file
    )

    required_columns = {
        "class_label",
        "split",
        "unique_vector_count",
    }

    missing_columns = (
        required_columns
        - set(distribution.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dağılım dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing_columns))
        )

    distribution["class_label"] = (
        distribution["class_label"]
        .astype(str)
    )

    distribution["split"] = (
        distribution["split"]
        .astype(str)
    )

    distribution["unique_vector_count"] = (
        pd.to_numeric(
            distribution["unique_vector_count"],
            errors="raise",
        )
        .astype("int64")
    )

    invalid_splits = (
        set(distribution["split"])
        - set(SPLIT_NAMES)
    )

    if invalid_splits:
        raise ValueError(
            "Geçersiz split adları var: "
            + ", ".join(sorted(invalid_splits))
        )

    classes = sorted(
        distribution[
            "class_label"
        ].unique().tolist()
    )

    split_counts = (
        distribution.groupby(
            "split"
        )["unique_vector_count"]
        .sum()
        .astype("int64")
        .to_dict()
    )

    return (
        classes,
        {
            str(split_name): int(count)
            for split_name, count
            in split_counts.items()
        },
    )


def load_audit_summary(
    audit_summary_file: Path,
) -> dict[str, Any]:
    """10 numaralı denetim özetini yükler."""

    if not audit_summary_file.exists():
        raise FileNotFoundError(
            "Float32 denetim özeti bulunamadı: "
            f"{audit_summary_file}"
        )

    with audit_summary_file.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        return json.load(
            file_handle
        )


def load_metadata_arrays(
    metadata_file: Path,
    classes: list[str],
    expected_split_counts: dict[str, int],
    batch_size: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """
    Metadata içindeki sınıf ve ham oluşum sayılarını split sırasıyla yükler.

    Metadata split bazında süzüldüğünde satır sırası, ilgili Parquet
    dosyasının satır sırasıyla aynıdır. Bu eşleşme işlem sırasında
    ayrıca doğrulanacaktır.
    """

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata bulunamadı: {metadata_file}"
        )

    label_to_code = {
        class_label: class_code
        for class_code, class_label in enumerate(
            classes
        )
    }

    label_parts: dict[
        str,
        list[np.ndarray],
    ] = {
        split_name: []
        for split_name in SPLIT_NAMES
    }

    raw_occurrence_parts: dict[
        str,
        list[np.ndarray],
    ] = {
        split_name: []
        for split_name in SPLIT_NAMES
    }

    parquet_file = pq.ParquetFile(
        metadata_file
    )

    total_rows = int(
        parquet_file.metadata.num_rows
    )

    progress = tqdm(
        total=total_rows,
        desc="Metadata dizileri hazırlanıyor",
        unit="satır",
    )

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=[
            "split",
            "class_label",
            "original_occurrence_count",
        ],
    ):
        frame = batch.to_pandas()

        split_series = frame[
            "split"
        ].astype(str)

        label_series = frame[
            "class_label"
        ].astype(str)

        occurrence_values = pd.to_numeric(
            frame[
                "original_occurrence_count"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.int64
        )

        encoded_labels = label_series.map(
            label_to_code
        )

        if encoded_labels.isna().any():
            unknown_labels = sorted(
                label_series[
                    encoded_labels.isna()
                ].unique().tolist()
            )

            raise ValueError(
                "Metadata içinde bilinmeyen sınıflar var: "
                + ", ".join(unknown_labels)
            )

        encoded_values = (
            encoded_labels.to_numpy(
                dtype=np.int16
            )
        )

        split_values = (
            split_series.to_numpy()
        )

        for split_name in SPLIT_NAMES:
            mask = (
                split_values == split_name
            )

            if not np.any(mask):
                continue

            label_parts[
                split_name
            ].append(
                encoded_values[mask].copy()
            )

            raw_occurrence_parts[
                split_name
            ].append(
                occurrence_values[mask].copy()
            )

        progress.update(
            len(frame)
        )

    progress.close()

    label_arrays: dict[
        str,
        np.ndarray,
    ] = {}

    raw_occurrence_arrays: dict[
        str,
        np.ndarray,
    ] = {}

    for split_name in SPLIT_NAMES:
        label_arrays[
            split_name
        ] = np.concatenate(
            label_parts[
                split_name
            ]
        )

        raw_occurrence_arrays[
            split_name
        ] = np.concatenate(
            raw_occurrence_parts[
                split_name
            ]
        )

        actual_count = int(
            len(
                label_arrays[
                    split_name
                ]
            )
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

    return (
        label_arrays,
        raw_occurrence_arrays,
    )


# ==========================================================
# FLOAT32 FINGERPRINTS
# ==========================================================

def create_float32_fingerprints(
    feature_frame: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """Float32 kanonik değerleri ve iki 64-bit parmak izini üretir."""

    values = feature_frame.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Float32 dönüşümünden sonra sonlu olmayan değer oluştu."
        )

    # -0.0 ile +0.0 aynı kanonik gösterime dönüştürülür.
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
        forward.view(np.int64),
        reverse.view(np.int64),
        float32_frame,
    )


# ==========================================================
# SQLITE
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


def create_database_schema(
    connection: sqlite3.Connection,
) -> None:
    """Kanonik grup tablolarını oluşturur."""

    connection.executescript(
        """
        CREATE TABLE run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE float32_groups (
            hash_forward INTEGER NOT NULL,
            hash_reverse INTEGER NOT NULL,

            class_code INTEGER NOT NULL,
            class_mask INTEGER NOT NULL,

            float64_unique_occurrence_count INTEGER NOT NULL,
            raw_occurrence_count INTEGER NOT NULL,

            old_split_mask INTEGER NOT NULL,
            representative_location INTEGER NOT NULL,

            PRIMARY KEY (
                hash_forward,
                hash_reverse
            )
        ) WITHOUT ROWID;
        """
    )

    connection.commit()


def insert_run_metadata(
    connection: sqlite3.Connection,
    classes: list[str],
) -> None:
    """Sınıf sıralamasını veritabanına kaydeder."""

    records = [
        (
            "classes_json",
            json.dumps(
                classes,
                ensure_ascii=False,
            ),
        ),
        (
            "representative_location_encoding",
            (
                "packed=(old_split_code << 40) | "
                "one_based_split_row_number"
            ),
        ),
        (
            "fingerprint_method",
            (
                "Two pandas 64-bit hashes from normal and "
                "reversed float32 feature-column order"
            ),
        ),
    ]

    connection.executemany(
        """
        INSERT INTO run_metadata (
            key,
            value
        )
        VALUES (?, ?)
        """,
        records,
    )

    connection.commit()


UPSERT_QUERY = """
    INSERT INTO float32_groups (
        hash_forward,
        hash_reverse,
        class_code,
        class_mask,
        float64_unique_occurrence_count,
        raw_occurrence_count,
        old_split_mask,
        representative_location
    )
    VALUES (?, ?, ?, ?, 1, ?, ?, ?)

    ON CONFLICT (
        hash_forward,
        hash_reverse
    )
    DO UPDATE SET

        class_code =
            CASE
                WHEN excluded.class_code
                     < float32_groups.class_code
                THEN excluded.class_code
                ELSE float32_groups.class_code
            END,

        class_mask =
            float32_groups.class_mask
            | excluded.class_mask,

        float64_unique_occurrence_count =
            float32_groups.float64_unique_occurrence_count
            + 1,

        raw_occurrence_count =
            float32_groups.raw_occurrence_count
            + excluded.raw_occurrence_count,

        old_split_mask =
            float32_groups.old_split_mask
            | excluded.old_split_mask,

        representative_location =
            CASE
                WHEN excluded.representative_location
                     < float32_groups.representative_location
                THEN excluded.representative_location
                ELSE float32_groups.representative_location
            END
"""


def scalar_query(
    connection: sqlite3.Connection,
    query: str,
) -> int:
    """Tek sayısal SQL sonucu döndürür."""

    result = connection.execute(
        query
    ).fetchone()

    if result is None or result[0] is None:
        return 0

    return int(result[0])


def decode_mask(
    mask: int,
    names: list[str] | tuple[str, ...],
) -> list[str]:
    """Bit maskesini isim listesine dönüştürür."""

    return [
        str(name)
        for index, name in enumerate(names)
        if mask & (1 << index)
    ]


def remove_sqlite_files(
    database_file: Path,
) -> None:
    """SQLite ana, WAL ve SHM dosyalarını siler."""

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

    data_dir = args.data_dir.resolve()
    schema_file = args.schema_file.resolve()
    distribution_file = (
        args.distribution_file.resolve()
    )
    audit_summary_file = (
        args.audit_summary_file.resolve()
    )
    database_file = args.database_file.resolve()
    report_dir = args.report_dir.resolve()

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size sıfırdan büyük olmalıdır."
        )

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Veri klasörü bulunamadı: {data_dir}"
        )

    feature_columns = load_feature_columns(
        schema_file
    )

    (
        classes,
        expected_split_counts,
    ) = load_classes_and_split_counts(
        distribution_file
    )

    audit_summary = load_audit_summary(
        audit_summary_file
    )

    expected_total_rows = int(
        audit_summary[
            "total_input_rows"
        ]
    )

    expected_unique_float32_groups = int(
        audit_summary[
            "unique_float32_vector_count"
        ]
    )

    expected_cross_split_groups = int(
        audit_summary[
            "cross_split_collision_group_count"
        ]
    )

    expected_cross_class_groups = int(
        audit_summary[
            "cross_class_conflict_group_count"
        ]
    )

    metadata_file = (
        data_dir
        / "metadata.parquet"
    )

    (
        metadata_label_arrays,
        metadata_raw_occurrence_arrays,
    ) = load_metadata_arrays(
        metadata_file=metadata_file,
        classes=classes,
        expected_split_counts=(
            expected_split_counts
        ),
        batch_size=args.batch_size,
    )

    database_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_database_file = (
        database_file.with_suffix(
            ".tmp.sqlite"
        )
    )

    if database_file.exists():
        if not args.overwrite:
            raise FileExistsError(
                "Kanonik SQLite veritabanı zaten mevcut. "
                "--overwrite kullan: "
                f"{database_file}"
            )

        remove_sqlite_files(
            database_file
        )

    remove_sqlite_files(
        temporary_database_file
    )

    free_space_gb = (
        shutil.disk_usage(
            database_file.parent
        ).free
        / (1024**3)
    )

    print("=" * 78)
    print("N-BaIoT Float32 Kanonik Grup İndeksi")
    print("=" * 78)
    print(f"Veri klasörü        : {data_dir}")
    print(f"Geçici veritabanı   : {temporary_database_file}")
    print(f"Nihai veritabanı    : {database_file}")
    print(f"Özellik sayısı      : {len(feature_columns)}")
    print(f"Sınıf sayısı        : {len(classes)}")
    print(f"Beklenen giriş      : {expected_total_rows:,}")
    print(
        "Beklenen float32 grup: "
        f"{expected_unique_float32_groups:,}"
    )
    print(f"Batch boyutu        : {args.batch_size:,}")
    print(f"Boş disk alanı      : {free_space_gb:.2f} GB")
    print("=" * 78)

    if free_space_gb < 2:
        print(
            "UYARI: Boş disk alanı 2 GB altında."
        )

    connection = sqlite3.connect(
        temporary_database_file,
        timeout=180,
    )

    total_processed_rows = 0

    try:
        configure_database(
            connection
        )

        create_database_schema(
            connection
        )

        insert_run_metadata(
            connection=connection,
            classes=classes,
        )

        label_to_code = {
            class_label: class_code
            for class_code, class_label in enumerate(
                classes
            )
        }

        for split_name in SPLIT_NAMES:
            split_code = SPLIT_TO_CODE[
                split_name
            ]

            parquet_path = (
                data_dir
                / f"{split_name}.parquet"
            )

            if not parquet_path.exists():
                raise FileNotFoundError(
                    f"Parquet bulunamadı: {parquet_path}"
                )

            parquet_file = pq.ParquetFile(
                parquet_path
            )

            expected_columns = (
                feature_columns
                + ["class_label"]
            )

            if parquet_file.schema_arrow.names != expected_columns:
                raise ValueError(
                    f"{split_name} Parquet şeması uyuşmuyor."
                )

            for column_name in feature_columns:
                arrow_type = (
                    parquet_file.schema_arrow.field(
                        column_name
                    ).type
                )

                if arrow_type != pa.float64():
                    raise TypeError(
                        f"{split_name}/{column_name} "
                        f"float64 değil: {arrow_type}"
                    )

            expected_split_count = int(
                expected_split_counts[
                    split_name
                ]
            )

            actual_split_count = int(
                parquet_file.metadata.num_rows
            )

            if actual_split_count != expected_split_count:
                raise RuntimeError(
                    f"{split_name} satır sayısı uyuşmuyor: "
                    f"beklenen={expected_split_count:,}, "
                    f"bulunan={actual_split_count:,}"
                )

            metadata_labels = (
                metadata_label_arrays[
                    split_name
                ]
            )

            metadata_raw_counts = (
                metadata_raw_occurrence_arrays[
                    split_name
                ]
            )

            split_offset = 0

            progress = tqdm(
                total=actual_split_count,
                desc=f"{split_name} gruplanıyor",
                unit="satır",
            )

            for batch in parquet_file.iter_batches(
                batch_size=args.batch_size,
                columns=expected_columns,
            ):
                frame = batch.to_pandas()

                feature_frame = frame[
                    feature_columns
                ]

                (
                    hash_forward,
                    hash_reverse,
                    _,
                ) = create_float32_fingerprints(
                    feature_frame
                )

                labels = (
                    frame["class_label"]
                    .astype(str)
                )

                encoded_labels = labels.map(
                    label_to_code
                )

                if encoded_labels.isna().any():
                    unknown_labels = sorted(
                        labels[
                            encoded_labels.isna()
                        ].unique().tolist()
                    )

                    raise ValueError(
                        "Bilinmeyen sınıf etiketi bulundu: "
                        + ", ".join(unknown_labels)
                    )

                label_codes = (
                    encoded_labels.to_numpy(
                        dtype=np.int16
                    )
                )

                batch_length = int(
                    len(frame)
                )

                metadata_label_batch = (
                    metadata_labels[
                        split_offset:
                        split_offset + batch_length
                    ]
                )

                metadata_raw_batch = (
                    metadata_raw_counts[
                        split_offset:
                        split_offset + batch_length
                    ]
                )

                if not np.array_equal(
                    label_codes,
                    metadata_label_batch,
                ):
                    mismatch_count = int(
                        np.count_nonzero(
                            label_codes
                            != metadata_label_batch
                        )
                    )

                    raise RuntimeError(
                        f"{split_name} Parquet ve metadata "
                        f"etiketleri uyuşmuyor: {mismatch_count:,}"
                    )

                one_based_row_numbers = np.arange(
                    split_offset + 1,
                    split_offset + batch_length + 1,
                    dtype=np.int64,
                )

                representative_locations = (
                    (
                        np.int64(split_code)
                        << np.int64(
                            REPRESENTATIVE_SHIFT
                        )
                    )
                    | one_based_row_numbers
                )

                split_mask_value = (
                    1 << split_code
                )

                records = [
                    (
                        int(hash_forward[index]),
                        int(hash_reverse[index]),
                        int(label_codes[index]),
                        int(
                            1
                            << int(
                                label_codes[index]
                            )
                        ),
                        int(metadata_raw_batch[index]),
                        int(split_mask_value),
                        int(
                            representative_locations[
                                index
                            ]
                        ),
                    )
                    for index in range(
                        batch_length
                    )
                ]

                with connection:
                    connection.executemany(
                        UPSERT_QUERY,
                        records,
                    )

                split_offset += (
                    batch_length
                )

                total_processed_rows += (
                    batch_length
                )

                progress.update(
                    batch_length
                )

            progress.close()

            if split_offset != expected_split_count:
                raise RuntimeError(
                    f"{split_name} işlenen satır sayısı uyuşmuyor: "
                    f"beklenen={expected_split_count:,}, "
                    f"işlenen={split_offset:,}"
                )

        # --------------------------------------------------
        # Indexes
        # --------------------------------------------------

        print(
            "Kanonik grup indeksleri oluşturuluyor..."
        )

        connection.execute(
            """
            CREATE INDEX
            idx_float32_groups_representative
            ON float32_groups (
                representative_location
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX
            idx_float32_groups_class_mask
            ON float32_groups (
                class_mask
            )
            """
        )

        connection.commit()

        # --------------------------------------------------
        # Aggregate statistics
        # --------------------------------------------------

        total_group_count = scalar_query(
            connection,
            """
            SELECT COUNT(*)
            FROM float32_groups
            """,
        )

        total_float64_occurrences = scalar_query(
            connection,
            """
            SELECT SUM(
                float64_unique_occurrence_count
            )
            FROM float32_groups
            """,
        )

        total_raw_occurrences = scalar_query(
            connection,
            """
            SELECT SUM(
                raw_occurrence_count
            )
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

        conflict_float64_rows = scalar_query(
            connection,
            """
            SELECT SUM(
                float64_unique_occurrence_count
            )
            FROM float32_groups
            WHERE (
                class_mask
                & (class_mask - 1)
            ) != 0
            """,
        )

        conflict_raw_rows = scalar_query(
            connection,
            """
            SELECT SUM(
                raw_occurrence_count
            )
            FROM float32_groups
            WHERE (
                class_mask
                & (class_mask - 1)
            ) != 0
            """,
        )

        cross_split_group_count = scalar_query(
            connection,
            """
            SELECT COUNT(*)
            FROM float32_groups
            WHERE (
                old_split_mask
                & (old_split_mask - 1)
            ) != 0
            """,
        )

        canonical_group_count = int(
            total_group_count
            - conflict_group_count
        )

        conflict_free_input_rows = int(
            total_float64_occurrences
            - conflict_float64_rows
        )

        same_class_rows_merged = int(
            conflict_free_input_rows
            - canonical_group_count
        )

        if (
            total_processed_rows
            != expected_total_rows
        ):
            raise RuntimeError(
                "Toplam işlenen satır sayısı uyuşmuyor: "
                f"beklenen={expected_total_rows:,}, "
                f"işlenen={total_processed_rows:,}"
            )

        if (
            total_float64_occurrences
            != expected_total_rows
        ):
            raise RuntimeError(
                "SQLite oluşum toplamı uyuşmuyor: "
                f"beklenen={expected_total_rows:,}, "
                f"bulunan={total_float64_occurrences:,}"
            )

        if (
            total_group_count
            != expected_unique_float32_groups
        ):
            raise RuntimeError(
                "Float32 grup sayısı 10 numaralı raporla "
                "uyuşmuyor: "
                f"beklenen={expected_unique_float32_groups:,}, "
                f"bulunan={total_group_count:,}"
            )

        if (
            conflict_group_count
            != expected_cross_class_groups
        ):
            raise RuntimeError(
                "Sınıf çelişkisi sayısı 10 numaralı raporla "
                "uyuşmuyor: "
                f"beklenen={expected_cross_class_groups:,}, "
                f"bulunan={conflict_group_count:,}"
            )

        if (
            cross_split_group_count
            != expected_cross_split_groups
        ):
            raise RuntimeError(
                "Splitler arası grup sayısı 10 numaralı "
                "raporla uyuşmuyor: "
                f"beklenen={expected_cross_split_groups:,}, "
                f"bulunan={cross_split_group_count:,}"
            )

        # --------------------------------------------------
        # Conflict report
        # --------------------------------------------------

        conflict_rows = connection.execute(
            """
            SELECT
                hash_forward,
                hash_reverse,
                class_mask,
                old_split_mask,
                float64_unique_occurrence_count,
                raw_occurrence_count,
                representative_location
            FROM float32_groups
            WHERE (
                class_mask
                & (class_mask - 1)
            ) != 0
            ORDER BY
                float64_unique_occurrence_count DESC
            """
        ).fetchall()

        conflict_records: list[
            dict[str, Any]
        ] = []

        for row in conflict_rows:
            (
                hash_forward,
                hash_reverse,
                class_mask,
                old_split_mask,
                float64_count,
                raw_count,
                representative_location,
            ) = row

            representative_split_code = int(
                int(representative_location)
                >> REPRESENTATIVE_SHIFT
            )

            representative_row_number = int(
                int(representative_location)
                & (
                    (1 << REPRESENTATIVE_SHIFT)
                    - 1
                )
            )

            conflict_records.append(
                {
                    "hash_forward": int(
                        hash_forward
                    ),
                    "hash_reverse": int(
                        hash_reverse
                    ),
                    "class_labels": ", ".join(
                        decode_mask(
                            int(class_mask),
                            classes,
                        )
                    ),
                    "old_splits": ", ".join(
                        decode_mask(
                            int(old_split_mask),
                            SPLIT_NAMES,
                        )
                    ),
                    "float64_unique_row_count": int(
                        float64_count
                    ),
                    "raw_occurrence_count": int(
                        raw_count
                    ),
                    "representative_source_split": (
                        SPLIT_NAMES[
                            representative_split_code
                        ]
                    ),
                    "representative_source_row_number": (
                        representative_row_number
                    ),
                    "decision": (
                        "exclude_entire_float32_group"
                    ),
                }
            )

        conflict_report = pd.DataFrame(
            conflict_records
        )

        conflict_report_file = (
            report_dir
            / "nbaiot_float32_canonical_conflicts.csv"
        )

        summary_file = (
            report_dir
            / "nbaiot_float32_canonical_index_summary.json"
        )

        conflict_report.to_csv(
            conflict_report_file,
            index=False,
            encoding="utf-8",
        )

        summary: dict[str, Any] = {
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "source_data_directory": str(
                data_dir
            ),
            "database_file": str(
                database_file
            ),
            "feature_count": int(
                len(feature_columns)
            ),
            "class_count": int(
                len(classes)
            ),
            "classes": classes,
            "total_input_float64_unique_rows": int(
                total_float64_occurrences
            ),
            "total_input_raw_occurrences": int(
                total_raw_occurrences
            ),
            "float32_unique_group_count": int(
                total_group_count
            ),
            "cross_old_split_group_count": int(
                cross_split_group_count
            ),
            "class_conflict_group_count": int(
                conflict_group_count
            ),
            "class_conflict_float64_unique_rows": int(
                conflict_float64_rows
            ),
            "class_conflict_raw_occurrences": int(
                conflict_raw_rows
            ),
            "conflict_free_float64_input_rows": int(
                conflict_free_input_rows
            ),
            "same_class_float32_rows_merged": int(
                same_class_rows_merged
            ),
            "canonical_conflict_free_group_count": int(
                canonical_group_count
            ),
            "conflict_policy": (
                "Every float32 group containing more than one "
                "class label is excluded in full."
            ),
            "representative_policy": (
                "Lowest old split code, then lowest one-based "
                "row number within that split."
            ),
            "previous_audit_consistency": {
                "unique_group_count_matches": True,
                "cross_split_group_count_matches": True,
                "cross_class_group_count_matches": True,
            },
            "conflict_report": str(
                conflict_report_file
            ),
        }

        write_json_atomic(
            data=summary,
            output_file=summary_file,
        )

        # --------------------------------------------------
        # Finalize SQLite database
        # --------------------------------------------------

        connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        )

        connection.close()
        connection = None

        temporary_database_file.replace(
            database_file
        )

        print()
        print("=" * 78)
        print("Float32 kanonik grup indeksi tamamlandı")
        print("=" * 78)
        print(
            "Giriş float64 benzersiz satır : "
            f"{total_float64_occurrences:,}"
        )
        print(
            "Benzersiz float32 grup        : "
            f"{total_group_count:,}"
        )
        print(
            "Eski splitler arası grup      : "
            f"{cross_split_group_count:,}"
        )
        print(
            "Sınıf çelişkili grup          : "
            f"{conflict_group_count:,}"
        )
        print(
            "Çelişkili gruplardaki satır   : "
            f"{conflict_float64_rows:,}"
        )
        print(
            "Aynı sınıfta birleştirilen    : "
            f"{same_class_rows_merged:,}"
        )
        print(
            "Kullanılabilir kanonik grup   : "
            f"{canonical_group_count:,}"
        )
        print()
        print(f"SQLite indeks : {database_file}")
        print(f"Çelişki raporu: {conflict_report_file}")
        print(f"JSON özet     : {summary_file}")
        print("=" * 78)

    except Exception:
        if connection is not None:
            connection.close()

        remove_sqlite_files(
            temporary_database_file
        )

        raise


if __name__ == "__main__":
    main()