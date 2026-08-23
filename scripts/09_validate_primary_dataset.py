"""
N-BaIoT işlenmiş ana deney veri setini doğrular.

Kontroller:
- Parquet şeması 115 float64 özellik ve class_label içeriyor mu?
- Train, validation ve test kayıt sayıları doğru mu?
- Metadata kayıtları doğru split'e atanmış mı?
- Metadata içinde tekrar parmak izi var mı?
- Parquet özelliklerinden yeniden hesaplanan parmak izleri metadata ile
  birebir uyuşuyor mu?
- Sınıf etiketleri metadata ile birebir uyuşuyor mu?
- Eksik veya sonsuz özellik değeri var mı?
- Sınıf dağılımları 07 numaralı split raporuyla uyuşuyor mu?

Bütün Parquet dosyaları parça parça okunur.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


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

DEFAULT_EXPECTED_DISTRIBUTION_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_primary_split_class_distribution.csv"
)

DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

UINT64_RANGE = 1 << 64
UINT64_MASK = UINT64_RANGE - 1

SPLIT_TO_CODE = {
    "train": 0,
    "validation": 1,
    "test": 2,
}

CODE_TO_SPLIT = {
    value: key
    for key, value in SPLIT_TO_CODE.items()
}


def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT işlenmiş ana deney "
            "Parquet veri setini doğrular."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--schema-file",
        type=Path,
        default=DEFAULT_SCHEMA_FILE,
    )

    parser.add_argument(
        "--expected-distribution-file",
        type=Path,
        default=DEFAULT_EXPECTED_DISTRIBUTION_FILE,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
    )

    return parser.parse_args()


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
    """JSON raporunu güvenli biçimde kaydeder."""

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


def load_feature_columns(
    schema_file: Path,
) -> list[str]:
    """115 özellik sütununun adlarını yükler."""

    if not schema_file.exists():
        raise FileNotFoundError(
            f"Şema dosyası bulunamadı: {schema_file}"
        )

    schema_frame = pd.read_csv(
        schema_file
    )

    if "column_name" not in schema_frame.columns:
        raise ValueError(
            "Şema dosyasında 'column_name' sütunu yok."
        )

    columns = (
        schema_frame["column_name"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if not columns:
        raise ValueError(
            "Özellik sütunu listesi boş."
        )

    if len(columns) != len(set(columns)):
        raise ValueError(
            "Şema dosyasında tekrarlanan sütun adı var."
        )

    return columns


def load_expected_distribution(
    distribution_file: Path,
) -> tuple[
    dict[str, int],
    dict[tuple[str, str], int],
    list[str],
]:
    """07 numaralı betiğin beklenen sınıf dağılımını yükler."""

    if not distribution_file.exists():
        raise FileNotFoundError(
            "Beklenen dağılım raporu bulunamadı: "
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
            "Dağılım raporunda eksik sütunlar var: "
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
        - set(SPLIT_TO_CODE)
    )

    if invalid_splits:
        raise ValueError(
            "Geçersiz split adları: "
            + ", ".join(sorted(invalid_splits))
        )

    split_counts = (
        distribution.groupby(
            "split"
        )["unique_vector_count"]
        .sum()
        .astype("int64")
        .to_dict()
    )

    class_split_counts = {
        (
            str(row.class_label),
            str(row.split),
        ): int(row.unique_vector_count)
        for row in distribution.itertuples(
            index=False
        )
    }

    classes = sorted(
        distribution["class_label"]
        .unique()
        .tolist()
    )

    return (
        {
            str(key): int(value)
            for key, value in split_counts.items()
        },
        class_split_counts,
        classes,
    )


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
    """Parmak izlerinden deterministik split kodlarını hesaplar."""

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


def create_row_fingerprints(
    feature_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Özellik satırlarının iki adet 64-bit özetini hesaplar."""

    forward = pd.util.hash_pandas_object(
        feature_frame,
        index=False,
        categorize=False,
    ).to_numpy(
        dtype=np.uint64
    )

    reverse = pd.util.hash_pandas_object(
        feature_frame.iloc[:, ::-1],
        index=False,
        categorize=False,
    ).to_numpy(
        dtype=np.uint64
    )

    return (
        forward.view(np.int64),
        reverse.view(np.int64),
    )


def encode_labels(
    labels: pd.Series,
    label_to_code: dict[str, int],
) -> np.ndarray:
    """Sınıf etiketlerini sayısal kodlara dönüştürür."""

    normalized = labels.astype(str)

    encoded = normalized.map(
        label_to_code
    )

    if encoded.isna().any():
        unknown_labels = sorted(
            normalized[
                encoded.isna()
            ].unique().tolist()
        )

        raise ValueError(
            "Bilinmeyen sınıf etiketleri bulundu: "
            + ", ".join(unknown_labels)
        )

    return encoded.to_numpy(
        dtype=np.int16
    )


def validate_metadata(
    metadata_file: Path,
    expected_split_counts: dict[str, int],
    expected_class_split_counts: dict[
        tuple[str, str],
        int,
    ],
    classes: list[str],
    seed: int,
    batch_size: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    """
    Metadata dosyasını doğrular ve split bazlı beklenen dizileri döndürür.
    """

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata dosyası bulunamadı: {metadata_file}"
        )

    parquet_file = pq.ParquetFile(
        metadata_file
    )

    required_columns = [
        "hash_forward",
        "hash_reverse",
        "split",
        "class_label",
        "source_device",
        "source_relative_path",
        "source_row_number",
        "original_occurrence_count",
        "source_device_count",
    ]

    if parquet_file.schema_arrow.names != required_columns:
        raise ValueError(
            "Metadata Parquet şeması beklenen şemayla uyuşmuyor."
        )

    label_to_code = {
        label: index
        for index, label in enumerate(classes)
    }

    hash_forward_parts: dict[
        str,
        list[np.ndarray],
    ] = {
        split: []
        for split in SPLIT_TO_CODE
    }

    hash_reverse_parts: dict[
        str,
        list[np.ndarray],
    ] = {
        split: []
        for split in SPLIT_TO_CODE
    }

    label_code_parts: dict[
        str,
        list[np.ndarray],
    ] = {
        split: []
        for split in SPLIT_TO_CODE
    }

    actual_class_split_counts: Counter[
        tuple[str, str]
    ] = Counter()

    total_metadata_rows = int(
        parquet_file.metadata.num_rows
    )

    progress = tqdm(
        total=total_metadata_rows,
        desc="Metadata doğrulanıyor",
        unit="satır",
    )

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=[
            "hash_forward",
            "hash_reverse",
            "split",
            "class_label",
        ],
    ):
        frame = batch.to_pandas()

        if frame.isna().any().any():
            raise ValueError(
                "Metadata içinde eksik değer bulundu."
            )

        hash_forward = frame[
            "hash_forward"
        ].to_numpy(
            dtype=np.int64,
            copy=False,
        )

        hash_reverse = frame[
            "hash_reverse"
        ].to_numpy(
            dtype=np.int64,
            copy=False,
        )

        split_names = frame[
            "split"
        ].astype(str)

        class_labels = frame[
            "class_label"
        ].astype(str)

        unknown_splits = (
            set(split_names.unique())
            - set(SPLIT_TO_CODE)
        )

        if unknown_splits:
            raise ValueError(
                "Metadata içinde geçersiz split adı var: "
                + ", ".join(sorted(unknown_splits))
            )

        calculated_codes = (
            calculate_split_codes(
                hash_forward=hash_forward,
                hash_reverse=hash_reverse,
                seed=seed,
            )
        )

        recorded_codes = split_names.map(
            SPLIT_TO_CODE
        ).to_numpy(
            dtype=np.uint8
        )

        if not np.array_equal(
            calculated_codes,
            recorded_codes,
        ):
            mismatch_count = int(
                np.count_nonzero(
                    calculated_codes
                    != recorded_codes
                )
            )

            raise RuntimeError(
                "Metadata split ataması parmak iziyle "
                f"uyuşmuyor. Hatalı satır={mismatch_count:,}"
            )

        label_codes = encode_labels(
            class_labels,
            label_to_code,
        )

        for split_name in SPLIT_TO_CODE:
            mask = (
                split_names.to_numpy()
                == split_name
            )

            if not np.any(mask):
                continue

            hash_forward_parts[
                split_name
            ].append(
                hash_forward[mask].copy()
            )

            hash_reverse_parts[
                split_name
            ].append(
                hash_reverse[mask].copy()
            )

            label_code_parts[
                split_name
            ].append(
                label_codes[mask].copy()
            )

            labels_in_split = (
                class_labels[
                    mask
                ]
                .value_counts()
            )

            for class_label, count in (
                labels_in_split.items()
            ):
                actual_class_split_counts[
                    (
                        str(class_label),
                        split_name,
                    )
                ] += int(count)

        progress.update(
            len(frame)
        )

    progress.close()

    expected_hash_forward: dict[
        str,
        np.ndarray,
    ] = {}

    expected_hash_reverse: dict[
        str,
        np.ndarray,
    ] = {}

    expected_label_codes: dict[
        str,
        np.ndarray,
    ] = {}

    split_records: list[
        dict[str, Any]
    ] = []

    for split_name in SPLIT_TO_CODE:
        expected_hash_forward[
            split_name
        ] = np.concatenate(
            hash_forward_parts[
                split_name
            ]
        )

        expected_hash_reverse[
            split_name
        ] = np.concatenate(
            hash_reverse_parts[
                split_name
            ]
        )

        expected_label_codes[
            split_name
        ] = np.concatenate(
            label_code_parts[
                split_name
            ]
        )

        actual_count = len(
            expected_hash_forward[
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

        forward = expected_hash_forward[
            split_name
        ]

        reverse = expected_hash_reverse[
            split_name
        ]

        order = np.lexsort(
            (
                reverse,
                forward,
            )
        )

        sorted_forward = forward[
            order
        ]

        sorted_reverse = reverse[
            order
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

        duplicate_count = int(
            duplicate_mask.sum()
        )

        if duplicate_count > 0:
            raise RuntimeError(
                f"Metadata {split_name} içinde "
                f"{duplicate_count:,} tekrar parmak izi var."
            )

        split_records.append(
            {
                "split": split_name,
                "metadata_row_count": actual_count,
                "duplicate_fingerprint_count": (
                    duplicate_count
                ),
                "split_assignment_valid": True,
            }
        )

    for key, expected_count in (
        expected_class_split_counts.items()
    ):
        actual_count = int(
            actual_class_split_counts[key]
        )

        if actual_count != expected_count:
            raise RuntimeError(
                "Metadata sınıf dağılımı uyuşmuyor: "
                f"class={key[0]}, split={key[1]}, "
                f"beklenen={expected_count:,}, "
                f"bulunan={actual_count:,}"
            )

    return (
        expected_hash_forward,
        expected_hash_reverse,
        expected_label_codes,
        split_records,
    )


def validate_split_file(
    split_name: str,
    parquet_file_path: Path,
    feature_columns: list[str],
    classes: list[str],
    expected_hash_forward: np.ndarray,
    expected_hash_reverse: np.ndarray,
    expected_label_codes: np.ndarray,
    expected_class_split_counts: dict[
        tuple[str, str],
        int,
    ],
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    """Tek bir train/validation/test Parquet dosyasını doğrular."""

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

    arrow_schema = parquet_file.schema_arrow

    if arrow_schema.names != expected_columns:
        raise ValueError(
            f"{split_name} sütun şeması uyuşmuyor."
        )

    for column_name in feature_columns:
        if arrow_schema.field(
            column_name
        ).type != pa.float64():
            raise TypeError(
                f"{split_name}/{column_name} float64 değil: "
                f"{arrow_schema.field(column_name).type}"
            )

    if arrow_schema.field(
        "class_label"
    ).type != pa.string():
        raise TypeError(
            f"{split_name}/class_label string değil."
        )

    expected_row_count = len(
        expected_hash_forward
    )

    parquet_row_count = int(
        parquet_file.metadata.num_rows
    )

    if parquet_row_count != expected_row_count:
        raise RuntimeError(
            f"{split_name} Parquet satır sayısı uyuşmuyor: "
            f"beklenen={expected_row_count:,}, "
            f"bulunan={parquet_row_count:,}"
        )

    label_to_code = {
        label: index
        for index, label in enumerate(classes)
    }

    class_counts: Counter[str] = Counter()

    missing_value_count = 0
    infinity_value_count = 0
    offset = 0

    progress = tqdm(
        total=parquet_row_count,
        desc=f"{split_name} doğrulanıyor",
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
            dtype=np.float64,
            copy=False,
        )

        current_missing = int(
            np.isnan(values).sum()
        )

        current_infinity = int(
            np.isinf(values).sum()
        )

        missing_value_count += (
            current_missing
        )

        infinity_value_count += (
            current_infinity
        )

        if current_missing > 0:
            raise RuntimeError(
                f"{split_name} içinde eksik değer bulundu."
            )

        if current_infinity > 0:
            raise RuntimeError(
                f"{split_name} içinde sonsuz değer bulundu."
            )

        (
            actual_forward,
            actual_reverse,
        ) = create_row_fingerprints(
            feature_frame
        )

        batch_length = len(frame)

        expected_forward_batch = (
            expected_hash_forward[
                offset:offset + batch_length
            ]
        )

        expected_reverse_batch = (
            expected_hash_reverse[
                offset:offset + batch_length
            ]
        )

        if not np.array_equal(
            actual_forward,
            expected_forward_batch,
        ):
            mismatch_count = int(
                np.count_nonzero(
                    actual_forward
                    != expected_forward_batch
                )
            )

            raise RuntimeError(
                f"{split_name} ileri parmak izleri "
                f"metadata ile uyuşmuyor: {mismatch_count:,}"
            )

        if not np.array_equal(
            actual_reverse,
            expected_reverse_batch,
        ):
            mismatch_count = int(
                np.count_nonzero(
                    actual_reverse
                    != expected_reverse_batch
                )
            )

            raise RuntimeError(
                f"{split_name} ters parmak izleri "
                f"metadata ile uyuşmuyor: {mismatch_count:,}"
            )

        calculated_split_codes = (
            calculate_split_codes(
                hash_forward=actual_forward,
                hash_reverse=actual_reverse,
                seed=seed,
            )
        )

        expected_split_code = (
            SPLIT_TO_CODE[
                split_name
            ]
        )

        if np.any(
            calculated_split_codes
            != expected_split_code
        ):
            mismatch_count = int(
                np.count_nonzero(
                    calculated_split_codes
                    != expected_split_code
                )
            )

            raise RuntimeError(
                f"{split_name} dosyasında yanlış split'e ait "
                f"{mismatch_count:,} kayıt var."
            )

        actual_label_codes = encode_labels(
            frame["class_label"],
            label_to_code,
        )

        expected_label_batch = (
            expected_label_codes[
                offset:offset + batch_length
            ]
        )

        if not np.array_equal(
            actual_label_codes,
            expected_label_batch,
        ):
            mismatch_count = int(
                np.count_nonzero(
                    actual_label_codes
                    != expected_label_batch
                )
            )

            raise RuntimeError(
                f"{split_name} sınıf etiketleri metadata ile "
                f"uyuşmuyor: {mismatch_count:,}"
            )

        value_counts = (
            frame["class_label"]
            .astype(str)
            .value_counts()
        )

        for class_label, count in (
            value_counts.items()
        ):
            class_counts[
                str(class_label)
            ] += int(count)

        offset += batch_length

        progress.update(
            batch_length
        )

    progress.close()

    if offset != expected_row_count:
        raise RuntimeError(
            f"{split_name} işlenen satır sayısı uyuşmuyor: "
            f"beklenen={expected_row_count:,}, "
            f"işlenen={offset:,}"
        )

    for class_label in classes:
        expected_count = int(
            expected_class_split_counts[
                (
                    class_label,
                    split_name,
                )
            ]
        )

        actual_count = int(
            class_counts[
                class_label
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
        "row_count": int(offset),
        "row_group_count": int(
            parquet_file.metadata.num_row_groups
        ),
        "column_count": int(
            len(expected_columns)
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "missing_value_count": int(
            missing_value_count
        ),
        "infinity_value_count": int(
            infinity_value_count
        ),
        "fingerprints_match_metadata": True,
        "labels_match_metadata": True,
        "split_assignment_valid": True,
        "file_size_bytes": int(
            parquet_file_path.stat().st_size
        ),
        "file_size_gb": float(
            parquet_file_path.stat().st_size
            / (1024**3)
        ),
    }


def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    data_dir = args.data_dir.resolve()
    schema_file = args.schema_file.resolve()
    expected_distribution_file = (
        args.expected_distribution_file.resolve()
    )
    report_dir = args.report_dir.resolve()

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size sıfırdan büyük olmalıdır."
        )

    if not data_dir.exists():
        raise FileNotFoundError(
            f"İşlenmiş veri klasörü bulunamadı: {data_dir}"
        )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_columns = load_feature_columns(
        schema_file
    )

    (
        expected_split_counts,
        expected_class_split_counts,
        classes,
    ) = load_expected_distribution(
        expected_distribution_file
    )

    metadata_file = (
        data_dir
        / "metadata.parquet"
    )

    print("=" * 78)
    print("N-BaIoT Ana Deney Veri Seti Doğrulaması")
    print("=" * 78)
    print(f"Veri klasörü   : {data_dir}")
    print(f"Metadata       : {metadata_file}")
    print(f"Özellik sayısı : {len(feature_columns)}")
    print(f"Sınıf sayısı   : {len(classes)}")
    print(f"Seed           : {args.seed}")
    print(f"Batch boyutu   : {args.batch_size:,}")
    print("=" * 78)

    (
        expected_hash_forward,
        expected_hash_reverse,
        expected_label_codes,
        metadata_records,
    ) = validate_metadata(
        metadata_file=metadata_file,
        expected_split_counts=(
            expected_split_counts
        ),
        expected_class_split_counts=(
            expected_class_split_counts
        ),
        classes=classes,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    split_records: list[
        dict[str, Any]
    ] = []

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_records.append(
            validate_split_file(
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
                expected_label_codes=(
                    expected_label_codes[
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
        / "nbaiot_primary_validation_by_split.csv"
    )

    summary_file = (
        report_dir
        / "nbaiot_primary_validation_summary.json"
    )

    split_report.to_csv(
        split_report_file,
        index=False,
        encoding="utf-8",
    )

    total_rows = int(
        split_report["row_count"].sum()
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

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "data_directory": str(
            data_dir
        ),
        "seed": int(
            args.seed
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "class_count": int(
            len(classes)
        ),
        "classes": classes,
        "total_row_count": total_rows,
        "metadata_row_count": int(
            sum(
                record["metadata_row_count"]
                for record in metadata_records
            )
        ),
        "total_missing_values": (
            total_missing_values
        ),
        "total_infinity_values": (
            total_infinity_values
        ),
        "metadata_validation": (
            metadata_records
        ),
        "split_validation": (
            split_records
        ),
        "all_parquet_fingerprints_match_metadata": bool(
            all(
                record[
                    "fingerprints_match_metadata"
                ]
                for record in split_records
            )
        ),
        "all_labels_match_metadata": bool(
            all(
                record[
                    "labels_match_metadata"
                ]
                for record in split_records
            )
        ),
        "all_split_assignments_valid": bool(
            all(
                record[
                    "split_assignment_valid"
                ]
                for record in split_records
            )
        ),
        "all_metadata_fingerprints_unique": bool(
            all(
                record[
                    "duplicate_fingerprint_count"
                ] == 0
                for record in metadata_records
            )
        ),
        "all_values_finite": bool(
            total_missing_values == 0
            and total_infinity_values == 0
        ),
        "validation_passed": True,
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
    print("Ana deney veri seti doğrulaması tamamlandı")
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
        "Toplam satır              : "
        f"{total_rows:,}"
    )
    print(
        "Metadata tekrar parmak izi: 0"
    )
    print(
        "Parmak izleri eşleşiyor   : True"
    )
    print(
        "Etiketler eşleşiyor       : True"
    )
    print(
        "Split atamaları geçerli   : True"
    )
    print(
        "Tüm değerler sonlu        : "
        f"{summary['all_values_finite']}"
    )
    print()
    print(f"Split raporu : {split_report_file}")
    print(f"JSON özet    : {summary_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()