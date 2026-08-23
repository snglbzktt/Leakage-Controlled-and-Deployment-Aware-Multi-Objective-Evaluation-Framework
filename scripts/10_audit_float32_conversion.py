"""
N-BaIoT float64 -> float32 dönüşüm çakışma denetimi.

Kontroller:
- Float32 dönüşümünden sonra kaç benzersiz vektör kalıyor?
- Yeni duplicate vektör oluşuyor mu?
- Aynı float32 vektörü farklı splitlerde bulunuyor mu?
- Aynı float32 vektörü farklı sınıflarda bulunuyor mu?
- Split başına yeni duplicate oranı nedir?

Bu betik veri dosyalarını değiştirmez.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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

DEFAULT_DISTRIBUTION_FILE = (
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


def parse_arguments() -> argparse.Namespace:
    """Komut satırı seçeneklerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT float32 dönüşümünden "
            "kaynaklanabilecek çakışmaları denetler."
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
        "--distribution-file",
        type=Path,
        default=DEFAULT_DISTRIBUTION_FILE,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--maximum-reported-groups",
        type=int,
        default=500,
    )

    return parser.parse_args()


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
    """JSON raporunu yarım dosya bırakmadan kaydeder."""

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


def load_expected_information(
    distribution_file: Path,
) -> tuple[dict[str, int], list[str]]:
    """Beklenen split büyüklüklerini ve sınıfları yükler."""

    if not distribution_file.exists():
        raise FileNotFoundError(
            "Sınıf dağılım raporu bulunamadı: "
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
        - set(SPLIT_NAMES)
    )

    if invalid_splits:
        raise ValueError(
            "Geçersiz split adları var: "
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

    classes = sorted(
        distribution[
            "class_label"
        ].unique().tolist()
    )

    return (
        {
            str(split_name): int(count)
            for split_name, count
            in split_counts.items()
        },
        classes,
    )


def create_float32_fingerprints(
    feature_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Özellikleri float32'ye dönüştürür ve iki 64-bit özet üretir.
    """

    values = feature_frame.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Float32 dönüşümünden sonra sonlu olmayan değer oluştu."
        )

    # -0.0 ve +0.0 değerlerini aynı gösterime getir.
    values[values == 0] = np.float32(0.0)

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
    )


def count_unique_pairs(
    hash_forward: np.ndarray,
    hash_reverse: np.ndarray,
) -> int:
    """İki hash dizisindeki benzersiz çift sayısını hesaplar."""

    if len(hash_forward) == 0:
        return 0

    order = np.lexsort(
        (
            hash_reverse,
            hash_forward,
        )
    )

    sorted_forward = hash_forward[
        order
    ]

    sorted_reverse = hash_reverse[
        order
    ]

    change_mask = (
        (
            sorted_forward[1:]
            != sorted_forward[:-1]
        )
        |
        (
            sorted_reverse[1:]
            != sorted_reverse[:-1]
        )
    )

    return int(
        1 + np.count_nonzero(change_mask)
    )


def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    data_dir = args.data_dir.resolve()
    schema_file = args.schema_file.resolve()
    distribution_file = (
        args.distribution_file.resolve()
    )
    report_dir = args.report_dir.resolve()

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size sıfırdan büyük olmalıdır."
        )

    if args.maximum_reported_groups < 0:
        raise ValueError(
            "Raporlanacak grup sayısı negatif olamaz."
        )

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Veri klasörü bulunamadı: {data_dir}"
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
        classes,
    ) = load_expected_information(
        distribution_file
    )

    label_to_code = {
        class_label: class_code
        for class_code, class_label in enumerate(
            classes
        )
    }

    code_to_label = {
        class_code: class_label
        for class_label, class_code
        in label_to_code.items()
    }

    forward_parts: list[np.ndarray] = []
    reverse_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    row_number_parts: list[np.ndarray] = []

    total_rows = 0

    print("=" * 78)
    print("N-BaIoT Float32 Dönüşüm Çakışma Denetimi")
    print("=" * 78)
    print(f"Veri klasörü   : {data_dir}")
    print(f"Özellik sayısı : {len(feature_columns)}")
    print(f"Sınıf sayısı   : {len(classes)}")
    print(f"Batch boyutu   : {args.batch_size:,}")
    print("=" * 78)

    for split_name in SPLIT_NAMES:
        parquet_path = (
            data_dir
            / f"{split_name}.parquet"
        )

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Parquet dosyası bulunamadı: {parquet_path}"
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
            arrow_type = parquet_file.schema_arrow.field(
                column_name
            ).type

            if arrow_type != pa.float64():
                raise TypeError(
                    f"{split_name}/{column_name} float64 değil: "
                    f"{arrow_type}"
                )

        expected_row_count = int(
            expected_split_counts[
                split_name
            ]
        )

        actual_row_count = int(
            parquet_file.metadata.num_rows
        )

        if actual_row_count != expected_row_count:
            raise RuntimeError(
                f"{split_name} satır sayısı uyuşmuyor: "
                f"beklenen={expected_row_count:,}, "
                f"bulunan={actual_row_count:,}"
            )

        split_code = SPLIT_TO_CODE[
            split_name
        ]

        current_row_number = 0

        progress = tqdm(
            total=actual_row_count,
            desc=f"{split_name} float32 denetimi",
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

            batch_length = int(
                len(frame)
            )

            forward_parts.append(
                hash_forward
            )

            reverse_parts.append(
                hash_reverse
            )

            split_parts.append(
                np.full(
                    batch_length,
                    split_code,
                    dtype=np.uint8,
                )
            )

            label_parts.append(
                encoded_labels.to_numpy(
                    dtype=np.int16
                )
            )

            row_number_parts.append(
                np.arange(
                    current_row_number + 1,
                    current_row_number
                    + batch_length + 1,
                    dtype=np.int64,
                )
            )

            current_row_number += (
                batch_length
            )

            total_rows += (
                batch_length
            )

            progress.update(
                batch_length
            )

        progress.close()

        if current_row_number != expected_row_count:
            raise RuntimeError(
                f"{split_name} işlenen satır sayısı uyuşmuyor."
            )

    print("Hash dizileri birleştiriliyor...")

    hash_forward = np.concatenate(
        forward_parts
    )

    hash_reverse = np.concatenate(
        reverse_parts
    )

    split_codes = np.concatenate(
        split_parts
    )

    label_codes = np.concatenate(
        label_parts
    )

    row_numbers = np.concatenate(
        row_number_parts
    )

    del forward_parts
    del reverse_parts
    del split_parts
    del label_parts
    del row_number_parts

    if len(hash_forward) != total_rows:
        raise RuntimeError(
            "Birleştirilmiş hash sayısı toplam satırla uyuşmuyor."
        )

    print("Global float32 çakışma grupları hesaplanıyor...")

    order = np.lexsort(
        (
            hash_reverse,
            hash_forward,
        )
    )

    sorted_forward = hash_forward[
        order
    ]

    sorted_reverse = hash_reverse[
        order
    ]

    sorted_splits = split_codes[
        order
    ]

    sorted_labels = label_codes[
        order
    ]

    sorted_rows = row_numbers[
        order
    ]

    group_change = (
        (
            sorted_forward[1:]
            != sorted_forward[:-1]
        )
        |
        (
            sorted_reverse[1:]
            != sorted_reverse[:-1]
        )
    )

    group_starts = np.concatenate(
        (
            np.asarray(
                [0],
                dtype=np.int64,
            ),
            np.flatnonzero(
                group_change
            ).astype(np.int64)
            + 1,
        )
    )

    group_ends = np.concatenate(
        (
            group_starts[1:],
            np.asarray(
                [total_rows],
                dtype=np.int64,
            ),
        )
    )

    group_sizes = (
        group_ends
        - group_starts
    )

    unique_float32_vectors = int(
        len(group_starts)
    )

    duplicate_rows_beyond_first = int(
        total_rows
        - unique_float32_vectors
    )

    duplicate_group_mask = (
        group_sizes > 1
    )

    duplicate_group_indices = (
        np.flatnonzero(
            duplicate_group_mask
        )
    )

    split_minimum = np.minimum.reduceat(
        sorted_splits,
        group_starts,
    )

    split_maximum = np.maximum.reduceat(
        sorted_splits,
        group_starts,
    )

    label_minimum = np.minimum.reduceat(
        sorted_labels,
        group_starts,
    )

    label_maximum = np.maximum.reduceat(
        sorted_labels,
        group_starts,
    )

    cross_split_mask = (
        split_minimum
        != split_maximum
    )

    cross_class_mask = (
        label_minimum
        != label_maximum
    )

    cross_split_group_count = int(
        np.count_nonzero(
            cross_split_mask
        )
    )

    cross_class_group_count = int(
        np.count_nonzero(
            cross_class_mask
        )
    )

    cross_split_and_class_group_count = int(
        np.count_nonzero(
            cross_split_mask
            & cross_class_mask
        )
    )

    rows_in_cross_split_groups = int(
        group_sizes[
            cross_split_mask
        ].sum()
    )

    rows_in_cross_class_groups = int(
        group_sizes[
            cross_class_mask
        ].sum()
    )

    # ------------------------------------------------------
    # Split-level reports
    # ------------------------------------------------------

    split_records: list[
        dict[str, Any]
    ] = []

    for split_name in SPLIT_NAMES:
        split_code = SPLIT_TO_CODE[
            split_name
        ]

        split_mask = (
            split_codes == split_code
        )

        split_forward = hash_forward[
            split_mask
        ]

        split_reverse = hash_reverse[
            split_mask
        ]

        split_row_count = int(
            len(split_forward)
        )

        unique_count = count_unique_pairs(
            split_forward,
            split_reverse,
        )

        duplicate_count = int(
            split_row_count
            - unique_count
        )

        duplicate_percentage = (
            duplicate_count
            / split_row_count
            * 100
            if split_row_count > 0
            else 0.0
        )

        split_records.append(
            {
                "split": split_name,
                "original_float64_unique_rows": (
                    split_row_count
                ),
                "unique_float32_vectors": (
                    unique_count
                ),
                "float32_duplicate_rows_beyond_first": (
                    duplicate_count
                ),
                "float32_duplicate_percentage": (
                    duplicate_percentage
                ),
            }
        )

    split_report = pd.DataFrame(
        split_records
    )

    # ------------------------------------------------------
    # Largest collision candidate groups
    # ------------------------------------------------------

    collision_columns = [
        "hash_forward",
        "hash_reverse",
        "occurrence_count",
        "split_count",
        "class_count",
        "splits",
        "class_labels",
        "row_references",
    ]

    collision_records: list[
        dict[str, Any]
    ] = []

    if len(duplicate_group_indices) > 0:
        ranked_indices = (
            duplicate_group_indices[
                np.argsort(
                    group_sizes[
                        duplicate_group_indices
                    ]
                )[::-1]
            ]
        )

        ranked_indices = ranked_indices[
            :args.maximum_reported_groups
        ]

        for group_index in ranked_indices:
            start = int(
                group_starts[
                    group_index
                ]
            )

            end = int(
                group_ends[
                    group_index
                ]
            )

            group_split_codes = sorted(
                np.unique(
                    sorted_splits[
                        start:end
                    ]
                ).astype(int).tolist()
            )

            group_label_codes = sorted(
                np.unique(
                    sorted_labels[
                        start:end
                    ]
                ).astype(int).tolist()
            )

            split_names = [
                SPLIT_NAMES[
                    split_code
                ]
                for split_code
                in group_split_codes
            ]

            class_labels = [
                code_to_label[
                    label_code
                ]
                for label_code
                in group_label_codes
            ]

            reference_count = min(
                end - start,
                20,
            )

            references = []

            for local_index in range(
                start,
                start + reference_count,
            ):
                reference_split = SPLIT_NAMES[
                    int(
                        sorted_splits[
                            local_index
                        ]
                    )
                ]

                reference_row = int(
                    sorted_rows[
                        local_index
                    ]
                )

                references.append(
                    f"{reference_split}:{reference_row}"
                )

            collision_records.append(
                {
                    "hash_forward": int(
                        sorted_forward[start]
                    ),
                    "hash_reverse": int(
                        sorted_reverse[start]
                    ),
                    "occurrence_count": int(
                        end - start
                    ),
                    "split_count": int(
                        len(split_names)
                    ),
                    "class_count": int(
                        len(class_labels)
                    ),
                    "splits": ", ".join(
                        split_names
                    ),
                    "class_labels": ", ".join(
                        class_labels
                    ),
                    "row_references": "; ".join(
                        references
                    ),
                }
            )

    collision_report = pd.DataFrame(
        collision_records,
        columns=collision_columns,
    )

    split_report_file = (
        report_dir
        / "nbaiot_float32_collision_by_split.csv"
    )

    collision_report_file = (
        report_dir
        / "nbaiot_float32_collision_groups.csv"
    )

    summary_file = (
        report_dir
        / "nbaiot_float32_collision_summary.json"
    )

    split_report.to_csv(
        split_report_file,
        index=False,
        encoding="utf-8",
    )

    collision_report.to_csv(
        collision_report_file,
        index=False,
        encoding="utf-8",
    )

    no_new_collisions = bool(
        duplicate_rows_beyond_first == 0
    )

    no_cross_split_collisions = bool(
        cross_split_group_count == 0
    )

    no_cross_class_conflicts = bool(
        cross_class_group_count == 0
    )

    if no_new_collisions:
        recommendation = (
            "Float32 dönüşümü yeni duplicate üretmedi. "
            "Mevcut splitler korunarak float32 veri üretilebilir."
        )

    elif (
        no_cross_split_collisions
        and no_cross_class_conflicts
    ):
        recommendation = (
            "Float32 dönüşümü yalnızca split içi duplicate üretti. "
            "Float32 üretim sırasında split içi yeniden deduplication gerekir."
        )

    else:
        recommendation = (
            "Float32 dönüşümü splitler arası veya sınıflar arası "
            "çakışma üretti. Float32 uzayında yeniden deduplication "
            "ve yeniden split oluşturulmalıdır."
        )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "data_directory": str(
            data_dir
        ),
        "input_dtype": "float64",
        "audited_dtype": "float32",
        "feature_count": int(
            len(feature_columns)
        ),
        "class_count": int(
            len(classes)
        ),
        "total_input_rows": int(
            total_rows
        ),
        "unique_float32_vector_count": int(
            unique_float32_vectors
        ),
        "float32_collision_group_count": int(
            len(duplicate_group_indices)
        ),
        "float32_duplicate_rows_beyond_first": int(
            duplicate_rows_beyond_first
        ),
        "float32_duplicate_percentage": float(
            duplicate_rows_beyond_first
            / total_rows
            * 100
            if total_rows > 0
            else 0.0
        ),
        "cross_split_collision_group_count": int(
            cross_split_group_count
        ),
        "rows_in_cross_split_collision_groups": int(
            rows_in_cross_split_groups
        ),
        "cross_class_conflict_group_count": int(
            cross_class_group_count
        ),
        "rows_in_cross_class_conflict_groups": int(
            rows_in_cross_class_groups
        ),
        "cross_split_and_class_group_count": int(
            cross_split_and_class_group_count
        ),
        "no_new_float32_collisions": (
            no_new_collisions
        ),
        "existing_split_remains_leakage_free": (
            no_cross_split_collisions
        ),
        "class_labels_remain_unambiguous": (
            no_cross_class_conflicts
        ),
        "safe_to_materialize_float32_without_changes": bool(
            no_new_collisions
            and no_cross_split_collisions
            and no_cross_class_conflicts
        ),
        "recommendation": recommendation,
        "split_report": str(
            split_report_file
        ),
        "collision_group_report": str(
            collision_report_file
        ),
    }

    write_json_atomic(
        data=summary,
        output_file=summary_file,
    )

    print()
    print("=" * 78)
    print("Float32 dönüşüm denetimi tamamlandı")
    print("=" * 78)
    print(
        "Toplam float64 vektör       : "
        f"{total_rows:,}"
    )
    print(
        "Benzersiz float32 vektör    : "
        f"{unique_float32_vectors:,}"
    )
    print(
        "Yeni duplicate fazla kayıt  : "
        f"{duplicate_rows_beyond_first:,}"
    )
    print(
        "Yeni duplicate oranı        : "
        f"{summary['float32_duplicate_percentage']:.6f}%"
    )
    print(
        "Splitler arası çakışma grubu: "
        f"{cross_split_group_count:,}"
    )
    print(
        "Sınıflar arası çelişki grubu: "
        f"{cross_class_group_count:,}"
    )
    print(
        "Değişikliksiz float32 güvenli: "
        f"{summary['safe_to_materialize_float32_without_changes']}"
    )
    print()
    print(f"Öneri        : {recommendation}")
    print(f"Split raporu : {split_report_file}")
    print(f"Grup raporu  : {collision_report_file}")
    print(f"JSON özet    : {summary_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()