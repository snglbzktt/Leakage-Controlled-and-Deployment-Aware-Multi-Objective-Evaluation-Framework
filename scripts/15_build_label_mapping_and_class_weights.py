"""
N-BaIoT nihai etiket eşlemesini ve sınıf ağırlıklarını oluşturur.

Bilimsel kurallar:
- Sınıf sırası float32 kanonik SQLite veritabanından alınır.
- Sınıf ağırlıkları yalnızca train splitinden hesaplanır.
- Validation ve test sayıları yalnızca dağılım denetimi için kullanılır.
- Uniform, inverse-frequency ve inverse-square-root ağırlıkları üretilir.
- Hangi ağırlığın ana deneyde kullanılacağı bu rapor incelendikten sonra
  deney protokolünde sabitlenecektir.

Çıktılar:
models/preprocessing/nbaiot_label_mapping_seed2026.json
models/preprocessing/nbaiot_class_weights_seed2026.npz
results/reports/nbaiot_final_class_distribution.csv
results/reports/nbaiot_label_and_class_weight_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
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

DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "models"
    / "preprocessing"
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


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT sınıf eşleme dosyasını ve "
            "train tabanlı sınıf ağırlıklarını oluşturur."
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
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Etiket ve ağırlık artifactlarının kaydedileceği klasör.",
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
        help="Veri seti tanımlayıcı tohumu.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Parquet etiket okuma parça büyüklüğü.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Mevcut çıktıların üzerine yazar.",
    )

    return parser.parse_args()


# ==========================================================
# FILE HELPERS
# ==========================================================

def json_default(value: object) -> object:
    """NumPy ve Path nesnelerini JSON uyumlu hâle getirir."""

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
    """JSON dosyasını atomik olarak yazar."""

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


def write_csv_atomic(
    frame: pd.DataFrame,
    output_file: Path,
) -> None:
    """CSV dosyasını atomik olarak yazar."""

    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    frame.to_csv(
        temporary_file,
        index=False,
        encoding="utf-8",
    )

    temporary_file.replace(output_file)


def calculate_sha256(
    file_path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    """Dosyanın SHA-256 özetini hesaplar."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        while True:
            block = file_handle.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


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


def load_classes_from_database(
    database_file: Path,
    expected_seed: int,
) -> list[str]:
    """Nihai sınıf sırasını SQLite veritabanından yükler."""

    if not database_file.exists():
        raise FileNotFoundError(
            f"Kanonik SQLite bulunamadı: {database_file}"
        )

    connection = sqlite3.connect(
        database_file,
        timeout=120,
    )

    try:
        required_tables = {
            "run_metadata",
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
                "SQLite içinde gerekli tablolar eksik: "
                + ", ".join(missing_tables)
            )

        classes_json = read_run_metadata(
            connection,
            "classes_json",
        )

        if classes_json is None:
            raise RuntimeError(
                "SQLite içinde classes_json bulunamadı."
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

        if len(classes) != len(set(classes)):
            raise RuntimeError(
                "SQLite sınıf listesinde tekrar var."
            )

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

        database_class_codes = {
            int(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT class_code
                FROM canonical_assignments
                ORDER BY class_code
                """
            ).fetchall()
        }

        expected_class_codes = set(
            range(len(classes))
        )

        if database_class_codes != expected_class_codes:
            raise RuntimeError(
                "SQLite sınıf kodları sınıf listesiyle uyuşmuyor."
            )

        return classes

    finally:
        connection.close()


# ==========================================================
# PARQUET LABEL COUNTS
# ==========================================================

def count_split_labels(
    parquet_path: Path,
    split_name: str,
    classes: list[str],
    batch_size: int,
) -> tuple[Counter[str], int]:
    """Bir Parquet splitindeki sınıf etiketlerini sayar."""

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet dosyası bulunamadı: {parquet_path}"
        )

    parquet_file = pq.ParquetFile(
        parquet_path
    )

    if "class_label" not in parquet_file.schema_arrow.names:
        raise ValueError(
            f"{split_name} Parquet dosyasında class_label yok."
        )

    class_label_type = parquet_file.schema_arrow.field(
        "class_label"
    ).type

    if class_label_type != pa.string():
        raise TypeError(
            f"{split_name}/class_label string değil: "
            f"{class_label_type}"
        )

    expected_row_count = int(
        parquet_file.metadata.num_rows
    )

    if expected_row_count <= 0:
        raise RuntimeError(
            f"{split_name} Parquet dosyası boş."
        )

    allowed_classes = set(
        classes
    )

    counts: Counter[str] = Counter()

    processed_rows = 0

    progress = tqdm(
        total=expected_row_count,
        desc=f"{split_name} etiketleri sayılıyor",
        unit="satır",
    )

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["class_label"],
    ):
        frame = batch.to_pandas()

        labels = frame[
            "class_label"
        ].astype(str)

        if labels.isna().any():
            raise RuntimeError(
                f"{split_name} içinde eksik sınıf etiketi bulundu."
            )

        unknown_classes = (
            set(labels.unique())
            - allowed_classes
        )

        if unknown_classes:
            raise RuntimeError(
                f"{split_name} içinde bilinmeyen sınıflar var: "
                + ", ".join(sorted(unknown_classes))
            )

        value_counts = labels.value_counts()

        for class_label, count in value_counts.items():
            counts[
                str(class_label)
            ] += int(count)

        batch_length = int(
            len(labels)
        )

        processed_rows += batch_length

        progress.update(
            batch_length
        )

    progress.close()

    if processed_rows != expected_row_count:
        raise RuntimeError(
            f"{split_name} işlenen satır sayısı uyuşmuyor: "
            f"beklenen={expected_row_count:,}, "
            f"işlenen={processed_rows:,}"
        )

    missing_classes = [
        class_label
        for class_label in classes
        if counts[class_label] <= 0
    ]

    if missing_classes:
        raise RuntimeError(
            f"{split_name} splitinde bulunmayan sınıflar var: "
            + ", ".join(missing_classes)
        )

    return (
        counts,
        processed_rows,
    )


# ==========================================================
# CLASS WEIGHTS
# ==========================================================

def normalize_mean_one(
    values: np.ndarray,
) -> np.ndarray:
    """Pozitif ağırlıkları aritmetik ortalaması 1 olacak şekilde ölçekler."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "Ağırlık dizisi tek boyutlu ve boş olmayan olmalıdır."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Ağırlık dizisinde sonlu olmayan değer var."
        )

    if np.any(values <= 0):
        raise ValueError(
            "Bütün sınıf ağırlıkları pozitif olmalıdır."
        )

    mean_value = float(
        values.mean()
    )

    if mean_value <= 0:
        raise ValueError(
            "Ağırlık ortalaması pozitif değil."
        )

    return (
        values
        / mean_value
    )


def calculate_weight_schemes(
    train_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Üç farklı sınıf ağırlık şemasını hesaplar."""

    train_counts = np.asarray(
        train_counts,
        dtype=np.float64,
    )

    if np.any(train_counts <= 0):
        raise ValueError(
            "Sınıf ağırlığı hesaplamak için bütün train "
            "sayıları pozitif olmalıdır."
        )

    class_count = int(
        len(train_counts)
    )

    train_total = float(
        train_counts.sum()
    )

    uniform = np.ones(
        class_count,
        dtype=np.float64,
    )

    inverse_frequency_raw = (
        train_total
        / (
            class_count
            * train_counts
        )
    )

    inverse_square_root_raw = np.sqrt(
        inverse_frequency_raw
    )

    return {
        "uniform": normalize_mean_one(
            uniform
        ),
        "inverse_frequency_mean1": normalize_mean_one(
            inverse_frequency_raw
        ),
        "inverse_square_root_frequency_mean1": normalize_mean_one(
            inverse_square_root_raw
        ),
    }


def save_weight_artifact(
    output_file: Path,
    classes: list[str],
    train_counts: np.ndarray,
    weight_schemes: dict[str, np.ndarray],
    seed: int,
) -> None:
    """Sınıf ağırlıklarını taşınabilir NPZ biçiminde kaydeder."""

    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    with temporary_file.open("wb") as file_handle:
        np.savez_compressed(
            file_handle,
            class_names=np.asarray(
                classes,
                dtype=np.str_,
            ),
            class_indices=np.arange(
                len(classes),
                dtype=np.int64,
            ),
            train_counts=np.asarray(
                train_counts,
                dtype=np.int64,
            ),
            uniform=np.asarray(
                weight_schemes["uniform"],
                dtype=np.float32,
            ),
            inverse_frequency_mean1=np.asarray(
                weight_schemes[
                    "inverse_frequency_mean1"
                ],
                dtype=np.float32,
            ),
            inverse_square_root_frequency_mean1=np.asarray(
                weight_schemes[
                    "inverse_square_root_frequency_mean1"
                ],
                dtype=np.float32,
            ),
            seed=np.asarray(
                [seed],
                dtype=np.int64,
            ),
        )

    temporary_file.replace(
        output_file
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    data_dir = args.data_dir.resolve()
    database_file = args.database_file.resolve()
    artifact_dir = args.artifact_dir.resolve()
    report_dir = args.report_dir.resolve()

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size sıfırdan büyük olmalıdır."
        )

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Float32 veri klasörü bulunamadı: {data_dir}"
        )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping_file = (
        artifact_dir
        / f"nbaiot_label_mapping_seed{args.seed}.json"
    )

    weight_file = (
        artifact_dir
        / f"nbaiot_class_weights_seed{args.seed}.npz"
    )

    distribution_file = (
        report_dir
        / "nbaiot_final_class_distribution.csv"
    )

    summary_file = (
        report_dir
        / "nbaiot_label_and_class_weight_summary.json"
    )

    output_files = (
        mapping_file,
        weight_file,
        distribution_file,
        summary_file,
    )

    existing_files = [
        file_path
        for file_path in output_files
        if file_path.exists()
    ]

    if existing_files and not args.overwrite:
        raise FileExistsError(
            "Çıktı dosyaları zaten mevcut. "
            "--overwrite kullan: "
            + ", ".join(
                str(file_path)
                for file_path in existing_files
            )
        )

    classes = load_classes_from_database(
        database_file=database_file,
        expected_seed=args.seed,
    )

    class_to_index = {
        class_label: class_index
        for class_index, class_label in enumerate(
            classes
        )
    }

    index_to_class = {
        str(class_index): class_label
        for class_label, class_index
        in class_to_index.items()
    }

    print("=" * 78)
    print("N-BaIoT Etiket Eşleme ve Sınıf Ağırlıkları")
    print("=" * 78)
    print(f"Veri klasörü       : {data_dir}")
    print(f"Kanonik SQLite     : {database_file}")
    print(f"Sınıf sayısı       : {len(classes)}")
    print(f"Etiket sırası      : SQLite classes_json")
    print(f"Ağırlık fit kaynağı: Yalnızca train")
    print(f"Batch boyutu       : {args.batch_size:,}")
    print("=" * 78)

    split_counts: dict[
        str,
        Counter[str],
    ] = {}

    split_totals: dict[
        str,
        int,
    ] = {}

    for split_name in SPLIT_NAMES:
        (
            class_counts,
            split_total,
        ) = count_split_labels(
            parquet_path=(
                data_dir
                / f"{split_name}.parquet"
            ),
            split_name=split_name,
            classes=classes,
            batch_size=args.batch_size,
        )

        split_counts[
            split_name
        ] = class_counts

        split_totals[
            split_name
        ] = split_total

    train_counts_array = np.asarray(
        [
            split_counts[
                "train"
            ][class_label]
            for class_label in classes
        ],
        dtype=np.int64,
    )

    weight_schemes = (
        calculate_weight_schemes(
            train_counts_array
        )
    )

    class_records: list[
        dict[str, Any]
    ] = []

    global_total = int(
        sum(
            split_totals.values()
        )
    )

    global_class_counts = {
        class_label: int(
            sum(
                split_counts[
                    split_name
                ][class_label]
                for split_name in SPLIT_NAMES
            )
        )
        for class_label in classes
    }

    for class_index, class_label in enumerate(
        classes
    ):
        global_class_total = int(
            global_class_counts[
                class_label
            ]
        )

        for split_name in SPLIT_NAMES:
            sample_count = int(
                split_counts[
                    split_name
                ][class_label]
            )

            class_records.append(
                {
                    "class_index": int(
                        class_index
                    ),
                    "class_label": (
                        class_label
                    ),
                    "split": (
                        split_name
                    ),
                    "sample_count": (
                        sample_count
                    ),
                    "percentage_within_split": float(
                        sample_count
                        / split_totals[
                            split_name
                        ]
                        * 100
                    ),
                    "percentage_of_class_in_split": float(
                        sample_count
                        / global_class_total
                        * 100
                    ),
                    "global_class_sample_count": (
                        global_class_total
                    ),
                    "global_class_percentage": float(
                        global_class_total
                        / global_total
                        * 100
                    ),
                    "uniform_weight": float(
                        weight_schemes[
                            "uniform"
                        ][class_index]
                    ),
                    "inverse_frequency_weight_mean1": float(
                        weight_schemes[
                            "inverse_frequency_mean1"
                        ][class_index]
                    ),
                    "inverse_square_root_weight_mean1": float(
                        weight_schemes[
                            "inverse_square_root_frequency_mean1"
                        ][class_index]
                    ),
                }
            )

    distribution_frame = pd.DataFrame(
        class_records
    )

    mapping_data: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": (
            f"nbaiot_float32_canonical_seed{args.seed}"
        ),
        "seed": int(
            args.seed
        ),
        "class_count": int(
            len(classes)
        ),
        "class_order_source": (
            "classes_json in nbaiot_float32_canonical.sqlite"
        ),
        "class_names": classes,
        "class_to_index": (
            class_to_index
        ),
        "index_to_class": (
            index_to_class
        ),
        "label_dtype": "int64",
        "unknown_label_policy": (
            "Raise an error. Unknown labels are not permitted."
        ),
    }

    write_json_atomic(
        data=mapping_data,
        output_file=mapping_file,
    )

    save_weight_artifact(
        output_file=weight_file,
        classes=classes,
        train_counts=train_counts_array,
        weight_schemes=weight_schemes,
        seed=args.seed,
    )

    write_csv_atomic(
        frame=distribution_frame,
        output_file=distribution_file,
    )

    minimum_train_count = int(
        train_counts_array.min()
    )

    maximum_train_count = int(
        train_counts_array.max()
    )

    imbalance_ratio = float(
        maximum_train_count
        / minimum_train_count
    )

    smallest_class_index = int(
        np.argmin(
            train_counts_array
        )
    )

    largest_class_index = int(
        np.argmax(
            train_counts_array
        )
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
        "class_count": int(
            len(classes)
        ),
        "classes": classes,
        "split_sample_counts": {
            split_name: int(
                split_totals[
                    split_name
                ]
            )
            for split_name in SPLIT_NAMES
        },
        "global_sample_count": int(
            global_total
        ),
        "weight_fit_split": "train",
        "validation_and_test_used_for_weights": False,
        "minimum_train_class_count": (
            minimum_train_count
        ),
        "maximum_train_class_count": (
            maximum_train_count
        ),
        "smallest_train_class": (
            classes[
                smallest_class_index
            ]
        ),
        "largest_train_class": (
            classes[
                largest_class_index
            ]
        ),
        "train_imbalance_ratio_max_to_min": (
            imbalance_ratio
        ),
        "weight_schemes": {
            scheme_name: {
                classes[index]: float(
                    scheme_values[index]
                )
                for index in range(
                    len(classes)
                )
            }
            for scheme_name, scheme_values
            in weight_schemes.items()
        },
        "weight_normalization": (
            "Each weight vector has arithmetic mean equal to 1."
        ),
        "primary_metric": "Macro F1",
        "loss_selection_status": (
            "Not yet locked. Weight schemes are generated "
            "for protocol selection before model training."
        ),
        "label_mapping_file": str(
            mapping_file
        ),
        "class_weight_file": str(
            weight_file
        ),
        "distribution_report": str(
            distribution_file
        ),
        "label_mapping_sha256": (
            calculate_sha256(
                mapping_file
            )
        ),
        "class_weight_sha256": (
            calculate_sha256(
                weight_file
            )
        ),
        "validation_passed": True,
    }

    write_json_atomic(
        data=summary,
        output_file=summary_file,
    )

    print()
    print("=" * 78)
    print("Etiket eşleme ve sınıf ağırlıkları tamamlandı")
    print("=" * 78)

    print("Nihai sınıf sırası:")

    for class_index, class_label in enumerate(
        classes
    ):
        print(
            f"  {class_index:2d} -> "
            f"{class_label:16s} | "
            f"train={train_counts_array[class_index]:,}"
        )

    print()
    print(
        "Train toplam örnek          : "
        f"{split_totals['train']:,}"
    )
    print(
        "Validation toplam örnek     : "
        f"{split_totals['validation']:,}"
    )
    print(
        "Test toplam örnek           : "
        f"{split_totals['test']:,}"
    )
    print(
        "En küçük train sınıfı       : "
        f"{classes[smallest_class_index]} "
        f"({minimum_train_count:,})"
    )
    print(
        "En büyük train sınıfı       : "
        f"{classes[largest_class_index]} "
        f"({maximum_train_count:,})"
    )
    print(
        "Train dengesizlik oranı     : "
        f"{imbalance_ratio:.4f}"
    )
    print()
    print(f"Etiket eşleme : {mapping_file}")
    print(f"Sınıf ağırlığı: {weight_file}")
    print(f"Dağılım raporu: {distribution_file}")
    print(f"JSON özet     : {summary_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()