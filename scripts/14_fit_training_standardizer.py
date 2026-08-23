"""
N-BaIoT eğitim verisinden StandardScaler parametrelerini üretir.

Bilimsel kural:
- Ortalama ve standart sapma yalnızca train.parquet üzerinden hesaplanır.
- Validation ve test verileri fit işlemine katılmaz.
- Validation ve test yalnızca dönüşümün sonlu olduğunu doğrulamak için okunur.
- Ölçeklenmiş veri diske yeniden yazılmaz.
- Eğitim sırasında dönüşüm çevrim içi uygulanacaktır.

Çıktılar:
models/preprocessing/nbaiot_standard_scaler_seed2026.npz
results/reports/nbaiot_training_standardizer_statistics.csv
results/reports/nbaiot_training_standardizer_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
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

DEFAULT_SCHEMA_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_reference_schema.csv"
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
            "N-BaIoT train splitinden StandardScaler "
            "parametrelerini hesaplar ve doğrular."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Float32 kanonik veri klasörü.",
    )

    parser.add_argument(
        "--schema-file",
        type=Path,
        default=DEFAULT_SCHEMA_FILE,
        help="Referans özellik şeması.",
    )

    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Ölçekleyici parametrelerinin kaydedileceği klasör.",
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
        default=50_000,
        help="Parquet okuma parça büyüklüğü.",
    )

    parser.add_argument(
        "--verification-tolerance",
        type=float,
        default=1e-4,
        help=(
            "Dönüştürülmüş train ortalama ve standart "
            "sapma doğrulama toleransı."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Mevcut çıktıların üzerine yazar.",
    )

    return parser.parse_args()


# ==========================================================
# JSON AND FILE HELPERS
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
    """Bir dosyanın SHA-256 özetini hesaplar."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        while True:
            block = file_handle.read(
                block_size
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


# ==========================================================
# SCHEMA HELPERS
# ==========================================================

def load_feature_columns(
    schema_file: Path,
) -> list[str]:
    """115 özellik sütununun adlarını yükler."""

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

    if len(feature_columns) != len(
        set(feature_columns)
    ):
        raise ValueError(
            "Şema dosyasında tekrarlanan sütun adı var."
        )

    return feature_columns


def validate_parquet_schema(
    parquet_file: pq.ParquetFile,
    feature_columns: list[str],
    split_name: str,
) -> int:
    """Parquet şemasını ve satır sayısını doğrular."""

    expected_columns = (
        feature_columns
        + ["class_label"]
    )

    arrow_schema = parquet_file.schema_arrow

    if arrow_schema.names != expected_columns:
        raise ValueError(
            f"{split_name} sütun sırası referans şemayla uyuşmuyor."
        )

    for feature_name in feature_columns:
        feature_type = arrow_schema.field(
            feature_name
        ).type

        if feature_type != pa.float32():
            raise TypeError(
                f"{split_name}/{feature_name} float32 değil: "
                f"{feature_type}"
            )

    if arrow_schema.field(
        "class_label"
    ).type != pa.string():
        raise TypeError(
            f"{split_name}/class_label string değil."
        )

    row_count = int(
        parquet_file.metadata.num_rows
    )

    if row_count <= 0:
        raise RuntimeError(
            f"{split_name} Parquet dosyası boş."
        )

    return row_count


# ==========================================================
# FIT
# ==========================================================

def fit_training_standardizer(
    train_file: Path,
    feature_columns: list[str],
    batch_size: int,
) -> tuple[
    StandardScaler,
    np.ndarray,
    np.ndarray,
    int,
]:
    """StandardScaler parametrelerini yalnızca train verisinden öğrenir."""

    parquet_file = pq.ParquetFile(
        train_file
    )

    train_row_count = validate_parquet_schema(
        parquet_file=parquet_file,
        feature_columns=feature_columns,
        split_name="train",
    )

    feature_count = len(
        feature_columns
    )

    train_minimum = np.full(
        feature_count,
        np.inf,
        dtype=np.float64,
    )

    train_maximum = np.full(
        feature_count,
        -np.inf,
        dtype=np.float64,
    )

    scaler = StandardScaler(
        copy=True,
        with_mean=True,
        with_std=True,
    )

    processed_rows = 0

    progress = tqdm(
        total=train_row_count,
        desc="StandardScaler train verisine fit ediliyor",
        unit="satır",
    )

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=feature_columns,
    ):
        frame = batch.to_pandas()

        values = frame.to_numpy(
            dtype=np.float64,
            copy=True,
        )

        if values.shape[1] != feature_count:
            raise RuntimeError(
                "Okunan özellik sayısı referans şemayla uyuşmuyor."
            )

        if not np.isfinite(values).all():
            raise RuntimeError(
                "Train verisinde NaN veya sonsuz değer bulundu."
            )

        scaler.partial_fit(
            values
        )

        train_minimum = np.minimum(
            train_minimum,
            values.min(axis=0),
        )

        train_maximum = np.maximum(
            train_maximum,
            values.max(axis=0),
        )

        batch_length = int(
            len(values)
        )

        processed_rows += (
            batch_length
        )

        progress.update(
            batch_length
        )

    progress.close()

    if processed_rows != train_row_count:
        raise RuntimeError(
            "İşlenen train satır sayısı uyuşmuyor: "
            f"beklenen={train_row_count:,}, "
            f"işlenen={processed_rows:,}"
        )

    fitted_sample_count = int(
        np.asarray(
            scaler.n_samples_seen_
        ).reshape(-1)[0]
    )

    if fitted_sample_count != train_row_count:
        raise RuntimeError(
            "StandardScaler örnek sayısı uyuşmuyor: "
            f"beklenen={train_row_count:,}, "
            f"scaler={fitted_sample_count:,}"
        )

    if (
        scaler.mean_ is None
        or scaler.var_ is None
        or scaler.scale_ is None
    ):
        raise RuntimeError(
            "StandardScaler parametreleri oluşturulamadı."
        )

    if not np.isfinite(
        scaler.mean_
    ).all():
        raise RuntimeError(
            "Scaler ortalamalarında sonlu olmayan değer var."
        )

    if not np.isfinite(
        scaler.var_
    ).all():
        raise RuntimeError(
            "Scaler varyanslarında sonlu olmayan değer var."
        )

    if not np.isfinite(
        scaler.scale_
    ).all():
        raise RuntimeError(
            "Scaler ölçeklerinde sonlu olmayan değer var."
        )

    if np.any(
        scaler.scale_ <= 0
    ):
        raise RuntimeError(
            "Scaler içinde sıfır veya negatif ölçek bulundu."
        )

    return (
        scaler,
        train_minimum,
        train_maximum,
        train_row_count,
    )


# ==========================================================
# TRANSFORMATION AUDIT
# ==========================================================

def audit_scaled_splits(
    data_dir: Path,
    feature_columns: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
    """
    Bütün splitlerde standardizasyon dönüşümünü doğrular.

    Fit işlemi yapılmaz. Train'den öğrenilen mean ve scale kullanılır.
    """

    feature_count = len(
        feature_columns
    )

    split_records: list[
        dict[str, Any]
    ] = []

    train_scaled_sum = np.zeros(
        feature_count,
        dtype=np.float64,
    )

    train_scaled_square_sum = np.zeros(
        feature_count,
        dtype=np.float64,
    )

    train_scaled_count = 0

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

        row_count = validate_parquet_schema(
            parquet_file=parquet_file,
            feature_columns=feature_columns,
            split_name=split_name,
        )

        scaled_minimum = np.full(
            feature_count,
            np.inf,
            dtype=np.float64,
        )

        scaled_maximum = np.full(
            feature_count,
            -np.inf,
            dtype=np.float64,
        )

        processed_rows = 0
        raw_nonfinite_count = 0
        scaled_nonfinite_count = 0

        progress = tqdm(
            total=row_count,
            desc=f"{split_name} dönüşümü doğrulanıyor",
            unit="satır",
        )

        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=feature_columns,
        ):
            frame = batch.to_pandas()

            values = frame.to_numpy(
                dtype=np.float64,
                copy=True,
            )

            current_raw_nonfinite = int(
                np.count_nonzero(
                    ~np.isfinite(values)
                )
            )

            raw_nonfinite_count += (
                current_raw_nonfinite
            )

            if current_raw_nonfinite > 0:
                raise RuntimeError(
                    f"{split_name} ham verisinde sonlu "
                    "olmayan değer bulundu."
                )

            scaled_values = (
                values - mean
            ) / scale

            current_scaled_nonfinite = int(
                np.count_nonzero(
                    ~np.isfinite(
                        scaled_values
                    )
                )
            )

            scaled_nonfinite_count += (
                current_scaled_nonfinite
            )

            if current_scaled_nonfinite > 0:
                raise RuntimeError(
                    f"{split_name} standardizasyonundan sonra "
                    "sonlu olmayan değer oluştu."
                )

            scaled_minimum = np.minimum(
                scaled_minimum,
                scaled_values.min(
                    axis=0
                ),
            )

            scaled_maximum = np.maximum(
                scaled_maximum,
                scaled_values.max(
                    axis=0
                ),
            )

            batch_length = int(
                len(scaled_values)
            )

            if split_name == "train":
                train_scaled_sum += (
                    scaled_values.sum(
                        axis=0,
                        dtype=np.float64,
                    )
                )

                train_scaled_square_sum += (
                    np.square(
                        scaled_values
                    ).sum(
                        axis=0,
                        dtype=np.float64,
                    )
                )

                train_scaled_count += (
                    batch_length
                )

            processed_rows += (
                batch_length
            )

            progress.update(
                batch_length
            )

        progress.close()

        if processed_rows != row_count:
            raise RuntimeError(
                f"{split_name} işlenen satır sayısı uyuşmuyor."
            )

        split_records.append(
            {
                "split": split_name,
                "row_count": int(
                    processed_rows
                ),
                "raw_nonfinite_value_count": int(
                    raw_nonfinite_count
                ),
                "scaled_nonfinite_value_count": int(
                    scaled_nonfinite_count
                ),
                "global_scaled_minimum": float(
                    scaled_minimum.min()
                ),
                "global_scaled_maximum": float(
                    scaled_maximum.max()
                ),
                "global_scaled_maximum_absolute": float(
                    max(
                        abs(
                            scaled_minimum.min()
                        ),
                        abs(
                            scaled_maximum.max()
                        ),
                    )
                ),
                "transformation_valid": True,
            }
        )

    if train_scaled_count <= 0:
        raise RuntimeError(
            "Train dönüşüm doğrulamasında örnek bulunamadı."
        )

    train_scaled_mean = (
        train_scaled_sum
        / train_scaled_count
    )

    train_scaled_variance = (
        train_scaled_square_sum
        / train_scaled_count
        - np.square(
            train_scaled_mean
        )
    )

    train_scaled_variance = np.maximum(
        train_scaled_variance,
        0.0,
    )

    train_scaled_standard_deviation = np.sqrt(
        train_scaled_variance
    )

    return (
        split_records,
        train_scaled_mean,
        train_scaled_standard_deviation,
    )


# ==========================================================
# ARTIFACT
# ==========================================================

def save_scaler_artifact(
    output_file: Path,
    feature_columns: list[str],
    mean: np.ndarray,
    variance: np.ndarray,
    scale: np.ndarray,
    train_minimum: np.ndarray,
    train_maximum: np.ndarray,
    train_sample_count: int,
    seed: int,
) -> None:
    """Ölçekleyici parametrelerini taşınabilir NPZ biçiminde kaydeder."""

    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    with temporary_file.open(
        "wb"
    ) as file_handle:
        np.savez_compressed(
            file_handle,
            feature_names=np.asarray(
                feature_columns,
                dtype=np.str_,
            ),
            mean=np.asarray(
                mean,
                dtype=np.float64,
            ),
            variance=np.asarray(
                variance,
                dtype=np.float64,
            ),
            scale=np.asarray(
                scale,
                dtype=np.float64,
            ),
            train_minimum=np.asarray(
                train_minimum,
                dtype=np.float64,
            ),
            train_maximum=np.asarray(
                train_maximum,
                dtype=np.float64,
            ),
            train_sample_count=np.asarray(
                [train_sample_count],
                dtype=np.int64,
            ),
            seed=np.asarray(
                [seed],
                dtype=np.int64,
            ),
            source_dtype=np.asarray(
                ["float32"],
                dtype=np.str_,
            ),
            parameter_dtype=np.asarray(
                ["float64"],
                dtype=np.str_,
            ),
            method=np.asarray(
                ["StandardScaler population variance ddof=0"],
                dtype=np.str_,
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
    schema_file = args.schema_file.resolve()
    artifact_dir = args.artifact_dir.resolve()
    report_dir = args.report_dir.resolve()

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size sıfırdan büyük olmalıdır."
        )

    if args.verification_tolerance <= 0:
        raise ValueError(
            "Doğrulama toleransı sıfırdan büyük olmalıdır."
        )

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Veri klasörü bulunamadı: {data_dir}"
        )

    feature_columns = load_feature_columns(
        schema_file
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_file = (
        artifact_dir
        / f"nbaiot_standard_scaler_seed{args.seed}.npz"
    )

    statistics_file = (
        report_dir
        / "nbaiot_training_standardizer_statistics.csv"
    )

    summary_file = (
        report_dir
        / "nbaiot_training_standardizer_summary.json"
    )

    output_files = (
        artifact_file,
        statistics_file,
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
                str(path)
                for path in existing_files
            )
        )

    train_file = (
        data_dir
        / "train.parquet"
    )

    if not train_file.exists():
        raise FileNotFoundError(
            f"Train Parquet bulunamadı: {train_file}"
        )

    print("=" * 78)
    print("N-BaIoT Eğitim Standardizasyon Parametreleri")
    print("=" * 78)
    print(f"Veri klasörü     : {data_dir}")
    print(f"Train dosyası    : {train_file}")
    print(f"Özellik sayısı   : {len(feature_columns)}")
    print(f"Fit kaynağı      : Yalnızca train")
    print(f"Batch boyutu     : {args.batch_size:,}")
    print(f"Çıktı artifactı  : {artifact_file}")
    print("=" * 78)

    (
        scaler,
        train_minimum,
        train_maximum,
        train_row_count,
    ) = fit_training_standardizer(
        train_file=train_file,
        feature_columns=feature_columns,
        batch_size=args.batch_size,
    )

    mean = np.asarray(
        scaler.mean_,
        dtype=np.float64,
    )

    variance = np.asarray(
        scaler.var_,
        dtype=np.float64,
    )

    scale = np.asarray(
        scaler.scale_,
        dtype=np.float64,
    )

    zero_variance_mask = (
        variance == 0.0
    )

    zero_variance_count = int(
        np.count_nonzero(
            zero_variance_mask
        )
    )

    print()
    print(
        "Train'den öğrenilen dönüşüm bütün splitlerde doğrulanıyor..."
    )

    (
        split_records,
        train_scaled_mean,
        train_scaled_standard_deviation,
    ) = audit_scaled_splits(
        data_dir=data_dir,
        feature_columns=feature_columns,
        mean=mean,
        scale=scale,
        batch_size=args.batch_size,
    )

    nonconstant_mask = (
        ~zero_variance_mask
    )

    maximum_absolute_scaled_train_mean = float(
        np.max(
            np.abs(
                train_scaled_mean
            )
        )
    )

    if np.any(
        nonconstant_mask
    ):
        maximum_scaled_train_std_error = float(
            np.max(
                np.abs(
                    train_scaled_standard_deviation[
                        nonconstant_mask
                    ]
                    - 1.0
                )
            )
        )
    else:
        maximum_scaled_train_std_error = 0.0

    mean_verification_passed = bool(
        maximum_absolute_scaled_train_mean
        <= args.verification_tolerance
    )

    std_verification_passed = bool(
        maximum_scaled_train_std_error
        <= args.verification_tolerance
    )

    all_transformations_valid = bool(
        all(
            record[
                "transformation_valid"
            ]
            for record in split_records
        )
    )

    all_values_finite = bool(
        all(
            record[
                "raw_nonfinite_value_count"
            ] == 0
            and record[
                "scaled_nonfinite_value_count"
            ] == 0
            for record in split_records
        )
    )

    validation_passed = bool(
        all_transformations_valid
        and all_values_finite
        and mean_verification_passed
        and std_verification_passed
    )

    if not validation_passed:
        raise RuntimeError(
            "Standardizasyon doğrulaması başarısız: "
            f"max_abs_mean={maximum_absolute_scaled_train_mean:.8e}, "
            f"max_std_error={maximum_scaled_train_std_error:.8e}"
        )

    statistics = pd.DataFrame(
        {
            "feature_index": np.arange(
                len(feature_columns),
                dtype=np.int64,
            ),
            "feature_name": feature_columns,
            "train_mean": mean,
            "train_variance_ddof0": variance,
            "train_standard_deviation_ddof0": (
                np.sqrt(
                    variance
                )
            ),
            "scaler_scale": scale,
            "train_minimum": train_minimum,
            "train_maximum": train_maximum,
            "train_range": (
                train_maximum
                - train_minimum
            ),
            "zero_variance": (
                zero_variance_mask
            ),
            "scaled_train_mean": (
                train_scaled_mean
            ),
            "scaled_train_standard_deviation": (
                train_scaled_standard_deviation
            ),
        }
    )

    write_csv_atomic(
        frame=statistics,
        output_file=statistics_file,
    )

    save_scaler_artifact(
        output_file=artifact_file,
        feature_columns=feature_columns,
        mean=mean,
        variance=variance,
        scale=scale,
        train_minimum=train_minimum,
        train_maximum=train_maximum,
        train_sample_count=train_row_count,
        seed=args.seed,
    )

    artifact_sha256 = calculate_sha256(
        artifact_file
    )

    parameter_memory_float64_bytes = int(
        mean.nbytes
        + scale.nbytes
    )

    parameter_memory_float32_bytes = int(
        len(feature_columns)
        * 2
        * np.dtype(
            np.float32
        ).itemsize
    )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "data_directory": str(
            data_dir
        ),
        "fit_split": "train",
        "validation_splits_used_only_for_audit": [
            "validation",
            "test",
        ],
        "data_leakage_prevention": (
            "Mean and variance were fitted exclusively on train.parquet."
        ),
        "seed": int(
            args.seed
        ),
        "method": "StandardScaler",
        "with_mean": True,
        "with_standard_deviation": True,
        "variance_definition": (
            "Population variance with ddof=0, matching scikit-learn."
        ),
        "source_feature_dtype": "float32",
        "parameter_storage_dtype": "float64",
        "training_output_dtype_policy": (
            "Scaled arrays will be converted to float32 "
            "inside the PyTorch data pipeline."
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "train_sample_count": int(
            train_row_count
        ),
        "zero_variance_feature_count": int(
            zero_variance_count
        ),
        "zero_variance_features": [
            feature_columns[index]
            for index in np.flatnonzero(
                zero_variance_mask
            )
        ],
        "maximum_absolute_scaled_train_mean": (
            maximum_absolute_scaled_train_mean
        ),
        "maximum_scaled_train_standard_deviation_error": (
            maximum_scaled_train_std_error
        ),
        "verification_tolerance": float(
            args.verification_tolerance
        ),
        "scaled_train_mean_verification_passed": (
            mean_verification_passed
        ),
        "scaled_train_standard_deviation_verification_passed": (
            std_verification_passed
        ),
        "all_values_finite": (
            all_values_finite
        ),
        "all_split_transformations_valid": (
            all_transformations_valid
        ),
        "validation_passed": (
            validation_passed
        ),
        "split_audit": (
            split_records
        ),
        "scaler_artifact": str(
            artifact_file
        ),
        "scaler_artifact_sha256": (
            artifact_sha256
        ),
        "statistics_report": str(
            statistics_file
        ),
        "mean_and_scale_memory_float64_bytes": (
            parameter_memory_float64_bytes
        ),
        "mean_and_scale_memory_float32_bytes": (
            parameter_memory_float32_bytes
        ),
        "deployment_note": (
            "Preprocessing memory and computation must be included "
            "in simulated TinyML resource reporting."
        ),
    }

    write_json_atomic(
        data=summary,
        output_file=summary_file,
    )

    print()
    print("=" * 78)
    print("Eğitim standardizasyonu başarıyla oluşturuldu")
    print("=" * 78)
    print(
        "Fit için kullanılan split       : train"
    )
    print(
        "Train örnek sayısı              : "
        f"{train_row_count:,}"
    )
    print(
        "Özellik sayısı                  : "
        f"{len(feature_columns)}"
    )
    print(
        "Sıfır varyanslı özellik         : "
        f"{zero_variance_count}"
    )
    print(
        "Ölçekli train maksimum |ortalama|: "
        f"{maximum_absolute_scaled_train_mean:.8e}"
    )
    print(
        "Ölçekli train maksimum std hatası: "
        f"{maximum_scaled_train_std_error:.8e}"
    )
    print(
        "Tüm dönüşümler sonlu           : "
        f"{all_values_finite}"
    )
    print(
        "Doğrulama geçti                : "
        f"{validation_passed}"
    )
    print()
    print(f"Scaler artifactı : {artifact_file}")
    print(f"İstatistik raporu: {statistics_file}")
    print(f"JSON özet        : {summary_file}")
    print(f"SHA-256          : {artifact_sha256}")
    print("=" * 78)


if __name__ == "__main__":
    main()