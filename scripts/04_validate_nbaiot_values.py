"""
N-BaIoT sayısal veri kalite doğrulama betiği.

Bu betik bütün N-BaIoT CSV dosyalarını parça parça okuyarak:

- Sütun adlarını ve sırasını doğrular.
- Manifestteki satır sayılarıyla gerçek satır sayılarını karşılaştırır.
- Eksik değerleri sayar.
- Pozitif ve negatif sonsuz değerleri sayar.
- Sayısal olarak okunamayan dosyaları tespit eder.
- Her özelliğin minimum ve maksimum değerlerini hesaplar.
- Sabit özellikleri belirler.
- Dosya ve özellik bazlı CSV raporları üretir.
- JSON özet raporunu güvenli ve atomik biçimde kaydeder.

Veri setinin tamamı aynı anda RAM'e yüklenmez.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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

DEFAULT_SCHEMA_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_reference_schema.csv"
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
            "N-BaIoT sayısal veri kalite "
            "kontrollerini gerçekleştirir."
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
        "--schema-file",
        type=Path,
        default=DEFAULT_SCHEMA_FILE,
        help="Referans sütun şeması CSV dosyası.",
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
        default=100_000,
        help="Her seferde okunacak CSV satırı sayısı.",
    )

    return parser.parse_args()


# ==========================================================
# VALIDATION HELPERS
# ==========================================================

def load_reference_schema(
    schema_file: Path,
) -> list[str]:
    """Referans özellik sütunlarını yükler."""

    if not schema_file.exists():
        raise FileNotFoundError(
            f"Referans şema bulunamadı: {schema_file}"
        )

    schema = pd.read_csv(schema_file)

    if "column_name" not in schema.columns:
        raise ValueError(
            "Referans şema dosyasında "
            "'column_name' sütunu bulunmuyor."
        )

    columns = (
        schema["column_name"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if not columns:
        raise ValueError(
            "Referans şema dosyası boş."
        )

    if len(columns) != len(set(columns)):
        raise ValueError(
            "Referans şemada tekrarlanan sütun adları var."
        )

    return columns


def validate_manifest(
    manifest: pd.DataFrame,
) -> None:
    """Manifest dosyasının yapısını doğrular."""

    required_columns = {
        "device",
        "attack_family",
        "attack_type",
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
            "Manifest içinde başarısız olarak işaretlenmiş "
            f"{len(failed_files)} dosya bulunuyor."
        )

    if manifest["relative_path"].duplicated().any():
        raise ValueError(
            "Manifestte tekrarlanan dosya yolları bulunuyor."
        )


# ==========================================================
# JSON HELPERS
# ==========================================================

def json_default(value: object) -> object:
    """
    NumPy ve Path nesnelerini JSON uyumlu Python türlerine dönüştürür.
    """

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
        f"{type(value).__name__} türü JSON ile uyumlu değil."
    )


def write_json_atomic(
    data: dict[str, Any],
    output_file: Path,
) -> None:
    """
    JSON raporunu önce geçici dosyaya, ardından nihai dosyaya yazar.

    Program yazım sırasında kesilirse yarım JSON dosyası bırakılmaz.
    """

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


def safe_float(
    value: float | np.floating,
) -> float | None:
    """Sonlu olmayan değerleri JSON için None yapar."""

    numeric_value = float(value)

    if not np.isfinite(numeric_value):
        return None

    return numeric_value


# ==========================================================
# NUMERICAL STATISTICS
# ==========================================================

def update_column_statistics(
    values: np.ndarray,
    missing_counts: np.ndarray,
    positive_infinity_counts: np.ndarray,
    negative_infinity_counts: np.ndarray,
    finite_counts: np.ndarray,
    minimum_values: np.ndarray,
    maximum_values: np.ndarray,
) -> None:
    """
    Bir veri parçasından sütun bazlı kalite istatistiklerini günceller.
    """

    missing_counts += np.isnan(
        values
    ).sum(axis=0)

    positive_infinity_counts += np.isposinf(
        values
    ).sum(axis=0)

    negative_infinity_counts += np.isneginf(
        values
    ).sum(axis=0)

    column_count = values.shape[1]

    for column_index in range(column_count):
        column_values = values[:, column_index]

        finite_values = column_values[
            np.isfinite(column_values)
        ]

        if finite_values.size == 0:
            continue

        finite_counts[column_index] += int(
            finite_values.size
        )

        column_minimum = float(
            finite_values.min()
        )

        column_maximum = float(
            finite_values.max()
        )

        if column_minimum < minimum_values[column_index]:
            minimum_values[column_index] = column_minimum

        if column_maximum > maximum_values[column_index]:
            maximum_values[column_index] = column_maximum


# ==========================================================
# FILE INSPECTION
# ==========================================================

def inspect_csv_file(
    file_path: Path,
    expected_columns: list[str],
    chunk_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Tek bir CSV dosyasını parça parça doğrular.
    """

    feature_count = len(expected_columns)

    missing_counts = np.zeros(
        feature_count,
        dtype=np.int64,
    )

    positive_infinity_counts = np.zeros(
        feature_count,
        dtype=np.int64,
    )

    negative_infinity_counts = np.zeros(
        feature_count,
        dtype=np.int64,
    )

    finite_counts = np.zeros(
        feature_count,
        dtype=np.int64,
    )

    minimum_values = np.full(
        feature_count,
        np.inf,
        dtype=np.float64,
    )

    maximum_values = np.full(
        feature_count,
        -np.inf,
        dtype=np.float64,
    )

    # ------------------------------------------------------
    # Header validation
    # ------------------------------------------------------

    header_frame = pd.read_csv(
        file_path,
        nrows=0,
    )

    actual_columns = [
        str(column).strip()
        for column in header_frame.columns
    ]

    if actual_columns != expected_columns:
        raise ValueError(
            "Sütun adları veya sütun sırası "
            "referans şemayla uyuşmuyor."
        )

    # ------------------------------------------------------
    # Chunked numerical reading
    # ------------------------------------------------------

    actual_row_count = 0
    chunk_count = 0

    reader = pd.read_csv(
        file_path,
        chunksize=chunk_size,
        dtype=np.float64,
        low_memory=False,
        on_bad_lines="error",
    )

    for chunk in reader:
        chunk.columns = [
            str(column).strip()
            for column in chunk.columns
        ]

        if list(chunk.columns) != expected_columns:
            raise ValueError(
                "Dosyanın bir parçasında sütun şeması değişti."
            )

        values = chunk.to_numpy(
            dtype=np.float64,
            copy=False,
        )

        if values.ndim != 2:
            raise ValueError(
                "CSV verisi iki boyutlu bir tablo olarak okunamadı."
            )

        if values.shape[1] != feature_count:
            raise ValueError(
                "Okunan özellik sayısı referans şemayla uyuşmuyor."
            )

        actual_row_count += int(
            values.shape[0]
        )

        chunk_count += 1

        update_column_statistics(
            values=values,
            missing_counts=missing_counts,
            positive_infinity_counts=(
                positive_infinity_counts
            ),
            negative_infinity_counts=(
                negative_infinity_counts
            ),
            finite_counts=finite_counts,
            minimum_values=minimum_values,
            maximum_values=maximum_values,
        )

    total_missing = int(
        missing_counts.sum()
    )

    total_positive_infinity = int(
        positive_infinity_counts.sum()
    )

    total_negative_infinity = int(
        negative_infinity_counts.sum()
    )

    file_record: dict[str, Any] = {
        "actual_row_count": int(actual_row_count),
        "column_count": int(feature_count),
        "chunk_count": int(chunk_count),
        "missing_value_count": total_missing,
        "positive_infinity_count": (
            total_positive_infinity
        ),
        "negative_infinity_count": (
            total_negative_infinity
        ),
        "non_finite_value_count": int(
            total_missing
            + total_positive_infinity
            + total_negative_infinity
        ),
    }

    column_statistics = {
        "missing": missing_counts,
        "positive_infinity": (
            positive_infinity_counts
        ),
        "negative_infinity": (
            negative_infinity_counts
        ),
        "finite": finite_counts,
        "minimum": minimum_values,
        "maximum": maximum_values,
    }

    return file_record, column_statistics


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    data_root = args.data_root.resolve()
    manifest_file = args.manifest_file.resolve()
    schema_file = args.schema_file.resolve()
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

    expected_columns = load_reference_schema(
        schema_file
    )

    feature_count = len(expected_columns)

    # ------------------------------------------------------
    # Global statistics
    # ------------------------------------------------------

    global_missing = np.zeros(
        feature_count,
        dtype=np.int64,
    )

    global_positive_infinity = np.zeros(
        feature_count,
        dtype=np.int64,
    )

    global_negative_infinity = np.zeros(
        feature_count,
        dtype=np.int64,
    )

    global_finite = np.zeros(
        feature_count,
        dtype=np.int64,
    )

    global_minimum = np.full(
        feature_count,
        np.inf,
        dtype=np.float64,
    )

    global_maximum = np.full(
        feature_count,
        -np.inf,
        dtype=np.float64,
    )

    file_records: list[dict[str, Any]] = []

    print("=" * 78)
    print("N-BaIoT Sayısal Veri Kalite Doğrulaması")
    print("=" * 78)
    print(f"Veri klasörü  : {data_root}")
    print(f"Dosya sayısı  : {len(manifest)}")
    print(f"Özellik sayısı: {feature_count}")
    print(f"Parça boyutu  : {args.chunk_size:,}")
    print("=" * 78)

    # ------------------------------------------------------
    # Process every CSV
    # ------------------------------------------------------

    for manifest_row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="CSV değerleri doğrulanıyor",
        unit="dosya",
    ):
        relative_path = Path(
            str(manifest_row.relative_path)
        )

        file_path = (
            data_root
            / relative_path
        )

        expected_row_count = int(
            manifest_row.row_count
        )

        record: dict[str, Any] = {
            "device": str(
                manifest_row.device
            ),
            "attack_family": str(
                manifest_row.attack_family
            ),
            "attack_type": str(
                manifest_row.attack_type
            ),
            "class_label": str(
                manifest_row.class_label
            ),
            "relative_path": (
                relative_path.as_posix()
            ),
            "expected_row_count": (
                expected_row_count
            ),
            "actual_row_count": None,
            "row_count_matches": False,
            "column_count": None,
            "chunk_count": None,
            "missing_value_count": None,
            "positive_infinity_count": None,
            "negative_infinity_count": None,
            "non_finite_value_count": None,
            "status": "success",
            "error": None,
        }

        try:
            (
                file_result,
                column_result,
            ) = inspect_csv_file(
                file_path=file_path,
                expected_columns=expected_columns,
                chunk_size=args.chunk_size,
            )

            record.update(file_result)

            record["row_count_matches"] = bool(
                int(record["actual_row_count"])
                == expected_row_count
            )

            global_missing += (
                column_result["missing"]
            )

            global_positive_infinity += (
                column_result[
                    "positive_infinity"
                ]
            )

            global_negative_infinity += (
                column_result[
                    "negative_infinity"
                ]
            )

            global_finite += (
                column_result["finite"]
            )

            global_minimum = np.minimum(
                global_minimum,
                column_result["minimum"],
            )

            global_maximum = np.maximum(
                global_maximum,
                column_result["maximum"],
            )

        except Exception as error:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = (
                f"{type(error).__name__}: {error}"
            )

        file_records.append(record)

    # ------------------------------------------------------
    # File report
    # ------------------------------------------------------

    file_quality = pd.DataFrame(
        file_records
    )

    # ------------------------------------------------------
    # Column report
    # ------------------------------------------------------

    column_records: list[dict[str, Any]] = []

    for column_index, column_name in enumerate(
        expected_columns
    ):
        minimum = safe_float(
            global_minimum[column_index]
        )

        maximum = safe_float(
            global_maximum[column_index]
        )

        finite_value_count = int(
            global_finite[column_index]
        )

        is_constant = bool(
            finite_value_count > 0
            and minimum is not None
            and maximum is not None
            and minimum == maximum
        )

        column_records.append(
            {
                "column_index": int(column_index),
                "column_name": str(column_name),
                "finite_value_count": (
                    finite_value_count
                ),
                "missing_value_count": int(
                    global_missing[column_index]
                ),
                "positive_infinity_count": int(
                    global_positive_infinity[
                        column_index
                    ]
                ),
                "negative_infinity_count": int(
                    global_negative_infinity[
                        column_index
                    ]
                ),
                "non_finite_value_count": int(
                    global_missing[column_index]
                    + global_positive_infinity[
                        column_index
                    ]
                    + global_negative_infinity[
                        column_index
                    ]
                ),
                "minimum": minimum,
                "maximum": maximum,
                "is_constant": is_constant,
            }
        )

    column_quality = pd.DataFrame(
        column_records
    )

    # ------------------------------------------------------
    # Output files
    # ------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_report = (
        output_dir
        / "nbaiot_data_quality_by_file.csv"
    )

    column_report = (
        output_dir
        / "nbaiot_data_quality_by_column.csv"
    )

    summary_report = (
        output_dir
        / "nbaiot_data_quality_summary.json"
    )

    file_quality.to_csv(
        file_report,
        index=False,
        encoding="utf-8",
    )

    column_quality.to_csv(
        column_report,
        index=False,
        encoding="utf-8",
    )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    error_files = file_quality[
        file_quality["status"] == "error"
    ]

    successful_files = file_quality[
        file_quality["status"] == "success"
    ]

    row_mismatch_files = successful_files[
        successful_files[
            "row_count_matches"
        ] != True  # noqa: E712
    ]

    non_finite_files = successful_files[
        successful_files[
            "non_finite_value_count"
        ].fillna(0)
        > 0
    ]

    constant_columns = (
        column_quality.loc[
            column_quality["is_constant"],
            "column_name",
        ]
        .astype(str)
        .tolist()
    )

    total_missing_values = int(
        global_missing.sum()
    )

    total_positive_infinity_values = int(
        global_positive_infinity.sum()
    )

    total_negative_infinity_values = int(
        global_negative_infinity.sum()
    )

    total_non_finite_values = int(
        total_missing_values
        + total_positive_infinity_values
        + total_negative_infinity_values
    )

    total_actual_rows = int(
        successful_files[
            "actual_row_count"
        ]
        .fillna(0)
        .sum()
    )

    total_expected_rows = int(
        file_quality[
            "expected_row_count"
        ].sum()
    )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "data_root": str(data_root),
        "manifest_file": str(manifest_file),
        "schema_file": str(schema_file),
        "chunk_size": int(args.chunk_size),
        "file_count": int(
            len(file_quality)
        ),
        "successful_file_count": int(
            len(successful_files)
        ),
        "error_file_count": int(
            len(error_files)
        ),
        "row_count_mismatch_file_count": int(
            len(row_mismatch_files)
        ),
        "feature_count": int(
            feature_count
        ),
        "total_expected_rows": (
            total_expected_rows
        ),
        "total_actual_rows": (
            total_actual_rows
        ),
        "total_missing_values": (
            total_missing_values
        ),
        "total_positive_infinity_values": (
            total_positive_infinity_values
        ),
        "total_negative_infinity_values": (
            total_negative_infinity_values
        ),
        "total_non_finite_values": (
            total_non_finite_values
        ),
        "files_with_non_finite_values": int(
            len(non_finite_files)
        ),
        "constant_column_count": int(
            len(constant_columns)
        ),
        "constant_columns": (
            constant_columns
        ),
        "all_files_numeric": bool(
            len(error_files) == 0
        ),
        "all_row_counts_match": bool(
            len(row_mismatch_files) == 0
            and len(error_files) == 0
        ),
        "all_values_finite": bool(
            total_non_finite_values == 0
        ),
        "errors": (
            error_files[
                [
                    "relative_path",
                    "error",
                ]
            ].to_dict(
                orient="records"
            )
        ),
    }

    write_json_atomic(
        data=summary,
        output_file=summary_report,
    )

    # ------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------

    print()
    print("=" * 78)
    print("Veri kalite doğrulaması tamamlandı")
    print("=" * 78)
    print(
        "Başarılı dosya       : "
        f"{summary['successful_file_count']}"
    )
    print(
        "Hatalı dosya         : "
        f"{summary['error_file_count']}"
    )
    print(
        "Satır uyuşmazlığı    : "
        f"{summary['row_count_mismatch_file_count']}"
    )
    print(
        "Toplam satır         : "
        f"{summary['total_actual_rows']:,}"
    )
    print(
        "Eksik değer          : "
        f"{summary['total_missing_values']:,}"
    )
    print(
        "Pozitif sonsuz       : "
        f"{summary['total_positive_infinity_values']:,}"
    )
    print(
        "Negatif sonsuz       : "
        f"{summary['total_negative_infinity_values']:,}"
    )
    print(
        "Sabit özellik        : "
        f"{summary['constant_column_count']}"
    )
    print(
        "Tüm değerler sonlu   : "
        f"{summary['all_values_finite']}"
    )
    print(
        "Tüm dosyalar sayısal : "
        f"{summary['all_files_numeric']}"
    )
    print(
        "Satır sayıları doğru : "
        f"{summary['all_row_counts_match']}"
    )
    print()
    print(f"Dosya raporu : {file_report}")
    print(f"Sütun raporu : {column_report}")
    print(f"JSON özet    : {summary_report}")
    print("=" * 78)


if __name__ == "__main__":
    main()