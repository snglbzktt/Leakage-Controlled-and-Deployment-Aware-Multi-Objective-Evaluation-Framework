"""
N-BaIoT family_3 bütün ana varyantların birleşik istatistiksel analizi.

Birleştirilen ana varyantlar:
- B0
- P25, P50, P75
- DQ, PTQ, QAT
- P25-QAT, P50-QAT, P75-QAT

Toplam:
- 3 model
- 10 varyant
- 5 eşleştirilmiş seed
- 150 ana test sonucu
- 450 sınıf bazlı sonuç

Analizler:
1. Ana ve güvenlik metriklerinin birleşik doğrulanması
2. Friedman omnibus testi
3. Kendall's W etki büyüklüğü
4. Eşleştirilmiş iki yönlü Wilcoxon signed-rank testi
5. Model ve sonuç değişkeni başına 45 karşılaştırmada Holm düzeltmesi
6. Eşleştirilmiş rank-biserial correlation
7. B0'a göre eşleştirilmiş performans farkları
8. Boyut, MAC, gecikme ve throughput ödünleşimi
9. Model içi ve global Pareto önü

Bilimsel sınırlar:
- Test sonuçları eğitim veya model seçiminde kullanılmaz.
- Verimlilik ölçümleri yalnızca mevcut host CPU ve runtime ortamına aittir.
- MCU gecikmesi, enerji tüketimi veya gerçek donanım dağıtımı iddia edilmez.
- Beş seed nedeniyle Wilcoxon test gücü sınırlıdır; p-değerleri,
  etki büyüklükleri ve eşleştirilmiş farklarla birlikte yorumlanır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import (
    friedmanchisquare,
    rankdata,
    t as student_t,
    wilcoxon,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

DEFAULT_FP32_RUN_FILE = (
    DEFAULT_REPORT_DIRECTORY
    / "nbaiot_family3_fp32_baseline_runs.csv"
)

DEFAULT_PRUNING_RUN_FILE = (
    DEFAULT_REPORT_DIRECTORY
    / "nbaiot_family3_structured_pruning_runs.csv"
)

DEFAULT_QUANTIZATION_RUN_FILE = (
    DEFAULT_REPORT_DIRECTORY
    / "nbaiot_family3_quantization_runs.csv"
)

DEFAULT_PRUNING_QAT_RUN_FILE = (
    DEFAULT_REPORT_DIRECTORY
    / "nbaiot_family3_pruning_qat_runs.csv"
)

DEFAULT_FP32_PER_CLASS_FILE = (
    DEFAULT_REPORT_DIRECTORY
    / "nbaiot_family3_fp32_baseline_per_class_runs.csv"
)

DEFAULT_PRUNING_PER_CLASS_FILE = (
    DEFAULT_REPORT_DIRECTORY
    / "nbaiot_family3_structured_pruning_per_class_runs.csv"
)

DEFAULT_QUANTIZATION_PER_CLASS_FILE = (
    DEFAULT_REPORT_DIRECTORY
    / "nbaiot_family3_quantization_per_class_runs.csv"
)

DEFAULT_PRUNING_QAT_PER_CLASS_FILE = (
    DEFAULT_REPORT_DIRECTORY
    / "nbaiot_family3_pruning_qat_per_class_runs.csv"
)

DEFAULT_EFFICIENCY_RUN_FILE = (
    DEFAULT_REPORT_DIRECTORY
    / "nbaiot_family3_efficiency_profile_runs.csv"
)

MODEL_NAMES = (
    "tinyml_mlp",
    "compact_dnn",
    "tiny_1d_cnn",
)

VARIANT_NAMES = (
    "B0",
    "P25",
    "P50",
    "P75",
    "DQ",
    "PTQ",
    "QAT",
    "P25-QAT",
    "P50-QAT",
    "P75-QAT",
)

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

TARGET_CLASSES = (
    "benign",
    "gafgyt",
    "mirai",
)

GLOBAL_ENDPOINTS: dict[str, dict[str, str]] = {
    "test_macro_f1": {
        "display_name": "Test Macro F1",
        "endpoint_family": "primary",
        "direction": "higher_is_better",
    },
    "test_mcc": {
        "display_name": "Test MCC",
        "endpoint_family": "secondary_global",
        "direction": "higher_is_better",
    },
    "test_balanced_accuracy": {
        "display_name": "Test Balanced Accuracy",
        "endpoint_family": "secondary_global",
        "direction": "higher_is_better",
    },
}

STANDARD_GLOBAL_COLUMNS = (
    "test_loss",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
    "test_mcc",
)

ALPHA = 0.05


def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "B0, budama, quantization ve budama+QAT "
            "sonuçlarını birleşik olarak analiz eder."
        )
    )

    parser.add_argument(
        "--fp32-run-file",
        type=Path,
        default=DEFAULT_FP32_RUN_FILE,
    )

    parser.add_argument(
        "--pruning-run-file",
        type=Path,
        default=DEFAULT_PRUNING_RUN_FILE,
    )

    parser.add_argument(
        "--quantization-run-file",
        type=Path,
        default=DEFAULT_QUANTIZATION_RUN_FILE,
    )

    parser.add_argument(
        "--pruning-qat-run-file",
        type=Path,
        default=DEFAULT_PRUNING_QAT_RUN_FILE,
    )

    parser.add_argument(
        "--fp32-per-class-file",
        type=Path,
        default=DEFAULT_FP32_PER_CLASS_FILE,
    )

    parser.add_argument(
        "--pruning-per-class-file",
        type=Path,
        default=DEFAULT_PRUNING_PER_CLASS_FILE,
    )

    parser.add_argument(
        "--quantization-per-class-file",
        type=Path,
        default=DEFAULT_QUANTIZATION_PER_CLASS_FILE,
    )

    parser.add_argument(
        "--pruning-qat-per-class-file",
        type=Path,
        default=DEFAULT_PRUNING_QAT_PER_CLASS_FILE,
    )

    parser.add_argument(
        "--efficiency-run-file",
        type=Path,
        default=DEFAULT_EFFICIENCY_RUN_FILE,
    )

    parser.add_argument(
        "--report-directory",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


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
    document: dict[str, Any],
    output_file: Path,
) -> None:
    """JSON dosyasını atomik biçimde yazar."""

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
            document,
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
    """CSV dosyasını atomik biçimde yazar."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    if not file_path.exists():
        raise FileNotFoundError(
            f"SHA-256 girdisi bulunamadı: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        while True:
            block = file_handle.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def require_file(file_path: Path) -> Path:
    """Girdi dosyasının bulunduğunu doğrular."""

    resolved = file_path.resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            f"Girdi dosyası bulunamadı: {resolved}"
        )

    return resolved


def normalize_boolean_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """Boolean sütununu güvenli biçimde dönüştürür."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if normalized.isna().any():
        invalid = sorted(
            series[
                normalized.isna()
            ].astype(str).unique().tolist()
        )

        raise ValueError(
            f"{column_name} içinde geçersiz boolean değerleri var: "
            + ", ".join(invalid)
        )

    return normalized.astype(bool)


def validate_optional_test_policy(
    frame: pd.DataFrame,
    source_name: str,
) -> None:
    """Varsa test değerlendirme politikası sütunlarını doğrular."""

    if "test_evaluation_attempt_count" in frame.columns:
        attempts = pd.to_numeric(
            frame["test_evaluation_attempt_count"],
            errors="raise",
        ).astype("int64")

        if (attempts != 1).any():
            raise RuntimeError(
                f"{source_name}: test değerlendirme girişimi 1 olmayan kayıt var."
            )

    if "test_used_for_selection" in frame.columns:
        flags = normalize_boolean_series(
            frame["test_used_for_selection"],
            f"{source_name}/test_used_for_selection",
        )

        if flags.any():
            raise RuntimeError(
                f"{source_name}: test sonucu model seçimine katılmış."
            )


def coerce_numeric_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    source_name: str,
) -> None:
    """Belirtilen sütunları sonlu sayısal değerlere dönüştürür."""

    for column_name in columns:
        frame[column_name] = pd.to_numeric(
            frame[column_name],
            errors="raise",
        )

        values = frame[column_name].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{source_name}/{column_name} içinde sonlu olmayan değer var."
            )


def expected_global_keys() -> set[tuple[str, str, int]]:
    """Beklenen 150 model-varyant-seed anahtarını üretir."""

    return {
        (
            model_name,
            variant_name,
            seed,
        )
        for model_name in MODEL_NAMES
        for variant_name in VARIANT_NAMES
        for seed in SEEDS
    }


def expected_per_class_keys() -> set[
    tuple[str, str, int, str]
]:
    """Beklenen 450 sınıf bazlı anahtarı üretir."""

    return {
        (
            model_name,
            variant_name,
            seed,
            class_name,
        )
        for model_name in MODEL_NAMES
        for variant_name in VARIANT_NAMES
        for seed in SEEDS
        for class_name in TARGET_CLASSES
    }


def standardize_global_source(
    *,
    input_file: Path,
    source_name: str,
    variant_column: str | None,
    fixed_variant: str | None,
    expected_count: int,
) -> pd.DataFrame:
    """Bir ana sonuç dosyasını ortak formata dönüştürür."""

    frame = pd.read_csv(
        require_file(input_file)
    )

    required_columns = {
        "model_name",
        "seed",
        *STANDARD_GLOBAL_COLUMNS,
    }

    if variant_column is not None:
        required_columns.add(
            variant_column
        )

    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(
            f"{source_name} dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing))
        )

    if len(frame) != expected_count:
        raise ValueError(
            f"{source_name} kayıt sayısı {expected_count} değil: "
            f"{len(frame)}"
        )

    frame = frame.copy()

    frame["model_name"] = frame[
        "model_name"
    ].astype(str)

    frame["seed"] = pd.to_numeric(
        frame["seed"],
        errors="raise",
    ).astype("int64")

    if fixed_variant is not None:
        frame["variant_name"] = fixed_variant

    elif variant_column is not None:
        frame["variant_name"] = frame[
            variant_column
        ].astype(str)

    else:
        raise RuntimeError(
            "Varyant sütunu veya sabit varyant belirtilmedi."
        )

    frame["source_family"] = source_name

    validate_optional_test_policy(
        frame,
        source_name,
    )

    coerce_numeric_columns(
        frame,
        STANDARD_GLOBAL_COLUMNS,
        source_name,
    )

    selected_columns = [
        "model_name",
        "variant_name",
        "seed",
        "source_family",
        *STANDARD_GLOBAL_COLUMNS,
    ]

    return frame[
        selected_columns
    ].copy()


def load_all_global_results(
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Dört deney ailesini 150 satırlık ortak tabloya birleştirir."""

    fp32 = standardize_global_source(
        input_file=args.fp32_run_file,
        source_name="fp32_baseline",
        variant_column=None,
        fixed_variant="B0",
        expected_count=15,
    )

    pruning = standardize_global_source(
        input_file=args.pruning_run_file,
        source_name="structured_pruning",
        variant_column="pruning_name",
        fixed_variant=None,
        expected_count=45,
    )

    quantization = standardize_global_source(
        input_file=args.quantization_run_file,
        source_name="quantization",
        variant_column="method_name",
        fixed_variant=None,
        expected_count=45,
    )

    pruning_qat = standardize_global_source(
        input_file=args.pruning_qat_run_file,
        source_name="pruning_qat",
        variant_column="variant_name",
        fixed_variant=None,
        expected_count=45,
    )

    combined = pd.concat(
        [
            fp32,
            pruning,
            quantization,
            pruning_qat,
        ],
        ignore_index=True,
    )

    if len(combined) != 150:
        raise RuntimeError(
            f"Birleşik ana sonuç sayısı 150 değil: {len(combined)}"
        )

    if combined.duplicated(
        subset=[
            "model_name",
            "variant_name",
            "seed",
        ]
    ).any():
        raise RuntimeError(
            "Birleşik ana sonuçlarda tekrarlanan anahtar var."
        )

    actual_keys = {
        (
            str(row.model_name),
            str(row.variant_name),
            int(row.seed),
        )
        for row in combined.itertuples(
            index=False
        )
    }

    expected_keys = expected_global_keys()

    if actual_keys != expected_keys:
        raise RuntimeError(
            "Birleşik ana model-varyant-seed matrisi eksik veya beklenmeyen kayıt içeriyor. "
            f"Eksik={len(expected_keys - actual_keys)}, "
            f"beklenmeyen={len(actual_keys - expected_keys)}"
        )

    return combined.sort_values(
        by=[
            "model_name",
            "variant_name",
            "seed",
        ]
    ).reset_index(
        drop=True
    )


def standardize_per_class_source(
    *,
    input_file: Path,
    source_name: str,
    variant_column: str | None,
    fixed_variant: str | None,
    expected_count: int,
) -> pd.DataFrame:
    """Sınıf bazlı sonuç dosyasını ortak formata dönüştürür."""

    frame = pd.read_csv(
        require_file(input_file)
    )

    required_columns = {
        "model_name",
        "seed",
        "class_name",
        "precision",
        "recall",
        "f1",
        "false_negative_rate",
    }

    if variant_column is not None:
        required_columns.add(
            variant_column
        )

    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(
            f"{source_name} sınıf dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing))
        )

    if len(frame) != expected_count:
        raise ValueError(
            f"{source_name} sınıf kayıt sayısı {expected_count} değil: "
            f"{len(frame)}"
        )

    frame = frame.copy()

    frame["model_name"] = frame[
        "model_name"
    ].astype(str)

    frame["seed"] = pd.to_numeric(
        frame["seed"],
        errors="raise",
    ).astype("int64")

    frame["class_name"] = frame[
        "class_name"
    ].astype(str)

    if fixed_variant is not None:
        frame["variant_name"] = fixed_variant

    elif variant_column is not None:
        frame["variant_name"] = frame[
            variant_column
        ].astype(str)

    else:
        raise RuntimeError(
            "Sınıf sonucu için varyant tanımı yok."
        )

    frame["source_family"] = source_name

    numeric_columns = (
        "precision",
        "recall",
        "f1",
        "false_negative_rate",
    )

    coerce_numeric_columns(
        frame,
        numeric_columns,
        source_name,
    )

    if (
        frame["false_negative_rate"] < 0
    ).any() or (
        frame["false_negative_rate"] > 1
    ).any():
        raise ValueError(
            f"{source_name}: FNR [0, 1] aralığı dışında."
        )

    return frame[
        [
            "model_name",
            "variant_name",
            "seed",
            "class_name",
            "source_family",
            *numeric_columns,
        ]
    ].copy()


def load_all_per_class_results(
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Dört deney ailesinin sınıf sonuçlarını birleştirir."""

    fp32 = standardize_per_class_source(
        input_file=args.fp32_per_class_file,
        source_name="fp32_baseline",
        variant_column=None,
        fixed_variant="B0",
        expected_count=45,
    )

    pruning = standardize_per_class_source(
        input_file=args.pruning_per_class_file,
        source_name="structured_pruning",
        variant_column="pruning_name",
        fixed_variant=None,
        expected_count=135,
    )

    quantization = standardize_per_class_source(
        input_file=args.quantization_per_class_file,
        source_name="quantization",
        variant_column="method_name",
        fixed_variant=None,
        expected_count=135,
    )

    pruning_qat = standardize_per_class_source(
        input_file=args.pruning_qat_per_class_file,
        source_name="pruning_qat",
        variant_column="variant_name",
        fixed_variant=None,
        expected_count=135,
    )

    combined = pd.concat(
        [
            fp32,
            pruning,
            quantization,
            pruning_qat,
        ],
        ignore_index=True,
    )

    if len(combined) != 450:
        raise RuntimeError(
            f"Birleşik sınıf sonucu sayısı 450 değil: {len(combined)}"
        )

    if combined.duplicated(
        subset=[
            "model_name",
            "variant_name",
            "seed",
            "class_name",
        ]
    ).any():
        raise RuntimeError(
            "Birleşik sınıf sonuçlarında tekrarlanan anahtar var."
        )

    actual_keys = {
        (
            str(row.model_name),
            str(row.variant_name),
            int(row.seed),
            str(row.class_name),
        )
        for row in combined.itertuples(
            index=False
        )
    }

    expected_keys = expected_per_class_keys()

    if actual_keys != expected_keys:
        raise RuntimeError(
            "Birleşik sınıf matrisi eksik veya beklenmeyen kayıt içeriyor. "
            f"Eksik={len(expected_keys - actual_keys)}, "
            f"beklenmeyen={len(actual_keys - expected_keys)}"
        )

    return combined.sort_values(
        by=[
            "model_name",
            "variant_name",
            "seed",
            "class_name",
        ]
    ).reset_index(
        drop=True
    )


def load_efficiency_results(
    args: argparse.Namespace,
    global_results: pd.DataFrame,
) -> pd.DataFrame:
    """31. aşama ana profillerini yükler ve test sonuçlarıyla eşler."""

    frame = pd.read_csv(
        require_file(
            args.efficiency_run_file
        )
    )

    required_columns = {
        "model_name",
        "variant_name",
        "seed",
        "analysis_role",
        "runtime_name",
        "precision",
        "parameter_count",
        "weight_layer_macs",
        "deployment_artifact_size_bytes",
        "test_macro_f1",
        "latency_median_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "throughput_samples_per_second",
    }

    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(
            "Verimlilik dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing))
        )

    frame = frame.copy()

    frame["analysis_role"] = frame[
        "analysis_role"
    ].astype(str)

    frame = frame[
        frame["analysis_role"]
        == "primary"
    ].copy()

    if len(frame) != 150:
        raise RuntimeError(
            f"Ana verimlilik profil sayısı 150 değil: {len(frame)}"
        )

    frame["model_name"] = frame[
        "model_name"
    ].astype(str)

    frame["variant_name"] = frame[
        "variant_name"
    ].astype(str)

    frame["seed"] = pd.to_numeric(
        frame["seed"],
        errors="raise",
    ).astype("int64")

    numeric_columns = (
        "parameter_count",
        "weight_layer_macs",
        "deployment_artifact_size_bytes",
        "test_macro_f1",
        "latency_median_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "throughput_samples_per_second",
    )

    coerce_numeric_columns(
        frame,
        numeric_columns,
        "efficiency_profile",
    )

    if frame.duplicated(
        subset=[
            "model_name",
            "variant_name",
            "seed",
        ]
    ).any():
        raise RuntimeError(
            "Verimlilik profillerinde tekrarlanan anahtar var."
        )

    actual_keys = {
        (
            str(row.model_name),
            str(row.variant_name),
            int(row.seed),
        )
        for row in frame.itertuples(
            index=False
        )
    }

    expected_keys = expected_global_keys()

    if actual_keys != expected_keys:
        raise RuntimeError(
            "Verimlilik matrisi 150 ana sonuçla uyuşmuyor."
        )

    policy_checks = {
        "train_split_used": False,
        "validation_split_used": False,
        "test_split_used_for_profiling": False,
        "host_cpu_profile_only": True,
        "mcu_measurement": False,
    }

    for column_name, expected_value in policy_checks.items():
        if column_name not in frame.columns:
            continue

        flags = normalize_boolean_series(
            frame[column_name],
            f"efficiency/{column_name}",
        )

        if not bool(
            (flags == expected_value).all()
        ):
            raise RuntimeError(
                f"Verimlilik politikası uyuşmuyor: {column_name}"
            )

    reference = global_results[
        [
            "model_name",
            "variant_name",
            "seed",
            "test_macro_f1",
        ]
    ].rename(
        columns={
            "test_macro_f1": "reference_test_macro_f1",
        }
    )

    merged = frame.merge(
        reference,
        on=[
            "model_name",
            "variant_name",
            "seed",
        ],
        how="left",
        validate="one_to_one",
    )

    differences = np.abs(
        merged["test_macro_f1"].to_numpy(
            dtype=np.float64
        )
        - merged[
            "reference_test_macro_f1"
        ].to_numpy(
            dtype=np.float64
        )
    )

    if not np.all(
        differences <= 1e-10
    ):
        raise RuntimeError(
            "Verimlilik raporundaki Macro F1 değerleri ana sonuçlarla uyuşmuyor. "
            f"Maksimum fark={differences.max():.12g}"
        )

    return merged.sort_values(
        by=[
            "model_name",
            "variant_name",
            "seed",
        ]
    ).reset_index(
        drop=True
    )


def calculate_summary_statistics(
    values: np.ndarray,
) -> dict[str, float | int]:
    """Özet istatistik ve iki yönlü %95 Student-t GA hesaplar."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "İstatistik dizisi tek boyutlu ve boş olmayan olmalıdır."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "İstatistik dizisinde sonlu olmayan değer var."
        )

    sample_count = int(
        len(values)
    )

    mean_value = float(
        values.mean()
    )

    median_value = float(
        np.median(values)
    )

    minimum_value = float(
        values.min()
    )

    maximum_value = float(
        values.max()
    )

    if sample_count > 1:
        standard_deviation = float(
            values.std(ddof=1)
        )

        standard_error = float(
            standard_deviation
            / np.sqrt(sample_count)
        )

        critical_value = float(
            student_t.ppf(
                0.975,
                df=sample_count - 1,
            )
        )

        margin = float(
            critical_value
            * standard_error
        )

    else:
        standard_deviation = 0.0
        standard_error = 0.0
        margin = 0.0

    return {
        "sample_count": sample_count,
        "mean": mean_value,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "median": median_value,
        "minimum": minimum_value,
        "maximum": maximum_value,
        "ci95_lower": float(
            mean_value - margin
        ),
        "ci95_upper": float(
            mean_value + margin
        ),
    }


def holm_adjust(
    p_values: list[float],
) -> list[float]:
    """Holm step-down düzeltmesini uygular."""

    if not p_values:
        return []

    values = np.asarray(
        p_values,
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Holm düzeltmesinde sonlu olmayan p-değeri var."
        )

    if np.any(values < 0) or np.any(values > 1):
        raise ValueError(
            "P-değerleri [0, 1] aralığında olmalıdır."
        )

    count = len(values)

    order = np.argsort(
        values,
        kind="mergesort",
    )

    sorted_values = values[
        order
    ]

    adjusted_sorted = np.empty(
        count,
        dtype=np.float64,
    )

    running_maximum = 0.0

    for index, raw_p_value in enumerate(
        sorted_values
    ):
        candidate = min(
            1.0,
            float(raw_p_value)
            * (
                count - index
            ),
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted_sorted[
            index
        ] = running_maximum

    adjusted = np.empty(
        count,
        dtype=np.float64,
    )

    adjusted[
        order
    ] = adjusted_sorted

    return adjusted.tolist()


def safe_friedman_test(
    variant_arrays: list[np.ndarray],
) -> tuple[float, float]:
    """Bütün varyantlar aynıysa güvenli Friedman sonucu üretir."""

    stacked = np.column_stack(
        variant_arrays
    )

    if np.allclose(
        stacked,
        stacked[:, [0]],
        rtol=0.0,
        atol=0.0,
    ):
        return 0.0, 1.0

    result = friedmanchisquare(
        *variant_arrays
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def safe_wilcoxon(
    advantage_values: np.ndarray,
) -> tuple[float, float, str]:
    """Bütün eşleştirilmiş farklar sıfırsa güvenli sonuç üretir."""

    differences = np.asarray(
        advantage_values,
        dtype=np.float64,
    )

    if np.allclose(
        differences,
        0.0,
        rtol=0.0,
        atol=1e-15,
    ):
        return (
            0.0,
            1.0,
            "all_paired_differences_zero",
        )

    result = wilcoxon(
        differences,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="auto",
    )

    return (
        float(result.statistic),
        float(result.pvalue),
        "scipy_auto",
    )


def interpret_kendall_w(
    value: float,
) -> str:
    """Kendall W büyüklüğünü sınıflandırır."""

    if value < 0.10:
        return "negligible"

    if value < 0.30:
        return "small"

    if value < 0.50:
        return "moderate"

    return "large"


def interpret_rank_biserial(
    value: float,
) -> str:
    """Rank-biserial korelasyonu sınıflandırır."""

    absolute_value = abs(
        float(value)
    )

    if absolute_value < 0.10:
        return "negligible"

    if absolute_value < 0.30:
        return "small"

    if absolute_value < 0.50:
        return "medium"

    return "large"


def calculate_rank_biserial(
    advantage_values: np.ndarray,
) -> dict[str, Any]:
    """Eşleştirilmiş rank-biserial correlation hesaplar."""

    differences = np.asarray(
        advantage_values,
        dtype=np.float64,
    )

    nonzero = differences[
        ~np.isclose(
            differences,
            0.0,
            rtol=0.0,
            atol=1e-15,
        )
    ]

    if len(nonzero) == 0:
        return {
            "rank_biserial_correlation": 0.0,
            "effect_magnitude": "negligible",
            "positive_rank_sum": 0.0,
            "negative_rank_sum": 0.0,
            "effective_nonzero_pair_count": 0,
        }

    ranks = rankdata(
        np.abs(nonzero),
        method="average",
    )

    positive_sum = float(
        ranks[
            nonzero > 0
        ].sum()
    )

    negative_sum = float(
        ranks[
            nonzero < 0
        ].sum()
    )

    total = (
        positive_sum
        + negative_sum
    )

    effect = float(
        (
            positive_sum
            - negative_sum
        )
        / total
    )

    return {
        "rank_biserial_correlation": effect,
        "effect_magnitude": interpret_rank_biserial(
            effect
        ),
        "positive_rank_sum": positive_sum,
        "negative_rank_sum": negative_sum,
        "effective_nonzero_pair_count": int(
            len(nonzero)
        ),
    }


def build_paired_matrix(
    frame: pd.DataFrame,
    *,
    model_name: str,
    value_column: str,
) -> pd.DataFrame:
    """Seed satırlı, varyant sütunlu eşleştirilmiş matris oluşturur."""

    subset = frame[
        frame["model_name"]
        == model_name
    ]

    matrix = subset.pivot(
        index="seed",
        columns="variant_name",
        values=value_column,
    )

    matrix = matrix.reindex(
        index=list(SEEDS),
        columns=list(VARIANT_NAMES),
    )

    if matrix.isna().any().any():
        raise RuntimeError(
            f"{model_name}/{value_column}: eşleştirilmiş matris eksik."
        )

    return matrix


def calculate_average_ranks(
    matrix: pd.DataFrame,
    direction: str,
) -> dict[str, float]:
    """Her seed içinde varyant ranklarını hesaplar."""

    rank_rows: list[np.ndarray] = []

    for row in matrix.to_numpy(
        dtype=np.float64
    ):
        if direction == "higher_is_better":
            ranks = rankdata(
                -row,
                method="average",
            )

        elif direction == "lower_is_better":
            ranks = rankdata(
                row,
                method="average",
            )

        else:
            raise ValueError(
                f"Geçersiz yön: {direction}"
            )

        rank_rows.append(ranks)

    mean_ranks = np.vstack(
        rank_rows
    ).mean(
        axis=0
    )

    return {
        variant_name: float(
            mean_ranks[index]
        )
        for index, variant_name in enumerate(
            VARIANT_NAMES
        )
    }


def analyze_endpoint(
    *,
    model_name: str,
    endpoint_id: str,
    endpoint_display_name: str,
    endpoint_family: str,
    direction: str,
    paired_matrix: pd.DataFrame,
    alpha: float,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Tek model ve sonuç değişkeni için tüm testleri çalıştırır."""

    variant_arrays = [
        paired_matrix[
            variant_name
        ].to_numpy(
            dtype=np.float64
        )
        for variant_name in VARIANT_NAMES
    ]

    friedman_statistic, friedman_p_value = (
        safe_friedman_test(
            variant_arrays
        )
    )

    subject_count = len(
        paired_matrix
    )

    treatment_count = len(
        VARIANT_NAMES
    )

    kendall_w = float(
        friedman_statistic
        / (
            subject_count
            * (
                treatment_count - 1
            )
        )
    )

    kendall_w = float(
        np.clip(
            kendall_w,
            0.0,
            1.0,
        )
    )

    friedman_significant = bool(
        friedman_p_value < alpha
    )

    average_ranks = calculate_average_ranks(
        paired_matrix,
        direction,
    )

    omnibus_record = {
        "model_name": model_name,
        "endpoint_id": endpoint_id,
        "endpoint_display_name": endpoint_display_name,
        "endpoint_family": endpoint_family,
        "direction": direction,
        "seed_count": int(subject_count),
        "variant_count": int(treatment_count),
        "friedman_chi_square": friedman_statistic,
        "friedman_degrees_of_freedom": int(
            treatment_count - 1
        ),
        "friedman_p_value": friedman_p_value,
        "alpha": float(alpha),
        "friedman_significant": friedman_significant,
        "kendall_w": kendall_w,
        "kendall_w_magnitude": interpret_kendall_w(
            kendall_w
        ),
    }

    descriptive_records: list[
        dict[str, Any]
    ] = []

    for variant_name in VARIANT_NAMES:
        statistics = calculate_summary_statistics(
            paired_matrix[
                variant_name
            ].to_numpy(
                dtype=np.float64
            )
        )

        descriptive_records.append(
            {
                "model_name": model_name,
                "endpoint_id": endpoint_id,
                "endpoint_display_name": endpoint_display_name,
                "endpoint_family": endpoint_family,
                "direction": direction,
                "variant_name": variant_name,
                "average_rank": average_ranks[
                    variant_name
                ],
                **statistics,
            }
        )

    pairwise_records: list[
        dict[str, Any]
    ] = []

    raw_p_values: list[float] = []

    for variant_a, variant_b in combinations(
        VARIANT_NAMES,
        2,
    ):
        values_a = paired_matrix[
            variant_a
        ].to_numpy(
            dtype=np.float64
        )

        values_b = paired_matrix[
            variant_b
        ].to_numpy(
            dtype=np.float64
        )

        raw_difference = (
            values_a
            - values_b
        )

        if direction == "higher_is_better":
            advantage = raw_difference

        else:
            advantage = -raw_difference

        (
            wilcoxon_statistic,
            raw_p_value,
            method_note,
        ) = safe_wilcoxon(
            advantage
        )

        effect = calculate_rank_biserial(
            advantage
        )

        win_count = int(
            np.sum(
                advantage > 1e-15
            )
        )

        loss_count = int(
            np.sum(
                advantage < -1e-15
            )
        )

        tie_count = int(
            len(advantage)
            - win_count
            - loss_count
        )

        raw_p_values.append(
            raw_p_value
        )

        pairwise_records.append(
            {
                "model_name": model_name,
                "endpoint_id": endpoint_id,
                "endpoint_display_name": endpoint_display_name,
                "endpoint_family": endpoint_family,
                "direction": direction,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "paired_seed_count": int(
                    len(advantage)
                ),
                "variant_a_mean": float(
                    values_a.mean()
                ),
                "variant_b_mean": float(
                    values_b.mean()
                ),
                "mean_metric_difference_a_minus_b": float(
                    raw_difference.mean()
                ),
                "median_metric_difference_a_minus_b": float(
                    np.median(
                        raw_difference
                    )
                ),
                "mean_advantage_a_over_b": float(
                    advantage.mean()
                ),
                "wilcoxon_statistic": wilcoxon_statistic,
                "wilcoxon_raw_p_value": raw_p_value,
                "wilcoxon_method": method_note,
                **effect,
                "variant_a_seed_wins": win_count,
                "seed_ties": tie_count,
                "variant_a_seed_losses": loss_count,
                "friedman_p_value": friedman_p_value,
                "friedman_significant": friedman_significant,
            }
        )

    adjusted_p_values = holm_adjust(
        raw_p_values
    )

    for record, adjusted_p_value in zip(
        pairwise_records,
        adjusted_p_values,
        strict=True,
    ):
        record[
            "holm_adjusted_p_value"
        ] = float(
            adjusted_p_value
        )

        record[
            "holm_significant"
        ] = bool(
            adjusted_p_value < alpha
        )

        record[
            "gatekept_significant"
        ] = bool(
            friedman_significant
            and adjusted_p_value < alpha
        )

        if not friedman_significant:
            status = (
                "not_confirmatory_because_friedman_not_significant"
            )

        elif adjusted_p_value < alpha:
            status = "significant_after_holm"

        else:
            status = "not_significant_after_holm"

        record[
            "interpretation_status"
        ] = status

    return (
        omnibus_record,
        pairwise_records,
        descriptive_records,
    )


def build_b0_delta_records(
    *,
    model_name: str,
    endpoint_id: str,
    endpoint_display_name: str,
    endpoint_family: str,
    direction: str,
    paired_matrix: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Her sıkıştırılmış varyantın B0'a göre farkını hesaplar."""

    baseline_values = paired_matrix[
        "B0"
    ].to_numpy(
        dtype=np.float64
    )

    records: list[
        dict[str, Any]
    ] = []

    for variant_name in VARIANT_NAMES:
        if variant_name == "B0":
            continue

        variant_values = paired_matrix[
            variant_name
        ].to_numpy(
            dtype=np.float64
        )

        raw_delta = (
            variant_values
            - baseline_values
        )

        if direction == "higher_is_better":
            advantage = raw_delta

        else:
            advantage = -raw_delta

        raw_statistics = calculate_summary_statistics(
            raw_delta
        )

        advantage_statistics = calculate_summary_statistics(
            advantage
        )

        improvements = int(
            np.sum(
                advantage > 1e-15
            )
        )

        degradations = int(
            np.sum(
                advantage < -1e-15
            )
        )

        ties = int(
            len(advantage)
            - improvements
            - degradations
        )

        records.append(
            {
                "model_name": model_name,
                "endpoint_id": endpoint_id,
                "endpoint_display_name": endpoint_display_name,
                "endpoint_family": endpoint_family,
                "direction": direction,
                "baseline_variant": "B0",
                "variant_name": variant_name,
                "paired_seed_count": int(
                    len(raw_delta)
                ),
                "mean_raw_delta_variant_minus_b0": raw_statistics[
                    "mean"
                ],
                "raw_delta_standard_deviation": raw_statistics[
                    "standard_deviation"
                ],
                "raw_delta_median": raw_statistics[
                    "median"
                ],
                "raw_delta_minimum": raw_statistics[
                    "minimum"
                ],
                "raw_delta_maximum": raw_statistics[
                    "maximum"
                ],
                "raw_delta_ci95_lower": raw_statistics[
                    "ci95_lower"
                ],
                "raw_delta_ci95_upper": raw_statistics[
                    "ci95_upper"
                ],
                "mean_performance_advantage": advantage_statistics[
                    "mean"
                ],
                "performance_advantage_ci95_lower": advantage_statistics[
                    "ci95_lower"
                ],
                "performance_advantage_ci95_upper": advantage_statistics[
                    "ci95_upper"
                ],
                "variant_seed_improvements": improvements,
                "seed_ties": ties,
                "variant_seed_degradations": degradations,
            }
        )

    return records


def build_statistical_outputs(
    *,
    global_results: pd.DataFrame,
    per_class_results: pd.DataFrame,
    alpha: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Bütün sonuç değişkenleri için istatistik tablolarını üretir."""

    omnibus_records: list[
        dict[str, Any]
    ] = []

    pairwise_records: list[
        dict[str, Any]
    ] = []

    descriptive_records: list[
        dict[str, Any]
    ] = []

    delta_records: list[
        dict[str, Any]
    ] = []

    for model_name in MODEL_NAMES:
        for endpoint_id, metadata in (
            GLOBAL_ENDPOINTS.items()
        ):
            matrix = build_paired_matrix(
                global_results,
                model_name=model_name,
                value_column=endpoint_id,
            )

            (
                omnibus_record,
                endpoint_pairwise,
                endpoint_descriptives,
            ) = analyze_endpoint(
                model_name=model_name,
                endpoint_id=endpoint_id,
                endpoint_display_name=metadata[
                    "display_name"
                ],
                endpoint_family=metadata[
                    "endpoint_family"
                ],
                direction=metadata[
                    "direction"
                ],
                paired_matrix=matrix,
                alpha=alpha,
            )

            omnibus_records.append(
                omnibus_record
            )

            pairwise_records.extend(
                endpoint_pairwise
            )

            descriptive_records.extend(
                endpoint_descriptives
            )

            delta_records.extend(
                build_b0_delta_records(
                    model_name=model_name,
                    endpoint_id=endpoint_id,
                    endpoint_display_name=metadata[
                        "display_name"
                    ],
                    endpoint_family=metadata[
                        "endpoint_family"
                    ],
                    direction=metadata[
                        "direction"
                    ],
                    paired_matrix=matrix,
                )
            )

        for class_name in TARGET_CLASSES:
            class_frame = per_class_results[
                per_class_results[
                    "class_name"
                ]
                == class_name
            ]

            endpoint_id = (
                f"{class_name}_false_negative_rate"
            )

            endpoint_display_name = (
                f"{class_name} False Negative Rate"
            )

            matrix = build_paired_matrix(
                class_frame,
                model_name=model_name,
                value_column=(
                    "false_negative_rate"
                ),
            )

            (
                omnibus_record,
                endpoint_pairwise,
                endpoint_descriptives,
            ) = analyze_endpoint(
                model_name=model_name,
                endpoint_id=endpoint_id,
                endpoint_display_name=endpoint_display_name,
                endpoint_family=(
                    "secondary_security"
                ),
                direction="lower_is_better",
                paired_matrix=matrix,
                alpha=alpha,
            )

            omnibus_records.append(
                omnibus_record
            )

            pairwise_records.extend(
                endpoint_pairwise
            )

            descriptive_records.extend(
                endpoint_descriptives
            )

            delta_records.extend(
                build_b0_delta_records(
                    model_name=model_name,
                    endpoint_id=endpoint_id,
                    endpoint_display_name=endpoint_display_name,
                    endpoint_family=(
                        "secondary_security"
                    ),
                    direction="lower_is_better",
                    paired_matrix=matrix,
                )
            )

    omnibus_frame = pd.DataFrame(
        omnibus_records
    )

    pairwise_frame = pd.DataFrame(
        pairwise_records
    )

    descriptive_frame = pd.DataFrame(
        descriptive_records
    )

    delta_frame = pd.DataFrame(
        delta_records
    )

    expected_endpoint_count = (
        len(MODEL_NAMES)
        * (
            len(GLOBAL_ENDPOINTS)
            + len(TARGET_CLASSES)
        )
    )

    expected_pairwise_count = (
        expected_endpoint_count
        * 45
    )

    expected_descriptive_count = (
        expected_endpoint_count
        * len(VARIANT_NAMES)
    )

    expected_delta_count = (
        expected_endpoint_count
        * (
            len(VARIANT_NAMES) - 1
        )
    )

    if len(omnibus_frame) != expected_endpoint_count:
        raise RuntimeError(
            "Friedman analiz sayısı beklenen değerde değil."
        )

    if len(pairwise_frame) != expected_pairwise_count:
        raise RuntimeError(
            "Wilcoxon karşılaştırma sayısı beklenen değerde değil."
        )

    if len(descriptive_frame) != expected_descriptive_count:
        raise RuntimeError(
            "Betimsel istatistik sayısı beklenen değerde değil."
        )

    if len(delta_frame) != expected_delta_count:
        raise RuntimeError(
            "B0 fark kaydı sayısı beklenen değerde değil."
        )

    return (
        omnibus_frame,
        pairwise_frame,
        descriptive_frame,
        delta_frame,
    )


def constant_integer_value(
    subset: pd.DataFrame,
    column_name: str,
    model_name: str,
    variant_name: str,
) -> int:
    """Seedler boyunca sabit olması gereken tamsayıyı doğrular."""

    values = pd.to_numeric(
        subset[column_name],
        errors="raise",
    ).astype("int64")

    unique_values = values.unique()

    if len(unique_values) != 1:
        raise RuntimeError(
            f"{model_name}/{variant_name}: "
            f"{column_name} seedler boyunca sabit değil."
        )

    return int(
        unique_values[0]
    )


def build_efficiency_tradeoff(
    efficiency_results: pd.DataFrame,
) -> pd.DataFrame:
    """30 model-varyant için birleşik ödünleşim tablosu oluşturur."""

    records: list[
        dict[str, Any]
    ] = []

    for model_name in MODEL_NAMES:
        for variant_name in VARIANT_NAMES:
            subset = efficiency_results[
                (
                    efficiency_results[
                        "model_name"
                    ]
                    == model_name
                )
                & (
                    efficiency_results[
                        "variant_name"
                    ]
                    == variant_name
                )
            ]

            if len(subset) != len(SEEDS):
                raise RuntimeError(
                    f"{model_name}/{variant_name}: beş verimlilik profili yok."
                )

            metric_columns = {
                "test_macro_f1": "test_macro_f1",
                "deployment_artifact_size_bytes": (
                    "deployment_artifact_size_bytes"
                ),
                "latency_median_ms": (
                    "latency_median_ms"
                ),
                "latency_p95_ms": (
                    "latency_p95_ms"
                ),
                "latency_p99_ms": (
                    "latency_p99_ms"
                ),
                "throughput_samples_per_second": (
                    "throughput_samples_per_second"
                ),
            }

            statistics = {
                output_prefix: calculate_summary_statistics(
                    pd.to_numeric(
                        subset[column_name],
                        errors="raise",
                    ).to_numpy(
                        dtype=np.float64
                    )
                )
                for output_prefix, column_name
                in metric_columns.items()
            }

            records.append(
                {
                    "model_name": model_name,
                    "variant_name": variant_name,
                    "runtime_name": str(
                        subset.iloc[0][
                            "runtime_name"
                        ]
                    ),
                    "precision": str(
                        subset.iloc[0][
                            "precision"
                        ]
                    ),
                    "parameter_count": constant_integer_value(
                        subset,
                        "parameter_count",
                        model_name,
                        variant_name,
                    ),
                    "weight_layer_macs": constant_integer_value(
                        subset,
                        "weight_layer_macs",
                        model_name,
                        variant_name,
                    ),
                    "mean_test_macro_f1": statistics[
                        "test_macro_f1"
                    ][
                        "mean"
                    ],
                    "test_macro_f1_standard_deviation": statistics[
                        "test_macro_f1"
                    ][
                        "standard_deviation"
                    ],
                    "test_macro_f1_ci95_lower": statistics[
                        "test_macro_f1"
                    ][
                        "ci95_lower"
                    ],
                    "test_macro_f1_ci95_upper": statistics[
                        "test_macro_f1"
                    ][
                        "ci95_upper"
                    ],
                    "mean_deployment_artifact_size_bytes": statistics[
                        "deployment_artifact_size_bytes"
                    ][
                        "mean"
                    ],
                    "artifact_size_standard_deviation": statistics[
                        "deployment_artifact_size_bytes"
                    ][
                        "standard_deviation"
                    ],
                    "mean_latency_median_ms": statistics[
                        "latency_median_ms"
                    ][
                        "mean"
                    ],
                    "latency_median_standard_deviation": statistics[
                        "latency_median_ms"
                    ][
                        "standard_deviation"
                    ],
                    "mean_latency_p95_ms": statistics[
                        "latency_p95_ms"
                    ][
                        "mean"
                    ],
                    "mean_latency_p99_ms": statistics[
                        "latency_p99_ms"
                    ][
                        "mean"
                    ],
                    "mean_throughput_samples_per_second": statistics[
                        "throughput_samples_per_second"
                    ][
                        "mean"
                    ],
                    "throughput_standard_deviation": statistics[
                        "throughput_samples_per_second"
                    ][
                        "standard_deviation"
                    ],
                    "host_cpu_profile_only": True,
                    "mcu_measurement": False,
                }
            )

    return pd.DataFrame(
        records
    )


def row_dominates(
    candidate: pd.Series,
    target: pd.Series,
) -> bool:
    """Dört amaçlı Pareto baskınlığını değerlendirir."""

    candidate_costs = np.asarray(
        [
            -float(
                candidate[
                    "mean_test_macro_f1"
                ]
            ),
            float(
                candidate[
                    "mean_deployment_artifact_size_bytes"
                ]
            ),
            float(
                candidate[
                    "mean_latency_median_ms"
                ]
            ),
            float(
                candidate[
                    "weight_layer_macs"
                ]
            ),
        ],
        dtype=np.float64,
    )

    target_costs = np.asarray(
        [
            -float(
                target[
                    "mean_test_macro_f1"
                ]
            ),
            float(
                target[
                    "mean_deployment_artifact_size_bytes"
                ]
            ),
            float(
                target[
                    "mean_latency_median_ms"
                ]
            ),
            float(
                target[
                    "weight_layer_macs"
                ]
            ),
        ],
        dtype=np.float64,
    )

    no_worse = bool(
        np.all(
            candidate_costs
            <= target_costs
        )
    )

    strictly_better = bool(
        np.any(
            candidate_costs
            < target_costs
        )
    )

    return (
        no_worse
        and strictly_better
    )


def identify_pareto_front(
    tradeoff_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Model içi ve global Pareto önlerini oluşturur."""

    records: list[
        dict[str, Any]
    ] = []

    scopes: list[
        tuple[str, str | None, pd.DataFrame]
    ] = [
        (
            "global",
            None,
            tradeoff_frame,
        )
    ]

    scopes.extend(
        (
            "within_model",
            model_name,
            tradeoff_frame[
                tradeoff_frame[
                    "model_name"
                ]
                == model_name
            ],
        )
        for model_name in MODEL_NAMES
    )

    for scope_name, scope_model, subset in scopes:
        subset = subset.reset_index(
            drop=True
        )

        for target_index, target in subset.iterrows():
            dominated_by: list[str] = []

            for candidate_index, candidate in subset.iterrows():
                if candidate_index == target_index:
                    continue

                if row_dominates(
                    candidate,
                    target,
                ):
                    dominated_by.append(
                        f"{candidate['model_name']}/{candidate['variant_name']}"
                    )

            records.append(
                {
                    "pareto_scope": scope_name,
                    "scope_model_name": scope_model,
                    "model_name": str(
                        target[
                            "model_name"
                        ]
                    ),
                    "variant_name": str(
                        target[
                            "variant_name"
                        ]
                    ),
                    "is_pareto_optimal": bool(
                        len(dominated_by) == 0
                    ),
                    "dominated_by_count": int(
                        len(dominated_by)
                    ),
                    "dominated_by": ";".join(
                        dominated_by
                    ),
                    "mean_test_macro_f1": float(
                        target[
                            "mean_test_macro_f1"
                        ]
                    ),
                    "mean_deployment_artifact_size_bytes": float(
                        target[
                            "mean_deployment_artifact_size_bytes"
                        ]
                    ),
                    "mean_latency_median_ms": float(
                        target[
                            "mean_latency_median_ms"
                        ]
                    ),
                    "weight_layer_macs": int(
                        target[
                            "weight_layer_macs"
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def select_extreme_candidate(
    frame: pd.DataFrame,
    *,
    column_name: str,
    ascending: bool,
) -> dict[str, Any]:
    """Bir mühendislik amacının en iyi adayını seçer."""

    ordered = frame.sort_values(
        by=[
            column_name,
            "model_name",
            "variant_name",
        ],
        ascending=[
            ascending,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    row = ordered.iloc[0]

    return {
        "model_name": str(
            row[
                "model_name"
            ]
        ),
        "variant_name": str(
            row[
                "variant_name"
            ]
        ),
        "metric": column_name,
        "value": float(
            row[
                column_name
            ]
        ),
        "runtime_name": str(
            row[
                "runtime_name"
            ]
        ),
    }


def build_engineering_summary(
    tradeoff_frame: pd.DataFrame,
    pareto_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Keyfi bileşik puan kullanmadan amaç bazlı adayları raporlar."""

    summary: dict[str, Any] = {
        "global": {},
        "within_model": {},
    }

    global_pareto = pareto_frame[
        (
            pareto_frame[
                "pareto_scope"
            ]
            == "global"
        )
        & (
            pareto_frame[
                "is_pareto_optimal"
            ]
        )
    ]

    summary[
        "global"
    ] = {
        "best_accuracy": select_extreme_candidate(
            tradeoff_frame,
            column_name=(
                "mean_test_macro_f1"
            ),
            ascending=False,
        ),
        "smallest_artifact": select_extreme_candidate(
            tradeoff_frame,
            column_name=(
                "mean_deployment_artifact_size_bytes"
            ),
            ascending=True,
        ),
        "lowest_host_latency": select_extreme_candidate(
            tradeoff_frame,
            column_name=(
                "mean_latency_median_ms"
            ),
            ascending=True,
        ),
        "highest_host_throughput": select_extreme_candidate(
            tradeoff_frame,
            column_name=(
                "mean_throughput_samples_per_second"
            ),
            ascending=False,
        ),
        "pareto_candidates": [
            {
                "model_name": str(
                    row.model_name
                ),
                "variant_name": str(
                    row.variant_name
                ),
            }
            for row in global_pareto.itertuples(
                index=False
            )
        ],
    }

    for model_name in MODEL_NAMES:
        subset = tradeoff_frame[
            tradeoff_frame[
                "model_name"
            ]
            == model_name
        ]

        model_pareto = pareto_frame[
            (
                pareto_frame[
                    "pareto_scope"
                ]
                == "within_model"
            )
            & (
                pareto_frame[
                    "scope_model_name"
                ]
                == model_name
            )
            & (
                pareto_frame[
                    "is_pareto_optimal"
                ]
            )
        ]

        summary[
            "within_model"
        ][
            model_name
        ] = {
            "best_accuracy": select_extreme_candidate(
                subset,
                column_name=(
                    "mean_test_macro_f1"
                ),
                ascending=False,
            ),
            "smallest_artifact": select_extreme_candidate(
                subset,
                column_name=(
                    "mean_deployment_artifact_size_bytes"
                ),
                ascending=True,
            ),
            "lowest_host_latency": select_extreme_candidate(
                subset,
                column_name=(
                    "mean_latency_median_ms"
                ),
                ascending=True,
            ),
            "highest_host_throughput": select_extreme_candidate(
                subset,
                column_name=(
                    "mean_throughput_samples_per_second"
                ),
                ascending=False,
            ),
            "pareto_variants": [
                str(
                    row.variant_name
                )
                for row in model_pareto.itertuples(
                    index=False
                )
            ],
        }

    return summary


def build_primary_summary(
    omnibus_frame: pd.DataFrame,
    descriptive_frame: pd.DataFrame,
    pairwise_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Macro F1 sonuçlarını JSON için özetler."""

    summary: dict[
        str,
        Any
    ] = {}

    for model_name in MODEL_NAMES:
        omnibus_row = omnibus_frame[
            (
                omnibus_frame[
                    "model_name"
                ]
                == model_name
            )
            & (
                omnibus_frame[
                    "endpoint_id"
                ]
                == "test_macro_f1"
            )
        ].iloc[0]

        rankings = descriptive_frame[
            (
                descriptive_frame[
                    "model_name"
                ]
                == model_name
            )
            & (
                descriptive_frame[
                    "endpoint_id"
                ]
                == "test_macro_f1"
            )
        ].sort_values(
            by=[
                "mean",
                "standard_deviation",
            ],
            ascending=[
                False,
                True,
            ],
        )

        b0_comparisons = pairwise_frame[
            (
                pairwise_frame[
                    "model_name"
                ]
                == model_name
            )
            & (
                pairwise_frame[
                    "endpoint_id"
                ]
                == "test_macro_f1"
            )
            & (
                pairwise_frame[
                    "variant_a"
                ]
                == "B0"
            )
        ]

        summary[
            model_name
        ] = {
            "friedman_chi_square": float(
                omnibus_row[
                    "friedman_chi_square"
                ]
            ),
            "friedman_degrees_of_freedom": int(
                omnibus_row[
                    "friedman_degrees_of_freedom"
                ]
            ),
            "friedman_p_value": float(
                omnibus_row[
                    "friedman_p_value"
                ]
            ),
            "friedman_significant": bool(
                omnibus_row[
                    "friedman_significant"
                ]
            ),
            "kendall_w": float(
                omnibus_row[
                    "kendall_w"
                ]
            ),
            "kendall_w_magnitude": str(
                omnibus_row[
                    "kendall_w_magnitude"
                ]
            ),
            "variant_ranking": [
                {
                    "rank": int(index + 1),
                    "variant_name": str(
                        row.variant_name
                    ),
                    "mean_test_macro_f1": float(
                        row.mean
                    ),
                    "standard_deviation": float(
                        row.standard_deviation
                    ),
                    "average_rank": float(
                        row.average_rank
                    ),
                }
                for index, row in enumerate(
                    rankings.itertuples(
                        index=False
                    )
                )
            ],
            "b0_pairwise_comparisons": [
                {
                    "comparison": (
                        f"{row.variant_a} vs {row.variant_b}"
                    ),
                    "raw_p_value": float(
                        row.wilcoxon_raw_p_value
                    ),
                    "holm_adjusted_p_value": float(
                        row.holm_adjusted_p_value
                    ),
                    "rank_biserial_correlation": float(
                        row.rank_biserial_correlation
                    ),
                    "effect_magnitude": str(
                        row.effect_magnitude
                    ),
                    "b0_seed_wins": int(
                        row.variant_a_seed_wins
                    ),
                    "seed_ties": int(
                        row.seed_ties
                    ),
                    "b0_seed_losses": int(
                        row.variant_a_seed_losses
                    ),
                    "interpretation_status": str(
                        row.interpretation_status
                    ),
                }
                for row in b0_comparisons.itertuples(
                    index=False
                )
            ],
        }

    return summary


def main() -> None:
    """Birleşik 32. aşama analizini çalıştırır."""

    args = parse_arguments()

    alpha = float(
        args.alpha
    )

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "Alpha 0 ile 1 arasında olmalıdır."
        )

    report_directory = (
        args.report_directory.resolve()
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = {
        "overall_runs": (
            report_directory
            / "nbaiot_family3_overall_runs.csv"
        ),
        "overall_per_class_runs": (
            report_directory
            / "nbaiot_family3_overall_per_class_runs.csv"
        ),
        "friedman": (
            report_directory
            / "nbaiot_family3_overall_friedman_results.csv"
        ),
        "wilcoxon": (
            report_directory
            / "nbaiot_family3_overall_wilcoxon_holm_results.csv"
        ),
        "descriptives": (
            report_directory
            / "nbaiot_family3_overall_statistical_descriptives.csv"
        ),
        "b0_deltas": (
            report_directory
            / "nbaiot_family3_overall_paired_deltas_vs_b0.csv"
        ),
        "tradeoff": (
            report_directory
            / "nbaiot_family3_overall_efficiency_tradeoff.csv"
        ),
        "pareto": (
            report_directory
            / "nbaiot_family3_overall_pareto_front.csv"
        ),
        "summary": (
            report_directory
            / "nbaiot_family3_overall_statistical_summary.json"
        ),
    }

    if (
        any(
            file_path.exists()
            for file_path in output_files.values()
        )
        and not args.overwrite
    ):
        raise FileExistsError(
            "32. aşama çıktıları zaten mevcut. "
            "--overwrite kullan."
        )

    print("=" * 78)
    print("N-BaIoT Family-3 Birleşik İstatistik ve Ödünleşim Analizi")
    print("=" * 78)
    print(
        "Modeller   : "
        + ", ".join(
            MODEL_NAMES
        )
    )
    print(
        "Varyantlar : "
        + ", ".join(
            VARIANT_NAMES
        )
    )
    print(
        "Seedler    : "
        + ", ".join(
            str(seed)
            for seed in SEEDS
        )
    )
    print(
        "Ana sonuç  : 150"
    )
    print(
        "Sınıf sonucu: 450"
    )
    print(
        f"Alpha      : {alpha}"
    )
    print("=" * 78)

    global_results = load_all_global_results(
        args
    )

    per_class_results = (
        load_all_per_class_results(
            args
        )
    )

    efficiency_results = (
        load_efficiency_results(
            args,
            global_results,
        )
    )

    (
        omnibus_frame,
        pairwise_frame,
        descriptive_frame,
        delta_frame,
    ) = build_statistical_outputs(
        global_results=global_results,
        per_class_results=per_class_results,
        alpha=alpha,
    )

    tradeoff_frame = (
        build_efficiency_tradeoff(
            efficiency_results
        )
    )

    pareto_frame = identify_pareto_front(
        tradeoff_frame
    )

    engineering_summary = (
        build_engineering_summary(
            tradeoff_frame,
            pareto_frame,
        )
    )

    primary_summary = build_primary_summary(
        omnibus_frame,
        descriptive_frame,
        pairwise_frame,
    )

    write_csv_atomic(
        global_results,
        output_files[
            "overall_runs"
        ],
    )

    write_csv_atomic(
        per_class_results,
        output_files[
            "overall_per_class_runs"
        ],
    )

    write_csv_atomic(
        omnibus_frame,
        output_files[
            "friedman"
        ],
    )

    write_csv_atomic(
        pairwise_frame,
        output_files[
            "wilcoxon"
        ],
    )

    write_csv_atomic(
        descriptive_frame,
        output_files[
            "descriptives"
        ],
    )

    write_csv_atomic(
        delta_frame,
        output_files[
            "b0_deltas"
        ],
    )

    write_csv_atomic(
        tradeoff_frame,
        output_files[
            "tradeoff"
        ],
    )

    write_csv_atomic(
        pareto_frame,
        output_files[
            "pareto"
        ],
    )

    input_files = {
        "fp32_run_file": require_file(
            args.fp32_run_file
        ),
        "pruning_run_file": require_file(
            args.pruning_run_file
        ),
        "quantization_run_file": require_file(
            args.quantization_run_file
        ),
        "pruning_qat_run_file": require_file(
            args.pruning_qat_run_file
        ),
        "fp32_per_class_file": require_file(
            args.fp32_per_class_file
        ),
        "pruning_per_class_file": require_file(
            args.pruning_per_class_file
        ),
        "quantization_per_class_file": require_file(
            args.quantization_per_class_file
        ),
        "pruning_qat_per_class_file": require_file(
            args.pruning_qat_per_class_file
        ),
        "efficiency_run_file": require_file(
            args.efficiency_run_file
        ),
    }

    summary_document = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "analysis_name": (
            "N-BaIoT family_3 unified compression "
            "statistics and host CPU trade-off analysis"
        ),
        "analysis_version": "1.0",
        "task": "family_3",
        "models": list(
            MODEL_NAMES
        ),
        "variants": list(
            VARIANT_NAMES
        ),
        "paired_seeds": list(
            SEEDS
        ),
        "global_result_count": int(
            len(global_results)
        ),
        "per_class_result_count": int(
            len(per_class_results)
        ),
        "efficiency_profile_count": int(
            len(efficiency_results)
        ),
        "primary_endpoint": (
            "test_macro_f1"
        ),
        "secondary_global_endpoints": [
            "test_mcc",
            "test_balanced_accuracy",
        ],
        "security_endpoints": [
            f"{class_name}_false_negative_rate"
            for class_name in TARGET_CLASSES
        ],
        "alpha": alpha,
        "statistical_tests": {
            "omnibus": (
                "Friedman chi-square test"
            ),
            "omnibus_effect_size": (
                "Kendall's W"
            ),
            "pairwise": (
                "Two-sided paired Wilcoxon signed-rank test"
            ),
            "pairwise_effect_size": (
                "Matched-pairs rank-biserial correlation"
            ),
            "multiple_testing": (
                "Holm step-down correction across 45 variant "
                "pairs separately within each model and endpoint"
            ),
            "gatekeeping": (
                "Pairwise findings are confirmatory only when "
                "the corresponding Friedman test is significant."
            ),
        },
        "primary_endpoint_results": (
            primary_summary
        ),
        "engineering_summary": (
            engineering_summary
        ),
        "pareto_objectives": {
            "maximize": [
                "mean_test_macro_f1",
            ],
            "minimize": [
                "mean_deployment_artifact_size_bytes",
                "mean_latency_median_ms",
                "weight_layer_macs",
            ],
            "composite_score_used": False,
        },
        "small_sample_note": (
            "Each comparison contains five paired seeds. "
            "For five non-zero differences, the minimum "
            "attainable two-sided exact Wilcoxon p-value is "
            "typically 0.0625. Effect sizes, paired deltas and "
            "directional consistency must therefore be interpreted "
            "with p-values."
        ),
        "efficiency_scope_note": (
            "Latency, throughput and RSS measurements are host CPU "
            "measurements from the recorded PyTorch or ONNX Runtime "
            "execution environment. They are not MCU measurements "
            "and cross-runtime comparisons must be interpreted as "
            "deployment-stack comparisons."
        ),
        "test_data_policy": (
            "Test results were produced only after validation-based "
            "model selection. Test data were not used for training, "
            "pruning, calibration policy selection, QAT early stopping "
            "or hyperparameter selection."
        ),
        "input_artifacts": {
            name: {
                "path": str(
                    file_path
                ),
                "sha256": calculate_sha256(
                    file_path
                ),
            }
            for name, file_path in input_files.items()
        },
        "output_artifacts": {
            name: str(
                file_path
            )
            for name, file_path in output_files.items()
            if name != "summary"
        },
        "validation_passed": True,
    }

    write_json_atomic(
        summary_document,
        output_files[
            "summary"
        ],
    )

    print()
    print("=" * 78)
    print("Birleşik İstatistik ve Ödünleşim Analizi Tamamlandı")
    print("=" * 78)

    for model_name in MODEL_NAMES:
        result = primary_summary[
            model_name
        ]

        print()
        print(
            f"{model_name}:"
        )
        print(
            "  Friedman chi-square = "
            f"{result['friedman_chi_square']:.6f}"
        )
        print(
            "  Friedman p          = "
            f"{result['friedman_p_value']:.8f}"
        )
        print(
            "  Anlamlı             = "
            f"{result['friedman_significant']}"
        )
        print(
            "  Kendall W           = "
            f"{result['kendall_w']:.6f} "
            f"({result['kendall_w_magnitude']})"
        )
        print(
            "  İlk üç Macro F1 varyantı:"
        )

        for ranking in result[
            "variant_ranking"
        ][
            :3
        ]:
            print(
                f"    {ranking['rank']}. "
                f"{ranking['variant_name']:8s} | "
                f"ortalama="
                f"{ranking['mean_test_macro_f1']:.6f} | "
                f"SS="
                f"{ranking['standard_deviation']:.6f}"
            )

    global_engineering = (
        engineering_summary[
            "global"
        ]
    )

    print()
    print("Global mühendislik uçları:")
    print(
        "  En yüksek doğruluk : "
        f"{global_engineering['best_accuracy']['model_name']}/"
        f"{global_engineering['best_accuracy']['variant_name']}"
    )
    print(
        "  En küçük artifact  : "
        f"{global_engineering['smallest_artifact']['model_name']}/"
        f"{global_engineering['smallest_artifact']['variant_name']}"
    )
    print(
        "  En düşük gecikme   : "
        f"{global_engineering['lowest_host_latency']['model_name']}/"
        f"{global_engineering['lowest_host_latency']['variant_name']}"
    )
    print(
        "  En yüksek throughput: "
        f"{global_engineering['highest_host_throughput']['model_name']}/"
        f"{global_engineering['highest_host_throughput']['variant_name']}"
    )

    print()
    print(
        f"Ana sonuç sayısı       : {len(global_results)}"
    )
    print(
        f"Sınıf sonucu sayısı    : {len(per_class_results)}"
    )
    print(
        f"Verimlilik profili     : {len(efficiency_results)}"
    )
    print(
        f"Friedman analiz sayısı : {len(omnibus_frame)}"
    )
    print(
        f"Wilcoxon analiz sayısı : {len(pairwise_frame)}"
    )
    print(
        "Test seçimde kullanıldı: False"
    )
    print(
        "MCU ölçümü             : False"
    )
    print(
        "Doğrulama geçti        : True"
    )
    print()
    print(
        f"Birleşik sonuçlar : {output_files['overall_runs']}"
    )
    print(
        f"Friedman          : {output_files['friedman']}"
    )
    print(
        f"Wilcoxon + Holm   : {output_files['wilcoxon']}"
    )
    print(
        f"B0 farkları       : {output_files['b0_deltas']}"
    )
    print(
        f"Ödünleşim         : {output_files['tradeoff']}"
    )
    print(
        f"Pareto önü        : {output_files['pareto']}"
    )
    print(
        f"JSON özet         : {output_files['summary']}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
