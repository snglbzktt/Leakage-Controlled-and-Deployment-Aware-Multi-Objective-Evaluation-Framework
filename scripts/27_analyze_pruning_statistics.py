"""
N-BaIoT family_3 FP32 ve yapılandırılmış budama sonuçlarının
eşleştirilmiş istatistiksel analizi.

Karşılaştırılan varyantlar:
- FP32
- P25
- P50
- P75

Her mimari ayrı analiz edilir:
- tinyml_mlp
- compact_dnn
- tiny_1d_cnn

Analizler:
1. Friedman omnibus testi
2. Kendall's W omnibus etki büyüklüğü
3. İki yönlü eşleştirilmiş Wilcoxon signed-rank testi
4. Model ve sonuç değişkeni başına altı karşılaştırmada Holm düzeltmesi
5. Eşleştirilmiş rank-biserial correlation
6. FP32'ye göre eşleştirilmiş performans değişimi
7. Parametre, MAC ve Macro F1 ödünleşim özeti

Bilimsel kurallar:
- test_macro_f1 birincil sonuç değişkenidir.
- MCC ve balanced accuracy ikincil genel metriklerdir.
- Sınıf bazlı FNR güvenlik odaklı ikincil metriklerdir.
- İkili sonuçlar ilgili Friedman testi anlamlı olduğunda
  doğrulayıcı olarak yorumlanır.
- Test sonuçları eğitim veya model seçiminde kullanılmaz.
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

DEFAULT_FP32_RUN_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_family3_fp32_baseline_runs.csv"
)

DEFAULT_PRUNING_RUN_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_family3_structured_pruning_runs.csv"
)

DEFAULT_FP32_PER_CLASS_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_family3_fp32_baseline_per_class_runs.csv"
)

DEFAULT_PRUNING_PER_CLASS_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_family3_structured_pruning_per_class_runs.csv"
)

DEFAULT_PRUNING_PROTOCOL_FILE = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_structured_pruning_protocol_v1.json"
)

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

MODEL_NAMES = (
    "tinyml_mlp",
    "compact_dnn",
    "tiny_1d_cnn",
)

VARIANT_NAMES = (
    "FP32",
    "P25",
    "P50",
    "P75",
)

PRUNING_VARIANTS = (
    "P25",
    "P50",
    "P75",
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

RESOURCE_COLUMNS = (
    "original_parameter_count",
    "compact_parameter_count",
    "parameter_reduction_ratio",
    "original_weight_layer_macs",
    "compact_weight_layer_macs",
    "mac_reduction_ratio",
)

ALPHA = 0.05


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "FP32, P25, P50 ve P75 sonuçlarını her mimari için "
            "eşleştirilmiş istatistiksel testlerle karşılaştırır."
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
        "--pruning-protocol-file",
        type=Path,
        default=DEFAULT_PRUNING_PROTOCOL_FILE,
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


def read_json(input_file: Path) -> dict[str, Any]:
    """JSON dosyasını sözlük olarak yükler."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"JSON dosyası bulunamadı: {input_file}"
        )

    with input_file.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        document = json.load(file_handle)

    if not isinstance(document, dict):
        raise TypeError(
            f"JSON kökü sözlük değil: {input_file}"
        )

    return document


def write_json_atomic(
    document: dict[str, Any],
    output_file: Path,
) -> None:
    """JSON dosyasını atomik olarak yazar."""

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
    """CSV dosyasını atomik olarak yazar."""

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


# ==========================================================
# VALIDATION HELPERS
# ==========================================================

def normalize_boolean_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """Boolean, 0/1 ve metinsel boolean değerlerini doğrular."""

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
        invalid_values = sorted(
            series[
                normalized.isna()
            ].astype(str).unique().tolist()
        )

        raise ValueError(
            f"{column_name} içinde geçersiz boolean değerleri var: "
            + ", ".join(invalid_values)
        )

    return normalized.astype(bool)


def validate_pruning_protocol(
    protocol_file: Path,
) -> dict[str, Any]:
    """Kilitli budama protokolünü doğrular."""

    protocol = read_json(protocol_file)

    if protocol.get("protocol_version") != "1.0":
        raise ValueError(
            "Beklenmeyen budama protokol sürümü."
        )

    if (
        protocol.get("status")
        != "locked_before_pruning_main_experiments"
    ):
        raise ValueError(
            "Budama protokolü ana deneylerden önce kilitlenmemiş."
        )

    if protocol.get("task") != "family_3":
        raise ValueError(
            "Budama protokol görevi family_3 değil."
        )

    if tuple(protocol["models"]) != MODEL_NAMES:
        raise ValueError(
            "Budama protokol model sırası uyuşmuyor."
        )

    if tuple(
        int(seed)
        for seed in protocol["training_seeds"]
    ) != SEEDS:
        raise ValueError(
            "Budama protokol seed sırası uyuşmuyor."
        )

    if tuple(protocol["pruning_names"]) != PRUNING_VARIANTS:
        raise ValueError(
            "Budama protokol varyantları P25/P50/P75 ile uyuşmuyor."
        )

    if int(protocol["planned_run_count"]) != 45:
        raise ValueError(
            "Budama protokolündeki planlanan çalışma sayısı 45 değil."
        )

    return protocol


def validate_expected_combinations(
    frame: pd.DataFrame,
    *,
    include_class: bool,
) -> None:
    """Beklenen model-varyant-seed kombinasyonlarını doğrular."""

    if include_class:
        expected_records = {
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

        actual_records = {
            (
                str(row.model_name),
                str(row.variant_name),
                int(row.seed),
                str(row.class_name),
            )
            for row in frame.itertuples(index=False)
        }

    else:
        expected_records = {
            (
                model_name,
                variant_name,
                seed,
            )
            for model_name in MODEL_NAMES
            for variant_name in VARIANT_NAMES
            for seed in SEEDS
        }

        actual_records = {
            (
                str(row.model_name),
                str(row.variant_name),
                int(row.seed),
            )
            for row in frame.itertuples(index=False)
        }

    if actual_records != expected_records:
        missing_records = expected_records - actual_records
        unexpected_records = actual_records - expected_records

        raise ValueError(
            "Model-varyant-seed kombinasyonları uyuşmuyor. "
            f"Eksik={len(missing_records)}, "
            f"beklenmeyen={len(unexpected_records)}"
        )


# ==========================================================
# INPUT DATA
# ==========================================================

def load_fp32_runs(input_file: Path) -> pd.DataFrame:
    """FP32 model-seed sonuçlarını yükler."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"FP32 sonuç dosyası bulunamadı: {input_file}"
        )

    frame = pd.read_csv(input_file)

    required_columns = {
        "model_name",
        "seed",
        "test_evaluation_attempt_count",
        "test_used_for_selection",
        *GLOBAL_ENDPOINTS.keys(),
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            "FP32 sonuç dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing_columns))
        )

    frame = frame.copy()

    frame["model_name"] = frame["model_name"].astype(str)

    frame["seed"] = pd.to_numeric(
        frame["seed"],
        errors="raise",
    ).astype("int64")

    frame["variant_name"] = "FP32"

    if len(frame) != 15:
        raise ValueError(
            f"FP32 sonuç sayısı 15 değil: {len(frame)}"
        )

    if frame.duplicated(
        subset=[
            "model_name",
            "seed",
        ]
    ).any():
        raise ValueError(
            "FP32 dosyasında tekrarlanan model-seed kaydı var."
        )

    attempts = pd.to_numeric(
        frame["test_evaluation_attempt_count"],
        errors="raise",
    ).astype("int64")

    if (attempts != 1).any():
        raise ValueError(
            "FP32 test değerlendirme girişimlerinden biri 1 değil."
        )

    selection_flags = normalize_boolean_series(
        frame["test_used_for_selection"],
        "FP32 test_used_for_selection",
    )

    if selection_flags.any():
        raise RuntimeError(
            "FP32 test sonuçlarından biri model seçimine katılmış."
        )

    for metric_name in GLOBAL_ENDPOINTS:
        frame[metric_name] = pd.to_numeric(
            frame[metric_name],
            errors="raise",
        )

        if not np.isfinite(
            frame[metric_name].to_numpy(
                dtype=np.float64
            )
        ).all():
            raise ValueError(
                f"FP32 {metric_name} içinde sonlu olmayan değer var."
            )

    return frame


def load_pruning_runs(input_file: Path) -> pd.DataFrame:
    """P25/P50/P75 model-seed sonuçlarını yükler."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"Budama sonuç dosyası bulunamadı: {input_file}"
        )

    frame = pd.read_csv(input_file)

    required_columns = {
        "model_name",
        "pruning_name",
        "seed",
        "test_evaluation_attempt_count",
        "test_used_for_selection",
        *GLOBAL_ENDPOINTS.keys(),
        *RESOURCE_COLUMNS,
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            "Budama sonuç dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing_columns))
        )

    frame = frame.copy()

    frame["model_name"] = frame["model_name"].astype(str)

    frame["pruning_name"] = frame[
        "pruning_name"
    ].astype(str)

    frame["variant_name"] = frame[
        "pruning_name"
    ]

    frame["seed"] = pd.to_numeric(
        frame["seed"],
        errors="raise",
    ).astype("int64")

    if len(frame) != 45:
        raise ValueError(
            f"Budama sonuç sayısı 45 değil: {len(frame)}"
        )

    if frame.duplicated(
        subset=[
            "model_name",
            "pruning_name",
            "seed",
        ]
    ).any():
        raise ValueError(
            "Budama dosyasında tekrarlanan "
            "model-varyant-seed kaydı var."
        )

    actual_variants = tuple(
        sorted(
            frame["pruning_name"].unique().tolist()
        )
    )

    if actual_variants != tuple(
        sorted(PRUNING_VARIANTS)
    ):
        raise ValueError(
            "Budama varyantları P25/P50/P75 ile uyuşmuyor."
        )

    attempts = pd.to_numeric(
        frame["test_evaluation_attempt_count"],
        errors="raise",
    ).astype("int64")

    if (attempts != 1).any():
        raise ValueError(
            "Budama test değerlendirme girişimlerinden biri 1 değil."
        )

    selection_flags = normalize_boolean_series(
        frame["test_used_for_selection"],
        "Budama test_used_for_selection",
    )

    if selection_flags.any():
        raise RuntimeError(
            "Budama test sonuçlarından biri model seçimine katılmış."
        )

    numeric_columns = (
        *GLOBAL_ENDPOINTS.keys(),
        *RESOURCE_COLUMNS,
    )

    for column_name in numeric_columns:
        frame[column_name] = pd.to_numeric(
            frame[column_name],
            errors="raise",
        )

        if not np.isfinite(
            frame[column_name].to_numpy(
                dtype=np.float64
            )
        ).all():
            raise ValueError(
                f"Budama {column_name} içinde sonlu olmayan değer var."
            )

    return frame


def combine_run_results(
    fp32_frame: pd.DataFrame,
    pruning_frame: pd.DataFrame,
) -> pd.DataFrame:
    """FP32 ve budama sonuçlarını ortak uzun formata getirir."""

    columns = [
        "model_name",
        "variant_name",
        "seed",
        *GLOBAL_ENDPOINTS.keys(),
    ]

    combined = pd.concat(
        [
            fp32_frame[columns],
            pruning_frame[columns],
        ],
        ignore_index=True,
    )

    expected_count = (
        len(MODEL_NAMES)
        * len(VARIANT_NAMES)
        * len(SEEDS)
    )

    if len(combined) != expected_count:
        raise ValueError(
            f"Birleşik sonuç sayısı 60 değil: {len(combined)}"
        )

    if combined.duplicated(
        subset=[
            "model_name",
            "variant_name",
            "seed",
        ]
    ).any():
        raise ValueError(
            "Birleşik sonuçlarda tekrarlanan kayıt var."
        )

    validate_expected_combinations(
        combined,
        include_class=False,
    )

    return combined


def load_fp32_per_class(input_file: Path) -> pd.DataFrame:
    """FP32 sınıf bazlı sonuçlarını yükler."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"FP32 sınıf sonuçları bulunamadı: {input_file}"
        )

    frame = pd.read_csv(input_file)

    required_columns = {
        "model_name",
        "seed",
        "class_name",
        "false_negative_rate",
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            "FP32 sınıf sonuçlarında eksik sütunlar var: "
            + ", ".join(sorted(missing_columns))
        )

    frame = frame.copy()

    frame["model_name"] = frame["model_name"].astype(str)

    frame["seed"] = pd.to_numeric(
        frame["seed"],
        errors="raise",
    ).astype("int64")

    frame["class_name"] = frame["class_name"].astype(str)

    frame["false_negative_rate"] = pd.to_numeric(
        frame["false_negative_rate"],
        errors="raise",
    )

    frame["variant_name"] = "FP32"

    if len(frame) != 45:
        raise ValueError(
            f"FP32 sınıf sonuç sayısı 45 değil: {len(frame)}"
        )

    if frame.duplicated(
        subset=[
            "model_name",
            "seed",
            "class_name",
        ]
    ).any():
        raise ValueError(
            "FP32 sınıf sonuçlarında tekrarlanan kayıt var."
        )

    return frame


def load_pruning_per_class(input_file: Path) -> pd.DataFrame:
    """Budanmış modellerin sınıf bazlı sonuçlarını yükler."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"Budama sınıf sonuçları bulunamadı: {input_file}"
        )

    frame = pd.read_csv(input_file)

    required_columns = {
        "model_name",
        "pruning_name",
        "seed",
        "class_name",
        "false_negative_rate",
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            "Budama sınıf sonuçlarında eksik sütunlar var: "
            + ", ".join(sorted(missing_columns))
        )

    frame = frame.copy()

    frame["model_name"] = frame["model_name"].astype(str)

    frame["pruning_name"] = frame[
        "pruning_name"
    ].astype(str)

    frame["variant_name"] = frame[
        "pruning_name"
    ]

    frame["seed"] = pd.to_numeric(
        frame["seed"],
        errors="raise",
    ).astype("int64")

    frame["class_name"] = frame["class_name"].astype(str)

    frame["false_negative_rate"] = pd.to_numeric(
        frame["false_negative_rate"],
        errors="raise",
    )

    if len(frame) != 135:
        raise ValueError(
            "Budama sınıf sonuç sayısı 135 değil: "
            f"{len(frame)}"
        )

    if frame.duplicated(
        subset=[
            "model_name",
            "pruning_name",
            "seed",
            "class_name",
        ]
    ).any():
        raise ValueError(
            "Budama sınıf sonuçlarında tekrarlanan kayıt var."
        )

    return frame


def combine_per_class_results(
    fp32_frame: pd.DataFrame,
    pruning_frame: pd.DataFrame,
) -> pd.DataFrame:
    """FP32 ve budama sınıf sonuçlarını birleştirir."""

    columns = [
        "model_name",
        "variant_name",
        "seed",
        "class_name",
        "false_negative_rate",
    ]

    combined = pd.concat(
        [
            fp32_frame[columns],
            pruning_frame[columns],
        ],
        ignore_index=True,
    )

    expected_count = (
        len(MODEL_NAMES)
        * len(VARIANT_NAMES)
        * len(SEEDS)
        * len(TARGET_CLASSES)
    )

    if len(combined) != expected_count:
        raise ValueError(
            "Birleşik sınıf sonucu sayısı 180 değil: "
            f"{len(combined)}"
        )

    if not np.isfinite(
        combined[
            "false_negative_rate"
        ].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "Birleşik FNR sonuçlarında sonlu olmayan değer var."
        )

    if (
        combined["false_negative_rate"] < 0
    ).any() or (
        combined["false_negative_rate"] > 1
    ).any():
        raise ValueError(
            "FNR sonuçlarından biri [0, 1] aralığı dışında."
        )

    validate_expected_combinations(
        combined,
        include_class=True,
    )

    return combined


# ==========================================================
# STATISTICS
# ==========================================================

def calculate_summary_statistics(
    values: np.ndarray,
) -> dict[str, float | int]:
    """Özet istatistikleri ve %95 Student-t güven aralığını hesaplar."""

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

    sample_count = int(len(values))
    mean_value = float(values.mean())
    median_value = float(np.median(values))
    minimum_value = float(values.min())
    maximum_value = float(values.max())

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

        confidence_margin = float(
            critical_value
            * standard_error
        )

    else:
        standard_deviation = 0.0
        standard_error = 0.0
        confidence_margin = 0.0

    return {
        "sample_count": sample_count,
        "mean": mean_value,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "median": median_value,
        "minimum": minimum_value,
        "maximum": maximum_value,
        "ci95_lower": float(
            mean_value - confidence_margin
        ),
        "ci95_upper": float(
            mean_value + confidence_margin
        ),
    }


def holm_adjust(
    p_values: list[float],
) -> list[float]:
    """Holm step-down çoklu karşılaştırma düzeltmesini uygular."""

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

    comparison_count = len(values)

    sorted_indices = np.argsort(
        values,
        kind="mergesort",
    )

    sorted_values = values[
        sorted_indices
    ]

    adjusted_sorted = np.empty(
        comparison_count,
        dtype=np.float64,
    )

    running_maximum = 0.0

    for rank_index, raw_p_value in enumerate(
        sorted_values
    ):
        multiplier = (
            comparison_count
            - rank_index
        )

        candidate = min(
            1.0,
            float(raw_p_value)
            * multiplier,
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted_sorted[
            rank_index
        ] = running_maximum

    adjusted = np.empty(
        comparison_count,
        dtype=np.float64,
    )

    adjusted[
        sorted_indices
    ] = adjusted_sorted

    return adjusted.tolist()


def interpret_rank_biserial(
    effect_size: float,
) -> str:
    """Rank-biserial korelasyon büyüklüğünü sınıflandırır."""

    absolute_value = abs(
        float(effect_size)
    )

    if absolute_value < 0.10:
        return "negligible"

    if absolute_value < 0.30:
        return "small"

    if absolute_value < 0.50:
        return "medium"

    return "large"


def interpret_kendall_w(
    effect_size: float,
) -> str:
    """Kendall W büyüklüğünü sınıflandırır."""

    value = float(effect_size)

    if value < 0.10:
        return "negligible"

    if value < 0.30:
        return "small"

    if value < 0.50:
        return "moderate"

    return "large"


def safe_friedman_test(
    model_values: list[np.ndarray],
) -> tuple[float, float]:
    """Bütün eşleştirilmiş değerler aynı olduğunda güvenli Friedman testi."""

    stacked = np.column_stack(
        model_values
    )

    if np.allclose(
        stacked,
        stacked[:, [0]],
        rtol=0.0,
        atol=0.0,
    ):
        return 0.0, 1.0

    result = friedmanchisquare(
        *model_values
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def safe_wilcoxon(
    advantage_values: np.ndarray,
) -> tuple[float, float, str]:
    """Bütün eşleştirilmiş farklar sıfırken güvenli Wilcoxon testi."""

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


def calculate_rank_biserial(
    advantage_values: np.ndarray,
) -> dict[str, Any]:
    """
    Eşleştirilmiş rank-biserial correlation hesaplar.

    Pozitif sonuç variant_a'nın variant_b'den daha iyi olduğunu gösterir.
    """

    differences = np.asarray(
        advantage_values,
        dtype=np.float64,
    )

    nonzero_mask = ~np.isclose(
        differences,
        0.0,
        rtol=0.0,
        atol=1e-15,
    )

    nonzero_differences = differences[
        nonzero_mask
    ]

    effective_count = int(
        len(nonzero_differences)
    )

    if effective_count == 0:
        return {
            "rank_biserial_correlation": 0.0,
            "effect_magnitude": "negligible",
            "positive_rank_sum": 0.0,
            "negative_rank_sum": 0.0,
            "effective_nonzero_pair_count": 0,
        }

    absolute_ranks = rankdata(
        np.abs(nonzero_differences),
        method="average",
    )

    positive_rank_sum = float(
        absolute_ranks[
            nonzero_differences > 0
        ].sum()
    )

    negative_rank_sum = float(
        absolute_ranks[
            nonzero_differences < 0
        ].sum()
    )

    total_rank_sum = (
        positive_rank_sum
        + negative_rank_sum
    )

    effect_size = float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / total_rank_sum
    )

    return {
        "rank_biserial_correlation": effect_size,
        "effect_magnitude": interpret_rank_biserial(
            effect_size
        ),
        "positive_rank_sum": positive_rank_sum,
        "negative_rank_sum": negative_rank_sum,
        "effective_nonzero_pair_count": effective_count,
    }


def build_paired_matrix(
    frame: pd.DataFrame,
    *,
    model_name: str,
    value_column: str,
) -> pd.DataFrame:
    """Seedleri satır, varyantları sütun yapan eşleştirilmiş matris."""

    model_frame = frame[
        frame["model_name"]
        == model_name
    ]

    paired_matrix = model_frame.pivot(
        index="seed",
        columns="variant_name",
        values=value_column,
    )

    paired_matrix = paired_matrix.reindex(
        index=list(SEEDS),
        columns=list(VARIANT_NAMES),
    )

    if paired_matrix.isna().any().any():
        raise ValueError(
            f"{model_name}/{value_column} matrisinde eksik değer var."
        )

    return paired_matrix


def calculate_average_ranks(
    paired_matrix: pd.DataFrame,
    direction: str,
) -> dict[str, float]:
    """Her seed içinde varyant sıralarını ve ortalama rank değerlerini hesaplar."""

    rank_rows: list[np.ndarray] = []

    for row in paired_matrix.to_numpy(
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
                f"Geçersiz metrik yönü: {direction}"
            )

        rank_rows.append(ranks)

    rank_matrix = np.vstack(
        rank_rows
    )

    mean_ranks = rank_matrix.mean(
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
    """Tek model ve sonuç değişkeni için istatistiksel analiz yapar."""

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
        "friedman_degrees_of_freedom": (
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
        values = paired_matrix[
            variant_name
        ].to_numpy(
            dtype=np.float64
        )

        statistics = calculate_summary_statistics(
            values
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

    raw_p_values: list[
        float
    ] = []

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
            advantage_values = raw_difference

        else:
            advantage_values = -raw_difference

        (
            wilcoxon_statistic,
            raw_p_value,
            method_note,
        ) = safe_wilcoxon(
            advantage_values
        )

        effect = calculate_rank_biserial(
            advantage_values
        )

        win_count = int(
            np.sum(
                advantage_values > 1e-15
            )
        )

        loss_count = int(
            np.sum(
                advantage_values < -1e-15
            )
        )

        tie_count = int(
            len(advantage_values)
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
                    len(advantage_values)
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
                    np.median(raw_difference)
                ),
                "mean_advantage_a_over_b": float(
                    advantage_values.mean()
                ),
                "median_advantage_a_over_b": float(
                    np.median(advantage_values)
                ),
                "wilcoxon_statistic": wilcoxon_statistic,
                "wilcoxon_raw_p_value": raw_p_value,
                "wilcoxon_method": method_note,
                "rank_biserial_correlation": effect[
                    "rank_biserial_correlation"
                ],
                "effect_magnitude": effect[
                    "effect_magnitude"
                ],
                "positive_rank_sum": effect[
                    "positive_rank_sum"
                ],
                "negative_rank_sum": effect[
                    "negative_rank_sum"
                ],
                "effective_nonzero_pair_count": effect[
                    "effective_nonzero_pair_count"
                ],
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
            interpretation = (
                "not_confirmatory_because_friedman_not_significant"
            )

        elif adjusted_p_value < alpha:
            interpretation = (
                "significant_after_holm"
            )

        else:
            interpretation = (
                "not_significant_after_holm"
            )

        record[
            "interpretation_status"
        ] = interpretation

    return (
        omnibus_record,
        pairwise_records,
        descriptive_records,
    )


# ==========================================================
# PAIRED DELTAS AND TRADE-OFFS
# ==========================================================

def build_paired_delta_records(
    *,
    model_name: str,
    endpoint_id: str,
    endpoint_display_name: str,
    endpoint_family: str,
    direction: str,
    paired_matrix: pd.DataFrame,
) -> list[dict[str, Any]]:
    """P25/P50/P75 sonuçlarının FP32'ye göre eşleştirilmiş farklarını özetler."""

    fp32_values = paired_matrix[
        "FP32"
    ].to_numpy(
        dtype=np.float64
    )

    records: list[
        dict[str, Any]
    ] = []

    for pruning_variant in PRUNING_VARIANTS:
        pruning_values = paired_matrix[
            pruning_variant
        ].to_numpy(
            dtype=np.float64
        )

        raw_delta = (
            pruning_values
            - fp32_values
        )

        if direction == "higher_is_better":
            performance_advantage = raw_delta

        else:
            performance_advantage = -raw_delta

        raw_statistics = calculate_summary_statistics(
            raw_delta
        )

        advantage_statistics = calculate_summary_statistics(
            performance_advantage
        )

        improved_count = int(
            np.sum(
                performance_advantage > 1e-15
            )
        )

        degraded_count = int(
            np.sum(
                performance_advantage < -1e-15
            )
        )

        unchanged_count = int(
            len(performance_advantage)
            - improved_count
            - degraded_count
        )

        records.append(
            {
                "model_name": model_name,
                "endpoint_id": endpoint_id,
                "endpoint_display_name": endpoint_display_name,
                "endpoint_family": endpoint_family,
                "direction": direction,
                "pruning_variant": pruning_variant,
                "paired_seed_count": int(
                    len(raw_delta)
                ),
                "mean_raw_delta_pruned_minus_fp32": raw_statistics[
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
                "pruned_seed_improvements": improved_count,
                "seed_ties": unchanged_count,
                "pruned_seed_degradations": degraded_count,
            }
        )

    return records


def ensure_group_constant(
    frame: pd.DataFrame,
    column_name: str,
    model_name: str,
    pruning_name: str,
) -> float:
    """Bir kaynak metriğinin beş seed boyunca sabit olduğunu doğrular."""

    unique_values = frame[
        column_name
    ].drop_duplicates()

    if len(unique_values) != 1:
        raise RuntimeError(
            f"{model_name}/{pruning_name} için "
            f"{column_name} seedler boyunca sabit değil."
        )

    return float(
        unique_values.iloc[0]
    )


def build_tradeoff_summary(
    pruning_frame: pd.DataFrame,
    delta_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Parametre, MAC ve Macro F1 ödünleşim tablosunu oluşturur."""

    records: list[
        dict[str, Any]
    ] = []

    macro_delta_frame = delta_frame[
        delta_frame["endpoint_id"]
        == "test_macro_f1"
    ]

    for model_name in MODEL_NAMES:
        for pruning_name in PRUNING_VARIANTS:
            run_subset = pruning_frame[
                (
                    pruning_frame["model_name"]
                    == model_name
                )
                & (
                    pruning_frame["pruning_name"]
                    == pruning_name
                )
            ]

            if len(run_subset) != len(SEEDS):
                raise RuntimeError(
                    f"{model_name}/{pruning_name} için "
                    "beş budama sonucu bulunmuyor."
                )

            macro_statistics = calculate_summary_statistics(
                run_subset[
                    "test_macro_f1"
                ].to_numpy(
                    dtype=np.float64
                )
            )

            delta_row = macro_delta_frame[
                (
                    macro_delta_frame["model_name"]
                    == model_name
                )
                & (
                    macro_delta_frame["pruning_variant"]
                    == pruning_name
                )
            ]

            if len(delta_row) != 1:
                raise RuntimeError(
                    f"{model_name}/{pruning_name} Macro F1 "
                    "delta kaydı bulunamadı."
                )

            delta_row = delta_row.iloc[0]

            original_parameters = int(
                ensure_group_constant(
                    run_subset,
                    "original_parameter_count",
                    model_name,
                    pruning_name,
                )
            )

            compact_parameters = int(
                ensure_group_constant(
                    run_subset,
                    "compact_parameter_count",
                    model_name,
                    pruning_name,
                )
            )

            parameter_reduction = ensure_group_constant(
                run_subset,
                "parameter_reduction_ratio",
                model_name,
                pruning_name,
            )

            original_macs = int(
                ensure_group_constant(
                    run_subset,
                    "original_weight_layer_macs",
                    model_name,
                    pruning_name,
                )
            )

            compact_macs = int(
                ensure_group_constant(
                    run_subset,
                    "compact_weight_layer_macs",
                    model_name,
                    pruning_name,
                )
            )

            mac_reduction = ensure_group_constant(
                run_subset,
                "mac_reduction_ratio",
                model_name,
                pruning_name,
            )

            records.append(
                {
                    "model_name": model_name,
                    "pruning_name": pruning_name,
                    "seed_count": len(SEEDS),
                    "original_parameter_count": original_parameters,
                    "compact_parameter_count": compact_parameters,
                    "parameter_reduction_ratio": parameter_reduction,
                    "parameter_reduction_percent": (
                        parameter_reduction * 100.0
                    ),
                    "original_weight_layer_macs": original_macs,
                    "compact_weight_layer_macs": compact_macs,
                    "mac_reduction_ratio": mac_reduction,
                    "mac_reduction_percent": (
                        mac_reduction * 100.0
                    ),
                    "mean_test_macro_f1": macro_statistics[
                        "mean"
                    ],
                    "test_macro_f1_standard_deviation": macro_statistics[
                        "standard_deviation"
                    ],
                    "test_macro_f1_ci95_lower": macro_statistics[
                        "ci95_lower"
                    ],
                    "test_macro_f1_ci95_upper": macro_statistics[
                        "ci95_upper"
                    ],
                    "mean_macro_f1_delta_vs_fp32": float(
                        delta_row[
                            "mean_raw_delta_pruned_minus_fp32"
                        ]
                    ),
                    "macro_f1_delta_ci95_lower": float(
                        delta_row[
                            "raw_delta_ci95_lower"
                        ]
                    ),
                    "macro_f1_delta_ci95_upper": float(
                        delta_row[
                            "raw_delta_ci95_upper"
                        ]
                    ),
                    "seed_improvements_vs_fp32": int(
                        delta_row[
                            "pruned_seed_improvements"
                        ]
                    ),
                    "seed_ties_vs_fp32": int(
                        delta_row[
                            "seed_ties"
                        ]
                    ),
                    "seed_degradations_vs_fp32": int(
                        delta_row[
                            "pruned_seed_degradations"
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Bütün FP32-budama istatistiksel analizlerini çalıştırır."""

    args = parse_arguments()

    alpha = float(args.alpha)

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "Alpha 0 ile 1 arasında olmalıdır."
        )

    fp32_run_file = args.fp32_run_file.resolve()
    pruning_run_file = args.pruning_run_file.resolve()

    fp32_per_class_file = (
        args.fp32_per_class_file.resolve()
    )

    pruning_per_class_file = (
        args.pruning_per_class_file.resolve()
    )

    pruning_protocol_file = (
        args.pruning_protocol_file.resolve()
    )

    report_directory = (
        args.report_directory.resolve()
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    friedman_file = (
        report_directory
        / "nbaiot_family3_pruning_friedman_results.csv"
    )

    pairwise_file = (
        report_directory
        / "nbaiot_family3_pruning_wilcoxon_holm_results.csv"
    )

    descriptive_file = (
        report_directory
        / "nbaiot_family3_pruning_statistical_descriptives.csv"
    )

    delta_file = (
        report_directory
        / "nbaiot_family3_pruning_paired_deltas.csv"
    )

    tradeoff_file = (
        report_directory
        / "nbaiot_family3_pruning_tradeoff_summary.csv"
    )

    summary_file = (
        report_directory
        / "nbaiot_family3_pruning_statistical_summary.json"
    )

    output_files = (
        friedman_file,
        pairwise_file,
        descriptive_file,
        delta_file,
        tradeoff_file,
        summary_file,
    )

    if (
        any(
            output_file.exists()
            for output_file in output_files
        )
        and not args.overwrite
    ):
        raise FileExistsError(
            "Budama istatistik çıktıları zaten mevcut. "
            "--overwrite kullan."
        )

    protocol = validate_pruning_protocol(
        pruning_protocol_file
    )

    fp32_runs = load_fp32_runs(
        fp32_run_file
    )

    pruning_runs = load_pruning_runs(
        pruning_run_file
    )

    run_results = combine_run_results(
        fp32_runs,
        pruning_runs,
    )

    fp32_per_class = load_fp32_per_class(
        fp32_per_class_file
    )

    pruning_per_class = load_pruning_per_class(
        pruning_per_class_file
    )

    per_class_results = combine_per_class_results(
        fp32_per_class,
        pruning_per_class,
    )

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

    print("=" * 78)
    print("N-BaIoT Family-3 FP32 ve Budama İstatistiksel Analizi")
    print("=" * 78)
    print(
        "Modeller   : "
        + ", ".join(MODEL_NAMES)
    )
    print(
        "Varyantlar : "
        + ", ".join(VARIANT_NAMES)
    )
    print(
        "Seedler    : "
        + ", ".join(
            str(seed)
            for seed in SEEDS
        )
    )
    print(
        "Ana metrik : test_macro_f1"
    )
    print(
        "Holm ailesi: model ve sonuç değişkeni başına 6 ikili karşılaştırma"
    )
    print(
        f"Alpha      : {alpha}"
    )
    print("=" * 78)

    # ------------------------------------------------------
    # Global endpoints
    # ------------------------------------------------------

    for model_name in MODEL_NAMES:
        for endpoint_id, metadata in (
            GLOBAL_ENDPOINTS.items()
        ):
            paired_matrix = build_paired_matrix(
                run_results,
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
                paired_matrix=paired_matrix,
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
                build_paired_delta_records(
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
                    paired_matrix=paired_matrix,
                )
            )

    # ------------------------------------------------------
    # Per-class FNR endpoints
    # ------------------------------------------------------

    for model_name in MODEL_NAMES:
        for class_name in TARGET_CLASSES:
            class_frame = per_class_results[
                per_class_results["class_name"]
                == class_name
            ]

            endpoint_id = (
                f"{class_name}_false_negative_rate"
            )

            endpoint_display_name = (
                f"{class_name} False Negative Rate"
            )

            paired_matrix = build_paired_matrix(
                class_frame,
                model_name=model_name,
                value_column="false_negative_rate",
            )

            (
                omnibus_record,
                endpoint_pairwise,
                endpoint_descriptives,
            ) = analyze_endpoint(
                model_name=model_name,
                endpoint_id=endpoint_id,
                endpoint_display_name=endpoint_display_name,
                endpoint_family="secondary_security",
                direction="lower_is_better",
                paired_matrix=paired_matrix,
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
                build_paired_delta_records(
                    model_name=model_name,
                    endpoint_id=endpoint_id,
                    endpoint_display_name=endpoint_display_name,
                    endpoint_family="secondary_security",
                    direction="lower_is_better",
                    paired_matrix=paired_matrix,
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

    if len(omnibus_frame) != expected_endpoint_count:
        raise RuntimeError(
            "Beklenen Friedman analiz sayısı oluşmadı."
        )

    expected_pairwise_count = (
        expected_endpoint_count
        * 6
    )

    if len(pairwise_frame) != expected_pairwise_count:
        raise RuntimeError(
            "Beklenen Wilcoxon analiz sayısı oluşmadı."
        )

    if (
        pairwise_frame["paired_seed_count"]
        != len(SEEDS)
    ).any():
        raise RuntimeError(
            "İkili analizlerden biri beş eşleştirilmiş seed içermiyor."
        )

    tradeoff_frame = build_tradeoff_summary(
        pruning_runs,
        delta_frame,
    )

    write_csv_atomic(
        omnibus_frame,
        friedman_file,
    )

    write_csv_atomic(
        pairwise_frame,
        pairwise_file,
    )

    write_csv_atomic(
        descriptive_frame,
        descriptive_file,
    )

    write_csv_atomic(
        delta_frame,
        delta_file,
    )

    write_csv_atomic(
        tradeoff_frame,
        tradeoff_file,
    )

    # ------------------------------------------------------
    # Primary endpoint summary
    # ------------------------------------------------------

    primary_summary: dict[
        str,
        Any
    ] = {}

    for model_name in MODEL_NAMES:
        primary_omnibus = omnibus_frame[
            (
                omnibus_frame["model_name"]
                == model_name
            )
            & (
                omnibus_frame["endpoint_id"]
                == "test_macro_f1"
            )
        ].iloc[0]

        primary_descriptives = (
            descriptive_frame[
                (
                    descriptive_frame["model_name"]
                    == model_name
                )
                & (
                    descriptive_frame["endpoint_id"]
                    == "test_macro_f1"
                )
            ]
            .copy()
            .sort_values(
                by=[
                    "mean",
                    "standard_deviation",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        primary_pairwise = pairwise_frame[
            (
                pairwise_frame["model_name"]
                == model_name
            )
            & (
                pairwise_frame["endpoint_id"]
                == "test_macro_f1"
            )
        ].copy()

        fp32_comparisons = primary_pairwise[
            primary_pairwise["variant_a"]
            == "FP32"
        ]

        primary_summary[
            model_name
        ] = {
            "friedman_chi_square": float(
                primary_omnibus[
                    "friedman_chi_square"
                ]
            ),
            "friedman_degrees_of_freedom": int(
                primary_omnibus[
                    "friedman_degrees_of_freedom"
                ]
            ),
            "friedman_p_value": float(
                primary_omnibus[
                    "friedman_p_value"
                ]
            ),
            "friedman_significant": bool(
                primary_omnibus[
                    "friedman_significant"
                ]
            ),
            "kendall_w": float(
                primary_omnibus[
                    "kendall_w"
                ]
            ),
            "kendall_w_magnitude": str(
                primary_omnibus[
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
                    primary_descriptives.itertuples(
                        index=False
                    )
                )
            ],
            "fp32_pairwise_comparisons": [
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
                    "fp32_seed_wins": int(
                        row.variant_a_seed_wins
                    ),
                    "seed_ties": int(
                        row.seed_ties
                    ),
                    "fp32_seed_losses": int(
                        row.variant_a_seed_losses
                    ),
                    "interpretation_status": str(
                        row.interpretation_status
                    ),
                }
                for row in fp32_comparisons.itertuples(
                    index=False
                )
            ],
        }

    summary_document: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "analysis_name": (
            "N-BaIoT family_3 paired FP32 versus "
            "structured pruning statistical analysis"
        ),
        "analysis_version": "1.0",
        "task": "family_3",
        "models": list(MODEL_NAMES),
        "variants": list(VARIANT_NAMES),
        "paired_seeds": list(SEEDS),
        "paired_seed_count": len(SEEDS),
        "primary_endpoint": "test_macro_f1",
        "secondary_global_endpoints": [
            "test_mcc",
            "test_balanced_accuracy",
        ],
        "security_endpoints": [
            f"{class_name}_false_negative_rate"
            for class_name in TARGET_CLASSES
        ],
        "alpha": float(alpha),
        "statistical_tests": {
            "omnibus": "Friedman chi-square test",
            "omnibus_effect_size": "Kendall's W",
            "pairwise": (
                "Two-sided paired Wilcoxon signed-rank test"
            ),
            "pairwise_effect_size": (
                "Matched-pairs rank-biserial correlation"
            ),
            "multiple_testing": (
                "Holm step-down correction across the six "
                "variant pairs separately within each model "
                "and endpoint"
            ),
            "gatekeeping": (
                "Pairwise findings are confirmatory only when "
                "the corresponding Friedman test is significant."
            ),
        },
        "effect_size_orientation": (
            "Positive pairwise rank-biserial values indicate "
            "that variant_a performs better than variant_b after "
            "accounting for endpoint direction."
        ),
        "primary_endpoint_results": primary_summary,
        "small_sample_note": (
            "Each comparison contains five paired seeds. For five "
            "non-zero paired differences, the minimum attainable "
            "two-sided exact Wilcoxon p-value is typically 0.0625. "
            "Therefore effect sizes, paired deltas and directional "
            "consistency must be interpreted together with p-values."
        ),
        "test_data_policy": (
            "All test results were generated after validation-based "
            "checkpoint selection. Test data were not used for "
            "training, pruning, fine-tuning, early stopping or "
            "hyperparameter selection."
        ),
        "input_artifacts": {
            "fp32_run_file": {
                "path": str(fp32_run_file),
                "sha256": calculate_sha256(
                    fp32_run_file
                ),
            },
            "pruning_run_file": {
                "path": str(pruning_run_file),
                "sha256": calculate_sha256(
                    pruning_run_file
                ),
            },
            "fp32_per_class_file": {
                "path": str(fp32_per_class_file),
                "sha256": calculate_sha256(
                    fp32_per_class_file
                ),
            },
            "pruning_per_class_file": {
                "path": str(
                    pruning_per_class_file
                ),
                "sha256": calculate_sha256(
                    pruning_per_class_file
                ),
            },
            "pruning_protocol_file": {
                "path": str(pruning_protocol_file),
                "sha256": calculate_sha256(
                    pruning_protocol_file
                ),
            },
        },
        "output_artifacts": {
            "friedman_results": str(friedman_file),
            "wilcoxon_holm_results": str(pairwise_file),
            "descriptive_statistics": str(
                descriptive_file
            ),
            "paired_deltas": str(delta_file),
            "tradeoff_summary": str(tradeoff_file),
        },
        "validation_passed": True,
    }

    write_json_atomic(
        summary_document,
        summary_file,
    )

    print()
    print("=" * 78)
    print("FP32 ve Budama İstatistiksel Analizi Tamamlandı")
    print("=" * 78)

    for model_name in MODEL_NAMES:
        primary_result = primary_summary[
            model_name
        ]

        print()
        print(f"{model_name}:")
        print(
            "  Friedman chi-square = "
            f"{primary_result['friedman_chi_square']:.6f}"
        )
        print(
            "  Friedman p          = "
            f"{primary_result['friedman_p_value']:.8f}"
        )
        print(
            "  Anlamlı             = "
            f"{primary_result['friedman_significant']}"
        )
        print(
            "  Kendall W           = "
            f"{primary_result['kendall_w']:.6f} "
            f"({primary_result['kendall_w_magnitude']})"
        )

        print("  Macro F1 sıralaması:")

        for ranking in primary_result[
            "variant_ranking"
        ]:
            print(
                f"    {ranking['rank']}. "
                f"{ranking['variant_name']:4s} | "
                f"ortalama="
                f"{ranking['mean_test_macro_f1']:.6f} | "
                f"SS="
                f"{ranking['standard_deviation']:.6f} | "
                f"ortalama rank="
                f"{ranking['average_rank']:.3f}"
            )

        print("  FP32 karşılaştırmaları:")

        for comparison in primary_result[
            "fp32_pairwise_comparisons"
        ]:
            print(
                "    "
                f"{comparison['comparison']:12s} | "
                f"raw_p="
                f"{comparison['raw_p_value']:.8f} | "
                f"Holm_p="
                f"{comparison['holm_adjusted_p_value']:.8f} | "
                f"r_rb="
                f"{comparison['rank_biserial_correlation']:.4f} "
                f"({comparison['effect_magnitude']})"
            )

    print()
    print(
        f"Doğrulanan model-varyant-seed sonucu : "
        f"{len(run_results)}"
    )
    print(
        f"Friedman analiz sayısı               : "
        f"{len(omnibus_frame)}"
    )
    print(
        f"Wilcoxon karşılaştırma sayısı         : "
        f"{len(pairwise_frame)}"
    )
    print(
        "Test seçimde kullanıldı              : False"
    )
    print(
        "Doğrulama geçti                      : True"
    )
    print()
    print(
        f"Friedman sonuçları : {friedman_file}"
    )
    print(
        f"Wilcoxon + Holm    : {pairwise_file}"
    )
    print(
        f"Betimsel sonuçlar  : {descriptive_file}"
    )
    print(
        f"Eşleştirilmiş farklar: {delta_file}"
    )
    print(
        f"Kaynak ödünleşimi  : {tradeoff_file}"
    )
    print(
        f"JSON özet          : {summary_file}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()