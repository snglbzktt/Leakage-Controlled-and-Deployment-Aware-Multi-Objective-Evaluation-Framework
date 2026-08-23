"""
N-BaIoT family_3 FP32 baseline istatistiksel analizi.

Girdiler:
- 23. aşamanın model-seed test sonuçları
- 23. aşamanın sınıf bazlı test sonuçları
- 22. aşamada kilitlenen FP32 protokolü

Analiz:
1. Friedman omnibus testi
2. Kendall's W omnibus etki büyüklüğü
3. Eşleştirilmiş Wilcoxon signed-rank testleri
4. Her sonucun üç ikili karşılaştırması için Holm düzeltmesi
5. Rank-biserial correlation
6. Seed bazında kazanma, beraberlik ve kaybetme sayıları

Bilimsel yorumlama:
- test_macro_f1 ana doğrulayıcı metriktir.
- Sınıf bazlı FNR güvenlik odaklı ikincil sonuçtur.
- Diğer genel metrikler keşifsel/ikincil sonuçlardır.
- Wilcoxon ikili sonuçları, ilgili Friedman testi anlamlıysa
  doğrulayıcı olarak yorumlanır.
- Holm düzeltmesi her sonuç değişkeninin üç model çifti
  içerisinde ayrı uygulanır.
- Test sonuçları eğitim veya hiperparametre seçiminde kullanılmaz.
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
    wilcoxon,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUN_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_family3_fp32_baseline_runs.csv"
)

DEFAULT_PER_CLASS_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_family3_fp32_baseline_per_class_runs.csv"
)

DEFAULT_PROTOCOL_FILE = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_fp32_baseline_protocol_v1.json"
)

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

EXPECTED_MODELS = (
    "tinyml_mlp",
    "compact_dnn",
    "tiny_1d_cnn",
)

EXPECTED_SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

EXPECTED_CLASSES = (
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
    "test_accuracy": {
        "display_name": "Test Accuracy",
        "endpoint_family": "secondary_global",
        "direction": "higher_is_better",
    },
    "test_macro_precision": {
        "display_name": "Test Macro Precision",
        "endpoint_family": "secondary_global",
        "direction": "higher_is_better",
    },
    "test_macro_recall": {
        "display_name": "Test Macro Recall",
        "endpoint_family": "secondary_global",
        "direction": "higher_is_better",
    },
    "test_weighted_f1": {
        "display_name": "Test Weighted F1",
        "endpoint_family": "secondary_global",
        "direction": "higher_is_better",
    },
}

ALPHA = 0.05


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "Üç FP32 modelin eşleştirilmiş beş seed sonucunu "
            "Friedman, Wilcoxon, Holm ve etki büyüklükleriyle analiz eder."
        )
    )

    parser.add_argument(
        "--run-file",
        type=Path,
        default=DEFAULT_RUN_FILE,
    )

    parser.add_argument(
        "--per-class-file",
        type=Path,
        default=DEFAULT_PER_CLASS_FILE,
    )

    parser.add_argument(
        "--protocol-file",
        type=Path,
        default=DEFAULT_PROTOCOL_FILE,
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


def read_json(
    input_file: Path,
) -> dict[str, Any]:
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


# ==========================================================
# INPUT VALIDATION
# ==========================================================

def load_and_validate_protocol(
    protocol_file: Path,
) -> dict[str, Any]:
    """Kilitli FP32 protokolünü doğrular."""

    protocol = read_json(protocol_file)

    if protocol.get("protocol_version") != "1.0":
        raise ValueError(
            "Beklenmeyen FP32 protokol sürümü."
        )

    if protocol.get("task") != "family_3":
        raise ValueError(
            "FP32 protokol görevi family_3 değil."
        )

    protocol_models = tuple(
        str(model_name)
        for model_name in protocol["models"]
    )

    if protocol_models != EXPECTED_MODELS:
        raise ValueError(
            "Protokoldeki model sırası beklenen sırayla uyuşmuyor."
        )

    protocol_seeds = tuple(
        int(seed)
        for seed in protocol["training_seeds"]
    )

    if protocol_seeds != EXPECTED_SEEDS:
        raise ValueError(
            "Protokoldeki seed sırası beklenen sırayla uyuşmuyor."
        )

    target_classes = tuple(
        str(class_name)
        for class_name in protocol["target_classes"]
    )

    if target_classes != EXPECTED_CLASSES:
        raise ValueError(
            "Protokoldeki family_3 sınıf sırası uyuşmuyor."
        )

    return protocol


def load_and_validate_runs(
    run_file: Path,
) -> pd.DataFrame:
    """Model-seed FP32 test sonuçlarını yükler ve doğrular."""

    if not run_file.exists():
        raise FileNotFoundError(
            f"FP32 seed sonuçları bulunamadı: {run_file}"
        )

    frame = pd.read_csv(run_file)

    required_columns = {
        "model_name",
        "seed",
        "test_evaluation_attempt_count",
        "test_used_for_selection",
        *GLOBAL_ENDPOINTS.keys(),
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "FP32 sonuç dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing_columns))
        )

    frame = frame.copy()

    frame["model_name"] = frame[
        "model_name"
    ].astype(str)

    frame["seed"] = pd.to_numeric(
        frame["seed"],
        errors="raise",
    ).astype("int64")

    if len(frame) != (
        len(EXPECTED_MODELS)
        * len(EXPECTED_SEEDS)
    ):
        raise ValueError(
            "FP32 model-seed sonuç sayısı 15 değil: "
            f"bulunan={len(frame)}"
        )

    if frame.duplicated(
        subset=[
            "model_name",
            "seed",
        ]
    ).any():
        raise ValueError(
            "FP32 sonuçlarında tekrar eden model-seed kaydı var."
        )

    expected_pairs = {
        (
            model_name,
            seed,
        )
        for model_name in EXPECTED_MODELS
        for seed in EXPECTED_SEEDS
    }

    actual_pairs = {
        (
            str(row.model_name),
            int(row.seed),
        )
        for row in frame.itertuples(index=False)
    }

    if actual_pairs != expected_pairs:
        raise ValueError(
            "FP32 sonuçlarındaki model-seed kümesi protokolle uyuşmuyor."
        )

    if (
        pd.to_numeric(
            frame["test_evaluation_attempt_count"],
            errors="raise",
        ).astype("int64")
        != 1
    ).any():
        raise ValueError(
            "Test değerlendirme girişimlerinden biri 1 değil."
        )

    selection_values = (
        frame["test_used_for_selection"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not selection_values.isin(
        {
            "false",
            "0",
            "no",
        }
    ).all():
        raise RuntimeError(
            "Test sonuçlarından biri model seçimine katılmış."
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
                f"{metric_name} içinde sonlu olmayan değer var."
            )

    return frame


def load_and_validate_per_class(
    per_class_file: Path,
) -> pd.DataFrame:
    """Sınıf bazlı FP32 sonuçlarını yükler ve doğrular."""

    if not per_class_file.exists():
        raise FileNotFoundError(
            f"Sınıf bazlı FP32 sonuçları bulunamadı: {per_class_file}"
        )

    frame = pd.read_csv(per_class_file)

    required_columns = {
        "model_name",
        "seed",
        "class_index",
        "class_name",
        "false_negative_rate",
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Sınıf bazlı sonuç dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing_columns))
        )

    frame = frame.copy()

    frame["model_name"] = (
        frame["model_name"]
        .astype(str)
    )

    frame["seed"] = pd.to_numeric(
        frame["seed"],
        errors="raise",
    ).astype("int64")

    frame["class_index"] = pd.to_numeric(
        frame["class_index"],
        errors="raise",
    ).astype("int64")

    frame["class_name"] = (
        frame["class_name"]
        .astype(str)
    )

    frame["false_negative_rate"] = pd.to_numeric(
        frame["false_negative_rate"],
        errors="raise",
    )

    expected_record_count = (
        len(EXPECTED_MODELS)
        * len(EXPECTED_SEEDS)
        * len(EXPECTED_CLASSES)
    )

    if len(frame) != expected_record_count:
        raise ValueError(
            "Sınıf bazlı FP32 sonuç sayısı 45 değil: "
            f"bulunan={len(frame)}"
        )

    if frame.duplicated(
        subset=[
            "model_name",
            "seed",
            "class_name",
        ]
    ).any():
        raise ValueError(
            "Tekrarlanan model-seed-sınıf kaydı var."
        )

    actual_classes = tuple(
        frame[
            [
                "class_index",
                "class_name",
            ]
        ]
        .drop_duplicates()
        .sort_values("class_index")
        ["class_name"]
        .tolist()
    )

    if actual_classes != EXPECTED_CLASSES:
        raise ValueError(
            "Sınıf bazlı sonuçlardaki sınıf sırası uyuşmuyor."
        )

    if not np.isfinite(
        frame["false_negative_rate"].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "FNR değerlerinde sonlu olmayan sayı var."
        )

    if (
        frame["false_negative_rate"] < 0
    ).any() or (
        frame["false_negative_rate"] > 1
    ).any():
        raise ValueError(
            "FNR değerlerinden biri [0, 1] aralığı dışında."
        )

    for model_name in EXPECTED_MODELS:
        for class_name in EXPECTED_CLASSES:
            actual_seeds = tuple(
                sorted(
                    frame[
                        (
                            frame["model_name"]
                            == model_name
                        )
                        & (
                            frame["class_name"]
                            == class_name
                        )
                    ]["seed"].tolist()
                )
            )

            if actual_seeds != tuple(
                sorted(EXPECTED_SEEDS)
            ):
                raise ValueError(
                    f"{model_name}/{class_name} için seed kümesi eksik."
                )

    return frame


# ==========================================================
# STATISTICAL HELPERS
# ==========================================================

def holm_adjust(
    p_values: list[float],
) -> list[float]:
    """
    Holm step-down düzeltmesini uygular.

    Düzeltilmiş p-değerleri orijinal karşılaştırma sırasına döndürülür.
    """

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
            "Holm girdileri [0, 1] aralığında olmalıdır."
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
    """Rank-biserial korelasyon büyüklüğünü sezgisel etiketler."""

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
    """Kendall W büyüklüğünü sezgisel etiketler."""

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
    """Tam eşit değerlerde hata vermeyen Friedman testi."""

    stacked = np.column_stack(
        model_values
    )

    if np.allclose(
        stacked,
        stacked[:, [0]],
        rtol=0.0,
        atol=0.0,
    ):
        return (
            0.0,
            1.0,
        )

    result = friedmanchisquare(
        *model_values
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def calculate_rank_biserial(
    advantage_values: np.ndarray,
) -> dict[str, Any]:
    """
    Eşleştirilmiş rank-biserial correlation hesaplar.

    advantage_values pozitifse model A model B'den daha iyidir.
    Sıfır farklar etki büyüklüğü hesabından çıkarılır.
    """

    differences = np.asarray(
        advantage_values,
        dtype=np.float64,
    )

    if differences.ndim != 1:
        raise ValueError(
            "Eşleştirilmiş fark dizisi tek boyutlu olmalıdır."
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

    effective_sample_count = int(
        len(nonzero_differences)
    )

    if effective_sample_count == 0:
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

    rank_biserial = float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / total_rank_sum
        if total_rank_sum > 0
        else 0.0
    )

    return {
        "rank_biserial_correlation": (
            rank_biserial
        ),
        "effect_magnitude": (
            interpret_rank_biserial(
                rank_biserial
            )
        ),
        "positive_rank_sum": (
            positive_rank_sum
        ),
        "negative_rank_sum": (
            negative_rank_sum
        ),
        "effective_nonzero_pair_count": (
            effective_sample_count
        ),
    }


def safe_wilcoxon(
    advantage_values: np.ndarray,
) -> tuple[float, float, str]:
    """Sıfır farkların tamamında hata vermeyen Wilcoxon testi."""

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


def build_paired_matrix(
    frame: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Seedleri satır, modelleri sütun yapan eşleştirilmiş matris üretir."""

    pivot = frame.pivot(
        index="seed",
        columns="model_name",
        values=value_column,
    )

    pivot = pivot.reindex(
        index=list(EXPECTED_SEEDS),
        columns=list(EXPECTED_MODELS),
    )

    if pivot.isna().any().any():
        raise ValueError(
            f"{value_column} eşleştirilmiş matrisinde eksik değer var."
        )

    return pivot


# ==========================================================
# ENDPOINT ANALYSIS
# ==========================================================

def analyze_endpoint(
    *,
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
    """Tek bir sonuç değişkeni için omnibus ve ikili analizleri çalıştırır."""

    if direction not in {
        "higher_is_better",
        "lower_is_better",
    }:
        raise ValueError(
            f"Geçersiz metrik yönü: {direction}"
        )

    model_arrays = [
        paired_matrix[
            model_name
        ].to_numpy(
            dtype=np.float64
        )
        for model_name in EXPECTED_MODELS
    ]

    for values in model_arrays:
        if not np.isfinite(values).all():
            raise ValueError(
                f"{endpoint_id} içinde sonlu olmayan değer var."
            )

    friedman_statistic, friedman_p_value = (
        safe_friedman_test(
            model_arrays
        )
    )

    subject_count = int(
        len(paired_matrix)
    )

    treatment_count = int(
        len(EXPECTED_MODELS)
    )

    kendall_w = float(
        friedman_statistic
        / (
            subject_count
            * (
                treatment_count - 1
            )
        )
        if subject_count > 0
        and treatment_count > 1
        else 0.0
    )

    kendall_w = float(
        np.clip(
            kendall_w,
            0.0,
            1.0,
        )
    )

    omnibus_significant = bool(
        friedman_p_value < alpha
    )

    omnibus_record = {
        "endpoint_id": endpoint_id,
        "endpoint_display_name": (
            endpoint_display_name
        ),
        "endpoint_family": (
            endpoint_family
        ),
        "direction": direction,
        "seed_count": subject_count,
        "model_count": treatment_count,
        "friedman_chi_square": (
            friedman_statistic
        ),
        "friedman_degrees_of_freedom": (
            treatment_count - 1
        ),
        "friedman_p_value": (
            friedman_p_value
        ),
        "alpha": float(alpha),
        "friedman_significant": (
            omnibus_significant
        ),
        "kendall_w": kendall_w,
        "kendall_w_magnitude": (
            interpret_kendall_w(
                kendall_w
            )
        ),
    }

    descriptive_records: list[
        dict[str, Any]
    ] = []

    for model_name in EXPECTED_MODELS:
        values = paired_matrix[
            model_name
        ].to_numpy(
            dtype=np.float64
        )

        descriptive_records.append(
            {
                "endpoint_id": endpoint_id,
                "endpoint_display_name": (
                    endpoint_display_name
                ),
                "endpoint_family": (
                    endpoint_family
                ),
                "direction": direction,
                "model_name": model_name,
                "seed_count": int(
                    len(values)
                ),
                "mean": float(
                    values.mean()
                ),
                "standard_deviation": float(
                    values.std(ddof=1)
                    if len(values) > 1
                    else 0.0
                ),
                "median": float(
                    np.median(values)
                ),
                "minimum": float(
                    values.min()
                ),
                "maximum": float(
                    values.max()
                ),
            }
        )

    pairwise_records: list[
        dict[str, Any]
    ] = []

    raw_p_values: list[
        float
    ] = []

    for model_a, model_b in combinations(
        EXPECTED_MODELS,
        2,
    ):
        values_a = paired_matrix[
            model_a
        ].to_numpy(
            dtype=np.float64
        )

        values_b = paired_matrix[
            model_b
        ].to_numpy(
            dtype=np.float64
        )

        raw_differences = (
            values_a
            - values_b
        )

        if direction == "higher_is_better":
            advantage_values = (
                raw_differences
            )

        else:
            advantage_values = (
                -raw_differences
            )

        wilcoxon_statistic, raw_p_value, method_note = (
            safe_wilcoxon(
                advantage_values
            )
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
                "endpoint_id": endpoint_id,
                "endpoint_display_name": (
                    endpoint_display_name
                ),
                "endpoint_family": (
                    endpoint_family
                ),
                "direction": direction,
                "model_a": model_a,
                "model_b": model_b,
                "paired_seed_count": int(
                    len(advantage_values)
                ),
                "model_a_mean": float(
                    values_a.mean()
                ),
                "model_b_mean": float(
                    values_b.mean()
                ),
                "mean_metric_difference_a_minus_b": float(
                    raw_differences.mean()
                ),
                "median_metric_difference_a_minus_b": float(
                    np.median(
                        raw_differences
                    )
                ),
                "mean_advantage_a_over_b": float(
                    advantage_values.mean()
                ),
                "median_advantage_a_over_b": float(
                    np.median(
                        advantage_values
                    )
                ),
                "wilcoxon_statistic": (
                    wilcoxon_statistic
                ),
                "wilcoxon_raw_p_value": (
                    raw_p_value
                ),
                "wilcoxon_method": (
                    method_note
                ),
                "rank_biserial_correlation": (
                    effect[
                        "rank_biserial_correlation"
                    ]
                ),
                "effect_magnitude": (
                    effect[
                        "effect_magnitude"
                    ]
                ),
                "positive_rank_sum": (
                    effect[
                        "positive_rank_sum"
                    ]
                ),
                "negative_rank_sum": (
                    effect[
                        "negative_rank_sum"
                    ]
                ),
                "effective_nonzero_pair_count": (
                    effect[
                        "effective_nonzero_pair_count"
                    ]
                ),
                "model_a_seed_wins": (
                    win_count
                ),
                "seed_ties": tie_count,
                "model_a_seed_losses": (
                    loss_count
                ),
                "friedman_p_value": (
                    friedman_p_value
                ),
                "friedman_significant": (
                    omnibus_significant
                ),
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
            adjusted_p_value
            < alpha
        )

        record[
            "gatekept_significant"
        ] = bool(
            omnibus_significant
            and adjusted_p_value < alpha
        )

        if not omnibus_significant:
            interpretation_status = (
                "not_confirmatory_because_friedman_not_significant"
            )

        elif adjusted_p_value < alpha:
            interpretation_status = (
                "significant_after_holm"
            )

        else:
            interpretation_status = (
                "not_significant_after_holm"
            )

        record[
            "interpretation_status"
        ] = interpretation_status

    return (
        omnibus_record,
        pairwise_records,
        descriptive_records,
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Bütün FP32 istatistiksel analizlerini çalıştırır."""

    args = parse_arguments()

    alpha = float(
        args.alpha
    )

    if not (
        0.0
        < alpha
        < 1.0
    ):
        raise ValueError(
            "Alpha 0 ile 1 arasında olmalıdır."
        )

    run_file = (
        args.run_file.resolve()
    )

    per_class_file = (
        args.per_class_file.resolve()
    )

    protocol_file = (
        args.protocol_file.resolve()
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
        / "nbaiot_family3_fp32_friedman_results.csv"
    )

    pairwise_file = (
        report_directory
        / "nbaiot_family3_fp32_wilcoxon_holm_results.csv"
    )

    descriptive_file = (
        report_directory
        / "nbaiot_family3_fp32_statistical_descriptives.csv"
    )

    summary_file = (
        report_directory
        / "nbaiot_family3_fp32_statistical_summary.json"
    )

    output_files = (
        friedman_file,
        pairwise_file,
        descriptive_file,
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
            "İstatistik analiz çıktıları zaten mevcut. "
            "--overwrite kullan."
        )

    protocol = load_and_validate_protocol(
        protocol_file
    )

    run_frame = load_and_validate_runs(
        run_file
    )

    per_class_frame = (
        load_and_validate_per_class(
            per_class_file
        )
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

    print("=" * 78)
    print("N-BaIoT Family-3 FP32 İstatistiksel Analizi")
    print("=" * 78)
    print(
        "Modeller       : "
        + ", ".join(EXPECTED_MODELS)
    )
    print(
        "Eşleştirilmiş seedler: "
        + ", ".join(
            str(seed)
            for seed in EXPECTED_SEEDS
        )
    )
    print(
        "Ana metrik     : test_macro_f1"
    )
    print(
        "Güvenlik metriği: sınıf bazlı false_negative_rate"
    )
    print(f"Alpha          : {alpha}")
    print(
        "Holm kapsamı   : Sonuç değişkeni başına 3 model çifti"
    )
    print("=" * 78)

    # ------------------------------------------------------
    # Global endpoints
    # ------------------------------------------------------

    for endpoint_id, metadata in (
        GLOBAL_ENDPOINTS.items()
    ):
        paired_matrix = build_paired_matrix(
            frame=run_frame,
            value_column=endpoint_id,
        )

        (
            omnibus_record,
            endpoint_pairwise,
            endpoint_descriptives,
        ) = analyze_endpoint(
            endpoint_id=endpoint_id,
            endpoint_display_name=(
                metadata[
                    "display_name"
                ]
            ),
            endpoint_family=(
                metadata[
                    "endpoint_family"
                ]
            ),
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

    # ------------------------------------------------------
    # Per-class FNR endpoints
    # ------------------------------------------------------

    for class_name in EXPECTED_CLASSES:
        class_frame = per_class_frame[
            per_class_frame[
                "class_name"
            ]
            == class_name
        ].copy()

        endpoint_id = (
            f"{class_name}_false_negative_rate"
        )

        paired_matrix = build_paired_matrix(
            frame=class_frame,
            value_column=(
                "false_negative_rate"
            ),
        )

        (
            omnibus_record,
            endpoint_pairwise,
            endpoint_descriptives,
        ) = analyze_endpoint(
            endpoint_id=endpoint_id,
            endpoint_display_name=(
                f"{class_name} False Negative Rate"
            ),
            endpoint_family=(
                "secondary_security"
            ),
            direction=(
                "lower_is_better"
            ),
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

    omnibus_frame = pd.DataFrame(
        omnibus_records
    )

    pairwise_frame = pd.DataFrame(
        pairwise_records
    )

    descriptive_frame = pd.DataFrame(
        descriptive_records
    )

    expected_endpoint_count = (
        len(GLOBAL_ENDPOINTS)
        + len(EXPECTED_CLASSES)
    )

    if len(omnibus_frame) != (
        expected_endpoint_count
    ):
        raise RuntimeError(
            "Friedman sonuç değişkeni sayısı beklenen değerle uyuşmuyor."
        )

    expected_pairwise_count = (
        expected_endpoint_count
        * 3
    )

    if len(pairwise_frame) != (
        expected_pairwise_count
    ):
        raise RuntimeError(
            "Wilcoxon karşılaştırma sayısı beklenen değerle uyuşmuyor."
        )

    if (
        pairwise_frame[
            "paired_seed_count"
        ]
        != len(EXPECTED_SEEDS)
    ).any():
        raise RuntimeError(
            "Wilcoxon analizlerinden biri beş eşleştirilmiş seed içermiyor."
        )

    write_csv_atomic(
        frame=omnibus_frame,
        output_file=friedman_file,
    )

    write_csv_atomic(
        frame=pairwise_frame,
        output_file=pairwise_file,
    )

    write_csv_atomic(
        frame=descriptive_frame,
        output_file=descriptive_file,
    )

    primary_friedman = omnibus_frame[
        omnibus_frame[
            "endpoint_id"
        ]
        == "test_macro_f1"
    ].iloc[0]

    primary_pairwise = pairwise_frame[
        pairwise_frame[
            "endpoint_id"
        ]
        == "test_macro_f1"
    ].copy()

    primary_descriptive = (
        descriptive_frame[
            descriptive_frame[
                "endpoint_id"
            ]
            == "test_macro_f1"
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
        .reset_index(
            drop=True
        )
    )

    primary_ranking = [
        {
            "rank": int(index + 1),
            "model_name": str(
                row.model_name
            ),
            "mean_test_macro_f1": float(
                row.mean
            ),
            "standard_deviation": float(
                row.standard_deviation
            ),
            "median": float(
                row.median
            ),
            "minimum": float(
                row.minimum
            ),
            "maximum": float(
                row.maximum
            ),
        }
        for index, row in enumerate(
            primary_descriptive.itertuples(
                index=False
            )
        )
    ]

    primary_pairwise_summary = [
        {
            "model_a": str(
                row.model_a
            ),
            "model_b": str(
                row.model_b
            ),
            "raw_p_value": float(
                row.wilcoxon_raw_p_value
            ),
            "holm_adjusted_p_value": float(
                row.holm_adjusted_p_value
            ),
            "holm_significant": bool(
                row.holm_significant
            ),
            "gatekept_significant": bool(
                row.gatekept_significant
            ),
            "rank_biserial_correlation": float(
                row.rank_biserial_correlation
            ),
            "effect_magnitude": str(
                row.effect_magnitude
            ),
            "model_a_seed_wins": int(
                row.model_a_seed_wins
            ),
            "seed_ties": int(
                row.seed_ties
            ),
            "model_a_seed_losses": int(
                row.model_a_seed_losses
            ),
            "interpretation_status": str(
                row.interpretation_status
            ),
        }
        for row in primary_pairwise.itertuples(
            index=False
        )
    ]

    summary_document: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "analysis_name": (
            "N-BaIoT family_3 FP32 paired multi-seed statistical analysis"
        ),
        "analysis_version": "1.0",
        "task": "family_3",
        "models": list(
            EXPECTED_MODELS
        ),
        "paired_seeds": list(
            EXPECTED_SEEDS
        ),
        "paired_seed_count": int(
            len(EXPECTED_SEEDS)
        ),
        "alpha": float(alpha),
        "primary_endpoint": (
            "test_macro_f1"
        ),
        "security_endpoints": [
            f"{class_name}_false_negative_rate"
            for class_name in EXPECTED_CLASSES
        ],
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
            "pairwise_multiple_testing": (
                "Holm step-down adjustment across the three "
                "model pairs separately within each endpoint"
            ),
            "pairwise_effect_size": (
                "Matched-pairs rank-biserial correlation"
            ),
            "gatekeeping": (
                "Pairwise findings are confirmatory only when "
                "the corresponding Friedman test is significant."
            ),
        },
        "effect_size_orientation": (
            "Positive pairwise rank-biserial values indicate that "
            "model_a is better than model_b after accounting for "
            "the metric direction."
        ),
        "primary_endpoint_result": {
            "friedman_chi_square": float(
                primary_friedman[
                    "friedman_chi_square"
                ]
            ),
            "friedman_degrees_of_freedom": int(
                primary_friedman[
                    "friedman_degrees_of_freedom"
                ]
            ),
            "friedman_p_value": float(
                primary_friedman[
                    "friedman_p_value"
                ]
            ),
            "friedman_significant": bool(
                primary_friedman[
                    "friedman_significant"
                ]
            ),
            "kendall_w": float(
                primary_friedman[
                    "kendall_w"
                ]
            ),
            "kendall_w_magnitude": str(
                primary_friedman[
                    "kendall_w_magnitude"
                ]
            ),
            "model_ranking": (
                primary_ranking
            ),
            "pairwise_results": (
                primary_pairwise_summary
            ),
        },
        "small_sample_note": (
            "The analysis contains five paired training seeds per "
            "architecture. Non-significant findings may reflect both "
            "small differences and limited statistical power."
        ),
        "test_data_policy": (
            "Test results are analyzed only after the FP32 training "
            "protocol and model checkpoints were locked. No test result "
            "was used for training, early stopping, hyperparameter "
            "selection or checkpoint selection."
        ),
        "input_artifacts": {
            "run_file": {
                "path": str(
                    run_file
                ),
                "sha256": calculate_sha256(
                    run_file
                ),
            },
            "per_class_file": {
                "path": str(
                    per_class_file
                ),
                "sha256": calculate_sha256(
                    per_class_file
                ),
            },
            "protocol_file": {
                "path": str(
                    protocol_file
                ),
                "sha256": calculate_sha256(
                    protocol_file
                ),
            },
        },
        "output_artifacts": {
            "friedman_results": str(
                friedman_file
            ),
            "wilcoxon_holm_results": str(
                pairwise_file
            ),
            "descriptive_statistics": str(
                descriptive_file
            ),
        },
        "validation_passed": True,
    }

    write_json_atomic(
        document=summary_document,
        output_file=summary_file,
    )

    print()
    print("=" * 78)
    print("FP32 İstatistiksel Analizi Tamamlandı")
    print("=" * 78)

    print(
        "Ana metrik Friedman testi:"
    )
    print(
        f"  chi-square = "
        f"{primary_friedman['friedman_chi_square']:.6f}"
    )
    print(
        f"  df         = "
        f"{int(primary_friedman['friedman_degrees_of_freedom'])}"
    )
    print(
        f"  p          = "
        f"{primary_friedman['friedman_p_value']:.8f}"
    )
    print(
        f"  anlamlı    = "
        f"{bool(primary_friedman['friedman_significant'])}"
    )
    print(
        f"  Kendall W  = "
        f"{primary_friedman['kendall_w']:.6f} "
        f"({primary_friedman['kendall_w_magnitude']})"
    )

    print()
    print(
        "Test Macro F1 sıralaması:"
    )

    for ranking_record in primary_ranking:
        print(
            f"  {ranking_record['rank']}. "
            f"{ranking_record['model_name']:14s} | "
            f"ortalama="
            f"{ranking_record['mean_test_macro_f1']:.6f} | "
            f"SS="
            f"{ranking_record['standard_deviation']:.6f}"
        )

    print()
    print(
        "Eşleştirilmiş Wilcoxon + Holm:"
    )

    for row in primary_pairwise.itertuples(
        index=False
    ):
        print(
            f"  {row.model_a:14s} vs "
            f"{row.model_b:14s} | "
            f"raw_p={row.wilcoxon_raw_p_value:.8f} | "
            f"Holm_p={row.holm_adjusted_p_value:.8f} | "
            f"r_rb={row.rank_biserial_correlation:.4f} "
            f"({row.effect_magnitude}) | "
            f"sonuç={row.interpretation_status}"
        )

    print()
    print(
        "Doğrulanan model-seed sonucu : 15"
    )
    print(
        "Eşleştirilmiş seed sayısı    : 5"
    )
    print(
        "Test seçimde kullanıldı      : False"
    )
    print(
        "Doğrulama geçti              : True"
    )
    print()
    print(f"Friedman sonuçları : {friedman_file}")
    print(f"Wilcoxon + Holm    : {pairwise_file}")
    print(f"Betimsel sonuçlar  : {descriptive_file}")
    print(f"JSON özet          : {summary_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()