"""
33. Aşama — IEEE Access için yayın kalitesinde tablo ve şekil üretimi.

Girdiler:
- results/reports/nbaiot_family3_overall_statistical_descriptives.csv
- results/reports/nbaiot_family3_overall_efficiency_tradeoff.csv
- results/reports/nbaiot_family3_overall_friedman_results.csv
- results/reports/nbaiot_family3_overall_wilcoxon_holm_results.csv
- results/reports/nbaiot_family3_overall_per_class_runs.csv
- results/reports/nbaiot_family3_overall_pareto_front.csv

Çıktılar:
- Yayın tabloları (CSV)
- Makale şekilleri (PNG ve PDF)
- Türkçe sonuç özeti (Markdown)
- Üretim manifesti (JSON)

Bilimsel sınırlar:
- Yeniden eğitim veya test değerlendirmesi yapılmaz.
- Yalnızca 32. aşamada doğrulanmış raporlar kullanılır.
- Gecikme ve throughput sonuçları host CPU ortamına aittir.
- MCU gecikmesi, enerji tüketimi veya gerçek cihaz dağıtımı iddia edilmez.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "publication"
    / "stage33"
)

DESCRIPTIVES_FILE_NAME = (
    "nbaiot_family3_overall_statistical_descriptives.csv"
)

EFFICIENCY_FILE_NAME = (
    "nbaiot_family3_overall_efficiency_tradeoff.csv"
)

FRIEDMAN_FILE_NAME = (
    "nbaiot_family3_overall_friedman_results.csv"
)

WILCOXON_FILE_NAME = (
    "nbaiot_family3_overall_wilcoxon_holm_results.csv"
)

PER_CLASS_FILE_NAME = (
    "nbaiot_family3_overall_per_class_runs.csv"
)

PARETO_FILE_NAME = (
    "nbaiot_family3_overall_pareto_front.csv"
)

MODEL_ORDER = (
    "tinyml_mlp",
    "compact_dnn",
    "tiny_1d_cnn",
)

MODEL_DISPLAY = {
    "tinyml_mlp": "TinyML-MLP",
    "compact_dnn": "Compact-DNN",
    "tiny_1d_cnn": "Tiny-1D-CNN",
}

VARIANT_ORDER = (
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

CLASS_ORDER = (
    "benign",
    "gafgyt",
    "mirai",
)

CLASS_DISPLAY = {
    "benign": "Benign",
    "gafgyt": "Gafgyt",
    "mirai": "Mirai",
}

MODEL_MARKERS = {
    "tinyml_mlp": "o",
    "compact_dnn": "s",
    "tiny_1d_cnn": "^",
}


def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "32. aşama raporlarından yayın kalitesinde "
            "tablo ve şekiller üretir."
        )
    )

    parser.add_argument(
        "--report-directory",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def require_file(file_path: Path) -> Path:
    """Girdi dosyasının bulunduğunu doğrular."""

    resolved = file_path.resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            f"Girdi dosyası bulunamadı: {resolved}"
        )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Girdi yolu dosya değil: {resolved}"
        )

    return resolved


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


def json_default(value: object) -> object:
    """NumPy ve Path değerlerini JSON uyumlu hâle getirir."""

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        numeric_value = float(value)

        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return None

        return numeric_value

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"{type(value).__name__} JSON ile uyumlu değil."
    )


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
        encoding="utf-8-sig",
    )

    temporary_file.replace(output_file)


def write_text_atomic(
    text: str,
    output_file: Path,
) -> None:
    """Metin dosyasını atomik biçimde yazar."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    temporary_file.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )

    temporary_file.replace(output_file)


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


def ensure_numeric(
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
                f"{source_name}/{column_name} içinde "
                "sonlu olmayan değer var."
            )


def validate_expected_values(
    frame: pd.DataFrame,
    column_name: str,
    expected_values: tuple[str, ...],
    source_name: str,
) -> None:
    """Kategorik sütundaki değerleri doğrular."""

    actual_values = set(
        frame[column_name]
        .astype(str)
        .unique()
        .tolist()
    )

    expected_set = set(expected_values)

    if actual_values != expected_set:
        raise RuntimeError(
            f"{source_name}/{column_name} değerleri uyuşmuyor. "
            f"Eksik={sorted(expected_set - actual_values)}, "
            f"beklenmeyen={sorted(actual_values - expected_set)}"
        )


def load_inputs(
    report_directory: Path,
) -> dict[str, pd.DataFrame]:
    """32. aşama çıktılarının tümünü yükler ve doğrular."""

    input_paths = {
        "descriptives": require_file(
            report_directory
            / DESCRIPTIVES_FILE_NAME
        ),
        "efficiency": require_file(
            report_directory
            / EFFICIENCY_FILE_NAME
        ),
        "friedman": require_file(
            report_directory
            / FRIEDMAN_FILE_NAME
        ),
        "wilcoxon": require_file(
            report_directory
            / WILCOXON_FILE_NAME
        ),
        "per_class": require_file(
            report_directory
            / PER_CLASS_FILE_NAME
        ),
        "pareto": require_file(
            report_directory
            / PARETO_FILE_NAME
        ),
    }

    frames = {
        name: pd.read_csv(path)
        for name, path in input_paths.items()
    }

    descriptives = frames["descriptives"]

    required_descriptive_columns = {
        "model_name",
        "endpoint_id",
        "variant_name",
        "average_rank",
        "sample_count",
        "mean",
        "standard_deviation",
        "ci95_lower",
        "ci95_upper",
    }

    missing = (
        required_descriptive_columns
        - set(descriptives.columns)
    )

    if missing:
        raise ValueError(
            "Betimsel istatistik dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing))
        )

    ensure_numeric(
        descriptives,
        (
            "average_rank",
            "sample_count",
            "mean",
            "standard_deviation",
            "ci95_lower",
            "ci95_upper",
        ),
        "descriptives",
    )

    validate_expected_values(
        descriptives,
        "model_name",
        MODEL_ORDER,
        "descriptives",
    )

    validate_expected_values(
        descriptives[
            descriptives["endpoint_id"]
            == "test_macro_f1"
        ],
        "variant_name",
        VARIANT_ORDER,
        "descriptives/test_macro_f1",
    )

    efficiency = frames["efficiency"]

    required_efficiency_columns = {
        "model_name",
        "variant_name",
        "runtime_name",
        "precision",
        "parameter_count",
        "weight_layer_macs",
        "mean_test_macro_f1",
        "test_macro_f1_standard_deviation",
        "mean_deployment_artifact_size_bytes",
        "mean_latency_median_ms",
        "mean_latency_p95_ms",
        "mean_latency_p99_ms",
        "mean_throughput_samples_per_second",
        "host_cpu_profile_only",
        "mcu_measurement",
    }

    missing = (
        required_efficiency_columns
        - set(efficiency.columns)
    )

    if missing:
        raise ValueError(
            "Verimlilik dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing))
        )

    ensure_numeric(
        efficiency,
        (
            "parameter_count",
            "weight_layer_macs",
            "mean_test_macro_f1",
            "test_macro_f1_standard_deviation",
            "mean_deployment_artifact_size_bytes",
            "mean_latency_median_ms",
            "mean_latency_p95_ms",
            "mean_latency_p99_ms",
            "mean_throughput_samples_per_second",
        ),
        "efficiency",
    )

    if len(efficiency) != 30:
        raise RuntimeError(
            f"Verimlilik ödünleşim tablosu 30 satır değil: "
            f"{len(efficiency)}"
        )

    validate_expected_values(
        efficiency,
        "model_name",
        MODEL_ORDER,
        "efficiency",
    )

    validate_expected_values(
        efficiency,
        "variant_name",
        VARIANT_ORDER,
        "efficiency",
    )

    host_flags = normalize_boolean_series(
        efficiency["host_cpu_profile_only"],
        "efficiency/host_cpu_profile_only",
    )

    mcu_flags = normalize_boolean_series(
        efficiency["mcu_measurement"],
        "efficiency/mcu_measurement",
    )

    if not host_flags.all():
        raise RuntimeError(
            "Bütün verimlilik kayıtları host CPU profili değil."
        )

    if mcu_flags.any():
        raise RuntimeError(
            "MCU ölçümü olarak işaretlenmiş kayıt bulundu."
        )

    friedman = frames["friedman"]

    required_friedman_columns = {
        "model_name",
        "endpoint_id",
        "friedman_chi_square",
        "friedman_degrees_of_freedom",
        "friedman_p_value",
        "friedman_significant",
        "kendall_w",
        "kendall_w_magnitude",
    }

    missing = (
        required_friedman_columns
        - set(friedman.columns)
    )

    if missing:
        raise ValueError(
            "Friedman dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing))
        )

    ensure_numeric(
        friedman,
        (
            "friedman_chi_square",
            "friedman_degrees_of_freedom",
            "friedman_p_value",
            "kendall_w",
        ),
        "friedman",
    )

    wilcoxon = frames["wilcoxon"]

    required_wilcoxon_columns = {
        "model_name",
        "endpoint_id",
        "variant_a",
        "variant_b",
        "holm_adjusted_p_value",
        "rank_biserial_correlation",
        "effect_magnitude",
        "gatekept_significant",
    }

    missing = (
        required_wilcoxon_columns
        - set(wilcoxon.columns)
    )

    if missing:
        raise ValueError(
            "Wilcoxon-Holm dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing))
        )

    ensure_numeric(
        wilcoxon,
        (
            "holm_adjusted_p_value",
            "rank_biserial_correlation",
        ),
        "wilcoxon",
    )

    per_class = frames["per_class"]

    required_per_class_columns = {
        "model_name",
        "variant_name",
        "seed",
        "class_name",
        "false_negative_rate",
    }

    missing = (
        required_per_class_columns
        - set(per_class.columns)
    )

    if missing:
        raise ValueError(
            "Sınıf bazlı sonuç dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing))
        )

    ensure_numeric(
        per_class,
        (
            "seed",
            "false_negative_rate",
        ),
        "per_class",
    )

    if len(per_class) != 450:
        raise RuntimeError(
            f"Sınıf bazlı sonuç sayısı 450 değil: "
            f"{len(per_class)}"
        )

    validate_expected_values(
        per_class,
        "class_name",
        CLASS_ORDER,
        "per_class",
    )

    pareto = frames["pareto"]

    required_pareto_columns = {
        "pareto_scope",
        "scope_model_name",
        "model_name",
        "variant_name",
        "is_pareto_optimal",
        "dominated_by_count",
        "mean_test_macro_f1",
        "mean_deployment_artifact_size_bytes",
        "mean_latency_median_ms",
        "weight_layer_macs",
    }

    missing = (
        required_pareto_columns
        - set(pareto.columns)
    )

    if missing:
        raise ValueError(
            "Pareto dosyasında eksik sütunlar var: "
            + ", ".join(sorted(missing))
        )

    pareto["is_pareto_optimal"] = (
        normalize_boolean_series(
            pareto["is_pareto_optimal"],
            "pareto/is_pareto_optimal",
        )
    )

    frames["input_paths"] = input_paths

    return frames


def order_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Model ve varyant sırasını sabitler."""

    ordered = frame.copy()

    ordered["model_name"] = pd.Categorical(
        ordered["model_name"],
        categories=list(MODEL_ORDER),
        ordered=True,
    )

    ordered["variant_name"] = pd.Categorical(
        ordered["variant_name"],
        categories=list(VARIANT_ORDER),
        ordered=True,
    )

    return ordered.sort_values(
        by=[
            "model_name",
            "variant_name",
        ]
    ).reset_index(
        drop=True
    )


def format_mean_std(
    mean_value: float,
    standard_deviation: float,
    digits: int = 6,
) -> str:
    """Ortalama ± standart sapma metni üretir."""

    return (
        f"{mean_value:.{digits}f} "
        f"± {standard_deviation:.{digits}f}"
    )


def build_performance_table(
    descriptives: pd.DataFrame,
) -> pd.DataFrame:
    """Macro F1, MCC ve dengeli doğruluk yayın tablosunu üretir."""

    selected_endpoints = (
        "test_macro_f1",
        "test_mcc",
        "test_balanced_accuracy",
    )

    subset = descriptives[
        descriptives["endpoint_id"].isin(
            selected_endpoints
        )
    ].copy()

    if len(subset) != 90:
        raise RuntimeError(
            f"Performans betimsel kayıt sayısı 90 değil: "
            f"{len(subset)}"
        )

    records: list[dict[str, Any]] = []

    for model_name in MODEL_ORDER:
        for variant_name in VARIANT_ORDER:
            group = subset[
                (
                    subset["model_name"]
                    == model_name
                )
                & (
                    subset["variant_name"]
                    == variant_name
                )
            ]

            endpoint_rows = {
                str(row.endpoint_id): row
                for row in group.itertuples(
                    index=False
                )
            }

            if set(endpoint_rows) != set(selected_endpoints):
                raise RuntimeError(
                    f"{model_name}/{variant_name}: "
                    "performans sonuç değişkenleri eksik."
                )

            macro = endpoint_rows["test_macro_f1"]
            mcc = endpoint_rows["test_mcc"]
            balanced = endpoint_rows[
                "test_balanced_accuracy"
            ]

            records.append(
                {
                    "Model": MODEL_DISPLAY[
                        model_name
                    ],
                    "Varyant": variant_name,
                    "Macro F1 Ortalama": float(
                        macro.mean
                    ),
                    "Macro F1 SS": float(
                        macro.standard_deviation
                    ),
                    "Macro F1 %95 GA Alt": float(
                        macro.ci95_lower
                    ),
                    "Macro F1 %95 GA Üst": float(
                        macro.ci95_upper
                    ),
                    "Macro F1 Ortalama ± SS": (
                        format_mean_std(
                            float(macro.mean),
                            float(
                                macro.standard_deviation
                            ),
                        )
                    ),
                    "Macro F1 Ortalama Rank": float(
                        macro.average_rank
                    ),
                    "MCC Ortalama": float(
                        mcc.mean
                    ),
                    "MCC SS": float(
                        mcc.standard_deviation
                    ),
                    "Dengeli Doğruluk Ortalama": float(
                        balanced.mean
                    ),
                    "Dengeli Doğruluk SS": float(
                        balanced.standard_deviation
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def build_efficiency_table(
    efficiency: pd.DataFrame,
) -> pd.DataFrame:
    """Kaynak kullanımı ve host CPU performansı tablosunu üretir."""

    ordered = order_table(
        efficiency
    )

    result = pd.DataFrame(
        {
            "Model": ordered[
                "model_name"
            ].astype(str).map(
                MODEL_DISPLAY
            ),
            "Varyant": ordered[
                "variant_name"
            ].astype(str),
            "Runtime": ordered[
                "runtime_name"
            ].astype(str),
            "Hassasiyet": ordered[
                "precision"
            ].astype(str),
            "Parametre": ordered[
                "parameter_count"
            ].astype("int64"),
            "MAC": ordered[
                "weight_layer_macs"
            ].astype("int64"),
            "Artifact Boyutu (bayt)": ordered[
                "mean_deployment_artifact_size_bytes"
            ],
            "Artifact Boyutu (KiB)": (
                ordered[
                    "mean_deployment_artifact_size_bytes"
                ]
                / 1024.0
            ),
            "Macro F1": ordered[
                "mean_test_macro_f1"
            ],
            "Macro F1 SS": ordered[
                "test_macro_f1_standard_deviation"
            ],
            "Medyan Gecikme (ms)": ordered[
                "mean_latency_median_ms"
            ],
            "P95 Gecikme (ms)": ordered[
                "mean_latency_p95_ms"
            ],
            "P99 Gecikme (ms)": ordered[
                "mean_latency_p99_ms"
            ],
            "Throughput (örnek/s)": ordered[
                "mean_throughput_samples_per_second"
            ],
            "Ölçüm Ortamı": (
                "Host CPU"
            ),
            "MCU Ölçümü": False,
        }
    )

    return result


def build_statistical_table(
    friedman: pd.DataFrame,
    wilcoxon: pd.DataFrame,
) -> pd.DataFrame:
    """Ana Macro F1 Friedman ve Holm özet tablosunu üretir."""

    primary_friedman = friedman[
        friedman["endpoint_id"]
        == "test_macro_f1"
    ].copy()

    if len(primary_friedman) != 3:
        raise RuntimeError(
            "Macro F1 için üç Friedman kaydı bulunamadı."
        )

    records: list[dict[str, Any]] = []

    for model_name in MODEL_ORDER:
        omnibus = primary_friedman[
            primary_friedman["model_name"]
            == model_name
        ]

        if len(omnibus) != 1:
            raise RuntimeError(
                f"{model_name}: Macro F1 Friedman kaydı tekil değil."
            )

        omnibus_row = omnibus.iloc[0]

        model_pairwise = wilcoxon[
            (
                wilcoxon["model_name"]
                == model_name
            )
            & (
                wilcoxon["endpoint_id"]
                == "test_macro_f1"
            )
        ].copy()

        if len(model_pairwise) != 45:
            raise RuntimeError(
                f"{model_name}: 45 Macro F1 ikili karşılaştırması yok."
            )

        significant_flags = normalize_boolean_series(
            model_pairwise[
                "gatekept_significant"
            ],
            (
                f"{model_name}/"
                "gatekept_significant"
            ),
        )

        b0_comparisons = model_pairwise[
            (
                model_pairwise["variant_a"]
                == "B0"
            )
            | (
                model_pairwise["variant_b"]
                == "B0"
            )
        ]

        b0_significant = normalize_boolean_series(
            b0_comparisons[
                "gatekept_significant"
            ],
            (
                f"{model_name}/"
                "b0_gatekept_significant"
            ),
        )

        records.append(
            {
                "Model": MODEL_DISPLAY[
                    model_name
                ],
                "Friedman χ²": float(
                    omnibus_row[
                        "friedman_chi_square"
                    ]
                ),
                "Serbestlik Derecesi": int(
                    omnibus_row[
                        "friedman_degrees_of_freedom"
                    ]
                ),
                "Friedman p": float(
                    omnibus_row[
                        "friedman_p_value"
                    ]
                ),
                "Friedman Anlamlı": bool(
                    normalize_boolean_series(
                        pd.Series(
                            [
                                omnibus_row[
                                    "friedman_significant"
                                ]
                            ]
                        ),
                        "friedman_significant",
                    ).iloc[0]
                ),
                "Kendall W": float(
                    omnibus_row[
                        "kendall_w"
                    ]
                ),
                "Etki Büyüklüğü": str(
                    omnibus_row[
                        "kendall_w_magnitude"
                    ]
                ),
                "Holm Sonrası Anlamlı İkili Karşılaştırma": int(
                    significant_flags.sum()
                ),
                "Toplam İkili Karşılaştırma": int(
                    len(model_pairwise)
                ),
                "B0 ile Anlamlı Karşılaştırma": int(
                    b0_significant.sum()
                ),
                "B0 Karşılaştırma Sayısı": int(
                    len(b0_comparisons)
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def build_fnr_table(
    per_class: pd.DataFrame,
) -> pd.DataFrame:
    """Benign, Gafgyt ve Mirai FNR yayın tablosunu üretir."""

    grouped = (
        per_class.groupby(
            [
                "model_name",
                "variant_name",
                "class_name",
            ],
            as_index=False,
        )[
            "false_negative_rate"
        ]
        .agg(
            mean="mean",
            std="std",
            minimum="min",
            maximum="max",
            count="count",
        )
    )

    records: list[dict[str, Any]] = []

    for model_name in MODEL_ORDER:
        for variant_name in VARIANT_ORDER:
            subset = grouped[
                (
                    grouped["model_name"]
                    == model_name
                )
                & (
                    grouped["variant_name"]
                    == variant_name
                )
            ]

            if len(subset) != 3:
                raise RuntimeError(
                    f"{model_name}/{variant_name}: "
                    "üç sınıf FNR özeti yok."
                )

            row: dict[str, Any] = {
                "Model": MODEL_DISPLAY[
                    model_name
                ],
                "Varyant": variant_name,
            }

            for class_name in CLASS_ORDER:
                class_row = subset[
                    subset["class_name"]
                    == class_name
                ]

                if len(class_row) != 1:
                    raise RuntimeError(
                        f"{model_name}/{variant_name}/{class_name}: "
                        "FNR özeti tekil değil."
                    )

                values = class_row.iloc[0]
                prefix = CLASS_DISPLAY[
                    class_name
                ]

                row[
                    f"{prefix} FNR Ortalama"
                ] = float(
                    values["mean"]
                )

                row[
                    f"{prefix} FNR SS"
                ] = float(
                    values["std"]
                )

                row[
                    f"{prefix} FNR Ortalama ± SS"
                ] = format_mean_std(
                    float(values["mean"]),
                    float(values["std"]),
                )

            records.append(row)

    return pd.DataFrame(
        records
    )


def build_pareto_table(
    pareto: pd.DataFrame,
) -> pd.DataFrame:
    """Global ve model içi Pareto-optimal çözümleri seçer."""

    optimal = pareto[
        pareto["is_pareto_optimal"]
    ].copy()

    if optimal.empty:
        raise RuntimeError(
            "Pareto-optimal kayıt bulunamadı."
        )

    optimal["Scope"] = np.where(
        optimal["pareto_scope"]
        == "global",
        "Global",
        "Model İçi",
    )

    optimal["Model"] = (
        optimal["model_name"]
        .astype(str)
        .map(MODEL_DISPLAY)
    )

    optimal["Varyant"] = (
        optimal["variant_name"]
        .astype(str)
    )

    result = optimal[
        [
            "Scope",
            "Model",
            "Varyant",
            "mean_test_macro_f1",
            "mean_deployment_artifact_size_bytes",
            "mean_latency_median_ms",
            "weight_layer_macs",
            "dominated_by_count",
        ]
    ].rename(
        columns={
            "mean_test_macro_f1": (
                "Macro F1"
            ),
            "mean_deployment_artifact_size_bytes": (
                "Artifact Boyutu (bayt)"
            ),
            "mean_latency_median_ms": (
                "Medyan Gecikme (ms)"
            ),
            "weight_layer_macs": (
                "MAC"
            ),
            "dominated_by_count": (
                "Baskın Çözüm Sayısı"
            ),
        }
    )

    return result.sort_values(
        by=[
            "Scope",
            "Model",
            "Varyant",
        ]
    ).reset_index(
        drop=True
    )


def configure_publication_axes(
    axis: plt.Axes,
) -> None:
    """Ortak yayın biçimlendirmesini uygular."""

    axis.grid(
        True,
        linewidth=0.5,
        alpha=0.35,
    )

    axis.tick_params(
        axis="both",
        labelsize=9,
    )


def save_figure(
    figure: plt.Figure,
    png_file: Path,
    pdf_file: Path,
    dpi: int,
) -> None:
    """Şekli PNG ve PDF olarak kaydeder."""

    figure.tight_layout()

    figure.savefig(
        png_file,
        dpi=dpi,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_file,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_macro_f1_vs_artifact(
    efficiency: pd.DataFrame,
    pareto: pd.DataFrame,
    output_directory: Path,
    dpi: int,
) -> tuple[Path, Path]:
    """Macro F1–artifact boyutu ödünleşim grafiğini üretir."""

    figure, axis = plt.subplots(
        figsize=(8.2, 5.6)
    )

    global_pareto_keys = {
        (
            str(row.model_name),
            str(row.variant_name),
        )
        for row in pareto[
            (
                pareto["pareto_scope"]
                == "global"
            )
            & (
                pareto["is_pareto_optimal"]
            )
        ].itertuples(
            index=False
        )
    }

    for model_name in MODEL_ORDER:
        subset = efficiency[
            efficiency["model_name"]
            == model_name
        ]

        axis.scatter(
            subset[
                "mean_deployment_artifact_size_bytes"
            ]
            / 1024.0,
            subset[
                "mean_test_macro_f1"
            ],
            marker=MODEL_MARKERS[
                model_name
            ],
            s=55,
            label=MODEL_DISPLAY[
                model_name
            ],
        )

        for row in subset.itertuples(
            index=False
        ):
            key = (
                str(row.model_name),
                str(row.variant_name),
            )

            if key in global_pareto_keys:
                axis.annotate(
                    str(row.variant_name),
                    (
                        float(
                            row.mean_deployment_artifact_size_bytes
                        )
                        / 1024.0,
                        float(
                            row.mean_test_macro_f1
                        ),
                    ),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

    axis.set_xlabel(
        "Deployment Artifact Boyutu (KiB)"
    )

    axis.set_ylabel(
        "Ortalama Test Macro F1"
    )

    axis.set_title(
        "Doğruluk–Model Boyutu Ödünleşimi"
    )

    axis.legend(
        fontsize=9,
        frameon=True,
    )

    configure_publication_axes(
        axis
    )

    png_file = (
        output_directory
        / "figure_1_macro_f1_vs_artifact.png"
    )

    pdf_file = (
        output_directory
        / "figure_1_macro_f1_vs_artifact.pdf"
    )

    save_figure(
        figure,
        png_file,
        pdf_file,
        dpi,
    )

    return png_file, pdf_file


def plot_macro_f1_vs_latency(
    efficiency: pd.DataFrame,
    pareto: pd.DataFrame,
    output_directory: Path,
    dpi: int,
) -> tuple[Path, Path]:
    """Macro F1–host CPU gecikme grafiğini üretir."""

    figure, axis = plt.subplots(
        figsize=(8.2, 5.6)
    )

    global_pareto_keys = {
        (
            str(row.model_name),
            str(row.variant_name),
        )
        for row in pareto[
            (
                pareto["pareto_scope"]
                == "global"
            )
            & (
                pareto["is_pareto_optimal"]
            )
        ].itertuples(
            index=False
        )
    }

    for model_name in MODEL_ORDER:
        subset = efficiency[
            efficiency["model_name"]
            == model_name
        ]

        axis.scatter(
            subset[
                "mean_latency_median_ms"
            ],
            subset[
                "mean_test_macro_f1"
            ],
            marker=MODEL_MARKERS[
                model_name
            ],
            s=55,
            label=MODEL_DISPLAY[
                model_name
            ],
        )

        for row in subset.itertuples(
            index=False
        ):
            key = (
                str(row.model_name),
                str(row.variant_name),
            )

            if key in global_pareto_keys:
                axis.annotate(
                    str(row.variant_name),
                    (
                        float(
                            row.mean_latency_median_ms
                        ),
                        float(
                            row.mean_test_macro_f1
                        ),
                    ),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

    axis.set_xlabel(
        "Host CPU Medyan Gecikme (ms)"
    )

    axis.set_ylabel(
        "Ortalama Test Macro F1"
    )

    axis.set_title(
        "Doğruluk–Host CPU Gecikme Ödünleşimi"
    )

    axis.legend(
        fontsize=9,
        frameon=True,
    )

    configure_publication_axes(
        axis
    )

    png_file = (
        output_directory
        / "figure_2_macro_f1_vs_latency.png"
    )

    pdf_file = (
        output_directory
        / "figure_2_macro_f1_vs_latency.pdf"
    )

    save_figure(
        figure,
        png_file,
        pdf_file,
        dpi,
    )

    return png_file, pdf_file


def plot_macro_f1_heatmap(
    descriptives: pd.DataFrame,
    output_directory: Path,
    dpi: int,
) -> tuple[Path, Path]:
    """Model–varyant Macro F1 ısı haritasını üretir."""

    subset = descriptives[
        descriptives["endpoint_id"]
        == "test_macro_f1"
    ].copy()

    matrix = subset.pivot(
        index="model_name",
        columns="variant_name",
        values="mean",
    ).reindex(
        index=list(MODEL_ORDER),
        columns=list(VARIANT_ORDER),
    )

    if matrix.isna().any().any():
        raise RuntimeError(
            "Macro F1 ısı haritası matrisi eksik."
        )

    values = matrix.to_numpy(
        dtype=np.float64
    )

    figure, axis = plt.subplots(
        figsize=(11.0, 3.8)
    )

    image = axis.imshow(
        values,
        aspect="auto",
    )

    axis.set_xticks(
        np.arange(
            len(VARIANT_ORDER)
        )
    )

    axis.set_xticklabels(
        VARIANT_ORDER,
        rotation=45,
        ha="right",
        fontsize=9,
    )

    axis.set_yticks(
        np.arange(
            len(MODEL_ORDER)
        )
    )

    axis.set_yticklabels(
        [
            MODEL_DISPLAY[
                model_name
            ]
            for model_name in MODEL_ORDER
        ],
        fontsize=10,
    )

    for row_index in range(
        values.shape[0]
    ):
        for column_index in range(
            values.shape[1]
        ):
            axis.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:.4f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    axis.set_title(
        "Model ve Sıkıştırma Varyantlarına Göre Ortalama Test Macro F1"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Ortalama Macro F1"
    )

    png_file = (
        output_directory
        / "figure_3_macro_f1_heatmap.png"
    )

    pdf_file = (
        output_directory
        / "figure_3_macro_f1_heatmap.pdf"
    )

    save_figure(
        figure,
        png_file,
        pdf_file,
        dpi,
    )

    return png_file, pdf_file


def find_best_row(
    frame: pd.DataFrame,
    column_name: str,
    ascending: bool,
) -> pd.Series:
    """Belirli ölçüte göre en iyi satırı seçer."""

    return (
        frame.sort_values(
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
        )
        .iloc[0]
    )


def build_results_markdown(
    descriptives: pd.DataFrame,
    efficiency: pd.DataFrame,
    friedman: pd.DataFrame,
    wilcoxon: pd.DataFrame,
    pareto: pd.DataFrame,
) -> str:
    """Makalede düzenlenebilecek Türkçe sonuç metnini üretir."""

    macro = descriptives[
        descriptives["endpoint_id"]
        == "test_macro_f1"
    ].copy()

    best_accuracy = find_best_row(
        efficiency,
        "mean_test_macro_f1",
        ascending=False,
    )

    smallest_artifact = find_best_row(
        efficiency,
        "mean_deployment_artifact_size_bytes",
        ascending=True,
    )

    lowest_latency = find_best_row(
        efficiency,
        "mean_latency_median_ms",
        ascending=True,
    )

    highest_throughput = find_best_row(
        efficiency,
        "mean_throughput_samples_per_second",
        ascending=False,
    )

    global_pareto = pareto[
        (
            pareto["pareto_scope"]
            == "global"
        )
        & (
            pareto["is_pareto_optimal"]
        )
    ]

    lines: list[str] = []

    lines.append(
        "# 33. Aşama — Yayın İçin Sonuç Özeti"
    )

    lines.append("")

    lines.append(
        "## 1. Sınıflandırma performansı"
    )

    lines.append("")

    lines.append(
        "Toplam 3 model, 10 varyant ve 5 eşleştirilmiş seed "
        "üzerinden 150 test sonucu analiz edilmiştir. "
        "Birincil performans ölçütü Macro F1'dir."
    )

    lines.append("")

    for model_name in MODEL_ORDER:
        model_macro = macro[
            macro["model_name"]
            == model_name
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

        top_rows = model_macro.head(
            3
        )

        friedman_row = friedman[
            (
                friedman["model_name"]
                == model_name
            )
            & (
                friedman["endpoint_id"]
                == "test_macro_f1"
            )
        ].iloc[0]

        lines.append(
            f"### {MODEL_DISPLAY[model_name]}"
        )

        lines.append("")

        ranking_parts = []

        for index, row in enumerate(
            top_rows.itertuples(
                index=False
            ),
            start=1,
        ):
            ranking_parts.append(
                f"{index}. {row.variant_name} "
                f"({row.mean:.6f} ± "
                f"{row.standard_deviation:.6f})"
            )

        lines.append(
            "İlk üç varyant: "
            + "; ".join(ranking_parts)
            + "."
        )

        lines.append("")

        lines.append(
            f"Friedman testi varyantlar arasında anlamlı fark "
            f"göstermiştir: χ²({int(friedman_row['friedman_degrees_of_freedom'])}) "
            f"= {float(friedman_row['friedman_chi_square']):.6f}, "
            f"p = {float(friedman_row['friedman_p_value']):.8f}. "
            f"Kendall W = {float(friedman_row['kendall_w']):.6f} "
            f"({friedman_row['kendall_w_magnitude']}) olarak bulunmuştur."
        )

        lines.append("")

        model_pairwise = wilcoxon[
            (
                wilcoxon["model_name"]
                == model_name
            )
            & (
                wilcoxon["endpoint_id"]
                == "test_macro_f1"
            )
        ].copy()

        significant_count = int(
            normalize_boolean_series(
                model_pairwise[
                    "gatekept_significant"
                ],
                (
                    f"{model_name}/"
                    "gatekept_significant"
                ),
            ).sum()
        )

        lines.append(
            f"Friedman kapı kontrolü ve Holm düzeltmesi sonrasında "
            f"45 ikili karşılaştırmanın {significant_count} tanesi "
            f"anlamlı kalmıştır."
        )

        lines.append("")

    lines.append(
        "## 2. Mühendislik ödünleşimleri"
    )

    lines.append("")

    lines.append(
        f"En yüksek ortalama Macro F1, "
        f"{MODEL_DISPLAY[str(best_accuracy['model_name'])]}/"
        f"{best_accuracy['variant_name']} tarafından "
        f"{float(best_accuracy['mean_test_macro_f1']):.6f} "
        f"değeriyle elde edilmiştir."
    )

    lines.append("")

    lines.append(
        f"En küçük deployment artifact, "
        f"{MODEL_DISPLAY[str(smallest_artifact['model_name'])]}/"
        f"{smallest_artifact['variant_name']} için "
        f"{float(smallest_artifact['mean_deployment_artifact_size_bytes']) / 1024.0:.3f} "
        f"KiB olarak ölçülmüştür."
    )

    lines.append("")

    lines.append(
        f"En düşük host CPU medyan gecikmesi, "
        f"{MODEL_DISPLAY[str(lowest_latency['model_name'])]}/"
        f"{lowest_latency['variant_name']} için "
        f"{float(lowest_latency['mean_latency_median_ms']):.6f} ms'dir."
    )

    lines.append("")

    lines.append(
        f"En yüksek host CPU throughput değeri, "
        f"{MODEL_DISPLAY[str(highest_throughput['model_name'])]}/"
        f"{highest_throughput['variant_name']} için "
        f"{float(highest_throughput['mean_throughput_samples_per_second']):.3f} "
        f"örnek/s olarak bulunmuştur."
    )

    lines.append("")

    lines.append(
        "Bu sonuçlar, en yüksek doğruluk, en küçük model boyutu, "
        "en düşük gecikme ve en yüksek throughput hedeflerinin aynı "
        "model-varyant kombinasyonunda birleşmediğini göstermektedir."
    )

    lines.append("")

    lines.append(
        "## 3. Global Pareto-optimal çözümler"
    )

    lines.append("")

    if global_pareto.empty:
        lines.append(
            "Global Pareto-optimal çözüm bulunamadı."
        )

    else:
        for row in global_pareto.itertuples(
            index=False
        ):
            lines.append(
                f"- {MODEL_DISPLAY[str(row.model_name)]}/"
                f"{row.variant_name}: "
                f"Macro F1={float(row.mean_test_macro_f1):.6f}, "
                f"artifact={float(row.mean_deployment_artifact_size_bytes) / 1024.0:.3f} KiB, "
                f"medyan gecikme={float(row.mean_latency_median_ms):.6f} ms, "
                f"MAC={int(row.weight_layer_macs)}."
            )

    lines.append("")

    lines.append(
        "## 4. Geçerlilik sınırları"
    )

    lines.append("")

    lines.append(
        "Gecikme ve throughput değerleri yalnızca çalışmanın yürütüldüğü "
        "host CPU, işletim sistemi, PyTorch ve ONNX Runtime ortamı için "
        "geçerlidir. Bunlar MCU gecikmesi veya enerji ölçümü değildir. "
        "Beş seed kullanılması nedeniyle özellikle ikili Wilcoxon "
        "karşılaştırmalarında istatistiksel test gücü sınırlıdır; bu nedenle "
        "p-değerleri eşleştirilmiş farklar, güven aralıkları ve etki "
        "büyüklükleriyle birlikte yorumlanmalıdır."
    )

    lines.append("")

    lines.append(
        "## 5. Üretilen şekiller"
    )

    lines.append("")

    lines.append(
        "- Şekil 1: Macro F1–artifact boyutu ödünleşimi"
    )

    lines.append(
        "- Şekil 2: Macro F1–host CPU gecikme ödünleşimi"
    )

    lines.append(
        "- Şekil 3: Model–varyant Macro F1 ısı haritası"
    )

    lines.append("")

    return "\n".join(lines)


def validate_output_policy(
    output_directory: Path,
    overwrite: bool,
) -> None:
    """Mevcut çıktıların yanlışlıkla üzerine yazılmasını engeller."""

    expected_names = (
        "table_1_performance_summary.csv",
        "table_2_efficiency_summary.csv",
        "table_3_statistical_summary.csv",
        "table_4_classwise_fnr_summary.csv",
        "table_5_pareto_optimal_solutions.csv",
        "figure_1_macro_f1_vs_artifact.png",
        "figure_1_macro_f1_vs_artifact.pdf",
        "figure_2_macro_f1_vs_latency.png",
        "figure_2_macro_f1_vs_latency.pdf",
        "figure_3_macro_f1_heatmap.png",
        "figure_3_macro_f1_heatmap.pdf",
        "stage33_results_summary_tr.md",
        "stage33_manifest.json",
    )

    existing = [
        output_directory
        / file_name
        for file_name in expected_names
        if (
            output_directory
            / file_name
        ).exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "33. aşama çıktıları zaten mevcut. "
            "--overwrite kullan. Mevcut dosyalar: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )


def main() -> None:
    """33. aşama yayın çıktısı üretimini çalıştırır."""

    args = parse_arguments()

    report_directory = (
        args.report_directory.resolve()
    )

    output_directory = (
        args.output_directory.resolve()
    )

    dpi = int(
        args.dpi
    )

    if dpi < 150:
        raise ValueError(
            "Yayın şekilleri için DPI en az 150 olmalıdır."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_output_policy(
        output_directory,
        bool(args.overwrite),
    )

    print("=" * 78)
    print("33. Aşama — Yayın Kalitesinde Tablo ve Şekil Üretimi")
    print("=" * 78)
    print(f"Rapor klasörü : {report_directory}")
    print(f"Çıktı klasörü : {output_directory}")
    print(f"PNG DPI        : {dpi}")
    print("=" * 78)

    frames = load_inputs(
        report_directory
    )

    descriptives = frames[
        "descriptives"
    ]

    efficiency = frames[
        "efficiency"
    ]

    friedman = frames[
        "friedman"
    ]

    wilcoxon = frames[
        "wilcoxon"
    ]

    per_class = frames[
        "per_class"
    ]

    pareto = frames[
        "pareto"
    ]

    performance_table = (
        build_performance_table(
            descriptives
        )
    )

    efficiency_table = (
        build_efficiency_table(
            efficiency
        )
    )

    statistical_table = (
        build_statistical_table(
            friedman,
            wilcoxon,
        )
    )

    fnr_table = build_fnr_table(
        per_class
    )

    pareto_table = build_pareto_table(
        pareto
    )

    output_files = {
        "table_1": (
            output_directory
            / "table_1_performance_summary.csv"
        ),
        "table_2": (
            output_directory
            / "table_2_efficiency_summary.csv"
        ),
        "table_3": (
            output_directory
            / "table_3_statistical_summary.csv"
        ),
        "table_4": (
            output_directory
            / "table_4_classwise_fnr_summary.csv"
        ),
        "table_5": (
            output_directory
            / "table_5_pareto_optimal_solutions.csv"
        ),
        "summary_markdown": (
            output_directory
            / "stage33_results_summary_tr.md"
        ),
        "manifest": (
            output_directory
            / "stage33_manifest.json"
        ),
    }

    write_csv_atomic(
        performance_table,
        output_files["table_1"],
    )

    write_csv_atomic(
        efficiency_table,
        output_files["table_2"],
    )

    write_csv_atomic(
        statistical_table,
        output_files["table_3"],
    )

    write_csv_atomic(
        fnr_table,
        output_files["table_4"],
    )

    write_csv_atomic(
        pareto_table,
        output_files["table_5"],
    )

    figure_1_png, figure_1_pdf = (
        plot_macro_f1_vs_artifact(
            efficiency,
            pareto,
            output_directory,
            dpi,
        )
    )

    figure_2_png, figure_2_pdf = (
        plot_macro_f1_vs_latency(
            efficiency,
            pareto,
            output_directory,
            dpi,
        )
    )

    figure_3_png, figure_3_pdf = (
        plot_macro_f1_heatmap(
            descriptives,
            output_directory,
            dpi,
        )
    )

    results_markdown = (
        build_results_markdown(
            descriptives,
            efficiency,
            friedman,
            wilcoxon,
            pareto,
        )
    )

    write_text_atomic(
        results_markdown,
        output_files[
            "summary_markdown"
        ],
    )

    generated_files = {
        "table_1_performance_summary": (
            output_files["table_1"]
        ),
        "table_2_efficiency_summary": (
            output_files["table_2"]
        ),
        "table_3_statistical_summary": (
            output_files["table_3"]
        ),
        "table_4_classwise_fnr_summary": (
            output_files["table_4"]
        ),
        "table_5_pareto_optimal_solutions": (
            output_files["table_5"]
        ),
        "figure_1_png": figure_1_png,
        "figure_1_pdf": figure_1_pdf,
        "figure_2_png": figure_2_png,
        "figure_2_pdf": figure_2_pdf,
        "figure_3_png": figure_3_png,
        "figure_3_pdf": figure_3_pdf,
        "results_summary_markdown": (
            output_files[
                "summary_markdown"
            ]
        ),
    }

    manifest = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "stage": 33,
        "task": "family_3",
        "models": list(
            MODEL_ORDER
        ),
        "variants": list(
            VARIANT_ORDER
        ),
        "class_names": list(
            CLASS_ORDER
        ),
        "input_artifacts": {
            name: {
                "path": str(path),
                "sha256": calculate_sha256(
                    path
                ),
            }
            for name, path in frames[
                "input_paths"
            ].items()
        },
        "generated_artifacts": {
            name: {
                "path": str(path),
                "sha256": calculate_sha256(
                    path
                ),
            }
            for name, path in generated_files.items()
        },
        "table_row_counts": {
            "performance_summary": int(
                len(performance_table)
            ),
            "efficiency_summary": int(
                len(efficiency_table)
            ),
            "statistical_summary": int(
                len(statistical_table)
            ),
            "classwise_fnr_summary": int(
                len(fnr_table)
            ),
            "pareto_optimal_solutions": int(
                len(pareto_table)
            ),
        },
        "figure_count": 3,
        "figure_formats": [
            "PNG",
            "PDF",
        ],
        "png_dpi": dpi,
        "host_cpu_efficiency_only": True,
        "mcu_measurement": False,
        "test_re_evaluated": False,
        "model_retrained": False,
        "validation_passed": True,
    }

    write_json_atomic(
        manifest,
        output_files[
            "manifest"
        ],
    )

    print()
    print("=" * 78)
    print("33. Aşama Tamamlandı")
    print("=" * 78)
    print(
        f"Performans tablosu : {len(performance_table)} satır"
    )
    print(
        f"Verimlilik tablosu : {len(efficiency_table)} satır"
    )
    print(
        f"İstatistik tablosu : {len(statistical_table)} satır"
    )
    print(
        f"FNR tablosu        : {len(fnr_table)} satır"
    )
    print(
        f"Pareto tablosu     : {len(pareto_table)} satır"
    )
    print("Şekil sayısı       : 3")
    print("Şekil biçimleri    : PNG + PDF")
    print("Yeniden eğitim     : False")
    print("Test tekrarlandı   : False")
    print("MCU ölçümü         : False")
    print("Doğrulama geçti    : True")
    print()
    print(f"Çıktı klasörü      : {output_directory}")
    print("=" * 78)


if __name__ == "__main__":
    main()
