"""
N-BaIoT family_3 çok-seed FP32 ana baseline deneyleri.

Deney planı:
- Modeller:
    tinyml_mlp
    compact_dnn
    tiny_1d_cnn
- Seedler:
    42, 123, 2026, 3407, 8192
- Toplam:
    15 bağımsız eğitim çalışması

Bilimsel kurallar:
- Protokol 22. aşamada kilitlenmiş JSON dosyasından okunur.
- Model seçimi yalnızca validation Macro F1 ile yapılır.
- En iyi validation checkpointi seçildikten sonra test bir kez değerlendirilir.
- Test sonuçları hiperparametre veya checkpoint seçimine katılmaz.
- Tamamlanan deneyler yeniden çalıştırılmaz.
- Yarım kalan eğitimler yalnızca --restart-incomplete ile silinip başlatılır.
- Başlatılmış fakat tamamlanmamış test değerlendirmesi otomatik tekrarlanmaz.

Çıktılar:
results/reports/nbaiot_family3_fp32_baseline_runs.csv
results/reports/nbaiot_family3_fp32_baseline_aggregate.csv
results/reports/nbaiot_family3_fp32_baseline_per_class_runs.csv
results/reports/nbaiot_family3_fp32_baseline_per_class_aggregate.csv
results/reports/nbaiot_family3_fp32_baseline_confusion_matrices.csv
results/reports/nbaiot_family3_fp32_baseline_summary.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import t as student_t
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.data.nbaiot_pipeline import (  # noqa: E402
    create_nbaiot_dataloader,
    load_pipeline_assets,
)
from src.models.nbaiot_models import create_model  # noqa: E402
from src.training.baseline_trainer import (  # noqa: E402
    run_epoch,
    run_training,
    set_reproducible_environment,
)


DEFAULT_PROTOCOL_FILE = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_fp32_baseline_protocol_v1.json"
)

DEFAULT_CONFIG_DIRECTORY = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "fp32_baseline_v1"
)

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

DEFAULT_EXPERIMENT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "fp32_baseline_v1"
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

AGGREGATED_METRICS = (
    "test_loss",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
    "test_mcc",
)

PER_CLASS_METRICS = (
    "precision",
    "recall",
    "f1",
    "false_negative_rate",
)


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "Kilitli protokole göre 15 FP32 baseline "
            "eğitimini ve test değerlendirmesini çalıştırır."
        )
    )

    parser.add_argument(
        "--protocol-file",
        type=Path,
        default=DEFAULT_PROTOCOL_FILE,
    )

    parser.add_argument(
        "--config-directory",
        type=Path,
        default=DEFAULT_CONFIG_DIRECTORY,
    )

    parser.add_argument(
        "--experiment-directory",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIRECTORY,
    )

    parser.add_argument(
        "--report-directory",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )

    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help=(
            "Run summary oluşmamış yarım eğitim klasörlerini "
            "silerek yeniden başlatır."
        ),
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
    """JSON dosyasını yükler."""

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


def write_yaml_atomic(
    document: dict[str, Any],
    output_file: Path,
) -> None:
    """YAML dosyasını atomik biçimde yazar."""

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
        yaml.safe_dump(
            document,
            file_handle,
            sort_keys=False,
            allow_unicode=True,
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


def load_torch_checkpoint(
    checkpoint_file: Path,
) -> dict[str, Any]:
    """PyTorch checkpointini CPU üzerinde güvenli biçimde yükler."""

    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {checkpoint_file}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint kökü sözlük değil."
        )

    return checkpoint


# ==========================================================
# PROTOCOL
# ==========================================================

def load_and_validate_protocol(
    protocol_file: Path,
) -> tuple[
    dict[str, Any],
    str,
]:
    """Kilitli FP32 protokolünü yükler ve doğrular."""

    protocol = read_json(
        protocol_file
    )

    if (
        protocol.get("protocol_version")
        != "1.0"
    ):
        raise ValueError(
            "Beklenmeyen protokol sürümü."
        )

    if (
        protocol.get("status")
        != "locked_before_multi_seed_training"
    ):
        raise ValueError(
            "Protokol ana eğitimden önce kilitlenmiş durumda değil."
        )

    if protocol.get("task") != "family_3":
        raise ValueError(
            "Ana protokol görevi family_3 değil."
        )

    models = tuple(
        str(model_name)
        for model_name
        in protocol["models"]
    )

    if models != EXPECTED_MODELS:
        raise ValueError(
            "Protokoldeki model sırası beklenen sırayla uyuşmuyor."
        )

    seeds = tuple(
        int(seed)
        for seed
        in protocol["training_seeds"]
    )

    if seeds != EXPECTED_SEEDS:
        raise ValueError(
            "Protokoldeki seed sırası beklenen sırayla uyuşmuyor."
        )

    planned_run_count = int(
        protocol[
            "planned_run_count"
        ][
            "total_fp32_training_runs"
        ]
    )

    if planned_run_count != 15:
        raise ValueError(
            "Planlanan ana eğitim sayısı 15 değil."
        )

    common_configuration = protocol[
        "common_training_configuration"
    ]

    required_common_fields = {
        "batch_size",
        "max_epochs",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "weight_decay",
        "gradient_clip_norm",
        "torch_threads",
        "num_workers",
        "class_weight_scheme",
    }

    missing_common_fields = (
        required_common_fields
        - set(common_configuration)
    )

    if missing_common_fields:
        raise ValueError(
            "Protokolde eksik eğitim alanları var: "
            + ", ".join(
                sorted(
                    missing_common_fields
                )
            )
        )

    model_configuration = protocol[
        "model_specific_configuration"
    ]

    for model_name in EXPECTED_MODELS:
        if model_name not in model_configuration:
            raise ValueError(
                f"Protokolde model ayarı yok: {model_name}"
            )

        learning_rate = float(
            model_configuration[
                model_name
            ][
                "learning_rate"
            ]
        )

        if learning_rate <= 0:
            raise ValueError(
                f"{model_name} öğrenme oranı geçersiz."
            )

    protocol_sha256 = calculate_sha256(
        protocol_file
    )

    return (
        protocol,
        protocol_sha256,
    )


# ==========================================================
# CONFIGURATION
# ==========================================================

def build_run_config(
    protocol: dict[str, Any],
    model_name: str,
    seed: int,
    output_directory: Path,
) -> dict[str, Any]:
    """Tek bir model-seed ana deney yapılandırması oluşturur."""

    common = protocol[
        "common_training_configuration"
    ]

    model_configuration = protocol[
        "model_specific_configuration"
    ][model_name]

    experiment_name = (
        f"fp32_family3_{model_name}_seed{seed}"
    )

    return {
        "experiment": {
            "name": experiment_name,
            "mode": "main",
            "task": "family_3",
            "model": model_name,
            "seed": int(seed),
        },
        "data": {
            "batch_size": int(
                common["batch_size"]
            ),
            "num_workers": int(
                common["num_workers"]
            ),
        },
        "training": {
            "max_epochs": int(
                common["max_epochs"]
            ),
            "learning_rate": float(
                model_configuration[
                    "learning_rate"
                ]
            ),
            "weight_decay": float(
                common["weight_decay"]
            ),
            "gradient_clip_norm": float(
                common[
                    "gradient_clip_norm"
                ]
            ),
            "patience": int(
                common[
                    "early_stopping_patience"
                ]
            ),
            "min_delta": float(
                common[
                    "early_stopping_min_delta"
                ]
            ),
            "torch_threads": int(
                common["torch_threads"]
            ),
            "weight_scheme": str(
                common[
                    "class_weight_scheme"
                ]
            ),
        },
        "output": {
            "directory": str(
                output_directory
            ),
            "overwrite": False,
        },
    }


# ==========================================================
# TEST EVALUATION
# ==========================================================

def evaluate_test_once(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    model_name: str,
    seed: int,
    run_summary: dict[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    """
    Seçilen validation checkpointini test splitinde yalnızca bir kez değerlendirir.

    Başlatılmış bir test değerlendirmesi tamamlanmamışsa otomatik olarak
    tekrar değerlendirme yapılmaz.
    """

    checkpoint_file = Path(
        str(
            run_summary[
                "checkpoint_file"
            ]
        )
    ).resolve()

    test_metrics_file = (
        run_directory
        / "test_metrics.json"
    )

    test_per_class_file = (
        run_directory
        / "test_per_class_metrics.csv"
    )

    test_confusion_file = (
        run_directory
        / "test_confusion_matrix.csv"
    )

    evaluation_marker_file = (
        run_directory
        / "test_evaluation_manifest.json"
    )

    checkpoint_sha256 = calculate_sha256(
        checkpoint_file
    )

    # ------------------------------------------------------
    # Existing completed evaluation
    # ------------------------------------------------------

    if evaluation_marker_file.exists():
        marker = read_json(
            evaluation_marker_file
        )

        marker_status = str(
            marker.get("status")
        )

        if marker_status == "started":
            raise RuntimeError(
                "Test değerlendirmesi daha önce başlatılmış ancak "
                "tamamlanmamış görünüyor. Bilimsel olarak otomatik "
                "tekrar yapılmayacak. Dosya: "
                f"{evaluation_marker_file}"
            )

        if marker_status != "completed":
            raise RuntimeError(
                "Bilinmeyen test değerlendirme durumu: "
                f"{marker_status}"
            )

        if not test_metrics_file.exists():
            raise RuntimeError(
                "Test manifesti tamamlandı diyor ancak "
                "test_metrics.json bulunamadı."
            )

        if (
            marker.get("protocol_sha256")
            != protocol_sha256
        ):
            raise RuntimeError(
                "Mevcut test sonucu farklı bir protokole ait."
            )

        if (
            marker.get("checkpoint_sha256")
            != checkpoint_sha256
        ):
            raise RuntimeError(
                "Mevcut test sonucu farklı bir checkpointe ait."
            )

        if int(
            marker.get(
                "evaluation_attempt_count",
                0,
            )
        ) != 1:
            raise RuntimeError(
                "Test değerlendirme girişimi sayısı 1 değil."
            )

        return read_json(
            test_metrics_file
        )

    if test_metrics_file.exists():
        raise RuntimeError(
            "Test metrics mevcut ancak değerlendirme manifesti yok."
        )

    # ------------------------------------------------------
    # Mark evaluation as started
    # ------------------------------------------------------

    started_marker = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "started",
        "model_name": model_name,
        "seed": int(seed),
        "protocol_sha256": protocol_sha256,
        "checkpoint_file": str(
            checkpoint_file
        ),
        "checkpoint_sha256": (
            checkpoint_sha256
        ),
        "evaluation_attempt_count": 1,
        "test_used_for_model_selection": False,
    }

    write_json_atomic(
        document=started_marker,
        output_file=evaluation_marker_file,
    )

    common_configuration = protocol[
        "common_training_configuration"
    ]

    torch_threads = int(
        common_configuration[
            "torch_threads"
        ]
    )

    weight_scheme = str(
        common_configuration[
            "class_weight_scheme"
        ]
    )

    batch_size = int(
        common_configuration[
            "batch_size"
        ]
    )

    num_workers = int(
        common_configuration[
            "num_workers"
        ]
    )

    set_reproducible_environment(
        seed=seed,
        torch_threads=torch_threads,
    )

    assets = load_pipeline_assets(
        task_name="family_3",
        weight_scheme=weight_scheme,
        expected_seed=2026,
    )

    _, test_loader = create_nbaiot_dataloader(
        assets=assets,
        split_name="test",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        epoch=0,
        num_workers=num_workers,
        pin_memory=False,
    )

    model = create_model(
        model_name=model_name,
        input_features=assets.feature_count,
        num_classes=assets.class_count,
    )

    checkpoint = load_torch_checkpoint(
        checkpoint_file
    )

    if str(
        checkpoint["model_name"]
    ) != model_name:
        raise RuntimeError(
            "Checkpoint model adı deney modeliyle uyuşmuyor."
        )

    if str(
        checkpoint["task_name"]
    ) != "family_3":
        raise RuntimeError(
            "Checkpoint görevi family_3 değil."
        )

    if int(
        checkpoint["seed"]
    ) != seed:
        raise RuntimeError(
            "Checkpoint seed değeri deney seediyle uyuşmuyor."
        )

    checkpoint_classes = tuple(
        str(class_name)
        for class_name
        in checkpoint["class_names"]
    )

    if checkpoint_classes != (
        assets.target_classes
    ):
        raise RuntimeError(
            "Checkpoint sınıf sırası veri hattıyla uyuşmuyor."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    class_weights = torch.as_tensor(
        assets.class_weights,
        dtype=torch.float32,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    test_metrics = run_epoch(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        class_names=assets.target_classes,
        optimizer=None,
        gradient_clip_norm=0.0,
        maximum_batches=None,
    )

    expected_test_count = int(
        assets.task_definition[
            "split_totals"
        ]["test"]
    )

    if int(
        test_metrics[
            "sample_count"
        ]
    ) != expected_test_count:
        raise RuntimeError(
            "Test örnek sayısı görev tanımıyla uyuşmuyor: "
            f"beklenen={expected_test_count:,}, "
            f"işlenen={test_metrics['sample_count']:,}"
        )

    test_document: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_name": model_name,
        "task_name": "family_3",
        "seed": int(seed),
        "checkpoint_selection_split": (
            "validation"
        ),
        "checkpoint_selection_metric": (
            "macro_f1"
        ),
        "best_epoch": int(
            run_summary["best_epoch"]
        ),
        "best_validation_macro_f1": float(
            run_summary[
                "best_validation_macro_f1"
            ]
        ),
        "test_used_for_selection": False,
        "test_evaluation_attempt_count": 1,
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        **test_metrics,
    }

    write_json_atomic(
        document=test_document,
        output_file=test_metrics_file,
    )

    per_class_records: list[
        dict[str, Any]
    ] = []

    for class_index, class_name in enumerate(
        assets.target_classes
    ):
        class_metrics = test_metrics[
            "per_class"
        ][class_name]

        per_class_records.append(
            {
                "model_name": model_name,
                "seed": int(seed),
                "class_index": int(
                    class_index
                ),
                "class_name": class_name,
                "support": int(
                    class_metrics[
                        "support"
                    ]
                ),
                "predicted_count": int(
                    class_metrics[
                        "predicted_count"
                    ]
                ),
                "precision": float(
                    class_metrics[
                        "precision"
                    ]
                ),
                "recall": float(
                    class_metrics[
                        "recall"
                    ]
                ),
                "f1": float(
                    class_metrics["f1"]
                ),
                "false_negative_rate": float(
                    class_metrics[
                        "false_negative_rate"
                    ]
                ),
            }
        )

    write_csv_atomic(
        frame=pd.DataFrame(
            per_class_records
        ),
        output_file=test_per_class_file,
    )

    confusion_matrix = np.asarray(
        test_metrics[
            "confusion_matrix"
        ],
        dtype=np.int64,
    )

    confusion_frame = pd.DataFrame(
        confusion_matrix,
        index=assets.target_classes,
        columns=assets.target_classes,
    )

    confusion_frame.index.name = (
        "true_class"
    )

    confusion_frame.to_csv(
        test_confusion_file,
        encoding="utf-8",
    )

    completed_marker = {
        **started_marker,
        "completed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "completed",
        "test_metrics_file": str(
            test_metrics_file
        ),
        "test_per_class_file": str(
            test_per_class_file
        ),
        "test_confusion_file": str(
            test_confusion_file
        ),
    }

    write_json_atomic(
        document=completed_marker,
        output_file=evaluation_marker_file,
    )

    return test_document


# ==========================================================
# STATISTICS
# ==========================================================

def calculate_summary_statistics(
    values: np.ndarray,
) -> dict[str, float | int]:
    """Bir metrik için seed tabanlı özet ve %95 güven aralığı hesaplar."""

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
            values.std(
                ddof=1
            )
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
        "standard_deviation": (
            standard_deviation
        ),
        "standard_error": (
            standard_error
        ),
        "median": median_value,
        "minimum": minimum_value,
        "maximum": maximum_value,
        "ci95_lower": float(
            mean_value
            - confidence_margin
        ),
        "ci95_upper": float(
            mean_value
            + confidence_margin
        ),
    }


def aggregate_run_metrics(
    run_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Model bazında test metriklerini seedler üzerinden özetler."""

    records: list[
        dict[str, Any]
    ] = []

    for model_name in EXPECTED_MODELS:
        model_frame = run_frame[
            run_frame[
                "model_name"
            ]
            == model_name
        ]

        for metric_name in AGGREGATED_METRICS:
            statistics = (
                calculate_summary_statistics(
                    model_frame[
                        metric_name
                    ].to_numpy(
                        dtype=np.float64
                    )
                )
            )

            records.append(
                {
                    "model_name": model_name,
                    "metric": metric_name,
                    **statistics,
                }
            )

    return pd.DataFrame(
        records
    )


def aggregate_per_class_metrics(
    per_class_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Her model ve hedef sınıf için çok-seed istatistikleri hesaplar."""

    records: list[
        dict[str, Any]
    ] = []

    for model_name in EXPECTED_MODELS:
        model_frame = per_class_frame[
            per_class_frame[
                "model_name"
            ]
            == model_name
        ]

        class_names = (
            model_frame[
                [
                    "class_index",
                    "class_name",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                "class_index"
            )
        )

        for class_row in class_names.itertuples(
            index=False
        ):
            class_frame = model_frame[
                model_frame[
                    "class_name"
                ]
                == class_row.class_name
            ]

            for metric_name in PER_CLASS_METRICS:
                statistics = (
                    calculate_summary_statistics(
                        class_frame[
                            metric_name
                        ].to_numpy(
                            dtype=np.float64
                        )
                    )
                )

                records.append(
                    {
                        "model_name": model_name,
                        "class_index": int(
                            class_row.class_index
                        ),
                        "class_name": str(
                            class_row.class_name
                        ),
                        "metric": metric_name,
                        **statistics,
                    }
                )

    return pd.DataFrame(
        records
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Ana deneylerin tamamını çalıştırır."""

    args = parse_arguments()

    protocol_file = (
        args.protocol_file.resolve()
    )

    config_directory = (
        args.config_directory.resolve()
    )

    experiment_directory = (
        args.experiment_directory.resolve()
    )

    report_directory = (
        args.report_directory.resolve()
    )

    config_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        protocol,
        protocol_sha256,
    ) = load_and_validate_protocol(
        protocol_file
    )

    run_records: list[
        dict[str, Any]
    ] = []

    per_class_records: list[
        dict[str, Any]
    ] = []

    confusion_records: list[
        dict[str, Any]
    ] = []

    total_run_count = (
        len(EXPECTED_MODELS)
        * len(EXPECTED_SEEDS)
    )

    print("=" * 78)
    print("N-BaIoT Family-3 FP32 Ana Baseline Deneyleri")
    print("=" * 78)
    print(
        "Modeller        : "
        + ", ".join(
            EXPECTED_MODELS
        )
    )
    print(
        "Seedler         : "
        + ", ".join(
            str(seed)
            for seed in EXPECTED_SEEDS
        )
    )
    print(
        f"Toplam çalışma  : {total_run_count}"
    )
    print(
        "Model seçimi    : validation Macro F1"
    )
    print(
        "Test politikası : seçilen checkpointte bir kez"
    )
    print(
        f"Protokol SHA-256: {protocol_sha256}"
    )
    print("=" * 78)

    run_index = 0

    for model_name in EXPECTED_MODELS:
        for seed in EXPECTED_SEEDS:
            run_index += 1

            run_directory = (
                experiment_directory
                / model_name
                / f"seed{seed}"
            )

            config_file = (
                config_directory
                / (
                    f"family3_{model_name}_"
                    f"seed{seed}.yaml"
                )
            )

            run_summary_file = (
                run_directory
                / "run_summary.json"
            )

            configuration = build_run_config(
                protocol=protocol,
                model_name=model_name,
                seed=seed,
                output_directory=run_directory,
            )

            write_yaml_atomic(
                document=configuration,
                output_file=config_file,
            )

            print()
            print("=" * 78)
            print(
                f"Çalışma {run_index}/{total_run_count}"
            )
            print(f"Model : {model_name}")
            print(f"Seed  : {seed}")
            print("=" * 78)

            # ----------------------------------------------
            # Training or resume
            # ----------------------------------------------

            if run_summary_file.exists():
                run_summary = read_json(
                    run_summary_file
                )

                expected_config_sha256 = (
                    calculate_sha256(
                        config_file
                    )
                )

                if (
                    run_summary[
                        "config_sha256"
                    ]
                    != expected_config_sha256
                ):
                    raise RuntimeError(
                        "Tamamlanmış deneyin yapılandırması "
                        "mevcut protokolle uyuşmuyor: "
                        f"{model_name}/seed{seed}"
                    )

                print(
                    "Eğitim daha önce tamamlanmış; "
                    "checkpoint yeniden kullanılacak."
                )

            else:
                if run_directory.exists():
                    if args.restart_incomplete:
                        shutil.rmtree(
                            run_directory
                        )

                    else:
                        raise RuntimeError(
                            "Yarım kalmış eğitim klasörü bulundu: "
                            f"{run_directory}\n"
                            "Yeniden başlatmak için komuta "
                            "--restart-incomplete ekle."
                        )

                run_summary = run_training(
                    config_file=config_file
                )

            if bool(
                run_summary[
                    "test_split_evaluated"
                ]
            ):
                raise RuntimeError(
                    "Eğitim motoru model seçimi sırasında "
                    "test splitini değerlendirmiş."
                )

            if str(
                run_summary[
                    "model_name"
                ]
            ) != model_name:
                raise RuntimeError(
                    "Run summary model adı uyuşmuyor."
                )

            if int(
                run_summary["seed"]
            ) != seed:
                raise RuntimeError(
                    "Run summary seed değeri uyuşmuyor."
                )

            # ----------------------------------------------
            # Single test evaluation
            # ----------------------------------------------

            test_metrics = evaluate_test_once(
                protocol=protocol,
                protocol_sha256=(
                    protocol_sha256
                ),
                model_name=model_name,
                seed=seed,
                run_summary=run_summary,
                run_directory=run_directory,
            )

            run_records.append(
                {
                    "model_name": model_name,
                    "seed": int(seed),
                    "parameter_count": int(
                        run_summary[
                            "parameter_count"
                        ]
                    ),
                    "learning_rate": float(
                        protocol[
                            "model_specific_configuration"
                        ][model_name][
                            "learning_rate"
                        ]
                    ),
                    "best_epoch": int(
                        run_summary[
                            "best_epoch"
                        ]
                    ),
                    "epochs_completed": int(
                        run_summary[
                            "epochs_completed"
                        ]
                    ),
                    "best_validation_macro_f1": float(
                        run_summary[
                            "best_validation_macro_f1"
                        ]
                    ),
                    "training_seconds": float(
                        run_summary[
                            "total_training_seconds"
                        ]
                    ),
                    "checkpoint_size_bytes": int(
                        run_summary[
                            "checkpoint_size_bytes"
                        ]
                    ),
                    "test_sample_count": int(
                        test_metrics[
                            "sample_count"
                        ]
                    ),
                    "test_loss": float(
                        test_metrics["loss"]
                    ),
                    "test_accuracy": float(
                        test_metrics[
                            "accuracy"
                        ]
                    ),
                    "test_balanced_accuracy": float(
                        test_metrics[
                            "balanced_accuracy"
                        ]
                    ),
                    "test_macro_precision": float(
                        test_metrics[
                            "macro_precision"
                        ]
                    ),
                    "test_macro_recall": float(
                        test_metrics[
                            "macro_recall"
                        ]
                    ),
                    "test_macro_f1": float(
                        test_metrics[
                            "macro_f1"
                        ]
                    ),
                    "test_weighted_f1": float(
                        test_metrics[
                            "weighted_f1"
                        ]
                    ),
                    "test_mcc": float(
                        test_metrics[
                            "matthews_correlation_coefficient"
                        ]
                    ),
                    "test_elapsed_seconds": float(
                        test_metrics[
                            "elapsed_seconds"
                        ]
                    ),
                    "test_throughput_samples_per_second": float(
                        test_metrics[
                            "throughput_samples_per_second"
                        ]
                    ),
                    "test_evaluation_attempt_count": int(
                        test_metrics[
                            "test_evaluation_attempt_count"
                        ]
                    ),
                    "test_used_for_selection": bool(
                        test_metrics[
                            "test_used_for_selection"
                        ]
                    ),
                    "run_directory": str(
                        run_directory
                    ),
                    "checkpoint_file": str(
                        run_summary[
                            "checkpoint_file"
                        ]
                    ),
                    "test_metrics_file": str(
                        run_directory
                        / "test_metrics.json"
                    ),
                }
            )

            for class_index, class_name in enumerate(
                protocol[
                    "target_classes"
                ]
            ):
                class_metrics = (
                    test_metrics[
                        "per_class"
                    ][class_name]
                )

                per_class_records.append(
                    {
                        "model_name": model_name,
                        "seed": int(seed),
                        "class_index": int(
                            class_index
                        ),
                        "class_name": (
                            class_name
                        ),
                        "support": int(
                            class_metrics[
                                "support"
                            ]
                        ),
                        "predicted_count": int(
                            class_metrics[
                                "predicted_count"
                            ]
                        ),
                        "precision": float(
                            class_metrics[
                                "precision"
                            ]
                        ),
                        "recall": float(
                            class_metrics[
                                "recall"
                            ]
                        ),
                        "f1": float(
                            class_metrics["f1"]
                        ),
                        "false_negative_rate": float(
                            class_metrics[
                                "false_negative_rate"
                            ]
                        ),
                    }
                )

            confusion_matrix = np.asarray(
                test_metrics[
                    "confusion_matrix"
                ],
                dtype=np.int64,
            )

            for true_index, true_class in enumerate(
                protocol[
                    "target_classes"
                ]
            ):
                for predicted_index, predicted_class in enumerate(
                    protocol[
                        "target_classes"
                    ]
                ):
                    confusion_records.append(
                        {
                            "model_name": (
                                model_name
                            ),
                            "seed": int(seed),
                            "true_class_index": int(
                                true_index
                            ),
                            "true_class": (
                                true_class
                            ),
                            "predicted_class_index": int(
                                predicted_index
                            ),
                            "predicted_class": (
                                predicted_class
                            ),
                            "sample_count": int(
                                confusion_matrix[
                                    true_index,
                                    predicted_index,
                                ]
                            ),
                        }
                    )

            print(
                f"Test Macro F1 : "
                f"{test_metrics['macro_f1']:.6f}"
            )
            print(
                f"Test MCC      : "
                f"{test_metrics['matthews_correlation_coefficient']:.6f}"
            )
            print(
                "Test seçime katıldı: False"
            )

            gc.collect()

    # ======================================================
    # Global validation
    # ======================================================

    run_frame = pd.DataFrame(
        run_records
    )

    per_class_frame = pd.DataFrame(
        per_class_records
    )

    confusion_frame = pd.DataFrame(
        confusion_records
    )

    if len(run_frame) != total_run_count:
        raise RuntimeError(
            "Tamamlanan ana deney sayısı 15 değil."
        )

    if run_frame.duplicated(
        subset=[
            "model_name",
            "seed",
        ]
    ).any():
        raise RuntimeError(
            "Tekrarlanan model-seed sonucu bulundu."
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
        for row in run_frame.itertuples(
            index=False
        )
    }

    if actual_pairs != expected_pairs:
        raise RuntimeError(
            "Tamamlanan model-seed kümesi protokolle uyuşmuyor."
        )

    if (
        run_frame[
            "test_evaluation_attempt_count"
        ]
        != 1
    ).any():
        raise RuntimeError(
            "Bir veya daha fazla test değerlendirme girişimi 1 değil."
        )

    if (
        run_frame[
            "test_used_for_selection"
        ]
    ).any():
        raise RuntimeError(
            "Test sonuçlarından biri model seçimine katılmış."
        )

    for model_name in EXPECTED_MODELS:
        model_seed_count = int(
            run_frame[
                run_frame[
                    "model_name"
                ]
                == model_name
            ][
                "seed"
            ].nunique()
        )

        if model_seed_count != len(
            EXPECTED_SEEDS
        ):
            raise RuntimeError(
                f"{model_name} için beş bağımsız seed bulunmuyor."
            )

    aggregate_frame = (
        aggregate_run_metrics(
            run_frame
        )
    )

    per_class_aggregate_frame = (
        aggregate_per_class_metrics(
            per_class_frame
        )
    )

    # ======================================================
    # Ranking
    # ======================================================

    macro_f1_aggregate = (
        aggregate_frame[
            aggregate_frame[
                "metric"
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

    macro_f1_aggregate.insert(
        0,
        "validation_rank",
        range(
            1,
            len(
                macro_f1_aggregate
            )
            + 1,
        ),
    )

    # ======================================================
    # Reports
    # ======================================================

    run_report_file = (
        report_directory
        / "nbaiot_family3_fp32_baseline_runs.csv"
    )

    aggregate_report_file = (
        report_directory
        / "nbaiot_family3_fp32_baseline_aggregate.csv"
    )

    per_class_run_file = (
        report_directory
        / "nbaiot_family3_fp32_baseline_per_class_runs.csv"
    )

    per_class_aggregate_file = (
        report_directory
        / "nbaiot_family3_fp32_baseline_per_class_aggregate.csv"
    )

    confusion_report_file = (
        report_directory
        / "nbaiot_family3_fp32_baseline_confusion_matrices.csv"
    )

    summary_file = (
        report_directory
        / "nbaiot_family3_fp32_baseline_summary.json"
    )

    write_csv_atomic(
        frame=run_frame,
        output_file=run_report_file,
    )

    write_csv_atomic(
        frame=aggregate_frame,
        output_file=aggregate_report_file,
    )

    write_csv_atomic(
        frame=per_class_frame,
        output_file=per_class_run_file,
    )

    write_csv_atomic(
        frame=per_class_aggregate_frame,
        output_file=per_class_aggregate_file,
    )

    write_csv_atomic(
        frame=confusion_frame,
        output_file=confusion_report_file,
    )

    ranking_records = [
        {
            "rank": int(
                row.validation_rank
            ),
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
            "ci95_lower": float(
                row.ci95_lower
            ),
            "ci95_upper": float(
                row.ci95_upper
            ),
        }
        for row in macro_f1_aggregate.itertuples(
            index=False
        )
    ]

    summary_document: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "protocol_file": str(
            protocol_file
        ),
        "protocol_sha256": (
            protocol_sha256
        ),
        "protocol_version": "1.0",
        "task": "family_3",
        "model_count": int(
            len(EXPECTED_MODELS)
        ),
        "seed_count_per_model": int(
            len(EXPECTED_SEEDS)
        ),
        "total_training_run_count": int(
            total_run_count
        ),
        "total_test_evaluation_count": int(
            len(run_frame)
        ),
        "test_evaluations_per_model_seed": 1,
        "test_used_for_model_selection": False,
        "models": list(
            EXPECTED_MODELS
        ),
        "seeds": list(
            EXPECTED_SEEDS
        ),
        "primary_test_metric": (
            "macro_f1"
        ),
        "security_metric": (
            "per_class_false_negative_rate"
        ),
        "aggregation_unit": (
            "independent_training_seed"
        ),
        "confidence_interval_method": (
            "Two-sided 95% Student-t confidence interval "
            "over five independent training seeds."
        ),
        "macro_f1_ranking": (
            ranking_records
        ),
        "best_mean_test_macro_f1_model": (
            ranking_records[
                0
            ][
                "model_name"
            ]
        ),
        "run_report": str(
            run_report_file
        ),
        "aggregate_report": str(
            aggregate_report_file
        ),
        "per_class_run_report": str(
            per_class_run_file
        ),
        "per_class_aggregate_report": str(
            per_class_aggregate_file
        ),
        "confusion_matrix_report": str(
            confusion_report_file
        ),
        "statistical_hypothesis_testing_status": (
            "Not yet performed. Pairwise and omnibus tests "
            "will be conducted in the next stage."
        ),
        "validation_passed": True,
    }

    write_json_atomic(
        document=summary_document,
        output_file=summary_file,
    )

    print()
    print("=" * 78)
    print("FP32 Ana Baseline Deneyleri Tamamlandı")
    print("=" * 78)

    for record in ranking_records:
        print(
            f"{record['rank']}. "
            f"{record['model_name']:14s} | "
            f"Test Macro F1="
            f"{record['mean_test_macro_f1']:.6f} "
            f"± {record['standard_deviation']:.6f} | "
            f"%95 GA=["
            f"{record['ci95_lower']:.6f}, "
            f"{record['ci95_upper']:.6f}]"
        )

    print()
    print(
        "Tamamlanan eğitim    : "
        f"{total_run_count}"
    )
    print(
        "Test değerlendirmesi : "
        f"{len(run_frame)}"
    )
    print(
        "Seed başına test     : 1"
    )
    print(
        "Test seçime katıldı  : False"
    )
    print(
        "Doğrulama geçti      : True"
    )
    print()
    print(f"Seed sonuçları       : {run_report_file}")
    print(f"Toplu istatistikler  : {aggregate_report_file}")
    print(f"Sınıf sonuçları      : {per_class_run_file}")
    print(f"Sınıf istatistikleri : {per_class_aggregate_file}")
    print(f"Karışıklık matrisleri: {confusion_report_file}")
    print(f"JSON özet            : {summary_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()