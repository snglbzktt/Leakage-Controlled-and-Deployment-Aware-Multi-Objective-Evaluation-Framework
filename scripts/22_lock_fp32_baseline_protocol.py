"""
N-BaIoT family_3 FP32 ana baseline deney protokolünü kilitler.

Girdiler:
- 21. aşamada seçilen model bazlı öğrenme oranları
- Görev tanımı
- Model kayıt manifestosu
- Standardizasyon ve sınıf ağırlığı artifactları

Bilimsel kurallar:
- Öğrenme oranları yalnızca validation tabanlı tuning sonucundan alınır.
- Üç mimariye aynı maksimum epoch ve early-stopping bütçesi verilir.
- Ana deneyler beş bağımsız eğitim tohumu ile çalıştırılır.
- Her seed için checkpoint yalnızca validation Macro F1 ile seçilir.
- Test, seçilen checkpoint üzerinde yalnızca bir kez değerlendirilir.
- Test sonuçları hiçbir hiperparametre veya mimari seçimine katılmaz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SWEEP_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_family3_learning_rate_sweep_selected.csv"
)

DEFAULT_TASK_FILE = (
    PROJECT_ROOT
    / "models"
    / "preprocessing"
    / "nbaiot_task_definitions_seed2026.json"
)

DEFAULT_MODEL_REGISTRY_FILE = (
    PROJECT_ROOT
    / "models"
    / "architecture"
    / "nbaiot_model_registry_v1.json"
)

DEFAULT_SCALER_FILE = (
    PROJECT_ROOT
    / "models"
    / "preprocessing"
    / "nbaiot_standard_scaler_seed2026.npz"
)

DEFAULT_WEIGHT_FILE = (
    PROJECT_ROOT
    / "models"
    / "preprocessing"
    / "nbaiot_task_class_weights_seed2026.npz"
)

DEFAULT_PROTOCOL_DIRECTORY = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
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

MAIN_SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)


def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT family_3 FP32 baseline protokolünü "
            "öğrenme oranı taramasından kilitler."
        )
    )

    parser.add_argument(
        "--sweep-file",
        type=Path,
        default=DEFAULT_SWEEP_FILE,
    )

    parser.add_argument(
        "--task-file",
        type=Path,
        default=DEFAULT_TASK_FILE,
    )

    parser.add_argument(
        "--model-registry-file",
        type=Path,
        default=DEFAULT_MODEL_REGISTRY_FILE,
    )

    parser.add_argument(
        "--scaler-file",
        type=Path,
        default=DEFAULT_SCALER_FILE,
    )

    parser.add_argument(
        "--weight-file",
        type=Path,
        default=DEFAULT_WEIGHT_FILE,
    )

    parser.add_argument(
        "--protocol-directory",
        type=Path,
        default=DEFAULT_PROTOCOL_DIRECTORY,
    )

    parser.add_argument(
        "--report-directory",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0002,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--torch-threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


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


def validate_numeric_arguments(
    args: argparse.Namespace,
) -> None:
    """Sayısal protokol parametrelerini doğrular."""

    if args.max_epochs <= 0:
        raise ValueError(
            "max-epochs sıfırdan büyük olmalıdır."
        )

    if args.patience <= 0:
        raise ValueError(
            "patience sıfırdan büyük olmalıdır."
        )

    if args.patience > args.max_epochs:
        raise ValueError(
            "patience, max-epochs değerinden büyük olamaz."
        )

    if args.min_delta < 0:
        raise ValueError(
            "min-delta negatif olamaz."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size sıfırdan büyük olmalıdır."
        )

    if args.weight_decay < 0:
        raise ValueError(
            "weight-decay negatif olamaz."
        )

    if args.gradient_clip_norm < 0:
        raise ValueError(
            "gradient-clip-norm negatif olamaz."
        )

    if args.torch_threads <= 0:
        raise ValueError(
            "torch-threads sıfırdan büyük olmalıdır."
        )


def load_selected_learning_rates(
    sweep_file: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """21. aşamanın seçilen öğrenme oranlarını yükler."""

    if not sweep_file.exists():
        raise FileNotFoundError(
            f"Seçilen tuning raporu bulunamadı: {sweep_file}"
        )

    frame = pd.read_csv(sweep_file)

    required_columns = {
        "model_name",
        "learning_rate",
        "best_epoch",
        "epochs_completed",
        "best_validation_macro_f1",
        "best_validation_loss",
        "test_split_evaluated",
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Tuning raporunda eksik sütunlar var: "
            + ", ".join(sorted(missing_columns))
        )

    frame = frame.copy()

    frame["model_name"] = (
        frame["model_name"]
        .astype(str)
    )

    frame["learning_rate"] = pd.to_numeric(
        frame["learning_rate"],
        errors="raise",
    )

    frame["best_epoch"] = pd.to_numeric(
        frame["best_epoch"],
        errors="raise",
    ).astype("int64")

    frame["epochs_completed"] = pd.to_numeric(
        frame["epochs_completed"],
        errors="raise",
    ).astype("int64")

    frame["best_validation_macro_f1"] = pd.to_numeric(
        frame["best_validation_macro_f1"],
        errors="raise",
    )

    frame["best_validation_loss"] = pd.to_numeric(
        frame["best_validation_loss"],
        errors="raise",
    )

    if len(frame) != len(EXPECTED_MODELS):
        raise ValueError(
            "Seçilen tuning raporunda tam olarak üç model olmalıdır: "
            f"bulunan={len(frame)}"
        )

    if frame["model_name"].duplicated().any():
        raise ValueError(
            "Seçilen tuning raporunda tekrarlanan model var."
        )

    actual_models = set(
        frame["model_name"]
    )

    if actual_models != set(EXPECTED_MODELS):
        raise ValueError(
            "Seçilen model kümesi beklenen modellerle uyuşmuyor: "
            f"bulunan={sorted(actual_models)}"
        )

    if (
        frame["learning_rate"]
        <= 0
    ).any():
        raise ValueError(
            "Seçilen öğrenme oranlarından biri pozitif değil."
        )

    if (
        frame["best_epoch"]
        <= 0
    ).any():
        raise ValueError(
            "Geçersiz best_epoch değeri var."
        )

    if (
        frame["best_epoch"]
        > frame["epochs_completed"]
    ).any():
        raise ValueError(
            "best_epoch, tamamlanan epoch sayısını aşıyor."
        )

    if (
        frame["best_validation_macro_f1"]
        < 0
    ).any() or (
        frame["best_validation_macro_f1"]
        > 1
    ).any():
        raise ValueError(
            "Validation Macro F1 değeri [0, 1] dışında."
        )

    test_values = (
        frame["test_split_evaluated"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_test_access = ~test_values.isin(
        {
            "false",
            "0",
            "no",
        }
    )

    if invalid_test_access.any():
        raise RuntimeError(
            "Tuning çalışmalarından en az biri test splitini kullanmış."
        )

    frame = (
        frame.set_index("model_name")
        .loc[list(EXPECTED_MODELS)]
        .reset_index()
    )

    selected_learning_rates = {
        str(row.model_name): float(
            row.learning_rate
        )
        for row in frame.itertuples(
            index=False
        )
    }

    return (
        frame,
        selected_learning_rates,
    )


def validate_task_definition(
    task_file: Path,
) -> dict[str, Any]:
    """family_3 görev tanımını doğrular."""

    document = read_json(task_file)

    tasks = document.get("tasks")

    if not isinstance(tasks, dict):
        raise ValueError(
            "Görev tanımı dosyasında tasks nesnesi yok."
        )

    if "family_3" not in tasks:
        raise ValueError(
            "Görev tanımı dosyasında family_3 bulunamadı."
        )

    family_task = tasks["family_3"]

    expected_classes = [
        "benign",
        "gafgyt",
        "mirai",
    ]

    actual_classes = [
        str(class_name)
        for class_name
        in family_task["target_classes"]
    ]

    if actual_classes != expected_classes:
        raise ValueError(
            "family_3 hedef sınıf sırası beklenen sırayla uyuşmuyor."
        )

    if family_task["role"] != "primary":
        raise ValueError(
            "family_3 görevi primary olarak işaretlenmemiş."
        )

    if not bool(
        family_task[
            "support_rule_passed"
        ]
    ):
        raise ValueError(
            "family_3 destek yeterliliği başarısız."
        )

    return family_task


def validate_model_registry(
    registry_file: Path,
) -> dict[str, Any]:
    """Model kayıt manifestosunu doğrular."""

    document = read_json(registry_file)

    registered_models = tuple(
        str(model_name)
        for model_name
        in document[
            "registered_model_order"
        ]
    )

    if registered_models != EXPECTED_MODELS:
        raise ValueError(
            "Model kayıt sırası beklenen sırayla uyuşmuyor."
        )

    models = document.get("models")

    if not isinstance(models, dict):
        raise ValueError(
            "Model kayıt manifestosunda models nesnesi yok."
        )

    missing_models = (
        set(EXPECTED_MODELS)
        - set(models)
    )

    if missing_models:
        raise ValueError(
            "Model kayıt manifestosunda eksik mimariler var: "
            + ", ".join(sorted(missing_models))
        )

    return document


def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    validate_numeric_arguments(args)

    sweep_file = args.sweep_file.resolve()
    task_file = args.task_file.resolve()
    registry_file = args.model_registry_file.resolve()
    scaler_file = args.scaler_file.resolve()
    weight_file = args.weight_file.resolve()

    protocol_directory = (
        args.protocol_directory.resolve()
    )

    report_directory = (
        args.report_directory.resolve()
    )

    protocol_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    protocol_file = (
        protocol_directory
        / "nbaiot_family3_fp32_baseline_protocol_v1.json"
    )

    protocol_table_file = (
        report_directory
        / "nbaiot_family3_fp32_baseline_protocol.csv"
    )

    summary_file = (
        report_directory
        / "nbaiot_family3_fp32_baseline_protocol_summary.json"
    )

    output_files = (
        protocol_file,
        protocol_table_file,
        summary_file,
    )

    existing_files = [
        output_file
        for output_file in output_files
        if output_file.exists()
    ]

    if existing_files and not args.overwrite:
        raise FileExistsError(
            "Protokol çıktıları zaten mevcut. "
            "--overwrite kullan."
        )

    (
        selected_frame,
        selected_learning_rates,
    ) = load_selected_learning_rates(
        sweep_file
    )

    family_task = validate_task_definition(
        task_file
    )

    registry_document = validate_model_registry(
        registry_file
    )

    source_files = {
        "learning_rate_selection": sweep_file,
        "task_definition": task_file,
        "model_registry": registry_file,
        "scaler": scaler_file,
        "task_weights": weight_file,
    }

    source_hashes = {
        source_name: {
            "path": str(source_path),
            "sha256": calculate_sha256(
                source_path
            ),
        }
        for source_name, source_path
        in source_files.items()
    }

    model_protocols: dict[
        str,
        dict[str, Any],
    ] = {}

    protocol_rows: list[
        dict[str, Any]
    ] = []

    for model_name in EXPECTED_MODELS:
        selected_row = (
            selected_frame[
                selected_frame[
                    "model_name"
                ]
                == model_name
            ]
            .iloc[0]
        )

        model_specification = (
            registry_document[
                "models"
            ][model_name]
        )

        model_protocols[
            model_name
        ] = {
            "learning_rate": float(
                selected_learning_rates[
                    model_name
                ]
            ),
            "learning_rate_source": (
                "single-seed validation Macro F1 sweep"
            ),
            "tuning_seed": int(
                selected_row[
                    "seed"
                ]
            )
            if "seed" in selected_frame.columns
            else 42,
            "tuning_best_epoch": int(
                selected_row[
                    "best_epoch"
                ]
            ),
            "tuning_epochs_completed": int(
                selected_row[
                    "epochs_completed"
                ]
            ),
            "tuning_validation_macro_f1": float(
                selected_row[
                    "best_validation_macro_f1"
                ]
            ),
            "tuning_validation_loss": float(
                selected_row[
                    "best_validation_loss"
                ]
            ),
            "architecture_family": str(
                model_specification[
                    "architecture_family"
                ]
            ),
            "order_sensitive": bool(
                model_specification[
                    "order_sensitive"
                ]
            ),
        }

        protocol_rows.append(
            {
                "model_name": model_name,
                "learning_rate": float(
                    selected_learning_rates[
                        model_name
                    ]
                ),
                "max_epochs": int(
                    args.max_epochs
                ),
                "patience": int(
                    args.patience
                ),
                "min_delta": float(
                    args.min_delta
                ),
                "batch_size": int(
                    args.batch_size
                ),
                "weight_decay": float(
                    args.weight_decay
                ),
                "gradient_clip_norm": float(
                    args.gradient_clip_norm
                ),
                "weight_scheme": (
                    "inverse_square_root_frequency_mean1"
                ),
                "seed_count": int(
                    len(MAIN_SEEDS)
                ),
                "seeds": ",".join(
                    str(seed)
                    for seed in MAIN_SEEDS
                ),
                "model_selection_split": (
                    "validation"
                ),
                "model_selection_metric": (
                    "macro_f1"
                ),
                "final_evaluation_split": (
                    "test"
                ),
                "test_evaluations_per_seed": 1,
            }
        )

    protocol_document: dict[
        str,
        Any,
    ] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "protocol_name": (
            "N-BaIoT family_3 FP32 multi-seed baseline"
        ),
        "protocol_version": "1.0",
        "status": (
            "locked_before_multi_seed_training"
        ),
        "dataset": (
            "nbaiot_float32_canonical_seed2026"
        ),
        "task": "family_3",
        "task_role": "primary",
        "target_classes": [
            str(class_name)
            for class_name
            in family_task[
                "target_classes"
            ]
        ],
        "models": list(
            EXPECTED_MODELS
        ),
        "training_seeds": list(
            MAIN_SEEDS
        ),
        "seed_count": int(
            len(MAIN_SEEDS)
        ),
        "common_training_configuration": {
            "optimizer": "AdamW",
            "batch_size": int(
                args.batch_size
            ),
            "max_epochs": int(
                args.max_epochs
            ),
            "early_stopping_patience": int(
                args.patience
            ),
            "early_stopping_min_delta": float(
                args.min_delta
            ),
            "weight_decay": float(
                args.weight_decay
            ),
            "gradient_clip_norm": float(
                args.gradient_clip_norm
            ),
            "torch_threads": int(
                args.torch_threads
            ),
            "num_workers": 0,
            "loss": (
                "CrossEntropyLoss"
            ),
            "class_weight_scheme": (
                "inverse_square_root_frequency_mean1"
            ),
            "class_weights_fitted_from": (
                "train_only"
            ),
            "feature_scaler_fitted_from": (
                "train_only"
            ),
        },
        "model_specific_configuration": (
            model_protocols
        ),
        "model_selection": {
            "split": "validation",
            "primary_metric": "macro_f1",
            "selection_direction": "maximize",
            "checkpoint_policy": (
                "Best epoch within each model-seed run"
            ),
            "test_used_for_selection": False,
        },
        "test_evaluation": {
            "enabled": True,
            "timing": (
                "After loading the best validation checkpoint"
            ),
            "evaluations_per_model_seed": 1,
            "hyperparameter_changes_after_test": (
                "forbidden"
            ),
            "test_metrics": [
                "accuracy",
                "balanced_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
                "matthews_correlation_coefficient",
                "per_class_precision",
                "per_class_recall",
                "per_class_f1",
                "per_class_false_negative_rate",
                "confusion_matrix",
            ],
        },
        "planned_run_count": {
            "model_count": int(
                len(EXPECTED_MODELS)
            ),
            "seed_count": int(
                len(MAIN_SEEDS)
            ),
            "total_fp32_training_runs": int(
                len(EXPECTED_MODELS)
                * len(MAIN_SEEDS)
            ),
        },
        "reporting_policy": {
            "aggregation_unit": (
                "independent training seed"
            ),
            "report_mean": True,
            "report_standard_deviation": True,
            "report_median": True,
            "report_minimum": True,
            "report_maximum": True,
            "report_95_percent_confidence_interval": True,
            "primary_metric": (
                "test Macro F1"
            ),
            "security_metric": (
                "per-class false-negative rate"
            ),
        },
        "source_artifacts": (
            source_hashes
        ),
        "protocol_change_policy": (
            "Any change after this lock requires a new protocol "
            "version and must not overwrite version 1.0."
        ),
    }

    write_json_atomic(
        document=protocol_document,
        output_file=protocol_file,
    )

    protocol_frame = pd.DataFrame(
        protocol_rows
    )

    write_csv_atomic(
        frame=protocol_frame,
        output_file=protocol_table_file,
    )

    protocol_sha256 = calculate_sha256(
        protocol_file
    )

    summary_document: dict[
        str,
        Any,
    ] = {
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
        "status": (
            "locked_before_multi_seed_training"
        ),
        "task": "family_3",
        "models": list(
            EXPECTED_MODELS
        ),
        "seeds": list(
            MAIN_SEEDS
        ),
        "model_specific_learning_rates": (
            selected_learning_rates
        ),
        "max_epochs": int(
            args.max_epochs
        ),
        "patience": int(
            args.patience
        ),
        "planned_training_run_count": int(
            len(EXPECTED_MODELS)
            * len(MAIN_SEEDS)
        ),
        "test_selection_leakage_prevention": (
            "Test is evaluated once only after each best "
            "validation checkpoint has been selected."
        ),
        "protocol_table": str(
            protocol_table_file
        ),
        "validation_passed": True,
    }

    write_json_atomic(
        document=summary_document,
        output_file=summary_file,
    )

    print("=" * 78)
    print("N-BaIoT FP32 Ana Baseline Protokolü Kilitlendi")
    print("=" * 78)

    for model_name in EXPECTED_MODELS:
        model_data = model_protocols[
            model_name
        ]

        print(
            f"{model_name:14s} | "
            f"LR={model_data['learning_rate']:g} | "
            f"tuning_best_epoch="
            f"{model_data['tuning_best_epoch']} | "
            f"tuning_val_macro_f1="
            f"{model_data['tuning_validation_macro_f1']:.6f}"
        )

    print()
    print(
        "Ana seedler        : "
        + ", ".join(
            str(seed)
            for seed in MAIN_SEEDS
        )
    )
    print(
        "Toplam FP32 çalışma: "
        f"{len(EXPECTED_MODELS) * len(MAIN_SEEDS)}"
    )
    print(
        "Maksimum epoch     : "
        f"{args.max_epochs}"
    )
    print(
        "Early stopping     : "
        f"patience={args.patience}, "
        f"min_delta={args.min_delta}"
    )
    print(
        "Model seçimi       : validation Macro F1"
    )
    print(
        "Test politikası    : seçilen checkpointte seed başına bir kez"
    )
    print()
    print(f"Protokol dosyası : {protocol_file}")
    print(f"Protokol tablosu : {protocol_table_file}")
    print(f"JSON özet        : {summary_file}")
    print(f"Protokol SHA-256 : {protocol_sha256}")
    print("=" * 78)


if __name__ == "__main__":
    main()