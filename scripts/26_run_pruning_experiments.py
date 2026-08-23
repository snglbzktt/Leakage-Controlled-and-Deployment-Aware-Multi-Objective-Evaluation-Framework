"""
N-BaIoT family_3 fiziksel yapılandırılmış budama ana deneyleri.

Deney planı:
- Modeller: tinyml_mlp, compact_dnn, tiny_1d_cnn
- Budama düzeyleri: P25, P50, P75
- Seedler: 42, 123, 2026, 3407, 8192
- Toplam: 45 bağımsız budama + fine-tuning çalışması

Bilimsel kurallar:
- Her budanmış model, aynı model ve seede ait kilitli FP32
  checkpointten türetilir.
- P25/P50/P75 birbirinden türetilmez; her biri doğrudan
  FP32 modelden üretilir.
- Fine-tuning yalnızca train splitinde yapılır.
- Checkpoint seçimi yalnızca validation Macro F1 ile yapılır.
- Test, seçilmiş checkpoint üzerinde bir kez değerlendirilir.
- Test sonuçları budama, fine-tuning veya checkpoint seçiminde
  kullanılmaz.
- Tamamlanan çalışmalar yeniden çalıştırılmaz.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from torch import nn


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.compression.structured_pruning import (  # noqa: E402
    SUPPORTED_PRUNING_RATIOS,
    physically_prune_model,
    recreate_compact_model,
)
from src.data.nbaiot_pipeline import (  # noqa: E402
    create_nbaiot_dataloader,
    load_pipeline_assets,
)
from src.models.nbaiot_models import create_model  # noqa: E402
from src.training.baseline_trainer import (  # noqa: E402
    run_epoch,
    set_reproducible_environment,
)


DEFAULT_FP32_PROTOCOL_FILE = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_fp32_baseline_protocol_v1.json"
)

DEFAULT_PRUNING_ENGINE_FILE = (
    PROJECT_ROOT
    / "models"
    / "pruning"
    / "nbaiot_structured_pruning_engine_v1.json"
)

DEFAULT_FP32_RUN_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_family3_fp32_baseline_runs.csv"
)

DEFAULT_FP32_PER_CLASS_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_family3_fp32_baseline_per_class_runs.csv"
)

DEFAULT_FP32_EXPERIMENT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "fp32_baseline_v1"
)

DEFAULT_PRUNING_PROTOCOL_FILE = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_structured_pruning_protocol_v1.json"
)

DEFAULT_EXPERIMENT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "structured_pruning_v1"
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

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

PRUNING_RATIOS = tuple(
    float(value)
    for value in SUPPORTED_PRUNING_RATIOS
)

PRUNING_NAMES = tuple(
    f"P{int(round(ratio * 100))}"
    for ratio in PRUNING_RATIOS
)

TARGET_CLASSES = (
    "benign",
    "gafgyt",
    "mirai",
)

GLOBAL_TEST_METRICS = (
    "test_loss",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
    "test_mcc",
    "macro_f1_delta_vs_fp32",
)

PER_CLASS_METRICS = (
    "precision",
    "recall",
    "f1",
    "false_negative_rate",
    "false_negative_rate_delta_vs_fp32",
)


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "Kilitli FP32 checkpointlerden 45 fiziksel budama ve "
            "fine-tuning ana deneyi çalıştırır."
        )
    )

    parser.add_argument(
        "--fp32-protocol-file",
        type=Path,
        default=DEFAULT_FP32_PROTOCOL_FILE,
    )

    parser.add_argument(
        "--pruning-engine-file",
        type=Path,
        default=DEFAULT_PRUNING_ENGINE_FILE,
    )

    parser.add_argument(
        "--fp32-run-file",
        type=Path,
        default=DEFAULT_FP32_RUN_FILE,
    )

    parser.add_argument(
        "--fp32-per-class-file",
        type=Path,
        default=DEFAULT_FP32_PER_CLASS_FILE,
    )

    parser.add_argument(
        "--fp32-experiment-directory",
        type=Path,
        default=DEFAULT_FP32_EXPERIMENT_DIRECTORY,
    )

    parser.add_argument(
        "--pruning-protocol-file",
        type=Path,
        default=DEFAULT_PRUNING_PROTOCOL_FILE,
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
        "--fine-tune-lr-factor",
        type=float,
        default=0.10,
        help=(
            "FP32 öğrenme oranına uygulanacak "
            "fine-tuning çarpanı."
        ),
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0002,
    )

    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help=(
            "training_summary.json oluşmamış yarım çalışma "
            "klasörlerini silerek yeniden başlatır."
        ),
    )

    return parser.parse_args()


# ==========================================================
# FILE HELPERS
# ==========================================================

def json_default(
    value: object,
) -> object:
    """NumPy ve Path nesnelerini JSON uyumlu hâle getirir."""

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        numeric_value = float(
            value
        )

        if not np.isfinite(
            numeric_value
        ):
            return None

        return numeric_value

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        Path,
    ):
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
        document = json.load(
            file_handle
        )

    if not isinstance(
        document,
        dict,
    ):
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

    temporary_file = (
        output_file.with_suffix(
            output_file.suffix
            + ".tmp"
        )
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

    temporary_file.replace(
        output_file
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

    temporary_file = (
        output_file.with_suffix(
            output_file.suffix
            + ".tmp"
        )
    )

    frame.to_csv(
        temporary_file,
        index=False,
        encoding="utf-8",
    )

    temporary_file.replace(
        output_file
    )


def save_torch_atomic(
    document: dict[str, Any],
    output_file: Path,
) -> None:
    """PyTorch checkpointini atomik biçimde yazar."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = (
        output_file.with_suffix(
            output_file.suffix
            + ".tmp"
        )
    )

    torch.save(
        document,
        temporary_file,
    )

    temporary_file.replace(
        output_file
    )


def load_torch_checkpoint(
    checkpoint_file: Path,
) -> dict[str, Any]:
    """PyTorch checkpointini CPU üzerinde yükler."""

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

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            f"Checkpoint kökü sözlük değil: {checkpoint_file}"
        )

    return checkpoint


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

    with file_path.open(
        "rb"
    ) as file_handle:
        while True:
            block = file_handle.read(
                block_size
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


# ==========================================================
# VALIDATION AND PROTOCOL
# ==========================================================

def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """Komut satırı parametrelerini doğrular."""

    if args.fine_tune_lr_factor <= 0:
        raise ValueError(
            "fine-tune-lr-factor pozitif olmalıdır."
        )

    if args.max_epochs <= 0:
        raise ValueError(
            "max-epochs pozitif olmalıdır."
        )

    if args.patience <= 0:
        raise ValueError(
            "patience pozitif olmalıdır."
        )

    if args.patience > args.max_epochs:
        raise ValueError(
            "patience, max-epochs değerinden büyük olamaz."
        )

    if args.min_delta < 0:
        raise ValueError(
            "min-delta negatif olamaz."
        )


def validate_fp32_protocol(
    protocol: dict[str, Any],
) -> None:
    """FP32 protokolünü doğrular."""

    if (
        protocol.get(
            "protocol_version"
        )
        != "1.0"
    ):
        raise ValueError(
            "Beklenmeyen FP32 protokol sürümü."
        )

    if protocol.get(
        "task"
    ) != "family_3":
        raise ValueError(
            "FP32 protokol görevi family_3 değil."
        )

    if tuple(
        protocol[
            "models"
        ]
    ) != MODEL_NAMES:
        raise ValueError(
            "FP32 protokol model sırası uyuşmuyor."
        )

    protocol_seeds = tuple(
        int(seed)
        for seed
        in protocol[
            "training_seeds"
        ]
    )

    if protocol_seeds != SEEDS:
        raise ValueError(
            "FP32 protokol seed sırası uyuşmuyor."
        )

    if tuple(
        protocol[
            "target_classes"
        ]
    ) != TARGET_CLASSES:
        raise ValueError(
            "FP32 protokol sınıf sırası uyuşmuyor."
        )


def validate_pruning_engine(
    engine: dict[str, Any],
) -> None:
    """25. aşamadaki budama motoru manifestini doğrular."""

    if not bool(
        engine.get(
            "validation_passed"
        )
    ):
        raise ValueError(
            "Budama motoru doğrulaması başarılı değil."
        )

    if not bool(
        engine.get(
            "physical_compaction"
        )
    ):
        raise ValueError(
            "Budama motoru fiziksel küçültme uygulamıyor."
        )

    if bool(
        engine.get(
            "mask_based_pruning"
        )
    ):
        raise ValueError(
            "Budama motoru maskeli budama olarak işaretlenmiş."
        )

    if tuple(
        engine.get(
            "supported_models",
            [],
        )
    ) != MODEL_NAMES:
        raise ValueError(
            "Budama motoru model sırası uyuşmuyor."
        )

    engine_ratios = tuple(
        float(value)
        for value
        in engine.get(
            "supported_pruning_ratios",
            [],
        )
    )

    if engine_ratios != PRUNING_RATIOS:
        raise ValueError(
            "Budama motoru oranları P25/P50/P75 ile uyuşmuyor."
        )


def create_or_validate_pruning_protocol(
    *,
    protocol_file: Path,
    fp32_protocol_file: Path,
    fp32_protocol: dict[str, Any],
    pruning_engine_file: Path,
    pruning_engine: dict[str, Any],
    fine_tune_lr_factor: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
) -> tuple[
    dict[str, Any],
    str,
]:
    """Budama protokolünü oluşturur veya mevcut protokolü doğrular."""

    common = fp32_protocol[
        "common_training_configuration"
    ]

    expected_protocol = {
        "protocol_name": (
            "N-BaIoT family_3 physical structured "
            "pruning multi-seed"
        ),
        "protocol_version": "1.0",
        "status": (
            "locked_before_pruning_main_experiments"
        ),
        "task": "family_3",
        "models": list(
            MODEL_NAMES
        ),
        "training_seeds": list(
            SEEDS
        ),
        "pruning_ratios": list(
            PRUNING_RATIOS
        ),
        "pruning_names": list(
            PRUNING_NAMES
        ),
        "planned_run_count": 45,
        "source_policy": (
            "Each pruning level is produced independently "
            "from the matching FP32 model-seed checkpoint."
        ),
        "importance_method": (
            pruning_engine[
                "importance_method"
            ]
        ),
        "physical_compaction": True,
        "mask_based_pruning": False,
        "fine_tuning": {
            "optimizer": "AdamW",
            "learning_rate_rule": (
                "fp32_learning_rate_times_factor"
            ),
            "learning_rate_factor": float(
                fine_tune_lr_factor
            ),
            "max_epochs": int(
                max_epochs
            ),
            "early_stopping_patience": int(
                patience
            ),
            "early_stopping_min_delta": float(
                min_delta
            ),
            "batch_size": int(
                common[
                    "batch_size"
                ]
            ),
            "weight_decay": float(
                common[
                    "weight_decay"
                ]
            ),
            "gradient_clip_norm": float(
                common[
                    "gradient_clip_norm"
                ]
            ),
            "torch_threads": int(
                common[
                    "torch_threads"
                ]
            ),
            "num_workers": int(
                common[
                    "num_workers"
                ]
            ),
            "loss": (
                "CrossEntropyLoss"
            ),
            "class_weight_scheme": str(
                common[
                    "class_weight_scheme"
                ]
            ),
            "feature_scaler_fitted_from": (
                "train_only"
            ),
            "class_weights_fitted_from": (
                "train_only"
            ),
        },
        "model_selection": {
            "split": "validation",
            "metric": "macro_f1",
            "direction": "maximize",
            "test_used_for_selection": False,
        },
        "test_evaluation": {
            "evaluations_per_model_ratio_seed": 1,
            "timing": (
                "after_best_validation_checkpoint_selection"
            ),
            "hyperparameter_changes_after_test": (
                "forbidden"
            ),
        },
        "source_artifacts": {
            "fp32_protocol": {
                "path": str(
                    fp32_protocol_file
                ),
                "sha256": calculate_sha256(
                    fp32_protocol_file
                ),
            },
            "pruning_engine": {
                "path": str(
                    pruning_engine_file
                ),
                "sha256": calculate_sha256(
                    pruning_engine_file
                ),
            },
        },
    }

    if protocol_file.exists():
        protocol = read_json(
            protocol_file
        )

        comparison_keys = (
            "protocol_name",
            "protocol_version",
            "status",
            "task",
            "models",
            "training_seeds",
            "pruning_ratios",
            "pruning_names",
            "planned_run_count",
            "source_policy",
            "importance_method",
            "physical_compaction",
            "mask_based_pruning",
            "fine_tuning",
            "model_selection",
            "test_evaluation",
            "source_artifacts",
        )

        for key in comparison_keys:
            if (
                protocol.get(
                    key
                )
                != expected_protocol.get(
                    key
                )
            ):
                raise RuntimeError(
                    "Mevcut budama protokolü beklenen "
                    f"ayarlarla uyuşmuyor: {key}"
                )

    else:
        protocol = {
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            **expected_protocol,
        }

        write_json_atomic(
            document=protocol,
            output_file=protocol_file,
        )

    protocol_sha256 = calculate_sha256(
        protocol_file
    )

    return (
        protocol,
        protocol_sha256,
    )


# ==========================================================
# FP32 REFERENCES
# ==========================================================

def load_fp32_run_references(
    run_file: Path,
) -> pd.DataFrame:
    """FP32 model-seed sonuçlarını yükler."""

    if not run_file.exists():
        raise FileNotFoundError(
            f"FP32 sonuç dosyası bulunamadı: {run_file}"
        )

    frame = pd.read_csv(
        run_file
    )

    required_columns = {
        "model_name",
        "seed",
        "test_macro_f1",
        "test_mcc",
        "checkpoint_file",
    }

    missing_columns = (
        required_columns
        - set(
            frame.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "FP32 sonuç dosyasında eksik sütunlar var: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    frame = frame.copy()

    frame[
        "model_name"
    ] = frame[
        "model_name"
    ].astype(
        str
    )

    frame[
        "seed"
    ] = pd.to_numeric(
        frame[
            "seed"
        ],
        errors="raise",
    ).astype(
        "int64"
    )

    frame[
        "test_macro_f1"
    ] = pd.to_numeric(
        frame[
            "test_macro_f1"
        ],
        errors="raise",
    )

    frame[
        "test_mcc"
    ] = pd.to_numeric(
        frame[
            "test_mcc"
        ],
        errors="raise",
    )

    if len(
        frame
    ) != 15:
        raise ValueError(
            "FP32 sonuç sayısı 15 değil."
        )

    if frame.duplicated(
        subset=[
            "model_name",
            "seed",
        ]
    ).any():
        raise ValueError(
            "FP32 sonuçlarında tekrarlanan "
            "model-seed kaydı var."
        )

    return frame


def load_fp32_per_class_references(
    per_class_file: Path,
) -> pd.DataFrame:
    """FP32 sınıf bazlı model-seed sonuçlarını yükler."""

    if not per_class_file.exists():
        raise FileNotFoundError(
            "FP32 sınıf sonuçları bulunamadı: "
            f"{per_class_file}"
        )

    frame = pd.read_csv(
        per_class_file
    )

    required_columns = {
        "model_name",
        "seed",
        "class_name",
        "false_negative_rate",
    }

    missing_columns = (
        required_columns
        - set(
            frame.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "FP32 sınıf sonuçlarında eksik sütunlar var: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    frame = frame.copy()

    frame[
        "model_name"
    ] = frame[
        "model_name"
    ].astype(
        str
    )

    frame[
        "seed"
    ] = pd.to_numeric(
        frame[
            "seed"
        ],
        errors="raise",
    ).astype(
        "int64"
    )

    frame[
        "class_name"
    ] = frame[
        "class_name"
    ].astype(
        str
    )

    frame[
        "false_negative_rate"
    ] = pd.to_numeric(
        frame[
            "false_negative_rate"
        ],
        errors="raise",
    )

    if len(
        frame
    ) != 45:
        raise ValueError(
            "FP32 sınıf sonucu sayısı 45 değil."
        )

    if frame.duplicated(
        subset=[
            "model_name",
            "seed",
            "class_name",
        ]
    ).any():
        raise ValueError(
            "FP32 sınıf sonuçlarında "
            "tekrarlanan kayıt var."
        )

    return frame


# ==========================================================
# RUN CONFIGURATION
# ==========================================================

def build_run_configuration(
    *,
    model_name: str,
    seed: int,
    pruning_ratio: float,
    pruning_name: str,
    source_checkpoint_file: Path,
    source_checkpoint_sha256: str,
    fp32_learning_rate: float,
    pruning_protocol: dict[str, Any],
    pruning_protocol_sha256: str,
) -> dict[str, Any]:
    """Tek model-ratio-seed yapılandırmasını oluşturur."""

    fine_tuning = pruning_protocol[
        "fine_tuning"
    ]

    fine_tune_learning_rate = (
        float(
            fp32_learning_rate
        )
        * float(
            fine_tuning[
                "learning_rate_factor"
            ]
        )
    )

    return {
        "experiment_name": (
            f"structured_pruning_family3_{model_name}_"
            f"{pruning_name}_seed{seed}"
        ),
        "mode": "pruning_main",
        "task": "family_3",
        "model_name": model_name,
        "seed": int(
            seed
        ),
        "pruning_name": pruning_name,
        "pruning_ratio": float(
            pruning_ratio
        ),
        "source_checkpoint_file": str(
            source_checkpoint_file
        ),
        "source_checkpoint_sha256": (
            source_checkpoint_sha256
        ),
        "fp32_learning_rate": float(
            fp32_learning_rate
        ),
        "fine_tune_learning_rate": float(
            fine_tune_learning_rate
        ),
        "max_epochs": int(
            fine_tuning[
                "max_epochs"
            ]
        ),
        "patience": int(
            fine_tuning[
                "early_stopping_patience"
            ]
        ),
        "min_delta": float(
            fine_tuning[
                "early_stopping_min_delta"
            ]
        ),
        "batch_size": int(
            fine_tuning[
                "batch_size"
            ]
        ),
        "weight_decay": float(
            fine_tuning[
                "weight_decay"
            ]
        ),
        "gradient_clip_norm": float(
            fine_tuning[
                "gradient_clip_norm"
            ]
        ),
        "torch_threads": int(
            fine_tuning[
                "torch_threads"
            ]
        ),
        "num_workers": int(
            fine_tuning[
                "num_workers"
            ]
        ),
        "weight_scheme": str(
            fine_tuning[
                "class_weight_scheme"
            ]
        ),
        "selection_split": "validation",
        "selection_metric": "macro_f1",
        "test_used_for_selection": False,
        "pruning_protocol_sha256": (
            pruning_protocol_sha256
        ),
    }


def validate_source_checkpoint(
    *,
    checkpoint: dict[str, Any],
    model_name: str,
    seed: int,
    class_names: tuple[str, ...],
) -> None:
    """Kaynak FP32 checkpoint kimliğini doğrular."""

    if str(
        checkpoint.get(
            "model_name"
        )
    ) != model_name:
        raise RuntimeError(
            "Kaynak checkpoint model adı uyuşmuyor."
        )

    if str(
        checkpoint.get(
            "task_name"
        )
    ) != "family_3":
        raise RuntimeError(
            "Kaynak checkpoint görevi family_3 değil."
        )

    if int(
        checkpoint.get(
            "seed"
        )
    ) != seed:
        raise RuntimeError(
            "Kaynak checkpoint seed değeri uyuşmuyor."
        )

    checkpoint_classes = tuple(
        str(value)
        for value
        in checkpoint.get(
            "class_names",
            [],
        )
    )

    if checkpoint_classes != (
        class_names
    ):
        raise RuntimeError(
            "Kaynak checkpoint sınıf sırası uyuşmuyor."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Kaynak checkpoint içinde "
            "model_state_dict yok."
        )


# ==========================================================
# FINE-TUNING
# ==========================================================

def train_pruned_model(
    *,
    compact_model: nn.Module,
    pruning_metadata: dict[str, Any],
    run_configuration: dict[str, Any],
    assets: Any,
    run_directory: Path,
) -> dict[str, Any]:
    """Budanmış modeli train üzerinde fine-tune eder."""

    seed = int(
        run_configuration[
            "seed"
        ]
    )

    set_reproducible_environment(
        seed=seed,
        torch_threads=int(
            run_configuration[
                "torch_threads"
            ]
        ),
    )

    class_weights = torch.as_tensor(
        assets.class_weights,
        dtype=torch.float32,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        compact_model.parameters(),
        lr=float(
            run_configuration[
                "fine_tune_learning_rate"
            ]
        ),
        weight_decay=float(
            run_configuration[
                "weight_decay"
            ]
        ),
    )

    checkpoint_file = (
        run_directory
        / "best_checkpoint.pt"
    )

    history_file = (
        run_directory
        / "training_history.csv"
    )

    training_summary_file = (
        run_directory
        / "training_summary.json"
    )

    history_records: list[
        dict[str, Any]
    ] = []

    best_validation_macro_f1 = (
        -np.inf
    )

    best_epoch = 0
    epochs_without_improvement = 0

    training_start = (
        time.perf_counter()
    )

    for epoch in range(
        1,
        int(
            run_configuration[
                "max_epochs"
            ]
        )
        + 1,
    ):
        _, train_loader = (
            create_nbaiot_dataloader(
                assets=assets,
                split_name="train",
                batch_size=int(
                    run_configuration[
                        "batch_size"
                    ]
                ),
                shuffle=True,
                seed=seed,
                epoch=epoch,
                num_workers=int(
                    run_configuration[
                        "num_workers"
                    ]
                ),
                pin_memory=False,
            )
        )

        _, validation_loader = (
            create_nbaiot_dataloader(
                assets=assets,
                split_name="validation",
                batch_size=int(
                    run_configuration[
                        "batch_size"
                    ]
                ),
                shuffle=False,
                seed=seed,
                epoch=0,
                num_workers=int(
                    run_configuration[
                        "num_workers"
                    ]
                ),
                pin_memory=False,
            )
        )

        train_metrics = run_epoch(
            model=compact_model,
            data_loader=train_loader,
            criterion=criterion,
            class_names=(
                assets.target_classes
            ),
            optimizer=optimizer,
            gradient_clip_norm=float(
                run_configuration[
                    "gradient_clip_norm"
                ]
            ),
            maximum_batches=None,
        )

        validation_metrics = (
            run_epoch(
                model=compact_model,
                data_loader=(
                    validation_loader
                ),
                criterion=criterion,
                class_names=(
                    assets.target_classes
                ),
                optimizer=None,
                gradient_clip_norm=0.0,
                maximum_batches=None,
            )
        )

        validation_macro_f1 = float(
            validation_metrics[
                "macro_f1"
            ]
        )

        improved = bool(
            validation_macro_f1
            > best_validation_macro_f1
            + float(
                run_configuration[
                    "min_delta"
                ]
            )
        )

        if improved:
            best_validation_macro_f1 = (
                validation_macro_f1
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint_document = {
                "created_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "checkpoint_type": (
                    "best_validation_structured_"
                    "pruning_finetune"
                ),
                "model_name": (
                    run_configuration[
                        "model_name"
                    ]
                ),
                "task_name": "family_3",
                "seed": seed,
                "class_names": list(
                    assets.target_classes
                ),
                "best_epoch": int(
                    best_epoch
                ),
                "best_validation_macro_f1": float(
                    best_validation_macro_f1
                ),
                "model_state_dict": (
                    compact_model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "pruning_metadata": (
                    pruning_metadata
                ),
                "run_configuration": (
                    run_configuration
                ),
                "test_split_evaluated": False,
            }

            save_torch_atomic(
                document=checkpoint_document,
                output_file=checkpoint_file,
            )

        else:
            epochs_without_improvement += 1

        history_records.append(
            {
                "epoch": int(
                    epoch
                ),
                "train_loss": float(
                    train_metrics[
                        "loss"
                    ]
                ),
                "train_accuracy": float(
                    train_metrics[
                        "accuracy"
                    ]
                ),
                "train_macro_f1": float(
                    train_metrics[
                        "macro_f1"
                    ]
                ),
                "train_mcc": float(
                    train_metrics[
                        "matthews_correlation_coefficient"
                    ]
                ),
                "validation_loss": float(
                    validation_metrics[
                        "loss"
                    ]
                ),
                "validation_accuracy": float(
                    validation_metrics[
                        "accuracy"
                    ]
                ),
                "validation_macro_f1": float(
                    validation_metrics[
                        "macro_f1"
                    ]
                ),
                "validation_mcc": float(
                    validation_metrics[
                        "matthews_correlation_coefficient"
                    ]
                ),
                "improved": (
                    improved
                ),
                "epochs_without_improvement": int(
                    epochs_without_improvement
                ),
            }
        )

        print(
            f"    epoch={epoch:02d} | "
            f"train_macro_f1="
            f"{train_metrics['macro_f1']:.6f} | "
            f"val_macro_f1="
            f"{validation_macro_f1:.6f} | "
            f"best="
            f"{best_validation_macro_f1:.6f} | "
            f"improved={improved}"
        )

        if (
            epochs_without_improvement
            >= int(
                run_configuration[
                    "patience"
                ]
            )
        ):
            break

    total_training_seconds = float(
        time.perf_counter()
        - training_start
    )

    if (
        best_epoch <= 0
        or not checkpoint_file.exists()
    ):
        raise RuntimeError(
            "Fine-tuning en iyi checkpoint oluşturmadı."
        )

    history_frame = pd.DataFrame(
        history_records
    )

    write_csv_atomic(
        frame=history_frame,
        output_file=history_file,
    )

    training_summary = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_name": (
            run_configuration[
                "model_name"
            ]
        ),
        "seed": seed,
        "pruning_name": (
            run_configuration[
                "pruning_name"
            ]
        ),
        "pruning_ratio": float(
            run_configuration[
                "pruning_ratio"
            ]
        ),
        "best_epoch": int(
            best_epoch
        ),
        "epochs_completed": int(
            len(
                history_records
            )
        ),
        "best_validation_macro_f1": float(
            best_validation_macro_f1
        ),
        "total_training_seconds": (
            total_training_seconds
        ),
        "checkpoint_file": str(
            checkpoint_file
        ),
        "checkpoint_sha256": calculate_sha256(
            checkpoint_file
        ),
        "history_file": str(
            history_file
        ),
        "pruning_metadata": (
            pruning_metadata
        ),
        "run_configuration": (
            run_configuration
        ),
        "test_split_evaluated": False,
        "scientific_result_status": (
            "main_pruning_training_completed_"
            "test_not_yet_evaluated"
        ),
    }

    write_json_atomic(
        document=training_summary,
        output_file=training_summary_file,
    )

    return training_summary


# ==========================================================
# TEST
# ==========================================================

def evaluate_pruned_test_once(
    *,
    training_summary: dict[str, Any],
    assets: Any,
    run_directory: Path,
) -> dict[str, Any]:
    """En iyi budanmış checkpointi testte bir kez değerlendirir."""

    checkpoint_file = Path(
        str(
            training_summary[
                "checkpoint_file"
            ]
        )
    ).resolve()

    checkpoint_sha256 = (
        calculate_sha256(
            checkpoint_file
        )
    )

    marker_file = (
        run_directory
        / "test_evaluation_manifest.json"
    )

    test_metrics_file = (
        run_directory
        / "test_metrics.json"
    )

    per_class_file = (
        run_directory
        / "test_per_class_metrics.csv"
    )

    confusion_file = (
        run_directory
        / "test_confusion_matrix.csv"
    )

    if marker_file.exists():
        marker = read_json(
            marker_file
        )

        if marker.get(
            "status"
        ) == "started":
            raise RuntimeError(
                "Test değerlendirmesi daha önce başlatılmış "
                "ancak tamamlanmamış. Otomatik tekrar yapılmadı: "
                f"{marker_file}"
            )

        if marker.get(
            "status"
        ) != "completed":
            raise RuntimeError(
                "Bilinmeyen test değerlendirme durumu."
            )

        if marker.get(
            "checkpoint_sha256"
        ) != checkpoint_sha256:
            raise RuntimeError(
                "Mevcut test sonucu farklı checkpointe ait."
            )

        if not test_metrics_file.exists():
            raise RuntimeError(
                "Test manifesti tamamlandı ancak "
                "test_metrics.json yok."
            )

        return read_json(
            test_metrics_file
        )

    started_marker = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "started",
        "model_name": (
            training_summary[
                "model_name"
            ]
        ),
        "seed": int(
            training_summary[
                "seed"
            ]
        ),
        "pruning_name": (
            training_summary[
                "pruning_name"
            ]
        ),
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
        output_file=marker_file,
    )

    checkpoint = load_torch_checkpoint(
        checkpoint_file
    )

    pruning_metadata = (
        checkpoint[
            "pruning_metadata"
        ]
    )

    model = recreate_compact_model(
        pruning_metadata
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

    run_configuration = (
        checkpoint[
            "run_configuration"
        ]
    )

    _, test_loader = (
        create_nbaiot_dataloader(
            assets=assets,
            split_name="test",
            batch_size=int(
                run_configuration[
                    "batch_size"
                ]
            ),
            shuffle=False,
            seed=int(
                run_configuration[
                    "seed"
                ]
            ),
            epoch=0,
            num_workers=int(
                run_configuration[
                    "num_workers"
                ]
            ),
            pin_memory=False,
        )
    )

    test_metrics = run_epoch(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        class_names=(
            assets.target_classes
        ),
        optimizer=None,
        gradient_clip_norm=0.0,
        maximum_batches=None,
    )

    expected_test_count = int(
        assets.task_definition[
            "split_totals"
        ][
            "test"
        ]
    )

    if int(
        test_metrics[
            "sample_count"
        ]
    ) != expected_test_count:
        raise RuntimeError(
            "Test örnek sayısı görev tanımıyla uyuşmuyor."
        )

    test_document = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_name": (
            training_summary[
                "model_name"
            ]
        ),
        "task_name": "family_3",
        "seed": int(
            training_summary[
                "seed"
            ]
        ),
        "pruning_name": (
            training_summary[
                "pruning_name"
            ]
        ),
        "pruning_ratio": float(
            training_summary[
                "pruning_ratio"
            ]
        ),
        "best_epoch": int(
            training_summary[
                "best_epoch"
            ]
        ),
        "best_validation_macro_f1": float(
            training_summary[
                "best_validation_macro_f1"
            ]
        ),
        "checkpoint_selection_split": (
            "validation"
        ),
        "checkpoint_selection_metric": (
            "macro_f1"
        ),
        "test_used_for_selection": False,
        "test_evaluation_attempt_count": 1,
        "checkpoint_sha256": (
            checkpoint_sha256
        ),
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
        class_metrics = (
            test_metrics[
                "per_class"
            ][
                class_name
            ]
        )

        per_class_records.append(
            {
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
                    class_metrics[
                        "f1"
                    ]
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
        output_file=per_class_file,
    )

    confusion_matrix = np.asarray(
        test_metrics[
            "confusion_matrix"
        ],
        dtype=np.int64,
    )

    confusion_frame = pd.DataFrame(
        confusion_matrix,
        index=(
            assets.target_classes
        ),
        columns=(
            assets.target_classes
        ),
    )

    confusion_frame.index.name = (
        "true_class"
    )

    confusion_frame.to_csv(
        confusion_file,
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
            per_class_file
        ),
        "test_confusion_file": str(
            confusion_file
        ),
    }

    write_json_atomic(
        document=completed_marker,
        output_file=marker_file,
    )

    return test_document


# ==========================================================
# STATISTICS
# ==========================================================

def calculate_summary_statistics(
    values: np.ndarray,
) -> dict[str, float | int]:
    """Beş seed için özet ve %95 güven aralığı hesaplar."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        values.ndim != 1
        or len(
            values
        ) == 0
    ):
        raise ValueError(
            "İstatistik dizisi tek boyutlu ve "
            "boş olmayan olmalıdır."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "İstatistik dizisinde sonlu olmayan değer var."
        )

    sample_count = int(
        len(
            values
        )
    )

    mean_value = float(
        values.mean()
    )

    median_value = float(
        np.median(
            values
        )
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
            / np.sqrt(
                sample_count
            )
        )

        critical_value = float(
            student_t.ppf(
                0.975,
                df=(
                    sample_count
                    - 1
                ),
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
            - margin
        ),
        "ci95_upper": float(
            mean_value
            + margin
        ),
    }


def aggregate_global_metrics(
    run_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Model ve budama düzeyi bazında sonuçları özetler."""

    records: list[
        dict[str, Any]
    ] = []

    for model_name in MODEL_NAMES:
        for pruning_name in PRUNING_NAMES:
            subset = run_frame[
                (
                    run_frame[
                        "model_name"
                    ]
                    == model_name
                )
                & (
                    run_frame[
                        "pruning_name"
                    ]
                    == pruning_name
                )
            ]

            for metric_name in (
                GLOBAL_TEST_METRICS
            ):
                statistics = (
                    calculate_summary_statistics(
                        subset[
                            metric_name
                        ].to_numpy(
                            dtype=np.float64
                        )
                    )
                )

                records.append(
                    {
                        "model_name": (
                            model_name
                        ),
                        "pruning_name": (
                            pruning_name
                        ),
                        "metric": (
                            metric_name
                        ),
                        **statistics,
                    }
                )

    return pd.DataFrame(
        records
    )


def aggregate_per_class_metrics(
    per_class_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Model, budama düzeyi ve sınıf bazında özetler."""

    records: list[
        dict[str, Any]
    ] = []

    for model_name in MODEL_NAMES:
        for pruning_name in PRUNING_NAMES:
            for class_name in TARGET_CLASSES:
                subset = per_class_frame[
                    (
                        per_class_frame[
                            "model_name"
                        ]
                        == model_name
                    )
                    & (
                        per_class_frame[
                            "pruning_name"
                        ]
                        == pruning_name
                    )
                    & (
                        per_class_frame[
                            "class_name"
                        ]
                        == class_name
                    )
                ]

                for metric_name in (
                    PER_CLASS_METRICS
                ):
                    statistics = (
                        calculate_summary_statistics(
                            subset[
                                metric_name
                            ].to_numpy(
                                dtype=np.float64
                            )
                        )
                    )

                    records.append(
                        {
                            "model_name": (
                                model_name
                            ),
                            "pruning_name": (
                                pruning_name
                            ),
                            "class_name": (
                                class_name
                            ),
                            "metric": (
                                metric_name
                            ),
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
    """45 budama ve fine-tuning deneyini çalıştırır."""

    args = parse_arguments()

    validate_arguments(
        args
    )

    fp32_protocol_file = (
        args.fp32_protocol_file.resolve()
    )

    pruning_engine_file = (
        args.pruning_engine_file.resolve()
    )

    fp32_run_file = (
        args.fp32_run_file.resolve()
    )

    fp32_per_class_file = (
        args.fp32_per_class_file.resolve()
    )

    fp32_experiment_directory = (
        args.fp32_experiment_directory.resolve()
    )

    pruning_protocol_file = (
        args.pruning_protocol_file.resolve()
    )

    experiment_directory = (
        args.experiment_directory.resolve()
    )

    report_directory = (
        args.report_directory.resolve()
    )

    experiment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pruning_protocol_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fp32_protocol = read_json(
        fp32_protocol_file
    )

    validate_fp32_protocol(
        fp32_protocol
    )

    pruning_engine = read_json(
        pruning_engine_file
    )

    validate_pruning_engine(
        pruning_engine
    )

    (
        pruning_protocol,
        pruning_protocol_sha256,
    ) = create_or_validate_pruning_protocol(
        protocol_file=(
            pruning_protocol_file
        ),
        fp32_protocol_file=(
            fp32_protocol_file
        ),
        fp32_protocol=(
            fp32_protocol
        ),
        pruning_engine_file=(
            pruning_engine_file
        ),
        pruning_engine=(
            pruning_engine
        ),
        fine_tune_lr_factor=(
            args.fine_tune_lr_factor
        ),
        max_epochs=(
            args.max_epochs
        ),
        patience=(
            args.patience
        ),
        min_delta=(
            args.min_delta
        ),
    )

    fp32_runs = (
        load_fp32_run_references(
            fp32_run_file
        )
    )

    fp32_per_class = (
        load_fp32_per_class_references(
            fp32_per_class_file
        )
    )

    assets = load_pipeline_assets(
        task_name="family_3",
        weight_scheme=(
            pruning_protocol[
                "fine_tuning"
            ][
                "class_weight_scheme"
            ]
        ),
        expected_seed=2026,
    )

    if tuple(
        assets.target_classes
    ) != TARGET_CLASSES:
        raise RuntimeError(
            "Veri hattı sınıf sırası "
            "family_3 protokolüyle uyuşmuyor."
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
        len(
            MODEL_NAMES
        )
        * len(
            PRUNING_RATIOS
        )
        * len(
            SEEDS
        )
    )

    print("=" * 78)
    print(
        "N-BaIoT Family-3 Yapılandırılmış "
        "Budama Ana Deneyleri"
    )
    print("=" * 78)
    print(
        "Modeller        : "
        + ", ".join(
            MODEL_NAMES
        )
    )
    print(
        "Budama düzeyleri: "
        + ", ".join(
            PRUNING_NAMES
        )
    )
    print(
        "Seedler         : "
        + ", ".join(
            str(seed)
            for seed in SEEDS
        )
    )
    print(
        f"Toplam çalışma  : {total_run_count}"
    )
    print(
        "Fine-tune LR    : FP32 öğrenme oranının "
        f"{args.fine_tune_lr_factor:g} katı"
    )
    print(
        f"Maksimum epoch  : {args.max_epochs}"
    )
    print(
        f"Patience        : {args.patience}"
    )
    print(
        "Model seçimi    : validation Macro F1"
    )
    print(
        "Test politikası : seçilen checkpointte bir kez"
    )
    print(
        f"Protokol SHA-256: {pruning_protocol_sha256}"
    )
    print("=" * 78)

    run_index = 0

    for model_name in MODEL_NAMES:
        model_configuration = (
            fp32_protocol[
                "model_specific_configuration"
            ][
                model_name
            ]
        )

        fp32_learning_rate = float(
            model_configuration[
                "learning_rate"
            ]
        )

        for (
            pruning_ratio,
            pruning_name,
        ) in zip(
            PRUNING_RATIOS,
            PRUNING_NAMES,
            strict=True,
        ):
            for seed in SEEDS:
                run_index += 1

                run_directory = (
                    experiment_directory
                    / model_name
                    / pruning_name
                    / f"seed{seed}"
                )

                final_summary_file = (
                    run_directory
                    / "run_summary.json"
                )

                training_summary_file = (
                    run_directory
                    / "training_summary.json"
                )

                run_config_file = (
                    run_directory
                    / "run_config.json"
                )

                source_checkpoint_file = (
                    fp32_experiment_directory
                    / model_name
                    / f"seed{seed}"
                    / "best_checkpoint.pt"
                )

                source_checkpoint_sha256 = (
                    calculate_sha256(
                        source_checkpoint_file
                    )
                )

                run_configuration = (
                    build_run_configuration(
                        model_name=(
                            model_name
                        ),
                        seed=seed,
                        pruning_ratio=(
                            pruning_ratio
                        ),
                        pruning_name=(
                            pruning_name
                        ),
                        source_checkpoint_file=(
                            source_checkpoint_file
                        ),
                        source_checkpoint_sha256=(
                            source_checkpoint_sha256
                        ),
                        fp32_learning_rate=(
                            fp32_learning_rate
                        ),
                        pruning_protocol=(
                            pruning_protocol
                        ),
                        pruning_protocol_sha256=(
                            pruning_protocol_sha256
                        ),
                    )
                )

                print()
                print("=" * 78)
                print(
                    f"Çalışma {run_index}/{total_run_count}"
                )
                print(
                    f"Model  : {model_name}"
                )
                print(
                    f"Budama : {pruning_name}"
                )
                print(
                    f"Seed   : {seed}"
                )
                print(
                    "FT LR  : "
                    f"{run_configuration['fine_tune_learning_rate']:g}"
                )
                print("=" * 78)

                if final_summary_file.exists():
                    final_summary = read_json(
                        final_summary_file
                    )

                    if (
                        final_summary[
                            "pruning_protocol_sha256"
                        ]
                        != pruning_protocol_sha256
                    ):
                        raise RuntimeError(
                            "Tamamlanmış çalışma farklı "
                            "budama protokolüne ait."
                        )

                    if (
                        final_summary[
                            "source_checkpoint_sha256"
                        ]
                        != source_checkpoint_sha256
                    ):
                        raise RuntimeError(
                            "Tamamlanmış çalışma farklı "
                            "FP32 checkpointe ait."
                        )

                    print(
                        "Çalışma daha önce tamamlanmış; "
                        "sonuç yeniden kullanılıyor."
                    )

                    test_metrics = read_json(
                        Path(
                            final_summary[
                                "test_metrics_file"
                            ]
                        )
                    )

                    training_summary = read_json(
                        Path(
                            final_summary[
                                "training_summary_file"
                            ]
                        )
                    )

                    pruning_metadata = (
                        training_summary[
                            "pruning_metadata"
                        ]
                    )

                else:
                    if (
                        run_directory.exists()
                        and not training_summary_file.exists()
                    ):
                        if args.restart_incomplete:
                            shutil.rmtree(
                                run_directory
                            )

                        else:
                            raise RuntimeError(
                                "Yarım kalmış fine-tuning "
                                "klasörü bulundu: "
                                f"{run_directory}\n"
                                "Yeniden başlatmak için "
                                "--restart-incomplete kullan."
                            )

                    run_directory.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    if run_config_file.exists():
                        existing_config = read_json(
                            run_config_file
                        )

                        if (
                            existing_config
                            != run_configuration
                        ):
                            raise RuntimeError(
                                "Mevcut run_config.json "
                                "beklenen ayarlarla uyuşmuyor."
                            )

                    else:
                        write_json_atomic(
                            document=(
                                run_configuration
                            ),
                            output_file=(
                                run_config_file
                            ),
                        )

                    source_checkpoint = (
                        load_torch_checkpoint(
                            source_checkpoint_file
                        )
                    )

                    validate_source_checkpoint(
                        checkpoint=(
                            source_checkpoint
                        ),
                        model_name=(
                            model_name
                        ),
                        seed=seed,
                        class_names=(
                            assets.target_classes
                        ),
                    )

                    original_model = create_model(
                        model_name=(
                            model_name
                        ),
                        input_features=(
                            assets.feature_count
                        ),
                        num_classes=(
                            assets.class_count
                        ),
                    )

                    original_model.load_state_dict(
                        source_checkpoint[
                            "model_state_dict"
                        ],
                        strict=True,
                    )

                    original_model.cpu()
                    original_model.eval()

                    (
                        compact_model,
                        metadata_object,
                    ) = physically_prune_model(
                        model_name=(
                            model_name
                        ),
                        original_model=(
                            original_model
                        ),
                        pruning_ratio=(
                            pruning_ratio
                        ),
                    )

                    pruning_metadata = (
                        metadata_object.to_dict()
                    )

                    if training_summary_file.exists():
                        training_summary = read_json(
                            training_summary_file
                        )

                        if (
                            training_summary[
                                "run_configuration"
                            ]
                            != run_configuration
                        ):
                            raise RuntimeError(
                                "Mevcut training_summary "
                                "farklı ayarlara ait."
                            )

                        print(
                            "Fine-tuning daha önce tamamlanmış; "
                            "test aşamasına devam ediliyor."
                        )

                    else:
                        training_summary = (
                            train_pruned_model(
                                compact_model=(
                                    compact_model
                                ),
                                pruning_metadata=(
                                    pruning_metadata
                                ),
                                run_configuration=(
                                    run_configuration
                                ),
                                assets=assets,
                                run_directory=(
                                    run_directory
                                ),
                            )
                        )

                    test_metrics = (
                        evaluate_pruned_test_once(
                            training_summary=(
                                training_summary
                            ),
                            assets=assets,
                            run_directory=(
                                run_directory
                            ),
                        )
                    )

                    final_summary = {
                        "generated_at_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "model_name": (
                            model_name
                        ),
                        "seed": int(
                            seed
                        ),
                        "pruning_name": (
                            pruning_name
                        ),
                        "pruning_ratio": float(
                            pruning_ratio
                        ),
                        "pruning_protocol_file": str(
                            pruning_protocol_file
                        ),
                        "pruning_protocol_sha256": (
                            pruning_protocol_sha256
                        ),
                        "source_checkpoint_file": str(
                            source_checkpoint_file
                        ),
                        "source_checkpoint_sha256": (
                            source_checkpoint_sha256
                        ),
                        "training_summary_file": str(
                            training_summary_file
                        ),
                        "test_metrics_file": str(
                            run_directory
                            / "test_metrics.json"
                        ),
                        "test_evaluation_attempt_count": 1,
                        "test_used_for_model_selection": False,
                        "scientific_result_status": (
                            "main_structured_pruning_result"
                        ),
                        "validation_passed": True,
                    }

                    write_json_atomic(
                        document=(
                            final_summary
                        ),
                        output_file=(
                            final_summary_file
                        ),
                    )

                fp32_row = fp32_runs[
                    (
                        fp32_runs[
                            "model_name"
                        ]
                        == model_name
                    )
                    & (
                        fp32_runs[
                            "seed"
                        ]
                        == seed
                    )
                ].iloc[
                    0
                ]

                fp32_test_macro_f1 = float(
                    fp32_row[
                        "test_macro_f1"
                    ]
                )

                fp32_test_mcc = float(
                    fp32_row[
                        "test_mcc"
                    ]
                )

                run_records.append(
                    {
                        "model_name": (
                            model_name
                        ),
                        "pruning_name": (
                            pruning_name
                        ),
                        "pruning_ratio": float(
                            pruning_ratio
                        ),
                        "seed": int(
                            seed
                        ),
                        "fp32_learning_rate": float(
                            run_configuration[
                                "fp32_learning_rate"
                            ]
                        ),
                        "fine_tune_learning_rate": float(
                            run_configuration[
                                "fine_tune_learning_rate"
                            ]
                        ),
                        "best_epoch": int(
                            training_summary[
                                "best_epoch"
                            ]
                        ),
                        "epochs_completed": int(
                            training_summary[
                                "epochs_completed"
                            ]
                        ),
                        "best_validation_macro_f1": float(
                            training_summary[
                                "best_validation_macro_f1"
                            ]
                        ),
                        "training_seconds": float(
                            training_summary[
                                "total_training_seconds"
                            ]
                        ),
                        "original_parameter_count": int(
                            pruning_metadata[
                                "original_parameter_count"
                            ]
                        ),
                        "compact_parameter_count": int(
                            pruning_metadata[
                                "compact_parameter_count"
                            ]
                        ),
                        "parameter_reduction_ratio": float(
                            pruning_metadata[
                                "parameter_reduction_ratio"
                            ]
                        ),
                        "original_weight_layer_macs": int(
                            pruning_metadata[
                                "original_weight_layer_macs"
                            ]
                        ),
                        "compact_weight_layer_macs": int(
                            pruning_metadata[
                                "compact_weight_layer_macs"
                            ]
                        ),
                        "mac_reduction_ratio": float(
                            pruning_metadata[
                                "mac_reduction_ratio"
                            ]
                        ),
                        "fp32_test_macro_f1": (
                            fp32_test_macro_f1
                        ),
                        "test_loss": float(
                            test_metrics[
                                "loss"
                            ]
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
                        "fp32_test_mcc": (
                            fp32_test_mcc
                        ),
                        "test_mcc": float(
                            test_metrics[
                                "matthews_correlation_coefficient"
                            ]
                        ),
                        "macro_f1_delta_vs_fp32": float(
                            test_metrics[
                                "macro_f1"
                            ]
                            - fp32_test_macro_f1
                        ),
                        "test_sample_count": int(
                            test_metrics[
                                "sample_count"
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
                        "source_checkpoint_file": str(
                            source_checkpoint_file
                        ),
                        "best_checkpoint_file": str(
                            training_summary[
                                "checkpoint_file"
                            ]
                        ),
                    }
                )

                for (
                    class_index,
                    class_name,
                ) in enumerate(
                    TARGET_CLASSES
                ):
                    class_metrics = (
                        test_metrics[
                            "per_class"
                        ][
                            class_name
                        ]
                    )

                    fp32_class_row = (
                        fp32_per_class[
                            (
                                fp32_per_class[
                                    "model_name"
                                ]
                                == model_name
                            )
                            & (
                                fp32_per_class[
                                    "seed"
                                ]
                                == seed
                            )
                            & (
                                fp32_per_class[
                                    "class_name"
                                ]
                                == class_name
                            )
                        ].iloc[
                            0
                        ]
                    )

                    fp32_fnr = float(
                        fp32_class_row[
                            "false_negative_rate"
                        ]
                    )

                    pruned_fnr = float(
                        class_metrics[
                            "false_negative_rate"
                        ]
                    )

                    per_class_records.append(
                        {
                            "model_name": (
                                model_name
                            ),
                            "pruning_name": (
                                pruning_name
                            ),
                            "pruning_ratio": float(
                                pruning_ratio
                            ),
                            "seed": int(
                                seed
                            ),
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
                                class_metrics[
                                    "f1"
                                ]
                            ),
                            "fp32_false_negative_rate": (
                                fp32_fnr
                            ),
                            "false_negative_rate": (
                                pruned_fnr
                            ),
                            "false_negative_rate_delta_vs_fp32": float(
                                pruned_fnr
                                - fp32_fnr
                            ),
                        }
                    )

                confusion_matrix = np.asarray(
                    test_metrics[
                        "confusion_matrix"
                    ],
                    dtype=np.int64,
                )

                for (
                    true_index,
                    true_class,
                ) in enumerate(
                    TARGET_CLASSES
                ):
                    for (
                        predicted_index,
                        predicted_class,
                    ) in enumerate(
                        TARGET_CLASSES
                    ):
                        confusion_records.append(
                            {
                                "model_name": (
                                    model_name
                                ),
                                "pruning_name": (
                                    pruning_name
                                ),
                                "pruning_ratio": float(
                                    pruning_ratio
                                ),
                                "seed": int(
                                    seed
                                ),
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
                    "Test Macro F1       : "
                    f"{test_metrics['macro_f1']:.6f}"
                )
                print(
                    "FP32'ye göre değişim: "
                    f"{test_metrics['macro_f1'] - fp32_test_macro_f1:+.6f}"
                )
                print(
                    "Test seçime katıldı: False"
                )

                gc.collect()

    run_frame = pd.DataFrame(
        run_records
    )

    per_class_frame = pd.DataFrame(
        per_class_records
    )

    confusion_frame = pd.DataFrame(
        confusion_records
    )

    if len(
        run_frame
    ) != 45:
        raise RuntimeError(
            "Tamamlanan budama deney sayısı 45 değil."
        )

    if run_frame.duplicated(
        subset=[
            "model_name",
            "pruning_name",
            "seed",
        ]
    ).any():
        raise RuntimeError(
            "Tekrarlanan model-budama-seed "
            "sonucu bulundu."
        )

    if (
        run_frame[
            "test_evaluation_attempt_count"
        ]
        != 1
    ).any():
        raise RuntimeError(
            "Test değerlendirme girişimlerinden "
            "biri 1 değil."
        )

    if run_frame[
        "test_used_for_selection"
    ].any():
        raise RuntimeError(
            "Test sonuçlarından biri "
            "model seçimine katılmış."
        )

    aggregate_frame = (
        aggregate_global_metrics(
            run_frame
        )
    )

    per_class_aggregate_frame = (
        aggregate_per_class_metrics(
            per_class_frame
        )
    )

    run_report_file = (
        report_directory
        / "nbaiot_family3_structured_pruning_runs.csv"
    )

    aggregate_report_file = (
        report_directory
        / "nbaiot_family3_structured_pruning_aggregate.csv"
    )

    per_class_run_file = (
        report_directory
        / "nbaiot_family3_structured_pruning_per_class_runs.csv"
    )

    per_class_aggregate_file = (
        report_directory
        / "nbaiot_family3_structured_pruning_per_class_aggregate.csv"
    )

    confusion_report_file = (
        report_directory
        / "nbaiot_family3_structured_pruning_confusion_matrices.csv"
    )

    summary_file = (
        report_directory
        / "nbaiot_family3_structured_pruning_summary.json"
    )

    write_csv_atomic(
        run_frame,
        run_report_file,
    )

    write_csv_atomic(
        aggregate_frame,
        aggregate_report_file,
    )

    write_csv_atomic(
        per_class_frame,
        per_class_run_file,
    )

    write_csv_atomic(
        per_class_aggregate_frame,
        per_class_aggregate_file,
    )

    write_csv_atomic(
        confusion_frame,
        confusion_report_file,
    )

    ranking_frame = aggregate_frame[
        aggregate_frame[
            "metric"
        ]
        == "test_macro_f1"
    ].copy()

    ranking_frame = (
        ranking_frame.sort_values(
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

    ranking_records = [
        {
            "rank": int(
                index
                + 1
            ),
            "model_name": str(
                row.model_name
            ),
            "pruning_name": str(
                row.pruning_name
            ),
            "mean_test_macro_f1": float(
                row.mean
            ),
            "standard_deviation": float(
                row.standard_deviation
            ),
            "ci95_lower": float(
                row.ci95_lower
            ),
            "ci95_upper": float(
                row.ci95_upper
            ),
        }
        for index, row in enumerate(
            ranking_frame.itertuples(
                index=False
            )
        )
    ]

    summary_document = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "analysis_name": (
            "N-BaIoT family_3 physical "
            "structured pruning multi-seed"
        ),
        "protocol_file": str(
            pruning_protocol_file
        ),
        "protocol_sha256": (
            pruning_protocol_sha256
        ),
        "model_count": len(
            MODEL_NAMES
        ),
        "pruning_level_count": len(
            PRUNING_RATIOS
        ),
        "seed_count": len(
            SEEDS
        ),
        "total_run_count": len(
            run_frame
        ),
        "models": list(
            MODEL_NAMES
        ),
        "pruning_names": list(
            PRUNING_NAMES
        ),
        "seeds": list(
            SEEDS
        ),
        "physical_compaction": True,
        "mask_based_pruning": False,
        "test_evaluations_per_model_ratio_seed": 1,
        "test_used_for_model_selection": False,
        "primary_metric": (
            "test_macro_f1"
        ),
        "security_metric": (
            "per_class_false_negative_rate"
        ),
        "confidence_interval_method": (
            "Two-sided 95% Student-t confidence "
            "interval over five seeds."
        ),
        "macro_f1_ranking": (
            ranking_records
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
            "Not yet performed; paired FP32-versus-pruned "
            "tests are next."
        ),
        "validation_passed": True,
    }

    write_json_atomic(
        summary_document,
        summary_file,
    )

    print()
    print("=" * 78)
    print(
        "Yapılandırılmış Budama "
        "Ana Deneyleri Tamamlandı"
    )
    print("=" * 78)

    for record in ranking_records:
        print(
            f"{record['rank']:2d}. "
            f"{record['model_name']:14s} "
            f"{record['pruning_name']:3s} | "
            f"Macro F1="
            f"{record['mean_test_macro_f1']:.6f} "
            f"± {record['standard_deviation']:.6f} | "
            f"%95 GA=["
            f"{record['ci95_lower']:.6f}, "
            f"{record['ci95_upper']:.6f}]"
        )

    print()
    print(
        f"Tamamlanan çalışma     : {len(run_frame)}"
    )
    print(
        "Test değerlendirmesi   : 45"
    )
    print(
        "Test seçime katıldı    : False"
    )
    print(
        "Fiziksel küçültme      : True"
    )
    print(
        "Maskeli budama          : False"
    )
    print(
        "Doğrulama geçti         : True"
    )
    print()
    print(
        f"Seed sonuçları          : {run_report_file}"
    )
    print(
        f"Toplu istatistikler     : {aggregate_report_file}"
    )
    print(
        f"Sınıf sonuçları         : {per_class_run_file}"
    )
    print(
        f"Sınıf istatistikleri    : {per_class_aggregate_file}"
    )
    print(
        f"Karışıklık matrisleri   : {confusion_report_file}"
    )
    print(
        f"JSON özet               : {summary_file}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()