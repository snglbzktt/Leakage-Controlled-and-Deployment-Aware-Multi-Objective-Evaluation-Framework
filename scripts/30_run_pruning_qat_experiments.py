"""
N-BaIoT family_3 fiziksel yapılandırılmış budama + QAT ana deneyleri.

Deney matrisi:
- Modeller: tinyml_mlp, compact_dnn, tiny_1d_cnn
- Budama düzeyleri: P25, P50, P75
- Seedler: 42, 123, 2026, 3407, 8192
- Toplam: 3 x 3 x 5 = 45 bağımsız sonuç

Bilimsel protokol:
- Her P25-QAT, P50-QAT ve P75-QAT çalışması, aynı model ve seede ait
  validation-seçilmiş FP32 checkpointten bağımsız biçimde üretilir.
- Budama düzeyleri birbirinden türetilmez.
- Fiziksel yapılandırılmış budama uygulanır; maske tabanlı budama yoktur.
- QAT yalnızca train splitinde güncellenir.
- Model seçimi, her epoch sonunda dönüştürülen gerçek INT8 modelin
  validation Macro F1 değeriyle yapılır.
- Test, seçilmiş INT8 model üzerinde yalnızca bir kez değerlendirilir.
- Test verisi budama, eğitim, erken durdurma veya hiperparametre seçiminde
  kullanılmaz.

Uygulama notu:
- QAT yardımcıları, başarıyla doğrulanmış
  scripts/29b_run_quantization_experiments.py modülünden yeniden kullanılır.
- Quantized backend: oneDNN.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
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
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
from src.training.baseline_trainer import run_epoch  # noqa: E402


# ==========================================================
# LOAD VALIDATED QUANTIZATION HELPERS
# ==========================================================

QBASE_FILE = (
    PROJECT_ROOT
    / "scripts"
    / "29b_run_quantization_experiments.py"
)

if not QBASE_FILE.exists():
    raise FileNotFoundError(
        "29B quantization yardımcı betiği bulunamadı: "
        f"{QBASE_FILE}"
    )

_qbase_spec = importlib.util.spec_from_file_location(
    "nbaiot_quantization_stage29b",
    QBASE_FILE,
)

if _qbase_spec is None or _qbase_spec.loader is None:
    raise ImportError(
        "29B quantization yardımcı modülü yüklenemedi."
    )

qbase = importlib.util.module_from_spec(_qbase_spec)
_qbase_spec.loader.exec_module(qbase)


# ==========================================================
# CONSTANTS
# ==========================================================

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

DEFAULT_QAT_VALIDATION_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "nbaiot_fx_qat_pipeline_validation_summary.json"
)

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

DEFAULT_FP32_EXPERIMENT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "fp32_baseline_v1"
)

DEFAULT_PROTOCOL_FILE = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_pruning_qat_protocol_v1.json"
)

DEFAULT_EXPERIMENT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "pruning_qat_v1"
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

VARIANT_NAMES = tuple(
    f"{name}-QAT"
    for name in PRUNING_NAMES
)

TARGET_CLASSES = (
    "benign",
    "gafgyt",
    "mirai",
)

AGGREGATE_METRICS = (
    "test_loss",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
    "test_mcc",
    "macro_f1_delta_vs_fp32",
    "macro_f1_delta_vs_pruning_only",
)


# ==========================================================
# ARGUMENTS
# ==========================================================


def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT family_3 için 45 fiziksel budama + "
            "FX-QAT ana deneyini çalıştırır."
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
        "--qat-validation-file",
        type=Path,
        default=DEFAULT_QAT_VALIDATION_FILE,
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
        "--fp32-experiment-directory",
        type=Path,
        default=DEFAULT_FP32_EXPERIMENT_DIRECTORY,
    )
    parser.add_argument(
        "--protocol-file",
        type=Path,
        default=DEFAULT_PROTOCOL_FILE,
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
        "--backend",
        type=str,
        default="onednn",
    )
    parser.add_argument(
        "--qat-learning-rate-factor",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--qat-max-epochs",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--qat-patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--qat-min-delta",
        type=float,
        default=0.0002,
    )
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
    )

    return parser.parse_args()


# ==========================================================
# VALIDATION AND PROTOCOL
# ==========================================================


def validate_arguments(args: argparse.Namespace) -> None:
    """Kullanıcı parametrelerini doğrular."""

    if args.qat_learning_rate_factor <= 0:
        raise ValueError(
            "qat-learning-rate-factor pozitif olmalıdır."
        )

    if args.qat_max_epochs <= 0:
        raise ValueError(
            "qat-max-epochs pozitif olmalıdır."
        )

    if args.qat_patience <= 0:
        raise ValueError(
            "qat-patience pozitif olmalıdır."
        )

    if args.qat_patience > args.qat_max_epochs:
        raise ValueError(
            "qat-patience, qat-max-epochs değerinden büyük olamaz."
        )

    if args.qat_min_delta < 0:
        raise ValueError(
            "qat-min-delta negatif olamaz."
        )


def validate_pruning_engine(
    engine: dict[str, Any],
) -> None:
    """25. aşama budama motoru manifestini doğrular."""

    if not bool(engine.get("validation_passed")):
        raise ValueError(
            "Yapılandırılmış budama motoru doğrulanmamış."
        )

    if not bool(engine.get("physical_compaction")):
        raise ValueError(
            "Budama motoru fiziksel küçültme uygulamıyor."
        )

    if bool(engine.get("mask_based_pruning")):
        raise ValueError(
            "Budama motoru maskeli budama olarak işaretlenmiş."
        )

    if tuple(engine.get("supported_models", [])) != MODEL_NAMES:
        raise ValueError(
            "Budama motoru model sırası uyuşmuyor."
        )

    engine_ratios = tuple(
        float(value)
        for value in engine.get(
            "supported_pruning_ratios",
            [],
        )
    )

    if engine_ratios != PRUNING_RATIOS:
        raise ValueError(
            "Budama motoru oranları P25/P50/P75 ile uyuşmuyor."
        )


def create_or_validate_protocol(
    *,
    protocol_file: Path,
    fp32_protocol_file: Path,
    fp32_protocol: dict[str, Any],
    pruning_engine_file: Path,
    pruning_engine: dict[str, Any],
    qat_validation_file: Path,
    qat_validation: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    """Stage-30 protokolünü oluşturur veya mevcut protokolü doğrular."""

    common = fp32_protocol[
        "common_training_configuration"
    ]

    expected = {
        "protocol_name": (
            "N-BaIoT family_3 physical structured pruning plus "
            "FX QAT multi-seed"
        ),
        "protocol_version": "1.0",
        "status": "locked_before_pruning_qat_main_experiments",
        "task": "family_3",
        "models": list(MODEL_NAMES),
        "training_seeds": list(SEEDS),
        "pruning_ratios": list(PRUNING_RATIOS),
        "pruning_names": list(PRUNING_NAMES),
        "variant_names": list(VARIANT_NAMES),
        "planned_run_count": 45,
        "source_policy": (
            "Every pruning-QAT variant is produced independently "
            "from the matching validation-selected FP32 model-seed "
            "checkpoint. Pruning levels are never chained."
        ),
        "structured_pruning": {
            "importance_method": pruning_engine[
                "importance_method"
            ],
            "physical_compaction": True,
            "mask_based_pruning": False,
            "output_class_units_pruned": False,
        },
        "quantization_aware_training": {
            "framework": "PyTorch FX Graph Mode QAT",
            "backend": str(args.backend),
            "optimizer": "AdamW",
            "learning_rate_rule": (
                "fp32_learning_rate_times_factor"
            ),
            "learning_rate_factor": float(
                args.qat_learning_rate_factor
            ),
            "max_epochs": int(args.qat_max_epochs),
            "early_stopping_patience": int(
                args.qat_patience
            ),
            "early_stopping_min_delta": float(
                args.qat_min_delta
            ),
            "batch_size": int(common["batch_size"]),
            "weight_decay": float(common["weight_decay"]),
            "gradient_clip_norm": float(
                common["gradient_clip_norm"]
            ),
            "torch_threads": int(common["torch_threads"]),
            "num_workers": int(common["num_workers"]),
            "training_split": "train",
            "selection_split": "validation",
            "selection_metric": "converted_int8_macro_f1",
            "test_used_for_selection": False,
        },
        "test_evaluation": {
            "evaluations_per_model_pruning_seed": 1,
            "test_used_for_training": False,
            "test_used_for_pruning": False,
            "test_used_for_model_selection": False,
        },
        "source_artifacts": {
            "fp32_protocol": {
                "path": str(fp32_protocol_file),
                "sha256": qbase.calculate_sha256(
                    fp32_protocol_file
                ),
            },
            "pruning_engine": {
                "path": str(pruning_engine_file),
                "sha256": qbase.calculate_sha256(
                    pruning_engine_file
                ),
            },
            "qat_validation": {
                "path": str(qat_validation_file),
                "sha256": qbase.calculate_sha256(
                    qat_validation_file
                ),
            },
            "stage29b_helper": {
                "path": str(QBASE_FILE),
                "sha256": qbase.calculate_sha256(
                    QBASE_FILE
                ),
            },
        },
    }

    if protocol_file.exists():
        protocol = qbase.read_json(
            protocol_file
        )

        for key, expected_value in expected.items():
            if protocol.get(key) != expected_value:
                raise RuntimeError(
                    "Mevcut pruning-QAT protokolü beklenen "
                    f"ayarlarla uyuşmuyor: {key}"
                )

    else:
        protocol = {
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            **expected,
        }

        qbase.write_json_atomic(
            protocol,
            protocol_file,
        )

    return (
        protocol,
        qbase.calculate_sha256(
            protocol_file
        ),
    )


# ==========================================================
# REFERENCE RESULTS
# ==========================================================


def load_reference_runs(
    *,
    fp32_run_file: Path,
    pruning_run_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FP32 ve pruning-only eşleşen test sonuçlarını yükler."""

    fp32_frame = qbase.load_fp32_references(
        fp32_run_file
    )

    if not pruning_run_file.exists():
        raise FileNotFoundError(
            "Budama sonuç dosyası bulunamadı: "
            f"{pruning_run_file}"
        )

    pruning_frame = pd.read_csv(
        pruning_run_file
    )

    required_columns = {
        "model_name",
        "pruning_name",
        "seed",
        "test_macro_f1",
        "test_mcc",
    }

    missing_columns = (
        required_columns
        - set(pruning_frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Budama referans sonuçlarında eksik sütunlar var: "
            + ", ".join(sorted(missing_columns))
        )

    pruning_frame = pruning_frame.copy()
    pruning_frame["model_name"] = (
        pruning_frame["model_name"].astype(str)
    )
    pruning_frame["pruning_name"] = (
        pruning_frame["pruning_name"].astype(str)
    )
    pruning_frame["seed"] = pd.to_numeric(
        pruning_frame["seed"],
        errors="raise",
    ).astype("int64")
    pruning_frame["test_macro_f1"] = pd.to_numeric(
        pruning_frame["test_macro_f1"],
        errors="raise",
    )
    pruning_frame["test_mcc"] = pd.to_numeric(
        pruning_frame["test_mcc"],
        errors="raise",
    )

    if len(pruning_frame) != 45:
        raise ValueError(
            "Budama referans sonucu sayısı 45 değil."
        )

    if pruning_frame.duplicated(
        subset=[
            "model_name",
            "pruning_name",
            "seed",
        ]
    ).any():
        raise ValueError(
            "Budama referans sonuçlarında tekrarlanan kayıt var."
        )

    return fp32_frame, pruning_frame


# ==========================================================
# COMPACT QAT HELPERS
# ==========================================================


def create_compact_model_from_state(
    *,
    pruning_metadata: dict[str, Any],
    compact_state_dict: dict[str, Any],
) -> nn.Module:
    """Pruning metadata ve state_dict ile kompakt FP32 model oluşturur."""

    model = recreate_compact_model(
        pruning_metadata
    )

    model.load_state_dict(
        compact_state_dict,
        strict=True,
    )

    return model.cpu()



def create_compact_prepared_snapshot(
    *,
    pruning_metadata: dict[str, Any],
    compact_source_state_dict: dict[str, Any],
    prepared_state_dict: dict[str, Any],
    example_inputs: tuple[torch.Tensor, ...],
    backend_name: str,
    backend_config: Any | None,
) -> nn.Module:
    """Kompakt QAT prepared model snapshotı oluşturur."""

    source_model = create_compact_model_from_state(
        pruning_metadata=pruning_metadata,
        compact_state_dict=compact_source_state_dict,
    )

    prepared_model = qbase.prepare_qat_model(
        model=source_model,
        example_inputs=example_inputs,
        backend_name=backend_name,
        backend_config=backend_config,
    )

    prepared_model.load_state_dict(
        prepared_state_dict,
        strict=True,
    )

    return prepared_model



def get_example_inputs(
    *,
    assets: Any,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    """FX tracing için train splitinden örnek girdi alır."""

    _, loader = create_nbaiot_dataloader(
        assets=assets,
        split_name="train",
        batch_size=8,
        shuffle=False,
        seed=seed,
        epoch=0,
        num_workers=0,
        pin_memory=False,
    )

    features, _ = next(iter(loader))

    return (
        features.cpu().contiguous(),
    )


# ==========================================================
# TRAINING
# ==========================================================


def train_pruning_qat_model(
    *,
    model_name: str,
    seed: int,
    pruning_name: str,
    pruning_ratio: float,
    pruning_metadata: dict[str, Any],
    compact_source_state_dict: dict[str, Any],
    source_checkpoint_file: Path,
    assets: Any,
    fp32_learning_rate: float,
    protocol: dict[str, Any],
    protocol_sha256: str,
    backend_config: Any | None,
    backend_config_source: str | None,
    run_directory: Path,
) -> dict[str, Any]:
    """Fiziksel budanmış modeli FX-QAT ile eğitir."""

    qat_config = protocol[
        "quantization_aware_training"
    ]

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_summary_file = (
        run_directory
        / "training_summary.json"
    )
    history_file = (
        run_directory
        / "training_history.csv"
    )
    best_prepared_state_file = (
        run_directory
        / "best_prepared_qat_state_dict.pt"
    )

    if training_summary_file.exists():
        return qbase.read_json(
            training_summary_file
        )

    qbase.set_reproducible_environment(
        seed=seed,
        torch_threads=int(
            qat_config["torch_threads"]
        ),
    )

    torch.backends.quantized.engine = str(
        qat_config["backend"]
    )

    example_inputs = get_example_inputs(
        assets=assets,
        seed=seed,
    )

    compact_source_model = (
        create_compact_model_from_state(
            pruning_metadata=pruning_metadata,
            compact_state_dict=(
                compact_source_state_dict
            ),
        )
    )

    prepared_model = qbase.prepare_qat_model(
        model=compact_source_model,
        example_inputs=example_inputs,
        backend_name=str(
            qat_config["backend"]
        ),
        backend_config=backend_config,
    )

    fake_quant_count = (
        qbase.count_fake_quant_modules(
            prepared_model
        )
    )

    if fake_quant_count <= 0:
        raise RuntimeError(
            f"{model_name}/{pruning_name}/seed{seed}: "
            "QAT modeli fake-quant modülü içermiyor."
        )

    class_weights = torch.as_tensor(
        assets.class_weights,
        dtype=torch.float32,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    qat_learning_rate = float(
        fp32_learning_rate
        * float(
            qat_config["learning_rate_factor"]
        )
    )

    optimizer = torch.optim.AdamW(
        prepared_model.parameters(),
        lr=qat_learning_rate,
        weight_decay=float(
            qat_config["weight_decay"]
        ),
    )

    best_validation_macro_f1 = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_records: list[dict[str, Any]] = []

    training_start = time.perf_counter()

    for epoch in range(
        1,
        int(qat_config["max_epochs"]) + 1,
    ):
        _, train_loader = create_nbaiot_dataloader(
            assets=assets,
            split_name="train",
            batch_size=int(
                qat_config["batch_size"]
            ),
            shuffle=True,
            seed=seed,
            epoch=epoch,
            num_workers=int(
                qat_config["num_workers"]
            ),
            pin_memory=False,
        )

        train_metrics = run_epoch(
            model=prepared_model,
            data_loader=train_loader,
            criterion=criterion,
            class_names=assets.target_classes,
            optimizer=optimizer,
            gradient_clip_norm=float(
                qat_config[
                    "gradient_clip_norm"
                ]
            ),
            maximum_batches=None,
        )

        prepared_snapshot = (
            create_compact_prepared_snapshot(
                pruning_metadata=pruning_metadata,
                compact_source_state_dict=(
                    compact_source_state_dict
                ),
                prepared_state_dict=(
                    prepared_model.state_dict()
                ),
                example_inputs=example_inputs,
                backend_name=str(
                    qat_config["backend"]
                ),
                backend_config=backend_config,
            )
        )

        converted_validation_model = (
            qbase.convert_qat_model(
                prepared_model=prepared_snapshot,
                backend_config=backend_config,
            )
        )

        (
            quantized_module_count,
            _,
        ) = qbase.count_quantized_modules(
            converted_validation_model
        )

        (
            quantized_node_count,
            _,
        ) = qbase.count_quantized_graph_nodes(
            converted_validation_model
        )

        if (
            quantized_module_count
            + quantized_node_count
            <= 0
        ):
            raise RuntimeError(
                f"{model_name}/{pruning_name}/seed{seed}: "
                "QAT validation modeli quantized kapsam içermiyor."
            )

        _, validation_loader = (
            create_nbaiot_dataloader(
                assets=assets,
                split_name="validation",
                batch_size=int(
                    qat_config["batch_size"]
                ),
                shuffle=False,
                seed=seed,
                epoch=0,
                num_workers=int(
                    qat_config["num_workers"]
                ),
                pin_memory=False,
            )
        )

        validation_metrics = run_epoch(
            model=converted_validation_model,
            data_loader=validation_loader,
            criterion=criterion,
            class_names=assets.target_classes,
            optimizer=None,
            gradient_clip_norm=0.0,
            maximum_batches=None,
        )

        validation_macro_f1 = float(
            validation_metrics["macro_f1"]
        )

        improved = bool(
            validation_macro_f1
            > best_validation_macro_f1
            + float(
                qat_config[
                    "early_stopping_min_delta"
                ]
            )
        )

        if improved:
            best_validation_macro_f1 = (
                validation_macro_f1
            )
            best_epoch = epoch
            epochs_without_improvement = 0

            qbase.save_torch_atomic(
                prepared_model.state_dict(),
                best_prepared_state_file,
            )

        else:
            epochs_without_improvement += 1

        history_records.append(
            {
                "epoch": int(epoch),
                "train_loss": float(
                    train_metrics["loss"]
                ),
                "train_accuracy": float(
                    train_metrics["accuracy"]
                ),
                "train_macro_f1": float(
                    train_metrics["macro_f1"]
                ),
                "train_mcc": float(
                    train_metrics[
                        "matthews_correlation_coefficient"
                    ]
                ),
                "validation_loss_converted_int8": float(
                    validation_metrics["loss"]
                ),
                "validation_accuracy_converted_int8": float(
                    validation_metrics["accuracy"]
                ),
                "validation_macro_f1_converted_int8": (
                    validation_macro_f1
                ),
                "validation_mcc_converted_int8": float(
                    validation_metrics[
                        "matthews_correlation_coefficient"
                    ]
                ),
                "improved": improved,
                "epochs_without_improvement": int(
                    epochs_without_improvement
                ),
            }
        )

        print(
            f"    epoch={epoch:02d} | "
            f"train_macro_f1="
            f"{train_metrics['macro_f1']:.6f} | "
            f"val_int8_macro_f1="
            f"{validation_macro_f1:.6f} | "
            f"best="
            f"{best_validation_macro_f1:.6f} | "
            f"improved={improved}"
        )

        if (
            epochs_without_improvement
            >= int(
                qat_config[
                    "early_stopping_patience"
                ]
            )
        ):
            break

        del prepared_snapshot
        del converted_validation_model
        gc.collect()

    total_training_seconds = float(
        time.perf_counter()
        - training_start
    )

    if (
        best_epoch <= 0
        or not best_prepared_state_file.exists()
    ):
        raise RuntimeError(
            f"{model_name}/{pruning_name}/seed{seed}: "
            "QAT en iyi prepared state oluşturmadı."
        )

    qbase.write_csv_atomic(
        pd.DataFrame(history_records),
        history_file,
    )

    summary = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_name": model_name,
        "pruning_name": pruning_name,
        "variant_name": f"{pruning_name}-QAT",
        "pruning_ratio": float(pruning_ratio),
        "seed": int(seed),
        "source_checkpoint_file": str(
            source_checkpoint_file
        ),
        "source_checkpoint_sha256": (
            qbase.calculate_sha256(
                source_checkpoint_file
            )
        ),
        "pruning_qat_protocol_sha256": (
            protocol_sha256
        ),
        "backend": str(
            qat_config["backend"]
        ),
        "backend_config_source": (
            backend_config_source
        ),
        "fp32_learning_rate": float(
            fp32_learning_rate
        ),
        "qat_learning_rate": float(
            qat_learning_rate
        ),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(
            len(history_records)
        ),
        "best_validation_macro_f1_converted_int8": float(
            best_validation_macro_f1
        ),
        "fake_quant_module_count": int(
            fake_quant_count
        ),
        "total_training_seconds": (
            total_training_seconds
        ),
        "best_prepared_state_file": str(
            best_prepared_state_file
        ),
        "best_prepared_state_sha256": (
            qbase.calculate_sha256(
                best_prepared_state_file
            )
        ),
        "history_file": str(history_file),
        "pruning_metadata": pruning_metadata,
        "physical_compaction": True,
        "mask_based_pruning": False,
        "training_split": "train",
        "selection_split": "validation",
        "selection_metric": (
            "converted_int8_macro_f1"
        ),
        "test_split_evaluated": False,
        "training_completed": True,
    }

    qbase.write_json_atomic(
        summary,
        training_summary_file,
    )

    return summary


# ==========================================================
# TEST EVALUATION
# ==========================================================


def evaluate_pruning_qat_test_once(
    *,
    model_name: str,
    seed: int,
    pruning_name: str,
    pruning_metadata: dict[str, Any],
    compact_source_state_dict: dict[str, Any],
    assets: Any,
    protocol: dict[str, Any],
    backend_config: Any | None,
    training_summary: dict[str, Any],
    run_directory: Path,
    restart_incomplete: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Seçilmiş compact INT8 modeli testte yalnızca bir kez değerlendirir."""

    qat_config = protocol[
        "quantization_aware_training"
    ]

    test_metrics_file = (
        run_directory
        / "test_metrics.json"
    )
    artifact_metadata_file = (
        run_directory
        / "artifact_metadata.json"
    )
    converted_state_file = (
        run_directory
        / "converted_int8_state_dict.pt"
    )

    if (
        test_metrics_file.exists()
        and artifact_metadata_file.exists()
    ):
        metrics_document = qbase.read_json(
            test_metrics_file
        )

        metrics = {
            key: value
            for key, value in metrics_document.items()
            if key
            not in {
                "generated_at_utc",
                "model_name",
                "method_name",
                "seed",
                "task_name",
                "test_evaluation_attempt_count",
                "test_used_for_selection",
            }
        }

        return (
            metrics,
            qbase.read_json(
                artifact_metadata_file
            ),
        )

    example_inputs = get_example_inputs(
        assets=assets,
        seed=seed,
    )

    prepared_state = qbase.load_torch_checkpoint(
        Path(
            training_summary[
                "best_prepared_state_file"
            ]
        )
    )

    prepared_model = (
        create_compact_prepared_snapshot(
            pruning_metadata=pruning_metadata,
            compact_source_state_dict=(
                compact_source_state_dict
            ),
            prepared_state_dict=prepared_state,
            example_inputs=example_inputs,
            backend_name=str(
                qat_config["backend"]
            ),
            backend_config=backend_config,
        )
    )

    converted_model = qbase.convert_qat_model(
        prepared_model=prepared_model,
        backend_config=backend_config,
    )

    (
        quantized_module_count,
        quantized_module_types,
    ) = qbase.count_quantized_modules(
        converted_model
    )

    (
        quantized_node_count,
        quantized_node_targets,
    ) = qbase.count_quantized_graph_nodes(
        converted_model
    )

    if (
        quantized_module_count
        + quantized_node_count
        <= 0
    ):
        raise RuntimeError(
            f"{model_name}/{pruning_name}/seed{seed}: "
            "QAT test modeli quantized kapsam içermiyor."
        )

    qbase.save_torch_atomic(
        converted_model.state_dict(),
        converted_state_file,
    )

    graph_file = (
        run_directory
        / "converted_fx_graph.txt"
    )

    graph_file.write_text(
        str(converted_model.graph),
        encoding="utf-8",
    )

    marker_file = qbase.begin_test_manifest(
        run_directory=run_directory,
        model_name=model_name,
        method_name=f"{pruning_name}-QAT",
        seed=seed,
        artifact_file=converted_state_file,
        restart_incomplete=restart_incomplete,
    )

    class_weights = torch.as_tensor(
        assets.class_weights,
        dtype=torch.float32,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    _, test_loader = create_nbaiot_dataloader(
        assets=assets,
        split_name="test",
        batch_size=int(
            qat_config["batch_size"]
        ),
        shuffle=False,
        seed=seed,
        epoch=0,
        num_workers=int(
            qat_config["num_workers"]
        ),
        pin_memory=False,
    )

    metrics = run_epoch(
        model=converted_model,
        data_loader=test_loader,
        criterion=criterion,
        class_names=assets.target_classes,
        optimizer=None,
        gradient_clip_norm=0.0,
        maximum_batches=None,
    )

    (
        written_metrics_file,
        per_class_file,
        confusion_file,
    ) = qbase.write_test_artifacts(
        metrics=metrics,
        run_directory=run_directory,
        method_name=f"{pruning_name}-QAT",
        model_name=model_name,
        seed=seed,
    )

    qbase.complete_test_manifest(
        marker_file=marker_file,
        test_metrics_file=written_metrics_file,
        per_class_file=per_class_file,
        confusion_file=confusion_file,
    )

    artifact_metadata = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "method_name": f"{pruning_name}-QAT",
        "artifact_file": str(
            converted_state_file
        ),
        "artifact_sha256": (
            qbase.calculate_sha256(
                converted_state_file
            )
        ),
        "artifact_size_bytes": int(
            converted_state_file.stat().st_size
        ),
        "prepared_state_file": str(
            training_summary[
                "best_prepared_state_file"
            ]
        ),
        "prepared_state_sha256": (
            qbase.calculate_sha256(
                Path(
                    training_summary[
                        "best_prepared_state_file"
                    ]
                )
            )
        ),
        "quantized_module_count": int(
            quantized_module_count
        ),
        "quantized_module_types": (
            quantized_module_types
        ),
        "quantized_graph_node_count": int(
            quantized_node_count
        ),
        "quantized_graph_targets": (
            quantized_node_targets
        ),
        "graph_file": str(graph_file),
        "pruning_metadata": pruning_metadata,
        "physical_compaction": True,
        "mask_based_pruning": False,
        "native_artifact_note": (
            "PyTorch quantized state_dict size is runtime-specific."
        ),
    }

    qbase.write_json_atomic(
        artifact_metadata,
        artifact_metadata_file,
    )

    return metrics, artifact_metadata


# ==========================================================
# AGGREGATION
# ==========================================================


def aggregate_results(
    run_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Model ve birleşik varyant bazında metrikleri özetler."""

    records: list[dict[str, Any]] = []

    for model_name in MODEL_NAMES:
        for variant_name in VARIANT_NAMES:
            subset = run_frame[
                (
                    run_frame["model_name"]
                    == model_name
                )
                & (
                    run_frame["variant_name"]
                    == variant_name
                )
            ]

            for metric_name in AGGREGATE_METRICS:
                statistics = (
                    qbase.calculate_summary_statistics(
                        subset[metric_name].to_numpy(
                            dtype=np.float64
                        )
                    )
                )

                records.append(
                    {
                        "model_name": model_name,
                        "variant_name": variant_name,
                        "metric": metric_name,
                        **statistics,
                    }
                )

    return pd.DataFrame(records)



def aggregate_per_class(
    per_class_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Model, birleşik varyant ve sınıf bazında sonuçları özetler."""

    records: list[dict[str, Any]] = []

    for model_name in MODEL_NAMES:
        for variant_name in VARIANT_NAMES:
            for class_name in TARGET_CLASSES:
                subset = per_class_frame[
                    (
                        per_class_frame["model_name"]
                        == model_name
                    )
                    & (
                        per_class_frame["variant_name"]
                        == variant_name
                    )
                    & (
                        per_class_frame["class_name"]
                        == class_name
                    )
                ]

                for metric_name in (
                    "precision",
                    "recall",
                    "f1",
                    "false_negative_rate",
                ):
                    statistics = (
                        qbase.calculate_summary_statistics(
                            subset[metric_name].to_numpy(
                                dtype=np.float64
                            )
                        )
                    )

                    records.append(
                        {
                            "model_name": model_name,
                            "variant_name": variant_name,
                            "class_name": class_name,
                            "metric": metric_name,
                            **statistics,
                        }
                    )

    return pd.DataFrame(records)


# ==========================================================
# MAIN
# ==========================================================


def main() -> None:
    """45 bağımsız pruning-QAT çalışmasını yürütür."""

    args = parse_arguments()
    validate_arguments(args)

    fp32_protocol_file = (
        args.fp32_protocol_file.resolve()
    )
    pruning_engine_file = (
        args.pruning_engine_file.resolve()
    )
    qat_validation_file = (
        args.qat_validation_file.resolve()
    )
    fp32_run_file = (
        args.fp32_run_file.resolve()
    )
    pruning_run_file = (
        args.pruning_run_file.resolve()
    )
    fp32_experiment_directory = (
        args.fp32_experiment_directory.resolve()
    )
    protocol_file = (
        args.protocol_file.resolve()
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
    protocol_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fp32_protocol = qbase.read_json(
        fp32_protocol_file
    )
    qbase.validate_fp32_protocol(
        fp32_protocol
    )

    pruning_engine = qbase.read_json(
        pruning_engine_file
    )
    validate_pruning_engine(
        pruning_engine
    )

    qat_validation = qbase.read_json(
        qat_validation_file
    )
    qbase.validate_qat_validation(
        qat_validation,
        args.backend,
    )

    supported_engines = tuple(
        torch.backends.quantized.supported_engines
    )

    if args.backend not in supported_engines:
        raise RuntimeError(
            f"{args.backend} quantized engine desteklenmiyor. "
            f"Desteklenenler={supported_engines}"
        )

    torch.backends.quantized.engine = (
        args.backend
    )

    (
        protocol,
        protocol_sha256,
    ) = create_or_validate_protocol(
        protocol_file=protocol_file,
        fp32_protocol_file=fp32_protocol_file,
        fp32_protocol=fp32_protocol,
        pruning_engine_file=pruning_engine_file,
        pruning_engine=pruning_engine,
        qat_validation_file=qat_validation_file,
        qat_validation=qat_validation,
        args=args,
    )

    (
        fp32_references,
        pruning_references,
    ) = load_reference_runs(
        fp32_run_file=fp32_run_file,
        pruning_run_file=pruning_run_file,
    )

    assets = load_pipeline_assets(
        task_name="family_3",
        weight_scheme=(
            "inverse_square_root_frequency_mean1"
        ),
        expected_seed=2026,
    )

    if assets.feature_count != 115:
        raise RuntimeError(
            "Beklenen özellik sayısı 115 değil."
        )

    if tuple(assets.target_classes) != TARGET_CLASSES:
        raise RuntimeError(
            "family_3 sınıf sırası uyuşmuyor."
        )

    (
        backend_config,
        backend_config_source,
    ) = qbase.resolve_backend_config(
        args.backend
    )

    run_records: list[dict[str, Any]] = []
    per_class_records: list[dict[str, Any]] = []
    confusion_records: list[dict[str, Any]] = []

    total_run_count = (
        len(MODEL_NAMES)
        * len(PRUNING_RATIOS)
        * len(SEEDS)
    )

    print("=" * 78)
    print(
        "N-BaIoT Family-3 Fiziksel Budama + QAT Ana Deneyleri"
    )
    print("=" * 78)
    print(
        "Modeller        : "
        + ", ".join(MODEL_NAMES)
    )
    print(
        "Varyantlar      : "
        + ", ".join(VARIANT_NAMES)
    )
    print(
        "Seedler         : "
        + ", ".join(str(seed) for seed in SEEDS)
    )
    print(
        f"Toplam çalışma  : {total_run_count}"
    )
    print(
        f"Quantized engine: {args.backend}"
    )
    print(
        "Backend config  : "
        f"{backend_config_source or 'qconfig_mapping_only'}"
    )
    print(
        "QAT LR          : FP32 öğrenme oranının "
        f"{args.qat_learning_rate_factor:g} katı"
    )
    print(
        f"Maksimum epoch  : {args.qat_max_epochs}"
    )
    print(
        f"Patience        : {args.qat_patience}"
    )
    print(
        "Model seçimi    : converted INT8 validation Macro F1"
    )
    print(
        "Test politikası : seçilen INT8 modelde bir kez"
    )
    print(
        f"Protokol SHA-256: {protocol_sha256}"
    )
    print("=" * 78)

    run_index = 0

    for model_name in MODEL_NAMES:
        model_configuration = (
            fp32_protocol[
                "model_specific_configuration"
            ][model_name]
        )

        fp32_learning_rate = float(
            model_configuration[
                "learning_rate"
            ]
        )

        for (
            pruning_ratio,
            pruning_name,
            variant_name,
        ) in zip(
            PRUNING_RATIOS,
            PRUNING_NAMES,
            VARIANT_NAMES,
            strict=True,
        ):
            for seed in SEEDS:
                run_index += 1

                run_directory = (
                    experiment_directory
                    / model_name
                    / variant_name
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

                source_checkpoint_file = (
                    fp32_experiment_directory
                    / model_name
                    / f"seed{seed}"
                    / "best_checkpoint.pt"
                )

                source_checkpoint_sha256 = (
                    qbase.calculate_sha256(
                        source_checkpoint_file
                    )
                )

                print()
                print("=" * 78)
                print(
                    f"Çalışma {run_index}/{total_run_count}"
                )
                print(f"Model   : {model_name}")
                print(f"Varyant : {variant_name}")
                print(f"Seed    : {seed}")
                print("=" * 78)

                if final_summary_file.exists():
                    final_summary = qbase.read_json(
                        final_summary_file
                    )

                    if (
                        final_summary[
                            "pruning_qat_protocol_sha256"
                        ]
                        != protocol_sha256
                    ):
                        raise RuntimeError(
                            "Tamamlanmış çalışma farklı "
                            "pruning-QAT protokolüne ait."
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

                    training_summary = qbase.read_json(
                        Path(
                            final_summary[
                                "training_summary_file"
                            ]
                        )
                    )

                    test_document = qbase.read_json(
                        Path(
                            final_summary[
                                "test_metrics_file"
                            ]
                        )
                    )

                    test_metrics = {
                        key: value
                        for key, value
                        in test_document.items()
                        if key
                        not in {
                            "generated_at_utc",
                            "model_name",
                            "method_name",
                            "seed",
                            "task_name",
                            "test_evaluation_attempt_count",
                            "test_used_for_selection",
                        }
                    }

                    artifact_metadata = qbase.read_json(
                        Path(
                            final_summary[
                                "artifact_metadata_file"
                            ]
                        )
                    )

                    pruning_metadata = training_summary[
                        "pruning_metadata"
                    ]

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
                                "Yarım kalmış pruning-QAT klasörü "
                                f"bulundu: {run_directory}\n"
                                "Yeniden başlatmak için "
                                "--restart-incomplete kullan."
                            )

                    run_directory.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    source_checkpoint = (
                        qbase.load_torch_checkpoint(
                            source_checkpoint_file
                        )
                    )

                    qbase.validate_source_checkpoint(
                        checkpoint=source_checkpoint,
                        model_name=model_name,
                        seed=seed,
                        class_names=assets.target_classes,
                    )

                    original_model = create_model(
                        model_name=model_name,
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
                        pruning_metadata_object,
                    ) = physically_prune_model(
                        model_name=model_name,
                        original_model=original_model,
                        pruning_ratio=pruning_ratio,
                    )

                    pruning_metadata = (
                        pruning_metadata_object.to_dict()
                    )

                    compact_source_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value
                        in compact_model.state_dict().items()
                    }

                    if training_summary_file.exists():
                        training_summary = qbase.read_json(
                            training_summary_file
                        )

                        if (
                            training_summary[
                                "source_checkpoint_sha256"
                            ]
                            != source_checkpoint_sha256
                        ):
                            raise RuntimeError(
                                "Mevcut training_summary farklı "
                                "FP32 checkpointe ait."
                            )

                        if (
                            training_summary[
                                "pruning_qat_protocol_sha256"
                            ]
                            != protocol_sha256
                        ):
                            raise RuntimeError(
                                "Mevcut training_summary farklı "
                                "pruning-QAT protokolüne ait."
                            )

                        print(
                            "QAT eğitimi tamamlanmış; "
                            "test aşamasına devam ediliyor."
                        )

                    else:
                        training_summary = (
                            train_pruning_qat_model(
                                model_name=model_name,
                                seed=seed,
                                pruning_name=pruning_name,
                                pruning_ratio=pruning_ratio,
                                pruning_metadata=(
                                    pruning_metadata
                                ),
                                compact_source_state_dict=(
                                    compact_source_state_dict
                                ),
                                source_checkpoint_file=(
                                    source_checkpoint_file
                                ),
                                assets=assets,
                                fp32_learning_rate=(
                                    fp32_learning_rate
                                ),
                                protocol=protocol,
                                protocol_sha256=(
                                    protocol_sha256
                                ),
                                backend_config=backend_config,
                                backend_config_source=(
                                    backend_config_source
                                ),
                                run_directory=run_directory,
                            )
                        )

                    (
                        test_metrics,
                        artifact_metadata,
                    ) = evaluate_pruning_qat_test_once(
                        model_name=model_name,
                        seed=seed,
                        pruning_name=pruning_name,
                        pruning_metadata=pruning_metadata,
                        compact_source_state_dict=(
                            compact_source_state_dict
                        ),
                        assets=assets,
                        protocol=protocol,
                        backend_config=backend_config,
                        training_summary=training_summary,
                        run_directory=run_directory,
                        restart_incomplete=(
                            args.restart_incomplete
                        ),
                    )

                    final_summary = {
                        "generated_at_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "model_name": model_name,
                        "pruning_name": pruning_name,
                        "variant_name": variant_name,
                        "pruning_ratio": float(
                            pruning_ratio
                        ),
                        "seed": int(seed),
                        "pruning_qat_protocol_file": str(
                            protocol_file
                        ),
                        "pruning_qat_protocol_sha256": (
                            protocol_sha256
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
                        "artifact_metadata_file": str(
                            run_directory
                            / "artifact_metadata.json"
                        ),
                        "test_evaluation_attempt_count": 1,
                        "test_used_for_model_selection": False,
                        "physical_compaction": True,
                        "mask_based_pruning": False,
                        "validation_passed": True,
                    }

                    qbase.write_json_atomic(
                        final_summary,
                        final_summary_file,
                    )

                fp32_row = fp32_references[
                    (
                        fp32_references["model_name"]
                        == model_name
                    )
                    & (
                        fp32_references["seed"]
                        == seed
                    )
                ].iloc[0]

                pruning_row = pruning_references[
                    (
                        pruning_references["model_name"]
                        == model_name
                    )
                    & (
                        pruning_references["pruning_name"]
                        == pruning_name
                    )
                    & (
                        pruning_references["seed"]
                        == seed
                    )
                ].iloc[0]

                fp32_macro_f1 = float(
                    fp32_row["test_macro_f1"]
                )
                pruning_macro_f1 = float(
                    pruning_row["test_macro_f1"]
                )

                run_records.append(
                    {
                        "model_name": model_name,
                        "pruning_name": pruning_name,
                        "variant_name": variant_name,
                        "pruning_ratio": float(
                            pruning_ratio
                        ),
                        "seed": int(seed),
                        "fp32_learning_rate": float(
                            fp32_learning_rate
                        ),
                        "qat_learning_rate": float(
                            training_summary[
                                "qat_learning_rate"
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
                        "best_validation_macro_f1_converted_int8": float(
                            training_summary[
                                "best_validation_macro_f1_converted_int8"
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
                        "artifact_size_bytes": int(
                            artifact_metadata[
                                "artifact_size_bytes"
                            ]
                        ),
                        "fake_quant_module_count": int(
                            training_summary[
                                "fake_quant_module_count"
                            ]
                        ),
                        "quantized_module_count": int(
                            artifact_metadata[
                                "quantized_module_count"
                            ]
                        ),
                        "quantized_graph_node_count": int(
                            artifact_metadata[
                                "quantized_graph_node_count"
                            ]
                        ),
                        "fp32_test_macro_f1": (
                            fp32_macro_f1
                        ),
                        "pruning_only_test_macro_f1": (
                            pruning_macro_f1
                        ),
                        "test_loss": float(
                            test_metrics["loss"]
                        ),
                        "test_accuracy": float(
                            test_metrics["accuracy"]
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
                            test_metrics["macro_f1"]
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
                        "macro_f1_delta_vs_fp32": float(
                            test_metrics["macro_f1"]
                            - fp32_macro_f1
                        ),
                        "macro_f1_delta_vs_pruning_only": float(
                            test_metrics["macro_f1"]
                            - pruning_macro_f1
                        ),
                        "test_sample_count": int(
                            test_metrics["sample_count"]
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
                        "test_evaluation_attempt_count": 1,
                        "test_used_for_selection": False,
                        "physical_compaction": True,
                        "mask_based_pruning": False,
                        "run_directory": str(
                            run_directory
                        ),
                    }
                )

                for (
                    class_index,
                    class_name,
                ) in enumerate(TARGET_CLASSES):
                    class_metrics = test_metrics[
                        "per_class"
                    ][class_name]

                    per_class_records.append(
                        {
                            "model_name": model_name,
                            "pruning_name": pruning_name,
                            "variant_name": variant_name,
                            "pruning_ratio": float(
                                pruning_ratio
                            ),
                            "seed": int(seed),
                            "class_index": int(
                                class_index
                            ),
                            "class_name": class_name,
                            "support": int(
                                class_metrics["support"]
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
                                class_metrics["recall"]
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
                    TARGET_CLASSES
                ):
                    for (
                        predicted_index,
                        predicted_class,
                    ) in enumerate(TARGET_CLASSES):
                        confusion_records.append(
                            {
                                "model_name": model_name,
                                "pruning_name": pruning_name,
                                "variant_name": variant_name,
                                "pruning_ratio": float(
                                    pruning_ratio
                                ),
                                "seed": int(seed),
                                "true_class_index": int(
                                    true_index
                                ),
                                "true_class": true_class,
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
                    "Test Macro F1            : "
                    f"{test_metrics['macro_f1']:.6f}"
                )
                print(
                    "FP32'ye göre değişim     : "
                    f"{test_metrics['macro_f1'] - fp32_macro_f1:+.6f}"
                )
                print(
                    "Pruning-only değişimi    : "
                    f"{test_metrics['macro_f1'] - pruning_macro_f1:+.6f}"
                )
                print(
                    "Test seçime katıldı      : False"
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

    if len(run_frame) != 45:
        raise RuntimeError(
            "Tamamlanan pruning-QAT deney sayısı 45 değil."
        )

    if run_frame.duplicated(
        subset=[
            "model_name",
            "variant_name",
            "seed",
        ]
    ).any():
        raise RuntimeError(
            "Tekrarlanan model-varyant-seed sonucu bulundu."
        )

    if (
        run_frame[
            "test_evaluation_attempt_count"
        ]
        != 1
    ).any():
        raise RuntimeError(
            "Test değerlendirme girişimlerinden biri 1 değil."
        )

    if run_frame[
        "test_used_for_selection"
    ].any():
        raise RuntimeError(
            "Test sonuçlarından biri model seçimine katılmış."
        )

    if not run_frame[
        "physical_compaction"
    ].all():
        raise RuntimeError(
            "Çalışmalardan biri fiziksel küçültme içermiyor."
        )

    if run_frame[
        "mask_based_pruning"
    ].any():
        raise RuntimeError(
            "Çalışmalardan biri maskeli budama içeriyor."
        )

    quantized_coverage = (
        run_frame[
            "quantized_module_count"
        ]
        + run_frame[
            "quantized_graph_node_count"
        ]
    )

    if (
        quantized_coverage <= 0
    ).any():
        raise RuntimeError(
            "Çalışmalardan birinde gerçek quantized kapsam yok."
        )

    aggregate_frame = aggregate_results(
        run_frame
    )
    per_class_aggregate_frame = (
        aggregate_per_class(
            per_class_frame
        )
    )

    run_report_file = (
        report_directory
        / "nbaiot_family3_pruning_qat_runs.csv"
    )
    aggregate_report_file = (
        report_directory
        / "nbaiot_family3_pruning_qat_aggregate.csv"
    )
    per_class_run_file = (
        report_directory
        / "nbaiot_family3_pruning_qat_per_class_runs.csv"
    )
    per_class_aggregate_file = (
        report_directory
        / "nbaiot_family3_pruning_qat_per_class_aggregate.csv"
    )
    confusion_report_file = (
        report_directory
        / "nbaiot_family3_pruning_qat_confusion_matrices.csv"
    )
    summary_file = (
        report_directory
        / "nbaiot_family3_pruning_qat_summary.json"
    )

    qbase.write_csv_atomic(
        run_frame,
        run_report_file,
    )
    qbase.write_csv_atomic(
        aggregate_frame,
        aggregate_report_file,
    )
    qbase.write_csv_atomic(
        per_class_frame,
        per_class_run_file,
    )
    qbase.write_csv_atomic(
        per_class_aggregate_frame,
        per_class_aggregate_file,
    )
    qbase.write_csv_atomic(
        confusion_frame,
        confusion_report_file,
    )

    ranking_frame = aggregate_frame[
        aggregate_frame["metric"]
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
        .reset_index(drop=True)
    )

    ranking_records = [
        {
            "rank": int(index + 1),
            "model_name": str(
                row.model_name
            ),
            "variant_name": str(
                row.variant_name
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
            "N-BaIoT family_3 physical structured "
            "pruning plus FX QAT multi-seed"
        ),
        "protocol_file": str(protocol_file),
        "protocol_sha256": protocol_sha256,
        "model_count": len(MODEL_NAMES),
        "pruning_level_count": len(
            PRUNING_RATIOS
        ),
        "seed_count": len(SEEDS),
        "total_run_count": len(run_frame),
        "models": list(MODEL_NAMES),
        "variants": list(VARIANT_NAMES),
        "seeds": list(SEEDS),
        "backend": args.backend,
        "physical_compaction": True,
        "mask_based_pruning": False,
        "test_evaluations_per_combination": 1,
        "test_used_for_model_selection": False,
        "primary_metric": "test_macro_f1",
        "security_metric": (
            "per_class_false_negative_rate"
        ),
        "macro_f1_ranking": ranking_records,
        "run_report": str(run_report_file),
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
            "Not yet performed; final cross-variant statistics "
            "are planned after deployment profiling."
        ),
        "validation_passed": True,
    }

    qbase.write_json_atomic(
        summary_document,
        summary_file,
    )

    print()
    print("=" * 78)
    print(
        "Fiziksel Budama + QAT Ana Deneyleri Tamamlandı"
    )
    print("=" * 78)

    for record in ranking_records:
        print(
            f"{record['rank']:2d}. "
            f"{record['model_name']:14s} "
            f"{record['variant_name']:7s} | "
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
        "Gerçek quantized kapsam: True"
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
