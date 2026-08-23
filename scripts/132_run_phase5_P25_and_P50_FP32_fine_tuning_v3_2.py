from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

X_TRAIN_PATH = FINAL_CACHE / "X_train.npy"
Y_TRAIN_PATH = FINAL_CACHE / "y_train.npy"

X_VALIDATION_PATH = FINAL_CACHE / "X_validation.npy"
Y_VALIDATION_PATH = FINAL_CACHE / "y_validation.npy"
RAW_VALIDATION_PATH = (
    FINAL_CACHE / "raw_row_count_validation.npy"
)

X_TEST_PATH = FINAL_CACHE / "X_test.npy"
Y_TEST_PATH = FINAL_CACHE / "y_test.npy"
RAW_TEST_PATH = (
    FINAL_CACHE / "raw_row_count_test.npy"
)

PHASE5_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase5_fair_budget_compression_protocol_v3_2.json"
)

LOCKED_MATRIX = (
    AUDIT / "phase5_locked_configuration_matrix_v3_2.csv"
)

NOFT_LOCK = (
    AUDIT
    / "phase5_pruning_noFT_evaluation_locked_v3_2.json"
)

PRUNING_SOURCES_VERIFIED_LOCK = (
    AUDIT
    / "phase5_physical_pruning_sources_verified_locked_v3_2.json"
)

PRUNING_SOURCES_REGISTRY = (
    AUDIT
    / "phase5_physical_pruning_source_checkpoint_registry_v3_2.csv"
)

PRUNING_ENGINE_LOCK = (
    AUDIT
    / "phase5_physical_pruning_engine_locked_v3_2.json"
)

PRUNING_ENGINE_PATH = (
    ROOT
    / "src"
    / "compression"
    / "phase5_physical_pruning_engine_v3_2.py"
)

MODEL_SOURCE = (
    ROOT / "src" / "models" / "nbaiot_models.py"
)

PREPROCESSING_DIRECTORY = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "shared"
    / "preprocessing"
)

SCALER_NPZ = (
    PREPROCESSING_DIRECTORY
    / "phase5_train_only_standard_scaler_v3_2.npz"
)

CLASS_WEIGHTS_NPZ = (
    PREPROCESSING_DIRECTORY
    / "phase5_train_only_class_weights_v3_2.npz"
)

PREPROCESSING_LOCK = (
    AUDIT / "phase5_preprocessing_locked_v3_2.json"
)

AGGREGATE_RUNS_CSV = (
    AUDIT
    / "phase5_pruning_FP32_FT_all_runs_v3_2.csv"
)

GROUP_SUMMARY_CSV = (
    AUDIT
    / "phase5_pruning_FP32_FT_group_summary_v3_2.csv"
)

AGGREGATE_JSON = (
    AUDIT
    / "phase5_pruning_FP32_FT_summary_v3_2.json"
)

COMPLETION_JSON = (
    AUDIT
    / "phase5_pruning_FP32_FT_completed_v3_2.json"
)

PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
ENGINE_VERSION = "phase5_physical_pruning_engine_v3_2"

ARCHITECTURES = (
    "tinyml_mlp",
    "compact_dnn",
)

VARIANTS = (
    "P25",
    "P50",
)

SOURCE_VARIANT = {
    "P25": "P25-noFT",
    "P50": "P50-noFT",
}

CANONICAL_REPORT_NAME = {
    "P25": "P25-FP32-FT",
    "P50": "P50-FP32-FT",
}

OUTPUT_VARIANT_DIRECTORY = {
    "P25": "p25",
    "P50": "p50",
}

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

BASELINE_LEARNING_RATE = {
    "tinyml_mlp": 0.003,
    "compact_dnn": 0.001,
}

FINE_TUNING_LEARNING_RATE = {
    architecture: value * 0.1
    for architecture, value
    in BASELINE_LEARNING_RATE.items()
}

HIDDEN_DIMS = {
    "tinyml_mlp": [64, 32],
    "compact_dnn": [128, 64, 32],
}

EXPECTED_WIDTHS = {
    "tinyml_mlp": {
        "P25": [
            [115, 48],
            [48, 24],
            [24, 3],
        ],
        "P50": [
            [115, 32],
            [32, 16],
            [16, 3],
        ],
    },
    "compact_dnn": {
        "P25": [
            [115, 96],
            [96, 48],
            [48, 24],
            [24, 3],
        ],
        "P50": [
            [115, 64],
            [64, 32],
            [32, 16],
            [16, 3],
        ],
    },
}

EXPECTED_FEATURE_COUNT = 115
EXPECTED_CLASS_COUNT = 3

EXPECTED_CLASS_LABELS = np.asarray(
    [0, 1, 2],
    dtype=np.int64,
)

CLASS_NAMES = (
    "benign",
    "gafgyt",
    "mirai",
)

EXPECTED_TRAIN_ROWS = 1_534_583
EXPECTED_VALIDATION_ROWS = 371_797
EXPECTED_TEST_ROWS = 371_796
EXPECTED_VALIDATION_RAW_ROWS = 1_059_390
EXPECTED_TEST_RAW_ROWS = 1_059_393

BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 16_384
MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0002
WEIGHT_DECAY = 0.0001
CPU_THREADS = 4
MINIMUM_FREE_DISK_GIB = 2.0

PRIMARY_SELECTION_METRIC = (
    "validation_fingerprint_macro_f1"
)

FRAGILITY_MACRO_F1_THRESHOLD = 0.85
FRAGILITY_ATTACK_FNR_THRESHOLD = 0.40


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(
            chunk_size
        ):
            digest.update(block)

    return digest.hexdigest()


def atomic_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def atomic_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        temporary,
        path,
    )


def atomic_npz(
    path: Path,
    **values: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **values,
        )

    os.replace(
        temporary,
        path,
    )


def atomic_torch_save(
    value: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        value,
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


def file_record(
    path: Path,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    record = {
        "path": str(path),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }

    if relative_to is not None:
        record["relative_path"] = str(
            path.relative_to(
                relative_to
            )
        )

    return record


def set_all_seeds(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_module(
    path: Path,
    module_name: str,
) -> ModuleType:
    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"Could not import module: {path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    sys.modules[module_name] = module

    try:
        specification.loader.exec_module(
            module
        )
    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    return module


def callable_kwargs(
    candidate: Any,
    architecture: str,
) -> dict[str, Any]:
    signature = inspect.signature(
        candidate
    )

    kwargs: dict[str, Any] = {}

    for name, parameter in (
        signature.parameters.items()
    ):
        if name in {
            "self",
            "args",
            "kwargs",
        }:
            continue

        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        normalized = "".join(
            character
            for character in name.lower()
            if character.isalnum()
        )

        if normalized in {
            "inputdim",
            "inputsize",
            "inputfeatures",
            "nfeatures",
            "numfeatures",
            "infeatures",
            "featurecount",
        }:
            kwargs[name] = (
                EXPECTED_FEATURE_COUNT
            )
            continue

        if normalized in {
            "numclasses",
            "nclasses",
            "classcount",
            "outputdim",
            "outputsize",
            "outfeatures",
        }:
            kwargs[name] = (
                EXPECTED_CLASS_COUNT
            )
            continue

        if normalized in {
            "hiddendims",
            "hiddensizes",
            "hiddenlayers",
            "hiddenunits",
        }:
            kwargs[name] = list(
                HIDDEN_DIMS[
                    architecture
                ]
            )
            continue

        if (
            parameter.default
            is inspect.Parameter.empty
        ):
            raise RuntimeError(
                "Unresolved model constructor "
                f"parameter: {name}"
            )

    return kwargs


def instantiate_B0_model(
    model_module: ModuleType,
    model_symbol: str,
    architecture: str,
    seed: int,
) -> nn.Module:
    if not hasattr(
        model_module,
        model_symbol,
    ):
        raise RuntimeError(
            f"Missing model symbol: {model_symbol}"
        )

    candidate = getattr(
        model_module,
        model_symbol,
    )

    set_all_seeds(seed)

    model = candidate(
        **callable_kwargs(
            candidate,
            architecture,
        )
    )

    if not isinstance(
        model,
        nn.Module,
    ):
        raise RuntimeError(
            "Model constructor did not "
            "return torch.nn.Module."
        )

    return model


def extract_logits(
    output: Any,
) -> torch.Tensor:
    if isinstance(
        output,
        torch.Tensor,
    ):
        return output

    if isinstance(
        output,
        (tuple, list),
    ):
        for value in output:
            if isinstance(
                value,
                torch.Tensor,
            ):
                return value

    if isinstance(
        output,
        dict,
    ):
        for key in (
            "logits",
            "output",
            "outputs",
            "predictions",
        ):
            value = output.get(key)

            if isinstance(
                value,
                torch.Tensor,
            ):
                return value

    raise RuntimeError(
        "Model output does not contain logits."
    )


def transform_batch(
    source: np.ndarray,
    mean64: np.ndarray,
    scale64: np.ndarray,
) -> np.ndarray:
    transformed = (
        (
            np.asarray(
                source,
                dtype=np.float64,
            )
            - mean64
        )
        / scale64
    ).astype(np.float32)

    if not np.isfinite(
        transformed
    ).all():
        raise RuntimeError(
            "Non-finite value after "
            "train-only StandardScaler."
        )

    return transformed


def evaluate_model(
    model: nn.Module,
    x_source: np.ndarray,
    y_source: np.ndarray,
    mean64: np.ndarray,
    scale64: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    model.eval()

    y_true = np.asarray(
        y_source,
        dtype=np.int64,
    )

    predictions = np.empty(
        len(y_true),
        dtype=np.int8,
    )

    probabilities = np.empty(
        (
            len(y_true),
            EXPECTED_CLASS_COUNT,
        ),
        dtype=np.float32,
    )

    started = time.perf_counter()

    with torch.inference_mode():
        for start in range(
            0,
            len(y_true),
            EVAL_BATCH_SIZE,
        ):
            end = min(
                start + EVAL_BATCH_SIZE,
                len(y_true),
            )

            x_batch = torch.from_numpy(
                transform_batch(
                    x_source[start:end],
                    mean64,
                    scale64,
                )
            )

            logits = extract_logits(
                model(x_batch)
            )

            if tuple(
                logits.shape
            ) != (
                end - start,
                EXPECTED_CLASS_COUNT,
            ):
                raise RuntimeError(
                    "Evaluation output shape mismatch."
                )

            batch_probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )
            )

            probabilities[start:end] = (
                batch_probabilities
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            predictions[start:end] = (
                torch.argmax(
                    logits,
                    dim=1,
                )
                .cpu()
                .numpy()
                .astype(np.int8)
            )

    return (
        predictions,
        probabilities,
        y_true,
        time.perf_counter()
        - started,
    )


def metric_view(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray | None,
) -> dict[str, Any]:
    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=EXPECTED_CLASS_LABELS,
        average=None,
        sample_weight=sample_weight,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=EXPECTED_CLASS_LABELS,
        sample_weight=sample_weight,
    )

    if sample_weight is None:
        matrix = matrix.astype(
            np.int64
        )
    else:
        matrix = np.rint(
            matrix
        ).astype(np.int64)

    support_total = float(
        np.sum(support)
    )

    result: dict[str, Any] = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
                sample_weight=sample_weight,
            )
        ),
        "macro_precision": float(
            np.mean(precision)
        ),
        "macro_recall": float(
            np.mean(recall)
        ),
        "balanced_accuracy": float(
            np.mean(recall)
        ),
        "macro_f1": float(
            np.mean(f1)
        ),
        "weighted_f1": float(
            np.average(
                f1,
                weights=support,
            )
            if support_total > 0
            else 0.0
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
                sample_weight=sample_weight,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=EXPECTED_CLASS_LABELS,
                sample_weight=sample_weight,
            )
        ),
        "roc_auc_ovr_macro": float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=EXPECTED_CLASS_LABELS,
                multi_class="ovr",
                average="macro",
                sample_weight=sample_weight,
            )
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "per_class": {},
    }

    for index, class_name in enumerate(
        CLASS_NAMES
    ):
        result["per_class"][
            class_name
        ] = {
            "label": int(
                EXPECTED_CLASS_LABELS[
                    index
                ]
            ),
            "precision": float(
                precision[index]
            ),
            "recall": float(
                recall[index]
            ),
            "f1": float(
                f1[index]
            ),
            "support": float(
                support[index]
            ),
            "false_negative_rate": float(
                1.0 - recall[index]
            ),
        }

    return result


def clone_state_dict(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value
        in model.state_dict().items()
    }


def clone_optimizer_state(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return copy.deepcopy(
        optimizer.state_dict()
    )


def raw_tensor_bytes(
    model: nn.Module,
) -> int:
    return int(
        sum(
            value.numel()
            * value.element_size()
            for value
            in model.state_dict().values()
        )
    )


def save_confusion_csv(
    path: Path,
    matrix: list[list[int]],
) -> None:
    rows = []

    for class_name, values in zip(
        CLASS_NAMES,
        matrix,
    ):
        rows.append(
            {
                "true_class": class_name,
                "pred_benign": values[0],
                "pred_gafgyt": values[1],
                "pred_mirai": values[2],
            }
        )

    atomic_csv(
        path,
        rows,
        [
            "true_class",
            "pred_benign",
            "pred_gafgyt",
            "pred_mirai",
        ],
    )


def save_per_class_csv(
    path: Path,
    primary: dict[str, Any],
    weighted: dict[str, Any],
) -> None:
    rows = []

    for class_name in CLASS_NAMES:
        primary_row = primary[
            "per_class"
        ][class_name]

        weighted_row = weighted[
            "per_class"
        ][class_name]

        rows.append(
            {
                "class_name": class_name,
                "label": (
                    primary_row["label"]
                ),
                "fingerprint_precision": (
                    primary_row[
                        "precision"
                    ]
                ),
                "fingerprint_recall": (
                    primary_row[
                        "recall"
                    ]
                ),
                "fingerprint_f1": (
                    primary_row["f1"]
                ),
                "fingerprint_support": (
                    primary_row[
                        "support"
                    ]
                ),
                "fingerprint_fnr": (
                    primary_row[
                        "false_negative_rate"
                    ]
                ),
                "raw_weighted_precision": (
                    weighted_row[
                        "precision"
                    ]
                ),
                "raw_weighted_recall": (
                    weighted_row[
                        "recall"
                    ]
                ),
                "raw_weighted_f1": (
                    weighted_row["f1"]
                ),
                "raw_weighted_support": (
                    weighted_row[
                        "support"
                    ]
                ),
                "raw_weighted_fnr": (
                    weighted_row[
                        "false_negative_rate"
                    ]
                ),
            }
        )

    atomic_csv(
        path,
        rows,
        [
            "class_name",
            "label",
            "fingerprint_precision",
            "fingerprint_recall",
            "fingerprint_f1",
            "fingerprint_support",
            "fingerprint_fnr",
            "raw_weighted_precision",
            "raw_weighted_recall",
            "raw_weighted_f1",
            "raw_weighted_support",
            "raw_weighted_fnr",
        ],
    )


def artifact_inventory(
    output_directory: Path,
) -> list[dict[str, Any]]:
    excluded = {
        "run_manifest.json",
        "run_status.json",
    }

    return [
        file_record(
            path,
            relative_to=output_directory,
        )
        for path in sorted(
            output_directory.iterdir()
        )
        if path.is_file()
        and path.name not in excluded
    ]


def output_directory(
    architecture: str,
    variant: str,
    seed: int,
) -> Path:
    return (
        ROOT
        / "results"
        / "v2"
        / "phase5_compression"
        / architecture
        / OUTPUT_VARIANT_DIRECTORY[
            variant
        ]
        / f"seed_{seed}"
    )


def build_run_summary_from_files(
    run_directory: Path,
) -> dict[str, Any]:
    metrics_path = (
        run_directory
        / "metrics.json"
    )

    manifest_path = (
        run_directory
        / "run_manifest.json"
    )

    status_path = (
        run_directory
        / "run_status.json"
    )

    metrics = read_json(
        metrics_path
    )

    manifest = read_json(
        manifest_path
    )

    status = read_json(
        status_path
    )

    if not (
        status.get("status")
        == "completed"
        and manifest.get("status")
        == "completed"
        and manifest.get(
            "all_integrity_checks_passed"
        )
        is True
        and int(
            status.get(
                "test_evaluation_count"
            )
        )
        == 1
        and int(
            metrics.get(
                "test_evaluation_count"
            )
        )
        == 1
    ):
        raise RuntimeError(
            "Existing completed run failed "
            f"resume validation: {run_directory}"
        )

    return {
        "run_id": metrics["run_id"],
        "architecture": (
            metrics["architecture"]
        ),
        "variant": metrics["variant"],
        "canonical_report_name": (
            metrics[
                "canonical_report_name"
            ]
        ),
        "source_variant": (
            metrics["source"][
                "source_variant"
            ]
        ),
        "seed": int(metrics["seed"]),
        "best_epoch": int(
            metrics["selection"][
                "best_epoch"
            ]
        ),
        "epochs_completed": int(
            metrics["training"][
                "epochs_completed"
            ]
        ),
        "best_validation_macro_f1": float(
            metrics["selection"][
                "best_validation_macro_f1"
            ]
        ),
        "test_fingerprint_macro_f1": float(
            metrics["test"][
                "primary_fingerprint_level"
            ]["macro_f1"]
        ),
        "test_fingerprint_accuracy": float(
            metrics["test"][
                "primary_fingerprint_level"
            ]["accuracy"]
        ),
        "test_raw_weighted_macro_f1": float(
            metrics["test"][
                "secondary_raw_record_weighted"
            ]["macro_f1"]
        ),
        "test_gafgyt_fnr": float(
            metrics["test"][
                "primary_fingerprint_level"
            ]["per_class"][
                "gafgyt"
            ][
                "false_negative_rate"
            ]
        ),
        "test_mirai_fnr": float(
            metrics["test"][
                "primary_fingerprint_level"
            ]["per_class"][
                "mirai"
            ][
                "false_negative_rate"
            ]
        ),
        "parameter_count": int(
            metrics["model_complexity"][
                "parameter_count"
            ]
        ),
        "linear_macs_per_sample": int(
            metrics["model_complexity"][
                "linear_macs_per_sample"
            ]
        ),
        "raw_tensor_bytes": int(
            metrics["model_complexity"][
                "raw_tensor_bytes"
            ]
        ),
        "serialized_checkpoint_bytes": int(
            metrics["model_complexity"][
                "serialized_checkpoint_bytes"
            ]
        ),
        "training_elapsed_seconds": float(
            metrics["training"][
                "elapsed_seconds"
            ]
        ),
        "test_evaluation_count": 1,
        "fragility_gate_triggered": bool(
            metrics[
                "fragility_gate"
            ]["triggered"]
        ),
        "source_checkpoint_path": (
            metrics["source"][
                "source_checkpoint_path"
            ]
        ),
        "source_checkpoint_sha256": (
            metrics["source"][
                "source_checkpoint_sha256"
            ]
        ),
        "output_directory": str(
            run_directory
        ),
        "metrics_sha256": (
            sha256_file(
                metrics_path
            )
        ),
        "run_manifest_sha256": (
            sha256_file(
                manifest_path
            )
        ),
        "run_status": "resumed_completed",
    }


def aggregate_statistics(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(
            np.mean(array)
        ),
        "std_population": float(
            np.std(
                array,
                ddof=0,
            )
        ),
        "minimum": float(
            np.min(array)
        ),
        "maximum": float(
            np.max(array)
        ),
    }


required_paths = (
    FINAL_CACHE,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_VALIDATION_PATH,
    Y_VALIDATION_PATH,
    RAW_VALIDATION_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    RAW_TEST_PATH,
    PHASE5_PROTOCOL,
    LOCKED_MATRIX,
    NOFT_LOCK,
    PRUNING_SOURCES_VERIFIED_LOCK,
    PRUNING_SOURCES_REGISTRY,
    PRUNING_ENGINE_LOCK,
    PRUNING_ENGINE_PATH,
    MODEL_SOURCE,
    SCALER_NPZ,
    CLASS_WEIGHTS_NPZ,
    PREPROCESSING_LOCK,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for aggregate_path in (
    AGGREGATE_RUNS_CSV,
    GROUP_SUMMARY_CSV,
    AGGREGATE_JSON,
    COMPLETION_JSON,
):
    if aggregate_path.exists():
        raise FileExistsError(
            "FP32 fine-tuning aggregate "
            "artifact already exists; "
            f"refusing to overwrite: "
            f"{aggregate_path}"
        )

free_disk_gib = (
    shutil.disk_usage(
        ROOT
    ).free
    / (
        1024 ** 3
    )
)

if (
    free_disk_gib
    < MINIMUM_FREE_DISK_GIB
):
    raise RuntimeError(
        "Insufficient free disk space. "
        f"Required >= "
        f"{MINIMUM_FREE_DISK_GIB:.3f} GiB, "
        f"observed={free_disk_gib:.3f} GiB."
    )

protocol = read_json(
    PHASE5_PROTOCOL
)

noFT_lock = read_json(
    NOFT_LOCK
)

sources_verified_lock = read_json(
    PRUNING_SOURCES_VERIFIED_LOCK
)

engine_lock = read_json(
    PRUNING_ENGINE_LOCK
)

preprocessing_lock = read_json(
    PREPROCESSING_LOCK
)

matrix_rows = read_csv(
    LOCKED_MATRIX
)

source_rows = read_csv(
    PRUNING_SOURCES_REGISTRY
)

selected_matrix_rows = [
    row
    for row in matrix_rows
    if row["architecture"]
    in ARCHITECTURES
    and row["variant"]
    in VARIANTS
]

expected_keys = {
    (
        architecture,
        variant,
        seed,
    )
    for architecture in ARCHITECTURES
    for variant in VARIANTS
    for seed in SEEDS
}

matrix_lookup = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    ): row
    for row in selected_matrix_rows
}

source_lookup = {
    (
        row["architecture"],
        (
            "P25"
            if row["variant"]
            == "P25-noFT"
            else "P50"
        ),
        int(row["seed"]),
    ): row
    for row in source_rows
}

entry_checks = {
    "protocol_locked": (
        protocol.get("status")
        == "locked"
        and protocol.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
    ),
    "noFT_evaluation_locked": (
        noFT_lock.get("status")
        == "locked"
        and noFT_lock.get(
            "ready_for_P25_and_P50_fine_tuning"
        )
        is True
        and noFT_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "sources_independently_verified": (
        sources_verified_lock.get(
            "status"
        )
        == "locked"
        and sources_verified_lock.get(
            "ready_for_P25_noFT_and_P50_noFT_evaluation"
        )
        is True
        and sources_verified_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "source_registry_hash_matches": (
        sources_verified_lock.get(
            "source_registry_sha256"
        )
        == sha256_file(
            PRUNING_SOURCES_REGISTRY
        )
    ),
    "pruning_engine_locked": (
        engine_lock.get("status")
        == "locked"
        and engine_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "pruning_engine_hash_matches": (
        engine_lock.get(
            "engine_source_sha256"
        )
        == sha256_file(
            PRUNING_ENGINE_PATH
        )
    ),
    "preprocessing_locked": (
        preprocessing_lock.get(
            "status"
        )
        == "locked"
        and preprocessing_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "matrix_has_twenty_FT_rows": (
        set(
            matrix_lookup.keys()
        )
        == expected_keys
    ),
    "source_registry_has_twenty_rows": (
        set(
            source_lookup.keys()
        )
        == expected_keys
    ),
    "matrix_budget_matches": all(
        int(row["max_epochs"])
        == MAX_EPOCHS
        and int(row["batch_size"])
        == BATCH_SIZE
        and int(
            row[
                "early_stopping_patience"
            ]
        )
        == EARLY_STOPPING_PATIENCE
        and math.isclose(
            float(
                row[
                    "early_stopping_min_delta"
                ]
            ),
            EARLY_STOPPING_MIN_DELTA,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            float(
                row["learning_rate"]
            ),
            FINE_TUNING_LEARNING_RATE[
                row["architecture"]
            ],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and int(
            row[
                "phase5_test_evaluation_count"
            ]
        )
        == 1
        for row in selected_matrix_rows
    ),
    "legacy_checkpoint_reuse_disabled": (
        protocol[
            "legacy_reuse_policy"
        ][
            "legacy_checkpoints_directly_reused"
        ]
        is False
    ),
}

failed_entry_checks = [
    name
    for name, passed
    in entry_checks.items()
    if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "FP32 fine-tuning entry gate "
        "failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

torch.set_num_threads(
    CPU_THREADS
)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

torch.use_deterministic_algorithms(
    True
)

x_train = np.load(
    X_TRAIN_PATH,
    mmap_mode="r",
)

y_train = np.load(
    Y_TRAIN_PATH,
    mmap_mode="r",
)

x_validation = np.load(
    X_VALIDATION_PATH,
    mmap_mode="r",
)

y_validation = np.load(
    Y_VALIDATION_PATH,
    mmap_mode="r",
)

raw_validation = np.asarray(
    np.load(
        RAW_VALIDATION_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

x_test = np.load(
    X_TEST_PATH,
    mmap_mode="r",
)

y_test = np.load(
    Y_TEST_PATH,
    mmap_mode="r",
)

raw_test = np.asarray(
    np.load(
        RAW_TEST_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

shape_checks = {
    "train_shape_matches": (
        x_train.shape
        == (
            EXPECTED_TRAIN_ROWS,
            EXPECTED_FEATURE_COUNT,
        )
        and y_train.shape
        == (
            EXPECTED_TRAIN_ROWS,
        )
    ),
    "validation_shape_matches": (
        x_validation.shape
        == (
            EXPECTED_VALIDATION_ROWS,
            EXPECTED_FEATURE_COUNT,
        )
        and y_validation.shape
        == (
            EXPECTED_VALIDATION_ROWS,
        )
        and raw_validation.shape
        == (
            EXPECTED_VALIDATION_ROWS,
        )
    ),
    "test_shape_matches": (
        x_test.shape
        == (
            EXPECTED_TEST_ROWS,
            EXPECTED_FEATURE_COUNT,
        )
        and y_test.shape
        == (
            EXPECTED_TEST_ROWS,
        )
        and raw_test.shape
        == (
            EXPECTED_TEST_ROWS,
        )
    ),
    "validation_raw_total_matches": (
        int(
            raw_validation.sum()
        )
        == EXPECTED_VALIDATION_RAW_ROWS
    ),
    "test_raw_total_matches": (
        int(
            raw_test.sum()
        )
        == EXPECTED_TEST_RAW_ROWS
    ),
}

failed_shape_checks = [
    name
    for name, passed
    in shape_checks.items()
    if not passed
]

if failed_shape_checks:
    raise RuntimeError(
        "FP32 fine-tuning cache checks "
        "failed: "
        + ", ".join(
            failed_shape_checks
        )
    )

with np.load(
    SCALER_NPZ
) as values:
    scaler_mean64 = np.array(
        values["mean_float64"],
        copy=True,
    )

    scaler_scale64 = np.array(
        values["scale_float64"],
        copy=True,
    )

with np.load(
    CLASS_WEIGHTS_NPZ
) as values:
    class_weights32 = np.array(
        values[
            "class_weights_float32"
        ],
        copy=True,
    )

if (
    scaler_mean64.shape
    != (
        EXPECTED_FEATURE_COUNT,
    )
    or scaler_scale64.shape
    != (
        EXPECTED_FEATURE_COUNT,
    )
    or class_weights32.shape
    != (
        EXPECTED_CLASS_COUNT,
    )
):
    raise RuntimeError(
        "FP32 fine-tuning preprocessing "
        "artifact shape mismatch."
    )

class_weight_tensor = (
    torch.from_numpy(
        class_weights32
    )
)

model_module = load_module(
    MODEL_SOURCE,
    "phase5_pruning_ft_models",
)

engine_module = load_module(
    PRUNING_ENGINE_PATH,
    "phase5_pruning_ft_engine",
)

if (
    engine_module.ENGINE_VERSION
    != ENGINE_VERSION
):
    raise RuntimeError(
        "Pruning-engine version mismatch."
    )

print("=" * 92)
print("PHASE 5 P25 AND P50 FP32 FINE-TUNING")
print("=" * 92)
print(
    "Runs                            : 20"
)
print(
    "Architectures                   : "
    f"{list(ARCHITECTURES)}"
)
print(
    "Variants                        : "
    f"{list(VARIANTS)}"
)
print(
    "Canonical names                 : "
    "P25-FP32-FT, P50-FP32-FT"
)
print(
    "Seeds                           : "
    f"{list(SEEDS)}"
)
print(
    "Train fingerprints              : "
    f"{EXPECTED_TRAIN_ROWS:,}"
)
print(
    "Validation rows                 : "
    f"{EXPECTED_VALIDATION_ROWS:,}"
)
print(
    "Test rows                       : "
    f"{EXPECTED_TEST_ROWS:,}"
)
print(
    "Batch size                      : "
    f"{BATCH_SIZE}"
)
print(
    "Max epochs                      : "
    f"{MAX_EPOCHS}"
)
print(
    "Early stopping                  : "
    f"patience={EARLY_STOPPING_PATIENCE}, "
    f"min_delta={EARLY_STOPPING_MIN_DELTA}"
)
print(
    "Learning rates                  : "
    f"{FINE_TUNING_LEARNING_RATE}"
)
print(
    "Test evaluation policy          : "
    "exactly once per run"
)
print(
    "Resume policy                   : "
    "completed runs reused; incomplete "
    "pre-test runs restarted; incomplete "
    "post-test runs blocked"
)
print(
    "Free disk                       : "
    f"{free_disk_gib:.3f} GiB"
)
print()

run_summaries: list[
    dict[str, Any]
] = []

ordered_keys = sorted(
    expected_keys,
    key=lambda value: (
        value[0],
        value[1],
        value[2],
    ),
)

for run_index, (
    architecture,
    variant,
    seed,
) in enumerate(
    ordered_keys,
    start=1,
):
    source_row = source_lookup[
        (
            architecture,
            variant,
            seed,
        )
    ]

    matrix_row = matrix_lookup[
        (
            architecture,
            variant,
            seed,
        )
    ]

    run_id = (
        f"{architecture}__"
        f"{variant.lower()}__"
        f"seed_{seed}"
    )

    run_directory = output_directory(
        architecture,
        variant,
        seed,
    )

    status_path = (
        run_directory
        / "run_status.json"
    )

    if run_directory.exists():
        if (
            status_path.exists()
            and read_json(
                status_path
            ).get("status")
            == "completed"
        ):
            summary = (
                build_run_summary_from_files(
                    run_directory
                )
            )

            run_summaries.append(
                summary
            )

            print(
                f"[{run_index}/20] "
                f"{run_id} | "
                "status=resumed_completed | "
                "test inference=False",
                flush=True,
            )

            continue

        if status_path.exists():
            existing_status = read_json(
                status_path
            )

            test_started = bool(
                existing_status.get(
                    "test_inference_started",
                    False,
                )
            )

            test_count = int(
                existing_status.get(
                    "test_evaluation_count",
                    0,
                )
            )

            if (
                test_started
                or test_count > 0
            ):
                raise RuntimeError(
                    "Incomplete run has already "
                    "entered test evaluation. "
                    "Automatic restart is blocked "
                    "to preserve the single-test "
                    f"policy: {run_directory}"
                )

        print(
            f"[{run_index}/20] "
            f"{run_id} | "
            "removing incomplete pre-test "
            "directory and restarting",
            flush=True,
        )

        shutil.rmtree(
            run_directory
        )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    source_checkpoint_path = Path(
        source_row[
            "source_checkpoint_path"
        ]
    )

    source_manifest_path = Path(
        source_row[
            "source_manifest_path"
        ]
    )

    B0_checkpoint_path = Path(
        source_row[
            "source_B0_checkpoint_path"
        ]
    )

    for path in (
        source_checkpoint_path,
        source_manifest_path,
        B0_checkpoint_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    source_checkpoint_hash = (
        sha256_file(
            source_checkpoint_path
        )
    )

    B0_checkpoint_hash = (
        sha256_file(
            B0_checkpoint_path
        )
    )

    if (
        source_checkpoint_hash
        != source_row[
            "source_checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            f"{run_id}: source checkpoint "
            "hash mismatch."
        )

    if (
        B0_checkpoint_hash
        != source_row[
            "source_B0_checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            f"{run_id}: B0 checkpoint "
            "hash mismatch."
        )

    atomic_json(
        status_path,
        {
            "status": "running",
            "stage": "training",
            "run_id": run_id,
            "architecture": architecture,
            "variant": variant,
            "canonical_report_name": (
                CANONICAL_REPORT_NAME[
                    variant
                ]
            ),
            "source_variant": (
                SOURCE_VARIANT[
                    variant
                ]
            ),
            "seed": seed,
            "test_evaluation_count": 0,
            "test_inference_started": False,
            "started_at_utc": utc_now(),
        },
    )

    source_checkpoint = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    B0_checkpoint = torch.load(
        B0_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_symbol = (
        source_row["model_symbol"]
    )

    B0_model = instantiate_B0_model(
        model_module,
        model_symbol,
        architecture,
        seed,
    )

    B0_model.load_state_dict(
        B0_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    (
        model,
        regenerated_metadata,
    ) = engine_module.physically_prune_mlp(
        B0_model,
        (
            0.25
            if variant == "P25"
            else 0.50
        ),
        source_checkpoint_sha256=(
            B0_checkpoint_hash
        ),
    )

    model.load_state_dict(
        source_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    expected_widths = (
        EXPECTED_WIDTHS[
            architecture
        ][variant]
    )

    if not (
        source_checkpoint[
            "pruning_metadata"
        ]
        == regenerated_metadata
        and regenerated_metadata[
            "compacted_linear_widths"
        ]
        == expected_widths
        and engine_module.inspect_linear_widths(
            model
        )
        == expected_widths
    ):
        raise RuntimeError(
            f"{run_id}: source model "
            "reconstruction failed."
        )

    learning_rate = float(
        matrix_row[
            "learning_rate"
        ]
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    rng = np.random.default_rng(
        seed
    )

    best_macro_f1 = -math.inf
    best_epoch = 0
    best_state: (
        dict[str, torch.Tensor]
        | None
    ) = None

    best_optimizer_state: (
        dict[str, Any]
        | None
    ) = None

    epochs_without_improvement = 0

    history_rows: list[
        dict[str, Any]
    ] = []

    training_started = (
        time.perf_counter()
    )

    print(
        f"[{run_index}/20] "
        f"{run_id} | "
        f"source={SOURCE_VARIANT[variant]} | "
        f"lr={learning_rate}",
        flush=True,
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        epoch_started = (
            time.perf_counter()
        )

        model.train()

        permutation = rng.permutation(
            EXPECTED_TRAIN_ROWS
        )

        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_count = 0

        for start in range(
            0,
            EXPECTED_TRAIN_ROWS,
            BATCH_SIZE,
        ):
            end = min(
                start + BATCH_SIZE,
                EXPECTED_TRAIN_ROWS,
            )

            indices = permutation[
                start:end
            ]

            x_batch = torch.from_numpy(
                transform_batch(
                    x_train[indices],
                    scaler_mean64,
                    scaler_scale64,
                )
            )

            y_batch = torch.from_numpy(
                np.asarray(
                    y_train[indices],
                    dtype=np.int64,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = extract_logits(
                model(x_batch)
            )

            loss = criterion(
                logits,
                y_batch,
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    f"{run_id}: non-finite "
                    f"loss at epoch {epoch}."
                )

            loss.backward()
            optimizer.step()

            batch_count = end - start

            epoch_loss_sum += (
                float(loss.item())
                * batch_count
            )

            epoch_correct += int(
                (
                    torch.argmax(
                        logits,
                        dim=1,
                    )
                    == y_batch
                )
                .sum()
                .item()
            )

            epoch_count += (
                batch_count
            )

        (
            validation_predictions,
            validation_probabilities,
            validation_true,
            validation_elapsed,
        ) = evaluate_model(
            model,
            x_validation,
            y_validation,
            scaler_mean64,
            scaler_scale64,
        )

        validation_primary = metric_view(
            validation_true,
            validation_predictions,
            validation_probabilities,
            sample_weight=None,
        )

        validation_macro_f1 = float(
            validation_primary[
                "macro_f1"
            ]
        )

        improved = (
            best_state is None
            or validation_macro_f1
            > (
                best_macro_f1
                + EARLY_STOPPING_MIN_DELTA
            )
        )

        if improved:
            best_macro_f1 = (
                validation_macro_f1
            )

            best_epoch = epoch

            best_state = (
                clone_state_dict(
                    model
                )
            )

            best_optimizer_state = (
                clone_optimizer_state(
                    optimizer
                )
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        epoch_elapsed = (
            time.perf_counter()
            - epoch_started
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": (
                    epoch_loss_sum
                    / epoch_count
                ),
                "train_accuracy": (
                    epoch_correct
                    / epoch_count
                ),
                "validation_macro_f1": (
                    validation_macro_f1
                ),
                "validation_accuracy": (
                    validation_primary[
                        "accuracy"
                    ]
                ),
                "validation_gafgyt_fnr": (
                    validation_primary[
                        "per_class"
                    ]["gafgyt"][
                        "false_negative_rate"
                    ]
                ),
                "validation_mirai_fnr": (
                    validation_primary[
                        "per_class"
                    ]["mirai"][
                        "false_negative_rate"
                    ]
                ),
                "improved": improved,
                "epochs_without_improvement": (
                    epochs_without_improvement
                ),
                "epoch_elapsed_seconds": (
                    epoch_elapsed
                ),
                "validation_elapsed_seconds": (
                    validation_elapsed
                ),
            }
        )

        atomic_json(
            status_path,
            {
                "status": "running",
                "stage": "training",
                "run_id": run_id,
                "architecture": architecture,
                "variant": variant,
                "canonical_report_name": (
                    CANONICAL_REPORT_NAME[
                        variant
                    ]
                ),
                "source_variant": (
                    SOURCE_VARIANT[
                        variant
                    ]
                ),
                "seed": seed,
                "current_epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": (
                    best_macro_f1
                ),
                "test_evaluation_count": 0,
                "test_inference_started": False,
                "updated_at_utc": utc_now(),
            },
        )

        print(
            "  "
            f"epoch={epoch:02d} | "
            "train_loss="
            f"{epoch_loss_sum / epoch_count:.6f} | "
            "train_acc="
            f"{epoch_correct / epoch_count:.6f} | "
            "val_macro_f1="
            f"{validation_macro_f1:.9f} | "
            f"best={best_macro_f1:.9f} | "
            "wait="
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE} | "
            "elapsed="
            f"{epoch_elapsed / 60.0:.2f} min",
            flush=True,
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "  early stopping triggered",
                flush=True,
            )

            break

    training_elapsed = (
        time.perf_counter()
        - training_started
    )

    if (
        best_state is None
        or best_optimizer_state is None
        or best_epoch <= 0
    ):
        raise RuntimeError(
            f"{run_id}: no best "
            "checkpoint was selected."
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    training_history_path = (
        run_directory
        / "training_history.csv"
    )

    atomic_csv(
        training_history_path,
        history_rows,
        [
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_macro_f1",
            "validation_accuracy",
            "validation_gafgyt_fnr",
            "validation_mirai_fnr",
            "improved",
            "epochs_without_improvement",
            "epoch_elapsed_seconds",
            "validation_elapsed_seconds",
        ],
    )

    checkpoint_path = (
        run_directory
        / "best_checkpoint.pt"
    )

    checkpoint_payload = {
        "run_id": run_id,
        "architecture": architecture,
        "variant": variant,
        "canonical_report_name": (
            CANONICAL_REPORT_NAME[
                variant
            ]
        ),
        "source_variant": (
            SOURCE_VARIANT[
                variant
            ]
        ),
        "seed": seed,
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "engine_version": (
            ENGINE_VERSION
        ),
        "model_symbol": model_symbol,
        "model_state_dict": (
            best_state
        ),
        "optimizer_state_dict": (
            best_optimizer_state
        ),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": (
            best_macro_f1
        ),
        "linear_widths": (
            engine_module.inspect_linear_widths(
                model
            )
        ),
        "parameter_count": (
            engine_module.count_parameters(
                model
            )
        ),
        "linear_macs_per_sample": (
            engine_module.linear_macs_per_sample(
                model
            )
        ),
        "source_checkpoint_path": (
            str(
                source_checkpoint_path
            )
        ),
        "source_checkpoint_sha256": (
            source_checkpoint_hash
        ),
        "source_B0_checkpoint_path": (
            str(
                B0_checkpoint_path
            )
        ),
        "source_B0_checkpoint_sha256": (
            B0_checkpoint_hash
        ),
        "pruning_engine_path": (
            str(
                PRUNING_ENGINE_PATH
            )
        ),
        "pruning_engine_sha256": (
            sha256_file(
                PRUNING_ENGINE_PATH
            )
        ),
    }

    atomic_torch_save(
        checkpoint_payload,
        checkpoint_path,
    )

    (
        validation_predictions,
        validation_probabilities,
        validation_true,
        validation_elapsed,
    ) = evaluate_model(
        model,
        x_validation,
        y_validation,
        scaler_mean64,
        scaler_scale64,
    )

    validation_primary = metric_view(
        validation_true,
        validation_predictions,
        validation_probabilities,
        sample_weight=None,
    )

    validation_weighted = metric_view(
        validation_true,
        validation_predictions,
        validation_probabilities,
        sample_weight=raw_validation,
    )

    validation_predictions_path = (
        run_directory
        / "validation_predictions.npz"
    )

    atomic_npz(
        validation_predictions_path,
        y_true=(
            validation_true.astype(
                np.int64
            )
        ),
        y_pred=(
            validation_predictions.astype(
                np.int8
            )
        ),
        probabilities=(
            validation_probabilities.astype(
                np.float32
            )
        ),
        raw_row_count=(
            raw_validation.astype(
                np.int64
            )
        ),
    )

    atomic_json(
        status_path,
        {
            "status": "running",
            "stage": "validation_complete",
            "run_id": run_id,
            "architecture": architecture,
            "variant": variant,
            "canonical_report_name": (
                CANONICAL_REPORT_NAME[
                    variant
                ]
            ),
            "source_variant": (
                SOURCE_VARIANT[
                    variant
                ]
            ),
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": (
                best_macro_f1
            ),
            "test_evaluation_count": 0,
            "test_inference_started": False,
            "validation_predictions_sha256": (
                sha256_file(
                    validation_predictions_path
                )
            ),
            "updated_at_utc": utc_now(),
        },
    )

    atomic_json(
        status_path,
        {
            "status": "running",
            "stage": "test_inference_started",
            "run_id": run_id,
            "architecture": architecture,
            "variant": variant,
            "canonical_report_name": (
                CANONICAL_REPORT_NAME[
                    variant
                ]
            ),
            "source_variant": (
                SOURCE_VARIANT[
                    variant
                ]
            ),
            "seed": seed,
            "best_epoch": best_epoch,
            "test_evaluation_count": 1,
            "test_inference_started": True,
            "validation_predictions_sha256": (
                sha256_file(
                    validation_predictions_path
                )
            ),
            "test_started_at_utc": utc_now(),
        },
    )

    (
        test_predictions,
        test_probabilities,
        test_true,
        test_elapsed,
    ) = evaluate_model(
        model,
        x_test,
        y_test,
        scaler_mean64,
        scaler_scale64,
    )

    test_predictions_path = (
        run_directory
        / "test_predictions.npz"
    )

    atomic_npz(
        test_predictions_path,
        y_true=(
            test_true.astype(
                np.int64
            )
        ),
        y_pred=(
            test_predictions.astype(
                np.int8
            )
        ),
        probabilities=(
            test_probabilities.astype(
                np.float32
            )
        ),
        raw_row_count=(
            raw_test.astype(
                np.int64
            )
        ),
    )

    test_primary = metric_view(
        test_true,
        test_predictions,
        test_probabilities,
        sample_weight=None,
    )

    test_weighted = metric_view(
        test_true,
        test_predictions,
        test_probabilities,
        sample_weight=raw_test,
    )

    fragility_reasons = []

    if (
        test_primary[
            "macro_f1"
        ]
        < FRAGILITY_MACRO_F1_THRESHOLD
    ):
        fragility_reasons.append(
            "test_fingerprint_macro_f1_below_0.85"
        )

    for attack_name in (
        "gafgyt",
        "mirai",
    ):
        if (
            test_primary[
                "per_class"
            ][attack_name][
                "false_negative_rate"
            ]
            > FRAGILITY_ATTACK_FNR_THRESHOLD
        ):
            fragility_reasons.append(
                f"{attack_name}_FNR_above_0.40"
            )

    fragility_triggered = (
        len(fragility_reasons) > 0
    )

    validation_confusion_path = (
        run_directory
        / "validation_confusion_fingerprint.csv"
    )

    validation_weighted_confusion_path = (
        run_directory
        / "validation_confusion_raw_weighted.csv"
    )

    validation_per_class_path = (
        run_directory
        / "validation_per_class_metrics.csv"
    )

    test_confusion_path = (
        run_directory
        / "test_confusion_fingerprint.csv"
    )

    test_weighted_confusion_path = (
        run_directory
        / "test_confusion_raw_weighted.csv"
    )

    test_per_class_path = (
        run_directory
        / "test_per_class_metrics.csv"
    )

    save_confusion_csv(
        validation_confusion_path,
        validation_primary[
            "confusion_matrix"
        ],
    )

    save_confusion_csv(
        validation_weighted_confusion_path,
        validation_weighted[
            "confusion_matrix"
        ],
    )

    save_per_class_csv(
        validation_per_class_path,
        validation_primary,
        validation_weighted,
    )

    save_confusion_csv(
        test_confusion_path,
        test_primary[
            "confusion_matrix"
        ],
    )

    save_confusion_csv(
        test_weighted_confusion_path,
        test_weighted[
            "confusion_matrix"
        ],
    )

    save_per_class_csv(
        test_per_class_path,
        test_primary,
        test_weighted,
    )

    model_raw_tensor_bytes = (
        raw_tensor_bytes(
            model
        )
    )

    metrics_path = (
        run_directory
        / "metrics.json"
    )

    metrics = {
        "status": "completed",
        "phase": 5,
        "run_id": run_id,
        "architecture": architecture,
        "variant": variant,
        "canonical_report_name": (
            CANONICAL_REPORT_NAME[
                variant
            ]
        ),
        "seed": seed,
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "engine_version": (
            ENGINE_VERSION
        ),
        "source": {
            "source_variant": (
                SOURCE_VARIANT[
                    variant
                ]
            ),
            "source_checkpoint_path": (
                str(
                    source_checkpoint_path
                )
            ),
            "source_checkpoint_sha256": (
                source_checkpoint_hash
            ),
            "source_B0_checkpoint_path": (
                str(
                    B0_checkpoint_path
                )
            ),
            "source_B0_checkpoint_sha256": (
                B0_checkpoint_hash
            ),
            "pruning_engine_path": (
                str(
                    PRUNING_ENGINE_PATH
                )
            ),
            "pruning_engine_sha256": (
                sha256_file(
                    PRUNING_ENGINE_PATH
                )
            ),
            "physical_compaction": True,
            "generated_directly_from_B0": True,
        },
        "selection": {
            "metric": (
                PRIMARY_SELECTION_METRIC
            ),
            "best_epoch": best_epoch,
            "best_validation_macro_f1": (
                best_macro_f1
            ),
            "validation_used_for_selection": True,
            "test_used_for_selection": False,
        },
        "training": {
            "optimizer": "AdamW",
            "batch_size": (
                BATCH_SIZE
            ),
            "max_epochs": (
                MAX_EPOCHS
            ),
            "epochs_completed": (
                len(history_rows)
            ),
            "learning_rate": (
                learning_rate
            ),
            "baseline_learning_rate": (
                BASELINE_LEARNING_RATE[
                    architecture
                ]
            ),
            "learning_rate_fraction_of_B0": (
                0.1
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),
            "early_stopping_min_delta": (
                EARLY_STOPPING_MIN_DELTA
            ),
            "elapsed_seconds": (
                training_elapsed
            ),
            "raw_occurrence_weight_used": False,
        },
        "validation": {
            "primary_fingerprint_level": (
                validation_primary
            ),
            "secondary_raw_record_weighted": (
                validation_weighted
            ),
            "elapsed_seconds": (
                validation_elapsed
            ),
        },
        "test": {
            "primary_fingerprint_level": (
                test_primary
            ),
            "secondary_raw_record_weighted": (
                test_weighted
            ),
            "elapsed_seconds": (
                test_elapsed
            ),
        },
        "model_complexity": {
            "linear_widths": (
                engine_module.inspect_linear_widths(
                    model
                )
            ),
            "parameter_count": (
                engine_module.count_parameters(
                    model
                )
            ),
            "linear_macs_per_sample": (
                engine_module.linear_macs_per_sample(
                    model
                )
            ),
            "raw_tensor_bytes": (
                model_raw_tensor_bytes
            ),
            "serialized_checkpoint_bytes": (
                int(
                    checkpoint_path
                    .stat()
                    .st_size
                )
            ),
        },
        "data_access": {
            "train_used_for_updates": True,
            "validation_used_for_selection": True,
            "test_used_for_selection": False,
            "test_evaluation_count": 1,
            "raw_occurrence_weight_used_for_fit": False,
        },
        "test_evaluation_count": 1,
        "fragility_gate": {
            "macro_f1_threshold": (
                FRAGILITY_MACRO_F1_THRESHOLD
            ),
            "attack_FNR_threshold": (
                FRAGILITY_ATTACK_FNR_THRESHOLD
            ),
            "triggered": (
                fragility_triggered
            ),
            "reasons": (
                fragility_reasons
            ),
        },
        "completed_at_utc": utc_now(),
    }

    atomic_json(
        metrics_path,
        metrics,
    )

    artifacts = artifact_inventory(
        run_directory
    )

    manifest_path = (
        run_directory
        / "run_manifest.json"
    )

    manifest = {
        "status": "completed",
        "phase": 5,
        "run_id": run_id,
        "architecture": architecture,
        "variant": variant,
        "canonical_report_name": (
            CANONICAL_REPORT_NAME[
                variant
            ]
        ),
        "source_variant": (
            SOURCE_VARIANT[
                variant
            ]
        ),
        "seed": seed,
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "engine_version": (
            ENGINE_VERSION
        ),
        "model_symbol": (
            model_symbol
        ),
        "source_checkpoint": (
            file_record(
                source_checkpoint_path
            )
        ),
        "source_manifest": (
            file_record(
                source_manifest_path
            )
        ),
        "source_B0_checkpoint": (
            file_record(
                B0_checkpoint_path
            )
        ),
        "pruning_engine": (
            file_record(
                PRUNING_ENGINE_PATH
            )
        ),
        "phase5_protocol": (
            file_record(
                PHASE5_PROTOCOL
            )
        ),
        "locked_matrix": (
            file_record(
                LOCKED_MATRIX
            )
        ),
        "scaler": (
            file_record(
                SCALER_NPZ
            )
        ),
        "class_weights": (
            file_record(
                CLASS_WEIGHTS_NPZ
            )
        ),
        "noFT_evaluation_lock": (
            file_record(
                NOFT_LOCK
            )
        ),
        "verified_sources_lock": (
            file_record(
                PRUNING_SOURCES_VERIFIED_LOCK
            )
        ),
        "fit_scope": {
            "train_used_for_updates": True,
            "validation_used_for_selection": True,
            "test_used_for_selection": False,
            "test_evaluation_count": 1,
            "raw_occurrence_weight_used_for_fit": False,
        },
        "parameters": {
            "optimizer": "AdamW",
            "batch_size": (
                BATCH_SIZE
            ),
            "max_epochs": (
                MAX_EPOCHS
            ),
            "learning_rate": (
                learning_rate
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),
            "early_stopping_min_delta": (
                EARLY_STOPPING_MIN_DELTA
            ),
        },
        "best_epoch": best_epoch,
        "best_validation_macro_f1": (
            best_macro_f1
        ),
        "training_elapsed_seconds": (
            training_elapsed
        ),
        "model_complexity": (
            metrics[
                "model_complexity"
            ]
        ),
        "fragility_gate": (
            metrics[
                "fragility_gate"
            ]
        ),
        "artifacts": artifacts,
        "all_integrity_checks_passed": True,
        "completed_at_utc": utc_now(),
    }

    atomic_json(
        manifest_path,
        manifest,
    )

    atomic_json(
        status_path,
        {
            "status": "completed",
            "stage": "completed",
            "run_id": run_id,
            "architecture": architecture,
            "variant": variant,
            "canonical_report_name": (
                CANONICAL_REPORT_NAME[
                    variant
                ]
            ),
            "source_variant": (
                SOURCE_VARIANT[
                    variant
                ]
            ),
            "seed": seed,
            "best_epoch": best_epoch,
            "test_evaluation_count": 1,
            "test_inference_started": True,
            "test_predictions_sha256": (
                sha256_file(
                    test_predictions_path
                )
            ),
            "metrics_sha256": (
                sha256_file(
                    metrics_path
                )
            ),
            "run_manifest_sha256": (
                sha256_file(
                    manifest_path
                )
            ),
            "fragility_gate_triggered": (
                fragility_triggered
            ),
            "completed_at_utc": utc_now(),
        },
    )

    run_summary = {
        "run_id": run_id,
        "architecture": architecture,
        "variant": variant,
        "canonical_report_name": (
            CANONICAL_REPORT_NAME[
                variant
            ]
        ),
        "source_variant": (
            SOURCE_VARIANT[
                variant
            ]
        ),
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": (
            len(history_rows)
        ),
        "best_validation_macro_f1": (
            best_macro_f1
        ),
        "test_fingerprint_macro_f1": (
            test_primary[
                "macro_f1"
            ]
        ),
        "test_fingerprint_accuracy": (
            test_primary[
                "accuracy"
            ]
        ),
        "test_raw_weighted_macro_f1": (
            test_weighted[
                "macro_f1"
            ]
        ),
        "test_gafgyt_fnr": (
            test_primary[
                "per_class"
            ]["gafgyt"][
                "false_negative_rate"
            ]
        ),
        "test_mirai_fnr": (
            test_primary[
                "per_class"
            ]["mirai"][
                "false_negative_rate"
            ]
        ),
        "parameter_count": (
            engine_module.count_parameters(
                model
            )
        ),
        "linear_macs_per_sample": (
            engine_module.linear_macs_per_sample(
                model
            )
        ),
        "raw_tensor_bytes": (
            model_raw_tensor_bytes
        ),
        "serialized_checkpoint_bytes": (
            int(
                checkpoint_path
                .stat()
                .st_size
            )
        ),
        "training_elapsed_seconds": (
            training_elapsed
        ),
        "test_evaluation_count": 1,
        "fragility_gate_triggered": (
            fragility_triggered
        ),
        "source_checkpoint_path": (
            str(
                source_checkpoint_path
            )
        ),
        "source_checkpoint_sha256": (
            source_checkpoint_hash
        ),
        "output_directory": (
            str(
                run_directory
            )
        ),
        "metrics_sha256": (
            sha256_file(
                metrics_path
            )
        ),
        "run_manifest_sha256": (
            sha256_file(
                manifest_path
            )
        ),
        "run_status": "completed_new",
    }

    run_summaries.append(
        run_summary
    )

    print(
        "  completed | "
        f"best_epoch={best_epoch} | "
        "test_macro_f1="
        f"{test_primary['macro_f1']:.9f} | "
        "raw_weighted="
        f"{test_weighted['macro_f1']:.9f} | "
        "gafgyt_fnr="
        f"{test_primary['per_class']['gafgyt']['false_negative_rate']:.9f} | "
        "mirai_fnr="
        f"{test_primary['per_class']['mirai']['false_negative_rate']:.9f} | "
        "fragility="
        f"{fragility_triggered}",
        flush=True,
    )

    del model
    del B0_model
    del optimizer
    del criterion
    del source_checkpoint
    del B0_checkpoint
    del validation_predictions
    del validation_probabilities
    del test_predictions
    del test_probabilities

if len(
    run_summaries
) != 20:
    raise RuntimeError(
        "Expected twenty completed "
        "FP32 fine-tuning runs."
    )

run_summaries.sort(
    key=lambda row: (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    )
)

atomic_csv(
    AGGREGATE_RUNS_CSV,
    run_summaries,
    [
        "run_id",
        "architecture",
        "variant",
        "canonical_report_name",
        "source_variant",
        "seed",
        "best_epoch",
        "epochs_completed",
        "best_validation_macro_f1",
        "test_fingerprint_macro_f1",
        "test_fingerprint_accuracy",
        "test_raw_weighted_macro_f1",
        "test_gafgyt_fnr",
        "test_mirai_fnr",
        "parameter_count",
        "linear_macs_per_sample",
        "raw_tensor_bytes",
        "serialized_checkpoint_bytes",
        "training_elapsed_seconds",
        "test_evaluation_count",
        "fragility_gate_triggered",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "output_directory",
        "metrics_sha256",
        "run_manifest_sha256",
        "run_status",
    ],
)

group_rows = []
group_json = []

for architecture in ARCHITECTURES:
    for variant in VARIANTS:
        rows = [
            row
            for row in run_summaries
            if row["architecture"]
            == architecture
            and row["variant"]
            == variant
        ]

        if len(rows) != 5:
            raise RuntimeError(
                "Expected five runs for "
                f"{architecture} {variant}."
            )

        metric_names = (
            "best_validation_macro_f1",
            "test_fingerprint_accuracy",
            "test_fingerprint_macro_f1",
            "test_raw_weighted_macro_f1",
            "test_gafgyt_fnr",
            "test_mirai_fnr",
            "training_elapsed_seconds",
        )

        statistics = {
            metric_name: (
                aggregate_statistics(
                    [
                        float(
                            row[
                                metric_name
                            ]
                        )
                        for row in rows
                    ]
                )
            )
            for metric_name
            in metric_names
        }

        fragility_any = any(
            bool(
                row[
                    "fragility_gate_triggered"
                ]
            )
            for row in rows
        )

        group_row = {
            "architecture": (
                architecture
            ),
            "variant": variant,
            "canonical_report_name": (
                CANONICAL_REPORT_NAME[
                    variant
                ]
            ),
            "source_variant": (
                SOURCE_VARIANT[
                    variant
                ]
            ),
            "run_count": 5,
            "seeds": (
                json.dumps(
                    list(SEEDS),
                    separators=(
                        ",",
                        ":",
                    ),
                )
            ),
            "parameter_count": (
                rows[0][
                    "parameter_count"
                ]
            ),
            "linear_macs_per_sample": (
                rows[0][
                    "linear_macs_per_sample"
                ]
            ),
            "mean_best_validation_macro_f1": (
                statistics[
                    "best_validation_macro_f1"
                ]["mean"]
            ),
            "mean_test_fingerprint_accuracy": (
                statistics[
                    "test_fingerprint_accuracy"
                ]["mean"]
            ),
            "mean_test_fingerprint_macro_f1": (
                statistics[
                    "test_fingerprint_macro_f1"
                ]["mean"]
            ),
            "std_test_fingerprint_macro_f1": (
                statistics[
                    "test_fingerprint_macro_f1"
                ][
                    "std_population"
                ]
            ),
            "min_test_fingerprint_macro_f1": (
                statistics[
                    "test_fingerprint_macro_f1"
                ]["minimum"]
            ),
            "max_test_fingerprint_macro_f1": (
                statistics[
                    "test_fingerprint_macro_f1"
                ]["maximum"]
            ),
            "mean_test_raw_weighted_macro_f1": (
                statistics[
                    "test_raw_weighted_macro_f1"
                ]["mean"]
            ),
            "mean_test_gafgyt_fnr": (
                statistics[
                    "test_gafgyt_fnr"
                ]["mean"]
            ),
            "mean_test_mirai_fnr": (
                statistics[
                    "test_mirai_fnr"
                ]["mean"]
            ),
            "test_evaluation_count_per_run": 1,
            "fragility_triggered_in_any_run": (
                fragility_any
            ),
        }

        group_rows.append(
            group_row
        )

        group_json.append(
            {
                **group_row,
                "seeds": list(SEEDS),
                "aggregate_metrics": (
                    statistics
                ),
            }
        )

atomic_csv(
    GROUP_SUMMARY_CSV,
    group_rows,
    [
        "architecture",
        "variant",
        "canonical_report_name",
        "source_variant",
        "run_count",
        "seeds",
        "parameter_count",
        "linear_macs_per_sample",
        "mean_best_validation_macro_f1",
        "mean_test_fingerprint_accuracy",
        "mean_test_fingerprint_macro_f1",
        "std_test_fingerprint_macro_f1",
        "min_test_fingerprint_macro_f1",
        "max_test_fingerprint_macro_f1",
        "mean_test_raw_weighted_macro_f1",
        "mean_test_gafgyt_fnr",
        "mean_test_mirai_fnr",
        "test_evaluation_count_per_run",
        "fragility_triggered_in_any_run",
    ],
)

aggregate = {
    "status": "completed",
    "phase": 5,
    "artifact_name": (
        "P25_and_P50_FP32_fine_tuning"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "engine_version": (
        ENGINE_VERSION
    ),
    "completed_at_utc": utc_now(),
    "run_count": 20,
    "architecture_count": 2,
    "variant_count": 2,
    "seeds": list(SEEDS),
    "groups": group_json,
    "training_policy": {
        "optimizer": "AdamW",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "early_stopping_min_delta": (
            EARLY_STOPPING_MIN_DELTA
        ),
        "learning_rates": (
            FINE_TUNING_LEARNING_RATE
        ),
        "learning_rate_fraction_of_B0": (
            0.1
        ),
        "weight_decay": (
            WEIGHT_DECAY
        ),
        "validation_used_for_selection": True,
        "test_used_for_selection": False,
        "raw_occurrence_weight_used_for_fit": False,
    },
    "test_policy": {
        "test_evaluation_count_per_run": 1,
        "test_inference_total": 20,
        "test_used_for_selection": False,
    },
    "source_policy": {
        "P25_uses_locked_P25_noFT_source": True,
        "P50_uses_locked_P50_noFT_source": True,
        "sources_generated_directly_from_matching_B0": True,
    },
    "fragility_triggered_in_any_run": (
        any(
            bool(
                row[
                    "fragility_gate_triggered"
                ]
            )
            for row in run_summaries
        )
    ),
    "entry_checks": (
        entry_checks
    ),
    "shape_checks": (
        shape_checks
    ),
    "aggregate_runs_csv": (
        file_record(
            AGGREGATE_RUNS_CSV
        )
    ),
    "group_summary_csv": (
        file_record(
            GROUP_SUMMARY_CSV
        )
    ),
    "ready_for_independent_verification": True,
    "all_integrity_checks_passed": True,
}

atomic_json(
    AGGREGATE_JSON,
    aggregate,
)

completion = {
    "status": "completed",
    "phase": 5,
    "artifact_name": (
        "P25_and_P50_FP32_fine_tuning"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "completed_at_utc": utc_now(),
    "run_count": 20,
    "test_evaluation_count_per_run": 1,
    "test_inference_total": 20,
    "test_used_for_selection": False,
    "aggregate_json": str(
        AGGREGATE_JSON
    ),
    "aggregate_json_sha256": (
        sha256_file(
            AGGREGATE_JSON
        )
    ),
    "aggregate_runs_csv": str(
        AGGREGATE_RUNS_CSV
    ),
    "aggregate_runs_csv_sha256": (
        sha256_file(
            AGGREGATE_RUNS_CSV
        )
    ),
    "group_summary_csv": str(
        GROUP_SUMMARY_CSV
    ),
    "group_summary_csv_sha256": (
        sha256_file(
            GROUP_SUMMARY_CSV
        )
    ),
    "ready_for_independent_verification": True,
    "all_checks_passed": True,
}

atomic_json(
    COMPLETION_JSON,
    completion,
)

print()
print("=" * 92)
print("PHASE 5 P25 AND P50 FP32 FINE-TUNING SUMMARY")
print("=" * 92)
print(
    "Runs completed                  : 20"
)
print(
    "Test evaluation count per run   : 1"
)
print(
    "Test used for selection         : False"
)
print(
    "Fine-tuning learning-rate ratio : 0.1 x B0"
)
print()

for row in sorted(
    group_rows,
    key=lambda value: (
        value["architecture"],
        value["variant"],
    ),
):
    print(
        f"{row['architecture']:<16} "
        f"{row['canonical_report_name']:<13} | "
        "mean Macro-F1="
        f"{float(row['mean_test_fingerprint_macro_f1']):.9f} | "
        "std="
        f"{float(row['std_test_fingerprint_macro_f1']):.9f} | "
        "raw-weighted="
        f"{float(row['mean_test_raw_weighted_macro_f1']):.9f} | "
        "parameters="
        f"{int(row['parameter_count']):,} | "
        "MACs/sample="
        f"{int(row['linear_macs_per_sample']):,} | "
        "fragility="
        f"{row['fragility_triggered_in_any_run']}"
    )

print()
print(
    "Aggregate runs CSV              : "
    f"{AGGREGATE_RUNS_CSV}"
)
print(
    "Group summary CSV               : "
    f"{GROUP_SUMMARY_CSV}"
)
print(
    "Aggregate JSON                  : "
    f"{AGGREGATE_JSON}"
)
print(
    "Ready for independent verify    : True"
)
print(
    "All integrity checks passed     : True"
)
print(
    "PHASE 5 P25 AND P50 FP32 FINE-TUNING COMPLETED"
)
