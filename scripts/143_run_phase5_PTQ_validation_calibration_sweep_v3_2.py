from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import shutil
import sys
import time
import warnings
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

X_VALIDATION_PATH = FINAL_CACHE / "X_validation.npy"
Y_VALIDATION_PATH = FINAL_CACHE / "y_validation.npy"
RAW_VALIDATION_PATH = (
    FINAL_CACHE / "raw_row_count_validation.npy"
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

B0_PAIR_LOCK = (
    AUDIT / "phase5_B0_pair_locked_v3_2.json"
)

B0_CHECKPOINT_REGISTRY = (
    AUDIT / "phase5_B0_checkpoint_registry_v3_2.csv"
)

QAT_LOCK = (
    AUDIT / "phase5_QAT_locked_v3_2.json"
)

PTQ_PREFLIGHT_LOCK = (
    AUDIT
    / "phase5_PTQ_runtime_preflight_locked_v3_2.json"
)

PTQ_PREFLIGHT_MATRIX = (
    AUDIT
    / "phase5_PTQ_runtime_preflight_matrix_v3_2.csv"
)

CALIBRATION_INDEX_NPZ = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "shared"
    / "ptq"
    / "phase5_PTQ_calibration_indices_v3_2.npz"
)

MODEL_SOURCE = (
    ROOT / "src" / "models" / "nbaiot_models.py"
)

PTQ_ENGINE_PATH = (
    ROOT
    / "src"
    / "compression"
    / "phase5_ptq_engine_v3_2.py"
)

SCALER_NPZ = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "shared"
    / "preprocessing"
    / "phase5_train_only_standard_scaler_v3_2.npz"
)

AGGREGATE_CANDIDATES_CSV = (
    AUDIT
    / "phase5_PTQ_validation_sweep_all_candidates_v3_2.csv"
)

GROUP_SUMMARY_CSV = (
    AUDIT
    / "phase5_PTQ_validation_sweep_group_summary_v3_2.csv"
)

SELECTION_CSV = (
    AUDIT
    / "phase5_PTQ_validation_selection_v3_2.csv"
)

AGGREGATE_JSON = (
    AUDIT
    / "phase5_PTQ_validation_sweep_summary_v3_2.json"
)

COMPLETION_JSON = (
    AUDIT
    / "phase5_PTQ_validation_sweep_completed_v3_2.json"
)

PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
PTQ_ENGINE_VERSION = "phase5_ptq_engine_v3_2"

ARCHITECTURES = (
    "tinyml_mlp",
    "compact_dnn",
)

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

CALIBRATION_SIZES = (
    4096,
    16384,
    65536,
    262144,
)

EXPECTED_KEYS = {
    (
        architecture,
        seed,
        calibration_size,
    )
    for architecture in ARCHITECTURES
    for seed in SEEDS
    for calibration_size in CALIBRATION_SIZES
}

EXPECTED_B0_KEYS = {
    (
        architecture,
        seed,
    )
    for architecture in ARCHITECTURES
    for seed in SEEDS
}

EXPECTED_INPUT_FEATURES = 115
EXPECTED_OUTPUT_CLASSES = 3

HIDDEN_DIMS = {
    "tinyml_mlp": [64, 32],
    "compact_dnn": [128, 64, 32],
}

EXPECTED_WIDTHS = {
    "tinyml_mlp": [
        [115, 64],
        [64, 32],
        [32, 3],
    ],
    "compact_dnn": [
        [115, 128],
        [128, 64],
        [64, 32],
        [32, 3],
    ],
}

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
EXPECTED_VALIDATION_RAW_ROWS = 1_059_390

CALIBRATION_BATCH_SIZE = 16_384
EVAL_BATCH_SIZE = 16_384
CPU_THREADS = 4
MINIMUM_FREE_DISK_GIB = 1.5

PRIMARY_SELECTION_METRIC = (
    "mean_validation_fingerprint_macro_f1"
)

SELECTION_TIE_BREAKS = (
    "lower_std_population_then_smaller_calibration_size"
)


warnings.filterwarnings(
    "ignore",
    message="torch.ao.quantization is deprecated.*",
    category=DeprecationWarning,
)

warnings.filterwarnings(
    "ignore",
    message="Please use quant_min and quant_max.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "Default qconfig of oneDNN backend.*"
    ),
    category=UserWarning,
)


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
                EXPECTED_INPUT_FEATURES
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
                EXPECTED_OUTPUT_CLASSES
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


def instantiate_model(
    model_module: ModuleType,
    model_symbol: str,
    architecture: str,
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


def transform_rows(
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


def calibrate_model(
    prepared_model: nn.Module,
    x_train: np.ndarray,
    calibration_indices: np.ndarray,
    mean64: np.ndarray,
    scale64: np.ndarray,
    PTQ_engine_module: ModuleType,
) -> float:
    prepared_model.eval()

    started = time.perf_counter()

    with torch.inference_mode():
        for start in range(
            0,
            len(calibration_indices),
            CALIBRATION_BATCH_SIZE,
        ):
            end = min(
                start + CALIBRATION_BATCH_SIZE,
                len(calibration_indices),
            )

            batch_indices = (
                calibration_indices[
                    start:end
                ]
            )

            x_batch = torch.from_numpy(
                transform_rows(
                    x_train[
                        batch_indices
                    ],
                    mean64,
                    scale64,
                )
            )

            PTQ_engine_module.extract_logits(
                prepared_model(
                    x_batch
                )
            )

    return (
        time.perf_counter()
        - started
    )


def evaluate_validation(
    model: nn.Module,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    mean64: np.ndarray,
    scale64: np.ndarray,
    PTQ_engine_module: ModuleType,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    model.eval()

    y_true = np.asarray(
        y_validation,
        dtype=np.int64,
    )

    predictions = np.empty(
        len(y_true),
        dtype=np.int8,
    )

    probabilities = np.empty(
        (
            len(y_true),
            EXPECTED_OUTPUT_CLASSES,
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
                transform_rows(
                    x_validation[
                        start:end
                    ],
                    mean64,
                    scale64,
                )
            )

            logits = (
                PTQ_engine_module
                .extract_logits(
                    model(
                        x_batch
                    )
                )
            )

            if tuple(
                logits.shape
            ) != (
                end - start,
                EXPECTED_OUTPUT_CLASSES,
            ):
                raise RuntimeError(
                    "Validation output shape mismatch."
                )

            batch_probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )
            )

            probabilities[
                start:end
            ] = (
                batch_probabilities
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            predictions[
                start:end
            ] = (
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


def output_directory(
    architecture: str,
    seed: int,
    calibration_size: int,
) -> Path:
    return (
        ROOT
        / "results"
        / "v2"
        / "phase5_compression"
        / architecture
        / "ptq_validation_sweep"
        / f"calibration_{calibration_size}"
        / f"seed_{seed}"
    )


def artifact_inventory(
    run_directory: Path,
) -> list[dict[str, Any]]:
    excluded = {
        "run_manifest.json",
        "run_status.json",
    }

    return [
        file_record(
            path,
            relative_to=run_directory,
        )
        for path in sorted(
            run_directory.iterdir()
        )
        if path.is_file()
        and path.name not in excluded
    ]


def build_completed_summary(
    run_directory: Path,
) -> dict[str, Any]:
    metrics_path = (
        run_directory / "metrics.json"
    )

    manifest_path = (
        run_directory / "run_manifest.json"
    )

    status_path = (
        run_directory / "run_status.json"
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
                "test_access_count"
            )
        )
        == 0
        and int(
            metrics.get(
                "test_access_count"
            )
        )
        == 0
    ):
        raise RuntimeError(
            "Existing PTQ candidate failed "
            f"resume validation: {run_directory}"
        )

    return {
        "run_id": metrics["run_id"],
        "architecture": (
            metrics["architecture"]
        ),
        "seed": int(metrics["seed"]),
        "calibration_size": int(
            metrics["calibration"][
                "size"
            ]
        ),
        "validation_fingerprint_macro_f1": float(
            metrics["validation"][
                "primary_fingerprint_level"
            ]["macro_f1"]
        ),
        "validation_fingerprint_accuracy": float(
            metrics["validation"][
                "primary_fingerprint_level"
            ]["accuracy"]
        ),
        "validation_raw_weighted_macro_f1": float(
            metrics["validation"][
                "secondary_raw_record_weighted"
            ]["macro_f1"]
        ),
        "validation_gafgyt_fnr": float(
            metrics["validation"][
                "primary_fingerprint_level"
            ]["per_class"][
                "gafgyt"
            ][
                "false_negative_rate"
            ]
        ),
        "validation_mirai_fnr": float(
            metrics["validation"][
                "primary_fingerprint_level"
            ]["per_class"][
                "mirai"
            ][
                "false_negative_rate"
            ]
        ),
        "float_state_bytes": int(
            metrics["model_complexity"][
                "float_state_bytes"
            ]
        ),
        "INT8_state_bytes": int(
            metrics["model_complexity"][
                "INT8_state_bytes"
            ]
        ),
        "INT8_to_float_state_size_ratio": float(
            metrics["model_complexity"][
                "INT8_to_float_state_size_ratio"
            ]
        ),
        "calibration_elapsed_seconds": float(
            metrics["calibration"][
                "elapsed_seconds"
            ]
        ),
        "validation_elapsed_seconds": float(
            metrics["validation"][
                "elapsed_seconds"
            ]
        ),
        "validation_access_count": 1,
        "test_access_count": 0,
        "source_checkpoint_path": (
            metrics["source"][
                "source_B0_checkpoint_path"
            ]
        ),
        "source_checkpoint_sha256": (
            metrics["source"][
                "source_B0_checkpoint_sha256"
            ]
        ),
        "calibration_indices_sha256": (
            metrics["calibration"][
                "indices_sha256"
            ]
        ),
        "output_directory": str(
            run_directory
        ),
        "INT8_checkpoint_sha256": (
            sha256_file(
                run_directory
                / "ptq_int8_checkpoint.pt"
            )
        ),
        "validation_predictions_sha256": (
            sha256_file(
                run_directory
                / "validation_predictions.npz"
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
        "run_status": (
            "resumed_completed"
        ),
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
    X_TRAIN_PATH,
    X_VALIDATION_PATH,
    Y_VALIDATION_PATH,
    RAW_VALIDATION_PATH,
    PHASE5_PROTOCOL,
    LOCKED_MATRIX,
    B0_PAIR_LOCK,
    B0_CHECKPOINT_REGISTRY,
    QAT_LOCK,
    PTQ_PREFLIGHT_LOCK,
    PTQ_PREFLIGHT_MATRIX,
    CALIBRATION_INDEX_NPZ,
    MODEL_SOURCE,
    PTQ_ENGINE_PATH,
    SCALER_NPZ,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for aggregate_path in (
    AGGREGATE_CANDIDATES_CSV,
    GROUP_SUMMARY_CSV,
    SELECTION_CSV,
    AGGREGATE_JSON,
    COMPLETION_JSON,
):
    if aggregate_path.exists():
        raise FileExistsError(
            "PTQ validation-sweep aggregate "
            "artifact already exists; refusing "
            f"to overwrite: {aggregate_path}"
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

B0_pair_lock = read_json(
    B0_PAIR_LOCK
)

QAT_lock = read_json(
    QAT_LOCK
)

PTQ_preflight_lock = read_json(
    PTQ_PREFLIGHT_LOCK
)

matrix_rows = read_csv(
    LOCKED_MATRIX
)

B0_rows = read_csv(
    B0_CHECKPOINT_REGISTRY
)

PTQ_preflight_rows = read_csv(
    PTQ_PREFLIGHT_MATRIX
)

PTQ_matrix_keys = {
    (
        row["architecture"],
        int(row["seed"]),
    )
    for row in matrix_rows
    if row["architecture"]
    in ARCHITECTURES
    and row["variant"]
    == "PTQ"
}

B0_lookup = {
    (
        row["architecture"],
        int(row["seed"]),
    ): row
    for row in B0_rows
}

preflight_lookup = {
    (
        row["architecture"],
        int(row["seed"]),
    ): row
    for row in PTQ_preflight_rows
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
    "B0_pair_locked": (
        B0_pair_lock.get("status")
        == "locked"
        and B0_pair_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "B0_registry_hash_matches": (
        B0_pair_lock.get(
            "checkpoint_registry_sha256"
        )
        == sha256_file(
            B0_CHECKPOINT_REGISTRY
        )
    ),
    "QAT_locked": (
        QAT_lock.get("status")
        == "locked"
        and QAT_lock.get(
            "ready_for_PTQ_branch"
        )
        is True
        and QAT_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "PTQ_preflight_locked": (
        PTQ_preflight_lock.get(
            "status"
        )
        == "locked"
        and PTQ_preflight_lock.get(
            "ready_for_PTQ_validation_sweep"
        )
        is True
        and PTQ_preflight_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "PTQ_engine_hash_matches": (
        PTQ_preflight_lock.get(
            "PTQ_engine_source_sha256"
        )
        == sha256_file(
            PTQ_ENGINE_PATH
        )
    ),
    "calibration_indices_hash_matches": (
        PTQ_preflight_lock.get(
            "calibration_index_npz_sha256"
        )
        == sha256_file(
            CALIBRATION_INDEX_NPZ
        )
    ),
    "PTQ_preflight_matrix_hash_matches": (
        PTQ_preflight_lock.get(
            "matrix_csv_sha256"
        )
        == sha256_file(
            PTQ_PREFLIGHT_MATRIX
        )
    ),
    "PTQ_matrix_complete": (
        PTQ_matrix_keys
        == EXPECTED_B0_KEYS
    ),
    "B0_registry_complete": (
        set(
            B0_lookup.keys()
        )
        == EXPECTED_B0_KEYS
    ),
    "PTQ_preflight_matrix_complete": (
        set(
            preflight_lookup.keys()
        )
        == EXPECTED_B0_KEYS
    ),
    "calibration_sizes_match_lock": (
        tuple(
            PTQ_preflight_lock[
                "calibration_sizes"
            ]
        )
        == CALIBRATION_SIZES
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
        "PTQ validation-sweep entry gate "
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

selected_backend = str(
    PTQ_preflight_lock[
        "selected_backend"
    ]
)

if (
    selected_backend
    not in torch.backends.quantized.supported_engines
):
    raise RuntimeError(
        "Locked PTQ backend is not "
        "supported by the current runtime."
    )

torch.backends.quantized.engine = (
    selected_backend
)

x_train = np.load(
    X_TRAIN_PATH,
    mmap_mode="r",
)

x_validation = np.load(
    X_VALIDATION_PATH,
    mmap_mode="r",
)

y_validation = np.asarray(
    np.load(
        Y_VALIDATION_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

raw_validation = np.asarray(
    np.load(
        RAW_VALIDATION_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

if (
    x_train.shape
    != (
        EXPECTED_TRAIN_ROWS,
        EXPECTED_INPUT_FEATURES,
    )
):
    raise RuntimeError(
        "PTQ train cache shape mismatch."
    )

if not (
    x_validation.shape
    == (
        EXPECTED_VALIDATION_ROWS,
        EXPECTED_INPUT_FEATURES,
    )
    and y_validation.shape
    == (
        EXPECTED_VALIDATION_ROWS,
    )
    and raw_validation.shape
    == (
        EXPECTED_VALIDATION_ROWS,
    )
):
    raise RuntimeError(
        "PTQ validation cache shape mismatch."
    )

if int(
    raw_validation.sum()
) != EXPECTED_VALIDATION_RAW_ROWS:
    raise RuntimeError(
        "PTQ validation raw-row total mismatch."
    )

with np.load(
    SCALER_NPZ
) as values:
    mean64 = np.array(
        values["mean_float64"],
        copy=True,
    )

    scale64 = np.array(
        values["scale_float64"],
        copy=True,
    )

if (
    mean64.shape
    != (
        EXPECTED_INPUT_FEATURES,
    )
    or scale64.shape
    != (
        EXPECTED_INPUT_FEATURES,
    )
):
    raise RuntimeError(
        "PTQ scaler shape mismatch."
    )

calibration_archive = np.load(
    CALIBRATION_INDEX_NPZ
)

expected_calibration_keys = {
    f"seed_{seed}_size_{size}"
    for seed in SEEDS
    for size in CALIBRATION_SIZES
}

if set(
    calibration_archive.files
) != expected_calibration_keys:
    raise RuntimeError(
        "PTQ calibration archive key mismatch."
    )

model_module = load_module(
    MODEL_SOURCE,
    "phase5_PTQ_validation_models",
)

PTQ_engine_module = load_module(
    PTQ_ENGINE_PATH,
    "phase5_PTQ_validation_engine",
)

if (
    PTQ_engine_module.ENGINE_VERSION
    != PTQ_ENGINE_VERSION
):
    raise RuntimeError(
        "PTQ-engine version mismatch."
    )

print("=" * 92)
print("PHASE 5 PTQ VALIDATION CALIBRATION SWEEP")
print("=" * 92)
print(
    "Candidates                      : 40"
)
print(
    "Architectures                   : "
    f"{list(ARCHITECTURES)}"
)
print(
    "Seeds                           : "
    f"{list(SEEDS)}"
)
print(
    "Calibration sizes               : "
    f"{list(CALIBRATION_SIZES)}"
)
print(
    "PTQ backend                     : "
    f"{selected_backend}"
)
print(
    "Selection unit                  : "
    "architecture-level mean across 5 seeds"
)
print(
    "Primary selection metric        : "
    f"{PRIMARY_SELECTION_METRIC}"
)
print(
    "Tie-breaks                      : "
    f"{SELECTION_TIE_BREAKS}"
)
print(
    "Test access count               : 0"
)
print(
    "Free disk                       : "
    f"{free_disk_gib:.3f} GiB"
)
print()

candidate_rows: list[
    dict[str, Any]
] = []

ordered_keys = sorted(
    EXPECTED_KEYS,
    key=lambda value: (
        value[0],
        value[1],
        value[2],
    ),
)

for candidate_index, (
    architecture,
    seed,
    calibration_size,
) in enumerate(
    ordered_keys,
    start=1,
):
    run_id = (
        f"{architecture}__ptq_validation__"
        f"seed_{seed}__cal_{calibration_size}"
    )

    run_directory = output_directory(
        architecture,
        seed,
        calibration_size,
    )

    status_path = (
        run_directory / "run_status.json"
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
                build_completed_summary(
                    run_directory
                )
            )

            candidate_rows.append(
                summary
            )

            print(
                f"[{candidate_index}/40] "
                f"{run_id} | "
                "status=resumed_completed | "
                "test access=0",
                flush=True,
            )

            continue

        shutil.rmtree(
            run_directory
        )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    B0_row = B0_lookup[
        (
            architecture,
            seed,
        )
    ]

    preflight_row = (
        preflight_lookup[
            (
                architecture,
                seed,
            )
        ]
    )

    checkpoint_path = Path(
        B0_row["checkpoint_path"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint_hash = sha256_file(
        checkpoint_path
    )

    if not (
        checkpoint_hash
        == B0_row[
            "checkpoint_sha256"
        ]
        == preflight_row[
            "source_checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            f"{run_id}: source checkpoint "
            "hash mismatch."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_symbol = (
        B0_row["model_symbol"]
    )

    source_model = instantiate_model(
        model_module,
        model_symbol,
        architecture,
    )

    source_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    source_model.eval()

    if (
        PTQ_engine_module
        .float_linear_widths(
            source_model
        )
        != EXPECTED_WIDTHS[
            architecture
        ]
    ):
        raise RuntimeError(
            f"{run_id}: source topology mismatch."
        )

    calibration_key = (
        f"seed_{seed}_"
        f"size_{calibration_size}"
    )

    calibration_indices = np.asarray(
        calibration_archive[
            calibration_key
        ],
        dtype=np.int64,
    )

    if (
        calibration_indices.shape
        != (
            calibration_size,
        )
        or len(
            np.unique(
                calibration_indices
            )
        )
        != calibration_size
        or int(
            calibration_indices.min()
        )
        < 0
        or int(
            calibration_indices.max()
        )
        >= EXPECTED_TRAIN_ROWS
    ):
        raise RuntimeError(
            f"{run_id}: calibration "
            "indices are invalid."
        )

    calibration_indices_digest = (
        hashlib.sha256(
            calibration_indices.tobytes()
        ).hexdigest()
    )

    atomic_json(
        status_path,
        {
            "status": "running",
            "stage": "calibration",
            "run_id": run_id,
            "architecture": architecture,
            "seed": seed,
            "calibration_size": (
                calibration_size
            ),
            "validation_access_count": 0,
            "test_access_count": 0,
            "started_at_utc": utc_now(),
        },
    )

    prepared_model = (
        PTQ_engine_module
        .prepare_ptq_model(
            source_model,
            selected_backend,
        )
    )

    calibration_elapsed = (
        calibrate_model(
            prepared_model,
            x_train,
            calibration_indices,
            mean64,
            scale64,
            PTQ_engine_module,
        )
    )

    INT8_model = (
        PTQ_engine_module
        .convert_ptq_model(
            prepared_model
        )
    )

    INT8_model.eval()

    expected_linear_count = len(
        EXPECTED_WIDTHS[
            architecture
        ]
    )

    INT8_linear_count = len(
        PTQ_engine_module
        .static_quantized_linear_modules(
            INT8_model
        )
    )

    if (
        INT8_linear_count
        != expected_linear_count
    ):
        raise RuntimeError(
            f"{run_id}: not all Linear "
            "layers were converted."
        )

    weight_dtypes = (
        PTQ_engine_module
        .quantized_weight_dtypes(
            INT8_model
        )
    )

    if not (
        len(weight_dtypes)
        == expected_linear_count
        and all(
            dtype == "torch.qint8"
            for dtype in weight_dtypes
        )
    ):
        raise RuntimeError(
            f"{run_id}: unexpected INT8 "
            "weight dtype."
        )

    import io

    float_buffer = io.BytesIO()

    torch.save(
        source_model.state_dict(),
        float_buffer,
    )

    float_state_bytes = len(
        float_buffer.getvalue()
    )

    int8_buffer = io.BytesIO()

    torch.save(
        INT8_model.state_dict(),
        int8_buffer,
    )

    INT8_state_bytes = len(
        int8_buffer.getvalue()
    )

    INT8_checkpoint_path = (
        run_directory
        / "ptq_int8_checkpoint.pt"
    )

    atomic_torch_save(
        {
            "artifact_type": (
                "phase5_PTQ_validation_candidate"
            ),
            "protocol_version": (
                PROTOCOL_VERSION
            ),
            "PTQ_engine_version": (
                PTQ_ENGINE_VERSION
            ),
            "run_id": run_id,
            "architecture": architecture,
            "seed": seed,
            "calibration_size": (
                calibration_size
            ),
            "calibration_key": (
                calibration_key
            ),
            "calibration_indices_sha256": (
                calibration_indices_digest
            ),
            "model_symbol": (
                model_symbol
            ),
            "PTQ_backend": (
                selected_backend
            ),
            "source_B0_checkpoint_path": (
                str(
                    checkpoint_path
                )
            ),
            "source_B0_checkpoint_sha256": (
                checkpoint_hash
            ),
            "INT8_model_state_dict": (
                INT8_model.state_dict()
            ),
            "float_linear_widths": (
                EXPECTED_WIDTHS[
                    architecture
                ]
            ),
            "float_state_bytes": (
                float_state_bytes
            ),
            "INT8_state_bytes": (
                INT8_state_bytes
            ),
            "validation_only_candidate": True,
            "test_access_count": 0,
            "created_at_utc": utc_now(),
        },
        INT8_checkpoint_path,
    )

    atomic_json(
        status_path,
        {
            "status": "running",
            "stage": "validation",
            "run_id": run_id,
            "architecture": architecture,
            "seed": seed,
            "calibration_size": (
                calibration_size
            ),
            "validation_access_count": 1,
            "test_access_count": 0,
            "INT8_checkpoint_sha256": (
                sha256_file(
                    INT8_checkpoint_path
                )
            ),
            "updated_at_utc": utc_now(),
        },
    )

    (
        validation_predictions,
        validation_probabilities,
        validation_true,
        validation_elapsed,
    ) = evaluate_validation(
        INT8_model,
        x_validation,
        y_validation,
        mean64,
        scale64,
        PTQ_engine_module,
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

    metrics_path = (
        run_directory / "metrics.json"
    )

    metrics = {
        "status": "completed",
        "phase": 5,
        "artifact_name": (
            "PTQ_validation_candidate"
        ),
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "PTQ_engine_version": (
            PTQ_ENGINE_VERSION
        ),
        "run_id": run_id,
        "architecture": architecture,
        "seed": seed,
        "source": {
            "source_variant": "B0",
            "source_B0_checkpoint_path": (
                str(
                    checkpoint_path
                )
            ),
            "source_B0_checkpoint_sha256": (
                checkpoint_hash
            ),
        },
        "calibration": {
            "source_split": "train_only",
            "size": (
                calibration_size
            ),
            "key": (
                calibration_key
            ),
            "indices_sha256": (
                calibration_indices_digest
            ),
            "sampling": (
                "deterministic_stratified_nested"
            ),
            "elapsed_seconds": (
                calibration_elapsed
            ),
        },
        "quantization": {
            "method": (
                "post_training_static_quantization"
            ),
            "backend": (
                selected_backend
            ),
            "target_modules": [
                "torch.nn.Linear",
            ],
            "quantized_weight_dtype": (
                "torch.qint8"
            ),
            "training_performed": False,
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
            "used_for_calibration_size_selection": (
                True
            ),
        },
        "model_complexity": {
            "float_linear_widths": (
                EXPECTED_WIDTHS[
                    architecture
                ]
            ),
            "float_linear_count": (
                expected_linear_count
            ),
            "INT8_linear_count": (
                INT8_linear_count
            ),
            "float_state_bytes": (
                float_state_bytes
            ),
            "INT8_state_bytes": (
                INT8_state_bytes
            ),
            "INT8_to_float_state_size_ratio": (
                INT8_state_bytes
                / float_state_bytes
            ),
        },
        "validation_access_count": 1,
        "test_access_count": 0,
        "test_used_for_selection": False,
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
        "artifact_name": (
            "PTQ_validation_candidate"
        ),
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "PTQ_engine_version": (
            PTQ_ENGINE_VERSION
        ),
        "run_id": run_id,
        "architecture": architecture,
        "seed": seed,
        "calibration_size": (
            calibration_size
        ),
        "model_symbol": (
            model_symbol
        ),
        "source_B0_checkpoint": (
            file_record(
                checkpoint_path
            )
        ),
        "PTQ_preflight_lock": (
            file_record(
                PTQ_PREFLIGHT_LOCK
            )
        ),
        "calibration_index_archive": (
            file_record(
                CALIBRATION_INDEX_NPZ
            )
        ),
        "PTQ_engine": (
            file_record(
                PTQ_ENGINE_PATH
            )
        ),
        "scaler": (
            file_record(
                SCALER_NPZ
            )
        ),
        "selection_scope": {
            "validation_used_for_selection": (
                True
            ),
            "test_used_for_selection": (
                False
            ),
            "test_access_count": 0,
        },
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
            "seed": seed,
            "calibration_size": (
                calibration_size
            ),
            "validation_access_count": 1,
            "test_access_count": 0,
            "INT8_checkpoint_sha256": (
                sha256_file(
                    INT8_checkpoint_path
                )
            ),
            "validation_predictions_sha256": (
                sha256_file(
                    validation_predictions_path
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
            "completed_at_utc": utc_now(),
        },
    )

    candidate_rows.append(
        {
            "run_id": run_id,
            "architecture": architecture,
            "seed": seed,
            "calibration_size": (
                calibration_size
            ),
            "validation_fingerprint_macro_f1": (
                validation_primary[
                    "macro_f1"
                ]
            ),
            "validation_fingerprint_accuracy": (
                validation_primary[
                    "accuracy"
                ]
            ),
            "validation_raw_weighted_macro_f1": (
                validation_weighted[
                    "macro_f1"
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
            "float_state_bytes": (
                float_state_bytes
            ),
            "INT8_state_bytes": (
                INT8_state_bytes
            ),
            "INT8_to_float_state_size_ratio": (
                INT8_state_bytes
                / float_state_bytes
            ),
            "calibration_elapsed_seconds": (
                calibration_elapsed
            ),
            "validation_elapsed_seconds": (
                validation_elapsed
            ),
            "validation_access_count": 1,
            "test_access_count": 0,
            "source_checkpoint_path": (
                str(
                    checkpoint_path
                )
            ),
            "source_checkpoint_sha256": (
                checkpoint_hash
            ),
            "calibration_indices_sha256": (
                calibration_indices_digest
            ),
            "output_directory": (
                str(
                    run_directory
                )
            ),
            "INT8_checkpoint_sha256": (
                sha256_file(
                    INT8_checkpoint_path
                )
            ),
            "validation_predictions_sha256": (
                sha256_file(
                    validation_predictions_path
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
    )

    print(
        f"[{candidate_index}/40] "
        f"{architecture} seed={seed} "
        f"cal={calibration_size} | "
        "val_macro_f1="
        f"{validation_primary['macro_f1']:.9f} | "
        "raw_weighted="
        f"{validation_weighted['macro_f1']:.9f} | "
        "state_ratio="
        f"{INT8_state_bytes / float_state_bytes:.4f} | "
        "test access=0",
        flush=True,
    )

    del source_model
    del prepared_model
    del INT8_model
    del checkpoint
    del validation_predictions
    del validation_probabilities

calibration_archive.close()

if len(
    candidate_rows
) != 40:
    raise RuntimeError(
        "Expected forty completed PTQ "
        "validation candidates."
    )

candidate_rows.sort(
    key=lambda row: (
        row["architecture"],
        int(row["calibration_size"]),
        int(row["seed"]),
    )
)

atomic_csv(
    AGGREGATE_CANDIDATES_CSV,
    candidate_rows,
    [
        "run_id",
        "architecture",
        "seed",
        "calibration_size",
        "validation_fingerprint_macro_f1",
        "validation_fingerprint_accuracy",
        "validation_raw_weighted_macro_f1",
        "validation_gafgyt_fnr",
        "validation_mirai_fnr",
        "float_state_bytes",
        "INT8_state_bytes",
        "INT8_to_float_state_size_ratio",
        "calibration_elapsed_seconds",
        "validation_elapsed_seconds",
        "validation_access_count",
        "test_access_count",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "calibration_indices_sha256",
        "output_directory",
        "INT8_checkpoint_sha256",
        "validation_predictions_sha256",
        "metrics_sha256",
        "run_manifest_sha256",
        "run_status",
    ],
)

group_rows: list[
    dict[str, Any]
] = []

group_json: list[
    dict[str, Any]
] = []

for architecture in ARCHITECTURES:
    for calibration_size in CALIBRATION_SIZES:
        rows = [
            row
            for row in candidate_rows
            if row["architecture"]
            == architecture
            and int(
                row[
                    "calibration_size"
                ]
            )
            == calibration_size
        ]

        if len(rows) != 5:
            raise RuntimeError(
                "Expected five PTQ validation "
                f"candidates for {architecture} "
                f"calibration={calibration_size}."
            )

        metrics = {
            "validation_fingerprint_macro_f1": (
                aggregate_statistics(
                    [
                        float(
                            row[
                                "validation_fingerprint_macro_f1"
                            ]
                        )
                        for row in rows
                    ]
                )
            ),
            "validation_fingerprint_accuracy": (
                aggregate_statistics(
                    [
                        float(
                            row[
                                "validation_fingerprint_accuracy"
                            ]
                        )
                        for row in rows
                    ]
                )
            ),
            "validation_raw_weighted_macro_f1": (
                aggregate_statistics(
                    [
                        float(
                            row[
                                "validation_raw_weighted_macro_f1"
                            ]
                        )
                        for row in rows
                    ]
                )
            ),
            "validation_gafgyt_fnr": (
                aggregate_statistics(
                    [
                        float(
                            row[
                                "validation_gafgyt_fnr"
                            ]
                        )
                        for row in rows
                    ]
                )
            ),
            "validation_mirai_fnr": (
                aggregate_statistics(
                    [
                        float(
                            row[
                                "validation_mirai_fnr"
                            ]
                        )
                        for row in rows
                    ]
                )
            ),
            "calibration_elapsed_seconds": (
                aggregate_statistics(
                    [
                        float(
                            row[
                                "calibration_elapsed_seconds"
                            ]
                        )
                        for row in rows
                    ]
                )
            ),
        }

        group_row = {
            "architecture": architecture,
            "calibration_size": (
                calibration_size
            ),
            "run_count": 5,
            "seeds": json.dumps(
                list(SEEDS),
                separators=(
                    ",",
                    ":",
                ),
            ),
            "mean_validation_fingerprint_macro_f1": (
                metrics[
                    "validation_fingerprint_macro_f1"
                ]["mean"]
            ),
            "std_validation_fingerprint_macro_f1": (
                metrics[
                    "validation_fingerprint_macro_f1"
                ]["std_population"]
            ),
            "min_validation_fingerprint_macro_f1": (
                metrics[
                    "validation_fingerprint_macro_f1"
                ]["minimum"]
            ),
            "max_validation_fingerprint_macro_f1": (
                metrics[
                    "validation_fingerprint_macro_f1"
                ]["maximum"]
            ),
            "mean_validation_fingerprint_accuracy": (
                metrics[
                    "validation_fingerprint_accuracy"
                ]["mean"]
            ),
            "mean_validation_raw_weighted_macro_f1": (
                metrics[
                    "validation_raw_weighted_macro_f1"
                ]["mean"]
            ),
            "mean_validation_gafgyt_fnr": (
                metrics[
                    "validation_gafgyt_fnr"
                ]["mean"]
            ),
            "mean_validation_mirai_fnr": (
                metrics[
                    "validation_mirai_fnr"
                ]["mean"]
            ),
            "mean_calibration_elapsed_seconds": (
                metrics[
                    "calibration_elapsed_seconds"
                ]["mean"]
            ),
            "test_access_count": 0,
        }

        group_rows.append(
            group_row
        )

        group_json.append(
            {
                **group_row,
                "seeds": list(SEEDS),
                "aggregate_metrics": (
                    metrics
                ),
            }
        )

atomic_csv(
    GROUP_SUMMARY_CSV,
    group_rows,
    [
        "architecture",
        "calibration_size",
        "run_count",
        "seeds",
        "mean_validation_fingerprint_macro_f1",
        "std_validation_fingerprint_macro_f1",
        "min_validation_fingerprint_macro_f1",
        "max_validation_fingerprint_macro_f1",
        "mean_validation_fingerprint_accuracy",
        "mean_validation_raw_weighted_macro_f1",
        "mean_validation_gafgyt_fnr",
        "mean_validation_mirai_fnr",
        "mean_calibration_elapsed_seconds",
        "test_access_count",
    ],
)

selection_rows: list[
    dict[str, Any]
] = []

for architecture in ARCHITECTURES:
    architecture_groups = [
        row
        for row in group_rows
        if row["architecture"]
        == architecture
    ]

    ranked = sorted(
        architecture_groups,
        key=lambda row: (
            -float(
                row[
                    "mean_validation_fingerprint_macro_f1"
                ]
            ),
            float(
                row[
                    "std_validation_fingerprint_macro_f1"
                ]
            ),
            int(
                row[
                    "calibration_size"
                ]
            ),
        ),
    )

    for rank, row in enumerate(
        ranked,
        start=1,
    ):
        selection_rows.append(
            {
                "architecture": architecture,
                "rank": rank,
                "selected": (
                    rank == 1
                ),
                "calibration_size": int(
                    row[
                        "calibration_size"
                    ]
                ),
                "mean_validation_fingerprint_macro_f1": float(
                    row[
                        "mean_validation_fingerprint_macro_f1"
                    ]
                ),
                "std_validation_fingerprint_macro_f1": float(
                    row[
                        "std_validation_fingerprint_macro_f1"
                    ]
                ),
                "mean_validation_raw_weighted_macro_f1": float(
                    row[
                        "mean_validation_raw_weighted_macro_f1"
                    ]
                ),
                "selection_metric": (
                    PRIMARY_SELECTION_METRIC
                ),
                "tie_breaks": (
                    SELECTION_TIE_BREAKS
                ),
                "test_access_count": 0,
            }
        )

atomic_csv(
    SELECTION_CSV,
    selection_rows,
    [
        "architecture",
        "rank",
        "selected",
        "calibration_size",
        "mean_validation_fingerprint_macro_f1",
        "std_validation_fingerprint_macro_f1",
        "mean_validation_raw_weighted_macro_f1",
        "selection_metric",
        "tie_breaks",
        "test_access_count",
    ],
)

selected_calibration_sizes = {
    row["architecture"]: int(
        row["calibration_size"]
    )
    for row in selection_rows
    if bool(
        row["selected"]
    )
}

if set(
    selected_calibration_sizes.keys()
) != set(
    ARCHITECTURES
):
    raise RuntimeError(
        "PTQ calibration-size selection "
        "did not produce one winner per "
        "architecture."
    )

global_checks = {
    "forty_candidates_completed": (
        len(candidate_rows)
        == 40
    ),
    "eight_groups_completed": (
        len(group_rows)
        == 8
    ),
    "one_selection_per_architecture": (
        len(
            selected_calibration_sizes
        )
        == 2
    ),
    "validation_access_one_per_candidate": (
        all(
            int(
                row[
                    "validation_access_count"
                ]
            )
            == 1
            for row in candidate_rows
        )
    ),
    "test_access_zero_all_candidates": (
        all(
            int(
                row[
                    "test_access_count"
                ]
            )
            == 0
            for row in candidate_rows
        )
    ),
    "selection_test_access_zero": (
        all(
            int(
                row[
                    "test_access_count"
                ]
            )
            == 0
            for row in selection_rows
        )
    ),
}

failed_global_checks = [
    name
    for name, passed
    in global_checks.items()
    if not passed
]

if failed_global_checks:
    raise RuntimeError(
        "PTQ validation-sweep global "
        "checks failed: "
        + ", ".join(
            failed_global_checks
        )
    )

aggregate = {
    "status": "completed",
    "phase": 5,
    "artifact_name": (
        "PTQ_validation_calibration_sweep"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "PTQ_engine_version": (
        PTQ_ENGINE_VERSION
    ),
    "completed_at_utc": utc_now(),
    "candidate_count": 40,
    "group_count": 8,
    "architecture_count": 2,
    "seed_count": 5,
    "calibration_sizes": list(
        CALIBRATION_SIZES
    ),
    "selection_policy": {
        "selection_unit": (
            "architecture_level_mean_across_five_seeds"
        ),
        "primary_metric": (
            PRIMARY_SELECTION_METRIC
        ),
        "tie_breaks": (
            SELECTION_TIE_BREAKS
        ),
        "selected_calibration_sizes": (
            selected_calibration_sizes
        ),
        "validation_used_for_selection": (
            True
        ),
        "test_used_for_selection": (
            False
        ),
    },
    "test_policy": {
        "test_access_count": 0,
        "test_deferred_until_after_independent_verification": (
            True
        ),
    },
    "groups": group_json,
    "entry_checks": (
        entry_checks
    ),
    "global_checks": (
        global_checks
    ),
    "candidate_csv": (
        file_record(
            AGGREGATE_CANDIDATES_CSV
        )
    ),
    "group_summary_csv": (
        file_record(
            GROUP_SUMMARY_CSV
        )
    ),
    "selection_csv": (
        file_record(
            SELECTION_CSV
        )
    ),
    "ready_for_independent_verification": (
        True
    ),
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
        "PTQ_validation_calibration_sweep"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "PTQ_engine_version": (
        PTQ_ENGINE_VERSION
    ),
    "completed_at_utc": utc_now(),
    "candidate_count": 40,
    "group_count": 8,
    "selected_calibration_sizes": (
        selected_calibration_sizes
    ),
    "validation_access_count_per_candidate": (
        1
    ),
    "test_access_count": 0,
    "candidate_csv": str(
        AGGREGATE_CANDIDATES_CSV
    ),
    "candidate_csv_sha256": (
        sha256_file(
            AGGREGATE_CANDIDATES_CSV
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
    "selection_csv": str(
        SELECTION_CSV
    ),
    "selection_csv_sha256": (
        sha256_file(
            SELECTION_CSV
        )
    ),
    "aggregate_json": str(
        AGGREGATE_JSON
    ),
    "aggregate_json_sha256": (
        sha256_file(
            AGGREGATE_JSON
        )
    ),
    "ready_for_independent_verification": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    COMPLETION_JSON,
    completion,
)

print()
print("=" * 92)
print("PHASE 5 PTQ VALIDATION SWEEP SUMMARY")
print("=" * 92)
print(
    "Candidates completed            : 40"
)
print(
    "Architecture-size groups        : 8"
)
print(
    "Validation access per candidate : 1"
)
print(
    "Test access count               : 0"
)
print(
    "Selection unit                  : "
    "architecture-level mean across 5 seeds"
)
print(
    "Primary selection metric        : "
    f"{PRIMARY_SELECTION_METRIC}"
)
print(
    "Tie-breaks                      : "
    f"{SELECTION_TIE_BREAKS}"
)
print()

for architecture in ARCHITECTURES:
    architecture_rows = sorted(
        [
            row
            for row in group_rows
            if row["architecture"]
            == architecture
        ],
        key=lambda row: int(
            row["calibration_size"]
        ),
    )

    for row in architecture_rows:
        selected = (
            int(
                row[
                    "calibration_size"
                ]
            )
            == selected_calibration_sizes[
                architecture
            ]
        )

        print(
            f"{architecture:<16} "
            "cal="
            f"{int(row['calibration_size']):>6} | "
            "mean val Macro-F1="
            f"{float(row['mean_validation_fingerprint_macro_f1']):.9f} | "
            "std="
            f"{float(row['std_validation_fingerprint_macro_f1']):.9f} | "
            "raw-weighted="
            f"{float(row['mean_validation_raw_weighted_macro_f1']):.9f} | "
            "selected="
            f"{selected}"
        )

print()
print(
    "Selected calibration sizes      : "
    f"{selected_calibration_sizes}"
)
print(
    "Candidate CSV                   : "
    f"{AGGREGATE_CANDIDATES_CSV}"
)
print(
    "Group summary CSV               : "
    f"{GROUP_SUMMARY_CSV}"
)
print(
    "Selection CSV                   : "
    f"{SELECTION_CSV}"
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
    "PHASE 5 PTQ VALIDATION CALIBRATION SWEEP COMPLETED"
)
