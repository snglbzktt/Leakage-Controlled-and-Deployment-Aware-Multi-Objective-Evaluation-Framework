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

PHASE5_PROTOCOL_COMPLETION = (
    AUDIT / "phase5_protocol_locked_v3_2.json"
)

PHASE5_PROTOCOL_LOCK = (
    AUDIT / "phase5_protocol_lock_manifest_v3_2.json"
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

PREPROCESSING_COMPLETION = (
    AUDIT / "phase5_preprocessing_locked_v3_2.json"
)

PREPROCESSING_VERIFICATION = (
    AUDIT / "phase5_preprocessing_verification_v3_2.json"
)

PREPROCESSING_LOCK = (
    AUDIT / "phase5_preprocessing_lock_manifest_v3_2.json"
)

SMOKE_REPORT = (
    AUDIT / "phase5_neural_architecture_smoke_test_v3_2.json"
)

SMOKE_COMPLETION = (
    AUDIT
    / "phase5_neural_architecture_smoke_test_locked_v3_2.json"
)

MODEL_SOURCE = (
    ROOT / "src" / "models" / "nbaiot_models.py"
)

LOCKED_MATRIX = (
    AUDIT / "phase5_locked_configuration_matrix_v3_2.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "tinyml_mlp"
    / "b0"
)

AGGREGATE_JSON = (
    AUDIT / "phase5_tinyml_mlp_b0_aggregate_v3_2.json"
)

AGGREGATE_CSV = (
    AUDIT / "phase5_tinyml_mlp_b0_runs_v3_2.csv"
)

PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
DATA_PROTOCOL_VERSION = "tabular_baseline_protocol_v3_2"

ARCHITECTURE = "tinyml_mlp"
VARIANT = "B0"
SEEDS = [42, 123, 2026, 3407, 8192]

EXPECTED_FEATURE_COUNT = 115
EXPECTED_CLASS_COUNT = 3
EXPECTED_CLASS_LABELS = np.asarray([0, 1, 2], dtype=np.int64)
CLASS_NAMES = ("benign", "gafgyt", "mirai")

EXPECTED_TRAIN_ROWS = 1_534_583
EXPECTED_VALIDATION_ROWS = 371_797
EXPECTED_TEST_ROWS = 371_796
EXPECTED_VALIDATION_RAW_ROWS = 1_059_390
EXPECTED_TEST_RAW_ROWS = 1_059_393

EXPECTED_LINEAR_WIDTHS = [
    [115, 64],
    [64, 32],
    [32, 3],
]

BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 16384
MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 0.0002
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
CPU_THREADS = 4
MINIMUM_FREE_DISK_GIB = 1.5

PRIMARY_SELECTION_METRIC = (
    "validation_fingerprint_macro_f1"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(csv.DictReader(handle))


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)

    return digest.hexdigest()


def atomic_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    os.replace(temporary, path)


def atomic_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

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

    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_module(path: Path) -> ModuleType:
    module_name = "phase5_b0_locked_nbaiot_models"

    specification = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "Could not create model-source import specification."
        )

    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module

    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def callable_kwargs(
    candidate: Any,
) -> dict[str, Any]:
    signature = inspect.signature(candidate)
    kwargs: dict[str, Any] = {}

    for name, parameter in signature.parameters.items():
        if name in {"self", "args", "kwargs"}:
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
            kwargs[name] = EXPECTED_FEATURE_COUNT
            continue

        if normalized in {
            "numclasses",
            "nclasses",
            "classcount",
            "outputdim",
            "outputsize",
            "outfeatures",
        }:
            kwargs[name] = EXPECTED_CLASS_COUNT
            continue

        if normalized in {
            "hiddendims",
            "hiddensizes",
            "hiddenlayers",
            "hiddenunits",
        }:
            kwargs[name] = [64, 32]
            continue

        if parameter.default is inspect.Parameter.empty:
            raise RuntimeError(
                f"Unresolved required constructor parameter: {name}"
            )

    return kwargs


def linear_widths(model: nn.Module) -> list[list[int]]:
    return [
        [
            int(layer.in_features),
            int(layer.out_features),
        ]
        for layer in model.modules()
        if isinstance(layer, nn.Linear)
    ]


def instantiate_model(
    module: ModuleType,
    symbol_name: str,
    seed: int,
) -> nn.Module:
    if not hasattr(module, symbol_name):
        raise RuntimeError(
            f"Locked model symbol does not exist: {symbol_name}"
        )

    candidate = getattr(module, symbol_name)

    if not callable(candidate):
        raise RuntimeError(
            f"Locked model symbol is not callable: {symbol_name}"
        )

    set_all_seeds(seed)

    model = candidate(
        **callable_kwargs(candidate)
    )

    if not isinstance(model, nn.Module):
        raise RuntimeError(
            "Locked model symbol did not create torch.nn.Module."
        )

    observed_widths = linear_widths(model)

    if observed_widths != EXPECTED_LINEAR_WIDTHS:
        raise RuntimeError(
            "TinyML MLP topology mismatch: "
            f"{observed_widths}"
        )

    return model


def extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value

    if isinstance(output, dict):
        for key in (
            "logits",
            "output",
            "outputs",
            "predictions",
        ):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
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
            np.asarray(source, dtype=np.float64)
            - mean64
        )
        / scale64
    ).astype(np.float32)

    if not np.isfinite(transformed).all():
        raise RuntimeError(
            "Non-finite value after StandardScaler transform."
        )

    return transformed


def evaluate_model(
    model: nn.Module,
    x_source: np.ndarray,
    y_source: np.ndarray,
    mean64: np.ndarray,
    scale64: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
        (len(y_true), EXPECTED_CLASS_COUNT),
        dtype=np.float32,
    )

    total_loss = 0.0
    total_count = 0

    started = time.perf_counter()

    with torch.inference_mode():
        for start in range(
            0,
            len(y_true),
            batch_size,
        ):
            end = min(
                start + batch_size,
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

            if tuple(logits.shape) != (
                end - start,
                EXPECTED_CLASS_COUNT,
            ):
                raise RuntimeError(
                    "Evaluation logit shape mismatch."
                )

            batch_probabilities = torch.softmax(
                logits,
                dim=1,
            )

            probabilities[start:end] = (
                batch_probabilities.cpu().numpy().astype(
                    np.float32
                )
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

            batch_targets = torch.from_numpy(
                y_true[start:end]
            )

            batch_loss = nn.functional.cross_entropy(
                logits,
                batch_targets,
                reduction="sum",
            )

            total_loss += float(
                batch_loss.item()
            )

            total_count += end - start

    elapsed = time.perf_counter() - started

    return (
        predictions,
        probabilities,
        y_true,
        elapsed,
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
        matrix = matrix.astype(np.int64)
    else:
        matrix = np.rint(matrix).astype(np.int64)

    support_total = float(np.sum(support))

    result: dict[str, Any] = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
                sample_weight=sample_weight,
            )
        ),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(
            np.average(f1, weights=support)
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
        "confusion_matrix": matrix.tolist(),
        "per_class": {},
    }

    for index, class_name in enumerate(CLASS_NAMES):
        result["per_class"][class_name] = {
            "label": int(EXPECTED_CLASS_LABELS[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": float(support[index]),
            "false_negative_rate": float(
                1.0 - recall[index]
            ),
        }

    return result


def save_confusion_csv(
    path: Path,
    matrix: list[list[int]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *CLASS_NAMES])

        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row])


def save_per_class_csv(
    path: Path,
    primary: dict[str, Any],
    weighted: dict[str, Any],
) -> None:
    fieldnames = [
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
    ]

    rows = []

    for class_name in CLASS_NAMES:
        primary_row = primary["per_class"][class_name]
        weighted_row = weighted["per_class"][class_name]

        rows.append(
            {
                "class_name": class_name,
                "label": primary_row["label"],
                "fingerprint_precision": primary_row[
                    "precision"
                ],
                "fingerprint_recall": primary_row[
                    "recall"
                ],
                "fingerprint_f1": primary_row["f1"],
                "fingerprint_support": primary_row[
                    "support"
                ],
                "fingerprint_fnr": primary_row[
                    "false_negative_rate"
                ],
                "raw_weighted_precision": weighted_row[
                    "precision"
                ],
                "raw_weighted_recall": weighted_row[
                    "recall"
                ],
                "raw_weighted_f1": weighted_row["f1"],
                "raw_weighted_support": weighted_row[
                    "support"
                ],
                "raw_weighted_fnr": weighted_row[
                    "false_negative_rate"
                ],
            }
        )

    atomic_csv(path, rows, fieldnames)


def clone_state_dict(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def clone_optimizer_state(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return copy.deepcopy(optimizer.state_dict())


def parameter_count(model: nn.Module) -> int:
    return int(
        sum(
            value.numel()
            for value in model.parameters()
        )
    )


def linear_macs_per_sample(model: nn.Module) -> int:
    return int(
        sum(
            layer.in_features * layer.out_features
            for layer in model.modules()
            if isinstance(layer, nn.Linear)
        )
    )


def artifact_inventory(
    output_directory: Path,
) -> list[dict[str, Any]]:
    excluded = {
        "run_manifest.json",
        "run_status.json",
    }

    return [
        {
            "relative_path": path.name,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(output_directory.iterdir())
        if path.is_file() and path.name not in excluded
    ]


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
    PHASE5_PROTOCOL_COMPLETION,
    PHASE5_PROTOCOL_LOCK,
    SCALER_NPZ,
    CLASS_WEIGHTS_NPZ,
    PREPROCESSING_COMPLETION,
    PREPROCESSING_VERIFICATION,
    PREPROCESSING_LOCK,
    SMOKE_REPORT,
    SMOKE_COMPLETION,
    MODEL_SOURCE,
    LOCKED_MATRIX,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    AGGREGATE_JSON,
    AGGREGATE_CSV,
):
    if output_path.exists():
        raise FileExistsError(
            "TinyML B0 aggregate artifact already exists; "
            f"refusing to overwrite: {output_path}"
        )

protocol = read_json(PHASE5_PROTOCOL)
protocol_completion = read_json(
    PHASE5_PROTOCOL_COMPLETION
)
protocol_lock = read_json(PHASE5_PROTOCOL_LOCK)
preprocessing_completion = read_json(
    PREPROCESSING_COMPLETION
)
preprocessing_verification = read_json(
    PREPROCESSING_VERIFICATION
)
preprocessing_lock = read_json(
    PREPROCESSING_LOCK
)
smoke_report = read_json(SMOKE_REPORT)
smoke_completion = read_json(SMOKE_COMPLETION)

matrix_rows = read_csv(LOCKED_MATRIX)

selected_matrix_rows = [
    row
    for row in matrix_rows
    if row["architecture"] == ARCHITECTURE
    and row["variant"] == VARIANT
]

selected_matrix_rows.sort(
    key=lambda row: int(row["seed"])
)

selected_seeds = [
    int(row["seed"])
    for row in selected_matrix_rows
]

smoke_architecture = next(
    (
        row
        for row in smoke_completion["architectures"]
        if row["architecture"] == ARCHITECTURE
    ),
    None,
)

if smoke_architecture is None:
    raise RuntimeError(
        "TinyML MLP smoke-test architecture record is missing."
    )

MODEL_SYMBOL = str(
    smoke_architecture["selected_symbol"]
)

entry_checks = {
    "protocol_locked": (
        protocol.get("status") == "locked"
    ),
    "protocol_version_matches": (
        protocol.get("protocol_version")
        == PROTOCOL_VERSION
    ),
    "data_protocol_matches": (
        protocol.get("data_protocol_version")
        == DATA_PROTOCOL_VERSION
    ),
    "protocol_completion_locked": (
        protocol_completion.get("status") == "locked"
        and protocol_completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "protocol_hash_matches": (
        protocol_completion.get("protocol_sha256")
        == sha256_file(PHASE5_PROTOCOL)
    ),
    "protocol_lock_locked": (
        protocol_lock.get("status") == "locked"
        and protocol_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "preprocessing_locked": (
        preprocessing_completion.get("status")
        == "locked"
        and preprocessing_completion.get(
            "ready_for_B0_training"
        )
        is True
    ),
    "preprocessing_verification_passed": (
        preprocessing_verification.get("status")
        == "passed"
        and preprocessing_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "preprocessing_lock_locked": (
        preprocessing_lock.get("status") == "locked"
        and preprocessing_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "smoke_test_passed": (
        smoke_report.get("status") == "passed"
        and smoke_report.get(
            "all_checks_passed"
        )
        is True
    ),
    "smoke_test_locked": (
        smoke_completion.get("status") == "locked"
        and smoke_completion.get(
            "ready_for_B0_training_runs"
        )
        is True
    ),
    "smoke_report_hash_matches": (
        smoke_completion.get("report_sha256")
        == sha256_file(SMOKE_REPORT)
    ),
    "matrix_has_five_tinyml_B0_rows": (
        len(selected_matrix_rows) == 5
    ),
    "matrix_seed_set_matches": (
        selected_seeds == SEEDS
    ),
    "matrix_B0_budget_matches": all(
        int(row["max_epochs"]) == MAX_EPOCHS
        and int(row["batch_size"]) == BATCH_SIZE
        and math.isclose(
            float(row["learning_rate"]),
            LEARNING_RATE,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and int(row["early_stopping_patience"])
        == EARLY_STOPPING_PATIENCE
        and math.isclose(
            float(row["early_stopping_min_delta"]),
            EARLY_STOPPING_MIN_DELTA,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for row in selected_matrix_rows
    ),
    "matrix_test_count_is_one": all(
        int(row["phase5_test_evaluation_count"]) == 1
        for row in selected_matrix_rows
    ),
    "model_source_hash_matches_protocol": (
        protocol["architectures"]["model_source"][
            "sha256"
        ]
        == sha256_file(MODEL_SOURCE)
    ),
    "legacy_checkpoint_reuse_disabled": (
        protocol["legacy_reuse_policy"][
            "legacy_checkpoints_directly_reused"
        ]
        is False
    ),
}

failed_entry_checks = [
    name
    for name, passed in entry_checks.items()
    if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "TinyML B0 entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

free_disk_gib = (
    shutil.disk_usage(ROOT).free / (1024**3)
)

if free_disk_gib < MINIMUM_FREE_DISK_GIB:
    raise RuntimeError(
        "Insufficient free disk space: "
        f"{free_disk_gib:.3f} GiB"
    )

torch.set_num_threads(CPU_THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

torch.use_deterministic_algorithms(True)

x_train = np.load(X_TRAIN_PATH, mmap_mode="r")
y_train = np.load(Y_TRAIN_PATH, mmap_mode="r")

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

x_test = np.load(X_TEST_PATH, mmap_mode="r")
y_test = np.load(Y_TEST_PATH, mmap_mode="r")
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
        and y_train.shape == (EXPECTED_TRAIN_ROWS,)
    ),
    "validation_shape_matches": (
        x_validation.shape
        == (
            EXPECTED_VALIDATION_ROWS,
            EXPECTED_FEATURE_COUNT,
        )
        and y_validation.shape
        == (EXPECTED_VALIDATION_ROWS,)
    ),
    "test_shape_matches": (
        x_test.shape
        == (
            EXPECTED_TEST_ROWS,
            EXPECTED_FEATURE_COUNT,
        )
        and y_test.shape == (EXPECTED_TEST_ROWS,)
    ),
    "validation_raw_total_matches": (
        int(raw_validation.sum())
        == EXPECTED_VALIDATION_RAW_ROWS
    ),
    "test_raw_total_matches": (
        int(raw_test.sum())
        == EXPECTED_TEST_RAW_ROWS
    ),
    "all_x_dtypes_float32": (
        x_train.dtype == np.float32
        and x_validation.dtype == np.float32
        and x_test.dtype == np.float32
    ),
}

failed_shape_checks = [
    name
    for name, passed in shape_checks.items()
    if not passed
]

if failed_shape_checks:
    raise RuntimeError(
        "TinyML B0 cache checks failed: "
        + ", ".join(failed_shape_checks)
    )

with np.load(SCALER_NPZ) as values:
    scaler_mean64 = np.array(
        values["mean_float64"],
        copy=True,
    )
    scaler_scale64 = np.array(
        values["scale_float64"],
        copy=True,
    )

with np.load(CLASS_WEIGHTS_NPZ) as values:
    class_weights32 = np.array(
        values["class_weights_float32"],
        copy=True,
    )

if (
    scaler_mean64.shape != (EXPECTED_FEATURE_COUNT,)
    or scaler_scale64.shape != (EXPECTED_FEATURE_COUNT,)
    or class_weights32.shape != (EXPECTED_CLASS_COUNT,)
):
    raise RuntimeError(
        "TinyML B0 preprocessing artifact shape mismatch."
    )

module = load_model_module(MODEL_SOURCE)
class_weight_tensor = torch.from_numpy(
    class_weights32
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("=" * 92)
print("PHASE 5 TINYML MLP B0 FIVE-SEED TRAINING")
print("=" * 92)
print(f"Architecture       : {ARCHITECTURE}")
print(f"Model symbol       : {MODEL_SYMBOL}")
print(f"Seeds              : {SEEDS}")
print(f"Train fingerprints : {EXPECTED_TRAIN_ROWS:,}")
print(f"Validation rows    : {EXPECTED_VALIDATION_ROWS:,}")
print(f"Test rows          : {EXPECTED_TEST_ROWS:,}")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Max epochs         : {MAX_EPOCHS}")
print(f"Learning rate      : {LEARNING_RATE}")
print(f"CPU threads        : {CPU_THREADS}")
print(f"Free disk          : {free_disk_gib:.3f} GiB")
print()

run_summaries: list[dict[str, Any]] = []

for run_index, seed in enumerate(SEEDS, start=1):
    run_id = f"tinyml_mlp__b0__seed_{seed}"
    output_directory = OUTPUT_ROOT / f"seed_{seed}"
    status_path = output_directory / "run_status.json"

    if output_directory.exists():
        raise FileExistsError(
            "TinyML B0 output directory already exists; "
            f"refusing to overwrite: {output_directory}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    atomic_json(
        status_path,
        {
            "status": "running",
            "stage": "training",
            "run_id": run_id,
            "architecture": ARCHITECTURE,
            "variant": VARIANT,
            "seed": seed,
            "started_at_utc": utc_now(),
        },
    )

    set_all_seeds(seed)

    model = instantiate_model(
        module,
        MODEL_SYMBOL,
        seed,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    rng = np.random.default_rng(seed)

    best_macro_f1 = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []

    training_started = time.perf_counter()

    print(
        f"[{run_index}/5] {run_id}",
        flush=True,
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_started = time.perf_counter()
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

            indices = permutation[start:end]

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

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{run_id}: non-finite loss "
                    f"at epoch {epoch}."
                )

            loss.backward()
            optimizer.step()

            batch_count = end - start

            epoch_loss_sum += (
                float(loss.item()) * batch_count
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

            epoch_count += batch_count

        (
            validation_predictions,
            validation_probabilities,
            validation_y_true,
            validation_elapsed,
        ) = evaluate_model(
            model=model,
            x_source=x_validation,
            y_source=y_validation,
            mean64=scaler_mean64,
            scale64=scaler_scale64,
            batch_size=EVAL_BATCH_SIZE,
        )

        validation_primary = metric_view(
            y_true=validation_y_true,
            y_pred=validation_predictions,
            probabilities=validation_probabilities,
            sample_weight=None,
        )

        validation_macro_f1 = float(
            validation_primary["macro_f1"]
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
            best_macro_f1 = validation_macro_f1
            best_epoch = epoch
            best_state = clone_state_dict(model)
            best_optimizer_state = (
                clone_optimizer_state(optimizer)
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_elapsed = (
            time.perf_counter() - epoch_started
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": (
                    epoch_loss_sum / epoch_count
                ),
                "train_accuracy": (
                    epoch_correct / epoch_count
                ),
                "validation_macro_f1": (
                    validation_macro_f1
                ),
                "validation_accuracy": (
                    validation_primary["accuracy"]
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

        print(
            f"  epoch={epoch:02d} | "
            f"train_loss={epoch_loss_sum / epoch_count:.6f} | "
            f"train_acc={epoch_correct / epoch_count:.6f} | "
            f"val_macro_f1={validation_macro_f1:.9f} | "
            f"best={best_macro_f1:.9f} | "
            f"wait={epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE} | "
            f"elapsed={epoch_elapsed / 60.0:.2f} min",
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
        time.perf_counter() - training_started
    )

    if (
        best_state is None
        or best_optimizer_state is None
        or best_epoch <= 0
    ):
        raise RuntimeError(
            f"{run_id}: no best checkpoint was selected."
        )

    model.load_state_dict(best_state)
    model.eval()

    atomic_csv(
        output_directory / "training_history.csv",
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
        output_directory / "best_checkpoint.pt"
    )

    torch.save(
        {
            "run_id": run_id,
            "architecture": ARCHITECTURE,
            "variant": VARIANT,
            "seed": seed,
            "protocol_version": PROTOCOL_VERSION,
            "model_symbol": MODEL_SYMBOL,
            "model_state_dict": best_state,
            "optimizer_state_dict": (
                best_optimizer_state
            ),
            "best_epoch": best_epoch,
            "best_validation_macro_f1": (
                best_macro_f1
            ),
            "linear_widths": linear_widths(model),
            "parameter_count": parameter_count(model),
            "linear_macs_per_sample": (
                linear_macs_per_sample(model)
            ),
        },
        checkpoint_path,
    )

    (
        validation_predictions,
        validation_probabilities,
        validation_y_true,
        validation_elapsed,
    ) = evaluate_model(
        model=model,
        x_source=x_validation,
        y_source=y_validation,
        mean64=scaler_mean64,
        scale64=scaler_scale64,
        batch_size=EVAL_BATCH_SIZE,
    )

    validation_primary = metric_view(
        y_true=validation_y_true,
        y_pred=validation_predictions,
        probabilities=validation_probabilities,
        sample_weight=None,
    )

    validation_weighted = metric_view(
        y_true=validation_y_true,
        y_pred=validation_predictions,
        probabilities=validation_probabilities,
        sample_weight=raw_validation,
    )

    np.savez_compressed(
        output_directory
        / "validation_predictions.npz",
        y_true=validation_y_true.astype(np.int8),
        y_pred=validation_predictions,
        probabilities=validation_probabilities,
        raw_row_count=raw_validation,
    )

    atomic_json(
        status_path,
        {
            "status": "running",
            "stage": "single_test_evaluation",
            "run_id": run_id,
            "architecture": ARCHITECTURE,
            "variant": VARIANT,
            "seed": seed,
            "best_epoch": best_epoch,
            "updated_at_utc": utc_now(),
        },
    )

    test_evaluation_count = 0

    (
        test_predictions,
        test_probabilities,
        test_y_true,
        test_elapsed,
    ) = evaluate_model(
        model=model,
        x_source=x_test,
        y_source=y_test,
        mean64=scaler_mean64,
        scale64=scaler_scale64,
        batch_size=EVAL_BATCH_SIZE,
    )

    test_evaluation_count += 1

    if test_evaluation_count != 1:
        raise RuntimeError(
            f"{run_id}: test evaluation count "
            f"is {test_evaluation_count}, expected 1."
        )

    test_primary = metric_view(
        y_true=test_y_true,
        y_pred=test_predictions,
        probabilities=test_probabilities,
        sample_weight=None,
    )

    test_weighted = metric_view(
        y_true=test_y_true,
        y_pred=test_predictions,
        probabilities=test_probabilities,
        sample_weight=raw_test,
    )

    np.savez_compressed(
        output_directory / "test_predictions.npz",
        y_true=test_y_true.astype(np.int8),
        y_pred=test_predictions,
        probabilities=test_probabilities,
        raw_row_count=raw_test,
    )

    save_confusion_csv(
        output_directory
        / "validation_confusion_fingerprint.csv",
        validation_primary["confusion_matrix"],
    )

    save_confusion_csv(
        output_directory
        / "validation_confusion_raw_weighted.csv",
        validation_weighted["confusion_matrix"],
    )

    save_confusion_csv(
        output_directory
        / "test_confusion_fingerprint.csv",
        test_primary["confusion_matrix"],
    )

    save_confusion_csv(
        output_directory
        / "test_confusion_raw_weighted.csv",
        test_weighted["confusion_matrix"],
    )

    save_per_class_csv(
        output_directory
        / "validation_per_class_metrics.csv",
        validation_primary,
        validation_weighted,
    )

    save_per_class_csv(
        output_directory
        / "test_per_class_metrics.csv",
        test_primary,
        test_weighted,
    )

    raw_tensor_bytes = int(
        sum(
            value.numel() * value.element_size()
            for value in model.state_dict().values()
        )
    )

    fragility_gate = {
        "macro_f1_below_0_85": (
            test_primary["macro_f1"] < 0.85
        ),
        "gafgyt_fnr_above_0_40": (
            test_primary["per_class"]["gafgyt"][
                "false_negative_rate"
            ]
            > 0.40
        ),
        "mirai_fnr_above_0_40": (
            test_primary["per_class"]["mirai"][
                "false_negative_rate"
            ]
            > 0.40
        ),
    }

    fragility_gate["triggered"] = any(
        fragility_gate.values()
    )

    metrics = {
        "run_id": run_id,
        "architecture": ARCHITECTURE,
        "variant": VARIANT,
        "seed": seed,
        "protocol_version": PROTOCOL_VERSION,
        "selection": {
            "metric": PRIMARY_SELECTION_METRIC,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": (
                best_macro_f1
            ),
            "test_used_for_selection": False,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "epochs_completed": len(history_rows),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),
            "early_stopping_min_delta": (
                EARLY_STOPPING_MIN_DELTA
            ),
            "elapsed_seconds": training_elapsed,
            "raw_occurrence_weight_used": False,
        },
        "validation": {
            "primary_fingerprint_level": (
                validation_primary
            ),
            "secondary_raw_record_weighted": (
                validation_weighted
            ),
            "elapsed_seconds": validation_elapsed,
        },
        "test": {
            "primary_fingerprint_level": test_primary,
            "secondary_raw_record_weighted": (
                test_weighted
            ),
            "elapsed_seconds": test_elapsed,
        },
        "model_complexity": {
            "linear_widths": linear_widths(model),
            "parameter_count": parameter_count(model),
            "linear_macs_per_sample": (
                linear_macs_per_sample(model)
            ),
            "raw_tensor_bytes": raw_tensor_bytes,
            "serialized_checkpoint_bytes": int(
                checkpoint_path.stat().st_size
            ),
        },
        "test_evaluation_count": (
            test_evaluation_count
        ),
        "fragility_gate": fragility_gate,
    }

    atomic_json(
        output_directory / "metrics.json",
        metrics,
    )

    artifacts = artifact_inventory(
        output_directory
    )

    completed_at = utc_now()

    manifest = {
        "status": "completed",
        "completed_at_utc": completed_at,
        "run_id": run_id,
        "architecture": ARCHITECTURE,
        "variant": VARIANT,
        "seed": seed,
        "protocol_version": PROTOCOL_VERSION,
        "data_protocol_version": (
            DATA_PROTOCOL_VERSION
        ),
        "model_symbol": MODEL_SYMBOL,
        "model_source": file_record(MODEL_SOURCE),
        "phase5_protocol": file_record(
            PHASE5_PROTOCOL
        ),
        "locked_matrix": file_record(LOCKED_MATRIX),
        "scaler": file_record(SCALER_NPZ),
        "class_weights": file_record(
            CLASS_WEIGHTS_NPZ
        ),
        "smoke_test": file_record(SMOKE_REPORT),
        "legacy_checkpoint_reused": False,
        "fit_scope": {
            "train_used_for_updates": True,
            "validation_used_for_selection": True,
            "test_used_for_selection": False,
            "test_evaluation_count": (
                test_evaluation_count
            ),
            "raw_occurrence_weight_used_for_fit": False,
        },
        "parameters": {
            "optimizer": "AdamW",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
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
        "fragility_gate": fragility_gate,
        "artifacts": artifacts,
        "all_integrity_checks_passed": True,
    }

    atomic_json(
        output_directory / "run_manifest.json",
        manifest,
    )

    atomic_json(
        status_path,
        {
            "status": "completed",
            "stage": "completed",
            "run_id": run_id,
            "architecture": ARCHITECTURE,
            "variant": VARIANT,
            "seed": seed,
            "best_epoch": best_epoch,
            "completed_at_utc": completed_at,
            "test_evaluation_count": (
                test_evaluation_count
            ),
            "fragility_gate_triggered": (
                fragility_gate["triggered"]
            ),
        },
    )

    run_summary = {
        "run_id": run_id,
        "architecture": ARCHITECTURE,
        "variant": VARIANT,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history_rows),
        "best_validation_macro_f1": (
            best_macro_f1
        ),
        "test_fingerprint_macro_f1": (
            test_primary["macro_f1"]
        ),
        "test_fingerprint_accuracy": (
            test_primary["accuracy"]
        ),
        "test_raw_weighted_macro_f1": (
            test_weighted["macro_f1"]
        ),
        "test_gafgyt_fnr": (
            test_primary["per_class"]["gafgyt"][
                "false_negative_rate"
            ]
        ),
        "test_mirai_fnr": (
            test_primary["per_class"]["mirai"][
                "false_negative_rate"
            ]
        ),
        "parameter_count": parameter_count(model),
        "linear_macs_per_sample": (
            linear_macs_per_sample(model)
        ),
        "raw_tensor_bytes": raw_tensor_bytes,
        "serialized_checkpoint_bytes": int(
            checkpoint_path.stat().st_size
        ),
        "training_elapsed_seconds": (
            training_elapsed
        ),
        "test_evaluation_count": (
            test_evaluation_count
        ),
        "fragility_gate_triggered": (
            fragility_gate["triggered"]
        ),
        "output_directory": str(
            output_directory
        ),
        "metrics_sha256": sha256_file(
            output_directory / "metrics.json"
        ),
        "run_manifest_sha256": sha256_file(
            output_directory / "run_manifest.json"
        ),
    }

    run_summaries.append(run_summary)

    print(
        "  completed | "
        f"best_epoch={best_epoch} | "
        f"test_macro_f1={test_primary['macro_f1']:.9f} | "
        f"raw_weighted={test_weighted['macro_f1']:.9f} | "
        f"gafgyt_fnr="
        f"{test_primary['per_class']['gafgyt']['false_negative_rate']:.9f} | "
        f"mirai_fnr="
        f"{test_primary['per_class']['mirai']['false_negative_rate']:.9f} | "
        f"fragility={fragility_gate['triggered']}",
        flush=True,
    )

    del model
    del optimizer
    del criterion


def aggregate_values(
    key: str,
) -> dict[str, float]:
    values = np.asarray(
        [
            float(row[key])
            for row in run_summaries
        ],
        dtype=np.float64,
    )

    return {
        "mean": float(np.mean(values)),
        "std_population": float(
            np.std(values, ddof=0)
        ),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


aggregate_metrics = {
    "test_fingerprint_macro_f1": aggregate_values(
        "test_fingerprint_macro_f1"
    ),
    "test_fingerprint_accuracy": aggregate_values(
        "test_fingerprint_accuracy"
    ),
    "test_raw_weighted_macro_f1": aggregate_values(
        "test_raw_weighted_macro_f1"
    ),
    "test_gafgyt_fnr": aggregate_values(
        "test_gafgyt_fnr"
    ),
    "test_mirai_fnr": aggregate_values(
        "test_mirai_fnr"
    ),
    "best_validation_macro_f1": aggregate_values(
        "best_validation_macro_f1"
    ),
    "training_elapsed_seconds": aggregate_values(
        "training_elapsed_seconds"
    ),
}

atomic_csv(
    AGGREGATE_CSV,
    run_summaries,
    [
        "run_id",
        "architecture",
        "variant",
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
        "output_directory",
        "metrics_sha256",
        "run_manifest_sha256",
    ],
)

aggregate = {
    "status": "completed",
    "phase": 5,
    "architecture": ARCHITECTURE,
    "variant": VARIANT,
    "protocol_version": PROTOCOL_VERSION,
    "generated_at_utc": utc_now(),
    "run_count": len(run_summaries),
    "seeds": SEEDS,
    "model_symbol": MODEL_SYMBOL,
    "aggregate_metrics": aggregate_metrics,
    "runs": run_summaries,
    "test_evaluation_count_per_run": 1,
    "test_used_for_selection": False,
    "fragility_gate_triggered_in_any_run": any(
        bool(row["fragility_gate_triggered"])
        for row in run_summaries
    ),
    "aggregate_csv": str(AGGREGATE_CSV),
    "aggregate_csv_sha256": sha256_file(
        AGGREGATE_CSV
    ),
    "runner_script": str(
        Path(__file__).resolve()
    ),
    "runner_script_sha256": sha256_file(
        Path(__file__).resolve()
    ),
    "all_integrity_checks_passed": True,
}

atomic_json(
    AGGREGATE_JSON,
    aggregate,
)

macro = aggregate_metrics[
    "test_fingerprint_macro_f1"
]

weighted = aggregate_metrics[
    "test_raw_weighted_macro_f1"
]

print()
print("=" * 92)
print("PHASE 5 TINYML MLP B0 FIVE-SEED SUMMARY")
print("=" * 92)
print(f"Runs completed                 : {len(run_summaries)}")
print(f"Seeds                          : {SEEDS}")
print(
    "Mean test fingerprint Macro-F1 : "
    f"{macro['mean']:.9f}"
)
print(
    "Std test fingerprint Macro-F1  : "
    f"{macro['std_population']:.9f}"
)
print(
    "Min / max fingerprint Macro-F1 : "
    f"{macro['minimum']:.9f} / "
    f"{macro['maximum']:.9f}"
)
print(
    "Mean raw-weighted Macro-F1     : "
    f"{weighted['mean']:.9f}"
)
print(
    "Fragility in any run           : "
    f"{aggregate['fragility_gate_triggered_in_any_run']}"
)
print(f"Aggregate JSON                 : {AGGREGATE_JSON}")
print(f"Aggregate CSV                  : {AGGREGATE_CSV}")
print("All integrity checks passed    : True")
print("PHASE 5 TINYML MLP B0 FIVE-SEED TRAINING COMPLETED")
