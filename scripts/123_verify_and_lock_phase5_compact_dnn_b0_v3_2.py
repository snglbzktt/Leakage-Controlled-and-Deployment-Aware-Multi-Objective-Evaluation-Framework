from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
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

X_VALIDATION_PATH = FINAL_CACHE / "X_validation.npy"
Y_VALIDATION_PATH = FINAL_CACHE / "y_validation.npy"
RAW_VALIDATION_PATH = (
    FINAL_CACHE / "raw_row_count_validation.npy"
)

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

TINYML_B0_LOCK = (
    AUDIT
    / "phase5_tinyml_mlp_b0_locked_v3_2.json"
)

MODEL_SOURCE = (
    ROOT / "src" / "models" / "nbaiot_models.py"
)

LOCKED_MATRIX = (
    AUDIT / "phase5_locked_configuration_matrix_v3_2.csv"
)

RUN_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "compact_dnn"
    / "b0"
)

AGGREGATE_JSON = (
    AUDIT / "phase5_compact_dnn_b0_aggregate_v3_2.json"
)

AGGREGATE_CSV = (
    AUDIT / "phase5_compact_dnn_b0_runs_v3_2.csv"
)

OUTPUT_VERIFICATION = (
    AUDIT / "phase5_compact_dnn_b0_verification_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT / "phase5_compact_dnn_b0_locked_v3_2.json"
)

PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
DATA_PROTOCOL_VERSION = "tabular_baseline_protocol_v3_2"

ARCHITECTURE = "compact_dnn"
VARIANT = "B0"
SEEDS = [42, 123, 2026, 3407, 8192]

EXPECTED_FEATURE_COUNT = 115
EXPECTED_CLASS_COUNT = 3
EXPECTED_CLASS_LABELS = np.asarray(
    [0, 1, 2],
    dtype=np.int64,
)
CLASS_NAMES = ("benign", "gafgyt", "mirai")

EXPECTED_VALIDATION_ROWS = 371_797
EXPECTED_TEST_ROWS = 371_796
EXPECTED_VALIDATION_RAW_ROWS = 1_059_390
EXPECTED_TEST_RAW_ROWS = 1_059_393

EXPECTED_LINEAR_WIDTHS = [
    [115, 128],
    [128, 64],
    [64, 32],
    [32, 3],
]

BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 16_384
MAX_EPOCHS = 15
PATIENCE = 4
MIN_DELTA = 0.0002
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
CPU_THREADS = 4

FLOAT_TOLERANCE = 1e-10
PROBABILITY_ATOL = 2e-6


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


def file_record(
    path: Path,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }


def close_enough(
    observed: float,
    expected: float,
    tolerance: float = FLOAT_TOLERANCE,
) -> bool:
    return math.isclose(
        float(observed),
        float(expected),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def record_matches(
    record: dict[str, Any],
    path: Path,
) -> bool:
    return (
        str(record["path"]) == str(path)
        and int(record["size_bytes"])
        == int(path.stat().st_size)
        and str(record["sha256"])
        == sha256_file(path)
    )


def load_model_module(
    path: Path,
) -> ModuleType:
    module_name = (
        "phase5_compact_b0_verification_models"
    )

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
            "Could not create model-source "
            "import specification."
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
) -> dict[str, Any]:
    signature = inspect.signature(candidate)
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
            kwargs[name] = [128, 64, 32]
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
    module: ModuleType,
    symbol_name: str,
) -> nn.Module:
    if not hasattr(
        module,
        symbol_name,
    ):
        raise RuntimeError(
            "Locked model symbol is missing: "
            f"{symbol_name}"
        )

    candidate = getattr(
        module,
        symbol_name,
    )

    model = candidate(
        **callable_kwargs(candidate)
    )

    if not isinstance(
        model,
        nn.Module,
    ):
        raise RuntimeError(
            "Locked model symbol did not "
            "create torch.nn.Module."
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


def linear_widths(
    model: nn.Module,
) -> list[list[int]]:
    return [
        [
            int(layer.in_features),
            int(layer.out_features),
        ]
        for layer in model.modules()
        if isinstance(
            layer,
            nn.Linear,
        )
    ]


def parameter_count(
    model: nn.Module,
) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )


def linear_macs_per_sample(
    model: nn.Module,
) -> int:
    return int(
        sum(
            layer.in_features
            * layer.out_features
            for layer in model.modules()
            if isinstance(
                layer,
                nn.Linear,
            )
        )
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
            "Non-finite validation value "
            "after scaling."
        )

    return transformed


def infer_validation(
    model: nn.Module,
    x_validation: np.ndarray,
    mean64: np.ndarray,
    scale64: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.empty(
        len(x_validation),
        dtype=np.int8,
    )

    probabilities = np.empty(
        (
            len(x_validation),
            EXPECTED_CLASS_COUNT,
        ),
        dtype=np.float32,
    )

    model.eval()

    with torch.inference_mode():
        for start in range(
            0,
            len(x_validation),
            EVAL_BATCH_SIZE,
        ):
            end = min(
                start + EVAL_BATCH_SIZE,
                len(x_validation),
            )

            x_batch = torch.from_numpy(
                transform_batch(
                    x_validation[start:end],
                    mean64,
                    scale64,
                )
            )

            logits = extract_logits(
                model(x_batch)
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

    return predictions, probabilities


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


def metrics_match(
    recomputed: dict[str, Any],
    saved: dict[str, Any],
) -> bool:
    scalar_keys = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "log_loss",
        "roc_auc_ovr_macro",
    )

    for key in scalar_keys:
        if not close_enough(
            recomputed[key],
            saved[key],
        ):
            return False

    if (
        recomputed[
            "confusion_matrix"
        ]
        != saved[
            "confusion_matrix"
        ]
    ):
        return False

    for class_name in CLASS_NAMES:
        for key in (
            "label",
            "precision",
            "recall",
            "f1",
            "support",
            "false_negative_rate",
        ):
            observed = recomputed[
                "per_class"
            ][class_name][key]

            expected = saved[
                "per_class"
            ][class_name][key]

            if key == "label":
                if int(observed) != int(
                    expected
                ):
                    return False
            elif not close_enough(
                observed,
                expected,
            ):
                return False

    return True


def verify_manifest_inventory(
    run_directory: Path,
    manifest: dict[str, Any],
) -> bool:
    records = manifest[
        "artifacts"
    ]

    recorded_names = {
        str(record["relative_path"])
        for record in records
    }

    actual_names = {
        path.name
        for path in run_directory.iterdir()
        if path.is_file()
        and path.name
        not in {
            "run_manifest.json",
            "run_status.json",
        }
    }

    if (
        recorded_names
        != actual_names
    ):
        return False

    for record in records:
        path = (
            run_directory
            / str(
                record[
                    "relative_path"
                ]
            )
        )

        if not path.exists():
            return False

        if int(
            record["size_bytes"]
        ) != int(
            path.stat().st_size
        ):
            return False

        if str(
            record["sha256"]
        ) != sha256_file(path):
            return False

    return True


required_paths = (
    FINAL_CACHE,
    X_VALIDATION_PATH,
    Y_VALIDATION_PATH,
    RAW_VALIDATION_PATH,
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
    TINYML_B0_LOCK,
    MODEL_SOURCE,
    LOCKED_MATRIX,
    RUN_ROOT,
    AGGREGATE_JSON,
    AGGREGATE_CSV,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for path in (
    OUTPUT_VERIFICATION,
    OUTPUT_LOCK,
):
    if path.exists():
        raise FileExistsError(
            "Compact-DNN B0 verification artifact "
            "already exists; refusing to "
            f"overwrite: {path}"
        )

protocol = read_json(
    PHASE5_PROTOCOL
)

protocol_completion = read_json(
    PHASE5_PROTOCOL_COMPLETION
)

protocol_lock = read_json(
    PHASE5_PROTOCOL_LOCK
)

preprocessing_completion = read_json(
    PREPROCESSING_COMPLETION
)

preprocessing_verification = read_json(
    PREPROCESSING_VERIFICATION
)

preprocessing_lock = read_json(
    PREPROCESSING_LOCK
)

smoke_report = read_json(
    SMOKE_REPORT
)

smoke_completion = read_json(
    SMOKE_COMPLETION
)

tinyml_b0_lock = read_json(
    TINYML_B0_LOCK
)

aggregate = read_json(
    AGGREGATE_JSON
)

aggregate_rows = read_csv(
    AGGREGATE_CSV
)

matrix_rows = read_csv(
    LOCKED_MATRIX
)

matrix_b0_rows = [
    row
    for row in matrix_rows
    if row["architecture"]
    == ARCHITECTURE
    and row["variant"]
    == VARIANT
]

matrix_b0_rows.sort(
    key=lambda row: int(
        row["seed"]
    )
)

smoke_architecture = next(
    (
        row
        for row in smoke_completion[
            "architectures"
        ]
        if row["architecture"]
        == ARCHITECTURE
    ),
    None,
)

if smoke_architecture is None:
    raise RuntimeError(
        "Compact-DNN smoke architecture "
        "record is missing."
    )

MODEL_SYMBOL = str(
    smoke_architecture[
        "selected_symbol"
    ]
)

entry_checks = {
    "protocol_locked": (
        protocol.get("status")
        == "locked"
    ),
    "protocol_version_matches": (
        protocol.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
    ),
    "data_protocol_matches": (
        protocol.get(
            "data_protocol_version"
        )
        == DATA_PROTOCOL_VERSION
    ),
    "protocol_completion_hash_matches": (
        protocol_completion.get(
            "protocol_sha256"
        )
        == sha256_file(
            PHASE5_PROTOCOL
        )
    ),
    "protocol_lock_passed": (
        protocol_lock.get("status")
        == "locked"
        and protocol_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "preprocessing_lock_passed": (
        preprocessing_completion.get(
            "status"
        )
        == "locked"
        and preprocessing_completion.get(
            "ready_for_B0_training"
        )
        is True
        and preprocessing_verification.get(
            "status"
        )
        == "passed"
        and preprocessing_lock.get(
            "status"
        )
        == "locked"
    ),
    "smoke_test_passed": (
        smoke_report.get("status")
        == "passed"
        and smoke_report.get(
            "all_checks_passed"
        )
        is True
    ),
    "smoke_test_locked": (
        smoke_completion.get(
            "status"
        )
        == "locked"
        and smoke_completion.get(
            "ready_for_B0_training_runs"
        )
        is True
    ),
    "tinyml_B0_already_locked": (
        tinyml_b0_lock.get(
            "status"
        )
        == "locked"
        and tinyml_b0_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "aggregate_completed": (
        aggregate.get("status")
        == "completed"
        and aggregate.get(
            "all_integrity_checks_passed"
        )
        is True
    ),
    "aggregate_protocol_matches": (
        aggregate.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
    ),
    "aggregate_architecture_matches": (
        aggregate.get(
            "architecture"
        )
        == ARCHITECTURE
        and aggregate.get(
            "variant"
        )
        == VARIANT
    ),
    "aggregate_has_five_runs": (
        int(
            aggregate.get(
                "run_count"
            )
        )
        == 5
        and len(
            aggregate.get(
                "runs",
                [],
            )
        )
        == 5
        and len(
            aggregate_rows
        )
        == 5
    ),
    "aggregate_seed_set_matches": (
        list(
            aggregate.get(
                "seeds"
            )
        )
        == SEEDS
        and sorted(
            int(row["seed"])
            for row in aggregate_rows
        )
        == sorted(SEEDS)
    ),
    "aggregate_csv_hash_matches": (
        aggregate.get(
            "aggregate_csv_sha256"
        )
        == sha256_file(
            AGGREGATE_CSV
        )
    ),
    "matrix_has_five_B0_rows": (
        len(matrix_b0_rows)
        == 5
        and [
            int(row["seed"])
            for row in matrix_b0_rows
        ]
        == SEEDS
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
        "Compact-DNN B0 verification entry "
        "gate failed: "
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

x_validation = np.load(
    X_VALIDATION_PATH,
    mmap_mode="r",
)

y_validation_source = np.asarray(
    np.load(
        Y_VALIDATION_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

raw_validation_source = np.asarray(
    np.load(
        RAW_VALIDATION_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

y_test_source = np.asarray(
    np.load(
        Y_TEST_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

raw_test_source = np.asarray(
    np.load(
        RAW_TEST_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
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

source_checks = {
    "validation_shape_matches": (
        x_validation.shape
        == (
            EXPECTED_VALIDATION_ROWS,
            EXPECTED_FEATURE_COUNT,
        )
        and y_validation_source.shape
        == (
            EXPECTED_VALIDATION_ROWS,
        )
    ),
    "test_label_shape_matches": (
        y_test_source.shape
        == (
            EXPECTED_TEST_ROWS,
        )
    ),
    "validation_raw_total_matches": (
        int(
            raw_validation_source.sum()
        )
        == EXPECTED_VALIDATION_RAW_ROWS
    ),
    "test_raw_total_matches": (
        int(
            raw_test_source.sum()
        )
        == EXPECTED_TEST_RAW_ROWS
    ),
    "scaler_shapes_match": (
        scaler_mean64.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
        and scaler_scale64.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
    ),
}

failed_source_checks = [
    name
    for name, passed
    in source_checks.items()
    if not passed
]

if failed_source_checks:
    raise RuntimeError(
        "Compact-DNN B0 verification source "
        "checks failed: "
        + ", ".join(
            failed_source_checks
        )
    )

module = load_model_module(
    MODEL_SOURCE
)

print("=" * 92)
print("PHASE 5 COMPACT-DNN B0 INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Seeds                       : "
    f"{SEEDS}"
)
print(
    "Validation model inference  : Yes"
)
print(
    "Test model inference        : No"
)
print(
    "Test verification source    : "
    "Saved predictions only"
)
print()

verified_runs: list[
    dict[str, Any]
] = []

run_checks_by_seed: dict[
    str,
    dict[str, bool],
] = {}

for run_index, seed in enumerate(
    SEEDS,
    start=1,
):
    run_id = (
        f"compact_dnn__b0__seed_{seed}"
    )

    run_directory = (
        RUN_ROOT / f"seed_{seed}"
    )

    required_run_files = {
        "status": (
            run_directory
            / "run_status.json"
        ),
        "manifest": (
            run_directory
            / "run_manifest.json"
        ),
        "metrics": (
            run_directory
            / "metrics.json"
        ),
        "checkpoint": (
            run_directory
            / "best_checkpoint.pt"
        ),
        "history": (
            run_directory
            / "training_history.csv"
        ),
        "validation_predictions": (
            run_directory
            / "validation_predictions.npz"
        ),
        "test_predictions": (
            run_directory
            / "test_predictions.npz"
        ),
    }

    for path in (
        required_run_files.values()
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    status = read_json(
        required_run_files["status"]
    )

    manifest = read_json(
        required_run_files[
            "manifest"
        ]
    )

    metrics = read_json(
        required_run_files[
            "metrics"
        ]
    )

    history_rows = read_csv(
        required_run_files[
            "history"
        ]
    )

    aggregate_row = next(
        row
        for row in aggregate_rows
        if int(row["seed"]) == seed
    )

    aggregate_json_row = next(
        row
        for row in aggregate[
            "runs"
        ]
        if int(row["seed"]) == seed
    )

    checkpoint = torch.load(
        required_run_files[
            "checkpoint"
        ],
        map_location="cpu",
        weights_only=False,
    )

    model = instantiate_model(
        module,
        MODEL_SYMBOL,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    observed_widths = linear_widths(
        model
    )

    observed_parameter_count = (
        parameter_count(model)
    )

    observed_macs = (
        linear_macs_per_sample(
            model
        )
    )

    with np.load(
        required_run_files[
            "validation_predictions"
        ]
    ) as values:
        saved_validation_y = np.array(
            values["y_true"],
            copy=True,
        ).astype(np.int64)

        saved_validation_pred = np.array(
            values["y_pred"],
            copy=True,
        )

        saved_validation_prob = np.array(
            values["probabilities"],
            copy=True,
        )

        saved_validation_raw = np.array(
            values["raw_row_count"],
            copy=True,
        ).astype(np.int64)

    with np.load(
        required_run_files[
            "test_predictions"
        ]
    ) as values:
        saved_test_y = np.array(
            values["y_true"],
            copy=True,
        ).astype(np.int64)

        saved_test_pred = np.array(
            values["y_pred"],
            copy=True,
        )

        saved_test_prob = np.array(
            values["probabilities"],
            copy=True,
        )

        saved_test_raw = np.array(
            values["raw_row_count"],
            copy=True,
        ).astype(np.int64)

    (
        inferred_validation_pred,
        inferred_validation_prob,
    ) = infer_validation(
        model=model,
        x_validation=x_validation,
        mean64=scaler_mean64,
        scale64=scaler_scale64,
    )

    recomputed_validation_primary = (
        metric_view(
            y_true=saved_validation_y,
            y_pred=saved_validation_pred,
            probabilities=(
                saved_validation_prob
            ),
            sample_weight=None,
        )
    )

    recomputed_validation_weighted = (
        metric_view(
            y_true=saved_validation_y,
            y_pred=saved_validation_pred,
            probabilities=(
                saved_validation_prob
            ),
            sample_weight=(
                saved_validation_raw
            ),
        )
    )

    recomputed_test_primary = (
        metric_view(
            y_true=saved_test_y,
            y_pred=saved_test_pred,
            probabilities=(
                saved_test_prob
            ),
            sample_weight=None,
        )
    )

    recomputed_test_weighted = (
        metric_view(
            y_true=saved_test_y,
            y_pred=saved_test_pred,
            probabilities=(
                saved_test_prob
            ),
            sample_weight=(
                saved_test_raw
            ),
        )
    )

    best_epoch = int(
        metrics["selection"][
            "best_epoch"
        ]
    )

    history_best_row = next(
        (
            row
            for row in history_rows
            if int(row["epoch"])
            == best_epoch
        ),
        None,
    )

    if history_best_row is None:
        raise RuntimeError(
            f"{run_id}: best epoch "
            "is missing from history."
        )

    run_checks = {
        "run_directory_exists": (
            run_directory.exists()
        ),
        "status_completed": (
            status.get("status")
            == "completed"
            and status.get("stage")
            == "completed"
        ),
        "status_identity_matches": (
            status.get("run_id")
            == run_id
            and int(
                status.get("seed")
            )
            == seed
        ),
        "manifest_completed": (
            manifest.get("status")
            == "completed"
            and manifest.get(
                "all_integrity_checks_passed"
            )
            is True
        ),
        "manifest_identity_matches": (
            manifest.get("run_id")
            == run_id
            and int(
                manifest.get("seed")
            )
            == seed
            and manifest.get(
                "architecture"
            )
            == ARCHITECTURE
            and manifest.get(
                "variant"
            )
            == VARIANT
        ),
        "manifest_source_hashes_match": (
            record_matches(
                manifest[
                    "model_source"
                ],
                MODEL_SOURCE,
            )
            and record_matches(
                manifest[
                    "phase5_protocol"
                ],
                PHASE5_PROTOCOL,
            )
            and record_matches(
                manifest[
                    "locked_matrix"
                ],
                LOCKED_MATRIX,
            )
            and record_matches(
                manifest["scaler"],
                SCALER_NPZ,
            )
            and record_matches(
                manifest[
                    "class_weights"
                ],
                CLASS_WEIGHTS_NPZ,
            )
            and record_matches(
                manifest[
                    "smoke_test"
                ],
                SMOKE_REPORT,
            )
        ),
        "manifest_artifact_inventory_matches": (
            verify_manifest_inventory(
                run_directory,
                manifest,
            )
        ),
        "legacy_checkpoint_not_reused": (
            manifest.get(
                "legacy_checkpoint_reused"
            )
            is False
        ),
        "locked_training_parameters_match": (
            int(
                metrics["training"][
                    "batch_size"
                ]
            )
            == BATCH_SIZE
            and int(
                metrics["training"][
                    "max_epochs"
                ]
            )
            == MAX_EPOCHS
            and close_enough(
                metrics["training"][
                    "learning_rate"
                ],
                LEARNING_RATE,
            )
            and close_enough(
                metrics["training"][
                    "weight_decay"
                ],
                WEIGHT_DECAY,
            )
            and int(
                metrics["training"][
                    "early_stopping_patience"
                ]
            )
            == PATIENCE
            and close_enough(
                metrics["training"][
                    "early_stopping_min_delta"
                ],
                MIN_DELTA,
            )
        ),
        "test_not_used_for_selection": (
            metrics["selection"][
                "test_used_for_selection"
            ]
            is False
            and manifest[
                "fit_scope"
            ]["test_used_for_selection"]
            is False
        ),
        "test_evaluation_count_is_one": (
            int(
                metrics[
                    "test_evaluation_count"
                ]
            )
            == 1
            and int(
                manifest[
                    "fit_scope"
                ][
                    "test_evaluation_count"
                ]
            )
            == 1
            and int(
                status[
                    "test_evaluation_count"
                ]
            )
            == 1
        ),
        "checkpoint_identity_matches": (
            checkpoint.get("run_id")
            == run_id
            and int(
                checkpoint.get("seed")
            )
            == seed
            and checkpoint.get(
                "architecture"
            )
            == ARCHITECTURE
            and checkpoint.get(
                "variant"
            )
            == VARIANT
            and checkpoint.get(
                "model_symbol"
            )
            == MODEL_SYMBOL
        ),
        "checkpoint_contains_optimizer_state": (
            isinstance(
                checkpoint.get(
                    "optimizer_state_dict"
                ),
                dict,
            )
            and len(
                checkpoint.get(
                    "optimizer_state_dict"
                )
            )
            > 0
        ),
        "model_topology_matches": (
            observed_widths
            == EXPECTED_LINEAR_WIDTHS
        ),
        "model_complexity_matches": (
            observed_parameter_count
            == int(
                metrics[
                    "model_complexity"
                ][
                    "parameter_count"
                ]
            )
            == int(
                checkpoint[
                    "parameter_count"
                ]
            )
            and observed_macs
            == int(
                metrics[
                    "model_complexity"
                ][
                    "linear_macs_per_sample"
                ]
            )
            == int(
                checkpoint[
                    "linear_macs_per_sample"
                ]
            )
        ),
        "best_epoch_matches": (
            int(
                checkpoint[
                    "best_epoch"
                ]
            )
            == best_epoch
            == int(
                manifest[
                    "best_epoch"
                ]
            )
            == int(
                status[
                    "best_epoch"
                ]
            )
        ),
        "best_validation_metric_matches": (
            close_enough(
                checkpoint[
                    "best_validation_macro_f1"
                ],
                metrics[
                    "selection"
                ][
                    "best_validation_macro_f1"
                ],
            )
            and close_enough(
                manifest[
                    "best_validation_macro_f1"
                ],
                metrics[
                    "selection"
                ][
                    "best_validation_macro_f1"
                ],
            )
            and close_enough(
                float(
                    history_best_row[
                        "validation_macro_f1"
                    ]
                ),
                metrics[
                    "selection"
                ][
                    "best_validation_macro_f1"
                ],
            )
        ),
        "history_length_matches": (
            len(history_rows)
            == int(
                metrics["training"][
                    "epochs_completed"
                ]
            )
            == int(
                aggregate_row[
                    "epochs_completed"
                ]
            )
        ),
        "validation_saved_shapes_match": (
            saved_validation_y.shape
            == (
                EXPECTED_VALIDATION_ROWS,
            )
            and saved_validation_pred.shape
            == (
                EXPECTED_VALIDATION_ROWS,
            )
            and saved_validation_prob.shape
            == (
                EXPECTED_VALIDATION_ROWS,
                EXPECTED_CLASS_COUNT,
            )
            and saved_validation_raw.shape
            == (
                EXPECTED_VALIDATION_ROWS,
            )
        ),
        "test_saved_shapes_match": (
            saved_test_y.shape
            == (
                EXPECTED_TEST_ROWS,
            )
            and saved_test_pred.shape
            == (
                EXPECTED_TEST_ROWS,
            )
            and saved_test_prob.shape
            == (
                EXPECTED_TEST_ROWS,
                EXPECTED_CLASS_COUNT,
            )
            and saved_test_raw.shape
            == (
                EXPECTED_TEST_ROWS,
            )
        ),
        "prediction_dtypes_match": (
            saved_validation_pred.dtype
            == np.int8
            and saved_test_pred.dtype
            == np.int8
            and saved_validation_prob.dtype
            == np.float32
            and saved_test_prob.dtype
            == np.float32
        ),
        "probabilities_finite_and_normalized": (
            np.isfinite(
                saved_validation_prob
            ).all()
            and np.isfinite(
                saved_test_prob
            ).all()
            and np.allclose(
                saved_validation_prob.sum(
                    axis=1
                ),
                1.0,
                atol=1e-6,
                rtol=0.0,
            )
            and np.allclose(
                saved_test_prob.sum(
                    axis=1
                ),
                1.0,
                atol=1e-6,
                rtol=0.0,
            )
        ),
        "saved_labels_match_locked_cache": (
            np.array_equal(
                saved_validation_y,
                y_validation_source,
            )
            and np.array_equal(
                saved_test_y,
                y_test_source,
            )
        ),
        "saved_raw_counts_match_locked_cache": (
            np.array_equal(
                saved_validation_raw,
                raw_validation_source,
            )
            and np.array_equal(
                saved_test_raw,
                raw_test_source,
            )
        ),
        "independent_validation_predictions_match": (
            np.array_equal(
                inferred_validation_pred,
                saved_validation_pred,
            )
        ),
        "independent_validation_probabilities_match": (
            np.allclose(
                inferred_validation_prob,
                saved_validation_prob,
                atol=PROBABILITY_ATOL,
                rtol=0.0,
            )
        ),
        "validation_primary_metrics_recompute": (
            metrics_match(
                recomputed_validation_primary,
                metrics[
                    "validation"
                ][
                    "primary_fingerprint_level"
                ],
            )
        ),
        "validation_weighted_metrics_recompute": (
            metrics_match(
                recomputed_validation_weighted,
                metrics[
                    "validation"
                ][
                    "secondary_raw_record_weighted"
                ],
            )
        ),
        "test_primary_metrics_recompute_from_saved_predictions": (
            metrics_match(
                recomputed_test_primary,
                metrics[
                    "test"
                ][
                    "primary_fingerprint_level"
                ],
            )
        ),
        "test_weighted_metrics_recompute_from_saved_predictions": (
            metrics_match(
                recomputed_test_weighted,
                metrics[
                    "test"
                ][
                    "secondary_raw_record_weighted"
                ],
            )
        ),
        "fragility_flag_matches": (
            bool(
                metrics[
                    "fragility_gate"
                ]["triggered"]
            )
            is False
            and bool(
                status[
                    "fragility_gate_triggered"
                ]
            )
            is False
            and bool(
                aggregate_json_row[
                    "fragility_gate_triggered"
                ]
            )
            is False
        ),
        "aggregate_hashes_match": (
            aggregate_row[
                "metrics_sha256"
            ]
            == sha256_file(
                required_run_files[
                    "metrics"
                ]
            )
            and aggregate_row[
                "run_manifest_sha256"
            ]
            == sha256_file(
                required_run_files[
                    "manifest"
                ]
            )
        ),
        "aggregate_scalar_values_match": (
            close_enough(
                aggregate_row[
                    "test_fingerprint_macro_f1"
                ],
                recomputed_test_primary[
                    "macro_f1"
                ],
            )
            and close_enough(
                aggregate_row[
                    "test_raw_weighted_macro_f1"
                ],
                recomputed_test_weighted[
                    "macro_f1"
                ],
            )
            and close_enough(
                aggregate_row[
                    "test_gafgyt_fnr"
                ],
                recomputed_test_primary[
                    "per_class"
                ]["gafgyt"][
                    "false_negative_rate"
                ],
            )
            and close_enough(
                aggregate_row[
                    "test_mirai_fnr"
                ],
                recomputed_test_primary[
                    "per_class"
                ]["mirai"][
                    "false_negative_rate"
                ],
            )
        ),
    }

    failed_run_checks = [
        name
        for name, passed
        in run_checks.items()
        if not passed
    ]

    if failed_run_checks:
        raise RuntimeError(
            f"{run_id} verification failed: "
            + ", ".join(
                failed_run_checks
            )
        )

    run_checks_by_seed[
        str(seed)
    ] = run_checks

    verified_runs.append(
        {
            "run_id": run_id,
            "seed": seed,
            "model_symbol": (
                MODEL_SYMBOL
            ),
            "best_epoch": best_epoch,
            "epochs_completed": (
                len(history_rows)
            ),
            "test_fingerprint_macro_f1": (
                recomputed_test_primary[
                    "macro_f1"
                ]
            ),
            "test_fingerprint_accuracy": (
                recomputed_test_primary[
                    "accuracy"
                ]
            ),
            "test_raw_weighted_macro_f1": (
                recomputed_test_weighted[
                    "macro_f1"
                ]
            ),
            "test_gafgyt_fnr": (
                recomputed_test_primary[
                    "per_class"
                ]["gafgyt"][
                    "false_negative_rate"
                ]
            ),
            "test_mirai_fnr": (
                recomputed_test_primary[
                    "per_class"
                ]["mirai"][
                    "false_negative_rate"
                ]
            ),
            "parameter_count": (
                observed_parameter_count
            ),
            "linear_macs_per_sample": (
                observed_macs
            ),
            "test_evaluation_count": 1,
            "test_model_inference_during_verification": False,
            "all_checks_passed": True,
        }
    )

    print(
        f"[{run_index}/5] seed={seed} | "
        f"best_epoch={best_epoch} | "
        "validation inference=True | "
        "test inference=False | "
        "test Macro-F1="
        f"{recomputed_test_primary['macro_f1']:.9f} | "
        "all checks=True",
        flush=True,
    )

    del model
    del checkpoint


def aggregate_values(
    key: str,
) -> dict[str, float]:
    values = np.asarray(
        [
            float(row[key])
            for row in verified_runs
        ],
        dtype=np.float64,
    )

    return {
        "mean": float(
            np.mean(values)
        ),
        "std_population": float(
            np.std(
                values,
                ddof=0,
            )
        ),
        "minimum": float(
            np.min(values)
        ),
        "maximum": float(
            np.max(values)
        ),
    }


recomputed_aggregate = {
    "test_fingerprint_macro_f1": (
        aggregate_values(
            "test_fingerprint_macro_f1"
        )
    ),
    "test_fingerprint_accuracy": (
        aggregate_values(
            "test_fingerprint_accuracy"
        )
    ),
    "test_raw_weighted_macro_f1": (
        aggregate_values(
            "test_raw_weighted_macro_f1"
        )
    ),
    "test_gafgyt_fnr": (
        aggregate_values(
            "test_gafgyt_fnr"
        )
    ),
    "test_mirai_fnr": (
        aggregate_values(
            "test_mirai_fnr"
        )
    ),
}

aggregate_checks: dict[
    str,
    bool,
] = {}

for metric_name, values in (
    recomputed_aggregate.items()
):
    saved_values = aggregate[
        "aggregate_metrics"
    ][metric_name]

    for statistic_name in (
        "mean",
        "std_population",
        "minimum",
        "maximum",
    ):
        aggregate_checks[
            f"{metric_name}_{statistic_name}_matches"
        ] = close_enough(
            values[
                statistic_name
            ],
            saved_values[
                statistic_name
            ],
        )

aggregate_checks.update(
    {
        "all_five_test_counts_are_one": (
            all(
                int(
                    row[
                        "test_evaluation_count"
                    ]
                )
                == 1
                for row in verified_runs
            )
        ),
        "no_test_model_inference_in_verifier": (
            all(
                row[
                    "test_model_inference_during_verification"
                ]
                is False
                for row in verified_runs
            )
        ),
        "no_fragility_in_aggregate": (
            aggregate[
                "fragility_gate_triggered_in_any_run"
            ]
            is False
        ),
        "aggregate_mean_macro_f1_is_finite": (
            math.isfinite(
                recomputed_aggregate[
                    "test_fingerprint_macro_f1"
                ]["mean"]
            )
        ),
        "aggregate_std_macro_f1_is_finite": (
            math.isfinite(
                recomputed_aggregate[
                    "test_fingerprint_macro_f1"
                ]["std_population"]
            )
        ),
        "aggregate_mean_raw_weighted_is_finite": (
            math.isfinite(
                recomputed_aggregate[
                    "test_raw_weighted_macro_f1"
                ]["mean"]
            )
        ),
    }
)

failed_aggregate_checks = [
    name
    for name, passed
    in aggregate_checks.items()
    if not passed
]

if failed_aggregate_checks:
    raise RuntimeError(
        "Compact-DNN B0 aggregate verification "
        "failed: "
        + ", ".join(
            failed_aggregate_checks
        )
    )

all_checks = {
    **entry_checks,
    **source_checks,
    **aggregate_checks,
}

verification = {
    "status": "passed",
    "phase": 5,
    "artifact_name": (
        "compact_dnn_B0_five_seed_verification"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "data_protocol_version": (
        DATA_PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "architecture": ARCHITECTURE,
    "variant": VARIANT,
    "model_symbol": MODEL_SYMBOL,
    "seeds": SEEDS,
    "run_count": len(
        verified_runs
    ),
    "verification_policy": {
        "checkpoint_loaded_for_each_seed": True,
        "full_validation_inference_repeated": True,
        "validation_predictions_compared": True,
        "test_model_inference_repeated": False,
        "test_metrics_recomputed_from_saved_predictions": True,
        "reason_test_inference_not_repeated": (
            "Preserve the locked one-time "
            "test-evaluation policy."
        ),
    },
    "runs": verified_runs,
    "recomputed_aggregate_metrics": (
        recomputed_aggregate
    ),
    "run_checks_by_seed": (
        run_checks_by_seed
    ),
    "checks": all_checks,
    "source_artifacts": {
        "aggregate_json": (
            file_record(
                AGGREGATE_JSON
            )
        ),
        "aggregate_csv": (
            file_record(
                AGGREGATE_CSV
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
        "model_source": (
            file_record(
                MODEL_SOURCE
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
        "smoke_test": (
            file_record(
                SMOKE_REPORT
            )
        ),
    },
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_VERIFICATION,
    verification,
)

lock = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "compact_dnn_B0_five_seed_runs"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "architecture": ARCHITECTURE,
    "variant": VARIANT,
    "run_count": 5,
    "seeds": SEEDS,
    "mean_test_fingerprint_macro_f1": (
        recomputed_aggregate[
            "test_fingerprint_macro_f1"
        ]["mean"]
    ),
    "std_test_fingerprint_macro_f1": (
        recomputed_aggregate[
            "test_fingerprint_macro_f1"
        ]["std_population"]
    ),
    "mean_test_raw_weighted_macro_f1": (
        recomputed_aggregate[
            "test_raw_weighted_macro_f1"
        ]["mean"]
    ),
    "test_evaluation_count_per_run": 1,
    "test_model_inference_repeated_by_verifier": False,
    "fragility_triggered_in_any_run": False,
    "verification_report": str(
        OUTPUT_VERIFICATION
    ),
    "verification_sha256": (
        sha256_file(
            OUTPUT_VERIFICATION
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
    "aggregate_csv": str(
        AGGREGATE_CSV
    ),
    "aggregate_csv_sha256": (
        sha256_file(
            AGGREGATE_CSV
        )
    ),
    "ready_for_phase5_B0_pair_lock": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK,
    lock,
)

post_checks = {
    "verification_exists": (
        OUTPUT_VERIFICATION.exists()
    ),
    "lock_exists": (
        OUTPUT_LOCK.exists()
    ),
    "lock_verification_hash_matches": (
        read_json(
            OUTPUT_LOCK
        )["verification_sha256"]
        == sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "ready_for_B0_pair_lock": (
        read_json(
            OUTPUT_LOCK
        ).get(
            "ready_for_phase5_B0_pair_lock"
        )
        is True
    ),
}

failed_post_checks = [
    name
    for name, passed
    in post_checks.items()
    if not passed
]

if failed_post_checks:
    raise RuntimeError(
        "Compact-DNN B0 post-lock checks failed: "
        + ", ".join(
            failed_post_checks
        )
    )

macro = recomputed_aggregate[
    "test_fingerprint_macro_f1"
]

weighted = recomputed_aggregate[
    "test_raw_weighted_macro_f1"
]

print()
print("=" * 92)
print("PHASE 5 COMPACT-DNN B0 VERIFICATION AND LOCK SUMMARY")
print("=" * 92)
print(
    "Verified runs                   : "
    f"{len(verified_runs)}"
)
print(
    "Checkpoint loading              : PASSED"
)
print(
    "Independent validation inference: PASSED"
)
print(
    "Test model inference repeated   : False"
)
print(
    "Saved test metrics recomputed   : PASSED"
)
print(
    "Test evaluation count per run   : 1"
)
print(
    "Mean test fingerprint Macro-F1  : "
    f"{macro['mean']:.9f}"
)
print(
    "Std test fingerprint Macro-F1   : "
    f"{macro['std_population']:.9f}"
)
print(
    "Mean raw-weighted Macro-F1      : "
    f"{weighted['mean']:.9f}"
)
print(
    "Fragility in any run            : False"
)
print(
    "Compact-DNN B0 status           : LOCKED"
)
print(
    "Ready for B0 pair lock          : True"
)
print(
    "Verification report             : "
    f"{OUTPUT_VERIFICATION}"
)
print(
    "Lock file                       : "
    f"{OUTPUT_LOCK}"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 5 COMPACT-DNN B0 FIVE-SEED RUNS LOCKED"
)
