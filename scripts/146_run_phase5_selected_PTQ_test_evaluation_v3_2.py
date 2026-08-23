from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import io
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
X_TEST_PATH = FINAL_CACHE / "X_test.npy"
Y_TEST_PATH = FINAL_CACHE / "y_test.npy"
RAW_TEST_PATH = FINAL_CACHE / "raw_row_count_test.npy"

PHASE5_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase5_fair_budget_compression_protocol_v3_2.json"
)

B0_PAIR_LOCK = (
    AUDIT / "phase5_B0_pair_locked_v3_2.json"
)

B0_CHECKPOINT_REGISTRY = (
    AUDIT / "phase5_B0_checkpoint_registry_v3_2.csv"
)

PTQ_PREFLIGHT_LOCK = (
    AUDIT
    / "phase5_PTQ_runtime_preflight_locked_v3_2.json"
)

PTQ_SELECTION_LOCK = (
    AUDIT
    / "phase5_PTQ_validation_selection_locked_v3_2.json"
)

PTQ_VERIFIED_CANDIDATES_CSV = (
    AUDIT
    / "phase5_PTQ_validation_sweep_verified_candidates_v3_2.csv"
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

CALIBRATION_INDEX_NPZ = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "shared"
    / "ptq"
    / "phase5_PTQ_calibration_indices_v3_2.npz"
)

AGGREGATE_RUNS_CSV = (
    AUDIT
    / "phase5_PTQ_selected_test_all_runs_v3_2.csv"
)

GROUP_SUMMARY_CSV = (
    AUDIT
    / "phase5_PTQ_selected_test_group_summary_v3_2.csv"
)

AGGREGATE_JSON = (
    AUDIT
    / "phase5_PTQ_selected_test_summary_v3_2.json"
)

COMPLETION_JSON = (
    AUDIT
    / "phase5_PTQ_selected_test_completed_v3_2.json"
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

EXPECTED_KEYS = {
    (
        architecture,
        seed,
    )
    for architecture in ARCHITECTURES
    for seed in SEEDS
}

EXPECTED_SELECTED_SIZES = {
    "tinyml_mlp": 4096,
    "compact_dnn": 16384,
}

EXPECTED_INPUT_FEATURES = 115
EXPECTED_OUTPUT_CLASSES = 3
EXPECTED_TRAIN_ROWS = 1_534_583
EXPECTED_TEST_ROWS = 371_796
EXPECTED_TEST_RAW_ROWS = 1_059_393

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

STRUCTURE_CALIBRATION_ROWS = 257
EVAL_BATCH_SIZE = 16_384
CPU_THREADS = 4
MINIMUM_FREE_DISK_GIB = 1.0

FRAGILITY_MACRO_F1_THRESHOLD = 0.85
FRAGILITY_ATTACK_FNR_THRESHOLD = 0.40

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


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


def replace_with_retry(
    source: Path,
    destination: Path,
) -> None:
    last_error: OSError | None = None

    for attempt in range(
        1,
        WINDOWS_FILE_RETRY_COUNT + 1,
    ):
        try:
            os.replace(
                source,
                destination,
            )
            return
        except PermissionError as error:
            last_error = error

            if (
                attempt
                == WINDOWS_FILE_RETRY_COUNT
            ):
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept the destination file "
        "locked after "
        f"{WINDOWS_FILE_RETRY_COUNT} attempts: "
        f"{destination}"
    ) from last_error


def remove_tree_with_retry(
    path: Path,
) -> None:
    if not path.exists():
        return

    last_error: OSError | None = None

    for attempt in range(
        1,
        WINDOWS_FILE_RETRY_COUNT + 1,
    ):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as error:
            last_error = error

            if (
                attempt
                == WINDOWS_FILE_RETRY_COUNT
            ):
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept an incomplete run "
        "directory locked after "
        f"{WINDOWS_FILE_RETRY_COUNT} attempts: "
        f"{path}"
    ) from last_error


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

    replace_with_retry(
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

    replace_with_retry(
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

    replace_with_retry(
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


def evaluate_test(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
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
        y_test,
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
                    x_test[
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
                    "Test output shape mismatch."
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
                "label": primary_row["label"],
                "fingerprint_precision": (
                    primary_row["precision"]
                ),
                "fingerprint_recall": (
                    primary_row["recall"]
                ),
                "fingerprint_f1": (
                    primary_row["f1"]
                ),
                "fingerprint_support": (
                    primary_row["support"]
                ),
                "fingerprint_fnr": (
                    primary_row[
                        "false_negative_rate"
                    ]
                ),
                "raw_weighted_precision": (
                    weighted_row["precision"]
                ),
                "raw_weighted_recall": (
                    weighted_row["recall"]
                ),
                "raw_weighted_f1": (
                    weighted_row["f1"]
                ),
                "raw_weighted_support": (
                    weighted_row["support"]
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


def output_directory(
    architecture: str,
    seed: int,
) -> Path:
    return (
        ROOT
        / "results"
        / "v2"
        / "phase5_compression"
        / architecture
        / "ptq_selected_test"
        / f"seed_{seed}"
    )


def candidate_directory(
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
            "Existing selected PTQ test run "
            "failed resume validation: "
            f"{run_directory}"
        )

    return {
        "run_id": metrics["run_id"],
        "architecture": (
            metrics["architecture"]
        ),
        "seed": int(metrics["seed"]),
        "calibration_size": int(
            metrics["calibration_size"]
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
        "test_elapsed_seconds": float(
            metrics["test"][
                "elapsed_seconds"
            ]
        ),
        "test_evaluation_count": 1,
        "fragility_gate_triggered": bool(
            metrics[
                "fragility_gate"
            ]["triggered"]
        ),
        "candidate_directory": (
            metrics["source_candidate"][
                "candidate_directory"
            ]
        ),
        "candidate_INT8_checkpoint_sha256": (
            metrics["source_candidate"][
                "INT8_checkpoint_sha256"
            ]
        ),
        "output_directory": str(
            run_directory
        ),
        "test_predictions_sha256": (
            sha256_file(
                run_directory
                / "test_predictions.npz"
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
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    RAW_TEST_PATH,
    PHASE5_PROTOCOL,
    B0_PAIR_LOCK,
    B0_CHECKPOINT_REGISTRY,
    PTQ_PREFLIGHT_LOCK,
    PTQ_SELECTION_LOCK,
    PTQ_VERIFIED_CANDIDATES_CSV,
    MODEL_SOURCE,
    PTQ_ENGINE_PATH,
    SCALER_NPZ,
    CALIBRATION_INDEX_NPZ,
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
            "Selected PTQ test aggregate "
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

PTQ_preflight_lock = read_json(
    PTQ_PREFLIGHT_LOCK
)

PTQ_selection_lock = read_json(
    PTQ_SELECTION_LOCK
)

B0_rows = read_csv(
    B0_CHECKPOINT_REGISTRY
)

verified_candidate_rows = read_csv(
    PTQ_VERIFIED_CANDIDATES_CSV
)

B0_lookup = {
    (
        row["architecture"],
        int(row["seed"]),
    ): row
    for row in B0_rows
}

verified_candidate_lookup = {
    (
        row["architecture"],
        int(row["seed"]),
        int(row["calibration_size"]),
    ): row
    for row in verified_candidate_rows
}

selected_sizes = {
    key: int(value)
    for key, value
    in PTQ_selection_lock[
        "selected_calibration_sizes"
    ].items()
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
    "PTQ_preflight_locked": (
        PTQ_preflight_lock.get(
            "status"
        )
        == "locked"
        and PTQ_preflight_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "PTQ_selection_locked": (
        PTQ_selection_lock.get(
            "status"
        )
        == "locked"
        and PTQ_selection_lock.get(
            "ready_for_selected_PTQ_test_evaluation"
        )
        is True
        and PTQ_selection_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "verified_candidate_hash_matches": (
        PTQ_selection_lock.get(
            "verified_candidates_csv_sha256"
        )
        == sha256_file(
            PTQ_VERIFIED_CANDIDATES_CSV
        )
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
    "selected_sizes_match_expected": (
        selected_sizes
        == EXPECTED_SELECTED_SIZES
    ),
    "B0_registry_complete": (
        set(
            B0_lookup.keys()
        )
        == EXPECTED_KEYS
    ),
    "selected_candidates_verified": (
        all(
            (
                architecture,
                seed,
                selected_sizes[
                    architecture
                ],
            )
            in verified_candidate_lookup
            and (
                verified_candidate_lookup[
                    (
                        architecture,
                        seed,
                        selected_sizes[
                            architecture
                        ],
                    )
                ][
                    "all_checks_passed"
                ].strip().lower()
                == "true"
            )
            for architecture, seed
            in EXPECTED_KEYS
        )
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
        "Selected PTQ test entry gate failed: "
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

x_test = np.load(
    X_TEST_PATH,
    mmap_mode="r",
)

y_test = np.asarray(
    np.load(
        Y_TEST_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

raw_test = np.asarray(
    np.load(
        RAW_TEST_PATH,
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
        "PTQ selected test train-cache "
        "shape mismatch."
    )

if not (
    x_test.shape
    == (
        EXPECTED_TEST_ROWS,
        EXPECTED_INPUT_FEATURES,
    )
    and y_test.shape
    == (
        EXPECTED_TEST_ROWS,
    )
    and raw_test.shape
    == (
        EXPECTED_TEST_ROWS,
    )
):
    raise RuntimeError(
        "PTQ selected test cache shape mismatch."
    )

if int(
    raw_test.sum()
) != EXPECTED_TEST_RAW_ROWS:
    raise RuntimeError(
        "PTQ selected test raw-row total mismatch."
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

calibration_archive = np.load(
    CALIBRATION_INDEX_NPZ
)

model_module = load_module(
    MODEL_SOURCE,
    "phase5_PTQ_selected_test_models",
)

PTQ_engine_module = load_module(
    PTQ_ENGINE_PATH,
    "phase5_PTQ_selected_test_engine",
)

if (
    PTQ_engine_module.ENGINE_VERSION
    != PTQ_ENGINE_VERSION
):
    raise RuntimeError(
        "PTQ-engine version mismatch."
    )

print("=" * 92)
print("PHASE 5 SELECTED PTQ TEST EVALUATION")
print("=" * 92)
print(
    "Runs                            : 10"
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
    "Selected calibration sizes      : "
    f"{selected_sizes}"
)
print(
    "PTQ backend                     : "
    f"{selected_backend}"
)
print(
    "Selection source                : "
    "locked validation-only sweep"
)
print(
    "Validation inference repeated   : False"
)
print(
    "Test evaluation policy          : "
    "exactly once per run"
)
print(
    "Test used for selection         : False"
)
print(
    "Windows file-lock retry         : Enabled"
)
print(
    "Free disk                       : "
    f"{free_disk_gib:.3f} GiB"
)
print()

run_summaries: list[
    dict[str, Any]
] = []

for run_index, (
    architecture,
    seed,
) in enumerate(
    sorted(
        EXPECTED_KEYS,
        key=lambda value: (
            value[0],
            value[1],
        ),
    ),
    start=1,
):
    calibration_size = (
        selected_sizes[
            architecture
        ]
    )

    run_id = (
        f"{architecture}__ptq_selected_test__"
        f"seed_{seed}__cal_{calibration_size}"
    )

    run_directory = output_directory(
        architecture,
        seed,
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

            run_summaries.append(
                summary
            )

            print(
                f"[{run_index}/10] "
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

            if (
                bool(
                    existing_status.get(
                        "test_inference_started",
                        False,
                    )
                )
                or int(
                    existing_status.get(
                        "test_evaluation_count",
                        0,
                    )
                )
                > 0
            ):
                raise RuntimeError(
                    "Incomplete selected PTQ run "
                    "already entered test inference. "
                    "Automatic restart is blocked: "
                    f"{run_directory}"
                )

        remove_tree_with_retry(
            run_directory
        )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    candidate_dir = candidate_directory(
        architecture,
        seed,
        calibration_size,
    )

    candidate_checkpoint_path = (
        candidate_dir
        / "ptq_int8_checkpoint.pt"
    )

    candidate_metrics_path = (
        candidate_dir
        / "metrics.json"
    )

    candidate_manifest_path = (
        candidate_dir
        / "run_manifest.json"
    )

    candidate_status_path = (
        candidate_dir
        / "run_status.json"
    )

    for path in (
        candidate_checkpoint_path,
        candidate_metrics_path,
        candidate_manifest_path,
        candidate_status_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    candidate_metrics = read_json(
        candidate_metrics_path
    )

    candidate_manifest = read_json(
        candidate_manifest_path
    )

    candidate_status = read_json(
        candidate_status_path
    )

    verified_row = (
        verified_candidate_lookup[
            (
                architecture,
                seed,
                calibration_size,
            )
        ]
    )

    candidate_checkpoint_hash = (
        sha256_file(
            candidate_checkpoint_path
        )
    )

    if not (
        candidate_checkpoint_hash
        == verified_row[
            "INT8_checkpoint_sha256"
        ]
        == candidate_status[
            "INT8_checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            f"{run_id}: selected candidate "
            "checkpoint hash mismatch."
        )

    if not (
        candidate_metrics.get(
            "test_access_count"
        )
        == 0
        and candidate_status.get(
            "test_access_count"
        )
        == 0
        and candidate_manifest[
            "selection_scope"
        ][
            "test_access_count"
        ]
        == 0
    ):
        raise RuntimeError(
            f"{run_id}: selected candidate "
            "already accessed test data."
        )

    B0_row = B0_lookup[
        (
            architecture,
            seed,
        )
    ]

    B0_checkpoint_path = Path(
        B0_row["checkpoint_path"]
    )

    B0_checkpoint_hash = (
        sha256_file(
            B0_checkpoint_path
        )
    )

    if (
        B0_checkpoint_hash
        != B0_row[
            "checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            f"{run_id}: B0 checkpoint "
            "hash mismatch."
        )

    B0_checkpoint = torch.load(
        B0_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    candidate_checkpoint = torch.load(
        candidate_checkpoint_path,
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
        B0_checkpoint[
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

    prepared_model = (
        PTQ_engine_module
        .prepare_ptq_model(
            source_model,
            selected_backend,
        )
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

    structure_indices = (
        calibration_indices[
            :STRUCTURE_CALIBRATION_ROWS
        ]
    )

    structure_tensor = torch.from_numpy(
        transform_rows(
            x_train[
                structure_indices
            ],
            mean64,
            scale64,
        )
    )

    with torch.inference_mode():
        prepared_model(
            structure_tensor
        )

    INT8_model = (
        PTQ_engine_module
        .convert_ptq_model(
            prepared_model
        )
    )

    INT8_model.load_state_dict(
        candidate_checkpoint[
            "INT8_model_state_dict"
        ],
        strict=True,
    )

    INT8_model.eval()

    if (
        len(
            PTQ_engine_module
            .static_quantized_linear_modules(
                INT8_model
            )
        )
        != len(
            EXPECTED_WIDTHS[
                architecture
            ]
        )
    ):
        raise RuntimeError(
            f"{run_id}: INT8 Linear "
            "count mismatch."
        )

    if not all(
        dtype == "torch.qint8"
        for dtype in (
            PTQ_engine_module
            .quantized_weight_dtypes(
                INT8_model
            )
        )
    ):
        raise RuntimeError(
            f"{run_id}: INT8 weight "
            "dtype mismatch."
        )

    atomic_json(
        status_path,
        {
            "status": "running",
            "stage": "ready_for_test",
            "run_id": run_id,
            "architecture": architecture,
            "seed": seed,
            "calibration_size": (
                calibration_size
            ),
            "validation_inference_repeated": (
                False
            ),
            "test_evaluation_count": 0,
            "test_inference_started": False,
            "source_candidate_checkpoint_sha256": (
                candidate_checkpoint_hash
            ),
            "started_at_utc": utc_now(),
        },
    )

    atomic_json(
        status_path,
        {
            "status": "running",
            "stage": "test_inference_started",
            "run_id": run_id,
            "architecture": architecture,
            "seed": seed,
            "calibration_size": (
                calibration_size
            ),
            "validation_inference_repeated": (
                False
            ),
            "test_evaluation_count": 1,
            "test_inference_started": True,
            "source_candidate_checkpoint_sha256": (
                candidate_checkpoint_hash
            ),
            "test_started_at_utc": utc_now(),
        },
    )

    (
        test_predictions,
        test_probabilities,
        test_true,
        test_elapsed,
    ) = evaluate_test(
        INT8_model,
        x_test,
        y_test,
        mean64,
        scale64,
        PTQ_engine_module,
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

    metrics_path = (
        run_directory / "metrics.json"
    )

    metrics = {
        "status": "completed",
        "phase": 5,
        "artifact_name": (
            "selected_PTQ_test_evaluation"
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
        "selection": {
            "source": (
                "locked_validation_only_sweep"
            ),
            "architecture_level_selection": (
                True
            ),
            "selected_calibration_size": (
                calibration_size
            ),
            "validation_inference_repeated": (
                False
            ),
            "test_used_for_selection": (
                False
            ),
        },
        "source_candidate": {
            "candidate_directory": (
                str(
                    candidate_dir
                )
            ),
            "INT8_checkpoint_path": (
                str(
                    candidate_checkpoint_path
                )
            ),
            "INT8_checkpoint_sha256": (
                candidate_checkpoint_hash
            ),
            "candidate_metrics_sha256": (
                sha256_file(
                    candidate_metrics_path
                )
            ),
            "candidate_manifest_sha256": (
                sha256_file(
                    candidate_manifest_path
                )
            ),
            "candidate_test_access_count": (
                0
            ),
        },
        "source_B0": {
            "checkpoint_path": (
                str(
                    B0_checkpoint_path
                )
            ),
            "checkpoint_sha256": (
                B0_checkpoint_hash
            ),
        },
        "quantization": {
            "method": (
                "post_training_static_quantization"
            ),
            "backend": (
                selected_backend
            ),
            "quantized_weight_dtype": (
                "torch.qint8"
            ),
            "training_performed": False,
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
            "float_linear_widths": (
                candidate_checkpoint[
                    "float_linear_widths"
                ]
            ),
            "float_state_bytes": int(
                candidate_checkpoint[
                    "float_state_bytes"
                ]
            ),
            "INT8_state_bytes": int(
                candidate_checkpoint[
                    "INT8_state_bytes"
                ]
            ),
            "INT8_to_float_state_size_ratio": (
                int(
                    candidate_checkpoint[
                        "INT8_state_bytes"
                    ]
                )
                / int(
                    candidate_checkpoint[
                        "float_state_bytes"
                    ]
                )
            ),
        },
        "data_access": {
            "train_used_for_new_calibration": (
                False
            ),
            "validation_inference_repeated": (
                False
            ),
            "test_used_for_selection": (
                False
            ),
            "test_evaluation_count": 1,
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
        "artifact_name": (
            "selected_PTQ_test_evaluation"
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
        "selection_lock": (
            file_record(
                PTQ_SELECTION_LOCK
            )
        ),
        "source_candidate_checkpoint": (
            file_record(
                candidate_checkpoint_path
            )
        ),
        "source_candidate_metrics": (
            file_record(
                candidate_metrics_path
            )
        ),
        "source_candidate_manifest": (
            file_record(
                candidate_manifest_path
            )
        ),
        "source_B0_checkpoint": (
            file_record(
                B0_checkpoint_path
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
        "test_policy": {
            "test_evaluation_count": 1,
            "test_used_for_selection": False,
            "validation_inference_repeated": False,
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
            "validation_inference_repeated": (
                False
            ),
            "test_evaluation_count": 1,
            "test_inference_started": True,
            "source_candidate_checkpoint_sha256": (
                candidate_checkpoint_hash
            ),
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

    run_summaries.append(
        {
            "run_id": run_id,
            "architecture": architecture,
            "seed": seed,
            "calibration_size": (
                calibration_size
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
            "float_state_bytes": int(
                candidate_checkpoint[
                    "float_state_bytes"
                ]
            ),
            "INT8_state_bytes": int(
                candidate_checkpoint[
                    "INT8_state_bytes"
                ]
            ),
            "INT8_to_float_state_size_ratio": (
                int(
                    candidate_checkpoint[
                        "INT8_state_bytes"
                    ]
                )
                / int(
                    candidate_checkpoint[
                        "float_state_bytes"
                    ]
                )
            ),
            "test_elapsed_seconds": (
                test_elapsed
            ),
            "test_evaluation_count": 1,
            "fragility_gate_triggered": (
                fragility_triggered
            ),
            "candidate_directory": (
                str(
                    candidate_dir
                )
            ),
            "candidate_INT8_checkpoint_sha256": (
                candidate_checkpoint_hash
            ),
            "output_directory": (
                str(
                    run_directory
                )
            ),
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
            "run_status": "completed_new",
        }
    )

    print(
        f"[{run_index}/10] "
        f"{architecture} seed={seed} "
        f"cal={calibration_size} PTQ | "
        "test_macro_f1="
        f"{test_primary['macro_f1']:.9f} | "
        "raw_weighted="
        f"{test_weighted['macro_f1']:.9f} | "
        "INT8 ratio="
        f"{run_summaries[-1]['INT8_to_float_state_size_ratio']:.4f} | "
        "fragility="
        f"{fragility_triggered}",
        flush=True,
    )

    del B0_checkpoint
    del candidate_checkpoint
    del source_model
    del prepared_model
    del INT8_model
    del test_predictions
    del test_probabilities

calibration_archive.close()

if len(
    run_summaries
) != 10:
    raise RuntimeError(
        "Expected ten completed selected "
        "PTQ test runs."
    )

run_summaries.sort(
    key=lambda row: (
        row["architecture"],
        int(row["seed"]),
    )
)

atomic_csv(
    AGGREGATE_RUNS_CSV,
    run_summaries,
    [
        "run_id",
        "architecture",
        "seed",
        "calibration_size",
        "test_fingerprint_macro_f1",
        "test_fingerprint_accuracy",
        "test_raw_weighted_macro_f1",
        "test_gafgyt_fnr",
        "test_mirai_fnr",
        "float_state_bytes",
        "INT8_state_bytes",
        "INT8_to_float_state_size_ratio",
        "test_elapsed_seconds",
        "test_evaluation_count",
        "fragility_gate_triggered",
        "candidate_directory",
        "candidate_INT8_checkpoint_sha256",
        "output_directory",
        "test_predictions_sha256",
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
    rows = [
        row
        for row in run_summaries
        if row["architecture"]
        == architecture
    ]

    if len(rows) != 5:
        raise RuntimeError(
            "Expected five selected PTQ "
            f"runs for {architecture}."
        )

    metric_names = (
        "test_fingerprint_accuracy",
        "test_fingerprint_macro_f1",
        "test_raw_weighted_macro_f1",
        "test_gafgyt_fnr",
        "test_mirai_fnr",
        "INT8_to_float_state_size_ratio",
        "test_elapsed_seconds",
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
        for metric_name in metric_names
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
        "architecture": architecture,
        "calibration_size": (
            selected_sizes[
                architecture
            ]
        ),
        "run_count": 5,
        "seeds": json.dumps(
            list(SEEDS),
            separators=(
                ",",
                ":",
            ),
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
            ]["std_population"]
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
        "mean_INT8_to_float_state_size_ratio": (
            statistics[
                "INT8_to_float_state_size_ratio"
            ]["mean"]
        ),
        "mean_test_elapsed_seconds": (
            statistics[
                "test_elapsed_seconds"
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
        "calibration_size",
        "run_count",
        "seeds",
        "mean_test_fingerprint_accuracy",
        "mean_test_fingerprint_macro_f1",
        "std_test_fingerprint_macro_f1",
        "min_test_fingerprint_macro_f1",
        "max_test_fingerprint_macro_f1",
        "mean_test_raw_weighted_macro_f1",
        "mean_test_gafgyt_fnr",
        "mean_test_mirai_fnr",
        "mean_INT8_to_float_state_size_ratio",
        "mean_test_elapsed_seconds",
        "test_evaluation_count_per_run",
        "fragility_triggered_in_any_run",
    ],
)

global_checks = {
    "ten_runs_completed": (
        len(run_summaries)
        == 10
    ),
    "two_groups_completed": (
        len(group_rows)
        == 2
    ),
    "test_count_one_all_runs": (
        all(
            int(
                row[
                    "test_evaluation_count"
                ]
            )
            == 1
            for row in run_summaries
        )
    ),
    "selected_sizes_unchanged": (
        selected_sizes
        == EXPECTED_SELECTED_SIZES
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
        "Selected PTQ test global checks "
        "failed: "
        + ", ".join(
            failed_global_checks
        )
    )

aggregate = {
    "status": "completed",
    "phase": 5,
    "artifact_name": (
        "selected_PTQ_test_evaluation"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "PTQ_engine_version": (
        PTQ_ENGINE_VERSION
    ),
    "completed_at_utc": utc_now(),
    "run_count": 10,
    "group_count": 2,
    "selected_calibration_sizes": (
        selected_sizes
    ),
    "selection_policy": {
        "source": (
            "locked_validation_only_sweep"
        ),
        "architecture_level_selection": (
            True
        ),
        "test_used_for_selection": (
            False
        ),
    },
    "test_policy": {
        "test_evaluation_count_per_run": 1,
        "test_inference_total": 10,
        "validation_inference_repeated": (
            False
        ),
    },
    "groups": group_json,
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
    "global_checks": (
        global_checks
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
        "selected_PTQ_test_evaluation"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "PTQ_engine_version": (
        PTQ_ENGINE_VERSION
    ),
    "completed_at_utc": utc_now(),
    "run_count": 10,
    "group_count": 2,
    "selected_calibration_sizes": (
        selected_sizes
    ),
    "test_evaluation_count_per_run": 1,
    "test_inference_total": 10,
    "test_used_for_selection": False,
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
print("PHASE 5 SELECTED PTQ TEST SUMMARY")
print("=" * 92)
print(
    "Runs completed                  : 10"
)
print(
    "Groups completed                : 2"
)
print(
    "Selected calibration sizes      : "
    f"{selected_sizes}"
)
print(
    "Test evaluation count per run   : 1"
)
print(
    "Test used for selection         : False"
)
print(
    "Validation inference repeated   : False"
)
print()

for row in sorted(
    group_rows,
    key=lambda value: (
        value["architecture"]
    ),
):
    print(
        f"{row['architecture']:<16} "
        "PTQ | cal="
        f"{int(row['calibration_size']):>6} | "
        "mean Macro-F1="
        f"{float(row['mean_test_fingerprint_macro_f1']):.9f} | "
        "std="
        f"{float(row['std_test_fingerprint_macro_f1']):.9f} | "
        "raw-weighted="
        f"{float(row['mean_test_raw_weighted_macro_f1']):.9f} | "
        "INT8 ratio="
        f"{float(row['mean_INT8_to_float_state_size_ratio']):.4f} | "
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
    "PHASE 5 SELECTED PTQ TEST EVALUATION COMPLETED"
)
