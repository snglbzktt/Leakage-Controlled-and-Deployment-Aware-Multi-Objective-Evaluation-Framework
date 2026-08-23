from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
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

LOCKED_MATRIX = (
    AUDIT / "phase5_locked_configuration_matrix_v3_2.csv"
)

B0_PAIR_LOCK = (
    AUDIT / "phase5_B0_pair_locked_v3_2.json"
)

B0_CHECKPOINT_REGISTRY = (
    AUDIT / "phase5_B0_checkpoint_registry_v3_2.csv"
)

PRUNING_FP32_FT_LOCK = (
    AUDIT
    / "phase5_pruning_FP32_FT_locked_v3_2.json"
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
    / "phase5_FP32_FT_all_runs_v3_2.csv"
)

GROUP_SUMMARY_CSV = (
    AUDIT
    / "phase5_FP32_FT_group_summary_v3_2.csv"
)

AGGREGATE_JSON = (
    AUDIT
    / "phase5_FP32_FT_summary_v3_2.json"
)

COMPLETION_JSON = (
    AUDIT
    / "phase5_FP32_FT_completed_v3_2.json"
)

OUTPUT_VERIFIED_RUNS_CSV = (
    AUDIT
    / "phase5_FP32_FT_verified_runs_v3_2.csv"
)

OUTPUT_VERIFICATION = (
    AUDIT
    / "phase5_FP32_FT_verification_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase5_FP32_FT_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase5_FP32_FT_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
VARIANT = "FP32-FT"

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

BASELINE_LEARNING_RATE = {
    "tinyml_mlp": 0.003,
    "compact_dnn": 0.001,
}

FINE_TUNING_LEARNING_RATE = {
    architecture: value * 0.1
    for architecture, value
    in BASELINE_LEARNING_RATE.items()
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

HIDDEN_DIMS = {
    "tinyml_mlp": [64, 32],
    "compact_dnn": [128, 64, 32],
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

EXPECTED_VALIDATION_ROWS = 371_797
EXPECTED_TEST_ROWS = 371_796
EXPECTED_VALIDATION_RAW_ROWS = 1_059_390
EXPECTED_TEST_RAW_ROWS = 1_059_393

BATCH_SIZE = 4096
MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0002
WEIGHT_DECAY = 0.0001

EVAL_BATCH_SIZE = 16_384
CPU_THREADS = 4
FLOAT_TOLERANCE = 1e-10


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
) -> bool:
    return math.isclose(
        float(observed),
        float(expected),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    )


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
        "Model output did not contain logits."
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


def infer_validation(
    model: nn.Module,
    x_source: np.ndarray,
    mean64: np.ndarray,
    scale64: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    predictions = np.empty(
        len(x_source),
        dtype=np.int8,
    )

    probabilities = np.empty(
        (
            len(x_source),
            EXPECTED_CLASS_COUNT,
        ),
        dtype=np.float32,
    )

    model.eval()

    started = time.perf_counter()

    with torch.inference_mode():
        for start in range(
            0,
            len(x_source),
            EVAL_BATCH_SIZE,
        ):
            end = min(
                start + EVAL_BATCH_SIZE,
                len(x_source),
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
                    "Validation output shape mismatch."
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


def verify_manifest_artifacts(
    run_directory: Path,
    manifest: dict[str, Any],
) -> bool:
    records = manifest[
        "artifacts"
    ]

    expected_names = {
        str(
            record[
                "relative_path"
            ]
        )
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

    if expected_names != actual_names:
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


def parse_bool(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(
        value
    ).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Cannot parse boolean value: {value}"
    )


def history_selection_matches(
    history_rows: list[dict[str, str]],
    expected_best_epoch: int,
    expected_best_value: float,
) -> bool:
    if not history_rows:
        return False

    best_value = -math.inf
    best_epoch = 0
    wait = 0

    for row in history_rows:
        epoch = int(row["epoch"])

        value = float(
            row[
                "validation_macro_f1"
            ]
        )

        improved = (
            best_epoch == 0
            or value
            > (
                best_value
                + EARLY_STOPPING_MIN_DELTA
            )
        )

        if parse_bool(
            row["improved"]
        ) != improved:
            return False

        if improved:
            best_value = value
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if int(
            row[
                "epochs_without_improvement"
            ]
        ) != wait:
            return False

    return (
        best_epoch
        == expected_best_epoch
        and close_enough(
            best_value,
            expected_best_value,
        )
    )


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
    X_VALIDATION_PATH,
    Y_VALIDATION_PATH,
    RAW_VALIDATION_PATH,
    Y_TEST_PATH,
    RAW_TEST_PATH,
    PHASE5_PROTOCOL,
    LOCKED_MATRIX,
    B0_PAIR_LOCK,
    B0_CHECKPOINT_REGISTRY,
    PRUNING_FP32_FT_LOCK,
    MODEL_SOURCE,
    SCALER_NPZ,
    CLASS_WEIGHTS_NPZ,
    PREPROCESSING_LOCK,
    AGGREGATE_RUNS_CSV,
    GROUP_SUMMARY_CSV,
    AGGREGATE_JSON,
    COMPLETION_JSON,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_VERIFIED_RUNS_CSV,
    OUTPUT_VERIFICATION,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "FP32-FT verification artifact "
            "already exists; refusing to "
            f"overwrite: {output_path}"
        )

protocol = read_json(
    PHASE5_PROTOCOL
)

B0_pair_lock = read_json(
    B0_PAIR_LOCK
)

pruning_FT_lock = read_json(
    PRUNING_FP32_FT_LOCK
)

preprocessing_lock = read_json(
    PREPROCESSING_LOCK
)

aggregate = read_json(
    AGGREGATE_JSON
)

completion = read_json(
    COMPLETION_JSON
)

matrix_rows = read_csv(
    LOCKED_MATRIX
)

B0_rows = read_csv(
    B0_CHECKPOINT_REGISTRY
)

aggregate_rows = read_csv(
    AGGREGATE_RUNS_CSV
)

group_rows = read_csv(
    GROUP_SUMMARY_CSV
)

expected_keys = {
    (
        architecture,
        seed,
    )
    for architecture in ARCHITECTURES
    for seed in SEEDS
}

aggregate_lookup = {
    (
        row["architecture"],
        int(row["seed"]),
    ): row
    for row in aggregate_rows
}

B0_lookup = {
    (
        row["architecture"],
        int(row["seed"]),
    ): row
    for row in B0_rows
}

matrix_lookup = {
    (
        row["architecture"],
        int(row["seed"]),
    ): row
    for row in matrix_rows
    if row["architecture"]
    in ARCHITECTURES
    and row["variant"]
    == VARIANT
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
    "pruning_FP32_FT_locked": (
        pruning_FT_lock.get(
            "status"
        )
        == "locked"
        and pruning_FT_lock.get(
            "ready_for_remaining_phase5_compression_variants"
        )
        is True
        and pruning_FT_lock.get(
            "all_checks_passed"
        )
        is True
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
    "aggregate_completed": (
        aggregate.get("status")
        == "completed"
        and aggregate.get(
            "run_count"
        )
        == 10
        and aggregate.get(
            "all_integrity_checks_passed"
        )
        is True
    ),
    "completion_completed": (
        completion.get("status")
        == "completed"
        and completion.get(
            "run_count"
        )
        == 10
        and completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "aggregate_hash_matches_completion": (
        completion.get(
            "aggregate_json_sha256"
        )
        == sha256_file(
            AGGREGATE_JSON
        )
    ),
    "aggregate_runs_hash_matches_completion": (
        completion.get(
            "aggregate_runs_csv_sha256"
        )
        == sha256_file(
            AGGREGATE_RUNS_CSV
        )
    ),
    "group_summary_hash_matches_completion": (
        completion.get(
            "group_summary_csv_sha256"
        )
        == sha256_file(
            GROUP_SUMMARY_CSV
        )
    ),
    "ten_aggregate_rows": (
        len(aggregate_rows)
        == 10
    ),
    "two_group_rows": (
        len(group_rows)
        == 2
    ),
    "aggregate_matrix_matches": (
        set(
            aggregate_lookup.keys()
        )
        == expected_keys
    ),
    "B0_matrix_matches": (
        set(
            B0_lookup.keys()
        )
        == expected_keys
    ),
    "locked_matrix_matches": (
        set(
            matrix_lookup.keys()
        )
        == expected_keys
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
        "FP32-FT verification entry "
        "gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

x_validation = np.load(
    X_VALIDATION_PATH,
    mmap_mode="r",
)

y_validation_cache = np.asarray(
    np.load(
        Y_VALIDATION_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

raw_validation_cache = np.asarray(
    np.load(
        RAW_VALIDATION_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

y_test_cache = np.asarray(
    np.load(
        Y_TEST_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

raw_test_cache = np.asarray(
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

with np.load(
    CLASS_WEIGHTS_NPZ
) as values:
    class_weights32 = np.array(
        values[
            "class_weights_float32"
        ],
        copy=True,
    )

cache_checks = {
    "validation_shapes_match": (
        x_validation.shape
        == (
            EXPECTED_VALIDATION_ROWS,
            EXPECTED_FEATURE_COUNT,
        )
        and y_validation_cache.shape
        == (
            EXPECTED_VALIDATION_ROWS,
        )
        and raw_validation_cache.shape
        == (
            EXPECTED_VALIDATION_ROWS,
        )
    ),
    "test_shapes_match": (
        y_test_cache.shape
        == (
            EXPECTED_TEST_ROWS,
        )
        and raw_test_cache.shape
        == (
            EXPECTED_TEST_ROWS,
        )
    ),
    "raw_totals_match": (
        int(
            raw_validation_cache.sum()
        )
        == EXPECTED_VALIDATION_RAW_ROWS
        and int(
            raw_test_cache.sum()
        )
        == EXPECTED_TEST_RAW_ROWS
    ),
    "preprocessing_shapes_match": (
        scaler_mean64.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
        and scaler_scale64.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
        and class_weights32.shape
        == (
            EXPECTED_CLASS_COUNT,
        )
    ),
}

failed_cache_checks = [
    name
    for name, passed
    in cache_checks.items()
    if not passed
]

if failed_cache_checks:
    raise RuntimeError(
        "FP32-FT verification cache "
        "checks failed: "
        + ", ".join(
            failed_cache_checks
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

model_module = load_module(
    MODEL_SOURCE,
    "phase5_fp32_ft_verify_models",
)

print("=" * 92)
print("PHASE 5 FP32-FT INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Runs                            : 10"
)
print(
    "Checkpoint loading              : Yes"
)
print(
    "Optimizer-state loading         : Yes"
)
print(
    "Validation model inference      : Yes"
)
print(
    "Test model inference            : No"
)
print(
    "Test verification source        : Saved predictions only"
)
print()

verified_rows: list[
    dict[str, Any]
] = []

for index, key in enumerate(
    sorted(
        expected_keys,
        key=lambda value: (
            value[0],
            value[1],
        ),
    ),
    start=1,
):
    architecture, seed = key

    aggregate_row = (
        aggregate_lookup[key]
    )

    B0_row = B0_lookup[
        key
    ]

    matrix_row = matrix_lookup[
        key
    ]

    run_directory = Path(
        aggregate_row[
            "output_directory"
        ]
    )

    paths = {
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
        "history": (
            run_directory
            / "training_history.csv"
        ),
        "checkpoint": (
            run_directory
            / "best_checkpoint.pt"
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

    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    status = read_json(
        paths["status"]
    )

    manifest = read_json(
        paths["manifest"]
    )

    metrics = read_json(
        paths["metrics"]
    )

    history_rows = read_csv(
        paths["history"]
    )

    checkpoint = torch.load(
        paths["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )

    B0_checkpoint_path = Path(
        B0_row[
            "checkpoint_path"
        ]
    )

    if not B0_checkpoint_path.exists():
        raise FileNotFoundError(
            B0_checkpoint_path
        )

    B0_checkpoint = torch.load(
        B0_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_symbol = (
        manifest["model_symbol"]
    )

    model = instantiate_model(
        model_module,
        model_symbol,
        architecture,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINE_TUNING_LEARNING_RATE[
            architecture
        ],
        weight_decay=WEIGHT_DECAY,
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    with np.load(
        paths[
            "validation_predictions"
        ]
    ) as values:
        saved_validation_true = np.array(
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
        paths[
            "test_predictions"
        ]
    ) as values:
        saved_test_true = np.array(
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
        repeated_validation_pred,
        repeated_validation_prob,
        repeated_validation_seconds,
    ) = infer_validation(
        model,
        x_validation,
        scaler_mean64,
        scaler_scale64,
    )

    repeated_validation_primary = metric_view(
        y_validation_cache,
        repeated_validation_pred,
        repeated_validation_prob,
        sample_weight=None,
    )

    saved_validation_primary = metric_view(
        saved_validation_true,
        saved_validation_pred,
        saved_validation_prob,
        sample_weight=None,
    )

    saved_validation_weighted = metric_view(
        saved_validation_true,
        saved_validation_pred,
        saved_validation_prob,
        sample_weight=saved_validation_raw,
    )

    saved_test_primary = metric_view(
        saved_test_true,
        saved_test_pred,
        saved_test_prob,
        sample_weight=None,
    )

    saved_test_weighted = metric_view(
        saved_test_true,
        saved_test_pred,
        saved_test_prob,
        sample_weight=saved_test_raw,
    )

    expected_learning_rate = (
        FINE_TUNING_LEARNING_RATE[
            architecture
        ]
    )

    optimizer_group = (
        optimizer.param_groups[0]
    )

    checks = {
        "status_completed": (
            status.get("status")
            == "completed"
            and status.get("stage")
            == "completed"
        ),
        "identity_matches": (
            status.get(
                "architecture"
            )
            == architecture
            and status.get(
                "variant"
            )
            == VARIANT
            and int(
                status.get("seed")
            )
            == seed
            and manifest.get(
                "architecture"
            )
            == architecture
            and manifest.get(
                "variant"
            )
            == VARIANT
            and int(
                manifest.get("seed")
            )
            == seed
            and metrics.get(
                "architecture"
            )
            == architecture
            and metrics.get(
                "variant"
            )
            == VARIANT
            and int(
                metrics.get("seed")
            )
            == seed
            and checkpoint.get(
                "architecture"
            )
            == architecture
            and checkpoint.get(
                "variant"
            )
            == VARIANT
            and int(
                checkpoint.get("seed")
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
        "manifest_artifacts_match": (
            verify_manifest_artifacts(
                run_directory,
                manifest,
            )
        ),
        "aggregate_hashes_match": (
            aggregate_row[
                "metrics_sha256"
            ]
            == sha256_file(
                paths["metrics"]
            )
            and aggregate_row[
                "run_manifest_sha256"
            ]
            == sha256_file(
                paths["manifest"]
            )
        ),
        "status_hashes_match": (
            status.get(
                "metrics_sha256"
            )
            == sha256_file(
                paths["metrics"]
            )
            and status.get(
                "run_manifest_sha256"
            )
            == sha256_file(
                paths["manifest"]
            )
            and status.get(
                "test_predictions_sha256"
            )
            == sha256_file(
                paths[
                    "test_predictions"
                ]
            )
        ),
        "B0_checkpoint_hash_matches": (
            sha256_file(
                B0_checkpoint_path
            )
            == B0_row[
                "checkpoint_sha256"
            ]
            == metrics["source"][
                "source_B0_checkpoint_sha256"
            ]
            == checkpoint[
                "source_B0_checkpoint_sha256"
            ]
            == manifest[
                "source_B0_checkpoint"
            ]["sha256"]
            == aggregate_row[
                "source_checkpoint_sha256"
            ]
        ),
        "B0_state_loads_into_fresh_model": (
            set(
                B0_checkpoint[
                    "model_state_dict"
                ].keys()
            )
            == set(
                model.state_dict().keys()
            )
        ),
        "topology_matches": (
            linear_widths(
                model
            )
            == EXPECTED_WIDTHS[
                architecture
            ]
            and checkpoint[
                "linear_widths"
            ]
            == EXPECTED_WIDTHS[
                architecture
            ]
        ),
        "complexity_matches": (
            parameter_count(
                model
            )
            == int(
                metrics[
                    "model_complexity"
                ][
                    "parameter_count"
                ]
            )
            == int(
                manifest[
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
            == int(
                aggregate_row[
                    "parameter_count"
                ]
            )
            and linear_macs_per_sample(
                model
            )
            == int(
                metrics[
                    "model_complexity"
                ][
                    "linear_macs_per_sample"
                ]
            )
            == int(
                manifest[
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
            == int(
                aggregate_row[
                    "linear_macs_per_sample"
                ]
            )
            and raw_tensor_bytes(
                model
            )
            == int(
                metrics[
                    "model_complexity"
                ][
                    "raw_tensor_bytes"
                ]
            )
            == int(
                aggregate_row[
                    "raw_tensor_bytes"
                ]
            )
        ),
        "serialized_checkpoint_size_matches": (
            int(
                paths["checkpoint"]
                .stat()
                .st_size
            )
            == int(
                metrics[
                    "model_complexity"
                ][
                    "serialized_checkpoint_bytes"
                ]
            )
            == int(
                aggregate_row[
                    "serialized_checkpoint_bytes"
                ]
            )
        ),
        "locked_training_parameters_match": (
            metrics[
                "training"
            ]["optimizer"]
            == "AdamW"
            and int(
                metrics[
                    "training"
                ]["batch_size"]
            )
            == BATCH_SIZE
            and int(
                metrics[
                    "training"
                ]["max_epochs"]
            )
            == MAX_EPOCHS
            and int(
                metrics[
                    "training"
                ][
                    "early_stopping_patience"
                ]
            )
            == EARLY_STOPPING_PATIENCE
            and close_enough(
                metrics[
                    "training"
                ][
                    "early_stopping_min_delta"
                ],
                EARLY_STOPPING_MIN_DELTA,
            )
            and close_enough(
                metrics[
                    "training"
                ]["learning_rate"],
                expected_learning_rate,
            )
            and close_enough(
                metrics[
                    "training"
                ]["weight_decay"],
                WEIGHT_DECAY,
            )
            and close_enough(
                matrix_row[
                    "learning_rate"
                ],
                expected_learning_rate,
            )
            and int(
                matrix_row[
                    "batch_size"
                ]
            )
            == BATCH_SIZE
            and int(
                matrix_row[
                    "max_epochs"
                ]
            )
            == MAX_EPOCHS
        ),
        "optimizer_state_loads": (
            len(
                optimizer.state
            )
            > 0
            and close_enough(
                optimizer_group["lr"],
                expected_learning_rate,
            )
            and close_enough(
                optimizer_group[
                    "weight_decay"
                ],
                WEIGHT_DECAY,
            )
        ),
        "history_row_count_matches": (
            len(history_rows)
            == int(
                metrics[
                    "training"
                ][
                    "epochs_completed"
                ]
            )
            == int(
                aggregate_row[
                    "epochs_completed"
                ]
            )
            and 1
            <= len(history_rows)
            <= MAX_EPOCHS
        ),
        "history_selection_matches": (
            history_selection_matches(
                history_rows,
                int(
                    checkpoint[
                        "best_epoch"
                    ]
                ),
                float(
                    checkpoint[
                        "best_validation_macro_f1"
                    ]
                ),
            )
        ),
        "best_selection_values_match": (
            int(
                checkpoint[
                    "best_epoch"
                ]
            )
            == int(
                metrics[
                    "selection"
                ]["best_epoch"]
            )
            == int(
                manifest[
                    "best_epoch"
                ]
            )
            == int(
                aggregate_row[
                    "best_epoch"
                ]
            )
            and close_enough(
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
                checkpoint[
                    "best_validation_macro_f1"
                ],
                manifest[
                    "best_validation_macro_f1"
                ],
            )
            and close_enough(
                checkpoint[
                    "best_validation_macro_f1"
                ],
                aggregate_row[
                    "best_validation_macro_f1"
                ],
            )
        ),
        "saved_validation_shapes_match": (
            saved_validation_true.shape
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
        "saved_test_shapes_match": (
            saved_test_true.shape
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
        "saved_labels_match_cache": (
            np.array_equal(
                saved_validation_true,
                y_validation_cache,
            )
            and np.array_equal(
                saved_test_true,
                y_test_cache,
            )
        ),
        "saved_raw_counts_match_cache": (
            np.array_equal(
                saved_validation_raw,
                raw_validation_cache,
            )
            and np.array_equal(
                saved_test_raw,
                raw_test_cache,
            )
        ),
        "saved_probabilities_finite": (
            np.isfinite(
                saved_validation_prob
            ).all()
            and np.isfinite(
                saved_test_prob
            ).all()
        ),
        "saved_probabilities_normalized": (
            np.allclose(
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
        "repeated_validation_predictions_exact": (
            np.array_equal(
                repeated_validation_pred,
                saved_validation_pred,
            )
        ),
        "repeated_validation_probabilities_exact": (
            np.array_equal(
                repeated_validation_prob,
                saved_validation_prob,
            )
        ),
        "repeated_validation_metrics_match": (
            metrics_match(
                repeated_validation_primary,
                metrics[
                    "validation"
                ][
                    "primary_fingerprint_level"
                ],
            )
        ),
        "saved_validation_primary_metrics_match": (
            metrics_match(
                saved_validation_primary,
                metrics[
                    "validation"
                ][
                    "primary_fingerprint_level"
                ],
            )
        ),
        "saved_validation_weighted_metrics_match": (
            metrics_match(
                saved_validation_weighted,
                metrics[
                    "validation"
                ][
                    "secondary_raw_record_weighted"
                ],
            )
        ),
        "saved_test_primary_metrics_match": (
            metrics_match(
                saved_test_primary,
                metrics[
                    "test"
                ][
                    "primary_fingerprint_level"
                ],
            )
        ),
        "saved_test_weighted_metrics_match": (
            metrics_match(
                saved_test_weighted,
                metrics[
                    "test"
                ][
                    "secondary_raw_record_weighted"
                ],
            )
        ),
        "aggregate_metrics_match": (
            close_enough(
                aggregate_row[
                    "test_fingerprint_macro_f1"
                ],
                saved_test_primary[
                    "macro_f1"
                ],
            )
            and close_enough(
                aggregate_row[
                    "test_fingerprint_accuracy"
                ],
                saved_test_primary[
                    "accuracy"
                ],
            )
            and close_enough(
                aggregate_row[
                    "test_raw_weighted_macro_f1"
                ],
                saved_test_weighted[
                    "macro_f1"
                ],
            )
            and close_enough(
                aggregate_row[
                    "test_gafgyt_fnr"
                ],
                saved_test_primary[
                    "per_class"
                ]["gafgyt"][
                    "false_negative_rate"
                ],
            )
            and close_enough(
                aggregate_row[
                    "test_mirai_fnr"
                ],
                saved_test_primary[
                    "per_class"
                ]["mirai"][
                    "false_negative_rate"
                ],
            )
        ),
        "test_count_is_one": (
            int(
                status[
                    "test_evaluation_count"
                ]
            )
            == 1
            and int(
                metrics[
                    "test_evaluation_count"
                ]
            )
            == 1
            and int(
                metrics[
                    "data_access"
                ][
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
                aggregate_row[
                    "test_evaluation_count"
                ]
            )
            == 1
        ),
        "test_not_used_for_selection": (
            metrics[
                "selection"
            ][
                "test_used_for_selection"
            ]
            is False
            and metrics[
                "data_access"
            ][
                "test_used_for_selection"
            ]
            is False
            and manifest[
                "fit_scope"
            ][
                "test_used_for_selection"
            ]
            is False
        ),
        "raw_weights_not_used_for_fit": (
            metrics[
                "training"
            ][
                "raw_occurrence_weight_used"
            ]
            is False
            and metrics[
                "data_access"
            ][
                "raw_occurrence_weight_used_for_fit"
            ]
            is False
            and manifest[
                "fit_scope"
            ][
                "raw_occurrence_weight_used_for_fit"
            ]
            is False
        ),
        "fragility_flag_matches": (
            bool(
                metrics[
                    "fragility_gate"
                ]["triggered"]
            )
            == bool(
                status[
                    "fragility_gate_triggered"
                ]
            )
            == parse_bool(
                aggregate_row[
                    "fragility_gate_triggered"
                ]
            )
        ),
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed_checks:
        raise RuntimeError(
            f"{architecture} seed {seed} "
            "FP32-FT verification failed: "
            + ", ".join(
                failed_checks
            )
        )

    verified_rows.append(
        {
            "architecture": architecture,
            "variant": VARIANT,
            "seed": seed,
            "best_epoch": int(
                checkpoint[
                    "best_epoch"
                ]
            ),
            "epochs_completed": (
                len(history_rows)
            ),
            "best_validation_macro_f1": (
                repeated_validation_primary[
                    "macro_f1"
                ]
            ),
            "test_fingerprint_macro_f1": (
                saved_test_primary[
                    "macro_f1"
                ]
            ),
            "test_fingerprint_accuracy": (
                saved_test_primary[
                    "accuracy"
                ]
            ),
            "test_raw_weighted_macro_f1": (
                saved_test_weighted[
                    "macro_f1"
                ]
            ),
            "test_gafgyt_fnr": (
                saved_test_primary[
                    "per_class"
                ]["gafgyt"][
                    "false_negative_rate"
                ]
            ),
            "test_mirai_fnr": (
                saved_test_primary[
                    "per_class"
                ]["mirai"][
                    "false_negative_rate"
                ]
            ),
            "parameter_count": (
                parameter_count(
                    model
                )
            ),
            "linear_macs_per_sample": (
                linear_macs_per_sample(
                    model
                )
            ),
            "validation_repeat_seconds": (
                repeated_validation_seconds
            ),
            "test_evaluation_count": 1,
            "test_model_inference_repeated": False,
            "fragility_gate_triggered": (
                bool(
                    metrics[
                        "fragility_gate"
                    ]["triggered"]
                )
            ),
            "run_directory": (
                str(
                    run_directory
                )
            ),
            "checkpoint_sha256": (
                sha256_file(
                    paths["checkpoint"]
                )
            ),
            "metrics_sha256": (
                sha256_file(
                    paths["metrics"]
                )
            ),
            "run_manifest_sha256": (
                sha256_file(
                    paths["manifest"]
                )
            ),
            "all_checks_passed": True,
        }
    )

    print(
        f"[{index}/10] "
        f"{architecture} seed={seed} "
        "FP32-FT | "
        "checkpoint=loaded | "
        "validation=exact | "
        "test inference=False | "
        "fragility="
        f"{bool(metrics['fragility_gate']['triggered'])} | "
        "all checks=True",
        flush=True,
    )

    del model
    del optimizer
    del checkpoint
    del B0_checkpoint
    del repeated_validation_pred
    del repeated_validation_prob
    del saved_validation_pred
    del saved_validation_prob
    del saved_test_pred
    del saved_test_prob

atomic_csv(
    OUTPUT_VERIFIED_RUNS_CSV,
    verified_rows,
    [
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
        "validation_repeat_seconds",
        "test_evaluation_count",
        "test_model_inference_repeated",
        "fragility_gate_triggered",
        "run_directory",
        "checkpoint_sha256",
        "metrics_sha256",
        "run_manifest_sha256",
        "all_checks_passed",
    ],
)

group_lookup = {
    row["architecture"]: row
    for row in group_rows
}

verified_groups: list[
    dict[str, Any]
] = []

for architecture in ARCHITECTURES:
    rows = [
        row
        for row in verified_rows
        if row["architecture"]
        == architecture
    ]

    saved_group = group_lookup[
        architecture
    ]

    statistics = {
        "test_fingerprint_accuracy": (
            aggregate_statistics(
                [
                    row[
                        "test_fingerprint_accuracy"
                    ]
                    for row in rows
                ]
            )
        ),
        "test_fingerprint_macro_f1": (
            aggregate_statistics(
                [
                    row[
                        "test_fingerprint_macro_f1"
                    ]
                    for row in rows
                ]
            )
        ),
        "test_raw_weighted_macro_f1": (
            aggregate_statistics(
                [
                    row[
                        "test_raw_weighted_macro_f1"
                    ]
                    for row in rows
                ]
            )
        ),
        "test_gafgyt_fnr": (
            aggregate_statistics(
                [
                    row[
                        "test_gafgyt_fnr"
                    ]
                    for row in rows
                ]
            )
        ),
        "test_mirai_fnr": (
            aggregate_statistics(
                [
                    row[
                        "test_mirai_fnr"
                    ]
                    for row in rows
                ]
            )
        ),
    }

    group_checks = {
        "run_count_matches": (
            len(rows)
            == 5
            and int(
                saved_group[
                    "run_count"
                ]
            )
            == 5
        ),
        "variant_matches": (
            saved_group[
                "variant"
            ]
            == VARIANT
        ),
        "mean_accuracy_matches": (
            close_enough(
                saved_group[
                    "mean_test_fingerprint_accuracy"
                ],
                statistics[
                    "test_fingerprint_accuracy"
                ]["mean"],
            )
        ),
        "mean_macro_f1_matches": (
            close_enough(
                saved_group[
                    "mean_test_fingerprint_macro_f1"
                ],
                statistics[
                    "test_fingerprint_macro_f1"
                ]["mean"],
            )
        ),
        "std_macro_f1_matches": (
            close_enough(
                saved_group[
                    "std_test_fingerprint_macro_f1"
                ],
                statistics[
                    "test_fingerprint_macro_f1"
                ][
                    "std_population"
                ],
            )
        ),
        "min_macro_f1_matches": (
            close_enough(
                saved_group[
                    "min_test_fingerprint_macro_f1"
                ],
                statistics[
                    "test_fingerprint_macro_f1"
                ]["minimum"],
            )
        ),
        "max_macro_f1_matches": (
            close_enough(
                saved_group[
                    "max_test_fingerprint_macro_f1"
                ],
                statistics[
                    "test_fingerprint_macro_f1"
                ]["maximum"],
            )
        ),
        "mean_raw_weighted_matches": (
            close_enough(
                saved_group[
                    "mean_test_raw_weighted_macro_f1"
                ],
                statistics[
                    "test_raw_weighted_macro_f1"
                ]["mean"],
            )
        ),
        "mean_gafgyt_fnr_matches": (
            close_enough(
                saved_group[
                    "mean_test_gafgyt_fnr"
                ],
                statistics[
                    "test_gafgyt_fnr"
                ]["mean"],
            )
        ),
        "mean_mirai_fnr_matches": (
            close_enough(
                saved_group[
                    "mean_test_mirai_fnr"
                ],
                statistics[
                    "test_mirai_fnr"
                ]["mean"],
            )
        ),
        "test_count_one_matches": (
            int(
                saved_group[
                    "test_evaluation_count_per_run"
                ]
            )
            == 1
        ),
        "fragility_any_matches": (
            parse_bool(
                saved_group[
                    "fragility_triggered_in_any_run"
                ]
            )
            == any(
                row[
                    "fragility_gate_triggered"
                ]
                for row in rows
            )
        ),
    }

    failed_group_checks = [
        name
        for name, passed
        in group_checks.items()
        if not passed
    ]

    if failed_group_checks:
        raise RuntimeError(
            f"{architecture} FP32-FT "
            "group verification failed: "
            + ", ".join(
                failed_group_checks
            )
        )

    verified_groups.append(
        {
            "architecture": architecture,
            "variant": VARIANT,
            "run_count": 5,
            "aggregate_metrics": (
                statistics
            ),
            "fragility_triggered_in_any_run": (
                any(
                    row[
                        "fragility_gate_triggered"
                    ]
                    for row in rows
                )
            ),
            "all_checks_passed": True,
        }
    )

global_checks = {
    "ten_runs_verified": (
        len(
            verified_rows
        )
        == 10
    ),
    "all_run_checks_passed": (
        all(
            row[
                "all_checks_passed"
            ]
            for row in verified_rows
        )
    ),
    "two_groups_verified": (
        len(
            verified_groups
        )
        == 2
    ),
    "all_group_checks_passed": (
        all(
            row[
                "all_checks_passed"
            ]
            for row in verified_groups
        )
    ),
    "test_count_one_all_runs": (
        all(
            row[
                "test_evaluation_count"
            ]
            == 1
            for row in verified_rows
        )
    ),
    "test_inference_not_repeated": (
        all(
            row[
                "test_model_inference_repeated"
            ]
            is False
            for row in verified_rows
        )
    ),
    "no_fragility": (
        all(
            row[
                "fragility_gate_triggered"
            ]
            is False
            for row in verified_rows
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
        "FP32-FT global verification "
        "failed: "
        + ", ".join(
            failed_global_checks
        )
    )

verification = {
    "status": "passed",
    "phase": 5,
    "artifact_name": (
        "FP32_FT_from_locked_B0_"
        "independent_verification"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "run_count": 10,
    "group_count": 2,
    "verification_policy": {
        "checkpoint_loading_performed": True,
        "optimizer_state_loading_performed": True,
        "validation_model_inference_repeated": True,
        "test_model_inference_repeated": False,
        "saved_test_metrics_recomputed": True,
        "test_evaluation_count_verified_per_run": 1,
    },
    "verified_runs_csv": (
        file_record(
            OUTPUT_VERIFIED_RUNS_CSV
        )
    ),
    "verified_groups": (
        verified_groups
    ),
    "entry_checks": (
        entry_checks
    ),
    "cache_checks": (
        cache_checks
    ),
    "global_checks": (
        global_checks
    ),
    "source_artifacts": {
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
        "aggregate_json": (
            file_record(
                AGGREGATE_JSON
            )
        ),
        "completion_json": (
            file_record(
                COMPLETION_JSON
            )
        ),
        "B0_pair_lock": (
            file_record(
                B0_PAIR_LOCK
            )
        ),
        "B0_checkpoint_registry": (
            file_record(
                B0_CHECKPOINT_REGISTRY
            )
        ),
        "pruning_FP32_FT_lock": (
            file_record(
                PRUNING_FP32_FT_LOCK
            )
        ),
    },
    "ready_for_DQ_QAT_and_PTQ_branches": True,
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
        "FP32_FT_from_locked_B0"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "run_count": 10,
    "group_count": 2,
    "test_evaluation_count_per_run": 1,
    "test_model_inference_repeated_by_verifier": False,
    "fragility_triggered_in_any_run": False,
    "verified_runs_csv": str(
        OUTPUT_VERIFIED_RUNS_CSV
    ),
    "verified_runs_csv_sha256": (
        sha256_file(
            OUTPUT_VERIFIED_RUNS_CSV
        )
    ),
    "verification_report": str(
        OUTPUT_VERIFICATION
    ),
    "verification_report_sha256": (
        sha256_file(
            OUTPUT_VERIFICATION
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
    "aggregate_json": str(
        AGGREGATE_JSON
    ),
    "aggregate_json_sha256": (
        sha256_file(
            AGGREGATE_JSON
        )
    ),
    "ready_for_DQ_QAT_and_PTQ_branches": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK,
    lock,
)

lock_manifest = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "FP32_FT_from_locked_B0"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            PHASE5_PROTOCOL
        ),
        file_record(
            LOCKED_MATRIX
        ),
        file_record(
            B0_PAIR_LOCK
        ),
        file_record(
            B0_CHECKPOINT_REGISTRY
        ),
        file_record(
            PRUNING_FP32_FT_LOCK
        ),
        file_record(
            MODEL_SOURCE
        ),
        file_record(
            SCALER_NPZ
        ),
        file_record(
            CLASS_WEIGHTS_NPZ
        ),
        file_record(
            PREPROCESSING_LOCK
        ),
        file_record(
            AGGREGATE_RUNS_CSV
        ),
        file_record(
            GROUP_SUMMARY_CSV
        ),
        file_record(
            AGGREGATE_JSON
        ),
        file_record(
            COMPLETION_JSON
        ),
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_VERIFIED_RUNS_CSV
        ),
        file_record(
            OUTPUT_VERIFICATION
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "run_count": 10,
    "test_evaluation_count_per_run": 1,
    "test_model_inference_repeated_by_verifier": False,
    "ready_for_DQ_QAT_and_PTQ_branches": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

post_checks = {
    "verified_runs_csv_exists": (
        OUTPUT_VERIFIED_RUNS_CSV.exists()
    ),
    "verification_exists": (
        OUTPUT_VERIFICATION.exists()
    ),
    "lock_exists": (
        OUTPUT_LOCK.exists()
    ),
    "lock_manifest_exists": (
        OUTPUT_LOCK_MANIFEST.exists()
    ),
    "verification_hash_matches_lock": (
        read_json(
            OUTPUT_LOCK
        )[
            "verification_report_sha256"
        ]
        == sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "verified_runs_hash_matches_lock": (
        read_json(
            OUTPUT_LOCK
        )[
            "verified_runs_csv_sha256"
        ]
        == sha256_file(
            OUTPUT_VERIFIED_RUNS_CSV
        )
    ),
    "ready_for_quantization_branches": (
        read_json(
            OUTPUT_LOCK
        ).get(
            "ready_for_DQ_QAT_and_PTQ_branches"
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
        "FP32-FT post-lock checks failed: "
        + ", ".join(
            failed_post_checks
        )
    )

print()
print("=" * 92)
print("PHASE 5 FP32-FT VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Runs independently verified     : 10"
)
print(
    "Groups independently verified   : 2"
)
print(
    "Checkpoint loading              : PASSED"
)
print(
    "Optimizer-state loading         : PASSED"
)
print(
    "Independent validation inference: PASSED"
)
print(
    "Saved test metrics recomputed   : PASSED"
)
print(
    "Test model inference repeated   : False"
)
print(
    "Test evaluation count per run   : 1"
)
print(
    "Fragility                       : False"
)
print(
    "FP32-FT status                  : LOCKED"
)
print(
    "Ready for DQ/QAT/PTQ branches   : True"
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
    "PHASE 5 FP32-FT VERIFIED AND LOCKED"
)
