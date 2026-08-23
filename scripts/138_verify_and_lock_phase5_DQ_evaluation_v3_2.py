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
    AUDIT
    / "phase5_locked_configuration_matrix_v3_2.csv"
)

B0_PAIR_LOCK = (
    AUDIT
    / "phase5_B0_pair_locked_v3_2.json"
)

B0_CHECKPOINT_REGISTRY = (
    AUDIT
    / "phase5_B0_checkpoint_registry_v3_2.csv"
)

FP32_FT_LOCK = (
    AUDIT
    / "phase5_FP32_FT_locked_v3_2.json"
)

DQ_PREFLIGHT_LOCK = (
    AUDIT
    / "phase5_DQ_runtime_preflight_locked_v3_2.json"
)

DQ_PREFLIGHT_MATRIX = (
    AUDIT
    / "phase5_DQ_runtime_preflight_matrix_v3_2.csv"
)

MODEL_SOURCE = (
    ROOT
    / "src"
    / "models"
    / "nbaiot_models.py"
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

AGGREGATE_RUNS_CSV = (
    AUDIT
    / "phase5_DQ_all_runs_v3_2.csv"
)

GROUP_SUMMARY_CSV = (
    AUDIT
    / "phase5_DQ_group_summary_v3_2.csv"
)

AGGREGATE_JSON = (
    AUDIT
    / "phase5_DQ_summary_v3_2.json"
)

COMPLETION_JSON = (
    AUDIT
    / "phase5_DQ_completed_v3_2.json"
)

OUTPUT_VERIFIED_RUNS_CSV = (
    AUDIT
    / "phase5_DQ_verified_runs_v3_2.csv"
)

OUTPUT_VERIFICATION = (
    AUDIT
    / "phase5_DQ_verification_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase5_DQ_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase5_DQ_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = (
    "phase5_fair_budget_compression_v3_2"
)

VARIANT = "DQ"

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

EXPECTED_INPUT_FEATURES = 115
EXPECTED_OUTPUT_CLASSES = 3

HIDDEN_DIMS = {
    "tinyml_mlp": [64, 32],
    "compact_dnn": [128, 64, 32],
}

EXPECTED_FLOAT_WIDTHS = {
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

EXPECTED_VALIDATION_ROWS = 371_797
EXPECTED_TEST_ROWS = 371_796
EXPECTED_VALIDATION_RAW_ROWS = 1_059_390
EXPECTED_TEST_RAW_ROWS = 1_059_393

EVAL_BATCH_SIZE = 16_384
CPU_THREADS = 4
FLOAT_TOLERANCE = 1e-10


warnings.filterwarnings(
    "ignore",
    message=(
        "torch.ao.quantization is deprecated.*"
    ),
    category=DeprecationWarning,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "TypedStorage is deprecated.*"
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

    with path.open(
        "rb"
    ) as handle:
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

    module = (
        importlib.util.module_from_spec(
            specification
        )
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


def get_quantize_dynamic():
    candidate = getattr(
        torch.ao.quantization,
        "quantize_dynamic",
        None,
    )

    if candidate is not None:
        return candidate

    candidate = getattr(
        torch.quantization,
        "quantize_dynamic",
        None,
    )

    if candidate is not None:
        return candidate

    raise RuntimeError(
        "Dynamic quantization API is unavailable."
    )


def quantize_model(
    model: nn.Module,
) -> nn.Module:
    model.eval()

    quantized = get_quantize_dynamic()(
        model,
        {
            nn.Linear,
        },
        dtype=torch.qint8,
        inplace=False,
    )

    quantized.eval()

    return quantized


def float_linear_widths(
    model: nn.Module,
) -> list[list[int]]:
    return [
        [
            int(module.in_features),
            int(module.out_features),
        ]
        for module in model.modules()
        if isinstance(
            module,
            nn.Linear,
        )
    ]


def float_parameter_count(
    model: nn.Module,
) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )


def float_linear_macs_per_sample(
    model: nn.Module,
) -> int:
    return int(
        sum(
            module.in_features
            * module.out_features
            for module in model.modules()
            if isinstance(
                module,
                nn.Linear,
            )
        )
    )


def quantized_dynamic_linear_count(
    model: nn.Module,
) -> int:
    return sum(
        1
        for module in model.modules()
        if (
            "quantized.dynamic"
            in type(module).__module__.lower()
            and type(module).__name__.lower()
            == "linear"
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
            EXPECTED_OUTPUT_CLASSES,
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
    FP32_FT_LOCK,
    DQ_PREFLIGHT_LOCK,
    DQ_PREFLIGHT_MATRIX,
    MODEL_SOURCE,
    SCALER_NPZ,
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
            "DQ verification artifact already "
            "exists; refusing to overwrite: "
            f"{output_path}"
        )

protocol = read_json(
    PHASE5_PROTOCOL
)

B0_pair_lock = read_json(
    B0_PAIR_LOCK
)

FP32_FT_lock = read_json(
    FP32_FT_LOCK
)

DQ_preflight_lock = read_json(
    DQ_PREFLIGHT_LOCK
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

preflight_rows = read_csv(
    DQ_PREFLIGHT_MATRIX
)

aggregate_rows = read_csv(
    AGGREGATE_RUNS_CSV
)

group_rows = read_csv(
    GROUP_SUMMARY_CSV
)

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

preflight_lookup = {
    (
        row["architecture"],
        int(row["seed"]),
    ): row
    for row in preflight_rows
}

DQ_matrix_keys = {
    (
        row["architecture"],
        int(row["seed"]),
    )
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
    "FP32_FT_locked": (
        FP32_FT_lock.get("status")
        == "locked"
        and FP32_FT_lock.get(
            "ready_for_DQ_QAT_and_PTQ_branches"
        )
        is True
        and FP32_FT_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "DQ_preflight_locked": (
        DQ_preflight_lock.get(
            "status"
        )
        == "locked"
        and DQ_preflight_lock.get(
            "ready_for_DQ_evaluation"
        )
        is True
        and DQ_preflight_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "DQ_preflight_matrix_hash_matches": (
        DQ_preflight_lock.get(
            "matrix_csv_sha256"
        )
        == sha256_file(
            DQ_PREFLIGHT_MATRIX
        )
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
        == EXPECTED_KEYS
    ),
    "B0_matrix_matches": (
        set(
            B0_lookup.keys()
        )
        == EXPECTED_KEYS
    ),
    "preflight_matrix_matches": (
        set(
            preflight_lookup.keys()
        )
        == EXPECTED_KEYS
    ),
    "locked_DQ_matrix_matches": (
        DQ_matrix_keys
        == EXPECTED_KEYS
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
        "DQ verification entry gate failed: "
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
    mean64 = np.array(
        values["mean_float64"],
        copy=True,
    )

    scale64 = np.array(
        values["scale_float64"],
        copy=True,
    )

cache_checks = {
    "validation_shapes_match": (
        x_validation.shape
        == (
            EXPECTED_VALIDATION_ROWS,
            EXPECTED_INPUT_FEATURES,
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
    "scaler_shape_matches": (
        mean64.shape
        == (
            EXPECTED_INPUT_FEATURES,
        )
        and scale64.shape
        == (
            EXPECTED_INPUT_FEATURES,
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
        "DQ verification cache checks failed: "
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

model_module = load_module(
    MODEL_SOURCE,
    "phase5_DQ_verification_models",
)

active_engine = (
    torch.backends.quantized.engine
)

if (
    active_engine
    != DQ_preflight_lock.get(
        "quantization_engine"
    )
):
    raise RuntimeError(
        "Active quantization engine does "
        "not match the locked DQ preflight."
    )

print("=" * 92)
print("PHASE 5 DYNAMIC QUANTIZATION INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Runs                            : 10"
)
print(
    "DQ checkpoint loading           : Yes"
)
print(
    "Independent validation inference: Yes"
)
print(
    "Test model inference            : No"
)
print(
    "Test verification source        : Saved predictions only"
)
print(
    "Quantization engine             : "
    f"{active_engine}"
)
print()

verified_rows: list[
    dict[str, Any]
] = []

for index, key in enumerate(
    sorted(
        EXPECTED_KEYS,
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

    preflight_row = (
        preflight_lookup[key]
    )

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
        "DQ_checkpoint": (
            run_directory
            / "dq_checkpoint.pt"
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

    DQ_checkpoint = torch.load(
        paths["DQ_checkpoint"],
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
        manifest[
            "model_symbol"
        ]
    )

    float_model = instantiate_model(
        model_module,
        model_symbol,
        architecture,
    )

    float_model.load_state_dict(
        B0_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    float_model.eval()

    DQ_model = quantize_model(
        float_model
    )

    DQ_model.load_state_dict(
        DQ_checkpoint[
            "quantized_state_dict"
        ],
        strict=True,
    )

    DQ_model.eval()

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
        DQ_model,
        x_validation,
        mean64,
        scale64,
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
            and DQ_checkpoint.get(
                "architecture"
            )
            == architecture
            and DQ_checkpoint.get(
                "variant"
            )
            == VARIANT
            and int(
                DQ_checkpoint.get("seed")
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
            and aggregate_row[
                "DQ_checkpoint_sha256"
            ]
            == sha256_file(
                paths[
                    "DQ_checkpoint"
                ]
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
                "DQ_checkpoint_sha256"
            )
            == sha256_file(
                paths[
                    "DQ_checkpoint"
                ]
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
            == preflight_row[
                "source_checkpoint_sha256"
            ]
            == metrics["source"][
                "source_B0_checkpoint_sha256"
            ]
            == DQ_checkpoint[
                "source_B0_checkpoint_sha256"
            ]
            == manifest[
                "source_B0_checkpoint"
            ]["sha256"]
            == aggregate_row[
                "source_checkpoint_sha256"
            ]
        ),
        "DQ_engine_matches_lock": (
            metrics[
                "quantization"
            ]["engine"]
            == active_engine
            == DQ_checkpoint[
                "quantization_engine"
            ]
            == DQ_preflight_lock[
                "quantization_engine"
            ]
        ),
        "DQ_method_matches": (
            metrics[
                "quantization"
            ]["method"]
            == (
                "post_training_dynamic_quantization"
            )
            and DQ_checkpoint[
                "quantization_method"
            ]
            == (
                "post_training_dynamic_quantization"
            )
            and metrics[
                "quantization"
            ]["weight_dtype"]
            == "torch.qint8"
            and DQ_checkpoint[
                "quantized_dtype"
            ]
            == "torch.qint8"
        ),
        "float_topology_matches": (
            float_linear_widths(
                float_model
            )
            == EXPECTED_FLOAT_WIDTHS[
                architecture
            ]
            and DQ_checkpoint[
                "float_linear_widths"
            ]
            == EXPECTED_FLOAT_WIDTHS[
                architecture
            ]
        ),
        "all_linear_layers_quantized": (
            quantized_dynamic_linear_count(
                DQ_model
            )
            == len(
                EXPECTED_FLOAT_WIDTHS[
                    architecture
                ]
            )
            == int(
                metrics[
                    "model_complexity"
                ][
                    "DQ_linear_count"
                ]
            )
            == int(
                DQ_checkpoint[
                    "DQ_linear_count"
                ]
            )
        ),
        "complexity_matches": (
            float_parameter_count(
                float_model
            )
            == int(
                metrics[
                    "model_complexity"
                ][
                    "float_parameter_count"
                ]
            )
            == int(
                DQ_checkpoint[
                    "float_parameter_count"
                ]
            )
            == int(
                aggregate_row[
                    "float_parameter_count"
                ]
            )
            and float_linear_macs_per_sample(
                float_model
            )
            == int(
                metrics[
                    "model_complexity"
                ][
                    "float_linear_macs_per_sample"
                ]
            )
            == int(
                DQ_checkpoint[
                    "float_linear_macs_per_sample"
                ]
            )
            == int(
                aggregate_row[
                    "float_linear_macs_per_sample"
                ]
            )
            and int(
                metrics[
                    "model_complexity"
                ]["float_state_bytes"]
            )
            == int(
                DQ_checkpoint[
                    "float_state_bytes"
                ]
            )
            == int(
                aggregate_row[
                    "float_state_bytes"
                ]
            )
            and int(
                metrics[
                    "model_complexity"
                ]["DQ_state_bytes"]
            )
            == int(
                DQ_checkpoint[
                    "DQ_state_bytes"
                ]
            )
            == int(
                aggregate_row[
                    "DQ_state_bytes"
                ]
            )
            and close_enough(
                metrics[
                    "model_complexity"
                ][
                    "DQ_to_float_state_size_ratio"
                ],
                aggregate_row[
                    "DQ_to_float_state_size_ratio"
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
                EXPECTED_OUTPUT_CLASSES,
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
                EXPECTED_OUTPUT_CLASSES,
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
                    "validation_fingerprint_macro_f1"
                ],
                saved_validation_primary[
                    "macro_f1"
                ],
            )
            and close_enough(
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
                    "test_inference_count"
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
        "no_training_or_calibration": (
            metrics[
                "quantization"
            ][
                "training_performed"
            ]
            is False
            and metrics[
                "quantization"
            ][
                "calibration_used"
            ]
            is False
            and manifest[
                "fit_scope"
            ][
                "training_performed"
            ]
            is False
            and manifest[
                "fit_scope"
            ][
                "calibration_used"
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
            "DQ verification failed: "
            + ", ".join(
                failed_checks
            )
        )

    verified_rows.append(
        {
            "architecture": architecture,
            "variant": VARIANT,
            "seed": seed,
            "validation_fingerprint_macro_f1": (
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
            "float_parameter_count": (
                float_parameter_count(
                    float_model
                )
            ),
            "float_linear_macs_per_sample": (
                float_linear_macs_per_sample(
                    float_model
                )
            ),
            "DQ_to_float_state_size_ratio": (
                float(
                    metrics[
                        "model_complexity"
                    ][
                        "DQ_to_float_state_size_ratio"
                    ]
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
            "DQ_checkpoint_sha256": (
                sha256_file(
                    paths[
                        "DQ_checkpoint"
                    ]
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
        f"{architecture} seed={seed} DQ | "
        "checkpoint=loaded | "
        "validation=exact | "
        "test inference=False | "
        "fragility="
        f"{bool(metrics['fragility_gate']['triggered'])} | "
        "all checks=True",
        flush=True,
    )

    del float_model
    del DQ_model
    del B0_checkpoint
    del DQ_checkpoint
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
        "validation_fingerprint_macro_f1",
        "test_fingerprint_macro_f1",
        "test_fingerprint_accuracy",
        "test_raw_weighted_macro_f1",
        "test_gafgyt_fnr",
        "test_mirai_fnr",
        "float_parameter_count",
        "float_linear_macs_per_sample",
        "DQ_to_float_state_size_ratio",
        "validation_repeat_seconds",
        "test_evaluation_count",
        "test_model_inference_repeated",
        "fragility_gate_triggered",
        "run_directory",
        "DQ_checkpoint_sha256",
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
        "validation_fingerprint_macro_f1": (
            aggregate_statistics(
                [
                    row[
                        "validation_fingerprint_macro_f1"
                    ]
                    for row in rows
                ]
            )
        ),
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
        "DQ_to_float_state_size_ratio": (
            aggregate_statistics(
                [
                    row[
                        "DQ_to_float_state_size_ratio"
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
        "mean_validation_matches": (
            close_enough(
                saved_group[
                    "mean_validation_fingerprint_macro_f1"
                ],
                statistics[
                    "validation_fingerprint_macro_f1"
                ]["mean"],
            )
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
        "mean_state_ratio_matches": (
            close_enough(
                saved_group[
                    "mean_DQ_to_float_state_size_ratio"
                ],
                statistics[
                    "DQ_to_float_state_size_ratio"
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
            f"{architecture} DQ group "
            "verification failed: "
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
    "compact_DNN_has_no_fragility": (
        all(
            row[
                "fragility_gate_triggered"
            ]
            is False
            for row in verified_rows
            if row["architecture"]
            == "compact_dnn"
        )
    ),
    "TinyML_MLP_has_fragility": (
        any(
            row[
                "fragility_gate_triggered"
            ]
            is True
            for row in verified_rows
            if row["architecture"]
            == "tinyml_mlp"
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
        "DQ global verification failed: "
        + ", ".join(
            failed_global_checks
        )
    )

verification = {
    "status": "passed",
    "phase": 5,
    "artifact_name": (
        "dynamic_quantization_"
        "independent_verification"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "run_count": 10,
    "group_count": 2,
    "quantization_engine": (
        active_engine
    ),
    "verification_policy": {
        "DQ_checkpoint_loading_performed": True,
        "independent_validation_inference_repeated": True,
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
        "FP32_FT_lock": (
            file_record(
                FP32_FT_LOCK
            )
        ),
        "DQ_preflight_lock": (
            file_record(
                DQ_PREFLIGHT_LOCK
            )
        ),
        "DQ_preflight_matrix": (
            file_record(
                DQ_PREFLIGHT_MATRIX
            )
        ),
    },
    "ready_for_QAT_and_PTQ_branches": True,
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
        "dynamic_quantization_evaluation"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "run_count": 10,
    "group_count": 2,
    "quantization_engine": (
        active_engine
    ),
    "test_evaluation_count_per_run": 1,
    "test_model_inference_repeated_by_verifier": False,
    "fragility_triggered_in_any_run": True,
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
    "ready_for_QAT_and_PTQ_branches": True,
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
        "dynamic_quantization_evaluation"
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
            FP32_FT_LOCK
        ),
        file_record(
            DQ_PREFLIGHT_LOCK
        ),
        file_record(
            DQ_PREFLIGHT_MATRIX
        ),
        file_record(
            MODEL_SOURCE
        ),
        file_record(
            SCALER_NPZ
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
    "ready_for_QAT_and_PTQ_branches": True,
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
    "ready_for_QAT_and_PTQ": (
        read_json(
            OUTPUT_LOCK
        ).get(
            "ready_for_QAT_and_PTQ_branches"
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
        "DQ post-lock checks failed: "
        + ", ".join(
            failed_post_checks
        )
    )

print()
print("=" * 92)
print("PHASE 5 DYNAMIC QUANTIZATION VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Runs independently verified     : 10"
)
print(
    "Groups independently verified   : 2"
)
print(
    "DQ checkpoint loading           : PASSED"
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
    "Compact-DNN fragility           : False"
)
print(
    "TinyML MLP fragility            : True"
)
print(
    "DQ evaluation status            : LOCKED"
)
print(
    "Ready for QAT/PTQ branches      : True"
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
    "PHASE 5 DYNAMIC QUANTIZATION VERIFIED AND LOCKED"
)
