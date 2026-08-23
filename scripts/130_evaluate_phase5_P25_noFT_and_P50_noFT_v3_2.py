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

B0_PAIR_LOCK = (
    AUDIT / "phase5_B0_pair_locked_v3_2.json"
)

B0_CHECKPOINT_REGISTRY = (
    AUDIT / "phase5_B0_checkpoint_registry_v3_2.csv"
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

PRUNING_SOURCES_VERIFIED_LOCK = (
    AUDIT
    / "phase5_physical_pruning_sources_verified_locked_v3_2.json"
)

PRUNING_SOURCES_REGISTRY = (
    AUDIT
    / "phase5_physical_pruning_source_checkpoint_registry_v3_2.csv"
)

MODEL_SOURCE = (
    ROOT / "src" / "models" / "nbaiot_models.py"
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
    / "phase5_pruning_noFT_evaluation_all_runs_v3_2.csv"
)

GROUP_SUMMARY_CSV = (
    AUDIT
    / "phase5_pruning_noFT_evaluation_group_summary_v3_2.csv"
)

AGGREGATE_JSON = (
    AUDIT
    / "phase5_pruning_noFT_evaluation_summary_v3_2.json"
)

COMPLETION_JSON = (
    AUDIT
    / "phase5_pruning_noFT_evaluation_completed_v3_2.json"
)

PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
ENGINE_VERSION = "phase5_physical_pruning_engine_v3_2"

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

VARIANTS = (
    "P25-noFT",
    "P50-noFT",
)

VARIANT_DIRECTORY = {
    "P25-noFT": "p25_noft",
    "P50-noFT": "p50_noft",
}

VARIANT_RATIO = {
    "P25-noFT": 0.25,
    "P50-noFT": 0.50,
}

EXPECTED_WIDTHS = {
    "tinyml_mlp": {
        "P25-noFT": [
            [115, 48],
            [48, 24],
            [24, 3],
        ],
        "P50-noFT": [
            [115, 32],
            [32, 16],
            [16, 3],
        ],
    },
    "compact_dnn": {
        "P25-noFT": [
            [115, 96],
            [96, 48],
            [48, 24],
            [24, 3],
        ],
        "P50-noFT": [
            [115, 64],
            [64, 32],
            [32, 16],
            [16, 3],
        ],
    },
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

EVAL_BATCH_SIZE = 16_384
CPU_THREADS = 4
MINIMUM_FREE_DISK_GIB = 1.5

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
                    "Evaluation logit shape "
                    "mismatch."
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

    elapsed = (
        time.perf_counter()
        - started
    )

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
    rows: list[
        dict[str, Any]
    ] = []

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
    rows: list[
        dict[str, Any]
    ] = []

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
        / VARIANT_DIRECTORY[
            variant
        ]
        / f"seed_{seed}"
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
    FINAL_CACHE,
    X_VALIDATION_PATH,
    Y_VALIDATION_PATH,
    RAW_VALIDATION_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    RAW_TEST_PATH,
    PHASE5_PROTOCOL,
    LOCKED_MATRIX,
    B0_PAIR_LOCK,
    B0_CHECKPOINT_REGISTRY,
    PRUNING_ENGINE_LOCK,
    PRUNING_ENGINE_PATH,
    PRUNING_SOURCES_VERIFIED_LOCK,
    PRUNING_SOURCES_REGISTRY,
    MODEL_SOURCE,
    SCALER_NPZ,
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
            "NoFT evaluation aggregate "
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

B0_pair_lock = read_json(
    B0_PAIR_LOCK
)

engine_lock = read_json(
    PRUNING_ENGINE_LOCK
)

sources_verified_lock = read_json(
    PRUNING_SOURCES_VERIFIED_LOCK
)

matrix_rows = read_csv(
    LOCKED_MATRIX
)

B0_rows = read_csv(
    B0_CHECKPOINT_REGISTRY
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
        B0_pair_lock.get(
            "status"
        )
        == "locked"
        and B0_pair_lock.get(
            "ready_for_branch_source_generation"
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
    "pruning_engine_locked": (
        engine_lock.get(
            "status"
        )
        == "locked"
        and engine_lock.get(
            "ready_for_P25_noFT_and_P50_noFT_generation"
        )
        is True
    ),
    "engine_hash_matches": (
        engine_lock.get(
            "engine_source_sha256"
        )
        == sha256_file(
            PRUNING_ENGINE_PATH
        )
    ),
    "pruning_sources_independently_verified": (
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
    "matrix_has_twenty_noFT_rows": (
        len(
            selected_matrix_rows
        )
        == 20
    ),
    "source_registry_has_twenty_rows": (
        len(source_rows)
        == 20
    ),
    "B0_registry_has_ten_rows": (
        len(B0_rows)
        == 10
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
        "NoFT evaluation entry gate "
        "failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

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

source_lookup = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    ): row
    for row in source_rows
}

matrix_keys = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    )
    for row in selected_matrix_rows
}

if set(
    source_lookup.keys()
) != expected_keys:
    raise RuntimeError(
        "Source registry does not match "
        "the locked 20-run noFT matrix."
    )

if matrix_keys != expected_keys:
    raise RuntimeError(
        "Locked matrix does not match "
        "the expected 20-run noFT matrix."
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
    "scaler_shape_matches": (
        mean64.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
        and scale64.shape
        == (
            EXPECTED_FEATURE_COUNT,
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
        "NoFT evaluation cache checks "
        "failed: "
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
    "phase5_noft_evaluation_models",
)

engine_module = load_module(
    PRUNING_ENGINE_PATH,
    "phase5_noft_evaluation_engine",
)

if (
    engine_module.ENGINE_VERSION
    != ENGINE_VERSION
):
    raise RuntimeError(
        "Pruning-engine version mismatch."
    )

print("=" * 92)
print("PHASE 5 P25-noFT AND P50-noFT EVALUATION")
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
    "Seeds                           : "
    f"{list(SEEDS)}"
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
    "Test evaluation policy          : "
    "exactly once per run"
)
print(
    "Free disk                       : "
    f"{free_disk_gib:.3f} GiB"
)
print()

all_run_rows: list[
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
    run_id = (
        f"{architecture}__"
        f"{variant.lower().replace('-', '_')}__"
        f"seed_{seed}"
    )

    source_row = source_lookup[
        (
            architecture,
            variant,
            seed,
        )
    ]

    run_directory = output_directory(
        architecture,
        variant,
        seed,
    )

    if run_directory.exists():
        raise FileExistsError(
            "NoFT evaluation output "
            "directory already exists. "
            "Refusing to risk a repeated "
            "test evaluation: "
            f"{run_directory}"
        )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    status_path = (
        run_directory
        / "run_status.json"
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

    initial_status = {
        "status": "running",
        "stage": (
            "validation_pending"
        ),
        "run_id": run_id,
        "architecture": architecture,
        "variant": variant,
        "seed": seed,
        "test_evaluation_count": 0,
        "test_inference_started": False,
        "created_at_utc": utc_now(),
    }

    atomic_json(
        status_path,
        initial_status,
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
        VARIANT_RATIO[
            variant
        ],
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

    model.eval()

    expected_widths = (
        EXPECTED_WIDTHS[
            architecture
        ][variant]
    )

    source_checks = {
        "source_identity_matches": (
            source_checkpoint.get(
                "architecture"
            )
            == architecture
            and source_checkpoint.get(
                "variant"
            )
            == variant
            and int(
                source_checkpoint.get(
                    "seed"
                )
            )
            == seed
        ),
        "source_metadata_matches_regenerated": (
            source_checkpoint[
                "pruning_metadata"
            ]
            == regenerated_metadata
        ),
        "compacted_widths_match": (
            regenerated_metadata[
                "compacted_linear_widths"
            ]
            == expected_widths
            and engine_module.inspect_linear_widths(
                model
            )
            == expected_widths
        ),
        "physical_compaction_true": (
            regenerated_metadata[
                "physical_compaction"
            ]
            is True
            and regenerated_metadata[
                "mask_only_pruning"
            ]
            is False
        ),
        "direct_B0_derivation_true": (
            regenerated_metadata[
                "generated_directly_from_B0"
            ]
            is True
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
            f"{run_id}: source checks "
            "failed: "
            + ", ".join(
                failed_source_checks
            )
        )

    validation_pred, (
        validation_prob
    ), validation_true, (
        validation_seconds
    ) = evaluate_model(
        model,
        x_validation,
        y_validation,
        mean64,
        scale64,
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
            validation_pred.astype(
                np.int8
            )
        ),
        probabilities=(
            validation_prob.astype(
                np.float32
            )
        ),
        raw_row_count=(
            raw_validation.astype(
                np.int64
            )
        ),
    )

    validation_primary = metric_view(
        validation_true,
        validation_pred,
        validation_prob,
        sample_weight=None,
    )

    validation_weighted = metric_view(
        validation_true,
        validation_pred,
        validation_prob,
        sample_weight=raw_validation,
    )

    atomic_json(
        status_path,
        {
            **initial_status,
            "stage": (
                "validation_complete"
            ),
            "validation_predictions_sha256": (
                sha256_file(
                    validation_predictions_path
                )
            ),
            "validation_macro_f1": (
                validation_primary[
                    "macro_f1"
                ]
            ),
            "updated_at_utc": (
                utc_now()
            ),
        },
    )

    atomic_json(
        status_path,
        {
            **initial_status,
            "stage": (
                "test_inference_started"
            ),
            "test_evaluation_count": 1,
            "test_inference_started": True,
            "validation_predictions_sha256": (
                sha256_file(
                    validation_predictions_path
                )
            ),
            "validation_macro_f1": (
                validation_primary[
                    "macro_f1"
                ]
            ),
            "test_started_at_utc": (
                utc_now()
            ),
        },
    )

    test_pred, test_prob, (
        test_true
    ), test_seconds = evaluate_model(
        model,
        x_test,
        y_test,
        mean64,
        scale64,
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
            test_pred.astype(
                np.int8
            )
        ),
        probabilities=(
            test_prob.astype(
                np.float32
            )
        ),
        raw_row_count=(
            raw_test.astype(
                np.int64
            )
        ),
    )

    atomic_json(
        status_path,
        {
            **initial_status,
            "stage": (
                "test_predictions_saved"
            ),
            "test_evaluation_count": 1,
            "test_inference_started": True,
            "validation_predictions_sha256": (
                sha256_file(
                    validation_predictions_path
                )
            ),
            "test_predictions_sha256": (
                sha256_file(
                    test_predictions_path
                )
            ),
            "updated_at_utc": (
                utc_now()
            ),
        },
    )

    test_primary = metric_view(
        test_true,
        test_pred,
        test_prob,
        sample_weight=None,
    )

    test_weighted = metric_view(
        test_true,
        test_pred,
        test_prob,
        sample_weight=raw_test,
    )

    fragility_reasons: list[str] = []

    if (
        test_primary["macro_f1"]
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
        / "validation_confusion_matrix.csv"
    )

    validation_weighted_confusion_path = (
        run_directory
        / "validation_raw_weighted_confusion_matrix.csv"
    )

    validation_per_class_path = (
        run_directory
        / "validation_per_class_metrics.csv"
    )

    test_confusion_path = (
        run_directory
        / "test_confusion_matrix.csv"
    )

    test_weighted_confusion_path = (
        run_directory
        / "test_raw_weighted_confusion_matrix.csv"
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

    metrics_path = (
        run_directory
        / "metrics.json"
    )

    metrics = {
        "status": "completed",
        "phase": 5,
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "engine_version": (
            ENGINE_VERSION
        ),
        "run_id": run_id,
        "architecture": architecture,
        "variant": variant,
        "seed": seed,
        "evaluation_mode": (
            "fixed_no_fine_tuning"
        ),
        "selection": {
            "configuration_predeclared": True,
            "validation_used_for_selection": False,
            "test_used_for_selection": False,
            "checkpoint_selected_by": (
                "locked_physical_pruning_source"
            ),
        },
        "source": {
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
            "pruning_ratio": (
                VARIANT_RATIO[
                    variant
                ]
            ),
            "compacted_linear_widths": (
                expected_widths
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
            "serialized_source_checkpoint_bytes": (
                int(
                    source_checkpoint_path
                    .stat()
                    .st_size
                )
            ),
        },
        "validation": {
            "primary_fingerprint_level": (
                validation_primary
            ),
            "secondary_raw_record_weighted": (
                validation_weighted
            ),
            "inference_seconds": (
                validation_seconds
            ),
        },
        "test": {
            "primary_fingerprint_level": (
                test_primary
            ),
            "secondary_raw_record_weighted": (
                test_weighted
            ),
            "inference_seconds": (
                test_seconds
            ),
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
        "data_access": {
            "train_access_count": 0,
            "validation_inference_count": 1,
            "test_inference_count": 1,
            "test_used_for_selection": False,
        },
        "completed_at_utc": utc_now(),
    }

    atomic_json(
        metrics_path,
        metrics,
    )

    artifact_paths = [
        validation_predictions_path,
        test_predictions_path,
        validation_confusion_path,
        validation_weighted_confusion_path,
        validation_per_class_path,
        test_confusion_path,
        test_weighted_confusion_path,
        test_per_class_path,
        metrics_path,
    ]

    manifest_path = (
        run_directory
        / "run_manifest.json"
    )

    manifest = {
        "status": "completed",
        "phase": 5,
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "engine_version": (
            ENGINE_VERSION
        ),
        "run_id": run_id,
        "architecture": architecture,
        "variant": variant,
        "seed": seed,
        "evaluation_mode": (
            "fixed_no_fine_tuning"
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
        "verified_sources_lock": (
            file_record(
                PRUNING_SOURCES_VERIFIED_LOCK
            )
        ),
        "model_complexity": {
            "linear_widths": (
                expected_widths
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
            "serialized_source_checkpoint_bytes": (
                int(
                    source_checkpoint_path
                    .stat()
                    .st_size
                )
            ),
        },
        "fit_scope": {
            "fine_tuning_performed": False,
            "train_access_count": 0,
            "validation_inference_count": 1,
            "test_evaluation_count": 1,
            "test_used_for_selection": False,
        },
        "fragility_gate_triggered": (
            fragility_triggered
        ),
        "artifacts": [
            file_record(
                path,
                relative_to=run_directory,
            )
            for path in artifact_paths
        ],
        "all_integrity_checks_passed": True,
        "completed_at_utc": utc_now(),
    }

    atomic_json(
        manifest_path,
        manifest,
    )

    completed_status = {
        "status": "completed",
        "stage": "completed",
        "run_id": run_id,
        "architecture": architecture,
        "variant": variant,
        "seed": seed,
        "test_evaluation_count": 1,
        "test_inference_started": True,
        "validation_predictions_sha256": (
            sha256_file(
                validation_predictions_path
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
        "test_fingerprint_macro_f1": (
            test_primary[
                "macro_f1"
            ]
        ),
        "test_raw_weighted_macro_f1": (
            test_weighted[
                "macro_f1"
            ]
        ),
        "fragility_gate_triggered": (
            fragility_triggered
        ),
        "completed_at_utc": utc_now(),
    }

    atomic_json(
        status_path,
        completed_status,
    )

    all_run_rows.append(
        {
            "run_id": run_id,
            "architecture": architecture,
            "variant": variant,
            "seed": seed,
            "pruning_ratio": (
                VARIANT_RATIO[
                    variant
                ]
            ),
            "compacted_linear_widths": (
                json.dumps(
                    expected_widths,
                    separators=(
                        ",",
                        ":",
                    ),
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
            "serialized_source_checkpoint_bytes": (
                int(
                    source_checkpoint_path
                    .stat()
                    .st_size
                )
            ),
            "validation_fingerprint_macro_f1": (
                validation_primary[
                    "macro_f1"
                ]
            ),
            "validation_raw_weighted_macro_f1": (
                validation_weighted[
                    "macro_f1"
                ]
            ),
            "test_fingerprint_accuracy": (
                test_primary[
                    "accuracy"
                ]
            ),
            "test_fingerprint_macro_f1": (
                test_primary[
                    "macro_f1"
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
            "validation_inference_seconds": (
                validation_seconds
            ),
            "test_inference_seconds": (
                test_seconds
            ),
            "test_evaluation_count": 1,
            "test_used_for_selection": False,
            "fine_tuning_performed": False,
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
            "status": "completed",
        }
    )

    print(
        f"[{run_index}/20] "
        f"{architecture} seed={seed} "
        f"{variant} | "
        "val_macro_f1="
        f"{validation_primary['macro_f1']:.9f} | "
        "test_macro_f1="
        f"{test_primary['macro_f1']:.9f} | "
        "raw_weighted="
        f"{test_weighted['macro_f1']:.9f} | "
        "fragility="
        f"{fragility_triggered}",
        flush=True,
    )

    del model
    del B0_model
    del source_checkpoint
    del B0_checkpoint
    del validation_pred
    del validation_prob
    del test_pred
    del test_prob

if len(
    all_run_rows
) != 20:
    raise RuntimeError(
        "Expected twenty completed noFT "
        "evaluation runs."
    )

all_run_rows.sort(
    key=lambda row: (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    )
)

atomic_csv(
    AGGREGATE_RUNS_CSV,
    all_run_rows,
    [
        "run_id",
        "architecture",
        "variant",
        "seed",
        "pruning_ratio",
        "compacted_linear_widths",
        "parameter_count",
        "linear_macs_per_sample",
        "serialized_source_checkpoint_bytes",
        "validation_fingerprint_macro_f1",
        "validation_raw_weighted_macro_f1",
        "test_fingerprint_accuracy",
        "test_fingerprint_macro_f1",
        "test_raw_weighted_macro_f1",
        "test_gafgyt_fnr",
        "test_mirai_fnr",
        "validation_inference_seconds",
        "test_inference_seconds",
        "test_evaluation_count",
        "test_used_for_selection",
        "fine_tuning_performed",
        "fragility_gate_triggered",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "output_directory",
        "metrics_sha256",
        "run_manifest_sha256",
        "status",
    ],
)

group_rows: list[
    dict[str, Any]
] = []

group_json: list[
    dict[str, Any]
] = []

for architecture in ARCHITECTURES:
    for variant in VARIANTS:
        rows = [
            row
            for row in all_run_rows
            if row["architecture"]
            == architecture
            and row["variant"]
            == variant
        ]

        if len(rows) != 5:
            raise RuntimeError(
                "Expected five runs in "
                f"{architecture} {variant}."
            )

        metric_names = (
            "validation_fingerprint_macro_f1",
            "test_fingerprint_accuracy",
            "test_fingerprint_macro_f1",
            "test_raw_weighted_macro_f1",
            "test_gafgyt_fnr",
            "test_mirai_fnr",
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
            "mean_validation_fingerprint_macro_f1": (
                statistics[
                    "validation_fingerprint_macro_f1"
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
        "run_count",
        "seeds",
        "parameter_count",
        "linear_macs_per_sample",
        "mean_validation_fingerprint_macro_f1",
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
        "P25_noFT_and_P50_noFT_evaluation"
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
    "test_policy": {
        "test_evaluation_count_per_run": 1,
        "test_used_for_selection": False,
        "test_inference_total": 20,
    },
    "training_policy": {
        "fine_tuning_performed": False,
        "train_access_count": 0,
    },
    "fragility_triggered_in_any_run": (
        any(
            bool(
                row[
                    "fragility_gate_triggered"
                ]
            )
            for row in all_run_rows
        )
    ),
    "entry_checks": entry_checks,
    "cache_checks": cache_checks,
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
        "P25_noFT_and_P50_noFT_evaluation"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "completed_at_utc": utc_now(),
    "run_count": 20,
    "test_evaluation_count_per_run": 1,
    "test_inference_total": 20,
    "test_used_for_selection": False,
    "fine_tuning_performed": False,
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
print("PHASE 5 P25-noFT AND P50-noFT EVALUATION SUMMARY")
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
    "Fine-tuning performed           : False"
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
        f"{row['variant']:<8} | "
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
    "PHASE 5 P25-noFT AND P50-noFT EVALUATION COMPLETED"
)
