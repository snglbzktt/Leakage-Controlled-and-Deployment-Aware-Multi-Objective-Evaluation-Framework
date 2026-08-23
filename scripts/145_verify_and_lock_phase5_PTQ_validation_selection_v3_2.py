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

OUTPUT_VERIFIED_CANDIDATES_CSV = (
    AUDIT
    / "phase5_PTQ_validation_sweep_verified_candidates_v3_2.csv"
)

OUTPUT_VERIFICATION = (
    AUDIT
    / "phase5_PTQ_validation_sweep_verification_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase5_PTQ_validation_selection_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase5_PTQ_validation_selection_lock_manifest_v3_2.json"
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

EXPECTED_SELECTED_SIZES = {
    "tinyml_mlp": 4096,
    "compact_dnn": 16384,
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

RECONSTRUCTION_CALIBRATION_ROWS = 257
EVAL_BATCH_SIZE = 16_384
CPU_THREADS = 4
FLOAT_TOLERANCE = 1e-10

PRIMARY_SELECTION_METRIC = (
    "mean_validation_fingerprint_macro_f1"
)

SELECTION_TIE_BREAKS = (
    "lower_std_population_then_smaller_calibration_size"
)

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


def infer_validation(
    model: nn.Module,
    x_validation: np.ndarray,
    mean64: np.ndarray,
    scale64: np.ndarray,
    PTQ_engine_module: ModuleType,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    predictions = np.empty(
        len(x_validation),
        dtype=np.int8,
    )

    probabilities = np.empty(
        (
            len(x_validation),
            EXPECTED_OUTPUT_CLASSES,
        ),
        dtype=np.float32,
    )

    model.eval()

    started = time.perf_counter()

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
    AGGREGATE_CANDIDATES_CSV,
    GROUP_SUMMARY_CSV,
    SELECTION_CSV,
    AGGREGATE_JSON,
    COMPLETION_JSON,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_VERIFIED_CANDIDATES_CSV,
    OUTPUT_VERIFICATION,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "PTQ validation verification "
            "artifact already exists; refusing "
            f"to overwrite: {output_path}"
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
    PTQ_PREFLIGHT_MATRIX
)

candidate_rows = read_csv(
    AGGREGATE_CANDIDATES_CSV
)

group_rows = read_csv(
    GROUP_SUMMARY_CSV
)

selection_rows = read_csv(
    SELECTION_CSV
)

candidate_lookup = {
    (
        row["architecture"],
        int(row["seed"]),
        int(row["calibration_size"]),
    ): row
    for row in candidate_rows
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
    "aggregate_completed": (
        aggregate.get("status")
        == "completed"
        and aggregate.get(
            "candidate_count"
        )
        == 40
        and aggregate.get(
            "all_integrity_checks_passed"
        )
        is True
    ),
    "completion_completed": (
        completion.get("status")
        == "completed"
        and completion.get(
            "candidate_count"
        )
        == 40
        and completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "candidate_csv_hash_matches": (
        completion.get(
            "candidate_csv_sha256"
        )
        == sha256_file(
            AGGREGATE_CANDIDATES_CSV
        )
    ),
    "group_summary_hash_matches": (
        completion.get(
            "group_summary_csv_sha256"
        )
        == sha256_file(
            GROUP_SUMMARY_CSV
        )
    ),
    "selection_csv_hash_matches": (
        completion.get(
            "selection_csv_sha256"
        )
        == sha256_file(
            SELECTION_CSV
        )
    ),
    "aggregate_json_hash_matches": (
        completion.get(
            "aggregate_json_sha256"
        )
        == sha256_file(
            AGGREGATE_JSON
        )
    ),
    "candidate_matrix_complete": (
        set(
            candidate_lookup.keys()
        )
        == EXPECTED_KEYS
    ),
    "B0_registry_complete": (
        set(
            B0_lookup.keys()
        )
        == EXPECTED_B0_KEYS
    ),
    "preflight_matrix_complete": (
        set(
            preflight_lookup.keys()
        )
        == EXPECTED_B0_KEYS
    ),
    "locked_PTQ_matrix_complete": (
        PTQ_matrix_keys
        == EXPECTED_B0_KEYS
    ),
    "forty_candidate_rows": (
        len(candidate_rows)
        == 40
    ),
    "eight_group_rows": (
        len(group_rows)
        == 8
    ),
    "eight_selection_rows": (
        len(selection_rows)
        == 8
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
        "PTQ validation verification "
        "entry gate failed: "
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

if (
    x_train.shape
    != (
        EXPECTED_TRAIN_ROWS,
        EXPECTED_INPUT_FEATURES,
    )
):
    raise RuntimeError(
        "PTQ verification train-cache "
        "shape mismatch."
    )

if not (
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
):
    raise RuntimeError(
        "PTQ verification validation-cache "
        "shape mismatch."
    )

if int(
    raw_validation_cache.sum()
) != EXPECTED_VALIDATION_RAW_ROWS:
    raise RuntimeError(
        "PTQ verification validation raw "
        "row total mismatch."
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
    "phase5_PTQ_verify_models",
)

PTQ_engine_module = load_module(
    PTQ_ENGINE_PATH,
    "phase5_PTQ_verify_engine",
)

if (
    PTQ_engine_module.ENGINE_VERSION
    != PTQ_ENGINE_VERSION
):
    raise RuntimeError(
        "PTQ-engine version mismatch."
    )

print("=" * 92)
print("PHASE 5 PTQ VALIDATION SWEEP INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Candidates                      : 40"
)
print(
    "INT8 checkpoint loading         : Yes"
)
print(
    "Independent validation inference: Yes"
)
print(
    "Test model inference            : No"
)
print(
    "Test access count               : 0"
)
print(
    "Selection recomputation         : Yes"
)
print(
    "PTQ backend                     : "
    f"{selected_backend}"
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
            value[2],
        ),
    ),
    start=1,
):
    architecture, seed, calibration_size = key

    candidate_row = (
        candidate_lookup[key]
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

    run_directory = output_directory(
        architecture,
        seed,
        calibration_size,
    )

    paths = {
        "status": (
            run_directory / "run_status.json"
        ),
        "manifest": (
            run_directory / "run_manifest.json"
        ),
        "metrics": (
            run_directory / "metrics.json"
        ),
        "INT8_checkpoint": (
            run_directory
            / "ptq_int8_checkpoint.pt"
        ),
        "validation_predictions": (
            run_directory
            / "validation_predictions.npz"
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

    INT8_checkpoint = torch.load(
        paths["INT8_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )

    B0_checkpoint_path = Path(
        B0_row["checkpoint_path"]
    )

    if not B0_checkpoint_path.exists():
        raise FileNotFoundError(
            B0_checkpoint_path
        )

    B0_checkpoint_hash = sha256_file(
        B0_checkpoint_path
    )

    B0_checkpoint = torch.load(
        B0_checkpoint_path,
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

    calibration_indices_digest = (
        hashlib.sha256(
            calibration_indices.tobytes()
        ).hexdigest()
    )

    reconstruction_indices = (
        calibration_indices[
            :RECONSTRUCTION_CALIBRATION_ROWS
        ]
    )

    reconstruction_tensor = (
        torch.from_numpy(
            transform_rows(
                x_train[
                    reconstruction_indices
                ],
                mean64,
                scale64,
            )
        )
    )

    with torch.inference_mode():
        prepared_model(
            reconstruction_tensor
        )

    INT8_model = (
        PTQ_engine_module
        .convert_ptq_model(
            prepared_model
        )
    )

    INT8_model.load_state_dict(
        INT8_checkpoint[
            "INT8_model_state_dict"
        ],
        strict=True,
    )

    INT8_model.eval()

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

    (
        repeated_validation_pred,
        repeated_validation_prob,
        repeated_validation_seconds,
    ) = infer_validation(
        INT8_model,
        x_validation,
        mean64,
        scale64,
        PTQ_engine_module,
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
            and int(
                status.get("seed")
            )
            == seed
            and int(
                status.get(
                    "calibration_size"
                )
            )
            == calibration_size
            and manifest.get(
                "architecture"
            )
            == architecture
            and int(
                manifest.get("seed")
            )
            == seed
            and int(
                manifest.get(
                    "calibration_size"
                )
            )
            == calibration_size
            and metrics.get(
                "architecture"
            )
            == architecture
            and int(
                metrics.get("seed")
            )
            == seed
            and int(
                metrics[
                    "calibration"
                ]["size"]
            )
            == calibration_size
            and INT8_checkpoint.get(
                "architecture"
            )
            == architecture
            and int(
                INT8_checkpoint.get("seed")
            )
            == seed
            and int(
                INT8_checkpoint.get(
                    "calibration_size"
                )
            )
            == calibration_size
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
        "candidate_hashes_match": (
            candidate_row[
                "metrics_sha256"
            ]
            == sha256_file(
                paths["metrics"]
            )
            and candidate_row[
                "run_manifest_sha256"
            ]
            == sha256_file(
                paths["manifest"]
            )
            and candidate_row[
                "INT8_checkpoint_sha256"
            ]
            == sha256_file(
                paths[
                    "INT8_checkpoint"
                ]
            )
            and candidate_row[
                "validation_predictions_sha256"
            ]
            == sha256_file(
                paths[
                    "validation_predictions"
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
                "INT8_checkpoint_sha256"
            )
            == sha256_file(
                paths[
                    "INT8_checkpoint"
                ]
            )
            and status.get(
                "validation_predictions_sha256"
            )
            == sha256_file(
                paths[
                    "validation_predictions"
                ]
            )
        ),
        "source_checkpoint_hash_matches": (
            B0_checkpoint_hash
            == B0_row[
                "checkpoint_sha256"
            ]
            == preflight_row[
                "source_checkpoint_sha256"
            ]
            == metrics[
                "source"
            ][
                "source_B0_checkpoint_sha256"
            ]
            == INT8_checkpoint[
                "source_B0_checkpoint_sha256"
            ]
            == manifest[
                "source_B0_checkpoint"
            ]["sha256"]
            == candidate_row[
                "source_checkpoint_sha256"
            ]
        ),
        "calibration_indices_match": (
            calibration_indices.shape
            == (
                calibration_size,
            )
            and len(
                np.unique(
                    calibration_indices
                )
            )
            == calibration_size
            and calibration_indices_digest
            == metrics[
                "calibration"
            ][
                "indices_sha256"
            ]
            == INT8_checkpoint[
                "calibration_indices_sha256"
            ]
            == candidate_row[
                "calibration_indices_sha256"
            ]
        ),
        "PTQ_backend_matches_lock": (
            selected_backend
            == metrics[
                "quantization"
            ]["backend"]
            == INT8_checkpoint[
                "PTQ_backend"
            ]
            == preflight_row[
                "PTQ_backend"
            ]
        ),
        "PTQ_engine_version_matches": (
            metrics[
                "PTQ_engine_version"
            ]
            == PTQ_ENGINE_VERSION
            == INT8_checkpoint[
                "PTQ_engine_version"
            ]
        ),
        "float_topology_matches": (
            PTQ_engine_module
            .float_linear_widths(
                source_model
            )
            == EXPECTED_WIDTHS[
                architecture
            ]
            == INT8_checkpoint[
                "float_linear_widths"
            ]
            == metrics[
                "model_complexity"
            ][
                "float_linear_widths"
            ]
        ),
        "all_linears_quantized": (
            len(
                PTQ_engine_module
                .static_quantized_linear_modules(
                    INT8_model
                )
            )
            == len(
                EXPECTED_WIDTHS[
                    architecture
                ]
            )
            == int(
                metrics[
                    "model_complexity"
                ]["INT8_linear_count"]
            )
        ),
        "quantized_weights_qint8": (
            all(
                dtype == "torch.qint8"
                for dtype in (
                    PTQ_engine_module
                    .quantized_weight_dtypes(
                        INT8_model
                    )
                )
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
        "saved_labels_match_cache": (
            np.array_equal(
                saved_validation_true,
                y_validation_cache,
            )
        ),
        "saved_raw_counts_match_cache": (
            np.array_equal(
                saved_validation_raw,
                raw_validation_cache,
            )
        ),
        "saved_probabilities_finite": (
            np.isfinite(
                saved_validation_prob
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
        "candidate_metrics_match": (
            close_enough(
                candidate_row[
                    "validation_fingerprint_macro_f1"
                ],
                saved_validation_primary[
                    "macro_f1"
                ],
            )
            and close_enough(
                candidate_row[
                    "validation_fingerprint_accuracy"
                ],
                saved_validation_primary[
                    "accuracy"
                ],
            )
            and close_enough(
                candidate_row[
                    "validation_raw_weighted_macro_f1"
                ],
                saved_validation_weighted[
                    "macro_f1"
                ],
            )
            and close_enough(
                candidate_row[
                    "validation_gafgyt_fnr"
                ],
                saved_validation_primary[
                    "per_class"
                ]["gafgyt"][
                    "false_negative_rate"
                ],
            )
            and close_enough(
                candidate_row[
                    "validation_mirai_fnr"
                ],
                saved_validation_primary[
                    "per_class"
                ]["mirai"][
                    "false_negative_rate"
                ],
            )
        ),
        "validation_access_is_one": (
            int(
                status[
                    "validation_access_count"
                ]
            )
            == 1
            and int(
                metrics[
                    "validation_access_count"
                ]
            )
            == 1
            and int(
                candidate_row[
                    "validation_access_count"
                ]
            )
            == 1
        ),
        "test_access_is_zero": (
            int(
                status[
                    "test_access_count"
                ]
            )
            == 0
            and int(
                metrics[
                    "test_access_count"
                ]
            )
            == 0
            and int(
                candidate_row[
                    "test_access_count"
                ]
            )
            == 0
            and int(
                manifest[
                    "selection_scope"
                ][
                    "test_access_count"
                ]
            )
            == 0
        ),
        "test_not_used_for_selection": (
            metrics[
                "test_used_for_selection"
            ]
            is False
            and manifest[
                "selection_scope"
            ][
                "test_used_for_selection"
            ]
            is False
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
            f"calibration {calibration_size} "
            "PTQ validation verification "
            "failed: "
            + ", ".join(
                failed_checks
            )
        )

    verified_rows.append(
        {
            "architecture": architecture,
            "seed": seed,
            "calibration_size": (
                calibration_size
            ),
            "validation_fingerprint_macro_f1": (
                repeated_validation_primary[
                    "macro_f1"
                ]
            ),
            "validation_fingerprint_accuracy": (
                repeated_validation_primary[
                    "accuracy"
                ]
            ),
            "validation_raw_weighted_macro_f1": (
                saved_validation_weighted[
                    "macro_f1"
                ]
            ),
            "validation_gafgyt_fnr": (
                repeated_validation_primary[
                    "per_class"
                ]["gafgyt"][
                    "false_negative_rate"
                ]
            ),
            "validation_mirai_fnr": (
                repeated_validation_primary[
                    "per_class"
                ]["mirai"][
                    "false_negative_rate"
                ]
            ),
            "validation_repeat_seconds": (
                repeated_validation_seconds
            ),
            "validation_access_count": 1,
            "test_access_count": 0,
            "test_model_inference_repeated": False,
            "run_directory": str(
                run_directory
            ),
            "INT8_checkpoint_sha256": (
                sha256_file(
                    paths[
                        "INT8_checkpoint"
                    ]
                )
            ),
            "validation_predictions_sha256": (
                sha256_file(
                    paths[
                        "validation_predictions"
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
        f"[{index}/40] "
        f"{architecture} seed={seed} "
        f"cal={calibration_size} | "
        "INT8 checkpoint=loaded | "
        "validation=exact | "
        "test access=0 | "
        "all checks=True",
        flush=True,
    )

    del B0_checkpoint
    del source_model
    del prepared_model
    del INT8_model
    del INT8_checkpoint
    del repeated_validation_pred
    del repeated_validation_prob
    del saved_validation_pred
    del saved_validation_prob

calibration_archive.close()

atomic_csv(
    OUTPUT_VERIFIED_CANDIDATES_CSV,
    verified_rows,
    [
        "architecture",
        "seed",
        "calibration_size",
        "validation_fingerprint_macro_f1",
        "validation_fingerprint_accuracy",
        "validation_raw_weighted_macro_f1",
        "validation_gafgyt_fnr",
        "validation_mirai_fnr",
        "validation_repeat_seconds",
        "validation_access_count",
        "test_access_count",
        "test_model_inference_repeated",
        "run_directory",
        "INT8_checkpoint_sha256",
        "validation_predictions_sha256",
        "metrics_sha256",
        "run_manifest_sha256",
        "all_checks_passed",
    ],
)

group_lookup = {
    (
        row["architecture"],
        int(row["calibration_size"]),
    ): row
    for row in group_rows
}

recomputed_group_rows: list[
    dict[str, Any]
] = []

for architecture in ARCHITECTURES:
    for calibration_size in CALIBRATION_SIZES:
        rows = [
            row
            for row in verified_rows
            if row["architecture"]
            == architecture
            and int(
                row[
                    "calibration_size"
                ]
            )
            == calibration_size
        ]

        saved_group = group_lookup[
            (
                architecture,
                calibration_size,
            )
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
            "validation_fingerprint_accuracy": (
                aggregate_statistics(
                    [
                        row[
                            "validation_fingerprint_accuracy"
                        ]
                        for row in rows
                    ]
                )
            ),
            "validation_raw_weighted_macro_f1": (
                aggregate_statistics(
                    [
                        row[
                            "validation_raw_weighted_macro_f1"
                        ]
                        for row in rows
                    ]
                )
            ),
            "validation_gafgyt_fnr": (
                aggregate_statistics(
                    [
                        row[
                            "validation_gafgyt_fnr"
                        ]
                        for row in rows
                    ]
                )
            ),
            "validation_mirai_fnr": (
                aggregate_statistics(
                    [
                        row[
                            "validation_mirai_fnr"
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
            "mean_macro_f1_matches": (
                close_enough(
                    saved_group[
                        "mean_validation_fingerprint_macro_f1"
                    ],
                    statistics[
                        "validation_fingerprint_macro_f1"
                    ]["mean"],
                )
            ),
            "std_macro_f1_matches": (
                close_enough(
                    saved_group[
                        "std_validation_fingerprint_macro_f1"
                    ],
                    statistics[
                        "validation_fingerprint_macro_f1"
                    ]["std_population"],
                )
            ),
            "min_macro_f1_matches": (
                close_enough(
                    saved_group[
                        "min_validation_fingerprint_macro_f1"
                    ],
                    statistics[
                        "validation_fingerprint_macro_f1"
                    ]["minimum"],
                )
            ),
            "max_macro_f1_matches": (
                close_enough(
                    saved_group[
                        "max_validation_fingerprint_macro_f1"
                    ],
                    statistics[
                        "validation_fingerprint_macro_f1"
                    ]["maximum"],
                )
            ),
            "mean_accuracy_matches": (
                close_enough(
                    saved_group[
                        "mean_validation_fingerprint_accuracy"
                    ],
                    statistics[
                        "validation_fingerprint_accuracy"
                    ]["mean"],
                )
            ),
            "mean_raw_weighted_matches": (
                close_enough(
                    saved_group[
                        "mean_validation_raw_weighted_macro_f1"
                    ],
                    statistics[
                        "validation_raw_weighted_macro_f1"
                    ]["mean"],
                )
            ),
            "mean_gafgyt_fnr_matches": (
                close_enough(
                    saved_group[
                        "mean_validation_gafgyt_fnr"
                    ],
                    statistics[
                        "validation_gafgyt_fnr"
                    ]["mean"],
                )
            ),
            "mean_mirai_fnr_matches": (
                close_enough(
                    saved_group[
                        "mean_validation_mirai_fnr"
                    ],
                    statistics[
                        "validation_mirai_fnr"
                    ]["mean"],
                )
            ),
            "test_access_zero": (
                int(
                    saved_group[
                        "test_access_count"
                    ]
                )
                == 0
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
                f"{architecture} calibration "
                f"{calibration_size} PTQ "
                "group verification failed: "
                + ", ".join(
                    failed_group_checks
                )
            )

        recomputed_group_rows.append(
            {
                "architecture": architecture,
                "calibration_size": (
                    calibration_size
                ),
                "run_count": 5,
                "mean_validation_fingerprint_macro_f1": (
                    statistics[
                        "validation_fingerprint_macro_f1"
                    ]["mean"]
                ),
                "std_validation_fingerprint_macro_f1": (
                    statistics[
                        "validation_fingerprint_macro_f1"
                    ]["std_population"]
                ),
                "mean_validation_raw_weighted_macro_f1": (
                    statistics[
                        "validation_raw_weighted_macro_f1"
                    ]["mean"]
                ),
                "test_access_count": 0,
                "all_checks_passed": True,
            }
        )

recomputed_selection_rows: list[
    dict[str, Any]
] = []

recomputed_selected_sizes: dict[
    str,
    int,
] = {}

for architecture in ARCHITECTURES:
    architecture_groups = [
        row
        for row in recomputed_group_rows
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
        recomputed_selection_rows.append(
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
            }
        )

    recomputed_selected_sizes[
        architecture
    ] = int(
        ranked[0][
            "calibration_size"
        ]
    )

saved_selection_lookup = {
    (
        row["architecture"],
        int(row["rank"]),
    ): row
    for row in selection_rows
}

selection_checks = {
    "selected_sizes_match_expected": (
        recomputed_selected_sizes
        == EXPECTED_SELECTED_SIZES
    ),
    "selected_sizes_match_aggregate": (
        recomputed_selected_sizes
        == {
            key: int(value)
            for key, value
            in aggregate[
                "selection_policy"
            ][
                "selected_calibration_sizes"
            ].items()
        }
    ),
    "selected_sizes_match_completion": (
        recomputed_selected_sizes
        == {
            key: int(value)
            for key, value
            in completion[
                "selected_calibration_sizes"
            ].items()
        }
    ),
    "selection_rows_match": (
        all(
            (
                saved_selection_lookup[
                    (
                        row["architecture"],
                        int(row["rank"]),
                    )
                ][
                    "architecture"
                ]
                == row["architecture"]
                and int(
                    saved_selection_lookup[
                        (
                            row["architecture"],
                            int(row["rank"]),
                        )
                    ][
                        "calibration_size"
                    ]
                )
                == int(
                    row[
                        "calibration_size"
                    ]
                )
                and parse_bool(
                    saved_selection_lookup[
                        (
                            row["architecture"],
                            int(row["rank"]),
                        )
                    ][
                        "selected"
                    ]
                )
                == bool(
                    row["selected"]
                )
                and close_enough(
                    saved_selection_lookup[
                        (
                            row["architecture"],
                            int(row["rank"]),
                        )
                    ][
                        "mean_validation_fingerprint_macro_f1"
                    ],
                    row[
                        "mean_validation_fingerprint_macro_f1"
                    ],
                )
                and close_enough(
                    saved_selection_lookup[
                        (
                            row["architecture"],
                            int(row["rank"]),
                        )
                    ][
                        "std_validation_fingerprint_macro_f1"
                    ],
                    row[
                        "std_validation_fingerprint_macro_f1"
                    ],
                )
                and close_enough(
                    saved_selection_lookup[
                        (
                            row["architecture"],
                            int(row["rank"]),
                        )
                    ][
                        "mean_validation_raw_weighted_macro_f1"
                    ],
                    row[
                        "mean_validation_raw_weighted_macro_f1"
                    ],
                )
                and saved_selection_lookup[
                    (
                        row["architecture"],
                        int(row["rank"]),
                    )
                ][
                    "selection_metric"
                ]
                == PRIMARY_SELECTION_METRIC
                and saved_selection_lookup[
                    (
                        row["architecture"],
                        int(row["rank"]),
                    )
                ][
                    "tie_breaks"
                ]
                == SELECTION_TIE_BREAKS
                and int(
                    saved_selection_lookup[
                        (
                            row["architecture"],
                            int(row["rank"]),
                        )
                    ][
                        "test_access_count"
                    ]
                )
                == 0
            )
            for row in recomputed_selection_rows
        )
    ),
}

failed_selection_checks = [
    name
    for name, passed
    in selection_checks.items()
    if not passed
]

if failed_selection_checks:
    raise RuntimeError(
        "PTQ calibration-size selection "
        "verification failed: "
        + ", ".join(
            failed_selection_checks
        )
    )

global_checks = {
    "forty_candidates_verified": (
        len(
            verified_rows
        )
        == 40
    ),
    "all_candidate_checks_passed": (
        all(
            row[
                "all_checks_passed"
            ]
            for row in verified_rows
        )
    ),
    "eight_groups_verified": (
        len(
            recomputed_group_rows
        )
        == 8
    ),
    "validation_access_one_all_candidates": (
        all(
            row[
                "validation_access_count"
            ]
            == 1
            for row in verified_rows
        )
    ),
    "test_access_zero_all_candidates": (
        all(
            row[
                "test_access_count"
            ]
            == 0
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
    "two_selected_calibration_sizes": (
        len(
            recomputed_selected_sizes
        )
        == 2
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
        "PTQ validation verification "
        "global checks failed: "
        + ", ".join(
            failed_global_checks
        )
    )

verification = {
    "status": "passed",
    "phase": 5,
    "artifact_name": (
        "PTQ_validation_calibration_sweep_"
        "independent_verification"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "PTQ_engine_version": (
        PTQ_ENGINE_VERSION
    ),
    "verified_at_utc": utc_now(),
    "candidate_count": 40,
    "group_count": 8,
    "selected_calibration_sizes": (
        recomputed_selected_sizes
    ),
    "verification_policy": {
        "INT8_checkpoint_loading_performed": True,
        "independent_validation_inference_repeated": True,
        "test_model_inference_repeated": False,
        "test_access_count": 0,
        "group_statistics_recomputed": True,
        "selection_ranking_recomputed": True,
    },
    "verified_candidates_csv": (
        file_record(
            OUTPUT_VERIFIED_CANDIDATES_CSV
        )
    ),
    "entry_checks": (
        entry_checks
    ),
    "selection_checks": (
        selection_checks
    ),
    "global_checks": (
        global_checks
    ),
    "source_artifacts": {
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
    },
    "ready_for_selected_PTQ_test_evaluation": (
        True
    ),
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
        "PTQ_validation_calibration_selection"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "PTQ_engine_version": (
        PTQ_ENGINE_VERSION
    ),
    "locked_at_utc": utc_now(),
    "candidate_count": 40,
    "group_count": 8,
    "selected_calibration_sizes": (
        recomputed_selected_sizes
    ),
    "selection_metric": (
        PRIMARY_SELECTION_METRIC
    ),
    "tie_breaks": (
        SELECTION_TIE_BREAKS
    ),
    "validation_access_count_per_candidate": (
        1
    ),
    "test_access_count": 0,
    "test_model_inference_repeated_by_verifier": (
        False
    ),
    "verified_candidates_csv": str(
        OUTPUT_VERIFIED_CANDIDATES_CSV
    ),
    "verified_candidates_csv_sha256": (
        sha256_file(
            OUTPUT_VERIFIED_CANDIDATES_CSV
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
    "ready_for_selected_PTQ_test_evaluation": (
        True
    ),
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
        "PTQ_validation_calibration_selection"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "PTQ_engine_version": (
        PTQ_ENGINE_VERSION
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
            QAT_LOCK
        ),
        file_record(
            PTQ_PREFLIGHT_LOCK
        ),
        file_record(
            PTQ_PREFLIGHT_MATRIX
        ),
        file_record(
            CALIBRATION_INDEX_NPZ
        ),
        file_record(
            MODEL_SOURCE
        ),
        file_record(
            PTQ_ENGINE_PATH
        ),
        file_record(
            SCALER_NPZ
        ),
        file_record(
            AGGREGATE_CANDIDATES_CSV
        ),
        file_record(
            GROUP_SUMMARY_CSV
        ),
        file_record(
            SELECTION_CSV
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
            OUTPUT_VERIFIED_CANDIDATES_CSV
        ),
        file_record(
            OUTPUT_VERIFICATION
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "candidate_count": 40,
    "group_count": 8,
    "selected_calibration_sizes": (
        recomputed_selected_sizes
    ),
    "test_access_count": 0,
    "ready_for_selected_PTQ_test_evaluation": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print()
print("=" * 92)
print("PHASE 5 PTQ VALIDATION VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Candidates independently verified: 40"
)
print(
    "Groups independently verified   : 8"
)
print(
    "INT8 checkpoint loading         : PASSED"
)
print(
    "Independent validation inference: PASSED"
)
print(
    "Group statistics recomputed     : PASSED"
)
print(
    "Selection ranking recomputed    : PASSED"
)
print(
    "Test model inference repeated   : False"
)
print(
    "Test access count               : 0"
)
print(
    "Selected calibration sizes      : "
    f"{recomputed_selected_sizes}"
)
print(
    "PTQ validation selection status : LOCKED"
)
print(
    "Ready for selected PTQ test     : True"
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
    "PHASE 5 PTQ VALIDATION CALIBRATION SELECTION VERIFIED AND LOCKED"
)
