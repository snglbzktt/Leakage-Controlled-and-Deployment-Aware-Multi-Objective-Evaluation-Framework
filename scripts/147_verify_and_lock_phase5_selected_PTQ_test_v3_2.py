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

PTQ_SELECTED_TEST_RUNS_CSV = (
    AUDIT
    / "phase5_PTQ_selected_test_all_runs_v3_2.csv"
)

PTQ_SELECTED_TEST_GROUP_CSV = (
    AUDIT
    / "phase5_PTQ_selected_test_group_summary_v3_2.csv"
)

PTQ_SELECTED_TEST_SUMMARY_JSON = (
    AUDIT
    / "phase5_PTQ_selected_test_summary_v3_2.json"
)

PTQ_SELECTED_TEST_COMPLETION_JSON = (
    AUDIT
    / "phase5_PTQ_selected_test_completed_v3_2.json"
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

OUTPUT_VERIFIED_RUNS_CSV = (
    AUDIT
    / "phase5_PTQ_selected_test_verified_runs_v3_2.csv"
)

OUTPUT_VERIFICATION_JSON = (
    AUDIT
    / "phase5_PTQ_selected_test_verification_v3_2.json"
)

OUTPUT_LOCK_JSON = (
    AUDIT
    / "phase5_PTQ_selected_test_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST_JSON = (
    AUDIT
    / "phase5_PTQ_selected_test_lock_manifest_v3_2.json"
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
CPU_THREADS = 4
FLOAT_TOLERANCE = 1e-10

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


def selected_test_directory(
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


required_paths = (
    X_TRAIN_PATH,
    Y_TEST_PATH,
    RAW_TEST_PATH,
    PHASE5_PROTOCOL,
    B0_PAIR_LOCK,
    B0_CHECKPOINT_REGISTRY,
    PTQ_PREFLIGHT_LOCK,
    PTQ_SELECTION_LOCK,
    PTQ_VERIFIED_CANDIDATES_CSV,
    PTQ_SELECTED_TEST_RUNS_CSV,
    PTQ_SELECTED_TEST_GROUP_CSV,
    PTQ_SELECTED_TEST_SUMMARY_JSON,
    PTQ_SELECTED_TEST_COMPLETION_JSON,
    MODEL_SOURCE,
    PTQ_ENGINE_PATH,
    SCALER_NPZ,
    CALIBRATION_INDEX_NPZ,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_VERIFIED_RUNS_CSV,
    OUTPUT_VERIFICATION_JSON,
    OUTPUT_LOCK_JSON,
    OUTPUT_LOCK_MANIFEST_JSON,
):
    if output_path.exists():
        raise FileExistsError(
            "Selected PTQ verification "
            "artifact already exists; refusing "
            f"to overwrite: {output_path}"
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

selected_test_summary = read_json(
    PTQ_SELECTED_TEST_SUMMARY_JSON
)

selected_test_completion = read_json(
    PTQ_SELECTED_TEST_COMPLETION_JSON
)

B0_rows = read_csv(
    B0_CHECKPOINT_REGISTRY
)

verified_candidate_rows = read_csv(
    PTQ_VERIFIED_CANDIDATES_CSV
)

selected_test_rows = read_csv(
    PTQ_SELECTED_TEST_RUNS_CSV
)

selected_test_group_rows = read_csv(
    PTQ_SELECTED_TEST_GROUP_CSV
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

selected_test_lookup = {
    (
        row["architecture"],
        int(row["seed"]),
    ): row
    for row in selected_test_rows
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
            "all_checks_passed"
        )
        is True
    ),
    "selected_test_summary_completed": (
        selected_test_summary.get(
            "status"
        )
        == "completed"
        and selected_test_summary.get(
            "run_count"
        )
        == 10
        and selected_test_summary.get(
            "all_integrity_checks_passed"
        )
        is True
    ),
    "selected_test_completion_completed": (
        selected_test_completion.get(
            "status"
        )
        == "completed"
        and selected_test_completion.get(
            "run_count"
        )
        == 10
        and selected_test_completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "selected_test_runs_hash_matches": (
        selected_test_completion.get(
            "aggregate_runs_csv_sha256"
        )
        == sha256_file(
            PTQ_SELECTED_TEST_RUNS_CSV
        )
    ),
    "selected_test_group_hash_matches": (
        selected_test_completion.get(
            "group_summary_csv_sha256"
        )
        == sha256_file(
            PTQ_SELECTED_TEST_GROUP_CSV
        )
    ),
    "selected_test_summary_hash_matches": (
        selected_test_completion.get(
            "aggregate_json_sha256"
        )
        == sha256_file(
            PTQ_SELECTED_TEST_SUMMARY_JSON
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
    "selected_test_matrix_complete": (
        set(
            selected_test_lookup.keys()
        )
        == EXPECTED_KEYS
    ),
    "ten_selected_test_rows": (
        len(selected_test_rows)
        == 10
    ),
    "two_selected_test_group_rows": (
        len(selected_test_group_rows)
        == 2
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
        "Selected PTQ verification "
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
    y_test_cache.shape
    == (
        EXPECTED_TEST_ROWS,
    )
    and raw_test_cache.shape
    == (
        EXPECTED_TEST_ROWS,
    )
):
    raise RuntimeError(
        "PTQ verification test-cache "
        "shape mismatch."
    )

if int(
    raw_test_cache.sum()
) != EXPECTED_TEST_RAW_ROWS:
    raise RuntimeError(
        "PTQ verification test raw-row "
        "total mismatch."
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
    "phase5_PTQ_selected_verify_models",
)

PTQ_engine_module = load_module(
    PTQ_ENGINE_PATH,
    "phase5_PTQ_selected_verify_engine",
)

if (
    PTQ_engine_module.ENGINE_VERSION
    != PTQ_ENGINE_VERSION
):
    raise RuntimeError(
        "PTQ-engine version mismatch."
    )

print("=" * 92)
print("PHASE 5 SELECTED PTQ TEST INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Runs                            : 10"
)
print(
    "Selected calibration sizes      : "
    f"{selected_sizes}"
)
print(
    "Candidate INT8 checkpoint load  : Yes"
)
print(
    "Train-only smoke inference      : Yes"
)
print(
    "Test model inference            : No"
)
print(
    "Test verification source        : "
    "saved predictions only"
)
print(
    "Test evaluation count per run   : 1"
)
print(
    "PTQ backend                     : "
    f"{selected_backend}"
)
print()

verified_rows: list[
    dict[str, Any]
] = []

for index, (
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

    selected_row = (
        selected_test_lookup[
            (
                architecture,
                seed,
            )
        ]
    )

    run_directory = (
        selected_test_directory(
            architecture,
            seed,
        )
    )

    candidate_dir = (
        candidate_directory(
            architecture,
            seed,
            calibration_size,
        )
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
        "test_predictions": (
            run_directory
            / "test_predictions.npz"
        ),
        "candidate_checkpoint": (
            candidate_dir
            / "ptq_int8_checkpoint.pt"
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

    candidate_checkpoint_hash = (
        sha256_file(
            paths[
                "candidate_checkpoint"
            ]
        )
    )

    candidate_checkpoint = torch.load(
        paths["candidate_checkpoint"],
        map_location="cpu",
        weights_only=False,
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

    smoke_indices = (
        calibration_indices[
            STRUCTURE_CALIBRATION_ROWS:
            STRUCTURE_CALIBRATION_ROWS + 257
        ]
    )

    smoke_tensor = torch.from_numpy(
        transform_rows(
            x_train[
                smoke_indices
            ],
            mean64,
            scale64,
        )
    )

    with torch.inference_mode():
        smoke_logits = (
            PTQ_engine_module
            .extract_logits(
                INT8_model(
                    smoke_tensor
                )
            )
        )

    with np.load(
        paths["test_predictions"]
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

    recomputed_test_primary = metric_view(
        saved_test_true,
        saved_test_pred,
        saved_test_prob,
        sample_weight=None,
    )

    recomputed_test_weighted = metric_view(
        saved_test_true,
        saved_test_pred,
        saved_test_prob,
        sample_weight=saved_test_raw,
    )

    verified_candidate_row = (
        verified_candidate_lookup[
            (
                architecture,
                seed,
                calibration_size,
            )
        ]
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
                metrics.get(
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
        "run_hashes_match": (
            selected_row[
                "test_predictions_sha256"
            ]
            == sha256_file(
                paths[
                    "test_predictions"
                ]
            )
            and selected_row[
                "metrics_sha256"
            ]
            == sha256_file(
                paths["metrics"]
            )
            and selected_row[
                "run_manifest_sha256"
            ]
            == sha256_file(
                paths["manifest"]
            )
        ),
        "status_hashes_match": (
            status.get(
                "test_predictions_sha256"
            )
            == sha256_file(
                paths[
                    "test_predictions"
                ]
            )
            and status.get(
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
        ),
        "candidate_checkpoint_hash_matches": (
            candidate_checkpoint_hash
            == selected_row[
                "candidate_INT8_checkpoint_sha256"
            ]
            == metrics[
                "source_candidate"
            ][
                "INT8_checkpoint_sha256"
            ]
            == status[
                "source_candidate_checkpoint_sha256"
            ]
            == verified_candidate_row[
                "INT8_checkpoint_sha256"
            ]
            == manifest[
                "source_candidate_checkpoint"
            ]["sha256"]
        ),
        "B0_checkpoint_hash_matches": (
            B0_checkpoint_hash
            == B0_row[
                "checkpoint_sha256"
            ]
            == metrics[
                "source_B0"
            ][
                "checkpoint_sha256"
            ]
            == manifest[
                "source_B0_checkpoint"
            ]["sha256"]
        ),
        "selected_size_matches_lock": (
            calibration_size
            == EXPECTED_SELECTED_SIZES[
                architecture
            ]
            == int(
                metrics[
                    "selection"
                ][
                    "selected_calibration_size"
                ]
            )
        ),
        "PTQ_backend_matches_lock": (
            selected_backend
            == metrics[
                "quantization"
            ]["backend"]
            == candidate_checkpoint[
                "PTQ_backend"
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
            == candidate_checkpoint[
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
        "train_smoke_output_valid": (
            tuple(
                smoke_logits.shape
            )
            == (
                len(smoke_indices),
                EXPECTED_OUTPUT_CLASSES,
            )
            and bool(
                torch.isfinite(
                    smoke_logits
                ).all().item()
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
        "saved_test_labels_match_cache": (
            np.array_equal(
                saved_test_true,
                y_test_cache,
            )
        ),
        "saved_test_raw_counts_match_cache": (
            np.array_equal(
                saved_test_raw,
                raw_test_cache,
            )
        ),
        "saved_test_probabilities_finite": (
            np.isfinite(
                saved_test_prob
            ).all()
        ),
        "saved_test_probabilities_normalized": (
            np.allclose(
                saved_test_prob.sum(
                    axis=1
                ),
                1.0,
                atol=1e-6,
                rtol=0.0,
            )
        ),
        "saved_test_primary_metrics_match": (
            metrics_match(
                recomputed_test_primary,
                metrics[
                    "test"
                ][
                    "primary_fingerprint_level"
                ],
            )
        ),
        "saved_test_weighted_metrics_match": (
            metrics_match(
                recomputed_test_weighted,
                metrics[
                    "test"
                ][
                    "secondary_raw_record_weighted"
                ],
            )
        ),
        "aggregate_metrics_match": (
            close_enough(
                selected_row[
                    "test_fingerprint_macro_f1"
                ],
                recomputed_test_primary[
                    "macro_f1"
                ],
            )
            and close_enough(
                selected_row[
                    "test_fingerprint_accuracy"
                ],
                recomputed_test_primary[
                    "accuracy"
                ],
            )
            and close_enough(
                selected_row[
                    "test_raw_weighted_macro_f1"
                ],
                recomputed_test_weighted[
                    "macro_f1"
                ],
            )
            and close_enough(
                selected_row[
                    "test_gafgyt_fnr"
                ],
                recomputed_test_primary[
                    "per_class"
                ]["gafgyt"][
                    "false_negative_rate"
                ],
            )
            and close_enough(
                selected_row[
                    "test_mirai_fnr"
                ],
                recomputed_test_primary[
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
                    "test_policy"
                ][
                    "test_evaluation_count"
                ]
            )
            == 1
            and int(
                selected_row[
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
                "test_policy"
            ][
                "test_used_for_selection"
            ]
            is False
        ),
        "validation_not_repeated": (
            metrics[
                "selection"
            ][
                "validation_inference_repeated"
            ]
            is False
            and metrics[
                "data_access"
            ][
                "validation_inference_repeated"
            ]
            is False
            and manifest[
                "test_policy"
            ][
                "validation_inference_repeated"
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
                selected_row[
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
            "selected PTQ verification "
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
            "INT8_to_float_state_size_ratio": float(
                metrics[
                    "model_complexity"
                ][
                    "INT8_to_float_state_size_ratio"
                ]
            ),
            "test_evaluation_count": 1,
            "test_model_inference_repeated": False,
            "train_smoke_inference_performed": True,
            "fragility_gate_triggered": bool(
                metrics[
                    "fragility_gate"
                ]["triggered"]
            ),
            "run_directory": str(
                run_directory
            ),
            "candidate_checkpoint_sha256": (
                candidate_checkpoint_hash
            ),
            "test_predictions_sha256": (
                sha256_file(
                    paths[
                        "test_predictions"
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
        f"{architecture} seed={seed} "
        f"cal={calibration_size} PTQ | "
        "checkpoint=loaded | "
        "train smoke=passed | "
        "test inference=False | "
        "fragility="
        f"{bool(metrics['fragility_gate']['triggered'])} | "
        "all checks=True",
        flush=True,
    )

    del B0_checkpoint
    del candidate_checkpoint
    del source_model
    del prepared_model
    del INT8_model
    del smoke_logits
    del saved_test_pred
    del saved_test_prob

calibration_archive.close()

atomic_csv(
    OUTPUT_VERIFIED_RUNS_CSV,
    verified_rows,
    [
        "architecture",
        "seed",
        "calibration_size",
        "test_fingerprint_macro_f1",
        "test_fingerprint_accuracy",
        "test_raw_weighted_macro_f1",
        "test_gafgyt_fnr",
        "test_mirai_fnr",
        "INT8_to_float_state_size_ratio",
        "test_evaluation_count",
        "test_model_inference_repeated",
        "train_smoke_inference_performed",
        "fragility_gate_triggered",
        "run_directory",
        "candidate_checkpoint_sha256",
        "test_predictions_sha256",
        "metrics_sha256",
        "run_manifest_sha256",
        "all_checks_passed",
    ],
)

saved_group_lookup = {
    row["architecture"]: row
    for row in selected_test_group_rows
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

    saved_group = (
        saved_group_lookup[
            architecture
        ]
    )

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
        "INT8_to_float_state_size_ratio": (
            aggregate_statistics(
                [
                    row[
                        "INT8_to_float_state_size_ratio"
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
        "selected_size_matches": (
            int(
                saved_group[
                    "calibration_size"
                ]
            )
            == selected_sizes[
                architecture
            ]
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
                ]["std_population"],
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
        "mean_INT8_ratio_matches": (
            close_enough(
                saved_group[
                    "mean_INT8_to_float_state_size_ratio"
                ],
                statistics[
                    "INT8_to_float_state_size_ratio"
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
            f"{architecture} selected PTQ "
            "group verification failed: "
            + ", ".join(
                failed_group_checks
            )
        )

    verified_groups.append(
        {
            "architecture": architecture,
            "calibration_size": (
                selected_sizes[
                    architecture
                ]
            ),
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
    "two_groups_verified": (
        len(
            verified_groups
        )
        == 2
    ),
    "all_run_checks_passed": (
        all(
            row[
                "all_checks_passed"
            ]
            for row in verified_rows
        )
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
        "Selected PTQ verification "
        "global checks failed: "
        + ", ".join(
            failed_global_checks
        )
    )

verification = {
    "status": "passed",
    "phase": 5,
    "artifact_name": (
        "selected_PTQ_test_independent_verification"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "PTQ_engine_version": (
        PTQ_ENGINE_VERSION
    ),
    "verified_at_utc": utc_now(),
    "run_count": 10,
    "group_count": 2,
    "selected_calibration_sizes": (
        selected_sizes
    ),
    "verification_policy": {
        "candidate_INT8_checkpoint_loading_performed": (
            True
        ),
        "train_only_smoke_inference_performed": (
            True
        ),
        "test_model_inference_repeated": (
            False
        ),
        "saved_test_metrics_recomputed": (
            True
        ),
        "test_evaluation_count_verified_per_run": (
            1
        ),
        "group_statistics_recomputed": (
            True
        ),
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
    "global_checks": (
        global_checks
    ),
    "source_artifacts": {
        "selected_test_runs_csv": (
            file_record(
                PTQ_SELECTED_TEST_RUNS_CSV
            )
        ),
        "selected_test_group_csv": (
            file_record(
                PTQ_SELECTED_TEST_GROUP_CSV
            )
        ),
        "selected_test_summary_json": (
            file_record(
                PTQ_SELECTED_TEST_SUMMARY_JSON
            )
        ),
        "selected_test_completion_json": (
            file_record(
                PTQ_SELECTED_TEST_COMPLETION_JSON
            )
        ),
        "selection_lock": (
            file_record(
                PTQ_SELECTION_LOCK
            )
        ),
        "PTQ_engine": (
            file_record(
                PTQ_ENGINE_PATH
            )
        ),
    },
    "ready_for_phase5_compression_matrix_closure": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_VERIFICATION_JSON,
    verification,
)

lock = {
    "status": "locked",
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
    "locked_at_utc": utc_now(),
    "run_count": 10,
    "group_count": 2,
    "selected_calibration_sizes": (
        selected_sizes
    ),
    "test_evaluation_count_per_run": 1,
    "test_model_inference_repeated_by_verifier": (
        False
    ),
    "verified_runs_csv": str(
        OUTPUT_VERIFIED_RUNS_CSV
    ),
    "verified_runs_csv_sha256": (
        sha256_file(
            OUTPUT_VERIFIED_RUNS_CSV
        )
    ),
    "verification_report": str(
        OUTPUT_VERIFICATION_JSON
    ),
    "verification_report_sha256": (
        sha256_file(
            OUTPUT_VERIFICATION_JSON
        )
    ),
    "selected_test_runs_csv": str(
        PTQ_SELECTED_TEST_RUNS_CSV
    ),
    "selected_test_runs_csv_sha256": (
        sha256_file(
            PTQ_SELECTED_TEST_RUNS_CSV
        )
    ),
    "selected_test_group_csv": str(
        PTQ_SELECTED_TEST_GROUP_CSV
    ),
    "selected_test_group_csv_sha256": (
        sha256_file(
            PTQ_SELECTED_TEST_GROUP_CSV
        )
    ),
    "selected_test_summary_json": str(
        PTQ_SELECTED_TEST_SUMMARY_JSON
    ),
    "selected_test_summary_json_sha256": (
        sha256_file(
            PTQ_SELECTED_TEST_SUMMARY_JSON
        )
    ),
    "ready_for_phase5_compression_matrix_closure": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_JSON,
    lock,
)

lock_manifest = {
    "status": "locked",
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
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            PHASE5_PROTOCOL
        ),
        file_record(
            B0_PAIR_LOCK
        ),
        file_record(
            B0_CHECKPOINT_REGISTRY
        ),
        file_record(
            PTQ_PREFLIGHT_LOCK
        ),
        file_record(
            PTQ_SELECTION_LOCK
        ),
        file_record(
            PTQ_VERIFIED_CANDIDATES_CSV
        ),
        file_record(
            PTQ_SELECTED_TEST_RUNS_CSV
        ),
        file_record(
            PTQ_SELECTED_TEST_GROUP_CSV
        ),
        file_record(
            PTQ_SELECTED_TEST_SUMMARY_JSON
        ),
        file_record(
            PTQ_SELECTED_TEST_COMPLETION_JSON
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
            CALIBRATION_INDEX_NPZ
        ),
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_VERIFIED_RUNS_CSV
        ),
        file_record(
            OUTPUT_VERIFICATION_JSON
        ),
        file_record(
            OUTPUT_LOCK_JSON
        ),
    ],
    "run_count": 10,
    "group_count": 2,
    "selected_calibration_sizes": (
        selected_sizes
    ),
    "test_evaluation_count_per_run": 1,
    "test_model_inference_repeated_by_verifier": (
        False
    ),
    "ready_for_phase5_compression_matrix_closure": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST_JSON,
    lock_manifest,
)

print()
print("=" * 92)
print("PHASE 5 SELECTED PTQ TEST VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Runs independently verified     : 10"
)
print(
    "Groups independently verified   : 2"
)
print(
    "Candidate INT8 checkpoint load  : PASSED"
)
print(
    "Train-only smoke inference      : PASSED"
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
    "Selected calibration sizes      : "
    f"{selected_sizes}"
)
print(
    "Selected PTQ test status        : LOCKED"
)
print(
    "Ready for Phase 5 closure       : True"
)
print(
    "Verification report             : "
    f"{OUTPUT_VERIFICATION_JSON}"
)
print(
    "Lock file                       : "
    f"{OUTPUT_LOCK_JSON}"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 5 SELECTED PTQ TEST VERIFIED AND LOCKED"
)
