from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "protocols"
    / "phase11_input_robustness_protocol_v3_3.json"
)

PROTOCOL_LOCK_PATH = (
    AUDIT
    / "phase11_input_robustness_protocol_locked_v3_3.json"
)

RUN_MATRIX_PATH = (
    AUDIT
    / "phase11_input_robustness_run_matrix_v3_3.csv"
)

CACHE_ROOT = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

X_TEST_PATH = CACHE_ROOT / "X_test.npy"
Y_TEST_PATH = CACHE_ROOT / "y_test.npy"
RAW_WEIGHT_PATH = (
    CACHE_ROOT
    / "raw_row_count_test.npy"
)

B0_REGISTRY_PATH = (
    AUDIT
    / "phase5_B0_checkpoint_registry_v3_2.csv"
)

PHASE5_MASTER_MATRIX_PATH = (
    AUDIT
    / "phase5_compression_master_run_matrix_v3_2.csv"
)

PHASE9_SOURCE_REGISTRY_PATH = (
    AUDIT
    / "phase9_final_artifact_source_registry_v3_2.csv"
)

HGB_SUMMARY_PATH = (
    AUDIT
    / "hist_gradient_boosting_b0_summary_v3_2.json"
)

HGB_MODEL_PATH = (
    ROOT
    / "results"
    / "v2"
    / "tabular_baselines"
    / "runs"
    / "hist_gradient_boosting_b0__seed_2026"
    / "model.joblib"
)

BUNDLE_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "final_artifact"
    / "tinyml_mlp_P50_QAT_seed2026_v3_2"
)

MODEL_SOURCE_PATH = (
    BUNDLE_ROOT
    / "nbaiot_models.py"
)

QAT_ENGINE_PATH = (
    BUNDLE_ROOT
    / "phase5_qat_engine_v3_2.py"
)

BUNDLE_SCALER_PATH = (
    BUNDLE_ROOT
    / "train_only_standard_scaler_v3_2.npz"
)

RESULTS_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "phase11_input_robustness_v3_3"
)

STAGING_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "phase11_input_robustness_v3_3.building"
)

OUTPUT_EXECUTION = (
    AUDIT
    / "phase11_input_robustness_execution_v3_3.json"
)

OUTPUT_COMPLETION = (
    AUDIT
    / "phase11_input_robustness_completed_v3_3.json"
)

PROTOCOL_VERSION = "phase11_input_robustness_v3_3"

EXPECTED_FEATURE_COUNT = 115
EXPECTED_TEST_ROWS = 371_796
EXPECTED_CLASS_COUNT = 3
EXPECTED_CLASS_LABELS = [0, 1, 2]
EXPECTED_CLASS_NAMES = [
    "benign",
    "gafgyt",
    "mirai",
]

CANONICAL_SEED = 2026
PERTURBATION_CHUNK_SIZE = 8192
NEURAL_BATCH_SIZE = 4096
CPU_THREADS = 1
INTEROP_THREADS = 1
CLEAN_METRIC_ATOL = 1e-10

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated.*",
    category=UserWarning,
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
    message="Default qconfig of oneDNN backend.*",
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


def read_csv_rows(
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

            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept the destination locked "
        f"after {WINDOWS_FILE_RETRY_COUNT} attempts: "
        f"{destination}"
    ) from last_error


def remove_with_retry(
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
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return
        except PermissionError as error:
            last_error = error

            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        f"Windows kept the path locked: {path}"
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
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    replace_with_retry(
        temporary,
        path,
    )


def write_csv(
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
    hidden_dims: list[int],
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
            kwargs[name] = list(hidden_dims)
            continue

        if (
            parameter.default
            is inspect.Parameter.empty
        ):
            raise RuntimeError(
                "Unresolved model-constructor "
                f"parameter: {name}"
            )

    return kwargs


def instantiate_model(
    model_module: ModuleType,
    hidden_dims: list[int],
) -> nn.Module:
    model_symbol = "TinyMLMLP"

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
            hidden_dims,
        )
    )

    if not isinstance(model, nn.Module):
        raise RuntimeError(
            "Model constructor did not return "
            "torch.nn.Module."
        )

    return model


def extract_logits(
    output: Any,
) -> torch.Tensor:
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
        "Model output did not contain logits."
    )


def build_B0_model(
    model_module: ModuleType,
    checkpoint_path: Path,
) -> nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "B0 model_state_dict missing."
        )

    model = instantiate_model(
        model_module,
        [64, 32],
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()
    return model


def build_QAT_model(
    model_module: ModuleType,
    qat_module: ModuleType,
    checkpoint_path: Path,
    structure_tensor: torch.Tensor,
) -> nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    backend = checkpoint.get(
        "QAT_backend"
    )

    if (
        backend
        not in torch.backends.quantized.supported_engines
    ):
        raise RuntimeError(
            f"Unsupported QAT backend: {backend}"
        )

    torch.backends.quantized.engine = backend

    float_model = instantiate_model(
        model_module,
        [32, 16],
    )

    prepared_model = (
        qat_module.prepare_qat_model(
            float_model,
            backend,
        )
    )

    prepared_model.eval()

    with torch.inference_mode():
        prepared_model(
            structure_tensor
        )

    int8_model = (
        qat_module.convert_qat_model(
            prepared_model
        )
    )

    state_dict = checkpoint.get(
        "INT8_model_state_dict"
    )

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "QAT INT8_model_state_dict missing."
        )

    int8_model.load_state_dict(
        state_dict,
        strict=True,
    )

    int8_model.eval()
    return int8_model


def neural_predict(
    model: nn.Module,
    values: np.ndarray,
) -> np.ndarray:
    predictions = np.empty(
        values.shape[0],
        dtype=np.int64,
    )

    with torch.inference_mode():
        for start in range(
            0,
            values.shape[0],
            NEURAL_BATCH_SIZE,
        ):
            stop = min(
                start + NEURAL_BATCH_SIZE,
                values.shape[0],
            )

            tensor = torch.from_numpy(
                np.ascontiguousarray(
                    values[start:stop],
                    dtype=np.float32,
                )
            )

            logits = extract_logits(
                model(tensor)
            )

            if tuple(logits.shape) != (
                stop - start,
                EXPECTED_CLASS_COUNT,
            ):
                raise RuntimeError(
                    "Unexpected neural output shape."
                )

            if not bool(
                torch.isfinite(
                    logits
                ).all().item()
            ):
                raise RuntimeError(
                    "Non-finite neural output."
                )

            predictions[start:stop] = (
                torch.argmax(
                    logits,
                    dim=1,
                )
                .cpu()
                .numpy()
                .astype(
                    np.int64,
                    copy=False,
                )
            )

    return predictions


def perturb_chunk(
    source: np.ndarray,
    condition: dict[str, Any],
    rng: np.random.Generator | None,
    mean64: np.ndarray,
    scale64: np.ndarray,
) -> np.ndarray:
    family = condition["family"]
    severity = float(
        condition["severity"]
    )

    source64 = np.asarray(
        source,
        dtype=np.float64,
    )

    if family == "clean":
        perturbed64 = source64

    elif family == "gaussian_noise":
        if rng is None:
            raise RuntimeError(
                "Gaussian-noise RNG missing."
            )

        noise = rng.standard_normal(
            source64.shape
        )

        perturbed64 = (
            source64
            + severity
            * scale64
            * noise
        )

    elif family == "feature_masking":
        if rng is None:
            raise RuntimeError(
                "Feature-masking RNG missing."
            )

        mask = (
            rng.random(
                source64.shape
            )
            < severity
        )

        perturbed64 = np.where(
            mask,
            mean64,
            source64,
        )

    elif family == "scale_drift":
        factor = 1.0 + severity

        perturbed64 = (
            mean64
            + factor
            * (
                source64
                - mean64
            )
        )

    else:
        raise RuntimeError(
            f"Unknown condition family: {family}"
        )

    perturbed32 = np.ascontiguousarray(
        perturbed64,
        dtype=np.float32,
    )

    if not np.isfinite(
        perturbed32
    ).all():
        raise RuntimeError(
            "Perturbation produced non-finite values."
        )

    return perturbed32


def standardize_for_neural(
    values: np.ndarray,
    mean64: np.ndarray,
    scale64: np.ndarray,
) -> np.ndarray:
    standardized = (
        (
            np.asarray(
                values,
                dtype=np.float64,
            )
            - mean64
        )
        / scale64
    ).astype(
        np.float32
    )

    standardized = np.ascontiguousarray(
        standardized
    )

    if not np.isfinite(
        standardized
    ).all():
        raise RuntimeError(
            "Standardization produced "
            "non-finite values."
        )

    return standardized


def update_confusion(
    matrix: np.ndarray,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray | None = None,
) -> None:
    if weights is None:
        np.add.at(
            matrix,
            (
                true_labels,
                predictions,
            ),
            1,
        )
    else:
        np.add.at(
            matrix,
            (
                true_labels,
                predictions,
            ),
            weights,
        )


def metrics_from_confusion(
    matrix: np.ndarray,
) -> dict[str, Any]:
    matrix64 = np.asarray(
        matrix,
        dtype=np.float64,
    )

    true_positive = np.diag(
        matrix64
    )

    support = matrix64.sum(
        axis=1
    )

    predicted = matrix64.sum(
        axis=0
    )

    total = float(
        matrix64.sum()
    )

    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(
            true_positive
        ),
        where=support > 0,
    )

    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(
            true_positive
        ),
        where=predicted > 0,
    )

    f1 = np.divide(
        2.0
        * precision
        * recall,
        precision + recall,
        out=np.zeros_like(
            true_positive
        ),
        where=(
            precision + recall
        ) > 0,
    )

    accuracy = (
        float(
            true_positive.sum()
            / total
        )
        if total > 0
        else 0.0
    )

    return {
        "macro_f1": float(
            np.mean(f1)
        ),
        "accuracy": accuracy,
        "class_precision": [
            float(value)
            for value in precision
        ],
        "class_recall": [
            float(value)
            for value in recall
        ],
        "class_f1": [
            float(value)
            for value in f1
        ],
        "gafgyt_fnr": float(
            1.0 - recall[1]
        ),
        "mirai_fnr": float(
            1.0 - recall[2]
        ),
        "total_weight": total,
    }


def normalize_key(
    value: str,
) -> str:
    return "".join(
        character
        for character in value.lower()
        if character.isalnum()
    )


def numeric_from_row(
    row: dict[str, str],
    aliases: list[str],
    required_tokens: list[str],
) -> float:
    normalized = {
        normalize_key(key): value
        for key, value in row.items()
    }

    for alias in aliases:
        key = normalize_key(alias)

        if key in normalized:
            return float(
                normalized[key]
            )

    candidates: list[
        tuple[int, str, str]
    ] = []

    normalized_tokens = [
        normalize_key(token)
        for token in required_tokens
    ]

    for key, value in row.items():
        normalized_key = normalize_key(
            key
        )

        if all(
            token in normalized_key
            for token in normalized_tokens
        ):
            if any(
                forbidden in normalized_key
                for forbidden in (
                    "mean",
                    "std",
                    "delta",
                    "drop",
                    "ratio",
                )
            ):
                continue

            candidates.append(
                (
                    len(normalized_key),
                    key,
                    value,
                )
            )

    if len(candidates) == 1:
        return float(
            candidates[0][2]
        )

    if candidates:
        candidates.sort()
        return float(
            candidates[0][2]
        )

    raise KeyError(
        "Could not resolve metric from row: "
        + ", ".join(
            required_tokens
        )
    )


def flatten_json(
    value: Any,
    prefix: str = "",
) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.extend(
                flatten_json(
                    child,
                    child_prefix,
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(
                flatten_json(
                    child,
                    f"{prefix}[{index}]",
                )
            )

    else:
        rows.append(
            (
                prefix,
                value,
            )
        )

    return rows


def numeric_from_json(
    value: dict[str, Any],
    required_tokens: list[str],
) -> float:
    normalized_tokens = [
        normalize_key(token)
        for token in required_tokens
    ]

    candidates: list[
        tuple[int, str, float]
    ] = []

    for path, child in flatten_json(
        value
    ):
        if not isinstance(
            child,
            (int, float),
        ):
            continue

        normalized_path = normalize_key(
            path
        )

        if all(
            token in normalized_path
            for token in normalized_tokens
        ):
            if any(
                forbidden in normalized_path
                for forbidden in (
                    "mean",
                    "std",
                    "delta",
                    "drop",
                    "ratio",
                    "validation",
                    "train",
                )
            ):
                continue

            candidates.append(
                (
                    len(normalized_path),
                    path,
                    float(child),
                )
            )

    if not candidates:
        raise KeyError(
            "Could not resolve metric from JSON: "
            + ", ".join(
                required_tokens
            )
        )

    candidates.sort()
    return candidates[0][2]


def expected_metrics_from_row(
    row: dict[str, str],
) -> dict[str, float]:
    return {
        "macro_f1": numeric_from_row(
            row,
            [
                "test_fingerprint_macro_f1",
                "fingerprint_macro_f1",
            ],
            [
                "test",
                "fingerprint",
                "macro",
                "f1",
            ],
        ),
        "accuracy": numeric_from_row(
            row,
            [
                "test_fingerprint_accuracy",
                "fingerprint_accuracy",
            ],
            [
                "test",
                "fingerprint",
                "accuracy",
            ],
        ),
        "weighted_macro_f1": (
            numeric_from_row(
                row,
                [
                    "test_raw_weighted_macro_f1",
                    "raw_weighted_macro_f1",
                ],
                [
                    "test",
                    "raw",
                    "weighted",
                    "macro",
                    "f1",
                ],
            )
        ),
        "gafgyt_fnr": numeric_from_row(
            row,
            [
                "test_gafgyt_fnr",
                "gafgyt_fnr",
            ],
            [
                "test",
                "gafgyt",
                "fnr",
            ],
        ),
        "mirai_fnr": numeric_from_row(
            row,
            [
                "test_mirai_fnr",
                "mirai_fnr",
            ],
            [
                "test",
                "mirai",
                "fnr",
            ],
        ),
    }


required_paths = (
    PROTOCOL_PATH,
    PROTOCOL_LOCK_PATH,
    RUN_MATRIX_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    RAW_WEIGHT_PATH,
    B0_REGISTRY_PATH,
    PHASE5_MASTER_MATRIX_PATH,
    PHASE9_SOURCE_REGISTRY_PATH,
    HGB_SUMMARY_PATH,
    HGB_MODEL_PATH,
    MODEL_SOURCE_PATH,
    QAT_ENGINE_PATH,
    BUNDLE_SCALER_PATH,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    RESULTS_ROOT,
    STAGING_ROOT,
    OUTPUT_EXECUTION,
    OUTPUT_COMPLETION,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 11 execution output already "
            f"exists; refusing to overwrite: {output_path}"
        )

protocol = read_json(
    PROTOCOL_PATH
)

protocol_lock = read_json(
    PROTOCOL_LOCK_PATH
)

run_matrix = read_csv_rows(
    RUN_MATRIX_PATH
)

entry_checks = {
    "protocol_locked": (
        protocol.get("status")
        == "locked"
        and protocol.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
        and protocol.get(
            "all_checks_passed"
        )
        is True
    ),
    "protocol_lock_ready": (
        protocol_lock.get("status")
        == "locked"
        and protocol_lock.get(
            "ready_for_robustness_execution"
        )
        is True
        and protocol_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "protocol_hash_matches": (
        protocol_lock.get(
            "protocol_sha256"
        )
        == sha256_file(
            PROTOCOL_PATH
        )
    ),
    "run_matrix_hash_matches": (
        protocol_lock.get(
            "run_matrix_sha256"
        )
        == sha256_file(
            RUN_MATRIX_PATH
        )
    ),
    "run_matrix_rows_27": (
        len(run_matrix) == 27
    ),
    "results_absent": (
        not RESULTS_ROOT.exists()
        and not STAGING_ROOT.exists()
    ),
    "model_selection_disabled": (
        protocol[
            "selection_policy"
        ][
            "model_selection_performed"
        ]
        is False
        and protocol[
            "selection_policy"
        ][
            "results_may_change_final_model"
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
        "Phase 11 execution entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

B0_rows = read_csv_rows(
    B0_REGISTRY_PATH
)

B0_matches = [
    row
    for row in B0_rows
    if (
        row.get("architecture")
        == "tinyml_mlp"
        and row.get("variant")
        == "B0"
        and row.get("seed")
        == str(CANONICAL_SEED)
    )
]

if len(B0_matches) != 1:
    raise RuntimeError(
        "TinyML B0 seed-2026 registry row "
        "was not unique."
    )

B0_row = B0_matches[0]
B0_checkpoint_path = Path(
    B0_row["checkpoint_path"]
)

phase9_rows = read_csv_rows(
    PHASE9_SOURCE_REGISTRY_PATH
)

QAT_matches = [
    row
    for row in phase9_rows
    if (
        row.get("architecture")
        == "tinyml_mlp"
        and row.get("variant")
        == "P50-QAT"
        and row.get("seed")
        == str(CANONICAL_SEED)
    )
]

if len(QAT_matches) != 1:
    raise RuntimeError(
        "TinyML P50-QAT seed-2026 source row "
        "was not unique."
    )

QAT_row = QAT_matches[0]
QAT_checkpoint_path = Path(
    QAT_row["checkpoint_path"]
)

phase5_master_rows = read_csv_rows(
    PHASE5_MASTER_MATRIX_PATH
)

QAT_master_matches = [
    row
    for row in phase5_master_rows
    if (
        row.get(
            "architecture",
            "",
        ).lower()
        == "tinyml_mlp"
        and row.get(
            "variant",
            "",
        ).lower().replace(
            "_",
            "-",
        )
        == "p50-qat"
        and row.get("seed")
        == str(CANONICAL_SEED)
    )
]

QAT_metrics_json = read_json(
    Path(
        QAT_row["metrics_path"]
    )
)

if len(QAT_master_matches) == 1:
    try:
        expected_QAT = (
            expected_metrics_from_row(
                QAT_master_matches[0]
            )
        )
    except Exception:
        expected_QAT = {
            "macro_f1": numeric_from_json(
                QAT_metrics_json,
                [
                    "test",
                    "fingerprint",
                    "macro",
                    "f1",
                ],
            ),
            "accuracy": numeric_from_json(
                QAT_metrics_json,
                [
                    "test",
                    "fingerprint",
                    "accuracy",
                ],
            ),
            "weighted_macro_f1": (
                numeric_from_json(
                    QAT_metrics_json,
                    [
                        "test",
                        "raw",
                        "weighted",
                        "macro",
                        "f1",
                    ],
                )
            ),
            "gafgyt_fnr": numeric_from_json(
                QAT_metrics_json,
                [
                    "test",
                    "gafgyt",
                    "fnr",
                ],
            ),
            "mirai_fnr": numeric_from_json(
                QAT_metrics_json,
                [
                    "test",
                    "mirai",
                    "fnr",
                ],
            ),
        }
else:
    expected_QAT = {
        "macro_f1": numeric_from_json(
            QAT_metrics_json,
            [
                "test",
                "fingerprint",
                "macro",
                "f1",
            ],
        ),
        "accuracy": numeric_from_json(
            QAT_metrics_json,
            [
                "test",
                "fingerprint",
                "accuracy",
            ],
        ),
        "weighted_macro_f1": (
            numeric_from_json(
                QAT_metrics_json,
                [
                    "test",
                    "raw",
                    "weighted",
                    "macro",
                    "f1",
                ],
            )
        ),
        "gafgyt_fnr": numeric_from_json(
            QAT_metrics_json,
            [
                "test",
                "gafgyt",
                "fnr",
            ],
        ),
        "mirai_fnr": numeric_from_json(
            QAT_metrics_json,
            [
                "test",
                "mirai",
                "fnr",
            ],
        ),
    }

hgb_summary = read_json(
    HGB_SUMMARY_PATH
)

expected_clean = {
    "tinyml_mlp_B0": (
        expected_metrics_from_row(
            B0_row
        )
    ),
    "tinyml_mlp_P50_QAT": (
        expected_QAT
    ),
    "hist_gradient_boosting_B0": {
        "macro_f1": float(
            hgb_summary[
                "test_fingerprint_macro_f1"
            ]
        ),
        "accuracy": float(
            hgb_summary[
                "test_fingerprint_accuracy"
            ]
        ),
        "weighted_macro_f1": float(
            hgb_summary[
                "test_raw_weighted_macro_f1"
            ]
        ),
        "gafgyt_fnr": float(
            hgb_summary[
                "test_gafgyt_fnr"
            ]
        ),
        "mirai_fnr": float(
            hgb_summary[
                "test_mirai_fnr"
            ]
        ),
    },
}

x_test = np.load(
    X_TEST_PATH,
    mmap_mode="r",
    allow_pickle=False,
)

y_test = np.load(
    Y_TEST_PATH,
    mmap_mode="r",
    allow_pickle=False,
)

raw_weights = np.load(
    RAW_WEIGHT_PATH,
    mmap_mode="r",
    allow_pickle=False,
)

if tuple(x_test.shape) != (
    EXPECTED_TEST_ROWS,
    EXPECTED_FEATURE_COUNT,
):
    raise RuntimeError(
        "X_test shape mismatch."
    )

if tuple(y_test.shape) != (
    EXPECTED_TEST_ROWS,
):
    raise RuntimeError(
        "y_test shape mismatch."
    )

if tuple(raw_weights.shape) != (
    EXPECTED_TEST_ROWS,
):
    raise RuntimeError(
        "raw test-weight shape mismatch."
    )

with np.load(
    BUNDLE_SCALER_PATH,
    allow_pickle=False,
) as scaler:
    mean64 = np.array(
        scaler[
            "mean_float64"
        ],
        dtype=np.float64,
        copy=True,
    )

    scale64 = np.array(
        scaler[
            "scale_float64"
        ],
        dtype=np.float64,
        copy=True,
    )

if (
    mean64.shape
    != (
        EXPECTED_FEATURE_COUNT,
    )
    or scale64.shape
    != (
        EXPECTED_FEATURE_COUNT,
    )
):
    raise RuntimeError(
        "Bundled scaler shape mismatch."
    )

if (
    not np.isfinite(mean64).all()
    or not np.isfinite(scale64).all()
    or np.any(scale64 <= 0.0)
):
    raise RuntimeError(
        "Bundled scaler values are invalid."
    )

torch.set_num_threads(
    CPU_THREADS
)

try:
    torch.set_num_interop_threads(
        INTEROP_THREADS
    )
except RuntimeError:
    pass

torch.use_deterministic_algorithms(
    True
)

if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(BUNDLE_ROOT),
    )

model_module = load_module(
    MODEL_SOURCE_PATH,
    "phase11_nbaiot_models",
)

qat_module = load_module(
    QAT_ENGINE_PATH,
    "phase11_qat_engine",
)

clean_structure = standardize_for_neural(
    np.asarray(
        x_test[:32],
        dtype=np.float32,
    ),
    mean64,
    scale64,
)

structure_tensor = torch.from_numpy(
    clean_structure
)

B0_model = build_B0_model(
    model_module,
    B0_checkpoint_path,
)

QAT_model = build_QAT_model(
    model_module,
    qat_module,
    QAT_checkpoint_path,
    structure_tensor,
)

hgb_model = joblib.load(
    HGB_MODEL_PATH
)

smoke_checks = {
    "B0_batch32_shape": (
        neural_predict(
            B0_model,
            clean_structure,
        ).shape
        == (32,)
    ),
    "QAT_batch32_shape": (
        neural_predict(
            QAT_model,
            clean_structure,
        ).shape
        == (32,)
    ),
    "HGB_batch32_shape": (
        np.asarray(
            hgb_model.predict(
                np.asarray(
                    x_test[:32],
                    dtype=np.float32,
                )
            )
        ).shape
        == (32,)
    ),
}

failed_smoke_checks = [
    name
    for name, passed
    in smoke_checks.items()
    if not passed
]

if failed_smoke_checks:
    raise RuntimeError(
        "Phase 11 model smoke test failed: "
        + ", ".join(
            failed_smoke_checks
        )
    )

condition_specs = sorted(
    protocol["conditions"],
    key=lambda row: int(
        row["condition_order"]
    ),
)

if len(condition_specs) != 9:
    raise RuntimeError(
        "Locked condition count is not 9."
    )

print("=" * 92)
print("PHASE 11 INPUT ROBUSTNESS EXECUTION")
print("=" * 92)
print(
    "Models                          : 3"
)
print(
    "Conditions                      : 9"
)
print(
    "Planned evaluations             : 27"
)
print(
    "Test rows                       : 371796"
)
print(
    "Perturbation chunk size         : 8192"
)
print(
    "Neural inference batch size     : 4096"
)
print(
    "Shared perturbations            : True"
)
print(
    "Model retraining                : False"
)
print(
    "Test used for selection         : False"
)
print(
    "Final model change allowed      : False"
)
print()

execution_started_at = utc_now()
execution_start_time = time.perf_counter()
execution_succeeded = False

all_results: list[
    dict[str, Any]
] = []

condition_hashes: dict[
    str,
    str,
] = {}

try:
    STAGING_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    for condition_index, condition in enumerate(
        condition_specs,
        start=1,
    ):
        condition_id = str(
            condition["condition_id"]
        )

        condition_start = time.perf_counter()

        random_seed = condition.get(
            "random_seed"
        )

        rng = (
            None
            if random_seed is None
            else np.random.default_rng(
                int(random_seed)
            )
        )

        perturb_digest = hashlib.sha256()
        perturb_digest.update(
            (
                "phase11_perturbation_array_v1\0"
                + condition_id
                + "\0"
            ).encode(
                "utf-8"
            )
        )

        confusion = {
            "tinyml_mlp_B0": np.zeros(
                (
                    EXPECTED_CLASS_COUNT,
                    EXPECTED_CLASS_COUNT,
                ),
                dtype=np.int64,
            ),
            "tinyml_mlp_P50_QAT": np.zeros(
                (
                    EXPECTED_CLASS_COUNT,
                    EXPECTED_CLASS_COUNT,
                ),
                dtype=np.int64,
            ),
            "hist_gradient_boosting_B0": np.zeros(
                (
                    EXPECTED_CLASS_COUNT,
                    EXPECTED_CLASS_COUNT,
                ),
                dtype=np.int64,
            ),
        }

        weighted_confusion = {
            model_id: np.zeros(
                (
                    EXPECTED_CLASS_COUNT,
                    EXPECTED_CLASS_COUNT,
                ),
                dtype=np.int64,
            )
            for model_id in confusion
        }

        for start in range(
            0,
            EXPECTED_TEST_ROWS,
            PERTURBATION_CHUNK_SIZE,
        ):
            stop = min(
                start
                + PERTURBATION_CHUNK_SIZE,
                EXPECTED_TEST_ROWS,
            )

            source_chunk = np.asarray(
                x_test[start:stop],
                dtype=np.float32,
            )

            perturbed_chunk = (
                perturb_chunk(
                    source_chunk,
                    condition,
                    rng,
                    mean64,
                    scale64,
                )
            )

            perturb_digest.update(
                perturbed_chunk.tobytes(
                    order="C"
                )
            )

            true_chunk = np.asarray(
                y_test[start:stop],
                dtype=np.int64,
            )

            weight_chunk = np.asarray(
                raw_weights[start:stop],
                dtype=np.int64,
            )

            hgb_predictions = np.asarray(
                hgb_model.predict(
                    perturbed_chunk
                ),
                dtype=np.int64,
            )

            standardized_chunk = (
                standardize_for_neural(
                    perturbed_chunk,
                    mean64,
                    scale64,
                )
            )

            B0_predictions = neural_predict(
                B0_model,
                standardized_chunk,
            )

            QAT_predictions = neural_predict(
                QAT_model,
                standardized_chunk,
            )

            predictions_by_model = {
                "tinyml_mlp_B0": (
                    B0_predictions
                ),
                "tinyml_mlp_P50_QAT": (
                    QAT_predictions
                ),
                "hist_gradient_boosting_B0": (
                    hgb_predictions
                ),
            }

            for model_id, predictions in (
                predictions_by_model.items()
            ):
                if predictions.shape != (
                    stop - start,
                ):
                    raise RuntimeError(
                        "Prediction shape mismatch "
                        f"for {model_id}."
                    )

                if not np.isin(
                    predictions,
                    EXPECTED_CLASS_LABELS,
                ).all():
                    raise RuntimeError(
                        "Prediction outside class "
                        f"range for {model_id}."
                    )

                update_confusion(
                    confusion[model_id],
                    true_chunk,
                    predictions,
                )

                update_confusion(
                    weighted_confusion[
                        model_id
                    ],
                    true_chunk,
                    predictions,
                    weight_chunk,
                )

        condition_hash = (
            perturb_digest.hexdigest()
        )

        condition_hashes[
            condition_id
        ] = condition_hash

        condition_elapsed = (
            time.perf_counter()
            - condition_start
        )

        condition_rows: list[
            dict[str, Any]
        ] = []

        for model_id in (
            "tinyml_mlp_B0",
            "tinyml_mlp_P50_QAT",
            "hist_gradient_boosting_B0",
        ):
            fingerprint_metrics = (
                metrics_from_confusion(
                    confusion[model_id]
                )
            )

            weighted_metrics = (
                metrics_from_confusion(
                    weighted_confusion[
                        model_id
                    ]
                )
            )

            result = {
                "run_id": (
                    f"{model_id}__"
                    f"{condition_id}"
                ),
                "model_id": model_id,
                "condition_id": (
                    condition_id
                ),
                "condition_family": (
                    condition["family"]
                ),
                "severity": float(
                    condition["severity"]
                ),
                "perturbation_seed": (
                    None
                    if random_seed is None
                    else int(random_seed)
                ),
                "perturbed_feature_array_sha256": (
                    condition_hash
                ),
                "test_rows": (
                    EXPECTED_TEST_ROWS
                ),
                "represented_raw_rows": int(
                    np.asarray(
                        raw_weights,
                        dtype=np.int64,
                    ).sum()
                ),
                "fingerprint_macro_f1": (
                    fingerprint_metrics[
                        "macro_f1"
                    ]
                ),
                "fingerprint_accuracy": (
                    fingerprint_metrics[
                        "accuracy"
                    ]
                ),
                "fingerprint_class_precision": (
                    fingerprint_metrics[
                        "class_precision"
                    ]
                ),
                "fingerprint_class_recall": (
                    fingerprint_metrics[
                        "class_recall"
                    ]
                ),
                "fingerprint_class_f1": (
                    fingerprint_metrics[
                        "class_f1"
                    ]
                ),
                "gafgyt_fnr": (
                    fingerprint_metrics[
                        "gafgyt_fnr"
                    ]
                ),
                "mirai_fnr": (
                    fingerprint_metrics[
                        "mirai_fnr"
                    ]
                ),
                "raw_weighted_macro_f1": (
                    weighted_metrics[
                        "macro_f1"
                    ]
                ),
                "raw_weighted_accuracy": (
                    weighted_metrics[
                        "accuracy"
                    ]
                ),
                "raw_weighted_class_recall": (
                    weighted_metrics[
                        "class_recall"
                    ]
                ),
                "confusion_matrix": (
                    confusion[
                        model_id
                    ].tolist()
                ),
                "raw_weighted_confusion_matrix": (
                    weighted_confusion[
                        model_id
                    ].tolist()
                ),
                "severe_fragility": bool(
                    fingerprint_metrics[
                        "macro_f1"
                    ]
                    < 0.85
                    or fingerprint_metrics[
                        "gafgyt_fnr"
                    ]
                    > 0.40
                    or fingerprint_metrics[
                        "mirai_fnr"
                    ]
                    > 0.40
                ),
                "model_retraining": False,
                "test_used_for_model_selection": (
                    False
                ),
                "final_model_changed": False,
                "condition_elapsed_seconds": (
                    float(
                        condition_elapsed
                    )
                ),
            }

            condition_rows.append(
                result
            )

        if condition_id == "clean":
            clean_failures: list[str] = []

            for result in condition_rows:
                model_id = result[
                    "model_id"
                ]

                expected = expected_clean[
                    model_id
                ]

                observed_values = {
                    "macro_f1": result[
                        "fingerprint_macro_f1"
                    ],
                    "accuracy": result[
                        "fingerprint_accuracy"
                    ],
                    "weighted_macro_f1": result[
                        "raw_weighted_macro_f1"
                    ],
                    "gafgyt_fnr": result[
                        "gafgyt_fnr"
                    ],
                    "mirai_fnr": result[
                        "mirai_fnr"
                    ],
                }

                for metric_name, observed in (
                    observed_values.items()
                ):
                    reference = expected[
                        metric_name
                    ]

                    if not np.isclose(
                        observed,
                        reference,
                        rtol=0.0,
                        atol=CLEAN_METRIC_ATOL,
                    ):
                        clean_failures.append(
                            f"{model_id}:{metric_name}:"
                            f"observed={observed:.15f}:"
                            f"expected={reference:.15f}"
                        )

            if clean_failures:
                raise RuntimeError(
                    "Clean-condition reproduction "
                    "gate failed: "
                    + " | ".join(
                        clean_failures
                    )
                )

        for result in condition_rows:
            model_directory = (
                STAGING_ROOT
                / result["model_id"]
                / result["condition_id"]
            )

            model_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            atomic_json(
                model_directory
                / "metrics.json",
                {
                    "status": "completed",
                    "phase": 11,
                    "protocol_version": (
                        PROTOCOL_VERSION
                    ),
                    **result,
                    "all_checks_passed": (
                        True
                    ),
                },
            )

            all_results.append(
                result
            )

            print(
                f"[{condition_index}/9] "
                f"{result['condition_id']:<28} | "
                f"{result['model_id']:<29} | "
                f"Macro-F1="
                f"{result['fingerprint_macro_f1']:.9f} | "
                f"Weighted-F1="
                f"{result['raw_weighted_macro_f1']:.9f} | "
                f"G-FNR="
                f"{result['gafgyt_fnr']:.6f} | "
                f"M-FNR="
                f"{result['mirai_fnr']:.6f} | "
                f"fragile="
                f"{result['severe_fragility']}",
                flush=True,
            )

    if len(all_results) != 27:
        raise RuntimeError(
            "Completed result count is not 27."
        )

    clean_by_model = {
        row["model_id"]: row
        for row in all_results
        if row["condition_id"]
        == "clean"
    }

    for row in all_results:
        clean = clean_by_model[
            row["model_id"]
        ]

        macro_drop = (
            clean[
                "fingerprint_macro_f1"
            ]
            - row[
                "fingerprint_macro_f1"
            ]
        )

        weighted_drop = (
            clean[
                "raw_weighted_macro_f1"
            ]
            - row[
                "raw_weighted_macro_f1"
            ]
        )

        clean_macro = clean[
            "fingerprint_macro_f1"
        ]

        row[
            "macro_f1_drop_from_model_clean"
        ] = float(macro_drop)

        row[
            "macro_f1_retention_ratio"
        ] = float(
            row[
                "fingerprint_macro_f1"
            ]
            / clean_macro
            if clean_macro > 0
            else 0.0
        )

        row[
            "weighted_macro_f1_drop_from_model_clean"
        ] = float(
            weighted_drop
        )

        row[
            "material_sensitivity"
        ] = bool(
            macro_drop > 0.05
        )

    aggregate_rows: list[
        dict[str, Any]
    ] = []

    for row in all_results:
        aggregate_rows.append(
            {
                "run_id": row["run_id"],
                "model_id": (
                    row["model_id"]
                ),
                "condition_id": (
                    row["condition_id"]
                ),
                "condition_family": (
                    row[
                        "condition_family"
                    ]
                ),
                "severity": row["severity"],
                "perturbation_seed": (
                    ""
                    if row[
                        "perturbation_seed"
                    ]
                    is None
                    else row[
                        "perturbation_seed"
                    ]
                ),
                "perturbed_feature_array_sha256": (
                    row[
                        "perturbed_feature_array_sha256"
                    ]
                ),
                "test_rows": row["test_rows"],
                "represented_raw_rows": (
                    row[
                        "represented_raw_rows"
                    ]
                ),
                "fingerprint_macro_f1": (
                    row[
                        "fingerprint_macro_f1"
                    ]
                ),
                "fingerprint_accuracy": (
                    row[
                        "fingerprint_accuracy"
                    ]
                ),
                "gafgyt_fnr": (
                    row["gafgyt_fnr"]
                ),
                "mirai_fnr": (
                    row["mirai_fnr"]
                ),
                "raw_weighted_macro_f1": (
                    row[
                        "raw_weighted_macro_f1"
                    ]
                ),
                "raw_weighted_accuracy": (
                    row[
                        "raw_weighted_accuracy"
                    ]
                ),
                "macro_f1_drop_from_model_clean": (
                    row[
                        "macro_f1_drop_from_model_clean"
                    ]
                ),
                "macro_f1_retention_ratio": (
                    row[
                        "macro_f1_retention_ratio"
                    ]
                ),
                "weighted_macro_f1_drop_from_model_clean": (
                    row[
                        "weighted_macro_f1_drop_from_model_clean"
                    ]
                ),
                "severe_fragility": (
                    row[
                        "severe_fragility"
                    ]
                ),
                "material_sensitivity": (
                    row[
                        "material_sensitivity"
                    ]
                ),
                "model_retraining": False,
                "test_used_for_model_selection": (
                    False
                ),
                "final_model_changed": False,
            }
        )

    aggregate_csv_path = (
        STAGING_ROOT
        / "phase11_input_robustness_all_runs_v3_3.csv"
    )

    write_csv(
        aggregate_csv_path,
        aggregate_rows,
        fieldnames=list(
            aggregate_rows[0].keys()
        ),
    )

    model_summaries: dict[
        str,
        dict[str, Any],
    ] = {}

    for model_id in (
        "tinyml_mlp_B0",
        "tinyml_mlp_P50_QAT",
        "hist_gradient_boosting_B0",
    ):
        model_rows = [
            row
            for row in all_results
            if row["model_id"]
            == model_id
        ]

        clean = clean_by_model[
            model_id
        ]

        worst = min(
            model_rows,
            key=lambda row: row[
                "fingerprint_macro_f1"
            ],
        )

        model_summaries[
            model_id
        ] = {
            "clean_fingerprint_macro_f1": (
                clean[
                    "fingerprint_macro_f1"
                ]
            ),
            "clean_raw_weighted_macro_f1": (
                clean[
                    "raw_weighted_macro_f1"
                ]
            ),
            "worst_condition": (
                worst["condition_id"]
            ),
            "worst_fingerprint_macro_f1": (
                worst[
                    "fingerprint_macro_f1"
                ]
            ),
            "maximum_macro_f1_drop": max(
                row[
                    "macro_f1_drop_from_model_clean"
                ]
                for row in model_rows
            ),
            "severe_fragility_condition_count": sum(
                1
                for row in model_rows
                if row[
                    "severe_fragility"
                ]
            ),
            "material_sensitivity_condition_count": sum(
                1
                for row in model_rows
                if row[
                    "material_sensitivity"
                ]
            ),
        }

    summary_path = (
        STAGING_ROOT
        / "phase11_input_robustness_summary_v3_3.json"
    )

    summary = {
        "status": "completed",
        "phase": 11,
        "artifact_name": (
            "input_robustness_analysis"
        ),
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "completed_at_utc": utc_now(),
        "model_count": 3,
        "condition_count": 9,
        "evaluation_count": 27,
        "test_rows": (
            EXPECTED_TEST_ROWS
        ),
        "represented_raw_rows": int(
            np.asarray(
                raw_weights,
                dtype=np.int64,
            ).sum()
        ),
        "condition_perturbation_sha256": (
            condition_hashes
        ),
        "clean_reproduction_gate_passed": (
            True
        ),
        "model_summaries": (
            model_summaries
        ),
        "severe_fragility_evaluation_count": sum(
            1
            for row in all_results
            if row["severe_fragility"]
        ),
        "material_sensitivity_evaluation_count": sum(
            1
            for row in all_results
            if row[
                "material_sensitivity"
            ]
        ),
        "model_retraining_performed": False,
        "test_used_for_model_selection": (
            False
        ),
        "final_model_changed": False,
        "ready_for_independent_verification": (
            True
        ),
        "all_checks_passed": True,
    }

    atomic_json(
        summary_path,
        summary,
    )

    detailed_path = (
        STAGING_ROOT
        / "phase11_input_robustness_detailed_results_v3_3.json"
    )

    atomic_json(
        detailed_path,
        {
            "status": "completed",
            "phase": 11,
            "protocol_version": (
                PROTOCOL_VERSION
            ),
            "results": all_results,
            "all_checks_passed": True,
        },
    )

    replace_with_retry(
        STAGING_ROOT,
        RESULTS_ROOT,
    )

    execution_succeeded = True

finally:
    if not execution_succeeded:
        remove_with_retry(
            STAGING_ROOT
        )

aggregate_csv_path = (
    RESULTS_ROOT
    / "phase11_input_robustness_all_runs_v3_3.csv"
)

summary_path = (
    RESULTS_ROOT
    / "phase11_input_robustness_summary_v3_3.json"
)

detailed_path = (
    RESULTS_ROOT
    / "phase11_input_robustness_detailed_results_v3_3.json"
)

for path in (
    aggregate_csv_path,
    summary_path,
    detailed_path,
):
    if not path.exists():
        raise FileNotFoundError(path)

total_elapsed = (
    time.perf_counter()
    - execution_start_time
)

execution_checks = {
    "result_directory_created": (
        RESULTS_ROOT.is_dir()
    ),
    "aggregate_csv_created": (
        aggregate_csv_path.is_file()
    ),
    "summary_created": (
        summary_path.is_file()
    ),
    "detailed_results_created": (
        detailed_path.is_file()
    ),
    "completed_evaluations_27": (
        len(all_results) == 27
    ),
    "condition_hashes_9": (
        len(condition_hashes) == 9
    ),
    "clean_reproduction_passed": (
        True
    ),
    "model_retraining_false": all(
        row["model_retraining"]
        is False
        for row in all_results
    ),
    "test_selection_false": all(
        row[
            "test_used_for_model_selection"
        ]
        is False
        for row in all_results
    ),
    "final_model_unchanged": all(
        row[
            "final_model_changed"
        ]
        is False
        for row in all_results
    ),
}

failed_execution_checks = [
    name
    for name, passed
    in execution_checks.items()
    if not passed
]

if failed_execution_checks:
    raise RuntimeError(
        "Phase 11 execution closure failed: "
        + ", ".join(
            failed_execution_checks
        )
    )

execution_report = {
    "status": "completed",
    "phase": 11,
    "artifact_name": (
        "input_robustness_execution"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "execution_started_at_utc": (
        execution_started_at
    ),
    "completed_at_utc": utc_now(),
    "elapsed_seconds": float(
        total_elapsed
    ),
    "model_count": 3,
    "condition_count": 9,
    "completed_evaluation_count": 27,
    "results_root": str(
        RESULTS_ROOT
    ),
    "aggregate_csv": (
        file_record(
            aggregate_csv_path
        )
    ),
    "summary": (
        file_record(
            summary_path
        )
    ),
    "detailed_results": (
        file_record(
            detailed_path
        )
    ),
    "condition_perturbation_sha256": (
        condition_hashes
    ),
    "smoke_checks": (
        smoke_checks
    ),
    "execution_checks": (
        execution_checks
    ),
    "model_retraining_performed": False,
    "test_used_for_model_selection": (
        False
    ),
    "final_model_changed": False,
    "ready_for_independent_verification": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_EXECUTION,
    execution_report,
)

completion = {
    "status": "completed",
    "phase": 11,
    "artifact_name": (
        "input_robustness_analysis"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "completed_at_utc": utc_now(),
    "execution_report": (
        file_record(
            OUTPUT_EXECUTION
        )
    ),
    "results_root": str(
        RESULTS_ROOT
    ),
    "aggregate_csv_sha256": (
        sha256_file(
            aggregate_csv_path
        )
    ),
    "summary_sha256": (
        sha256_file(
            summary_path
        )
    ),
    "detailed_results_sha256": (
        sha256_file(
            detailed_path
        )
    ),
    "completed_evaluation_count": 27,
    "model_retraining_performed": False,
    "test_used_for_model_selection": (
        False
    ),
    "final_model_changed": False,
    "ready_for_independent_verification": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_COMPLETION,
    completion,
)

locked_summary = read_json(
    summary_path
)

print()
print("=" * 92)
print("PHASE 11 INPUT ROBUSTNESS EXECUTION SUMMARY")
print("=" * 92)
print(
    "Completed evaluations           : 27"
)
print(
    "Models                          : 3"
)
print(
    "Conditions                      : 9"
)
print(
    "Clean reproduction gate         : PASSED"
)
print(
    "Shared perturbations            : True"
)
print(
    "Model retraining performed      : False"
)
print(
    "Test used for model selection   : False"
)
print(
    "Final model changed             : False"
)
print(
    "Severe fragility evaluations    : "
    f"{locked_summary['severe_fragility_evaluation_count']}"
)
print(
    "Material sensitivity evaluations: "
    f"{locked_summary['material_sensitivity_evaluation_count']}"
)

for model_id, values in (
    locked_summary[
        "model_summaries"
    ].items()
):
    print(
        f"{model_id:<31}: "
        f"clean={values['clean_fingerprint_macro_f1']:.9f} | "
        f"worst={values['worst_fingerprint_macro_f1']:.9f} | "
        f"condition={values['worst_condition']} | "
        f"max_drop={values['maximum_macro_f1_drop']:.9f}"
    )

print(
    "Ready for independent verify    : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 11 ROBUSTNESS EXECUTION COMPLETED"
)
