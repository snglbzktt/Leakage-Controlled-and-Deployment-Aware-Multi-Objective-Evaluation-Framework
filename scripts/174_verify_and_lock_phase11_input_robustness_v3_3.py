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

EXECUTION_REPORT_PATH = (
    AUDIT
    / "phase11_input_robustness_execution_v3_3.json"
)

COMPLETION_PATH = (
    AUDIT
    / "phase11_input_robustness_completed_v3_3.json"
)

RESULTS_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "phase11_input_robustness_v3_3"
)

AGGREGATE_CSV_PATH = (
    RESULTS_ROOT
    / "phase11_input_robustness_all_runs_v3_3.csv"
)

SUMMARY_PATH = (
    RESULTS_ROOT
    / "phase11_input_robustness_summary_v3_3.json"
)

DETAILED_PATH = (
    RESULTS_ROOT
    / "phase11_input_robustness_detailed_results_v3_3.json"
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

PHASE9_SOURCE_REGISTRY_PATH = (
    AUDIT
    / "phase9_final_artifact_source_registry_v3_2.csv"
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

OUTPUT_VERIFICATION = (
    AUDIT
    / "phase11_input_robustness_verification_v3_3.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase11_input_robustness_locked_v3_3.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase11_input_robustness_lock_manifest_v3_3.json"
)

PROTOCOL_VERSION = "phase11_input_robustness_v3_3"

EXPECTED_FEATURE_COUNT = 115
EXPECTED_TEST_ROWS = 371_796
EXPECTED_CLASS_COUNT = 3
EXPECTED_CLASS_LABELS = [0, 1, 2]

CANONICAL_SEED = 2026
PERTURBATION_CHUNK_SIZE = 8192
INDEPENDENT_NEURAL_BATCH_SIZE = 2048
CPU_THREADS = 1
INTEROP_THREADS = 1

FLOAT_ATOL = 1e-12

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
                "Unresolved model constructor "
                f"parameter: {name}"
            )

    return kwargs


def instantiate_model(
    model_module: ModuleType,
    hidden_dims: list[int],
) -> nn.Module:
    candidate = getattr(
        model_module,
        "TinyMLMLP",
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
            INDEPENDENT_NEURAL_BATCH_SIZE,
        ):
            stop = min(
                start
                + INDEPENDENT_NEURAL_BATCH_SIZE,
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
    family = str(
        condition["family"]
    )

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

        perturbed64 = (
            source64
            + severity
            * scale64
            * rng.standard_normal(
                source64.shape
            )
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
    }


def assert_close(
    observed: float,
    expected: float,
    label: str,
) -> None:
    if not np.isclose(
        float(observed),
        float(expected),
        rtol=0.0,
        atol=FLOAT_ATOL,
    ):
        raise RuntimeError(
            f"{label} mismatch: "
            f"observed={observed:.15f}, "
            f"expected={expected:.15f}"
        )


required_paths = (
    PROTOCOL_PATH,
    PROTOCOL_LOCK_PATH,
    RUN_MATRIX_PATH,
    EXECUTION_REPORT_PATH,
    COMPLETION_PATH,
    RESULTS_ROOT,
    AGGREGATE_CSV_PATH,
    SUMMARY_PATH,
    DETAILED_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    RAW_WEIGHT_PATH,
    B0_REGISTRY_PATH,
    PHASE9_SOURCE_REGISTRY_PATH,
    HGB_MODEL_PATH,
    MODEL_SOURCE_PATH,
    QAT_ENGINE_PATH,
    BUNDLE_SCALER_PATH,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_VERIFICATION,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 11 verification output already "
            f"exists; refusing to overwrite: {output_path}"
        )

protocol = read_json(
    PROTOCOL_PATH
)

protocol_lock = read_json(
    PROTOCOL_LOCK_PATH
)

execution_report = read_json(
    EXECUTION_REPORT_PATH
)

completion = read_json(
    COMPLETION_PATH
)

summary = read_json(
    SUMMARY_PATH
)

detailed = read_json(
    DETAILED_PATH
)

aggregate_rows = read_csv_rows(
    AGGREGATE_CSV_PATH
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
    "execution_completed": (
        execution_report.get("status")
        == "completed"
        and execution_report.get(
            "completed_evaluation_count"
        )
        == 27
        and execution_report.get(
            "ready_for_independent_verification"
        )
        is True
        and execution_report.get(
            "all_checks_passed"
        )
        is True
    ),
    "completion_completed": (
        completion.get("status")
        == "completed"
        and completion.get(
            "completed_evaluation_count"
        )
        == 27
        and completion.get(
            "ready_for_independent_verification"
        )
        is True
        and completion.get(
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
    "execution_report_hash_matches": (
        completion[
            "execution_report"
        ][
            "sha256"
        ]
        == sha256_file(
            EXECUTION_REPORT_PATH
        )
    ),
    "aggregate_hash_matches": (
        completion.get(
            "aggregate_csv_sha256"
        )
        == sha256_file(
            AGGREGATE_CSV_PATH
        )
    ),
    "summary_hash_matches": (
        completion.get(
            "summary_sha256"
        )
        == sha256_file(
            SUMMARY_PATH
        )
    ),
    "detailed_hash_matches": (
        completion.get(
            "detailed_results_sha256"
        )
        == sha256_file(
            DETAILED_PATH
        )
    ),
    "run_matrix_rows_27": (
        len(run_matrix) == 27
    ),
    "aggregate_rows_27": (
        len(aggregate_rows) == 27
    ),
    "detailed_rows_27": (
        len(
            detailed.get(
                "results",
                [],
            )
        )
        == 27
    ),
    "model_selection_unchanged": (
        execution_report.get(
            "test_used_for_model_selection"
        )
        is False
        and execution_report.get(
            "final_model_changed"
        )
        is False
        and completion.get(
            "final_model_changed"
        )
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
        "Phase 11 independent verification "
        "entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

saved_results = {
    row["run_id"]: row
    for row in detailed[
        "results"
    ]
}

aggregate_lookup = {
    row["run_id"]: row
    for row in aggregate_rows
}

planned_run_ids = {
    row["run_id"]
    for row in run_matrix
}

if set(saved_results.keys()) != planned_run_ids:
    raise RuntimeError(
        "Detailed result run IDs do not match "
        "the locked run matrix."
    )

if set(aggregate_lookup.keys()) != planned_run_ids:
    raise RuntimeError(
        "Aggregate result run IDs do not match "
        "the locked run matrix."
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
        "TinyML B0 source row was not unique."
    )

B0_checkpoint_path = Path(
    B0_matches[0][
        "checkpoint_path"
    ]
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
        "TinyML P50-QAT source row was not unique."
    )

QAT_checkpoint_path = Path(
    QAT_matches[0][
        "checkpoint_path"
    ]
)

for path in (
    B0_checkpoint_path,
    QAT_checkpoint_path,
):
    if not path.exists():
        raise FileNotFoundError(path)

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
    "phase11_verify_models",
)

qat_module = load_module(
    QAT_ENGINE_PATH,
    "phase11_verify_qat",
)

structure_values = (
    (
        np.asarray(
            x_test[:32],
            dtype=np.float64,
        )
        - mean64
    )
    / scale64
).astype(np.float32)

structure_tensor = torch.from_numpy(
    np.ascontiguousarray(
        structure_values
    )
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

condition_specs = sorted(
    protocol["conditions"],
    key=lambda row: int(
        row["condition_order"]
    ),
)

print("=" * 92)
print("PHASE 11 INPUT ROBUSTNESS INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Models                          : 3"
)
print(
    "Conditions                      : 9"
)
print(
    "Evaluations to recompute        : 27"
)
print(
    "Independent neural batch size   : 2048"
)
print(
    "Model retraining                : False"
)
print(
    "Final model change allowed      : False"
)
print()

verification_started_at = utc_now()
verification_start_time = time.perf_counter()

recomputed_results: dict[
    str,
    dict[str, Any],
] = {}

condition_hashes: dict[
    str,
    str,
] = {}

for condition_index, condition in enumerate(
    condition_specs,
    start=1,
):
    condition_id = str(
        condition["condition_id"]
    )

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
            start + PERTURBATION_CHUNK_SIZE,
            EXPECTED_TEST_ROWS,
        )

        source_chunk = np.asarray(
            x_test[start:stop],
            dtype=np.float32,
        )

        perturbed_chunk = perturb_chunk(
            source_chunk,
            condition,
            rng,
            mean64,
            scale64,
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

        standardized_chunk = (
            standardize_for_neural(
                perturbed_chunk,
                mean64,
                scale64,
            )
        )

        predictions_by_model = {
            "tinyml_mlp_B0": (
                neural_predict(
                    B0_model,
                    standardized_chunk,
                )
            ),
            "tinyml_mlp_P50_QAT": (
                neural_predict(
                    QAT_model,
                    standardized_chunk,
                )
            ),
            "hist_gradient_boosting_B0": (
                np.asarray(
                    hgb_model.predict(
                        perturbed_chunk
                    ),
                    dtype=np.int64,
                )
            ),
        }

        for model_id, predictions in (
            predictions_by_model.items()
        ):
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

        run_id = (
            f"{model_id}__{condition_id}"
        )

        recomputed = {
            "run_id": run_id,
            "model_id": model_id,
            "condition_id": condition_id,
            "perturbed_feature_array_sha256": (
                condition_hash
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
        }

        recomputed_results[
            run_id
        ] = recomputed

        saved = saved_results[
            run_id
        ]

        if (
            saved[
                "perturbed_feature_array_sha256"
            ]
            != condition_hash
        ):
            raise RuntimeError(
                f"Perturbation hash mismatch: {run_id}"
            )

        if (
            saved[
                "confusion_matrix"
            ]
            != recomputed[
                "confusion_matrix"
            ]
        ):
            raise RuntimeError(
                f"Confusion matrix mismatch: {run_id}"
            )

        if (
            saved[
                "raw_weighted_confusion_matrix"
            ]
            != recomputed[
                "raw_weighted_confusion_matrix"
            ]
        ):
            raise RuntimeError(
                "Weighted confusion matrix mismatch: "
                f"{run_id}"
            )

        scalar_pairs = {
            "fingerprint_macro_f1": (
                recomputed[
                    "fingerprint_macro_f1"
                ]
            ),
            "fingerprint_accuracy": (
                recomputed[
                    "fingerprint_accuracy"
                ]
            ),
            "gafgyt_fnr": (
                recomputed[
                    "gafgyt_fnr"
                ]
            ),
            "mirai_fnr": (
                recomputed[
                    "mirai_fnr"
                ]
            ),
            "raw_weighted_macro_f1": (
                recomputed[
                    "raw_weighted_macro_f1"
                ]
            ),
            "raw_weighted_accuracy": (
                recomputed[
                    "raw_weighted_accuracy"
                ]
            ),
        }

        for metric_name, observed in (
            scalar_pairs.items()
        ):
            assert_close(
                observed,
                saved[metric_name],
                f"{run_id}:{metric_name}",
            )

        for vector_name in (
            "fingerprint_class_precision",
            "fingerprint_class_recall",
            "fingerprint_class_f1",
            "raw_weighted_class_recall",
        ):
            if not np.allclose(
                np.asarray(
                    recomputed[vector_name],
                    dtype=np.float64,
                ),
                np.asarray(
                    saved[vector_name],
                    dtype=np.float64,
                ),
                rtol=0.0,
                atol=FLOAT_ATOL,
            ):
                raise RuntimeError(
                    f"{run_id}:{vector_name} mismatch."
                )

        metrics_path = (
            RESULTS_ROOT
            / model_id
            / condition_id
            / "metrics.json"
        )

        if not metrics_path.is_file():
            raise FileNotFoundError(
                metrics_path
            )

        metrics_json = read_json(
            metrics_path
        )

        if (
            metrics_json.get(
                "run_id"
            )
            != run_id
            or metrics_json.get(
                "all_checks_passed"
            )
            is not True
        ):
            raise RuntimeError(
                "Per-run metrics identity mismatch: "
                f"{run_id}"
            )

        aggregate = aggregate_lookup[
            run_id
        ]

        assert_close(
            float(
                aggregate[
                    "fingerprint_macro_f1"
                ]
            ),
            saved[
                "fingerprint_macro_f1"
            ],
            f"{run_id}:aggregate_macro_f1",
        )

        assert_close(
            float(
                aggregate[
                    "raw_weighted_macro_f1"
                ]
            ),
            saved[
                "raw_weighted_macro_f1"
            ],
            f"{run_id}:aggregate_weighted_macro_f1",
        )

        print(
            f"[{condition_index}/9] "
            f"{condition_id:<28} | "
            f"{model_id:<29} | "
            "hash=True | confusion=True | "
            "metrics=True",
            flush=True,
        )

clean_by_model = {
    model_id: recomputed_results[
        f"{model_id}__clean"
    ]
    for model_id in (
        "tinyml_mlp_B0",
        "tinyml_mlp_P50_QAT",
        "hist_gradient_boosting_B0",
    )
}

derived_checks: dict[
    str,
    bool,
] = {}

for run_id, recomputed in (
    recomputed_results.items()
):
    saved = saved_results[
        run_id
    ]

    clean = clean_by_model[
        recomputed["model_id"]
    ]

    macro_drop = (
        clean[
            "fingerprint_macro_f1"
        ]
        - recomputed[
            "fingerprint_macro_f1"
        ]
    )

    weighted_drop = (
        clean[
            "raw_weighted_macro_f1"
        ]
        - recomputed[
            "raw_weighted_macro_f1"
        ]
    )

    retention = (
        recomputed[
            "fingerprint_macro_f1"
        ]
        / clean[
            "fingerprint_macro_f1"
        ]
    )

    severe_fragility = bool(
        recomputed[
            "fingerprint_macro_f1"
        ]
        < 0.85
        or recomputed[
            "gafgyt_fnr"
        ]
        > 0.40
        or recomputed[
            "mirai_fnr"
        ]
        > 0.40
    )

    material_sensitivity = bool(
        macro_drop > 0.05
    )

    assert_close(
        macro_drop,
        saved[
            "macro_f1_drop_from_model_clean"
        ],
        f"{run_id}:macro_drop",
    )

    assert_close(
        weighted_drop,
        saved[
            "weighted_macro_f1_drop_from_model_clean"
        ],
        f"{run_id}:weighted_drop",
    )

    assert_close(
        retention,
        saved[
            "macro_f1_retention_ratio"
        ],
        f"{run_id}:retention",
    )

    if (
        bool(
            saved[
                "severe_fragility"
            ]
        )
        != severe_fragility
    ):
        raise RuntimeError(
            f"{run_id}:severe_fragility mismatch."
        )

    if (
        bool(
            saved[
                "material_sensitivity"
            ]
        )
        != material_sensitivity
    ):
        raise RuntimeError(
            f"{run_id}:material_sensitivity mismatch."
        )

    aggregate = aggregate_lookup[
        run_id
    ]

    if (
        parse_bool(
            aggregate[
                "severe_fragility"
            ]
        )
        != severe_fragility
        or parse_bool(
            aggregate[
                "material_sensitivity"
            ]
        )
        != material_sensitivity
    ):
        raise RuntimeError(
            f"{run_id}:aggregate flag mismatch."
        )

    derived_checks[
        run_id
    ] = True

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
        saved_results[run_id]
        for run_id in sorted(
            saved_results.keys()
        )
        if saved_results[
            run_id
        ][
            "model_id"
        ]
        == model_id
    ]

    clean = next(
        row
        for row in model_rows
        if row[
            "condition_id"
        ]
        == "clean"
    )

    worst = min(
        model_rows,
        key=lambda row: row[
            "fingerprint_macro_f1"
        ],
    )

    recomputed_summary = {
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

    saved_summary = summary[
        "model_summaries"
    ][model_id]

    for key in (
        "clean_fingerprint_macro_f1",
        "clean_raw_weighted_macro_f1",
        "worst_fingerprint_macro_f1",
        "maximum_macro_f1_drop",
    ):
        assert_close(
            recomputed_summary[key],
            saved_summary[key],
            f"{model_id}:summary:{key}",
        )

    for key in (
        "worst_condition",
        "severe_fragility_condition_count",
        "material_sensitivity_condition_count",
    ):
        if (
            recomputed_summary[key]
            != saved_summary[key]
        ):
            raise RuntimeError(
                f"{model_id}:summary:{key} mismatch."
            )

    model_summaries[
        model_id
    ] = recomputed_summary

severe_count = sum(
    1
    for row in saved_results.values()
    if row[
        "severe_fragility"
    ]
)

material_count = sum(
    1
    for row in saved_results.values()
    if row[
        "material_sensitivity"
    ]
)

if severe_count != summary[
    "severe_fragility_evaluation_count"
]:
    raise RuntimeError(
        "Severe fragility summary count mismatch."
    )

if material_count != summary[
    "material_sensitivity_evaluation_count"
]:
    raise RuntimeError(
        "Material sensitivity summary count mismatch."
    )

if condition_hashes != summary[
    "condition_perturbation_sha256"
]:
    raise RuntimeError(
        "Summary condition-hash mapping mismatch."
    )

verification_checks = {
    "entry_checks_passed": all(
        entry_checks.values()
    ),
    "all_27_runs_recomputed": (
        len(recomputed_results) == 27
    ),
    "all_9_perturbation_hashes_verified": (
        len(condition_hashes) == 9
    ),
    "all_confusion_matrices_exact": True,
    "all_weighted_confusion_matrices_exact": (
        True
    ),
    "all_scalar_metrics_verified": True,
    "all_vector_metrics_verified": True,
    "all_derived_metrics_verified": all(
        derived_checks.values()
    ),
    "all_per_run_metrics_files_verified": (
        True
    ),
    "aggregate_csv_verified": True,
    "summary_verified": True,
    "model_retraining_false": (
        execution_report.get(
            "model_retraining_performed"
        )
        is False
    ),
    "test_selection_false": (
        execution_report.get(
            "test_used_for_model_selection"
        )
        is False
    ),
    "final_model_unchanged": (
        execution_report.get(
            "final_model_changed"
        )
        is False
    ),
}

failed_verification_checks = [
    name
    for name, passed
    in verification_checks.items()
    if not passed
]

if failed_verification_checks:
    raise RuntimeError(
        "Phase 11 independent verification failed: "
        + ", ".join(
            failed_verification_checks
        )
    )

elapsed_seconds = (
    time.perf_counter()
    - verification_start_time
)

verification = {
    "status": "passed",
    "phase": 11,
    "artifact_name": (
        "input_robustness_independent_verification"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verification_started_at_utc": (
        verification_started_at
    ),
    "verified_at_utc": utc_now(),
    "elapsed_seconds": float(
        elapsed_seconds
    ),
    "independent_neural_batch_size": (
        INDEPENDENT_NEURAL_BATCH_SIZE
    ),
    "model_count": 3,
    "condition_count": 9,
    "verified_evaluation_count": 27,
    "condition_perturbation_sha256": (
        condition_hashes
    ),
    "model_summaries": (
        model_summaries
    ),
    "severe_fragility_evaluation_count": (
        severe_count
    ),
    "material_sensitivity_evaluation_count": (
        material_count
    ),
    "entry_checks": (
        entry_checks
    ),
    "verification_checks": (
        verification_checks
    ),
    "model_inference_repeated": True,
    "model_retraining_performed": False,
    "test_used_for_model_selection": (
        False
    ),
    "final_model_changed": False,
    "ready_to_lock_phase11": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_VERIFICATION,
    verification,
)

lock = {
    "status": "locked",
    "phase": 11,
    "artifact_name": (
        "input_robustness_analysis"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "protocol": (
        file_record(
            PROTOCOL_PATH
        )
    ),
    "protocol_lock": (
        file_record(
            PROTOCOL_LOCK_PATH
        )
    ),
    "run_matrix": (
        file_record(
            RUN_MATRIX_PATH
        )
    ),
    "execution_report": (
        file_record(
            EXECUTION_REPORT_PATH
        )
    ),
    "completion": (
        file_record(
            COMPLETION_PATH
        )
    ),
    "aggregate_results": (
        file_record(
            AGGREGATE_CSV_PATH
        )
    ),
    "summary": (
        file_record(
            SUMMARY_PATH
        )
    ),
    "detailed_results": (
        file_record(
            DETAILED_PATH
        )
    ),
    "independent_verification": (
        file_record(
            OUTPUT_VERIFICATION
        )
    ),
    "model_count": 3,
    "condition_count": 9,
    "locked_evaluation_count": 27,
    "severe_fragility_evaluation_count": (
        severe_count
    ),
    "material_sensitivity_evaluation_count": (
        material_count
    ),
    "model_summaries": (
        model_summaries
    ),
    "model_retraining_performed": False,
    "test_used_for_model_selection": (
        False
    ),
    "final_model_changed": False,
    "phase11_locked": True,
    "ready_for_figures_and_reporting": (
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
    "phase": 11,
    "artifact_name": (
        "input_robustness_analysis"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            PROTOCOL_PATH
        ),
        file_record(
            PROTOCOL_LOCK_PATH
        ),
        file_record(
            RUN_MATRIX_PATH
        ),
        file_record(
            EXECUTION_REPORT_PATH
        ),
        file_record(
            COMPLETION_PATH
        ),
        file_record(
            X_TEST_PATH
        ),
        file_record(
            Y_TEST_PATH
        ),
        file_record(
            RAW_WEIGHT_PATH
        ),
        file_record(
            B0_checkpoint_path
        ),
        file_record(
            QAT_checkpoint_path
        ),
        file_record(
            HGB_MODEL_PATH
        ),
        file_record(
            MODEL_SOURCE_PATH
        ),
        file_record(
            QAT_ENGINE_PATH
        ),
        file_record(
            BUNDLE_SCALER_PATH
        ),
    ],
    "result_artifacts": [
        file_record(
            AGGREGATE_CSV_PATH
        ),
        file_record(
            SUMMARY_PATH
        ),
        file_record(
            DETAILED_PATH
        ),
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_VERIFICATION
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "locked_evaluation_count": 27,
    "model_retraining_performed": False,
    "test_used_for_model_selection": (
        False
    ),
    "final_model_changed": False,
    "phase11_locked": True,
    "ready_for_figures_and_reporting": (
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
print("PHASE 11 INPUT ROBUSTNESS VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Verified evaluations            : 27"
)
print(
    "Perturbation hashes verified    : 9"
)
print(
    "Confusion matrices exact        : True"
)
print(
    "Weighted confusion exact        : True"
)
print(
    "Scalar metrics verified         : True"
)
print(
    "Derived metrics verified        : True"
)
print(
    "Independent neural batch size   : 2048"
)
print(
    "Severe fragility evaluations    : "
    f"{severe_count}"
)
print(
    "Material sensitivity evaluations: "
    f"{material_count}"
)

for model_id, values in (
    model_summaries.items()
):
    print(
        f"{model_id:<31}: "
        f"clean={values['clean_fingerprint_macro_f1']:.9f} | "
        f"worst={values['worst_fingerprint_macro_f1']:.9f} | "
        f"condition={values['worst_condition']} | "
        f"max_drop={values['maximum_macro_f1_drop']:.9f}"
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
    "Phase 11 status                 : LOCKED"
)
print(
    "Ready for figures and reporting : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 11 ROBUSTNESS ANALYSIS VERIFIED AND LOCKED"
)
