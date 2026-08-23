from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import io
import json
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
from torch import nn


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "protocols"
    / "phase9_final_artifact_protocol_v3_2.json"
)

PREFLIGHT_LOCK_PATH = (
    AUDIT
    / "phase9_final_artifact_protocol_preflight_locked_v3_2.json"
)

SOURCE_REGISTRY_PATH = (
    AUDIT
    / "phase9_final_artifact_source_registry_v3_2.csv"
)

CONSTRUCTION_REPORT_PATH = (
    AUDIT
    / "phase9_final_artifact_construction_v3_2.json"
)

CONSTRUCTION_COMPLETION_PATH = (
    AUDIT
    / "phase9_final_artifact_constructed_v3_2.json"
)

PREPROCESSING_VERIFICATION_PATH = (
    AUDIT
    / "phase5_preprocessing_verification_v3_2.json"
)

PREPROCESSING_LOCK_PATH = (
    AUDIT
    / "phase5_preprocessing_locked_v3_2.json"
)

X_TRAIN_PATH = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
    / "X_train.npy"
)

ARTIFACT_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "final_artifact"
    / "tinyml_mlp_P50_QAT_seed2026_v3_2"
)

BUNDLE_CHECKPOINT_PATH = (
    ARTIFACT_ROOT
    / "model_int8_checkpoint.pt"
)

BUNDLE_MODEL_SOURCE_PATH = (
    ARTIFACT_ROOT
    / "nbaiot_models.py"
)

BUNDLE_PRUNING_ENGINE_PATH = (
    ARTIFACT_ROOT
    / "phase5_physical_pruning_engine_v3_2.py"
)

BUNDLE_QAT_ENGINE_PATH = (
    ARTIFACT_ROOT
    / "phase5_qat_engine_v3_2.py"
)

BUNDLE_SCALER_PATH = (
    ARTIFACT_ROOT
    / "train_only_standard_scaler_v3_2.npz"
)

BUNDLE_CONTRACT_PATH = (
    ARTIFACT_ROOT
    / "inference_contract.json"
)

BUNDLE_MANIFEST_PATH = (
    ARTIFACT_ROOT
    / "artifact_manifest.json"
)

BUNDLE_LOCK_PATH = (
    ARTIFACT_ROOT
    / "artifact_lock.json"
)

OUTPUT_VERIFICATION = (
    AUDIT
    / "phase9_final_artifact_verification_v3_2.json"
)

OUTPUT_FINAL_LOCK = (
    AUDIT
    / "phase9_final_deployment_artifact_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase9_final_deployment_artifact_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase9_final_artifact_v3_2"

SELECTED_MODEL_FAMILY = "tinyml_mlp::P50-QAT"
SELECTED_ARCHITECTURE = "tinyml_mlp"
SELECTED_VARIANT = "P50-QAT"
SELECTED_REPRESENTATION = "static_int8"
SELECTED_SEED = 2026

EXPECTED_FEATURE_COUNT = 115
EXPECTED_CLASS_COUNT = 3
EXPECTED_TRAIN_ROWS = 1_534_583
EXPECTED_CLASS_LABELS = [
    "benign",
    "gafgyt",
    "mirai",
]

TRAIN_INPUT_START_INDEX = 2026
SMOKE_BATCH_SIZES = (1, 32)
CPU_THREADS = 1
INTEROP_THREADS = 1

EXPECTED_PRELOCK_FILES = {
    "model_int8_checkpoint.pt",
    "nbaiot_models.py",
    "phase5_physical_pruning_engine_v3_2.py",
    "phase5_qat_engine_v3_2.py",
    "train_only_standard_scaler_v3_2.npz",
    "inference_contract.json",
    "artifact_manifest.json",
}

EXPECTED_LOCKED_FILES = {
    *EXPECTED_PRELOCK_FILES,
    "artifact_lock.json",
}

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

            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept the destination file locked "
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
        ),
        encoding="utf-8",
    )

    replace_with_retry(
        temporary,
        path,
    )


def file_record(
    path: Path,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }

    if relative_to is not None:
        record["relative_path"] = str(
            path.relative_to(relative_to)
        )

    return record


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
                "Unresolved model-constructor "
                f"parameter: {name}"
            )

    return kwargs


def instantiate_model(
    model_module: ModuleType,
    model_symbol: str,
    hidden_dims: list[int],
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
            "Non-finite value after scaler."
        )

    return transformed


def build_int8_model(
    model_module: ModuleType,
    qat_module: ModuleType,
    model_symbol: str,
    hidden_dims: list[int],
    backend: str,
    checkpoint_path: Path,
    structure_tensor: torch.Tensor,
) -> nn.Module:
    if (
        backend
        not in torch.backends.quantized.supported_engines
    ):
        raise RuntimeError(
            f"Unsupported quantized backend: {backend}"
        )

    torch.backends.quantized.engine = backend

    float_model = instantiate_model(
        model_module,
        model_symbol,
        hidden_dims,
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

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "INT8_model_state_dict"
    )

    if not isinstance(
        state_dict,
        dict,
    ):
        raise RuntimeError(
            "INT8_model_state_dict missing."
        )

    int8_model.load_state_dict(
        state_dict,
        strict=True,
    )

    int8_model.eval()
    return int8_model


def output_digest(
    tensor: torch.Tensor,
) -> str:
    return hashlib.sha256(
        tensor.detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()


required_paths = (
    PROTOCOL_PATH,
    PREFLIGHT_LOCK_PATH,
    SOURCE_REGISTRY_PATH,
    CONSTRUCTION_REPORT_PATH,
    CONSTRUCTION_COMPLETION_PATH,
    PREPROCESSING_VERIFICATION_PATH,
    PREPROCESSING_LOCK_PATH,
    X_TRAIN_PATH,
    ARTIFACT_ROOT,
    BUNDLE_CHECKPOINT_PATH,
    BUNDLE_MODEL_SOURCE_PATH,
    BUNDLE_PRUNING_ENGINE_PATH,
    BUNDLE_QAT_ENGINE_PATH,
    BUNDLE_SCALER_PATH,
    BUNDLE_CONTRACT_PATH,
    BUNDLE_MANIFEST_PATH,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_VERIFICATION,
    OUTPUT_FINAL_LOCK,
    OUTPUT_LOCK_MANIFEST,
    BUNDLE_LOCK_PATH,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 9 verification or lock "
            "artifact already exists; refusing "
            f"to overwrite: {output_path}"
        )

protocol = read_json(
    PROTOCOL_PATH
)

preflight_lock = read_json(
    PREFLIGHT_LOCK_PATH
)

construction_report = read_json(
    CONSTRUCTION_REPORT_PATH
)

construction_completion = read_json(
    CONSTRUCTION_COMPLETION_PATH
)

preprocessing_verification = read_json(
    PREPROCESSING_VERIFICATION_PATH
)

preprocessing_lock = read_json(
    PREPROCESSING_LOCK_PATH
)

contract = read_json(
    BUNDLE_CONTRACT_PATH
)

manifest = read_json(
    BUNDLE_MANIFEST_PATH
)

source_registry_rows = read_csv(
    SOURCE_REGISTRY_PATH
)

if len(source_registry_rows) != 1:
    raise RuntimeError(
        "Source registry must contain "
        "exactly one row."
    )

source_registry = source_registry_rows[0]

source_checkpoint_path = Path(
    source_registry["checkpoint_path"]
)

source_model_path = Path(
    protocol[
        "source_artifacts"
    ][
        "model_source"
    ]["path"]
)

source_qat_engine_path = Path(
    protocol[
        "source_artifacts"
    ][
        "QAT_engine"
    ]["path"]
)

source_scaler_path = Path(
    protocol[
        "source_artifacts"
    ][
        "scaler"
    ]["path"]
)

for path in (
    source_checkpoint_path,
    source_model_path,
    source_qat_engine_path,
    source_scaler_path,
):
    if not path.exists():
        raise FileNotFoundError(path)

observed_prelock_files = {
    path.name
    for path in ARTIFACT_ROOT.iterdir()
    if path.is_file()
}

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
    "preflight_locked": (
        preflight_lock.get("status")
        == "locked"
        and preflight_lock.get(
            "ready_for_artifact_construction"
        )
        is True
        and preflight_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "construction_report_completed": (
        construction_report.get(
            "status"
        )
        == "constructed"
        and construction_report.get(
            "ready_for_independent_verification"
        )
        is True
        and construction_report.get(
            "all_checks_passed"
        )
        is True
    ),
    "construction_completion_completed": (
        construction_completion.get(
            "status"
        )
        == "constructed"
        and construction_completion.get(
            "ready_for_independent_verification"
        )
        is True
        and construction_completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "protocol_hash_matches": (
        preflight_lock.get(
            "protocol_sha256"
        )
        == sha256_file(
            PROTOCOL_PATH
        )
    ),
    "source_registry_hash_matches": (
        preflight_lock.get(
            "source_registry_sha256"
        )
        == sha256_file(
            SOURCE_REGISTRY_PATH
        )
    ),
    "construction_report_hash_matches": (
        construction_completion.get(
            "construction_report_sha256"
        )
        == sha256_file(
            CONSTRUCTION_REPORT_PATH
        )
    ),
    "manifest_hash_matches_completion": (
        construction_completion.get(
            "artifact_manifest_sha256"
        )
        == sha256_file(
            BUNDLE_MANIFEST_PATH
        )
    ),
    "contract_hash_matches_completion": (
        construction_completion.get(
            "inference_contract_sha256"
        )
        == sha256_file(
            BUNDLE_CONTRACT_PATH
        )
    ),
    "checkpoint_hash_matches_completion": (
        construction_completion.get(
            "checkpoint_sha256"
        )
        == sha256_file(
            BUNDLE_CHECKPOINT_PATH
        )
    ),
    "prelock_file_set_exact": (
        observed_prelock_files
        == EXPECTED_PRELOCK_FILES
    ),
    "artifact_lock_absent": (
        not BUNDLE_LOCK_PATH.exists()
    ),
    "construction_final_lock_absent": (
        construction_report.get(
            "artifact_lock_present"
        )
        is False
        and construction_report.get(
            "single_checkpoint_locked"
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
        "Phase 9 final artifact verification "
        "entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

identity_checks = {
    "contract_identity": (
        contract.get("model_family")
        == SELECTED_MODEL_FAMILY
        and contract.get("architecture")
        == SELECTED_ARCHITECTURE
        and contract.get("variant")
        == SELECTED_VARIANT
        and contract.get("representation")
        == SELECTED_REPRESENTATION
        and int(
            contract.get(
                "canonical_seed"
            )
        )
        == SELECTED_SEED
    ),
    "manifest_identity": (
        manifest.get("model_family")
        == SELECTED_MODEL_FAMILY
        and manifest.get("architecture")
        == SELECTED_ARCHITECTURE
        and manifest.get("variant")
        == SELECTED_VARIANT
        and manifest.get("representation")
        == SELECTED_REPRESENTATION
        and int(
            manifest.get(
                "canonical_seed"
            )
        )
        == SELECTED_SEED
    ),
    "contract_input_shape": (
        int(
            contract[
                "input"
            ]["feature_count"]
        )
        == EXPECTED_FEATURE_COUNT
        and contract[
            "input"
        ]["dtype"]
        == "float32"
    ),
    "contract_output_shape": (
        int(
            contract[
                "output"
            ]["class_count"]
        )
        == EXPECTED_CLASS_COUNT
        and contract[
            "output"
        ][
            "class_labels_in_output_index_order"
        ]
        == EXPECTED_CLASS_LABELS
    ),
    "locked_class_order_matches": (
        preprocessing_verification.get(
            "class_labels"
        )
        == [0, 1, 2]
        and preprocessing_verification.get(
            "class_names"
        )
        == EXPECTED_CLASS_LABELS
        and preprocessing_lock.get(
            "verification_sha256"
        )
        == sha256_file(
            PREPROCESSING_VERIFICATION_PATH
        )
    ),
    "manifest_pending_verification": (
        manifest.get("status")
        == (
            "constructed_pending_"
            "independent_verification"
        )
        and manifest.get(
            "ready_for_independent_verification"
        )
        is True
        and manifest.get(
            "all_checks_passed"
        )
        is True
        and manifest.get(
            "artifact_lock_present"
        )
        is False
    ),
    "no_metric_seed_ranking": (
        manifest[
            "source_selection"
        ][
            "test_metric_ranking_used"
        ]
        is False
        and manifest[
            "source_selection"
        ][
            "validation_metric_ranking_used"
        ]
        is False
        and manifest[
            "source_selection"
        ][
            "deployment_metric_ranking_used"
        ]
        is False
    ),
}

failed_identity_checks = [
    name
    for name, passed
    in identity_checks.items()
    if not passed
]

if failed_identity_checks:
    raise RuntimeError(
        "Phase 9 artifact identity or "
        "contract verification failed: "
        + ", ".join(
            failed_identity_checks
        )
    )

manifest_bundle_records = {
    record["relative_path"]: record
    for record in manifest[
        "bundle_files"
    ]
}

expected_manifest_record_names = {
    "model_int8_checkpoint.pt",
    "nbaiot_models.py",
    "phase5_physical_pruning_engine_v3_2.py",
    "phase5_qat_engine_v3_2.py",
    "train_only_standard_scaler_v3_2.npz",
    "inference_contract.json",
}

if (
    set(
        manifest_bundle_records.keys()
    )
    != expected_manifest_record_names
):
    raise RuntimeError(
        "Manifest bundle-file record set "
        "is incomplete."
    )

bundle_integrity_checks: dict[
    str,
    bool,
] = {}

for relative_name, record in (
    manifest_bundle_records.items()
):
    bundle_path = (
        ARTIFACT_ROOT
        / relative_name
    )

    bundle_integrity_checks[
        relative_name
    ] = (
        bundle_path.exists()
        and int(
            record["size_bytes"]
        )
        == int(
            bundle_path.stat().st_size
        )
        and record["sha256"]
        == sha256_file(bundle_path)
    )

source_copy_checks = {
    "checkpoint_copy_exact": (
        sha256_file(
            BUNDLE_CHECKPOINT_PATH
        )
        == sha256_file(
            source_checkpoint_path
        )
        == source_registry[
            "checkpoint_sha256"
        ]
    ),
    "model_source_copy_exact": (
        sha256_file(
            BUNDLE_MODEL_SOURCE_PATH
        )
        == sha256_file(
            source_model_path
        )
        == protocol[
            "source_artifacts"
        ][
            "model_source"
        ]["sha256"]
    ),
    "qat_engine_copy_exact": (
        sha256_file(
            BUNDLE_QAT_ENGINE_PATH
        )
        == sha256_file(
            source_qat_engine_path
        )
        == protocol[
            "source_artifacts"
        ][
            "QAT_engine"
        ]["sha256"]
    ),
    "scaler_copy_exact": (
        sha256_file(
            BUNDLE_SCALER_PATH
        )
        == sha256_file(
            source_scaler_path
        )
        == protocol[
            "source_artifacts"
        ][
            "scaler"
        ]["sha256"]
        == preprocessing_lock.get(
            "scaler_sha256"
        )
    ),
}

failed_bundle_integrity = [
    name
    for name, passed
    in bundle_integrity_checks.items()
    if not passed
]

failed_source_copy_checks = [
    name
    for name, passed
    in source_copy_checks.items()
    if not passed
]

if (
    failed_bundle_integrity
    or failed_source_copy_checks
):
    raise RuntimeError(
        "Phase 9 bundle integrity failed: "
        + ", ".join(
            failed_bundle_integrity
            + failed_source_copy_checks
        )
    )

model_symbol = contract[
    "model_symbol"
]

hidden_dimensions = [
    int(value)
    for value in contract[
        "hidden_dimensions"
    ]
]

backend = contract[
    "quantization"
]["backend"]

checkpoint = torch.load(
    BUNDLE_CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

checkpoint_checks = {
    "state_dict_present": (
        isinstance(
            checkpoint.get(
                "INT8_model_state_dict"
            ),
            dict,
        )
        and len(
            checkpoint[
                "INT8_model_state_dict"
            ]
        )
        > 0
    ),
    "backend_matches_contract": (
        checkpoint.get(
            "QAT_backend"
        )
        == backend
    ),
    "backend_supported": (
        backend
        in torch.backends.quantized.supported_engines
    ),
}

failed_checkpoint_checks = [
    name
    for name, passed
    in checkpoint_checks.items()
    if not passed
]

if failed_checkpoint_checks:
    raise RuntimeError(
        "Phase 9 checkpoint verification "
        "failed: "
        + ", ".join(
            failed_checkpoint_checks
        )
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

x_train = np.load(
    X_TRAIN_PATH,
    mmap_mode="r",
)

if x_train.shape != (
    EXPECTED_TRAIN_ROWS,
    EXPECTED_FEATURE_COUNT,
):
    raise RuntimeError(
        "Train cache shape mismatch."
    )

with np.load(
    BUNDLE_SCALER_PATH
) as scaler_values:
    mean64 = np.array(
        scaler_values[
            "mean_float64"
        ],
        copy=True,
    )

    scale64 = np.array(
        scaler_values[
            "scale_float64"
        ],
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

input_rows = transform_rows(
    x_train[
        TRAIN_INPUT_START_INDEX:
        TRAIN_INPUT_START_INDEX + 32
    ],
    mean64,
    scale64,
)

batch_tensors = {
    1: torch.from_numpy(
        input_rows[:1]
    ),
    32: torch.from_numpy(
        input_rows[:32]
    ),
}

structure_tensor = batch_tensors[32]

print("=" * 92)
print("PHASE 9 FINAL DEPLOYMENT ARTIFACT INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Model family                    : tinyml_mlp::P50-QAT"
)
print(
    "Canonical checkpoint seed       : 2026"
)
print(
    "Representation                  : static_int8"
)
print(
    "Class labels                    : ['benign', 'gafgyt', 'mirai']"
)
print(
    "Bundle file integrity           : verified"
)
print(
    "Input source                    : train only"
)
print(
    "Validation data access          : False"
)
print(
    "Test data access                : False"
)
print(
    "Artifact lock present           : False"
)
print()

source_model_module = load_module(
    source_model_path,
    "phase9_verify_source_models",
)

source_qat_module = load_module(
    source_qat_engine_path,
    "phase9_verify_source_qat",
)

bundle_model_module = load_module(
    BUNDLE_MODEL_SOURCE_PATH,
    "phase9_verify_bundle_models",
)

bundle_qat_module = load_module(
    BUNDLE_QAT_ENGINE_PATH,
    "phase9_verify_bundle_qat",
)

source_model = build_int8_model(
    source_model_module,
    source_qat_module,
    model_symbol,
    hidden_dimensions,
    backend,
    source_checkpoint_path,
    structure_tensor,
)

bundle_model = build_int8_model(
    bundle_model_module,
    bundle_qat_module,
    model_symbol,
    hidden_dimensions,
    backend,
    BUNDLE_CHECKPOINT_PATH,
    structure_tensor,
)

inference_checks: dict[
    str,
    dict[str, Any],
] = {}

with torch.inference_mode():
    for batch_size in SMOKE_BATCH_SIZES:
        source_logits = extract_logits(
            source_model(
                batch_tensors[
                    batch_size
                ]
            )
        )

        bundle_logits = extract_logits(
            bundle_model(
                batch_tensors[
                    batch_size
                ]
            )
        )

        expected_shape = (
            batch_size,
            EXPECTED_CLASS_COUNT,
        )

        source_digest = output_digest(
            source_logits
        )

        bundle_digest = output_digest(
            bundle_logits
        )

        construction_record = (
            construction_report[
                "output_checks"
            ][
                f"batch_{batch_size}"
            ]
        )

        checks = {
            "source_shape": (
                tuple(
                    source_logits.shape
                )
                == expected_shape
            ),
            "bundle_shape": (
                tuple(
                    bundle_logits.shape
                )
                == expected_shape
            ),
            "source_finite": (
                bool(
                    torch.isfinite(
                        source_logits
                    ).all().item()
                )
            ),
            "bundle_finite": (
                bool(
                    torch.isfinite(
                        bundle_logits
                    ).all().item()
                )
            ),
            "source_bundle_exact": (
                torch.equal(
                    source_logits,
                    bundle_logits,
                )
            ),
            "source_digest_matches_construction": (
                source_digest
                == construction_record[
                    "source_output_sha256"
                ]
            ),
            "bundle_digest_matches_construction": (
                bundle_digest
                == construction_record[
                    "bundle_output_sha256"
                ]
            ),
            "construction_exact_match_flag": (
                construction_record[
                    "exact_match"
                ]
                is True
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
                "Independent inference verification "
                f"failed for batch {batch_size}: "
                + ", ".join(
                    failed_checks
                )
            )

        inference_checks[
            f"batch_{batch_size}"
        ] = {
            "shape": list(
                bundle_logits.shape
            ),
            "source_output_sha256": (
                source_digest
            ),
            "bundle_output_sha256": (
                bundle_digest
            ),
            "source_bundle_exact": True,
            "finite": True,
            "all_checks_passed": True,
        }

        print(
            f"batch={batch_size:<2} | "
            "shape=verified | finite=True | "
            "source_bundle_exact=True | "
            "construction_digest_match=True",
            flush=True,
        )

state_buffer = io.BytesIO()

torch.save(
    bundle_model.state_dict(),
    state_buffer,
)

state_buffer.seek(0)

roundtrip_state = torch.load(
    state_buffer,
    map_location="cpu",
    weights_only=False,
)

roundtrip_model = build_int8_model(
    bundle_model_module,
    bundle_qat_module,
    model_symbol,
    hidden_dimensions,
    backend,
    BUNDLE_CHECKPOINT_PATH,
    structure_tensor,
)

roundtrip_model.load_state_dict(
    roundtrip_state,
    strict=True,
)

roundtrip_model.eval()

roundtrip_checks: dict[
    str,
    dict[str, Any],
] = {}

with torch.inference_mode():
    for batch_size in SMOKE_BATCH_SIZES:
        bundle_logits = extract_logits(
            bundle_model(
                batch_tensors[
                    batch_size
                ]
            )
        )

        roundtrip_logits = extract_logits(
            roundtrip_model(
                batch_tensors[
                    batch_size
                ]
            )
        )

        digest = output_digest(
            roundtrip_logits
        )

        construction_digest = (
            construction_report[
                "roundtrip_checks"
            ][
                f"batch_{batch_size}"
            ][
                "output_sha256"
            ]
        )

        checks = {
            "exact_match": (
                torch.equal(
                    bundle_logits,
                    roundtrip_logits,
                )
            ),
            "digest_matches_construction": (
                digest
                == construction_digest
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
                "State roundtrip verification "
                f"failed for batch {batch_size}: "
                + ", ".join(
                    failed_checks
                )
            )

        roundtrip_checks[
            f"batch_{batch_size}"
        ] = {
            "output_sha256": digest,
            "exact_match": True,
            "construction_digest_match": (
                True
            ),
            "all_checks_passed": True,
        }

verification_checks = {
    "entry_checks_passed": all(
        entry_checks.values()
    ),
    "identity_checks_passed": all(
        identity_checks.values()
    ),
    "bundle_integrity_passed": all(
        bundle_integrity_checks.values()
    ),
    "source_copy_checks_passed": all(
        source_copy_checks.values()
    ),
    "checkpoint_checks_passed": all(
        checkpoint_checks.values()
    ),
    "batch_1_inference_verified": (
        inference_checks[
            "batch_1"
        ][
            "all_checks_passed"
        ]
    ),
    "batch_32_inference_verified": (
        inference_checks[
            "batch_32"
        ][
            "all_checks_passed"
        ]
    ),
    "batch_1_roundtrip_verified": (
        roundtrip_checks[
            "batch_1"
        ][
            "all_checks_passed"
        ]
    ),
    "batch_32_roundtrip_verified": (
        roundtrip_checks[
            "batch_32"
        ][
            "all_checks_passed"
        ]
    ),
    "validation_access_zero": True,
    "test_access_zero": True,
    "artifact_lock_not_yet_written": (
        not BUNDLE_LOCK_PATH.exists()
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
        "Phase 9 final artifact independent "
        "verification failed: "
        + ", ".join(
            failed_verification_checks
        )
    )

verification = {
    "status": "passed",
    "phase": 9,
    "artifact_name": (
        "final_deployment_artifact_"
        "independent_verification"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "artifact_root": str(
        ARTIFACT_ROOT
    ),
    "model_family": (
        SELECTED_MODEL_FAMILY
    ),
    "architecture": (
        SELECTED_ARCHITECTURE
    ),
    "variant": (
        SELECTED_VARIANT
    ),
    "representation": (
        SELECTED_REPRESENTATION
    ),
    "canonical_seed": (
        SELECTED_SEED
    ),
    "class_labels": (
        EXPECTED_CLASS_LABELS
    ),
    "entry_checks": (
        entry_checks
    ),
    "identity_checks": (
        identity_checks
    ),
    "bundle_integrity_checks": (
        bundle_integrity_checks
    ),
    "source_copy_checks": (
        source_copy_checks
    ),
    "checkpoint_checks": (
        checkpoint_checks
    ),
    "inference_checks": (
        inference_checks
    ),
    "roundtrip_checks": (
        roundtrip_checks
    ),
    "verification_checks": (
        verification_checks
    ),
    "data_access": {
        "train_data_access": True,
        "validation_data_access": False,
        "test_data_access": False,
    },
    "model_inference_repeated": True,
    "model_inference_purpose": (
        "independent artifact integrity "
        "verification only"
    ),
    "artifact_lock_written": False,
    "single_checkpoint_locked": False,
    "ready_to_lock_final_artifact": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_VERIFICATION,
    verification,
)

prelock_inventory = [
    file_record(
        ARTIFACT_ROOT / name,
        relative_to=ARTIFACT_ROOT,
    )
    for name in sorted(
        EXPECTED_PRELOCK_FILES
    )
]

artifact_lock = {
    "status": "locked",
    "phase": 9,
    "artifact_name": (
        "final_deployment_artifact"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "model_family": (
        SELECTED_MODEL_FAMILY
    ),
    "architecture": (
        SELECTED_ARCHITECTURE
    ),
    "variant": (
        SELECTED_VARIANT
    ),
    "representation": (
        SELECTED_REPRESENTATION
    ),
    "canonical_seed": (
        SELECTED_SEED
    ),
    "class_labels_in_output_index_order": (
        EXPECTED_CLASS_LABELS
    ),
    "checkpoint_selection_method": (
        "fixed_canonical_seed_no_metric_ranking"
    ),
    "checkpoint_sha256": (
        sha256_file(
            BUNDLE_CHECKPOINT_PATH
        )
    ),
    "artifact_manifest_sha256": (
        sha256_file(
            BUNDLE_MANIFEST_PATH
        )
    ),
    "inference_contract_sha256": (
        sha256_file(
            BUNDLE_CONTRACT_PATH
        )
    ),
    "independent_verification_report": (
        str(
            OUTPUT_VERIFICATION
        )
    ),
    "independent_verification_report_sha256": (
        sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "immutable_prelock_bundle_files": (
        prelock_inventory
    ),
    "validation_data_access": False,
    "test_data_access": False,
    "source_bundle_outputs_exact": True,
    "serialization_roundtrip_exact": (
        True
    ),
    "single_checkpoint_locked": True,
    "final_deployment_artifact_locked": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    BUNDLE_LOCK_PATH,
    artifact_lock,
)

observed_locked_files = {
    path.name
    for path in ARTIFACT_ROOT.iterdir()
    if path.is_file()
}

postlock_checks = {
    "artifact_lock_created": (
        BUNDLE_LOCK_PATH.exists()
    ),
    "locked_file_set_exact": (
        observed_locked_files
        == EXPECTED_LOCKED_FILES
    ),
    "artifact_lock_status_locked": (
        read_json(
            BUNDLE_LOCK_PATH
        ).get(
            "status"
        )
        == "locked"
    ),
    "verification_hash_matches_lock": (
        read_json(
            BUNDLE_LOCK_PATH
        )[
            "independent_verification_report_sha256"
        ]
        == sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "checkpoint_hash_matches_lock": (
        read_json(
            BUNDLE_LOCK_PATH
        )[
            "checkpoint_sha256"
        ]
        == sha256_file(
            BUNDLE_CHECKPOINT_PATH
        )
    ),
}

failed_postlock_checks = [
    name
    for name, passed
    in postlock_checks.items()
    if not passed
]

if failed_postlock_checks:
    raise RuntimeError(
        "Phase 9 artifact post-lock "
        "verification failed: "
        + ", ".join(
            failed_postlock_checks
        )
    )

locked_inventory = [
    file_record(
        path,
        relative_to=ARTIFACT_ROOT,
    )
    for path in sorted(
        ARTIFACT_ROOT.iterdir()
    )
    if path.is_file()
]

final_lock = {
    "status": "locked",
    "phase": 9,
    "artifact_name": (
        "final_deployment_artifact"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "artifact_root": str(
        ARTIFACT_ROOT
    ),
    "model_family": (
        SELECTED_MODEL_FAMILY
    ),
    "architecture": (
        SELECTED_ARCHITECTURE
    ),
    "variant": (
        SELECTED_VARIANT
    ),
    "representation": (
        SELECTED_REPRESENTATION
    ),
    "canonical_seed": (
        SELECTED_SEED
    ),
    "class_labels_in_output_index_order": (
        EXPECTED_CLASS_LABELS
    ),
    "artifact_lock": (
        file_record(
            BUNDLE_LOCK_PATH
        )
    ),
    "independent_verification": (
        file_record(
            OUTPUT_VERIFICATION
        )
    ),
    "locked_bundle_inventory": (
        locked_inventory
    ),
    "source_checkpoint_sha256": (
        sha256_file(
            source_checkpoint_path
        )
    ),
    "bundled_checkpoint_sha256": (
        sha256_file(
            BUNDLE_CHECKPOINT_PATH
        )
    ),
    "source_bundle_checkpoint_exact": (
        True
    ),
    "source_bundle_outputs_exact": (
        True
    ),
    "serialization_roundtrip_exact": (
        True
    ),
    "validation_data_access": False,
    "test_data_access": False,
    "single_checkpoint_locked": True,
    "final_deployment_artifact_locked": (
        True
    ),
    "ready_for_release_archive": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_FINAL_LOCK,
    final_lock,
)

lock_manifest = {
    "status": "locked",
    "phase": 9,
    "artifact_name": (
        "final_deployment_artifact"
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
            PREFLIGHT_LOCK_PATH
        ),
        file_record(
            SOURCE_REGISTRY_PATH
        ),
        file_record(
            CONSTRUCTION_REPORT_PATH
        ),
        file_record(
            CONSTRUCTION_COMPLETION_PATH
        ),
        file_record(
            PREPROCESSING_VERIFICATION_PATH
        ),
        file_record(
            PREPROCESSING_LOCK_PATH
        ),
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_VERIFICATION
        ),
        file_record(
            BUNDLE_LOCK_PATH
        ),
        file_record(
            OUTPUT_FINAL_LOCK
        ),
    ],
    "locked_bundle_inventory": (
        locked_inventory
    ),
    "model_family": (
        SELECTED_MODEL_FAMILY
    ),
    "canonical_seed": (
        SELECTED_SEED
    ),
    "single_checkpoint_locked": True,
    "final_deployment_artifact_locked": (
        True
    ),
    "ready_for_release_archive": (
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
print("PHASE 9 FINAL DEPLOYMENT ARTIFACT VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Bundle files verified           : 7"
)
print(
    "Locked bundle files             : 8"
)
print(
    "Model family                    : tinyml_mlp::P50-QAT"
)
print(
    "Canonical checkpoint seed       : 2026"
)
print(
    "Representation                  : static_int8"
)
print(
    "Class labels                    : ['benign', 'gafgyt', 'mirai']"
)
print(
    "Checkpoint copy exact           : True"
)
print(
    "Source/bundle batch 1 exact     : True"
)
print(
    "Source/bundle batch 32 exact    : True"
)
print(
    "State roundtrip batch 1 exact   : True"
)
print(
    "State roundtrip batch 32 exact  : True"
)
print(
    "Validation data access          : False"
)
print(
    "Test data access                : False"
)
print(
    "Artifact lock present           : True"
)
print(
    "Single checkpoint locked        : True"
)
print(
    "Final artifact status           : LOCKED"
)
print(
    "Ready for release archive       : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 9 FINAL DEPLOYMENT ARTIFACT VERIFIED AND LOCKED"
)
