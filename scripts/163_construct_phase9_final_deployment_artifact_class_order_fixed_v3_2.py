from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import inspect
import io
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

B0_REGISTRY_PATH = (
    AUDIT
    / "phase5_B0_checkpoint_registry_v3_2.csv"
)

X_TRAIN_PATH = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
    / "X_train.npy"
)

SOURCE_MODEL_PATH = (
    ROOT
    / "src"
    / "models"
    / "nbaiot_models.py"
)

SOURCE_PRUNING_ENGINE_PATH = (
    ROOT
    / "src"
    / "compression"
    / "phase5_physical_pruning_engine_v3_2.py"
)

SOURCE_QAT_ENGINE_PATH = (
    ROOT
    / "src"
    / "compression"
    / "phase5_qat_engine_v3_2.py"
)

SOURCE_SCALER_PATH = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "shared"
    / "preprocessing"
    / "phase5_train_only_standard_scaler_v3_2.npz"
)


PREPROCESSING_VERIFICATION_PATH = (
    AUDIT
    / "phase5_preprocessing_verification_v3_2.json"
)

PREPROCESSING_LOCK_PATH = (
    AUDIT
    / "phase5_preprocessing_locked_v3_2.json"
)

ARTIFACT_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "final_artifact"
    / "tinyml_mlp_P50_QAT_seed2026_v3_2"
)

STAGING_ROOT = ARTIFACT_ROOT.with_name(
    ARTIFACT_ROOT.name + ".building"
)

OUTPUT_REPORT = (
    AUDIT
    / "phase9_final_artifact_construction_v3_2.json"
)

OUTPUT_COMPLETION = (
    AUDIT
    / "phase9_final_artifact_constructed_v3_2.json"
)

PROTOCOL_VERSION = "phase9_final_artifact_v3_2"

SELECTED_ARCHITECTURE = "tinyml_mlp"
SELECTED_VARIANT = "P50-QAT"
SELECTED_REPRESENTATION = "static_int8"
SELECTED_SEED = 2026

EXPECTED_FEATURE_COUNT = 115
EXPECTED_CLASS_COUNT = 3
EXPECTED_TRAIN_ROWS = 1_534_583
PRUNED_HIDDEN_DIMS = [32, 16]

TRAIN_INPUT_START_INDEX = 2026
SMOKE_BATCH_SIZES = (1, 32)
CPU_THREADS = 1
INTEROP_THREADS = 1

BUNDLE_FILE_NAMES = {
    "checkpoint": "model_int8_checkpoint.pt",
    "model_source": "nbaiot_models.py",
    "pruning_engine": (
        "phase5_physical_pruning_engine_v3_2.py"
    ),
    "qat_engine": "phase5_qat_engine_v3_2.py",
    "scaler": (
        "train_only_standard_scaler_v3_2.npz"
    ),
    "contract": "inference_contract.json",
    "manifest": "artifact_manifest.json",
    "lock": "artifact_lock.json",
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
        "Windows kept the destination locked "
        f"after {WINDOWS_FILE_RETRY_COUNT} attempts: "
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

            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept the staging directory "
        f"locked: {path}"
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
    record = {
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
        PRUNED_HIDDEN_DIMS,
    )

    prepared_model = (
        qat_module.prepare_qat_model(
            float_model,
            backend,
        )
    )

    prepared_model.eval()

    with torch.inference_mode():
        prepared_model(structure_tensor)

    int8_model = qat_module.convert_qat_model(
        prepared_model
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    int8_model.load_state_dict(
        checkpoint[
            "INT8_model_state_dict"
        ],
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


LABEL_LIST_KEYS = {
    "class_names",
    "class_labels",
    "label_names",
    "target_names",
    "classes",
    "classes_",
    "class_order",
    "label_order",
    "target_classes",
}

LABEL_MAPPING_KEYS = {
    "class_to_idx",
    "label_to_idx",
    "class_mapping",
    "label_mapping",
    "idx_to_class",
    "idx_to_label",
}


def normalize_label_mapping(
    value: dict[Any, Any],
) -> list[str] | None:
    if len(value) != EXPECTED_CLASS_COUNT:
        return None

    if all(
        isinstance(key, str)
        and isinstance(index, (int, np.integer))
        for key, index in value.items()
    ):
        indexes = {
            int(index)
            for index in value.values()
        }

        if indexes == set(
            range(EXPECTED_CLASS_COUNT)
        ):
            return [
                next(
                    key
                    for key, index in value.items()
                    if int(index) == position
                )
                for position in range(
                    EXPECTED_CLASS_COUNT
                )
            ]

    converted: dict[int, str] = {}

    for key, label in value.items():
        try:
            index = int(key)
        except (
            TypeError,
            ValueError,
        ):
            converted = {}
            break

        if not isinstance(label, str):
            converted = {}
            break

        converted[index] = label

    if set(converted.keys()) == set(
        range(EXPECTED_CLASS_COUNT)
    ):
        return [
            converted[position]
            for position in range(
                EXPECTED_CLASS_COUNT
            )
        ]

    return None


def collect_label_candidates(
    value: Any,
    path: str,
    output: list[tuple[int, str, list[str]]],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            child_path = (
                f"{path}.{key_text}"
                if path
                else key_text
            )

            if key_lower in LABEL_LIST_KEYS:
                if (
                    isinstance(
                        child,
                        (list, tuple),
                    )
                    and len(child)
                    == EXPECTED_CLASS_COUNT
                    and all(
                        isinstance(item, str)
                        for item in child
                    )
                ):
                    output.append(
                        (
                            0,
                            child_path,
                            [
                                str(item)
                                for item in child
                            ],
                        )
                    )

            if key_lower in LABEL_MAPPING_KEYS:
                if isinstance(child, dict):
                    normalized = (
                        normalize_label_mapping(
                            child
                        )
                    )

                    if normalized is not None:
                        output.append(
                            (
                                0,
                                child_path,
                                normalized,
                            )
                        )

            collect_label_candidates(
                child,
                child_path,
                output,
            )

    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            collect_label_candidates(
                child,
                f"{path}[{index}]",
                output,
            )


def load_locked_class_labels(
) -> tuple[list[str], list[str]]:
    verification = read_json(
        PREPROCESSING_VERIFICATION_PATH
    )

    preprocessing_lock = read_json(
        PREPROCESSING_LOCK_PATH
    )

    source_scaler_sha256 = sha256_file(
        SOURCE_SCALER_PATH
    )

    checks = {
        "verification_passed": (
            verification.get("status")
            == "passed"
            and verification.get(
                "all_checks_passed"
            )
            is True
        ),
        "verification_protocol_matches": (
            verification.get(
                "protocol_version"
            )
            == (
                "phase5_fair_budget_"
                "compression_v3_2"
            )
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
        "lock_protocol_matches": (
            preprocessing_lock.get(
                "protocol_version"
            )
            == (
                "phase5_fair_budget_"
                "compression_v3_2"
            )
        ),
        "verification_hash_matches_lock": (
            preprocessing_lock.get(
                "verification_sha256"
            )
            == sha256_file(
                PREPROCESSING_VERIFICATION_PATH
            )
        ),
        "scaler_hash_matches_lock": (
            preprocessing_lock.get(
                "scaler_sha256"
            )
            == source_scaler_sha256
        ),
        "scaler_hash_matches_verification": (
            verification[
                "artifacts"
            ][
                "scaler"
            ][
                "sha256"
            ]
            == source_scaler_sha256
        ),
        "feature_count_matches": (
            int(
                verification[
                    "feature_count"
                ]
            )
            == EXPECTED_FEATURE_COUNT
            and int(
                preprocessing_lock[
                    "feature_count"
                ]
            )
            == EXPECTED_FEATURE_COUNT
        ),
        "validation_access_zero": (
            int(
                preprocessing_lock[
                    "validation_access_count"
                ]
            )
            == 0
        ),
        "test_access_zero": (
            int(
                preprocessing_lock[
                    "test_access_count"
                ]
            )
            == 0
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
            "Locked class-label source "
            "verification failed: "
            + ", ".join(
                failed_checks
            )
        )

    class_indices = list(
        verification[
            "class_labels"
        ]
    )

    class_names = list(
        verification[
            "class_names"
        ]
    )

    if class_indices != [0, 1, 2]:
        raise RuntimeError(
            "Locked class indices are not "
            "[0, 1, 2]."
        )

    if (
        len(class_names)
        != EXPECTED_CLASS_COUNT
        or not all(
            isinstance(name, str)
            and len(name) > 0
            for name in class_names
        )
        or len(set(class_names))
        != EXPECTED_CLASS_COUNT
    ):
        raise RuntimeError(
            "Locked class names are invalid."
        )

    expected_names = [
        "benign",
        "gafgyt",
        "mirai",
    ]

    if class_names != expected_names:
        raise RuntimeError(
            "Locked class-name order differs "
            "from the independently verified "
            f"expected order: {class_names}"
        )

    sources = [
        (
            str(
                PREPROCESSING_VERIFICATION_PATH
            )
            + ".class_labels"
        ),
        (
            str(
                PREPROCESSING_VERIFICATION_PATH
            )
            + ".class_names"
        ),
        (
            str(
                PREPROCESSING_LOCK_PATH
            )
            + ".verification_sha256"
        ),
    ]

    return class_names, sources


required_paths = (
    PROTOCOL_PATH,
    PREFLIGHT_LOCK_PATH,
    SOURCE_REGISTRY_PATH,
    B0_REGISTRY_PATH,
    X_TRAIN_PATH,
    SOURCE_MODEL_PATH,
    SOURCE_PRUNING_ENGINE_PATH,
    SOURCE_QAT_ENGINE_PATH,
    SOURCE_SCALER_PATH,
    PREPROCESSING_VERIFICATION_PATH,
    PREPROCESSING_LOCK_PATH,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_REPORT,
    OUTPUT_COMPLETION,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 9 construction audit "
            "artifact already exists; refusing "
            f"to overwrite: {output_path}"
        )

if ARTIFACT_ROOT.exists():
    raise FileExistsError(
        "Final artifact directory already "
        f"exists: {ARTIFACT_ROOT}"
    )

if STAGING_ROOT.exists():
    raise FileExistsError(
        "Staging directory already exists: "
        f"{STAGING_ROOT}"
    )

protocol = read_json(PROTOCOL_PATH)
preflight_lock = read_json(
    PREFLIGHT_LOCK_PATH
)

source_registry_rows = read_csv(
    SOURCE_REGISTRY_PATH
)

B0_rows = read_csv(
    B0_REGISTRY_PATH
)

entry_checks = {
    "protocol_locked": (
        protocol.get("status") == "locked"
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
    "protocol_hash_matches": (
        preflight_lock.get(
            "protocol_sha256"
        )
        == sha256_file(PROTOCOL_PATH)
    ),
    "source_registry_hash_matches": (
        preflight_lock.get(
            "source_registry_sha256"
        )
        == sha256_file(
            SOURCE_REGISTRY_PATH
        )
    ),
    "source_registry_row_count_1": (
        len(source_registry_rows) == 1
    ),
    "artifact_directory_absent": (
        not ARTIFACT_ROOT.exists()
    ),
    "staging_directory_absent": (
        not STAGING_ROOT.exists()
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
        "Phase 9 artifact construction "
        "entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

source_row = source_registry_rows[0]

if not (
    source_row["architecture"]
    == SELECTED_ARCHITECTURE
    and source_row["variant"]
    == SELECTED_VARIANT
    and source_row["representation"]
    == SELECTED_REPRESENTATION
    and int(source_row["seed"])
    == SELECTED_SEED
):
    raise RuntimeError(
        "Source-registry identity mismatch."
    )

source_checkpoint_path = Path(
    source_row["checkpoint_path"]
)

source_metrics_path = Path(
    source_row["metrics_path"]
)

for path in (
    source_checkpoint_path,
    source_metrics_path,
):
    if not path.exists():
        raise FileNotFoundError(path)

if (
    sha256_file(source_checkpoint_path)
    != source_row["checkpoint_sha256"]
):
    raise RuntimeError(
        "Source checkpoint hash mismatch."
    )

if (
    sha256_file(source_metrics_path)
    != source_row["metrics_sha256"]
):
    raise RuntimeError(
        "Source metrics hash mismatch."
    )

B0_matches = [
    row
    for row in B0_rows
    if row["architecture"]
    == SELECTED_ARCHITECTURE
    and int(row["seed"])
    == SELECTED_SEED
]

if len(B0_matches) != 1:
    raise RuntimeError(
        "B0 model-symbol lookup did not "
        "return exactly one row."
    )

model_symbol = B0_matches[0][
    "model_symbol"
]

source_checkpoint = torch.load(
    source_checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

class_labels, class_label_sources = (
    load_locked_class_labels()
)

torch.set_num_threads(CPU_THREADS)

try:
    torch.set_num_interop_threads(
        INTEROP_THREADS
    )
except RuntimeError:
    pass

torch.use_deterministic_algorithms(True)

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
    SOURCE_SCALER_PATH
) as values:
    mean64 = np.array(
        values["mean_float64"],
        copy=True,
    )

    scale64 = np.array(
        values["scale_float64"],
        copy=True,
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
print("PHASE 9 FINAL DEPLOYMENT ARTIFACT CONSTRUCTION - CLASS ORDER FIXED")
print("=" * 92)
print(
    "Model family                    : tinyml_mlp::P50-QAT"
)
print(
    "Canonical seed                  : 2026"
)
print(
    "Representation                  : static_int8"
)
print(
    "Model symbol                    : "
    f"{model_symbol}"
)
print(
    "Hidden dimensions               : [32, 16]"
)
print(
    "Class labels                    : "
    f"{class_labels}"
)
print(
    "Input source for smoke test      : train only"
)
print(
    "Validation data access          : False"
)
print(
    "Test data access                : False"
)
print(
    "Final artifact lock created     : False"
)
print()

construction_started = utc_now()
construction_succeeded = False

try:
    STAGING_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    copied_paths = {
        "checkpoint": (
            STAGING_ROOT
            / BUNDLE_FILE_NAMES["checkpoint"]
        ),
        "model_source": (
            STAGING_ROOT
            / BUNDLE_FILE_NAMES["model_source"]
        ),
        "pruning_engine": (
            STAGING_ROOT
            / BUNDLE_FILE_NAMES["pruning_engine"]
        ),
        "qat_engine": (
            STAGING_ROOT
            / BUNDLE_FILE_NAMES["qat_engine"]
        ),
        "scaler": (
            STAGING_ROOT
            / BUNDLE_FILE_NAMES["scaler"]
        ),
    }

    source_copy_map = {
        "checkpoint": source_checkpoint_path,
        "model_source": SOURCE_MODEL_PATH,
        "pruning_engine": (
            SOURCE_PRUNING_ENGINE_PATH
        ),
        "qat_engine": (
            SOURCE_QAT_ENGINE_PATH
        ),
        "scaler": SOURCE_SCALER_PATH,
    }

    for role, source_path in (
        source_copy_map.items()
    ):
        shutil.copyfile(
            source_path,
            copied_paths[role],
        )

        if (
            sha256_file(source_path)
            != sha256_file(
                copied_paths[role]
            )
        ):
            raise RuntimeError(
                "Byte-for-byte copy verification "
                f"failed for {role}."
            )

    inference_contract = {
        "status": "constructed_pending_independent_verification",
        "protocol_version": PROTOCOL_VERSION,
        "model_family": (
            "tinyml_mlp::P50-QAT"
        ),
        "architecture": (
            SELECTED_ARCHITECTURE
        ),
        "variant": SELECTED_VARIANT,
        "representation": (
            SELECTED_REPRESENTATION
        ),
        "canonical_seed": (
            SELECTED_SEED
        ),
        "model_symbol": model_symbol,
        "hidden_dimensions": (
            PRUNED_HIDDEN_DIMS
        ),
        "input": {
            "feature_count": (
                EXPECTED_FEATURE_COUNT
            ),
            "dtype": "float32",
            "shape": "[batch, 115]",
            "preprocessing": (
                BUNDLE_FILE_NAMES["scaler"]
            ),
        },
        "output": {
            "class_count": (
                EXPECTED_CLASS_COUNT
            ),
            "shape": "[batch, 3]",
            "semantics": "logits",
            "class_labels_in_output_index_order": (
                class_labels
            ),
            "class_label_source_paths": (
                class_label_sources
            ),
        },
        "quantization": {
            "method": (
                "eager static INT8 QAT"
            ),
            "backend": source_checkpoint[
                "QAT_backend"
            ],
            "state_dict_key": (
                "INT8_model_state_dict"
            ),
        },
        "runtime": {
            "device": "CPU",
            "cpu_threads": CPU_THREADS,
            "interop_threads": (
                INTEROP_THREADS
            ),
        },
        "data_access_during_artifact_smoke": {
            "train_data_access": True,
            "validation_data_access": False,
            "test_data_access": False,
        },
    }

    contract_path = (
        STAGING_ROOT
        / BUNDLE_FILE_NAMES["contract"]
    )

    atomic_json(
        contract_path,
        inference_contract,
    )

    source_model_module = load_module(
        SOURCE_MODEL_PATH,
        "phase9_source_models",
    )

    source_qat_module = load_module(
        SOURCE_QAT_ENGINE_PATH,
        "phase9_source_qat",
    )

    bundle_model_module = load_module(
        copied_paths["model_source"],
        "phase9_bundle_models",
    )

    bundle_qat_module = load_module(
        copied_paths["qat_engine"],
        "phase9_bundle_qat",
    )

    source_model = build_int8_model(
        source_model_module,
        source_qat_module,
        model_symbol,
        source_checkpoint[
            "QAT_backend"
        ],
        source_checkpoint_path,
        structure_tensor,
    )

    bundle_model = build_int8_model(
        bundle_model_module,
        bundle_qat_module,
        model_symbol,
        source_checkpoint[
            "QAT_backend"
        ],
        copied_paths["checkpoint"],
        structure_tensor,
    )

    output_checks: dict[str, Any] = {}

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

            if tuple(
                source_logits.shape
            ) != expected_shape:
                raise RuntimeError(
                    "Source output shape mismatch."
                )

            if tuple(
                bundle_logits.shape
            ) != expected_shape:
                raise RuntimeError(
                    "Bundle output shape mismatch."
                )

            if not bool(
                torch.isfinite(
                    bundle_logits
                ).all().item()
            ):
                raise RuntimeError(
                    "Non-finite bundle output."
                )

            if not torch.equal(
                source_logits,
                bundle_logits,
            ):
                raise RuntimeError(
                    "Copied artifact output does not "
                    "exactly match the source output."
                )

            output_checks[
                f"batch_{batch_size}"
            ] = {
                "shape": list(
                    bundle_logits.shape
                ),
                "source_output_sha256": (
                    output_digest(
                        source_logits
                    )
                ),
                "bundle_output_sha256": (
                    output_digest(
                        bundle_logits
                    )
                ),
                "exact_match": True,
                "finite": True,
            }

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
        source_checkpoint[
            "QAT_backend"
        ],
        copied_paths["checkpoint"],
        structure_tensor,
    )

    roundtrip_model.load_state_dict(
        roundtrip_state,
        strict=True,
    )

    roundtrip_model.eval()

    roundtrip_checks: dict[str, Any] = {}

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

            if not torch.equal(
                bundle_logits,
                roundtrip_logits,
            ):
                raise RuntimeError(
                    "Serialized state-dict roundtrip "
                    "output mismatch."
                )

            roundtrip_checks[
                f"batch_{batch_size}"
            ] = {
                "exact_match": True,
                "output_sha256": (
                    output_digest(
                        roundtrip_logits
                    )
                ),
            }

    manifest_path = (
        STAGING_ROOT
        / BUNDLE_FILE_NAMES["manifest"]
    )

    artifact_manifest = {
        "status": "constructed_pending_independent_verification",
        "phase": 9,
        "artifact_name": (
            "final_deployment_artifact"
        ),
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "constructed_at_utc": utc_now(),
        "model_family": (
            "tinyml_mlp::P50-QAT"
        ),
        "architecture": (
            SELECTED_ARCHITECTURE
        ),
        "variant": SELECTED_VARIANT,
        "representation": (
            SELECTED_REPRESENTATION
        ),
        "canonical_seed": (
            SELECTED_SEED
        ),
        "source_selection": {
            "method": (
                "fixed_canonical_seed_no_metric_ranking"
            ),
            "test_metric_ranking_used": False,
            "validation_metric_ranking_used": False,
            "deployment_metric_ranking_used": False,
        },
        "bundle_files": [
            file_record(
                copied_paths["checkpoint"],
                relative_to=STAGING_ROOT,
            ),
            file_record(
                copied_paths["model_source"],
                relative_to=STAGING_ROOT,
            ),
            file_record(
                copied_paths["pruning_engine"],
                relative_to=STAGING_ROOT,
            ),
            file_record(
                copied_paths["qat_engine"],
                relative_to=STAGING_ROOT,
            ),
            file_record(
                copied_paths["scaler"],
                relative_to=STAGING_ROOT,
            ),
            file_record(
                contract_path,
                relative_to=STAGING_ROOT,
            ),
        ],
        "source_artifacts": {
            **{
                role: file_record(path)
                for role, path
                in source_copy_map.items()
            },
            "class_label_verification": (
                file_record(
                    PREPROCESSING_VERIFICATION_PATH
                )
            ),
            "class_label_lock": (
                file_record(
                    PREPROCESSING_LOCK_PATH
                )
            ),
        },
        "smoke_verification": {
            "strict_state_dict_loading": True,
            "source_checkpoint_output_match": (
                output_checks
            ),
            "serialized_roundtrip": (
                roundtrip_checks
            ),
            "train_data_access": True,
            "validation_data_access": False,
            "test_data_access": False,
        },
        "artifact_lock_present": False,
        "ready_for_independent_verification": (
            True
        ),
        "all_checks_passed": True,
    }

    atomic_json(
        manifest_path,
        artifact_manifest,
    )

    staging_required_files = {
        BUNDLE_FILE_NAMES["checkpoint"],
        BUNDLE_FILE_NAMES["model_source"],
        BUNDLE_FILE_NAMES["pruning_engine"],
        BUNDLE_FILE_NAMES["qat_engine"],
        BUNDLE_FILE_NAMES["scaler"],
        BUNDLE_FILE_NAMES["contract"],
        BUNDLE_FILE_NAMES["manifest"],
    }

    observed_files = {
        path.name
        for path in STAGING_ROOT.iterdir()
        if path.is_file()
    }

    if observed_files != staging_required_files:
        raise RuntimeError(
            "Staging bundle file-set mismatch."
        )

    replace_with_retry(
        STAGING_ROOT,
        ARTIFACT_ROOT,
    )

    construction_succeeded = True

finally:
    if (
        not construction_succeeded
        and STAGING_ROOT.exists()
    ):
        remove_tree_with_retry(
            STAGING_ROOT
        )

if not ARTIFACT_ROOT.exists():
    raise RuntimeError(
        "Final artifact directory was not created."
    )

final_manifest_path = (
    ARTIFACT_ROOT
    / BUNDLE_FILE_NAMES["manifest"]
)

final_contract_path = (
    ARTIFACT_ROOT
    / BUNDLE_FILE_NAMES["contract"]
)

final_checkpoint_path = (
    ARTIFACT_ROOT
    / BUNDLE_FILE_NAMES["checkpoint"]
)

final_files = {
    path.name
    for path in ARTIFACT_ROOT.iterdir()
    if path.is_file()
}

expected_constructed_files = {
    BUNDLE_FILE_NAMES["checkpoint"],
    BUNDLE_FILE_NAMES["model_source"],
    BUNDLE_FILE_NAMES["pruning_engine"],
    BUNDLE_FILE_NAMES["qat_engine"],
    BUNDLE_FILE_NAMES["scaler"],
    BUNDLE_FILE_NAMES["contract"],
    BUNDLE_FILE_NAMES["manifest"],
}

construction_checks = {
    "artifact_directory_created": (
        ARTIFACT_ROOT.exists()
    ),
    "constructed_file_set_exact": (
        final_files
        == expected_constructed_files
    ),
    "artifact_lock_absent_pending_verification": (
        BUNDLE_FILE_NAMES["lock"]
        not in final_files
    ),
    "checkpoint_copy_hash_matches": (
        sha256_file(
            final_checkpoint_path
        )
        == sha256_file(
            source_checkpoint_path
        )
    ),
    "manifest_present": (
        final_manifest_path.exists()
    ),
    "contract_present": (
        final_contract_path.exists()
    ),
    "batch_1_source_bundle_match": (
        output_checks[
            "batch_1"
        ]["exact_match"]
    ),
    "batch_32_source_bundle_match": (
        output_checks[
            "batch_32"
        ]["exact_match"]
    ),
    "roundtrip_batch_1_match": (
        roundtrip_checks[
            "batch_1"
        ]["exact_match"]
    ),
    "roundtrip_batch_32_match": (
        roundtrip_checks[
            "batch_32"
        ]["exact_match"]
    ),
    "validation_access_zero": True,
    "test_access_zero": True,
    "single_checkpoint_not_locked": True,
}

failed_construction_checks = [
    name
    for name, passed
    in construction_checks.items()
    if not passed
]

if failed_construction_checks:
    raise RuntimeError(
        "Phase 9 artifact construction "
        "checks failed: "
        + ", ".join(
            failed_construction_checks
        )
    )

report = {
    "status": "constructed",
    "phase": 9,
    "artifact_name": (
        "final_deployment_artifact"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "construction_started_at_utc": (
        construction_started
    ),
    "constructed_at_utc": utc_now(),
    "artifact_root": str(
        ARTIFACT_ROOT
    ),
    "model_family": (
        "tinyml_mlp::P50-QAT"
    ),
    "architecture": (
        SELECTED_ARCHITECTURE
    ),
    "variant": SELECTED_VARIANT,
    "representation": (
        SELECTED_REPRESENTATION
    ),
    "canonical_seed": (
        SELECTED_SEED
    ),
    "model_symbol": model_symbol,
    "hidden_dimensions": (
        PRUNED_HIDDEN_DIMS
    ),
    "class_labels": class_labels,
    "class_label_source_paths": (
        class_label_sources
    ),
    "output_checks": output_checks,
    "roundtrip_checks": (
        roundtrip_checks
    ),
    "construction_checks": (
        construction_checks
    ),
    "artifact_files": [
        file_record(
            path,
            relative_to=ARTIFACT_ROOT,
        )
        for path in sorted(
            ARTIFACT_ROOT.iterdir()
        )
        if path.is_file()
    ],
    "artifact_lock_present": False,
    "single_checkpoint_locked": False,
    "ready_for_independent_verification": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_REPORT,
    report,
)

completion = {
    "status": "constructed",
    "phase": 9,
    "artifact_name": (
        "final_deployment_artifact"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "constructed_at_utc": utc_now(),
    "artifact_root": str(
        ARTIFACT_ROOT
    ),
    "artifact_manifest": str(
        final_manifest_path
    ),
    "artifact_manifest_sha256": (
        sha256_file(
            final_manifest_path
        )
    ),
    "inference_contract": str(
        final_contract_path
    ),
    "inference_contract_sha256": (
        sha256_file(
            final_contract_path
        )
    ),
    "checkpoint": str(
        final_checkpoint_path
    ),
    "checkpoint_sha256": (
        sha256_file(
            final_checkpoint_path
        )
    ),
    "construction_report": str(
        OUTPUT_REPORT
    ),
    "construction_report_sha256": (
        sha256_file(
            OUTPUT_REPORT
        )
    ),
    "artifact_lock_present": False,
    "single_checkpoint_locked": False,
    "ready_for_independent_verification": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_COMPLETION,
    completion,
)

print("=" * 92)
print("PHASE 9 FINAL DEPLOYMENT ARTIFACT CONSTRUCTION SUMMARY")
print("=" * 92)
print(
    "Artifact directory              : "
    f"{ARTIFACT_ROOT}"
)
print(
    "Model family                    : tinyml_mlp::P50-QAT"
)
print(
    "Canonical seed                  : 2026"
)
print(
    "Representation                  : static_int8"
)
print(
    "Class labels                    : "
    f"{class_labels}"
)
print(
    "Bundle files constructed        : 7"
)
print(
    "Checkpoint copied byte-for-byte : True"
)
print(
    "Source/bundle batch 1 match     : True"
)
print(
    "Source/bundle batch 32 match    : True"
)
print(
    "State roundtrip batch 1 match   : True"
)
print(
    "State roundtrip batch 32 match  : True"
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
print(
    "Single checkpoint locked        : False"
)
print(
    "Ready for independent verify    : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 9 FINAL ARTIFACT CONSTRUCTED"
)
