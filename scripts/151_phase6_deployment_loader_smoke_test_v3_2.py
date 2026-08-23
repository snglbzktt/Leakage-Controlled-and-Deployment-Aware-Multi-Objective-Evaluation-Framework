from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import inspect
import io
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
from torch import nn


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PHASE6_PREFLIGHT_LOCK = (
    AUDIT
    / "phase6_deployment_benchmark_preflight_locked_v3_2.json"
)

PHASE6_MATRIX = (
    AUDIT
    / "phase6_deployment_benchmark_locked_matrix_v3_2.csv"
)

PHASE5_MASTER_MATRIX = (
    AUDIT
    / "phase5_compression_master_run_matrix_v3_2.csv"
)

B0_REGISTRY = (
    AUDIT
    / "phase5_B0_checkpoint_registry_v3_2.csv"
)

PRUNING_REGISTRY = (
    AUDIT
    / "phase5_physical_pruning_source_checkpoint_registry_v3_2.csv"
)

MODEL_SOURCE = (
    ROOT
    / "src"
    / "models"
    / "nbaiot_models.py"
)

PRUNING_ENGINE_PATH = (
    ROOT
    / "src"
    / "compression"
    / "phase5_physical_pruning_engine_v3_2.py"
)

QAT_ENGINE_PATH = (
    ROOT
    / "src"
    / "compression"
    / "phase5_qat_engine_v3_2.py"
)

PTQ_ENGINE_PATH = (
    ROOT
    / "src"
    / "compression"
    / "phase5_ptq_engine_v3_2.py"
)

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

X_TRAIN_PATH = FINAL_CACHE / "X_train.npy"

SCALER_NPZ = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "shared"
    / "preprocessing"
    / "phase5_train_only_standard_scaler_v3_2.npz"
)

OUTPUT_MATRIX = (
    AUDIT
    / "phase6_deployment_loader_smoke_matrix_v3_2.csv"
)

OUTPUT_REPORT = (
    AUDIT
    / "phase6_deployment_loader_smoke_test_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase6_deployment_loader_smoke_test_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase6_deployment_loader_smoke_test_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase6_deployment_benchmark_v3_2"

ARCHITECTURES = (
    "tinyml_mlp",
    "compact_dnn",
)

VARIANTS = (
    "B0",
    "FP32-FT",
    "P25-noFT",
    "P25-FP32-FT",
    "P50-noFT",
    "P50-FP32-FT",
    "DQ",
    "QAT",
    "P25-QAT",
    "P50-QAT",
    "PTQ",
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
        variant,
        seed,
    )
    for architecture in ARCHITECTURES
    for variant in VARIANTS
    for seed in SEEDS
}

EXPECTED_RUN_COUNT = 110
EXPECTED_FEATURE_COUNT = 115
EXPECTED_CLASS_COUNT = 3
EXPECTED_TRAIN_ROWS = 1_534_583

HIDDEN_DIMS = {
    "tinyml_mlp": [64, 32],
    "compact_dnn": [128, 64, 32],
}

PRUNING_RATIO = {
    "P25-noFT": 0.25,
    "P25-FP32-FT": 0.25,
    "P25-QAT": 0.25,
    "P50-noFT": 0.50,
    "P50-FP32-FT": 0.50,
    "P50-QAT": 0.50,
}

PRUNING_SOURCE_VARIANT = {
    "P25-noFT": "P25-noFT",
    "P25-FP32-FT": "P25-noFT",
    "P25-QAT": "P25-noFT",
    "P50-noFT": "P50-noFT",
    "P50-FP32-FT": "P50-noFT",
    "P50-QAT": "P50-noFT",
}

FLOAT_VARIANTS = {
    "B0",
    "FP32-FT",
    "P25-noFT",
    "P25-FP32-FT",
    "P50-noFT",
    "P50-FP32-FT",
}

STATIC_INT8_VARIANTS = {
    "QAT",
    "P25-QAT",
    "P50-QAT",
    "PTQ",
}

SMOKE_BATCH_SIZES = (
    1,
    32,
)

TRAIN_INPUT_START_INDEX = 2026
CPU_THREADS = 1
INTEROP_THREADS = 1

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


def get_dynamic_quantizer():
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


def quantize_dynamic_model(
    float_model: nn.Module,
) -> nn.Module:
    model = get_dynamic_quantizer()(
        copy.deepcopy(
            float_model
        ).eval(),
        {
            nn.Linear,
        },
        dtype=torch.qint8,
    )

    model.eval()
    return model


def quantized_dynamic_linear_count(
    model: nn.Module,
) -> int:
    count = 0

    for module in model.modules():
        module_path = (
            type(module).__module__.lower()
        )

        class_name = (
            type(module).__name__.lower()
        )

        if (
            "quantized.dynamic" in module_path
            and class_name == "linear"
        ):
            count += 1

    return count


def static_quantized_linear_count(
    model: nn.Module,
) -> int:
    count = 0

    for module in model.modules():
        module_path = (
            type(module).__module__.lower()
        )

        class_name = (
            type(module).__name__.lower()
        )

        if (
            "quantized" in module_path
            and ".qat." not in module_path
            and "dynamic" not in module_path
            and class_name == "linear"
        ):
            count += 1

    return count


def serialized_state_size(
    model: nn.Module,
) -> int:
    buffer = io.BytesIO()

    torch.save(
        model.state_dict(),
        buffer,
    )

    return len(
        buffer.getvalue()
    )


def pruning_registry_row(
    pruning_lookup: dict[
        tuple[str, str, int],
        dict[str, str],
    ],
    architecture: str,
    variant: str,
    seed: int,
) -> dict[str, str]:
    source_variant = (
        PRUNING_SOURCE_VARIANT[
            variant
        ]
    )

    return pruning_lookup[
        (
            architecture,
            source_variant,
            seed,
        )
    ]


def build_base_or_pruned_model(
    architecture: str,
    variant: str,
    seed: int,
    model_module: ModuleType,
    pruning_engine_module: ModuleType,
    B0_lookup: dict[
        tuple[str, int],
        dict[str, str],
    ],
    pruning_lookup: dict[
        tuple[str, str, int],
        dict[str, str],
    ],
) -> tuple[
    nn.Module,
    str,
    Path,
    str,
]:
    B0_row = B0_lookup[
        (
            architecture,
            seed,
        )
    ]

    B0_checkpoint_path = Path(
        B0_row[
            "checkpoint_path"
        ]
    )

    B0_checkpoint_hash = sha256_file(
        B0_checkpoint_path
    )

    if (
        B0_checkpoint_hash
        != B0_row[
            "checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "B0 checkpoint hash mismatch."
        )

    B0_checkpoint = torch.load(
        B0_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_symbol = (
        B0_row["model_symbol"]
    )

    B0_model = instantiate_model(
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

    B0_model.eval()

    if variant not in PRUNING_RATIO:
        return (
            B0_model,
            model_symbol,
            B0_checkpoint_path,
            B0_checkpoint_hash,
        )

    source_row = pruning_registry_row(
        pruning_lookup,
        architecture,
        variant,
        seed,
    )

    (
        pruned_model,
        _,
    ) = (
        pruning_engine_module
        .physically_prune_mlp(
            B0_model,
            PRUNING_RATIO[
                variant
            ],
            source_checkpoint_sha256=(
                B0_checkpoint_hash
            ),
        )
    )

    pruned_model.eval()

    return (
        pruned_model,
        model_symbol,
        B0_checkpoint_path,
        B0_checkpoint_hash,
    )


def run_checkpoint_path(
    variant: str,
    run_directory: Path,
    metrics: dict[str, Any],
    architecture: str,
    seed: int,
    pruning_lookup: dict[
        tuple[str, str, int],
        dict[str, str],
    ],
) -> Path:
    if variant in {
        "B0",
        "FP32-FT",
        "P25-FP32-FT",
        "P50-FP32-FT",
    }:
        return (
            run_directory
            / "best_checkpoint.pt"
        )

    if variant in {
        "P25-noFT",
        "P50-noFT",
    }:
        row = pruning_registry_row(
            pruning_lookup,
            architecture,
            variant,
            seed,
        )

        return Path(
            row[
                "source_checkpoint_path"
            ]
        )

    if variant == "DQ":
        return (
            run_directory
            / "dq_checkpoint.pt"
        )

    if variant in {
        "QAT",
        "P25-QAT",
        "P50-QAT",
    }:
        return (
            run_directory
            / "best_int8_checkpoint.pt"
        )

    if variant == "PTQ":
        candidate = (
            metrics[
                "source_candidate"
            ][
                "INT8_checkpoint_path"
            ]
        )

        return Path(candidate)

    raise RuntimeError(
        f"Unsupported variant: {variant}"
    )


def load_deployment_model(
    architecture: str,
    variant: str,
    seed: int,
    run_directory: Path,
    metrics: dict[str, Any],
    model_module: ModuleType,
    pruning_engine_module: ModuleType,
    QAT_engine_module: ModuleType,
    PTQ_engine_module: ModuleType,
    B0_lookup: dict[
        tuple[str, int],
        dict[str, str],
    ],
    pruning_lookup: dict[
        tuple[str, str, int],
        dict[str, str],
    ],
    structure_tensor: torch.Tensor,
) -> tuple[
    nn.Module,
    Path,
    str,
    str,
]:
    checkpoint_path = run_checkpoint_path(
        variant,
        run_directory,
        metrics,
        architecture,
        seed,
        pruning_lookup,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint_hash = sha256_file(
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    source_model, _, _, _ = (
        build_base_or_pruned_model(
            architecture,
            variant,
            seed,
            model_module,
            pruning_engine_module,
            B0_lookup,
            pruning_lookup,
        )
    )

    if variant in FLOAT_VARIANTS:
        if variant in {
            "P25-noFT",
            "P50-noFT",
        }:
            state = checkpoint[
                "model_state_dict"
            ]
        else:
            state = checkpoint[
                "model_state_dict"
            ]

        source_model.load_state_dict(
            state,
            strict=True,
        )

        source_model.eval()

        return (
            source_model,
            checkpoint_path,
            checkpoint_hash,
            "float32",
        )

    if variant == "DQ":
        requested_engine = str(
            checkpoint.get(
                "quantization_engine",
                torch.backends.quantized.engine,
            )
        )

        if (
            requested_engine
            in torch.backends.quantized.supported_engines
        ):
            torch.backends.quantized.engine = (
                requested_engine
            )

        DQ_model = quantize_dynamic_model(
            source_model
        )

        DQ_model.load_state_dict(
            checkpoint[
                "quantized_state_dict"
            ],
            strict=True,
        )

        DQ_model.eval()

        return (
            DQ_model,
            checkpoint_path,
            checkpoint_hash,
            "dynamic_int8",
        )

    if variant in {
        "QAT",
        "P25-QAT",
        "P50-QAT",
    }:
        backend = str(
            checkpoint[
                "QAT_backend"
            ]
        )

        if (
            backend
            not in torch.backends.quantized.supported_engines
        ):
            raise RuntimeError(
                f"Unsupported QAT backend: {backend}"
            )

        torch.backends.quantized.engine = (
            backend
        )

        prepared_model = (
            QAT_engine_module
            .prepare_qat_model(
                source_model,
                backend,
            )
        )

        prepared_model.eval()

        with torch.inference_mode():
            prepared_model(
                structure_tensor
            )

        INT8_model = (
            QAT_engine_module
            .convert_qat_model(
                prepared_model
            )
        )

        INT8_model.load_state_dict(
            checkpoint[
                "INT8_model_state_dict"
            ],
            strict=True,
        )

        INT8_model.eval()

        return (
            INT8_model,
            checkpoint_path,
            checkpoint_hash,
            "static_int8",
        )

    if variant == "PTQ":
        backend = str(
            checkpoint[
                "PTQ_backend"
            ]
        )

        if (
            backend
            not in torch.backends.quantized.supported_engines
        ):
            raise RuntimeError(
                f"Unsupported PTQ backend: {backend}"
            )

        torch.backends.quantized.engine = (
            backend
        )

        prepared_model = (
            PTQ_engine_module
            .prepare_ptq_model(
                source_model,
                backend,
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
            checkpoint[
                "INT8_model_state_dict"
            ],
            strict=True,
        )

        INT8_model.eval()

        return (
            INT8_model,
            checkpoint_path,
            checkpoint_hash,
            "static_int8",
        )

    raise RuntimeError(
        f"Unsupported loader variant: {variant}"
    )


required_paths = (
    PHASE6_PREFLIGHT_LOCK,
    PHASE6_MATRIX,
    PHASE5_MASTER_MATRIX,
    B0_REGISTRY,
    PRUNING_REGISTRY,
    MODEL_SOURCE,
    PRUNING_ENGINE_PATH,
    QAT_ENGINE_PATH,
    PTQ_ENGINE_PATH,
    X_TRAIN_PATH,
    SCALER_NPZ,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_MATRIX,
    OUTPUT_REPORT,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 6 loader-smoke artifact "
            "already exists; refusing to "
            f"overwrite: {output_path}"
        )

preflight_lock = read_json(
    PHASE6_PREFLIGHT_LOCK
)

phase6_rows = read_csv(
    PHASE6_MATRIX
)

phase5_rows = read_csv(
    PHASE5_MASTER_MATRIX
)

B0_rows = read_csv(
    B0_REGISTRY
)

pruning_rows = read_csv(
    PRUNING_REGISTRY
)

entry_checks = {
    "preflight_locked": (
        preflight_lock.get("status")
        == "locked"
        and preflight_lock.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
        and preflight_lock.get(
            "ready_for_loader_smoke_test"
        )
        is True
        and preflight_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "phase6_matrix_hash_matches": (
        preflight_lock.get(
            "locked_matrix_sha256"
        )
        == sha256_file(
            PHASE6_MATRIX
        )
    ),
    "phase6_matrix_110": (
        len(phase6_rows)
        == EXPECTED_RUN_COUNT
    ),
    "phase5_matrix_110": (
        len(phase5_rows)
        == EXPECTED_RUN_COUNT
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
        "Phase 6 loader-smoke entry gate "
        "failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

phase5_lookup = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    ): row
    for row in phase5_rows
}

phase6_keys = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    )
    for row in phase6_rows
}

if (
    set(
        phase5_lookup.keys()
    )
    != EXPECTED_KEYS
    or phase6_keys
    != EXPECTED_KEYS
):
    raise RuntimeError(
        "Phase 6 loader-smoke matrix mismatch."
    )

B0_lookup = {
    (
        row["architecture"],
        int(row["seed"]),
    ): row
    for row in B0_rows
}

pruning_lookup = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    ): row
    for row in pruning_rows
}

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

if (
    x_train.shape
    != (
        EXPECTED_TRAIN_ROWS,
        EXPECTED_FEATURE_COUNT,
    )
):
    raise RuntimeError(
        "Train cache shape mismatch."
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

structure_tensor = (
    batch_tensors[32]
)

model_module = load_module(
    MODEL_SOURCE,
    "phase6_loader_smoke_models",
)

pruning_engine_module = load_module(
    PRUNING_ENGINE_PATH,
    "phase6_loader_smoke_pruning",
)

QAT_engine_module = load_module(
    QAT_ENGINE_PATH,
    "phase6_loader_smoke_qat",
)

PTQ_engine_module = load_module(
    PTQ_ENGINE_PATH,
    "phase6_loader_smoke_ptq",
)

print("=" * 92)
print("PHASE 6 DEPLOYMENT LOADER SMOKE TEST")
print("=" * 92)
print(
    "Configurations                  : 110"
)
print(
    "Batch sizes                     : [1, 32]"
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
    "CPU threads                     : 1"
)
print(
    "Checkpoint deserialization      : Yes"
)
print(
    "Model state loading             : strict"
)
print()

result_rows: list[
    dict[str, Any]
] = []

ordered_rows = sorted(
    phase6_rows,
    key=lambda row: int(
        row[
            "measurement_order"
        ]
    ),
)

for index, phase6_row in enumerate(
    ordered_rows,
    start=1,
):
    architecture = (
        phase6_row["architecture"]
    )

    variant = (
        phase6_row["variant"]
    )

    seed = int(
        phase6_row["seed"]
    )

    phase5_row = phase5_lookup[
        (
            architecture,
            variant,
            seed,
        )
    ]

    metrics_path = Path(
        phase5_row["metrics_path"]
    )

    run_directory = Path(
        phase5_row["run_directory"]
    )

    if not metrics_path.exists():
        raise FileNotFoundError(
            metrics_path
        )

    metrics_hash = sha256_file(
        metrics_path
    )

    if (
        metrics_hash
        != phase5_row[
            "metrics_sha256"
        ]
        or metrics_hash
        != phase6_row[
            "metrics_sha256"
        ]
    ):
        raise RuntimeError(
            "Metrics hash mismatch for "
            f"{architecture} {variant} "
            f"seed {seed}."
        )

    metrics = read_json(
        metrics_path
    )

    started = time.perf_counter()

    (
        model,
        checkpoint_path,
        checkpoint_hash,
        loaded_representation,
    ) = load_deployment_model(
        architecture,
        variant,
        seed,
        run_directory,
        metrics,
        model_module,
        pruning_engine_module,
        QAT_engine_module,
        PTQ_engine_module,
        B0_lookup,
        pruning_lookup,
        structure_tensor,
    )

    load_elapsed = (
        time.perf_counter()
        - started
    )

    output_shapes: dict[
        int,
        list[int],
    ] = {}

    output_digests: dict[
        int,
        str,
    ] = {}

    batch_elapsed: dict[
        int,
        float,
    ] = {}

    with torch.inference_mode():
        for batch_size in (
            SMOKE_BATCH_SIZES
        ):
            tensor = batch_tensors[
                batch_size
            ]

            inference_started = (
                time.perf_counter()
            )

            logits = extract_logits(
                model(
                    tensor
                )
            )

            batch_elapsed[
                batch_size
            ] = (
                time.perf_counter()
                - inference_started
            )

            if tuple(
                logits.shape
            ) != (
                batch_size,
                EXPECTED_CLASS_COUNT,
            ):
                raise RuntimeError(
                    "Smoke output shape mismatch "
                    f"for {architecture} {variant} "
                    f"seed {seed}, batch "
                    f"{batch_size}: "
                    f"{tuple(logits.shape)}"
                )

            if not bool(
                torch.isfinite(
                    logits
                ).all().item()
            ):
                raise RuntimeError(
                    "Non-finite smoke output "
                    f"for {architecture} {variant} "
                    f"seed {seed}."
                )

            output_shapes[
                batch_size
            ] = list(
                logits.shape
            )

            output_digests[
                batch_size
            ] = hashlib.sha256(
                logits.detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()

    expected_representation = (
        phase6_row[
            "representation"
        ]
    )

    if (
        loaded_representation
        != expected_representation
    ):
        raise RuntimeError(
            "Representation mismatch for "
            f"{architecture} {variant} "
            f"seed {seed}: loaded="
            f"{loaded_representation}, "
            f"expected="
            f"{expected_representation}"
        )

    if variant == "DQ":
        quantized_linear_count = (
            quantized_dynamic_linear_count(
                model
            )
        )
    elif variant in STATIC_INT8_VARIANTS:
        quantized_linear_count = (
            static_quantized_linear_count(
                model
            )
        )
    else:
        quantized_linear_count = 0

    if (
        variant == "DQ"
        and quantized_linear_count <= 0
    ):
        raise RuntimeError(
            "No dynamic INT8 Linear layers "
            f"found for {architecture} "
            f"seed {seed}."
        )

    if (
        variant in STATIC_INT8_VARIANTS
        and quantized_linear_count <= 0
    ):
        raise RuntimeError(
            "No static INT8 Linear layers "
            f"found for {architecture} "
            f"{variant} seed {seed}."
        )

    result_rows.append(
        {
            "measurement_order": int(
                phase6_row[
                    "measurement_order"
                ]
            ),
            "architecture": architecture,
            "variant": variant,
            "seed": seed,
            "expected_representation": (
                expected_representation
            ),
            "loaded_representation": (
                loaded_representation
            ),
            "checkpoint_path": str(
                checkpoint_path
            ),
            "checkpoint_sha256": (
                checkpoint_hash
            ),
            "checkpoint_file_bytes": int(
                checkpoint_path.stat().st_size
            ),
            "serialized_state_dict_bytes": (
                serialized_state_size(
                    model
                )
            ),
            "model_class": (
                type(model).__name__
            ),
            "quantized_linear_count": (
                quantized_linear_count
            ),
            "load_elapsed_seconds": (
                load_elapsed
            ),
            "batch_1_output_shape": (
                json.dumps(
                    output_shapes[1],
                    separators=(
                        ",",
                        ":",
                    ),
                )
            ),
            "batch_1_output_sha256": (
                output_digests[1]
            ),
            "batch_1_smoke_seconds": (
                batch_elapsed[1]
            ),
            "batch_32_output_shape": (
                json.dumps(
                    output_shapes[32],
                    separators=(
                        ",",
                        ":",
                    ),
                )
            ),
            "batch_32_output_sha256": (
                output_digests[32]
            ),
            "batch_32_smoke_seconds": (
                batch_elapsed[32]
            ),
            "train_data_access": True,
            "validation_data_access": False,
            "test_data_access": False,
            "strict_state_loading": True,
            "all_checks_passed": True,
        }
    )

    print(
        f"[{index}/110] "
        f"{architecture} {variant} "
        f"seed={seed} | "
        f"representation="
        f"{loaded_representation} | "
        f"checkpoint=loaded | "
        "batch1=passed | "
        "batch32=passed | "
        f"qlinear="
        f"{quantized_linear_count} | "
        "all checks=True",
        flush=True,
    )

    del model

atomic_csv(
    OUTPUT_MATRIX,
    result_rows,
    [
        "measurement_order",
        "architecture",
        "variant",
        "seed",
        "expected_representation",
        "loaded_representation",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_file_bytes",
        "serialized_state_dict_bytes",
        "model_class",
        "quantized_linear_count",
        "load_elapsed_seconds",
        "batch_1_output_shape",
        "batch_1_output_sha256",
        "batch_1_smoke_seconds",
        "batch_32_output_shape",
        "batch_32_output_sha256",
        "batch_32_smoke_seconds",
        "train_data_access",
        "validation_data_access",
        "test_data_access",
        "strict_state_loading",
        "all_checks_passed",
    ],
)

observed_keys = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    )
    for row in result_rows
}

global_checks = {
    "run_count_110": (
        len(result_rows)
        == EXPECTED_RUN_COUNT
    ),
    "configuration_matrix_exact": (
        observed_keys
        == EXPECTED_KEYS
    ),
    "all_strict_loads_passed": (
        all(
            row[
                "strict_state_loading"
            ]
            is True
            for row in result_rows
        )
    ),
    "all_batch_1_passed": (
        all(
            row[
                "batch_1_output_shape"
            ]
            == "[1,3]"
            for row in result_rows
        )
    ),
    "all_batch_32_passed": (
        all(
            row[
                "batch_32_output_shape"
            ]
            == "[32,3]"
            for row in result_rows
        )
    ),
    "validation_access_zero": (
        all(
            row[
                "validation_data_access"
            ]
            is False
            for row in result_rows
        )
    ),
    "test_access_zero": (
        all(
            row[
                "test_data_access"
            ]
            is False
            for row in result_rows
        )
    ),
    "all_checks_passed": (
        all(
            row[
                "all_checks_passed"
            ]
            is True
            for row in result_rows
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
        "Phase 6 loader-smoke global "
        "checks failed: "
        + ", ".join(
            failed_global_checks
        )
    )

representation_counts: dict[
    str,
    int,
] = {}

for row in result_rows:
    key = row[
        "loaded_representation"
    ]

    representation_counts[key] = (
        representation_counts.get(
            key,
            0,
        )
        + 1
    )

report = {
    "status": "passed",
    "phase": 6,
    "artifact_name": (
        "deployment_loader_smoke_test"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "run_count": (
        EXPECTED_RUN_COUNT
    ),
    "group_count": 22,
    "smoke_batch_sizes": list(
        SMOKE_BATCH_SIZES
    ),
    "input_policy": {
        "source_split": "train_only",
        "start_index": (
            TRAIN_INPUT_START_INDEX
        ),
        "validation_data_access": False,
        "test_data_access": False,
    },
    "runtime_policy": {
        "cpu_threads": (
            CPU_THREADS
        ),
        "interop_threads": (
            INTEROP_THREADS
        ),
        "deterministic_algorithms": (
            True
        ),
        "strict_state_loading": (
            True
        ),
    },
    "representation_counts": (
        representation_counts
    ),
    "entry_checks": (
        entry_checks
    ),
    "global_checks": (
        global_checks
    ),
    "smoke_matrix": (
        file_record(
            OUTPUT_MATRIX
        )
    ),
    "source_artifacts": {
        "phase6_preflight_lock": (
            file_record(
                PHASE6_PREFLIGHT_LOCK
            )
        ),
        "phase6_locked_matrix": (
            file_record(
                PHASE6_MATRIX
            )
        ),
        "phase5_master_matrix": (
            file_record(
                PHASE5_MASTER_MATRIX
            )
        ),
        "B0_registry": (
            file_record(
                B0_REGISTRY
            )
        ),
        "pruning_registry": (
            file_record(
                PRUNING_REGISTRY
            )
        ),
        "model_source": (
            file_record(
                MODEL_SOURCE
            )
        ),
        "pruning_engine": (
            file_record(
                PRUNING_ENGINE_PATH
            )
        ),
        "QAT_engine": (
            file_record(
                QAT_ENGINE_PATH
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
    },
    "ready_for_isolated_deployment_benchmark": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_REPORT,
    report,
)

lock = {
    "status": "locked",
    "phase": 6,
    "artifact_name": (
        "deployment_loader_smoke_test"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "run_count": (
        EXPECTED_RUN_COUNT
    ),
    "group_count": 22,
    "smoke_matrix": str(
        OUTPUT_MATRIX
    ),
    "smoke_matrix_sha256": (
        sha256_file(
            OUTPUT_MATRIX
        )
    ),
    "report": str(
        OUTPUT_REPORT
    ),
    "report_sha256": (
        sha256_file(
            OUTPUT_REPORT
        )
    ),
    "validation_data_access": False,
    "test_data_access": False,
    "ready_for_isolated_deployment_benchmark": (
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
    "phase": 6,
    "artifact_name": (
        "deployment_loader_smoke_test"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            PHASE6_PREFLIGHT_LOCK
        ),
        file_record(
            PHASE6_MATRIX
        ),
        file_record(
            PHASE5_MASTER_MATRIX
        ),
        file_record(
            B0_REGISTRY
        ),
        file_record(
            PRUNING_REGISTRY
        ),
        file_record(
            MODEL_SOURCE
        ),
        file_record(
            PRUNING_ENGINE_PATH
        ),
        file_record(
            QAT_ENGINE_PATH
        ),
        file_record(
            PTQ_ENGINE_PATH
        ),
        file_record(
            SCALER_NPZ
        ),
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_MATRIX
        ),
        file_record(
            OUTPUT_REPORT
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "run_count": (
        EXPECTED_RUN_COUNT
    ),
    "group_count": 22,
    "validation_data_access": False,
    "test_data_access": False,
    "ready_for_isolated_deployment_benchmark": (
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
print("PHASE 6 DEPLOYMENT LOADER SMOKE SUMMARY")
print("=" * 92)
print(
    "Configurations loaded           : 110"
)
print(
    "Float32 runs                    : "
    f"{representation_counts.get('float32', 0)}"
)
print(
    "Dynamic INT8 runs               : "
    f"{representation_counts.get('dynamic_int8', 0)}"
)
print(
    "Static INT8 runs                : "
    f"{representation_counts.get('static_int8', 0)}"
)
print(
    "Batch 1 inference               : PASSED"
)
print(
    "Batch 32 inference              : PASSED"
)
print(
    "Strict checkpoint loading       : PASSED"
)
print(
    "Validation data access          : False"
)
print(
    "Test data access                : False"
)
print(
    "Loader smoke status             : LOCKED"
)
print(
    "Ready for isolated benchmark    : True"
)
print(
    "Matrix                          : "
    f"{OUTPUT_MATRIX}"
)
print(
    "Report                          : "
    f"{OUTPUT_REPORT}"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 6 DEPLOYMENT LOADERS VERIFIED AND LOCKED"
)
