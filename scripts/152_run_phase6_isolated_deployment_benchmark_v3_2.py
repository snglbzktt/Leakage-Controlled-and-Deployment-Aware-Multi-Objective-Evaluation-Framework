from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PHASE6_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase6_deployment_benchmark_protocol_v3_2.json"
)

PHASE6_PREFLIGHT_LOCK = (
    AUDIT
    / "phase6_deployment_benchmark_preflight_locked_v3_2.json"
)

PHASE6_LOADER_LOCK = (
    AUDIT
    / "phase6_deployment_loader_smoke_test_locked_v3_2.json"
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

BENCHMARK_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "phase6_deployment_benchmark"
)

OUTPUT_ALL_RUNS = (
    AUDIT
    / "phase6_deployment_benchmark_all_runs_v3_2.csv"
)

OUTPUT_GROUP_SUMMARY = (
    AUDIT
    / "phase6_deployment_benchmark_group_summary_v3_2.csv"
)

OUTPUT_SUMMARY = (
    AUDIT
    / "phase6_deployment_benchmark_summary_v3_2.json"
)

OUTPUT_COMPLETION = (
    AUDIT
    / "phase6_deployment_benchmark_completed_v3_2.json"
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
EXPECTED_GROUP_COUNT = 22
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

BATCH_SIZES = (
    1,
    32,
)

WARMUP_ITERATIONS = {
    1: 50,
    32: 30,
}

MEASURED_ITERATIONS = {
    1: 500,
    32: 200,
}

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

            if (
                attempt
                == WINDOWS_FILE_RETRY_COUNT
            ):
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept an incomplete benchmark "
        "directory locked after "
        f"{WINDOWS_FILE_RETRY_COUNT} attempts: "
        f"{path}"
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
    return pruning_lookup[
        (
            architecture,
            PRUNING_SOURCE_VARIANT[
                variant
            ],
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
        return Path(
            metrics[
                "source_candidate"
            ][
                "INT8_checkpoint_path"
            ]
        )

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

    (
        source_model,
        _,
        _,
        _,
    ) = build_base_or_pruned_model(
        architecture,
        variant,
        seed,
        model_module,
        pruning_engine_module,
        B0_lookup,
    )

    if variant in FLOAT_VARIANTS:
        source_model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
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


def latency_statistics(
    latencies_ns: np.ndarray,
) -> dict[str, float]:
    values_ms = (
        latencies_ns.astype(
            np.float64
        )
        / 1_000_000.0
    )

    return {
        "mean_ms": float(
            np.mean(values_ms)
        ),
        "median_ms": float(
            np.median(values_ms)
        ),
        "std_ms": float(
            np.std(
                values_ms,
                ddof=0,
            )
        ),
        "p95_ms": float(
            np.percentile(
                values_ms,
                95,
            )
        ),
        "p99_ms": float(
            np.percentile(
                values_ms,
                99,
            )
        ),
        "minimum_ms": float(
            np.min(values_ms)
        ),
        "maximum_ms": float(
            np.max(values_ms)
        ),
    }


def run_output_directory(
    architecture: str,
    variant: str,
    seed: int,
) -> Path:
    safe_variant = (
        variant.replace(
            "-",
            "_",
        )
    )

    return (
        BENCHMARK_ROOT
        / architecture
        / safe_variant
        / f"seed_{seed}"
    )


def artifact_inventory(
    run_directory: Path,
) -> list[dict[str, Any]]:
    excluded = {
        "run_manifest.json",
        "run_status.json",
    }

    return [
        file_record(
            path,
            relative_to=run_directory,
        )
        for path in sorted(
            run_directory.iterdir()
        )
        if path.is_file()
        and path.name not in excluded
    ]


def worker(
    measurement_order: int,
) -> None:
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

    phase6_rows = read_csv(
        PHASE6_MATRIX
    )

    selected = [
        row
        for row in phase6_rows
        if int(
            row[
                "measurement_order"
            ]
        )
        == measurement_order
    ]

    if len(selected) != 1:
        raise RuntimeError(
            "Worker measurement-order lookup "
            "did not return exactly one row."
        )

    phase6_row = selected[0]

    architecture = (
        phase6_row[
            "architecture"
        ]
    )

    variant = phase6_row[
        "variant"
    ]

    seed = int(
        phase6_row[
            "seed"
        ]
    )

    phase5_rows = read_csv(
        PHASE5_MASTER_MATRIX
    )

    phase5_lookup = {
        (
            row["architecture"],
            row["variant"],
            int(row["seed"]),
        ): row
        for row in phase5_rows
    }

    phase5_row = phase5_lookup[
        (
            architecture,
            variant,
            seed,
        )
    ]

    metrics_path = Path(
        phase5_row[
            "metrics_path"
        ]
    )

    metrics_hash = sha256_file(
        metrics_path
    )

    if not (
        metrics_hash
        == phase5_row[
            "metrics_sha256"
        ]
        == phase6_row[
            "metrics_sha256"
        ]
    ):
        raise RuntimeError(
            "Metrics hash mismatch."
        )

    metrics = read_json(
        metrics_path
    )

    B0_rows = read_csv(
        B0_REGISTRY
    )

    pruning_rows = read_csv(
        PRUNING_REGISTRY
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
            values[
                "mean_float64"
            ],
            copy=True,
        )

        scale64 = np.array(
            values[
                "scale_float64"
            ],
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
        (
            "phase6_benchmark_models_"
            f"{measurement_order}"
        ),
    )

    pruning_engine_module = load_module(
        PRUNING_ENGINE_PATH,
        (
            "phase6_benchmark_pruning_"
            f"{measurement_order}"
        ),
    )

    QAT_engine_module = load_module(
        QAT_ENGINE_PATH,
        (
            "phase6_benchmark_qat_"
            f"{measurement_order}"
        ),
    )

    PTQ_engine_module = load_module(
        PTQ_ENGINE_PATH,
        (
            "phase6_benchmark_ptq_"
            f"{measurement_order}"
        ),
    )

    run_directory = run_output_directory(
        architecture,
        variant,
        seed,
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    status_path = (
        run_directory
        / "run_status.json"
    )

    atomic_json(
        status_path,
        {
            "status": "running",
            "stage": "model_loading",
            "measurement_order": (
                measurement_order
            ),
            "architecture": architecture,
            "variant": variant,
            "seed": seed,
            "validation_data_access": False,
            "test_data_access": False,
            "started_at_utc": utc_now(),
        },
    )

    process = psutil.Process(
        os.getpid()
    )

    gc.collect()

    RSS_before_model_load = int(
        process.memory_info().rss
    )

    load_started = (
        time.perf_counter_ns()
    )

    (
        model,
        checkpoint_path,
        checkpoint_hash,
        loaded_representation,
    ) = load_deployment_model(
        architecture,
        variant,
        seed,
        Path(
            phase5_row[
                "run_directory"
            ]
        ),
        metrics,
        model_module,
        pruning_engine_module,
        QAT_engine_module,
        PTQ_engine_module,
        B0_lookup,
        pruning_lookup,
        structure_tensor,
    )

    load_elapsed_ns = (
        time.perf_counter_ns()
        - load_started
    )

    gc.collect()

    RSS_after_model_load = int(
        process.memory_info().rss
    )

    model_load_RSS_delta = max(
        0,
        (
            RSS_after_model_load
            - RSS_before_model_load
        ),
    )

    serialized_bytes = (
        serialized_state_size(
            model
        )
    )

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
            "Loaded representation mismatch."
        )

    latency_arrays: dict[
        int,
        np.ndarray,
    ] = {}

    latency_summaries: dict[
        int,
        dict[str, float],
    ] = {}

    throughput: dict[
        int,
        float,
    ] = {}

    output_hashes: dict[
        int,
        str,
    ] = {}

    peak_RSS = (
        RSS_after_model_load
    )

    atomic_json(
        status_path,
        {
            "status": "running",
            "stage": "benchmarking",
            "measurement_order": (
                measurement_order
            ),
            "architecture": architecture,
            "variant": variant,
            "seed": seed,
            "validation_data_access": False,
            "test_data_access": False,
            "checkpoint_sha256": (
                checkpoint_hash
            ),
            "updated_at_utc": utc_now(),
        },
    )

    model.eval()

    with torch.inference_mode():
        for batch_size in BATCH_SIZES:
            tensor = batch_tensors[
                batch_size
            ]

            for _ in range(
                WARMUP_ITERATIONS[
                    batch_size
                ]
            ):
                logits = extract_logits(
                    model(
                        tensor
                    )
                )

                if tuple(
                    logits.shape
                ) != (
                    batch_size,
                    EXPECTED_CLASS_COUNT,
                ):
                    raise RuntimeError(
                        "Warmup output shape mismatch."
                    )

            latencies = np.empty(
                MEASURED_ITERATIONS[
                    batch_size
                ],
                dtype=np.int64,
            )

            final_logits = None

            for iteration in range(
                MEASURED_ITERATIONS[
                    batch_size
                ]
            ):
                started_ns = (
                    time.perf_counter_ns()
                )

                logits = extract_logits(
                    model(
                        tensor
                    )
                )

                elapsed_ns = (
                    time.perf_counter_ns()
                    - started_ns
                )

                latencies[
                    iteration
                ] = elapsed_ns

                current_RSS = int(
                    process.memory_info().rss
                )

                if current_RSS > peak_RSS:
                    peak_RSS = (
                        current_RSS
                    )

                final_logits = logits

            if final_logits is None:
                raise RuntimeError(
                    "No measured output was produced."
                )

            if not bool(
                torch.isfinite(
                    final_logits
                ).all().item()
            ):
                raise RuntimeError(
                    "Non-finite benchmark output."
                )

            latency_arrays[
                batch_size
            ] = latencies

            latency_summaries[
                batch_size
            ] = latency_statistics(
                latencies
            )

            total_seconds = (
                float(
                    np.sum(
                        latencies,
                        dtype=np.int64,
                    )
                )
                / 1_000_000_000.0
            )

            throughput[
                batch_size
            ] = (
                batch_size
                * len(latencies)
                / total_seconds
            )

            output_hashes[
                batch_size
            ] = hashlib.sha256(
                final_logits.detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()

    peak_inference_RSS_delta = max(
        0,
        (
            peak_RSS
            - RSS_after_model_load
        ),
    )

    raw_latency_path = (
        run_directory
        / "raw_latencies_ns.npz"
    )

    atomic_npz(
        raw_latency_path,
        batch_1=(
            latency_arrays[1]
        ),
        batch_32=(
            latency_arrays[32]
        ),
    )

    metrics_output_path = (
        run_directory
        / "deployment_metrics.json"
    )

    result = {
        "status": "completed",
        "phase": 6,
        "artifact_name": (
            "isolated_deployment_benchmark_run"
        ),
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "measurement_order": (
            measurement_order
        ),
        "architecture": architecture,
        "variant": variant,
        "seed": seed,
        "representation": (
            loaded_representation
        ),
        "source_metrics": {
            "path": str(
                metrics_path
            ),
            "sha256": (
                metrics_hash
            ),
        },
        "checkpoint": {
            "path": str(
                checkpoint_path
            ),
            "sha256": (
                checkpoint_hash
            ),
            "file_bytes": int(
                checkpoint_path.stat().st_size
            ),
            "serialized_state_dict_bytes": (
                serialized_bytes
            ),
        },
        "runtime": {
            "process_id": (
                os.getpid()
            ),
            "cpu_threads": (
                CPU_THREADS
            ),
            "interop_threads": (
                INTEROP_THREADS
            ),
            "load_elapsed_ms": (
                load_elapsed_ns
                / 1_000_000.0
            ),
            "RSS_before_model_load_bytes": (
                RSS_before_model_load
            ),
            "RSS_after_model_load_bytes": (
                RSS_after_model_load
            ),
            "model_load_RSS_delta_bytes": (
                model_load_RSS_delta
            ),
            "peak_process_RSS_bytes": (
                peak_RSS
            ),
            "peak_inference_RSS_delta_bytes": (
                peak_inference_RSS_delta
            ),
        },
        "batch_1": {
            "warmup_iterations": (
                WARMUP_ITERATIONS[1]
            ),
            "measured_iterations": (
                MEASURED_ITERATIONS[1]
            ),
            **latency_summaries[1],
            "samples_per_second": (
                throughput[1]
            ),
            "output_sha256": (
                output_hashes[1]
            ),
        },
        "batch_32": {
            "warmup_iterations": (
                WARMUP_ITERATIONS[32]
            ),
            "measured_iterations": (
                MEASURED_ITERATIONS[32]
            ),
            **latency_summaries[32],
            "samples_per_second": (
                throughput[32]
            ),
            "output_sha256": (
                output_hashes[32]
            ),
        },
        "data_access": {
            "train_data_access": True,
            "validation_data_access": False,
            "test_data_access": False,
        },
        "completed_at_utc": utc_now(),
    }

    atomic_json(
        metrics_output_path,
        result,
    )

    artifacts = artifact_inventory(
        run_directory
    )

    manifest_path = (
        run_directory
        / "run_manifest.json"
    )

    manifest = {
        "status": "completed",
        "phase": 6,
        "artifact_name": (
            "isolated_deployment_benchmark_run"
        ),
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "measurement_order": (
            measurement_order
        ),
        "architecture": architecture,
        "variant": variant,
        "seed": seed,
        "source_checkpoint": (
            file_record(
                checkpoint_path
            )
        ),
        "source_metrics": (
            file_record(
                metrics_path
            )
        ),
        "phase6_protocol": (
            file_record(
                PHASE6_PROTOCOL
            )
        ),
        "phase6_matrix": (
            file_record(
                PHASE6_MATRIX
            )
        ),
        "artifacts": artifacts,
        "validation_data_access": False,
        "test_data_access": False,
        "all_integrity_checks_passed": True,
        "completed_at_utc": utc_now(),
    }

    atomic_json(
        manifest_path,
        manifest,
    )

    atomic_json(
        status_path,
        {
            "status": "completed",
            "stage": "completed",
            "measurement_order": (
                measurement_order
            ),
            "architecture": architecture,
            "variant": variant,
            "seed": seed,
            "deployment_metrics_sha256": (
                sha256_file(
                    metrics_output_path
                )
            ),
            "raw_latencies_sha256": (
                sha256_file(
                    raw_latency_path
                )
            ),
            "run_manifest_sha256": (
                sha256_file(
                    manifest_path
                )
            ),
            "validation_data_access": False,
            "test_data_access": False,
            "completed_at_utc": utc_now(),
        },
    )

    print(
        json.dumps(
            {
                "status": "completed",
                "measurement_order": (
                    measurement_order
                ),
                "architecture": (
                    architecture
                ),
                "variant": variant,
                "seed": seed,
                "batch_1_median_ms": (
                    latency_summaries[1][
                        "median_ms"
                    ]
                ),
                "batch_32_median_ms": (
                    latency_summaries[32][
                        "median_ms"
                    ]
                ),
                "batch_32_samples_per_second": (
                    throughput[32]
                ),
                "model_load_RSS_delta_bytes": (
                    model_load_RSS_delta
                ),
                "peak_inference_RSS_delta_bytes": (
                    peak_inference_RSS_delta
                ),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


def load_completed_run(
    row: dict[str, str],
) -> dict[str, Any]:
    architecture = (
        row["architecture"]
    )

    variant = row["variant"]

    seed = int(
        row["seed"]
    )

    run_directory = (
        run_output_directory(
            architecture,
            variant,
            seed,
        )
    )

    status_path = (
        run_directory
        / "run_status.json"
    )

    metrics_path = (
        run_directory
        / "deployment_metrics.json"
    )

    manifest_path = (
        run_directory
        / "run_manifest.json"
    )

    raw_latency_path = (
        run_directory
        / "raw_latencies_ns.npz"
    )

    for path in (
        status_path,
        metrics_path,
        manifest_path,
        raw_latency_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    status = read_json(
        status_path
    )

    metrics = read_json(
        metrics_path
    )

    manifest = read_json(
        manifest_path
    )

    if not (
        status.get("status")
        == "completed"
        and metrics.get("status")
        == "completed"
        and manifest.get("status")
        == "completed"
        and manifest.get(
            "all_integrity_checks_passed"
        )
        is True
        and status.get(
            "deployment_metrics_sha256"
        )
        == sha256_file(
            metrics_path
        )
        and status.get(
            "raw_latencies_sha256"
        )
        == sha256_file(
            raw_latency_path
        )
        and status.get(
            "run_manifest_sha256"
        )
        == sha256_file(
            manifest_path
        )
        and metrics[
            "data_access"
        ][
            "validation_data_access"
        ]
        is False
        and metrics[
            "data_access"
        ][
            "test_data_access"
        ]
        is False
    ):
        raise RuntimeError(
            "Completed benchmark run failed "
            "resume validation: "
            f"{run_directory}"
        )

    return {
        "measurement_order": int(
            metrics[
                "measurement_order"
            ]
        ),
        "architecture": (
            metrics[
                "architecture"
            ]
        ),
        "variant": (
            metrics["variant"]
        ),
        "seed": int(
            metrics["seed"]
        ),
        "representation": (
            metrics[
                "representation"
            ]
        ),
        "checkpoint_file_bytes": int(
            metrics[
                "checkpoint"
            ]["file_bytes"]
        ),
        "serialized_state_dict_bytes": int(
            metrics[
                "checkpoint"
            ][
                "serialized_state_dict_bytes"
            ]
        ),
        "model_load_elapsed_ms": float(
            metrics[
                "runtime"
            ]["load_elapsed_ms"]
        ),
        "model_load_RSS_delta_bytes": int(
            metrics[
                "runtime"
            ][
                "model_load_RSS_delta_bytes"
            ]
        ),
        "peak_inference_RSS_delta_bytes": int(
            metrics[
                "runtime"
            ][
                "peak_inference_RSS_delta_bytes"
            ]
        ),
        "batch_1_mean_ms": float(
            metrics["batch_1"][
                "mean_ms"
            ]
        ),
        "batch_1_median_ms": float(
            metrics["batch_1"][
                "median_ms"
            ]
        ),
        "batch_1_std_ms": float(
            metrics["batch_1"][
                "std_ms"
            ]
        ),
        "batch_1_p95_ms": float(
            metrics["batch_1"][
                "p95_ms"
            ]
        ),
        "batch_1_p99_ms": float(
            metrics["batch_1"][
                "p99_ms"
            ]
        ),
        "batch_1_minimum_ms": float(
            metrics["batch_1"][
                "minimum_ms"
            ]
        ),
        "batch_1_maximum_ms": float(
            metrics["batch_1"][
                "maximum_ms"
            ]
        ),
        "batch_1_samples_per_second": float(
            metrics["batch_1"][
                "samples_per_second"
            ]
        ),
        "batch_32_mean_ms": float(
            metrics["batch_32"][
                "mean_ms"
            ]
        ),
        "batch_32_median_ms": float(
            metrics["batch_32"][
                "median_ms"
            ]
        ),
        "batch_32_std_ms": float(
            metrics["batch_32"][
                "std_ms"
            ]
        ),
        "batch_32_p95_ms": float(
            metrics["batch_32"][
                "p95_ms"
            ]
        ),
        "batch_32_p99_ms": float(
            metrics["batch_32"][
                "p99_ms"
            ]
        ),
        "batch_32_minimum_ms": float(
            metrics["batch_32"][
                "minimum_ms"
            ]
        ),
        "batch_32_maximum_ms": float(
            metrics["batch_32"][
                "maximum_ms"
            ]
        ),
        "batch_32_samples_per_second": float(
            metrics["batch_32"][
                "samples_per_second"
            ]
        ),
        "validation_data_access": False,
        "test_data_access": False,
        "run_directory": str(
            run_directory
        ),
        "deployment_metrics_sha256": (
            sha256_file(
                metrics_path
            )
        ),
        "raw_latencies_sha256": (
            sha256_file(
                raw_latency_path
            )
        ),
        "run_manifest_sha256": (
            sha256_file(
                manifest_path
            )
        ),
    }


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


def parent() -> None:
    required_paths = (
        PHASE6_PROTOCOL,
        PHASE6_PREFLIGHT_LOCK,
        PHASE6_LOADER_LOCK,
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
        OUTPUT_ALL_RUNS,
        OUTPUT_GROUP_SUMMARY,
        OUTPUT_SUMMARY,
        OUTPUT_COMPLETION,
    ):
        if output_path.exists():
            raise FileExistsError(
                "Phase 6 deployment aggregate "
                "artifact already exists; "
                "refusing to overwrite: "
                f"{output_path}"
            )

    protocol = read_json(
        PHASE6_PROTOCOL
    )

    preflight_lock = read_json(
        PHASE6_PREFLIGHT_LOCK
    )

    loader_lock = read_json(
        PHASE6_LOADER_LOCK
    )

    matrix_rows = read_csv(
        PHASE6_MATRIX
    )

    entry_checks = {
        "protocol_locked": (
            protocol.get("status")
            == "locked"
            and protocol.get(
                "protocol_version"
            )
            == PROTOCOL_VERSION
        ),
        "preflight_locked": (
            preflight_lock.get(
                "status"
            )
            == "locked"
            and preflight_lock.get(
                "all_checks_passed"
            )
            is True
        ),
        "loader_smoke_locked": (
            loader_lock.get(
                "status"
            )
            == "locked"
            and loader_lock.get(
                "ready_for_isolated_deployment_benchmark"
            )
            is True
            and loader_lock.get(
                "all_checks_passed"
            )
            is True
        ),
        "matrix_hash_matches": (
            preflight_lock.get(
                "locked_matrix_sha256"
            )
            == sha256_file(
                PHASE6_MATRIX
            )
        ),
        "loader_matrix_hash_matches": (
            loader_lock.get(
                "smoke_matrix_sha256"
            )
            == sha256_file(
                AUDIT
                / "phase6_deployment_loader_smoke_matrix_v3_2.csv"
            )
        ),
        "matrix_run_count_110": (
            len(matrix_rows)
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
            "Phase 6 isolated benchmark "
            "entry gate failed: "
            + ", ".join(
                failed_entry_checks
            )
        )

    observed_keys = {
        (
            row["architecture"],
            row["variant"],
            int(row["seed"]),
        )
        for row in matrix_rows
    }

    if observed_keys != EXPECTED_KEYS:
        raise RuntimeError(
            "Phase 6 benchmark matrix mismatch."
        )

    print("=" * 92)
    print("PHASE 6 ISOLATED DEPLOYMENT BENCHMARK")
    print("=" * 92)
    print(
        "Configurations                  : 110"
    )
    print(
        "Architecture-variant groups     : 22"
    )
    print(
        "Process isolation per run       : True"
    )
    print(
        "Primary latency batch size      : 1"
    )
    print(
        "Secondary throughput batch size : 32"
    )
    print(
        "Batch 1 warmup / measured       : "
        "50 / 500"
    )
    print(
        "Batch 32 warmup / measured      : "
        "30 / 200"
    )
    print(
        "CPU threads / interop threads   : 1 / 1"
    )
    print(
        "Validation data access          : False"
    )
    print(
        "Test data access                : False"
    )
    print(
        "Resume completed runs           : True"
    )
    print()

    ordered_rows = sorted(
        matrix_rows,
        key=lambda row: int(
            row[
                "measurement_order"
            ]
        ),
    )

    completed_rows: list[
        dict[str, Any]
    ] = []

    script_path = Path(
        __file__
    ).resolve()

    child_environment = (
        os.environ.copy()
    )

    child_environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )

    for index, row in enumerate(
        ordered_rows,
        start=1,
    ):
        architecture = (
            row["architecture"]
        )

        variant = row["variant"]

        seed = int(
            row["seed"]
        )

        measurement_order = int(
            row[
                "measurement_order"
            ]
        )

        run_directory = (
            run_output_directory(
                architecture,
                variant,
                seed,
            )
        )

        status_path = (
            run_directory
            / "run_status.json"
        )

        if (
            run_directory.exists()
            and status_path.exists()
            and read_json(
                status_path
            ).get("status")
            == "completed"
        ):
            completed_rows.append(
                load_completed_run(
                    row
                )
            )

            print(
                f"[{index}/110] "
                f"{architecture} {variant} "
                f"seed={seed} | "
                "status=resumed_completed",
                flush=True,
            )

            continue

        if run_directory.exists():
            remove_tree_with_retry(
                run_directory
            )

        command = [
            sys.executable,
            "-X",
            "utf8",
            "-u",
            str(
                script_path
            ),
            "--worker-order",
            str(
                measurement_order
            ),
        ]

        child = subprocess.run(
            command,
            cwd=str(ROOT),
            env=child_environment,
            capture_output=True,
            text=True,
            check=False,
        )

        if child.returncode != 0:
            print(
                child.stdout,
                end="",
            )
            print(
                child.stderr,
                end="",
                file=sys.stderr,
            )

            raise RuntimeError(
                "Deployment benchmark worker "
                f"failed for {architecture} "
                f"{variant} seed {seed}."
            )

        completed = (
            load_completed_run(
                row
            )
        )

        completed_rows.append(
            completed
        )

        print(
            f"[{index}/110] "
            f"{architecture} {variant} "
            f"seed={seed} | "
            "batch1 median="
            f"{completed['batch_1_median_ms']:.6f} ms | "
            "batch32 median="
            f"{completed['batch_32_median_ms']:.6f} ms | "
            "throughput="
            f"{completed['batch_32_samples_per_second']:.2f} samples/s | "
            "completed",
            flush=True,
        )

    if len(
        completed_rows
    ) != EXPECTED_RUN_COUNT:
        raise RuntimeError(
            "Expected 110 completed deployment "
            "benchmark runs."
        )

    completed_rows.sort(
        key=lambda row: int(
            row[
                "measurement_order"
            ]
        )
    )

    atomic_csv(
        OUTPUT_ALL_RUNS,
        completed_rows,
        [
            "measurement_order",
            "architecture",
            "variant",
            "seed",
            "representation",
            "checkpoint_file_bytes",
            "serialized_state_dict_bytes",
            "model_load_elapsed_ms",
            "model_load_RSS_delta_bytes",
            "peak_inference_RSS_delta_bytes",
            "batch_1_mean_ms",
            "batch_1_median_ms",
            "batch_1_std_ms",
            "batch_1_p95_ms",
            "batch_1_p99_ms",
            "batch_1_minimum_ms",
            "batch_1_maximum_ms",
            "batch_1_samples_per_second",
            "batch_32_mean_ms",
            "batch_32_median_ms",
            "batch_32_std_ms",
            "batch_32_p95_ms",
            "batch_32_p99_ms",
            "batch_32_minimum_ms",
            "batch_32_maximum_ms",
            "batch_32_samples_per_second",
            "validation_data_access",
            "test_data_access",
            "run_directory",
            "deployment_metrics_sha256",
            "raw_latencies_sha256",
            "run_manifest_sha256",
        ],
    )

    group_rows: list[
        dict[str, Any]
    ] = []

    metric_names = (
        "checkpoint_file_bytes",
        "serialized_state_dict_bytes",
        "model_load_elapsed_ms",
        "model_load_RSS_delta_bytes",
        "peak_inference_RSS_delta_bytes",
        "batch_1_median_ms",
        "batch_1_p95_ms",
        "batch_1_p99_ms",
        "batch_1_samples_per_second",
        "batch_32_median_ms",
        "batch_32_p95_ms",
        "batch_32_p99_ms",
        "batch_32_samples_per_second",
    )

    for architecture in ARCHITECTURES:
        for variant in VARIANTS:
            rows = [
                row
                for row in completed_rows
                if row[
                    "architecture"
                ]
                == architecture
                and row[
                    "variant"
                ]
                == variant
            ]

            if len(rows) != 5:
                raise RuntimeError(
                    "Expected five deployment "
                    f"runs for {architecture} "
                    f"{variant}."
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
                for metric_name in metric_names
            }

            group_rows.append(
                {
                    "architecture": (
                        architecture
                    ),
                    "variant": variant,
                    "representation": (
                        rows[0][
                            "representation"
                        ]
                    ),
                    "run_count": 5,
                    "seeds": json.dumps(
                        list(SEEDS),
                        separators=(
                            ",",
                            ":",
                        ),
                    ),
                    "mean_checkpoint_file_bytes": (
                        statistics[
                            "checkpoint_file_bytes"
                        ]["mean"]
                    ),
                    "mean_serialized_state_dict_bytes": (
                        statistics[
                            "serialized_state_dict_bytes"
                        ]["mean"]
                    ),
                    "mean_model_load_elapsed_ms": (
                        statistics[
                            "model_load_elapsed_ms"
                        ]["mean"]
                    ),
                    "mean_model_load_RSS_delta_bytes": (
                        statistics[
                            "model_load_RSS_delta_bytes"
                        ]["mean"]
                    ),
                    "mean_peak_inference_RSS_delta_bytes": (
                        statistics[
                            "peak_inference_RSS_delta_bytes"
                        ]["mean"]
                    ),
                    "mean_batch_1_median_ms": (
                        statistics[
                            "batch_1_median_ms"
                        ]["mean"]
                    ),
                    "std_batch_1_median_ms": (
                        statistics[
                            "batch_1_median_ms"
                        ]["std_population"]
                    ),
                    "mean_batch_1_p95_ms": (
                        statistics[
                            "batch_1_p95_ms"
                        ]["mean"]
                    ),
                    "mean_batch_1_p99_ms": (
                        statistics[
                            "batch_1_p99_ms"
                        ]["mean"]
                    ),
                    "mean_batch_1_samples_per_second": (
                        statistics[
                            "batch_1_samples_per_second"
                        ]["mean"]
                    ),
                    "mean_batch_32_median_ms": (
                        statistics[
                            "batch_32_median_ms"
                        ]["mean"]
                    ),
                    "std_batch_32_median_ms": (
                        statistics[
                            "batch_32_median_ms"
                        ]["std_population"]
                    ),
                    "mean_batch_32_p95_ms": (
                        statistics[
                            "batch_32_p95_ms"
                        ]["mean"]
                    ),
                    "mean_batch_32_p99_ms": (
                        statistics[
                            "batch_32_p99_ms"
                        ]["mean"]
                    ),
                    "mean_batch_32_samples_per_second": (
                        statistics[
                            "batch_32_samples_per_second"
                        ]["mean"]
                    ),
                    "validation_data_access": (
                        False
                    ),
                    "test_data_access": (
                        False
                    ),
                }
            )

    if len(
        group_rows
    ) != EXPECTED_GROUP_COUNT:
        raise RuntimeError(
            "Expected 22 deployment groups."
        )

    atomic_csv(
        OUTPUT_GROUP_SUMMARY,
        group_rows,
        [
            "architecture",
            "variant",
            "representation",
            "run_count",
            "seeds",
            "mean_checkpoint_file_bytes",
            "mean_serialized_state_dict_bytes",
            "mean_model_load_elapsed_ms",
            "mean_model_load_RSS_delta_bytes",
            "mean_peak_inference_RSS_delta_bytes",
            "mean_batch_1_median_ms",
            "std_batch_1_median_ms",
            "mean_batch_1_p95_ms",
            "mean_batch_1_p99_ms",
            "mean_batch_1_samples_per_second",
            "mean_batch_32_median_ms",
            "std_batch_32_median_ms",
            "mean_batch_32_p95_ms",
            "mean_batch_32_p99_ms",
            "mean_batch_32_samples_per_second",
            "validation_data_access",
            "test_data_access",
        ],
    )

    global_checks = {
        "run_count_110": (
            len(completed_rows)
            == EXPECTED_RUN_COUNT
        ),
        "group_count_22": (
            len(group_rows)
            == EXPECTED_GROUP_COUNT
        ),
        "configuration_matrix_exact": (
            {
                (
                    row["architecture"],
                    row["variant"],
                    int(row["seed"]),
                )
                for row in completed_rows
            }
            == EXPECTED_KEYS
        ),
        "process_isolation_per_run": (
            True
        ),
        "validation_access_zero": (
            all(
                row[
                    "validation_data_access"
                ]
                is False
                for row in completed_rows
            )
        ),
        "test_access_zero": (
            all(
                row[
                    "test_data_access"
                ]
                is False
                for row in completed_rows
            )
        ),
        "final_model_not_selected": (
            True
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
            "Phase 6 deployment benchmark "
            "global checks failed: "
            + ", ".join(
                failed_global_checks
            )
        )

    summary = {
        "status": "completed",
        "phase": 6,
        "artifact_name": (
            "isolated_deployment_benchmark"
        ),
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "completed_at_utc": utc_now(),
        "run_count": (
            EXPECTED_RUN_COUNT
        ),
        "group_count": (
            EXPECTED_GROUP_COUNT
        ),
        "architecture_count": 2,
        "variant_count": 11,
        "seed_count": 5,
        "runtime_policy": {
            "process_isolation_per_run": (
                True
            ),
            "cpu_threads": 1,
            "interop_threads": 1,
            "batch_1_warmup_iterations": (
                WARMUP_ITERATIONS[1]
            ),
            "batch_1_measured_iterations": (
                MEASURED_ITERATIONS[1]
            ),
            "batch_32_warmup_iterations": (
                WARMUP_ITERATIONS[32]
            ),
            "batch_32_measured_iterations": (
                MEASURED_ITERATIONS[32]
            ),
        },
        "data_access": {
            "train_data_access": True,
            "validation_data_access": False,
            "test_data_access": False,
        },
        "selection_policy": {
            "final_model_selected": False,
        },
        "entry_checks": (
            entry_checks
        ),
        "global_checks": (
            global_checks
        ),
        "all_runs_csv": (
            file_record(
                OUTPUT_ALL_RUNS
            )
        ),
        "group_summary_csv": (
            file_record(
                OUTPUT_GROUP_SUMMARY
            )
        ),
        "ready_for_independent_verification": (
            True
        ),
        "all_checks_passed": True,
    }

    atomic_json(
        OUTPUT_SUMMARY,
        summary,
    )

    completion = {
        "status": "completed",
        "phase": 6,
        "artifact_name": (
            "isolated_deployment_benchmark"
        ),
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "completed_at_utc": utc_now(),
        "run_count": (
            EXPECTED_RUN_COUNT
        ),
        "group_count": (
            EXPECTED_GROUP_COUNT
        ),
        "all_runs_csv": str(
            OUTPUT_ALL_RUNS
        ),
        "all_runs_csv_sha256": (
            sha256_file(
                OUTPUT_ALL_RUNS
            )
        ),
        "group_summary_csv": str(
            OUTPUT_GROUP_SUMMARY
        ),
        "group_summary_csv_sha256": (
            sha256_file(
                OUTPUT_GROUP_SUMMARY
            )
        ),
        "summary_json": str(
            OUTPUT_SUMMARY
        ),
        "summary_json_sha256": (
            sha256_file(
                OUTPUT_SUMMARY
            )
        ),
        "validation_data_access": False,
        "test_data_access": False,
        "final_model_selected": False,
        "ready_for_independent_verification": (
            True
        ),
        "all_checks_passed": True,
    }

    atomic_json(
        OUTPUT_COMPLETION,
        completion,
    )

    print()
    print("=" * 92)
    print("PHASE 6 ISOLATED DEPLOYMENT BENCHMARK SUMMARY")
    print("=" * 92)
    print(
        "Runs completed                  : 110"
    )
    print(
        "Groups completed                : 22"
    )
    print(
        "Process isolation per run       : True"
    )
    print(
        "Batch 1 measurements per run    : 500"
    )
    print(
        "Batch 32 measurements per run   : 200"
    )
    print(
        "Validation data access          : False"
    )
    print(
        "Test data access                : False"
    )
    print(
        "Final model selected            : False"
    )
    print()
    print(
        "Group summary:"
    )

    for architecture in ARCHITECTURES:
        print(
            architecture
        )

        for variant in VARIANTS:
            row = next(
                value
                for value in group_rows
                if value[
                    "architecture"
                ]
                == architecture
                and value[
                    "variant"
                ]
                == variant
            )

            print(
                "  "
                f"{variant:<14} | "
                "batch1 median="
                f"{float(row['mean_batch_1_median_ms']):.6f} ms | "
                "batch32 median="
                f"{float(row['mean_batch_32_median_ms']):.6f} ms | "
                "throughput="
                f"{float(row['mean_batch_32_samples_per_second']):.2f} samples/s | "
                "state="
                f"{float(row['mean_serialized_state_dict_bytes']) / 1024.0:.2f} KiB"
            )

        print()

    print(
        "All-runs CSV                   : "
        f"{OUTPUT_ALL_RUNS}"
    )
    print(
        "Group summary CSV               : "
        f"{OUTPUT_GROUP_SUMMARY}"
    )
    print(
        "Summary JSON                    : "
        f"{OUTPUT_SUMMARY}"
    )
    print(
        "Ready for independent verify    : True"
    )
    print(
        "All checks passed               : True"
    )
    print(
        "PHASE 6 ISOLATED DEPLOYMENT BENCHMARK COMPLETED"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker-order",
        type=int,
        default=None,
    )

    arguments = parser.parse_args()

    if arguments.worker_order is None:
        parent()
    else:
        worker(
            arguments.worker_order
        )


if __name__ == "__main__":
    main()
