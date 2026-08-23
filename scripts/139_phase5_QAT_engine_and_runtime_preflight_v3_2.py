from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import random
import sys
import tempfile
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

PRUNING_SOURCES_LOCK = (
    AUDIT
    / "phase5_physical_pruning_sources_verified_locked_v3_2.json"
)

PRUNING_SOURCES_REGISTRY = (
    AUDIT
    / "phase5_physical_pruning_source_checkpoint_registry_v3_2.csv"
)

FP32_FT_LOCK = (
    AUDIT
    / "phase5_FP32_FT_locked_v3_2.json"
)

DQ_LOCK = (
    AUDIT
    / "phase5_DQ_locked_v3_2.json"
)

MODEL_SOURCE = (
    ROOT
    / "src"
    / "models"
    / "nbaiot_models.py"
)

QAT_ENGINE_PATH = (
    ROOT
    / "src"
    / "compression"
    / "phase5_qat_engine_v3_2.py"
)

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

X_TRAIN_PATH = (
    FINAL_CACHE
    / "X_train.npy"
)

Y_TRAIN_PATH = (
    FINAL_CACHE
    / "y_train.npy"
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

CLASS_WEIGHTS_NPZ = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "shared"
    / "preprocessing"
    / "phase5_train_only_class_weights_v3_2.npz"
)

OUTPUT_MATRIX = (
    AUDIT
    / "phase5_QAT_runtime_preflight_matrix_v3_2.csv"
)

OUTPUT_REPORT = (
    AUDIT
    / "phase5_QAT_runtime_preflight_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase5_QAT_runtime_preflight_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase5_QAT_runtime_preflight_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = (
    "phase5_fair_budget_compression_v3_2"
)

QAT_ENGINE_VERSION = (
    "phase5_qat_engine_v3_2"
)

PRUNING_ENGINE_VERSION = (
    "phase5_physical_pruning_engine_v3_2"
)

ARCHITECTURES = (
    "tinyml_mlp",
    "compact_dnn",
)

VARIANTS = (
    "QAT",
    "P25-QAT",
    "P50-QAT",
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

SOURCE_VARIANT = {
    "QAT": "B0",
    "P25-QAT": "P25-noFT",
    "P50-QAT": "P50-noFT",
}

PRUNING_RATIO = {
    "P25-QAT": 0.25,
    "P50-QAT": 0.50,
}

BASELINE_LEARNING_RATE = {
    "tinyml_mlp": 0.003,
    "compact_dnn": 0.001,
}

QAT_LEARNING_RATE = {
    architecture: value * 0.1
    for architecture, value
    in BASELINE_LEARNING_RATE.items()
}

HIDDEN_DIMS = {
    "tinyml_mlp": [64, 32],
    "compact_dnn": [128, 64, 32],
}

EXPECTED_WIDTHS = {
    "tinyml_mlp": {
        "QAT": [
            [115, 64],
            [64, 32],
            [32, 3],
        ],
        "P25-QAT": [
            [115, 48],
            [48, 24],
            [24, 3],
        ],
        "P50-QAT": [
            [115, 32],
            [32, 16],
            [16, 3],
        ],
    },
    "compact_dnn": {
        "QAT": [
            [115, 128],
            [128, 64],
            [64, 32],
            [32, 3],
        ],
        "P25-QAT": [
            [115, 96],
            [96, 48],
            [48, 24],
            [24, 3],
        ],
        "P50-QAT": [
            [115, 64],
            [64, 32],
            [32, 16],
            [16, 3],
        ],
    },
}

EXPECTED_INPUT_FEATURES = 115
EXPECTED_OUTPUT_CLASSES = 3

MAX_EPOCHS = 8
BATCH_SIZE = 4096
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0002
WEIGHT_DECAY = 0.0001

SMOKE_BATCH_SIZE = 257
CPU_THREADS = 4


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
        "Please use quant_min and quant_max.*"
    ),
    category=UserWarning,
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


def set_all_seeds(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def state_digest(
    model: nn.Module,
) -> str:
    digest = hashlib.sha256()

    for name, tensor in sorted(
        model.state_dict().items()
    ):
        digest.update(
            name.encode("utf-8")
        )

        buffer = io.BytesIO()

        torch.save(
            tensor,
            buffer,
        )

        digest.update(
            buffer.getvalue()
        )

    return digest.hexdigest()


def serialized_state_size(
    state_dict: dict[str, Any],
) -> int:
    buffer = io.BytesIO()

    torch.save(
        state_dict,
        buffer,
    )

    return len(
        buffer.getvalue()
    )


def build_source_model(
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
    str,
]:
    if variant == "QAT":
        row = B0_lookup[
            (
                architecture,
                seed,
            )
        ]

        checkpoint_path = Path(
            row["checkpoint_path"]
        )

        checkpoint_hash = sha256_file(
            checkpoint_path
        )

        if (
            checkpoint_hash
            != row[
                "checkpoint_sha256"
            ]
        ):
            raise RuntimeError(
                "B0 checkpoint hash mismatch."
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model_symbol = (
            row["model_symbol"]
        )

        model = instantiate_model(
            model_module,
            model_symbol,
            architecture,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        model.eval()

        return (
            model,
            model_symbol,
            checkpoint_path,
            checkpoint_hash,
            "B0",
        )

    source_variant = (
        SOURCE_VARIANT[
            variant
        ]
    )

    row = pruning_lookup[
        (
            architecture,
            source_variant,
            seed,
        )
    ]

    source_checkpoint_path = Path(
        row[
            "source_checkpoint_path"
        ]
    )

    source_checkpoint_hash = (
        sha256_file(
            source_checkpoint_path
        )
    )

    if (
        source_checkpoint_hash
        != row[
            "source_checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "Pruning source checkpoint "
            "hash mismatch."
        )

    B0_checkpoint_path = Path(
        row[
            "source_B0_checkpoint_path"
        ]
    )

    B0_checkpoint_hash = (
        sha256_file(
            B0_checkpoint_path
        )
    )

    if (
        B0_checkpoint_hash
        != row[
            "source_B0_checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "Pruning source B0 checkpoint "
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
        row["model_symbol"]
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

    (
        model,
        regenerated_metadata,
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

    if (
        regenerated_metadata
        != source_checkpoint[
            "pruning_metadata"
        ]
    ):
        raise RuntimeError(
            "Pruning metadata mismatch."
        )

    model.load_state_dict(
        source_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    return (
        model,
        model_symbol,
        source_checkpoint_path,
        source_checkpoint_hash,
        source_variant,
    )


def execute_QAT_smoke_once(
    source_model: nn.Module,
    backend: str,
    learning_rate: float,
    seed: int,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    class_weights: torch.Tensor,
    QAT_engine_module: ModuleType,
) -> dict[str, Any]:
    set_all_seeds(seed)

    prepared = (
        QAT_engine_module
        .prepare_qat_model(
            source_model,
            backend,
        )
    )

    optimizer = torch.optim.AdamW(
        prepared.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = (
        QAT_engine_module
        .extract_logits(
            prepared(
                x_batch
            )
        )
    )

    loss = criterion(
        logits,
        y_batch,
    )

    if not torch.isfinite(
        loss
    ):
        raise RuntimeError(
            "Non-finite QAT smoke loss."
        )

    loss.backward()
    optimizer.step()

    prepared.eval()

    with torch.inference_mode():
        fake_quant_logits = (
            QAT_engine_module
            .extract_logits(
                prepared(
                    x_batch
                )
            )
        )

    converted = (
        QAT_engine_module
        .convert_qat_model(
            prepared
        )
    )

    with torch.inference_mode():
        quantized_logits = (
            QAT_engine_module
            .extract_logits(
                converted(
                    x_batch
                )
            )
        )

    return {
        "prepared_model": prepared,
        "converted_model": converted,
        "loss": float(
            loss.item()
        ),
        "fake_quant_logits": (
            fake_quant_logits
            .detach()
            .cpu()
        ),
        "quantized_logits": (
            quantized_logits
            .detach()
            .cpu()
        ),
    }


required_paths = (
    PHASE5_PROTOCOL,
    LOCKED_MATRIX,
    B0_PAIR_LOCK,
    B0_CHECKPOINT_REGISTRY,
    PRUNING_ENGINE_LOCK,
    PRUNING_ENGINE_PATH,
    PRUNING_SOURCES_LOCK,
    PRUNING_SOURCES_REGISTRY,
    FP32_FT_LOCK,
    DQ_LOCK,
    MODEL_SOURCE,
    QAT_ENGINE_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    SCALER_NPZ,
    CLASS_WEIGHTS_NPZ,
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
            "QAT preflight artifact already "
            "exists; refusing to overwrite: "
            f"{output_path}"
        )

protocol = read_json(
    PHASE5_PROTOCOL
)

B0_pair_lock = read_json(
    B0_PAIR_LOCK
)

pruning_engine_lock = read_json(
    PRUNING_ENGINE_LOCK
)

pruning_sources_lock = read_json(
    PRUNING_SOURCES_LOCK
)

FP32_FT_lock = read_json(
    FP32_FT_LOCK
)

DQ_lock = read_json(
    DQ_LOCK
)

matrix_rows = read_csv(
    LOCKED_MATRIX
)

B0_rows = read_csv(
    B0_CHECKPOINT_REGISTRY
)

pruning_rows = read_csv(
    PRUNING_SOURCES_REGISTRY
)

QAT_matrix_rows = [
    row
    for row in matrix_rows
    if row["architecture"]
    in ARCHITECTURES
    and row["variant"]
    in VARIANTS
]

QAT_matrix_lookup = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    ): row
    for row in QAT_matrix_rows
}

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
    "pruning_engine_locked": (
        pruning_engine_lock.get(
            "status"
        )
        == "locked"
        and pruning_engine_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "pruning_engine_hash_matches": (
        pruning_engine_lock.get(
            "engine_source_sha256"
        )
        == sha256_file(
            PRUNING_ENGINE_PATH
        )
    ),
    "pruning_sources_locked": (
        pruning_sources_lock.get(
            "status"
        )
        == "locked"
        and pruning_sources_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "pruning_registry_hash_matches": (
        pruning_sources_lock.get(
            "source_registry_sha256"
        )
        == sha256_file(
            PRUNING_SOURCES_REGISTRY
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
    "DQ_locked": (
        DQ_lock.get("status")
        == "locked"
        and DQ_lock.get(
            "ready_for_QAT_and_PTQ_branches"
        )
        is True
        and DQ_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "QAT_matrix_has_thirty_rows": (
        len(QAT_matrix_rows)
        == 30
        and set(
            QAT_matrix_lookup.keys()
        )
        == EXPECTED_KEYS
    ),
    "B0_registry_complete": (
        set(
            B0_lookup.keys()
        )
        == {
            (
                architecture,
                seed,
            )
            for architecture
            in ARCHITECTURES
            for seed in SEEDS
        }
    ),
    "pruning_registry_complete": (
        {
            (
                architecture,
                source_variant,
                seed,
            )
            for architecture
            in ARCHITECTURES
            for source_variant
            in (
                "P25-noFT",
                "P50-noFT",
            )
            for seed in SEEDS
        }
        .issubset(
            set(
                pruning_lookup.keys()
            )
        )
    ),
    "QAT_budget_matches": all(
        int(row["max_epochs"])
        == MAX_EPOCHS
        and int(row["batch_size"])
        == BATCH_SIZE
        and int(
            row[
                "early_stopping_patience"
            ]
        )
        == EARLY_STOPPING_PATIENCE
        and math.isclose(
            float(
                row[
                    "early_stopping_min_delta"
                ]
            ),
            EARLY_STOPPING_MIN_DELTA,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            float(
                row["learning_rate"]
            ),
            QAT_LEARNING_RATE[
                row["architecture"]
            ],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and int(
            row[
                "phase5_test_evaluation_count"
            ]
        )
        == 1
        for row in QAT_matrix_rows
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
        "QAT preflight entry gate failed: "
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

x_train = np.load(
    X_TRAIN_PATH,
    mmap_mode="r",
)

y_train = np.load(
    Y_TRAIN_PATH,
    mmap_mode="r",
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

with np.load(
    CLASS_WEIGHTS_NPZ
) as values:
    class_weights32 = np.array(
        values[
            "class_weights_float32"
        ],
        copy=True,
    )

smoke_numpy = (
    (
        np.asarray(
            x_train[
                :SMOKE_BATCH_SIZE
            ],
            dtype=np.float64,
        )
        - mean64
    )
    / scale64
).astype(np.float32)

smoke_labels = np.asarray(
    y_train[
        :SMOKE_BATCH_SIZE
    ],
    dtype=np.int64,
)

if not np.isfinite(
    smoke_numpy
).all():
    raise RuntimeError(
        "Non-finite QAT smoke batch."
    )

smoke_batch = torch.from_numpy(
    smoke_numpy
)

smoke_y = torch.from_numpy(
    smoke_labels
)

class_weights = torch.from_numpy(
    class_weights32
)

model_module = load_module(
    MODEL_SOURCE,
    "phase5_QAT_preflight_models",
)

pruning_engine_module = load_module(
    PRUNING_ENGINE_PATH,
    "phase5_QAT_preflight_pruning_engine",
)

QAT_engine_module = load_module(
    QAT_ENGINE_PATH,
    "phase5_QAT_engine",
)

if (
    pruning_engine_module.ENGINE_VERSION
    != PRUNING_ENGINE_VERSION
):
    raise RuntimeError(
        "Pruning-engine version mismatch."
    )

if (
    QAT_engine_module.ENGINE_VERSION
    != QAT_ENGINE_VERSION
):
    raise RuntimeError(
        "QAT-engine version mismatch."
    )

supported_backends = list(
    torch.backends.quantized.supported_engines
)

candidate_backends = []

for candidate in (
    torch.backends.quantized.engine,
    "x86",
    "fbgemm",
    "onednn",
    "qnnpack",
):
    if (
        candidate
        and candidate
        in supported_backends
        and candidate
        not in candidate_backends
    ):
        candidate_backends.append(
            candidate
        )

representative_model, _, _, _, _ = (
    build_source_model(
        "tinyml_mlp",
        "QAT",
        42,
        model_module,
        pruning_engine_module,
        B0_lookup,
        pruning_lookup,
    )
)

backend_attempts: list[
    dict[str, Any]
] = []

selected_backend: str | None = None

for candidate in candidate_backends:
    try:
        torch.backends.quantized.engine = (
            candidate
        )

        prepared = (
            QAT_engine_module
            .prepare_qat_model(
                representative_model,
                candidate,
            )
        )

        with torch.no_grad():
            prepared(
                smoke_batch[:32]
            )

        converted = (
            QAT_engine_module
            .convert_qat_model(
                prepared
            )
        )

        with torch.inference_mode():
            output = (
                QAT_engine_module
                .extract_logits(
                    converted(
                        smoke_batch[:32]
                    )
                )
            )

        expected_linear_count = (
            QAT_engine_module
            .float_linear_count(
                representative_model
            )
        )

        converted_linear_count = len(
            QAT_engine_module
            .static_quantized_linear_modules(
                converted
            )
        )

        passed = (
            tuple(output.shape)
            == (
                32,
                EXPECTED_OUTPUT_CLASSES,
            )
            and bool(
                torch.isfinite(
                    output
                ).all().item()
            )
            and converted_linear_count
            == expected_linear_count
        )

        backend_attempts.append(
            {
                "backend": candidate,
                "passed": passed,
                "error": None,
            }
        )

        if passed:
            selected_backend = candidate
            break

    except Exception as error:
        backend_attempts.append(
            {
                "backend": candidate,
                "passed": False,
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }
        )

if selected_backend is None:
    raise RuntimeError(
        "No supported QAT backend passed "
        "the local runtime smoke test."
    )

torch.backends.quantized.engine = (
    selected_backend
)

del representative_model

print("=" * 92)
print("PHASE 5 QAT ENGINE AND RUNTIME PREFLIGHT")
print("=" * 92)
print(
    "Configurations                  : 30"
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
    "Selected quantization backend   : "
    f"{selected_backend}"
)
print(
    "QAT engine version              : "
    f"{QAT_ENGINE_VERSION}"
)
print(
    "Train smoke rows per config     : "
    f"{SMOKE_BATCH_SIZE}"
)
print(
    "Validation access count         : 0"
)
print(
    "Test access count               : 0"
)
print()

rows: list[dict[str, Any]] = []

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
    architecture, variant, seed = key

    matrix_row = (
        QAT_matrix_lookup[key]
    )

    source_model, (
        model_symbol
    ), source_path, (
        source_hash
    ), source_variant = (
        build_source_model(
            architecture,
            variant,
            seed,
            model_module,
            pruning_engine_module,
            B0_lookup,
            pruning_lookup,
        )
    )

    source_digest_before = (
        state_digest(
            source_model
        )
    )

    expected_widths = (
        EXPECTED_WIDTHS[
            architecture
        ][variant]
    )

    if (
        QAT_engine_module
        .float_linear_widths(
            source_model
        )
        != expected_widths
    ):
        raise RuntimeError(
            f"{architecture} {variant} "
            f"seed {seed}: source topology "
            "mismatch."
        )

    learning_rate = float(
        matrix_row[
            "learning_rate"
        ]
    )

    first = execute_QAT_smoke_once(
        source_model,
        selected_backend,
        learning_rate,
        seed,
        smoke_batch,
        smoke_y,
        class_weights,
        QAT_engine_module,
    )

    second = execute_QAT_smoke_once(
        source_model,
        selected_backend,
        learning_rate,
        seed,
        smoke_batch,
        smoke_y,
        class_weights,
        QAT_engine_module,
    )

    prepared_a = (
        first[
            "prepared_model"
        ]
    )

    converted_a = (
        first[
            "converted_model"
        ]
    )

    converted_b = (
        second[
            "converted_model"
        ]
    )

    float_linear_total = (
        QAT_engine_module
        .float_linear_count(
            source_model
        )
    )

    QAT_linear_total = (
        QAT_engine_module
        .qat_linear_count(
            prepared_a
        )
    )

    fake_quant_total = (
        QAT_engine_module
        .fake_quantizer_count(
            prepared_a
        )
    )

    quantized_linear_total = len(
        QAT_engine_module
        .static_quantized_linear_modules(
            converted_a
        )
    )

    weight_dtypes = (
        QAT_engine_module
        .quantized_weight_dtypes(
            converted_a
        )
    )

    with tempfile.TemporaryDirectory() as directory:
        state_path = (
            Path(directory)
            / "qat_quantized_state.pt"
        )

        torch.save(
            converted_a.state_dict(),
            state_path,
        )

        loaded_state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
        )

        converted_b.load_state_dict(
            loaded_state,
            strict=True,
        )

        converted_b.eval()

        with torch.inference_mode():
            roundtrip_logits = (
                QAT_engine_module
                .extract_logits(
                    converted_b(
                        smoke_batch
                    )
                )
                .detach()
                .cpu()
            )

        roundtrip_file_size = int(
            state_path.stat().st_size
        )

    source_digest_after = (
        state_digest(
            source_model
        )
    )

    quantized_logits_a = (
        first[
            "quantized_logits"
        ]
    )

    quantized_logits_b = (
        second[
            "quantized_logits"
        ]
    )

    checks = {
        "source_topology_matches": (
            QAT_engine_module
            .float_linear_widths(
                source_model
            )
            == expected_widths
        ),
        "QAT_linear_count_matches": (
            QAT_linear_total
            == float_linear_total
            and QAT_linear_total > 0
        ),
        "fake_quantizers_present": (
            fake_quant_total > 0
        ),
        "converted_linear_count_matches": (
            quantized_linear_total
            == float_linear_total
            and quantized_linear_total > 0
        ),
        "quantized_weights_qint8": (
            len(weight_dtypes)
            == float_linear_total
            and all(
                dtype
                == "torch.qint8"
                for dtype in weight_dtypes
            )
        ),
        "loss_finite": (
            math.isfinite(
                first["loss"]
            )
            and math.isfinite(
                second["loss"]
            )
        ),
        "output_shapes_match": (
            tuple(
                first[
                    "fake_quant_logits"
                ].shape
            )
            == (
                SMOKE_BATCH_SIZE,
                EXPECTED_OUTPUT_CLASSES,
            )
            and tuple(
                quantized_logits_a.shape
            )
            == (
                SMOKE_BATCH_SIZE,
                EXPECTED_OUTPUT_CLASSES,
            )
        ),
        "outputs_finite": (
            bool(
                torch.isfinite(
                    first[
                        "fake_quant_logits"
                    ]
                ).all().item()
            )
            and bool(
                torch.isfinite(
                    quantized_logits_a
                ).all().item()
            )
        ),
        "deterministic_QAT_conversion": (
            torch.equal(
                quantized_logits_a,
                quantized_logits_b,
            )
        ),
        "serialization_roundtrip_exact": (
            torch.equal(
                quantized_logits_a,
                roundtrip_logits,
            )
        ),
        "source_model_unchanged": (
            source_digest_before
            == source_digest_after
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
            f"{architecture} {variant} "
            f"seed {seed} QAT preflight "
            "failed: "
            + ", ".join(
                failed_checks
            )
        )

    float_state_bytes = (
        serialized_state_size(
            source_model.state_dict()
        )
    )

    quantized_state_bytes = (
        serialized_state_size(
            converted_a.state_dict()
        )
    )

    rows.append(
        {
            "architecture": (
                architecture
            ),
            "variant": variant,
            "source_variant": (
                source_variant
            ),
            "seed": seed,
            "model_symbol": (
                model_symbol
            ),
            "source_checkpoint_path": (
                str(
                    source_path
                )
            ),
            "source_checkpoint_sha256": (
                source_hash
            ),
            "QAT_backend": (
                selected_backend
            ),
            "QAT_learning_rate": (
                learning_rate
            ),
            "float_linear_count": (
                float_linear_total
            ),
            "QAT_linear_count": (
                QAT_linear_total
            ),
            "fake_quantizer_count": (
                fake_quant_total
            ),
            "converted_quantized_linear_count": (
                quantized_linear_total
            ),
            "quantized_weight_dtypes": (
                json.dumps(
                    weight_dtypes,
                    separators=(
                        ",",
                        ":",
                    ),
                )
            ),
            "float_state_bytes": (
                float_state_bytes
            ),
            "quantized_state_bytes": (
                quantized_state_bytes
            ),
            "quantized_to_float_state_size_ratio": (
                quantized_state_bytes
                / float_state_bytes
            ),
            "roundtrip_file_size_bytes": (
                roundtrip_file_size
            ),
            "smoke_loss": (
                first["loss"]
            ),
            "deterministic_QAT_conversion": (
                True
            ),
            "serialization_roundtrip_exact": (
                True
            ),
            "source_model_unchanged": (
                True
            ),
            "train_smoke_rows_used": (
                SMOKE_BATCH_SIZE
            ),
            "validation_access_count": 0,
            "test_access_count": 0,
            "all_checks_passed": True,
        }
    )

    print(
        f"[{index}/30] "
        f"{architecture} seed={seed} "
        f"{variant} | "
        f"source={source_variant} | "
        f"QAT_linear={QAT_linear_total} | "
        f"INT8_linear={quantized_linear_total} | "
        "deterministic=True | "
        "roundtrip=True | "
        "validation=0 | test=0 | "
        "all checks=True",
        flush=True,
    )

    del source_model
    del prepared_a
    del converted_a
    del converted_b
    del first
    del second

atomic_csv(
    OUTPUT_MATRIX,
    rows,
    [
        "architecture",
        "variant",
        "source_variant",
        "seed",
        "model_symbol",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "QAT_backend",
        "QAT_learning_rate",
        "float_linear_count",
        "QAT_linear_count",
        "fake_quantizer_count",
        "converted_quantized_linear_count",
        "quantized_weight_dtypes",
        "float_state_bytes",
        "quantized_state_bytes",
        "quantized_to_float_state_size_ratio",
        "roundtrip_file_size_bytes",
        "smoke_loss",
        "deterministic_QAT_conversion",
        "serialization_roundtrip_exact",
        "source_model_unchanged",
        "train_smoke_rows_used",
        "validation_access_count",
        "test_access_count",
        "all_checks_passed",
    ],
)

global_checks = {
    "thirty_configurations_audited": (
        len(rows)
        == 30
    ),
    "all_QAT_linears_prepared": (
        all(
            int(
                row[
                    "QAT_linear_count"
                ]
            )
            == int(
                row[
                    "float_linear_count"
                ]
            )
            for row in rows
        )
    ),
    "all_linears_converted_to_INT8": (
        all(
            int(
                row[
                    "converted_quantized_linear_count"
                ]
            )
            == int(
                row[
                    "float_linear_count"
                ]
            )
            for row in rows
        )
    ),
    "all_quantized_weights_qint8": (
        all(
            all(
                dtype
                == "torch.qint8"
                for dtype in json.loads(
                    row[
                        "quantized_weight_dtypes"
                    ]
                )
            )
            for row in rows
        )
    ),
    "all_conversions_deterministic": (
        all(
            bool(
                row[
                    "deterministic_QAT_conversion"
                ]
            )
            for row in rows
        )
    ),
    "all_roundtrips_exact": (
        all(
            bool(
                row[
                    "serialization_roundtrip_exact"
                ]
            )
            for row in rows
        )
    ),
    "all_sources_unchanged": (
        all(
            bool(
                row[
                    "source_model_unchanged"
                ]
            )
            for row in rows
        )
    ),
    "validation_access_zero": (
        all(
            int(
                row[
                    "validation_access_count"
                ]
            )
            == 0
            for row in rows
        )
    ),
    "test_access_zero": (
        all(
            int(
                row[
                    "test_access_count"
                ]
            )
            == 0
            for row in rows
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
        "QAT preflight global checks failed: "
        + ", ".join(
            failed_global_checks
        )
    )

report = {
    "status": "passed",
    "phase": 5,
    "artifact_name": (
        "QAT_engine_and_runtime_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "QAT_engine_version": (
        QAT_ENGINE_VERSION
    ),
    "verified_at_utc": utc_now(),
    "architecture_count": 2,
    "variant_count": 3,
    "seed_count": 5,
    "configuration_count": 30,
    "variants": list(
        VARIANTS
    ),
    "source_policy": {
        "QAT": "matching locked B0 checkpoint",
        "P25-QAT": (
            "matching locked P25-noFT source"
        ),
        "P50-QAT": (
            "matching locked P50-noFT source"
        ),
    },
    "QAT_policy": {
        "selected_backend": (
            selected_backend
        ),
        "supported_backends": (
            supported_backends
        ),
        "backend_attempts": (
            backend_attempts
        ),
        "weight_dtype": (
            "torch.qint8"
        ),
        "activation_dtype": (
            "backend_default_QAT_activation_dtype"
        ),
        "target_modules": [
            "torch.nn.Linear",
        ],
        "optimizer": "AdamW",
        "weight_decay": (
            WEIGHT_DECAY
        ),
        "learning_rates": (
            QAT_LEARNING_RATE
        ),
    },
    "data_access": {
        "train_smoke_rows_used_per_configuration": (
            SMOKE_BATCH_SIZE
        ),
        "validation_access_count": 0,
        "test_access_count": 0,
    },
    "entry_checks": (
        entry_checks
    ),
    "global_checks": (
        global_checks
    ),
    "QAT_engine_source": (
        file_record(
            QAT_ENGINE_PATH
        )
    ),
    "matrix_csv": (
        file_record(
            OUTPUT_MATRIX
        )
    ),
    "ready_for_QAT_Q25_Q50_training": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_REPORT,
    report,
)

lock = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "QAT_engine_and_runtime_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "QAT_engine_version": (
        QAT_ENGINE_VERSION
    ),
    "locked_at_utc": utc_now(),
    "configuration_count": 30,
    "selected_backend": (
        selected_backend
    ),
    "quantized_weight_dtype": (
        "torch.qint8"
    ),
    "QAT_engine_source": str(
        QAT_ENGINE_PATH
    ),
    "QAT_engine_source_sha256": (
        sha256_file(
            QAT_ENGINE_PATH
        )
    ),
    "validation_access_count": 0,
    "test_access_count": 0,
    "matrix_csv": str(
        OUTPUT_MATRIX
    ),
    "matrix_csv_sha256": (
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
    "ready_for_QAT_Q25_Q50_training": True,
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
        "QAT_engine_and_runtime_preflight"
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
            PRUNING_ENGINE_LOCK
        ),
        file_record(
            PRUNING_ENGINE_PATH
        ),
        file_record(
            PRUNING_SOURCES_LOCK
        ),
        file_record(
            PRUNING_SOURCES_REGISTRY
        ),
        file_record(
            FP32_FT_LOCK
        ),
        file_record(
            DQ_LOCK
        ),
        file_record(
            MODEL_SOURCE
        ),
        file_record(
            QAT_ENGINE_PATH
        ),
        file_record(
            SCALER_NPZ
        ),
        file_record(
            CLASS_WEIGHTS_NPZ
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
    "configuration_count": 30,
    "selected_backend": (
        selected_backend
    ),
    "validation_access_count": 0,
    "test_access_count": 0,
    "ready_for_QAT_Q25_Q50_training": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print()
print("=" * 92)
print("PHASE 5 QAT ENGINE AND RUNTIME PREFLIGHT SUMMARY")
print("=" * 92)
print(
    "Configurations audited          : 30"
)
print(
    "Architectures                   : 2"
)
print(
    "Variants                        : QAT, P25-QAT, P50-QAT"
)
print(
    "Selected quantization backend   : "
    f"{selected_backend}"
)
print(
    "QAT Linear preparation          : PASSED"
)
print(
    "Static INT8 conversion          : PASSED"
)
print(
    "Quantized weight dtype          : torch.qint8"
)
print(
    "Deterministic QAT conversion    : PASSED"
)
print(
    "Serialization roundtrip exact   : PASSED"
)
print(
    "Source models unchanged         : PASSED"
)
print(
    "Validation access count         : 0"
)
print(
    "Test access count               : 0"
)
print(
    "QAT preflight status            : LOCKED"
)
print(
    "Ready for QAT/P25-QAT/P50-QAT   : True"
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
    "PHASE 5 QAT ENGINE AND RUNTIME PREFLIGHT LOCKED"
)
