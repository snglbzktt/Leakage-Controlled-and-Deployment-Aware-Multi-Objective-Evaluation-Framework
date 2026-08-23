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

QAT_LOCK = (
    AUDIT
    / "phase5_QAT_locked_v3_2.json"
)

MODEL_SOURCE = (
    ROOT
    / "src"
    / "models"
    / "nbaiot_models.py"
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
Y_TRAIN_PATH = FINAL_CACHE / "y_train.npy"

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

CALIBRATION_INDEX_CSV = (
    AUDIT
    / "phase5_PTQ_calibration_index_summary_v3_2.csv"
)

OUTPUT_MATRIX = (
    AUDIT
    / "phase5_PTQ_runtime_preflight_matrix_v3_2.csv"
)

OUTPUT_REPORT = (
    AUDIT
    / "phase5_PTQ_runtime_preflight_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase5_PTQ_runtime_preflight_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase5_PTQ_runtime_preflight_lock_manifest_v3_2.json"
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
    )
    for architecture in ARCHITECTURES
    for seed in SEEDS
}

EXPECTED_INPUT_FEATURES = 115
EXPECTED_OUTPUT_CLASSES = 3
EXPECTED_TRAIN_ROWS = 1_534_583

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

SMOKE_CALIBRATION_ROWS = 257
SMOKE_INFERENCE_ROWS = 257
CPU_THREADS = 4


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


def largest_remainder_counts(
    total: int,
    class_counts: np.ndarray,
) -> np.ndarray:
    proportions = (
        class_counts.astype(
            np.float64
        )
        / float(
            class_counts.sum()
        )
    )

    exact = proportions * total

    allocated = np.floor(
        exact
    ).astype(np.int64)

    remainder = int(
        total - allocated.sum()
    )

    order = np.argsort(
        -(
            exact
            - allocated
        ),
        kind="stable",
    )

    for index in order[
        :remainder
    ]:
        allocated[index] += 1

    if np.any(
        allocated <= 0
    ):
        raise RuntimeError(
            "Calibration allocation produced "
            "an empty class."
        )

    if int(
        allocated.sum()
    ) != total:
        raise RuntimeError(
            "Calibration allocation total mismatch."
        )

    return allocated


def generate_nested_calibration_indices(
    y_train: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    class_labels = np.asarray(
        sorted(
            np.unique(
                y_train
            ).tolist()
        ),
        dtype=np.int64,
    )

    if not np.array_equal(
        class_labels,
        np.asarray(
            [0, 1, 2],
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "Unexpected train class labels."
        )

    class_positions = {
        int(label): np.flatnonzero(
            y_train == label
        ).astype(np.int64)
        for label in class_labels
    }

    class_counts = np.asarray(
        [
            len(
                class_positions[
                    int(label)
                ]
            )
            for label in class_labels
        ],
        dtype=np.int64,
    )

    arrays: dict[
        str,
        np.ndarray,
    ] = {}

    summary_rows: list[
        dict[str, Any]
    ] = []

    for seed in SEEDS:
        rng = np.random.default_rng(
            seed
        )

        permutations = {
            label: rng.permutation(
                positions
            )
            for label, positions
            in class_positions.items()
        }

        previous_counts = np.zeros(
            len(class_labels),
            dtype=np.int64,
        )

        previous_index_set: set[int] = set()

        for size in CALIBRATION_SIZES:
            allocation = (
                largest_remainder_counts(
                    size,
                    class_counts,
                )
            )

            if np.any(
                allocation
                < previous_counts
            ):
                raise RuntimeError(
                    "Calibration allocation is "
                    "not nested by class."
                )

            selected_parts = []

            for class_offset, label in enumerate(
                class_labels
            ):
                selected_parts.append(
                    permutations[
                        int(label)
                    ][
                        :int(
                            allocation[
                                class_offset
                            ]
                        )
                    ]
                )

            selected = np.sort(
                np.concatenate(
                    selected_parts
                )
            ).astype(np.int64)

            if len(selected) != size:
                raise RuntimeError(
                    "Calibration index count mismatch."
                )

            if len(
                np.unique(
                    selected
                )
            ) != size:
                raise RuntimeError(
                    "Duplicate calibration index."
                )

            current_set = set(
                selected.tolist()
            )

            if not previous_index_set.issubset(
                current_set
            ):
                raise RuntimeError(
                    "Calibration sets are not nested."
                )

            key = (
                f"seed_{seed}_"
                f"size_{size}"
            )

            arrays[key] = selected

            observed_counts = np.bincount(
                np.asarray(
                    y_train[
                        selected
                    ],
                    dtype=np.int64,
                ),
                minlength=3,
            )

            summary_rows.append(
                {
                    "seed": seed,
                    "calibration_size": size,
                    "array_key": key,
                    "benign_count": int(
                        observed_counts[0]
                    ),
                    "gafgyt_count": int(
                        observed_counts[1]
                    ),
                    "mirai_count": int(
                        observed_counts[2]
                    ),
                    "minimum_index": int(
                        selected.min()
                    ),
                    "maximum_index": int(
                        selected.max()
                    ),
                    "unique_index_count": int(
                        len(
                            np.unique(
                                selected
                            )
                        )
                    ),
                    "nested_with_previous": True,
                }
            )

            previous_counts = allocation
            previous_index_set = current_set

    return (
        arrays,
        summary_rows,
    )


def prepare_calibrate_convert(
    source_model: nn.Module,
    backend: str,
    calibration_tensor: torch.Tensor,
    PTQ_engine_module: ModuleType,
) -> tuple[
    nn.Module,
    nn.Module,
]:
    prepared = (
        PTQ_engine_module
        .prepare_ptq_model(
            source_model,
            backend,
        )
    )

    with torch.inference_mode():
        prepared(
            calibration_tensor
        )

    converted = (
        PTQ_engine_module
        .convert_ptq_model(
            prepared
        )
    )

    return (
        prepared,
        converted,
    )


required_paths = (
    PHASE5_PROTOCOL,
    LOCKED_MATRIX,
    B0_PAIR_LOCK,
    B0_CHECKPOINT_REGISTRY,
    QAT_LOCK,
    MODEL_SOURCE,
    PTQ_ENGINE_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    SCALER_NPZ,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    CALIBRATION_INDEX_NPZ,
    CALIBRATION_INDEX_CSV,
    OUTPUT_MATRIX,
    OUTPUT_REPORT,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "PTQ preflight artifact already "
            "exists; refusing to overwrite: "
            f"{output_path}"
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

matrix_rows = read_csv(
    LOCKED_MATRIX
)

B0_rows = read_csv(
    B0_CHECKPOINT_REGISTRY
)

PTQ_matrix_rows = [
    row
    for row in matrix_rows
    if row["architecture"]
    in ARCHITECTURES
    and row["variant"]
    == "PTQ"
]

PTQ_matrix_keys = {
    (
        row["architecture"],
        int(row["seed"]),
    )
    for row in PTQ_matrix_rows
}

B0_lookup = {
    (
        row["architecture"],
        int(row["seed"]),
    ): row
    for row in B0_rows
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
    "PTQ_matrix_has_ten_rows": (
        len(PTQ_matrix_rows)
        == 10
        and PTQ_matrix_keys
        == EXPECTED_KEYS
    ),
    "B0_registry_has_ten_rows": (
        set(
            B0_lookup.keys()
        )
        == EXPECTED_KEYS
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
        "PTQ preflight entry gate failed: "
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

y_train = np.asarray(
    np.load(
        Y_TRAIN_PATH,
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
    or y_train.shape
    != (
        EXPECTED_TRAIN_ROWS,
    )
):
    raise RuntimeError(
        "PTQ train cache shape mismatch."
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

if (
    mean64.shape
    != (
        EXPECTED_INPUT_FEATURES,
    )
    or scale64.shape
    != (
        EXPECTED_INPUT_FEATURES,
    )
):
    raise RuntimeError(
        "PTQ scaler shape mismatch."
    )

(
    calibration_arrays,
    calibration_summary_rows,
) = generate_nested_calibration_indices(
    y_train
)

atomic_npz(
    CALIBRATION_INDEX_NPZ,
    **calibration_arrays,
)

atomic_csv(
    CALIBRATION_INDEX_CSV,
    calibration_summary_rows,
    [
        "seed",
        "calibration_size",
        "array_key",
        "benign_count",
        "gafgyt_count",
        "mirai_count",
        "minimum_index",
        "maximum_index",
        "unique_index_count",
        "nested_with_previous",
    ],
)

model_module = load_module(
    MODEL_SOURCE,
    "phase5_PTQ_preflight_models",
)

PTQ_engine_module = load_module(
    PTQ_ENGINE_PATH,
    "phase5_PTQ_engine",
)

if (
    PTQ_engine_module.ENGINE_VERSION
    != PTQ_ENGINE_VERSION
):
    raise RuntimeError(
        "PTQ-engine version mismatch."
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

representative_row = B0_lookup[
    (
        "tinyml_mlp",
        42,
    )
]

representative_checkpoint_path = Path(
    representative_row[
        "checkpoint_path"
    ]
)

representative_checkpoint = torch.load(
    representative_checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

representative_model = instantiate_model(
    model_module,
    representative_row[
        "model_symbol"
    ],
    "tinyml_mlp",
)

representative_model.load_state_dict(
    representative_checkpoint[
        "model_state_dict"
    ],
    strict=True,
)

representative_model.eval()

representative_indices = (
    calibration_arrays[
        "seed_42_size_4096"
    ][
        :SMOKE_CALIBRATION_ROWS
    ]
)

representative_calibration_tensor = (
    torch.from_numpy(
        transform_rows(
            x_train[
                representative_indices
            ],
            mean64,
            scale64,
        )
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

        prepared, converted = (
            prepare_calibrate_convert(
                representative_model,
                candidate,
                representative_calibration_tensor,
                PTQ_engine_module,
            )
        )

        with torch.inference_mode():
            output = (
                PTQ_engine_module
                .extract_logits(
                    converted(
                        representative_calibration_tensor
                    )
                )
            )

        expected_linear_count = (
            PTQ_engine_module
            .float_linear_count(
                representative_model
            )
        )

        converted_linear_count = len(
            PTQ_engine_module
            .static_quantized_linear_modules(
                converted
            )
        )

        passed = (
            tuple(output.shape)
            == (
                SMOKE_CALIBRATION_ROWS,
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
        "No supported PTQ backend passed "
        "the local runtime smoke test."
    )

torch.backends.quantized.engine = (
    selected_backend
)

del representative_checkpoint
del representative_model

print("=" * 92)
print("PHASE 5 PTQ ENGINE, RUNTIME AND CALIBRATION PREFLIGHT")
print("=" * 92)
print(
    "B0 configurations               : 10"
)
print(
    "Calibration sizes               : "
    f"{list(CALIBRATION_SIZES)}"
)
print(
    "Calibration plans               : 20"
)
print(
    "Nested stratified sets          : True"
)
print(
    "Selected quantization backend   : "
    f"{selected_backend}"
)
print(
    "PTQ engine version              : "
    f"{PTQ_ENGINE_VERSION}"
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
        ),
    ),
    start=1,
):
    architecture, seed = key

    B0_row = B0_lookup[
        key
    ]

    checkpoint_path = Path(
        B0_row["checkpoint_path"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint_hash = sha256_file(
        checkpoint_path
    )

    if (
        checkpoint_hash
        != B0_row[
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
        B0_row[
            "model_symbol"
        ]
    )

    source_model = instantiate_model(
        model_module,
        model_symbol,
        architecture,
    )

    source_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    source_model.eval()

    if (
        PTQ_engine_module
        .float_linear_widths(
            source_model
        )
        != EXPECTED_WIDTHS[
            architecture
        ]
    ):
        raise RuntimeError(
            "PTQ source topology mismatch."
        )

    source_digest_before = (
        state_digest(
            source_model
        )
    )

    calibration_key = (
        f"seed_{seed}_size_4096"
    )

    smoke_indices = (
        calibration_arrays[
            calibration_key
        ][
            :SMOKE_CALIBRATION_ROWS
        ]
    )

    calibration_tensor = torch.from_numpy(
        transform_rows(
            x_train[
                smoke_indices
            ],
            mean64,
            scale64,
        )
    )

    inference_indices = (
        calibration_arrays[
            calibration_key
        ][
            SMOKE_CALIBRATION_ROWS:
            SMOKE_CALIBRATION_ROWS
            + SMOKE_INFERENCE_ROWS
        ]
    )

    if len(
        inference_indices
    ) != SMOKE_INFERENCE_ROWS:
        raise RuntimeError(
            "PTQ smoke inference index "
            "count mismatch."
        )

    inference_tensor = torch.from_numpy(
        transform_rows(
            x_train[
                inference_indices
            ],
            mean64,
            scale64,
        )
    )

    prepared_a, converted_a = (
        prepare_calibrate_convert(
            source_model,
            selected_backend,
            calibration_tensor,
            PTQ_engine_module,
        )
    )

    prepared_b, converted_b = (
        prepare_calibrate_convert(
            source_model,
            selected_backend,
            calibration_tensor,
            PTQ_engine_module,
        )
    )

    with torch.inference_mode():
        logits_a = (
            PTQ_engine_module
            .extract_logits(
                converted_a(
                    inference_tensor
                )
            )
            .detach()
            .cpu()
        )

        logits_b = (
            PTQ_engine_module
            .extract_logits(
                converted_b(
                    inference_tensor
                )
            )
            .detach()
            .cpu()
        )

    float_linear_total = (
        PTQ_engine_module
        .float_linear_count(
            source_model
        )
    )

    observer_total = (
        PTQ_engine_module
        .observer_count(
            prepared_a
        )
    )

    quantized_linear_total = len(
        PTQ_engine_module
        .static_quantized_linear_modules(
            converted_a
        )
    )

    weight_dtypes = (
        PTQ_engine_module
        .quantized_weight_dtypes(
            converted_a
        )
    )

    with tempfile.TemporaryDirectory() as directory:
        state_path = (
            Path(directory)
            / "ptq_state.pt"
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
                PTQ_engine_module
                .extract_logits(
                    converted_b(
                        inference_tensor
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

    checks = {
        "source_topology_matches": (
            PTQ_engine_module
            .float_linear_widths(
                source_model
            )
            == EXPECTED_WIDTHS[
                architecture
            ]
        ),
        "observers_present": (
            observer_total > 0
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
                dtype == "torch.qint8"
                for dtype in weight_dtypes
            )
        ),
        "outputs_finite": (
            bool(
                torch.isfinite(
                    logits_a
                ).all().item()
            )
        ),
        "deterministic_PTQ_conversion": (
            torch.equal(
                logits_a,
                logits_b,
            )
        ),
        "serialization_roundtrip_exact": (
            torch.equal(
                logits_a,
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
            f"{architecture} seed {seed} "
            "PTQ preflight failed: "
            + ", ".join(
                failed_checks
            )
        )

    float_state_bytes = (
        serialized_state_size(
            source_model.state_dict()
        )
    )

    INT8_state_bytes = (
        serialized_state_size(
            converted_a.state_dict()
        )
    )

    rows.append(
        {
            "architecture": architecture,
            "seed": seed,
            "model_symbol": (
                model_symbol
            ),
            "source_checkpoint_path": (
                str(
                    checkpoint_path
                )
            ),
            "source_checkpoint_sha256": (
                checkpoint_hash
            ),
            "PTQ_backend": (
                selected_backend
            ),
            "calibration_plan_sizes": (
                json.dumps(
                    list(
                        CALIBRATION_SIZES
                    ),
                    separators=(
                        ",",
                        ":",
                    ),
                )
            ),
            "smoke_calibration_rows": (
                SMOKE_CALIBRATION_ROWS
            ),
            "smoke_inference_rows": (
                SMOKE_INFERENCE_ROWS
            ),
            "float_linear_count": (
                float_linear_total
            ),
            "observer_count": (
                observer_total
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
            "INT8_state_bytes": (
                INT8_state_bytes
            ),
            "INT8_to_float_state_size_ratio": (
                INT8_state_bytes
                / float_state_bytes
            ),
            "roundtrip_file_size_bytes": (
                roundtrip_file_size
            ),
            "deterministic_PTQ_conversion": True,
            "serialization_roundtrip_exact": True,
            "source_model_unchanged": True,
            "validation_access_count": 0,
            "test_access_count": 0,
            "all_checks_passed": True,
        }
    )

    print(
        f"[{index}/10] "
        f"{architecture} seed={seed} PTQ | "
        f"observers={observer_total} | "
        f"INT8_linear={quantized_linear_total} | "
        "deterministic=True | "
        "roundtrip=True | "
        "validation=0 | test=0 | "
        "all checks=True",
        flush=True,
    )

    del source_model
    del checkpoint
    del prepared_a
    del prepared_b
    del converted_a
    del converted_b

atomic_csv(
    OUTPUT_MATRIX,
    rows,
    [
        "architecture",
        "seed",
        "model_symbol",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "PTQ_backend",
        "calibration_plan_sizes",
        "smoke_calibration_rows",
        "smoke_inference_rows",
        "float_linear_count",
        "observer_count",
        "converted_quantized_linear_count",
        "quantized_weight_dtypes",
        "float_state_bytes",
        "INT8_state_bytes",
        "INT8_to_float_state_size_ratio",
        "roundtrip_file_size_bytes",
        "deterministic_PTQ_conversion",
        "serialization_roundtrip_exact",
        "source_model_unchanged",
        "validation_access_count",
        "test_access_count",
        "all_checks_passed",
    ],
)

global_checks = {
    "ten_B0_configurations_audited": (
        len(rows)
        == 10
    ),
    "twenty_calibration_plans_created": (
        len(
            calibration_summary_rows
        )
        == 20
    ),
    "all_calibration_sizes_present": (
        {
            int(
                row[
                    "calibration_size"
                ]
            )
            for row
            in calibration_summary_rows
        }
        == set(
            CALIBRATION_SIZES
        )
    ),
    "all_calibration_sets_nested": (
        all(
            bool(
                row[
                    "nested_with_previous"
                ]
            )
            for row
            in calibration_summary_rows
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
                    "deterministic_PTQ_conversion"
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
        "PTQ preflight global checks failed: "
        + ", ".join(
            failed_global_checks
        )
    )

report = {
    "status": "passed",
    "phase": 5,
    "artifact_name": (
        "PTQ_engine_runtime_and_"
        "calibration_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "PTQ_engine_version": (
        PTQ_ENGINE_VERSION
    ),
    "verified_at_utc": utc_now(),
    "architecture_count": 2,
    "seed_count": 5,
    "configuration_count": 10,
    "calibration_sizes": list(
        CALIBRATION_SIZES
    ),
    "calibration_plan_count": 20,
    "calibration_policy": {
        "source_split": "train_only",
        "sampling": (
            "deterministic_stratified_nested"
        ),
        "seed_specific": True,
        "shared_between_architectures_for_same_seed": True,
        "validation_used_for_calibration": False,
        "test_used_for_calibration": False,
    },
    "PTQ_policy": {
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
        "target_modules": [
            "torch.nn.Linear",
        ],
        "source_checkpoint": (
            "matching locked B0 checkpoint"
        ),
    },
    "data_access": {
        "train_used_for_calibration_plan": True,
        "validation_access_count": 0,
        "test_access_count": 0,
    },
    "entry_checks": (
        entry_checks
    ),
    "global_checks": (
        global_checks
    ),
    "PTQ_engine_source": (
        file_record(
            PTQ_ENGINE_PATH
        )
    ),
    "calibration_index_npz": (
        file_record(
            CALIBRATION_INDEX_NPZ
        )
    ),
    "calibration_index_summary_csv": (
        file_record(
            CALIBRATION_INDEX_CSV
        )
    ),
    "matrix_csv": (
        file_record(
            OUTPUT_MATRIX
        )
    ),
    "ready_for_PTQ_validation_sweep": True,
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
        "PTQ_engine_runtime_and_"
        "calibration_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "PTQ_engine_version": (
        PTQ_ENGINE_VERSION
    ),
    "locked_at_utc": utc_now(),
    "configuration_count": 10,
    "calibration_plan_count": 20,
    "calibration_sizes": list(
        CALIBRATION_SIZES
    ),
    "selected_backend": (
        selected_backend
    ),
    "quantized_weight_dtype": (
        "torch.qint8"
    ),
    "PTQ_engine_source": str(
        PTQ_ENGINE_PATH
    ),
    "PTQ_engine_source_sha256": (
        sha256_file(
            PTQ_ENGINE_PATH
        )
    ),
    "calibration_index_npz": str(
        CALIBRATION_INDEX_NPZ
    ),
    "calibration_index_npz_sha256": (
        sha256_file(
            CALIBRATION_INDEX_NPZ
        )
    ),
    "calibration_index_summary_csv": str(
        CALIBRATION_INDEX_CSV
    ),
    "calibration_index_summary_csv_sha256": (
        sha256_file(
            CALIBRATION_INDEX_CSV
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
    "ready_for_PTQ_validation_sweep": True,
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
        "PTQ_engine_runtime_and_"
        "calibration_preflight"
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
            QAT_LOCK
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
    ],
    "generated_artifacts": [
        file_record(
            CALIBRATION_INDEX_NPZ
        ),
        file_record(
            CALIBRATION_INDEX_CSV
        ),
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
    "configuration_count": 10,
    "calibration_plan_count": 20,
    "validation_access_count": 0,
    "test_access_count": 0,
    "ready_for_PTQ_validation_sweep": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print()
print("=" * 92)
print("PHASE 5 PTQ PREFLIGHT SUMMARY")
print("=" * 92)
print(
    "B0 configurations audited       : 10"
)
print(
    "Calibration plans created       : 20"
)
print(
    "Calibration sizes               : "
    f"{list(CALIBRATION_SIZES)}"
)
print(
    "Nested stratified calibration   : PASSED"
)
print(
    "Selected quantization backend   : "
    f"{selected_backend}"
)
print(
    "Observer insertion              : PASSED"
)
print(
    "Static INT8 conversion          : PASSED"
)
print(
    "Quantized weight dtype          : torch.qint8"
)
print(
    "Deterministic PTQ conversion    : PASSED"
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
    "PTQ preflight status            : LOCKED"
)
print(
    "Ready for validation sweep      : True"
)
print(
    "Calibration indices             : "
    f"{CALIBRATION_INDEX_NPZ}"
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
    "PHASE 5 PTQ ENGINE, RUNTIME AND CALIBRATION PREFLIGHT LOCKED"
)
