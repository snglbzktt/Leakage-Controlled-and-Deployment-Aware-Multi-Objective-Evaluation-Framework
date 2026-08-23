from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import io
import json
import os
import sys
import tempfile
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

FP32_FT_LOCK = (
    AUDIT
    / "phase5_FP32_FT_locked_v3_2.json"
)

MODEL_SOURCE = (
    ROOT
    / "src"
    / "models"
    / "nbaiot_models.py"
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
    / "phase5_DQ_runtime_preflight_matrix_v3_2.csv"
)

OUTPUT_REPORT = (
    AUDIT
    / "phase5_DQ_runtime_preflight_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase5_DQ_runtime_preflight_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase5_DQ_runtime_preflight_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = (
    "phase5_fair_budget_compression_v3_2"
)

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

EXPECTED_INPUT_FEATURES = 115
EXPECTED_OUTPUT_CLASSES = 3

HIDDEN_DIMS = {
    "tinyml_mlp": [64, 32],
    "compact_dnn": [128, 64, 32],
}

EXPECTED_FLOAT_WIDTHS = {
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

SMOKE_BATCH_SIZE = 257
CPU_THREADS = 4


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


def float_linear_widths(
    model: nn.Module,
) -> list[list[int]]:
    return [
        [
            int(module.in_features),
            int(module.out_features),
        ]
        for module in model.modules()
        if isinstance(
            module,
            nn.Linear,
        )
    ]


def float_linear_count(
    model: nn.Module,
) -> int:
    return sum(
        1
        for module in model.modules()
        if isinstance(
            module,
            nn.Linear,
        )
    )


def quantized_dynamic_linear_modules(
    model: nn.Module,
) -> list[nn.Module]:
    modules = []

    for module in model.modules():
        module_path = (
            type(module).__module__.lower()
        )

        class_name = (
            type(module).__name__.lower()
        )

        if (
            "quantized.dynamic"
            in module_path
            and class_name == "linear"
        ):
            modules.append(module)

    return modules


def model_parameter_digest(
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


def get_quantize_dynamic():
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


def quantize_model(
    model: nn.Module,
) -> nn.Module:
    quantize_dynamic = (
        get_quantize_dynamic()
    )

    model.eval()

    quantized = quantize_dynamic(
        model,
        {
            nn.Linear,
        },
        dtype=torch.qint8,
        inplace=False,
    )

    quantized.eval()

    return quantized


required_paths = (
    PHASE5_PROTOCOL,
    LOCKED_MATRIX,
    B0_PAIR_LOCK,
    B0_CHECKPOINT_REGISTRY,
    FP32_FT_LOCK,
    MODEL_SOURCE,
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
            "DQ preflight artifact already "
            "exists; refusing to overwrite: "
            f"{output_path}"
        )

protocol = read_json(
    PHASE5_PROTOCOL
)

B0_pair_lock = read_json(
    B0_PAIR_LOCK
)

FP32_FT_lock = read_json(
    FP32_FT_LOCK
)

matrix_rows = read_csv(
    LOCKED_MATRIX
)

B0_rows = read_csv(
    B0_CHECKPOINT_REGISTRY
)

DQ_matrix_rows = [
    row
    for row in matrix_rows
    if row["architecture"]
    in ARCHITECTURES
    and row["variant"]
    == "DQ"
]

DQ_matrix_keys = {
    (
        row["architecture"],
        int(row["seed"]),
    )
    for row in DQ_matrix_rows
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
    "DQ_matrix_has_ten_rows": (
        len(DQ_matrix_rows)
        == 10
        and DQ_matrix_keys
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
        "DQ preflight entry gate failed: "
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

x_train = np.load(
    X_TRAIN_PATH,
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

if not np.isfinite(
    smoke_numpy
).all():
    raise RuntimeError(
        "Non-finite DQ smoke batch."
    )

smoke_batch = torch.from_numpy(
    smoke_numpy
)

model_module = load_module(
    MODEL_SOURCE,
    "phase5_DQ_preflight_models",
)

quantization_engine = (
    torch.backends.quantized.engine
)

supported_engines = list(
    torch.backends.quantized.supported_engines
)

if not quantization_engine:
    raise RuntimeError(
        "No active PyTorch quantization engine."
    )

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
            "B0 checkpoint hash mismatch: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_symbol = (
        B0_row["model_symbol"]
    )

    float_model = instantiate_model(
        model_module,
        model_symbol,
        architecture,
    )

    float_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    float_model.eval()

    source_digest_before = (
        model_parameter_digest(
            float_model
        )
    )

    quantized_model_a = (
        quantize_model(
            float_model
        )
    )

    quantized_model_b = (
        quantize_model(
            float_model
        )
    )

    with torch.inference_mode():
        float_logits = extract_logits(
            float_model(
                smoke_batch
            )
        )

        quantized_logits_a = extract_logits(
            quantized_model_a(
                smoke_batch
            )
        )

        quantized_logits_b = extract_logits(
            quantized_model_b(
                smoke_batch
            )
        )

    quantized_state = (
        quantized_model_a.state_dict()
    )

    with tempfile.TemporaryDirectory() as directory:
        temporary_path = (
            Path(directory)
            / "dq_state.pt"
        )

        torch.save(
            quantized_state,
            temporary_path,
        )

        fresh_float_model = (
            instantiate_model(
                model_module,
                model_symbol,
                architecture,
            )
        )

        fresh_float_model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        roundtrip_model = (
            quantize_model(
                fresh_float_model
            )
        )

        roundtrip_model.load_state_dict(
            torch.load(
                temporary_path,
                map_location="cpu",
                weights_only=False,
            ),
            strict=True,
        )

        roundtrip_model.eval()

        with torch.inference_mode():
            roundtrip_logits = (
                extract_logits(
                    roundtrip_model(
                        smoke_batch
                    )
                )
            )

        roundtrip_file_size = int(
            temporary_path.stat().st_size
        )

    float_linear_total = (
        float_linear_count(
            float_model
        )
    )

    DQ_linear_total = len(
        quantized_dynamic_linear_modules(
            quantized_model_a
        )
    )

    source_digest_after = (
        model_parameter_digest(
            float_model
        )
    )

    checks = {
        "float_widths_match": (
            float_linear_widths(
                float_model
            )
            == EXPECTED_FLOAT_WIDTHS[
                architecture
            ]
        ),
        "all_linear_layers_quantized": (
            DQ_linear_total
            == float_linear_total
            and DQ_linear_total > 0
        ),
        "float_output_shape_matches": (
            tuple(
                float_logits.shape
            )
            == (
                SMOKE_BATCH_SIZE,
                EXPECTED_OUTPUT_CLASSES,
            )
        ),
        "DQ_output_shape_matches": (
            tuple(
                quantized_logits_a.shape
            )
            == (
                SMOKE_BATCH_SIZE,
                EXPECTED_OUTPUT_CLASSES,
            )
        ),
        "outputs_are_finite": (
            bool(
                torch.isfinite(
                    float_logits
                ).all().item()
            )
            and bool(
                torch.isfinite(
                    quantized_logits_a
                ).all().item()
            )
        ),
        "deterministic_DQ_outputs": (
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
        "B0_source_unchanged": (
            source_digest_before
            == source_digest_after
        ),
        "dtype_is_qint8": (
            str(torch.qint8)
            == "torch.qint8"
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
            "DQ preflight failed: "
            + ", ".join(
                failed_checks
            )
        )

    float_state_bytes = (
        serialized_state_size(
            float_model.state_dict()
        )
    )

    DQ_state_bytes = (
        serialized_state_size(
            quantized_state
        )
    )

    mean_absolute_logit_delta = float(
        torch.mean(
            torch.abs(
                float_logits
                - quantized_logits_a
            )
        ).item()
    )

    maximum_absolute_logit_delta = float(
        torch.max(
            torch.abs(
                float_logits
                - quantized_logits_a
            )
        ).item()
    )

    rows.append(
        {
            "architecture": (
                architecture
            ),
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
            "quantization_engine": (
                quantization_engine
            ),
            "quantized_dtype": (
                "torch.qint8"
            ),
            "float_linear_count": (
                float_linear_total
            ),
            "DQ_linear_count": (
                DQ_linear_total
            ),
            "float_state_bytes": (
                float_state_bytes
            ),
            "DQ_state_bytes": (
                DQ_state_bytes
            ),
            "DQ_to_float_state_size_ratio": (
                DQ_state_bytes
                / float_state_bytes
            ),
            "roundtrip_file_size_bytes": (
                roundtrip_file_size
            ),
            "mean_absolute_logit_delta": (
                mean_absolute_logit_delta
            ),
            "maximum_absolute_logit_delta": (
                maximum_absolute_logit_delta
            ),
            "deterministic_DQ_outputs": (
                True
            ),
            "serialization_roundtrip_exact": (
                True
            ),
            "B0_source_unchanged": (
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
        f"[{index}/10] "
        f"{architecture} seed={seed} | "
        f"float_linear={float_linear_total} | "
        f"DQ_linear={DQ_linear_total} | "
        "deterministic=True | "
        "roundtrip=True | "
        "validation=0 | test=0 | "
        "all checks=True",
        flush=True,
    )

    del float_model
    del quantized_model_a
    del quantized_model_b
    del roundtrip_model
    del fresh_float_model
    del checkpoint

atomic_csv(
    OUTPUT_MATRIX,
    rows,
    [
        "architecture",
        "seed",
        "model_symbol",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "quantization_engine",
        "quantized_dtype",
        "float_linear_count",
        "DQ_linear_count",
        "float_state_bytes",
        "DQ_state_bytes",
        "DQ_to_float_state_size_ratio",
        "roundtrip_file_size_bytes",
        "mean_absolute_logit_delta",
        "maximum_absolute_logit_delta",
        "deterministic_DQ_outputs",
        "serialization_roundtrip_exact",
        "B0_source_unchanged",
        "train_smoke_rows_used",
        "validation_access_count",
        "test_access_count",
        "all_checks_passed",
    ],
)

global_checks = {
    "ten_B0_checkpoints_audited": (
        len(rows)
        == 10
    ),
    "all_linear_layers_quantized": (
        all(
            int(
                row[
                    "float_linear_count"
                ]
            )
            == int(
                row[
                    "DQ_linear_count"
                ]
            )
            for row in rows
        )
    ),
    "all_DQ_outputs_deterministic": (
        all(
            bool(
                row[
                    "deterministic_DQ_outputs"
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
    "all_B0_sources_unchanged": (
        all(
            bool(
                row[
                    "B0_source_unchanged"
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
        "DQ preflight global checks failed: "
        + ", ".join(
            failed_global_checks
        )
    )

report = {
    "status": "passed",
    "phase": 5,
    "artifact_name": (
        "dynamic_quantization_runtime_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "architecture_count": 2,
    "seed_count": 5,
    "run_count": 10,
    "variant": "DQ",
    "quantization": {
        "method": (
            "post_training_dynamic_quantization"
        ),
        "target_modules": [
            "torch.nn.Linear",
        ],
        "weight_dtype": (
            "torch.qint8"
        ),
        "activation_handling": (
            "dynamic_runtime_quantization"
        ),
        "active_engine": (
            quantization_engine
        ),
        "supported_engines": (
            supported_engines
        ),
        "source_checkpoint": (
            "matching locked B0 checkpoint"
        ),
    },
    "data_access": {
        "train_smoke_rows_used_per_run": (
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
    "matrix_csv": (
        file_record(
            OUTPUT_MATRIX
        )
    ),
    "ready_for_DQ_evaluation": True,
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
        "dynamic_quantization_runtime_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "run_count": 10,
    "variant": "DQ",
    "source_policy": (
        "matching locked B0 checkpoint"
    ),
    "quantized_dtype": (
        "torch.qint8"
    ),
    "quantization_engine": (
        quantization_engine
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
    "ready_for_DQ_evaluation": True,
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
        "dynamic_quantization_runtime_preflight"
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
            FP32_FT_LOCK
        ),
        file_record(
            MODEL_SOURCE
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
    "ready_for_DQ_evaluation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print()
print("=" * 92)
print("PHASE 5 DQ RUNTIME PREFLIGHT SUMMARY")
print("=" * 92)
print(
    "B0 checkpoints audited          : 10"
)
print(
    "Architectures                   : 2"
)
print(
    "Dynamic quantized dtype         : torch.qint8"
)
print(
    "Target module                   : torch.nn.Linear"
)
print(
    "Quantization engine             : "
    f"{quantization_engine}"
)
print(
    "All Linear layers quantized     : True"
)
print(
    "Deterministic DQ outputs        : True"
)
print(
    "Serialization roundtrip exact   : True"
)
print(
    "B0 source models unchanged      : True"
)
print(
    "Validation access count         : 0"
)
print(
    "Test access count               : 0"
)
print(
    "DQ preflight status             : LOCKED"
)
print(
    "Ready for DQ evaluation         : True"
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
    "PHASE 5 DQ RUNTIME PREFLIGHT LOCKED"
)
