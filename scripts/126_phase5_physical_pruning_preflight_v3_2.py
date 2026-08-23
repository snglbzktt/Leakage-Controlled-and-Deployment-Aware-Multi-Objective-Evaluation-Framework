from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
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

B0_PAIR_LOCK = (
    AUDIT / "phase5_B0_pair_locked_v3_2.json"
)

B0_PAIR_LOCK_MANIFEST = (
    AUDIT / "phase5_B0_pair_lock_manifest_v3_2.json"
)

B0_CHECKPOINT_REGISTRY = (
    AUDIT / "phase5_B0_checkpoint_registry_v3_2.csv"
)

MODEL_SOURCE = (
    ROOT / "src" / "models" / "nbaiot_models.py"
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

OUTPUT_REPORT = (
    AUDIT / "phase5_physical_pruning_preflight_v3_2.json"
)

OUTPUT_MODULES_CSV = (
    AUDIT / "phase5_B0_leaf_module_inventory_v3_2.csv"
)

OUTPUT_PLAN_CSV = (
    AUDIT / "phase5_physical_pruning_source_plan_v3_2.csv"
)

OUTPUT_LOCK = (
    AUDIT / "phase5_physical_pruning_preflight_locked_v3_2.json"
)

PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"

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

PRUNING_RATIOS = (
    0.25,
    0.50,
)

EXPECTED_INPUT_FEATURES = 115
EXPECTED_OUTPUT_CLASSES = 3
DUMMY_BATCH_SIZE = 32
CPU_THREADS = 4

EXPECTED_LINEAR_WIDTHS = {
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

HIDDEN_DIMS = {
    "tinyml_mlp": [64, 32],
    "compact_dnn": [128, 64, 32],
}

SUPPORTED_STATELESS_MODULES = (
    nn.ReLU,
    nn.ReLU6,
    nn.GELU,
    nn.SiLU,
    nn.LeakyReLU,
    nn.ELU,
    nn.SELU,
    nn.Tanh,
    nn.Sigmoid,
    nn.Dropout,
    nn.Dropout1d,
    nn.Identity,
    nn.Flatten,
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


def load_model_module(
    path: Path,
) -> ModuleType:
    module_name = (
        "phase5_pruning_preflight_models"
    )

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
            "Could not create model-source "
            "import specification."
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
    module: ModuleType,
    symbol_name: str,
    architecture: str,
) -> nn.Module:
    if not hasattr(
        module,
        symbol_name,
    ):
        raise RuntimeError(
            "Model symbol is missing: "
            f"{symbol_name}"
        )

    candidate = getattr(
        module,
        symbol_name,
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
            "Model symbol did not create "
            "torch.nn.Module."
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
        "Model output does not contain logits."
    )


def linear_widths(
    model: nn.Module,
) -> list[list[int]]:
    return [
        [
            int(layer.in_features),
            int(layer.out_features),
        ]
        for layer in model.modules()
        if isinstance(
            layer,
            nn.Linear,
        )
    ]


def parameter_count(
    model: nn.Module,
) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )


def linear_macs_per_sample(
    model: nn.Module,
) -> int:
    return int(
        sum(
            layer.in_features
            * layer.out_features
            for layer in model.modules()
            if isinstance(
                layer,
                nn.Linear,
            )
        )
    )


def leaf_modules(
    model: nn.Module,
) -> list[
    tuple[str, nn.Module]
]:
    return [
        (name, child)
        for name, child
        in model.named_modules()
        if name
        and len(
            list(
                child.children()
            )
        )
        == 0
    ]


def tensor_shape(
    value: Any,
) -> list[int] | None:
    if isinstance(
        value,
        torch.Tensor,
    ):
        return list(value.shape)

    if isinstance(
        value,
        (tuple, list),
    ):
        for item in value:
            if isinstance(
                item,
                torch.Tensor,
            ):
                return list(item.shape)

    if isinstance(
        value,
        dict,
    ):
        for item in value.values():
            if isinstance(
                item,
                torch.Tensor,
            ):
                return list(item.shape)

    return None


def inspect_forward_shapes(
    model: nn.Module,
    batch: torch.Tensor,
) -> list[dict[str, Any]]:
    records: list[
        dict[str, Any]
    ] = []

    handles = []

    for name, child in leaf_modules(
        model
    ):
        def hook(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
            module_name: str = name,
        ) -> None:
            input_shape = None

            for item in inputs:
                input_shape = tensor_shape(
                    item
                )

                if input_shape is not None:
                    break

            records.append(
                {
                    "module_name": (
                        module_name
                    ),
                    "module_type": (
                        type(module).__name__
                    ),
                    "input_shape": (
                        input_shape
                    ),
                    "output_shape": (
                        tensor_shape(
                            output
                        )
                    ),
                }
            )

        handles.append(
            child.register_forward_hook(
                hook
            )
        )

    try:
        model.eval()

        with torch.inference_mode():
            logits = extract_logits(
                model(batch)
            )

        if tuple(
            logits.shape
        ) != (
            len(batch),
            EXPECTED_OUTPUT_CLASSES,
        ):
            raise RuntimeError(
                "Model output shape mismatch: "
                f"{tuple(logits.shape)}"
            )

        if not torch.isfinite(
            logits
        ).all():
            raise RuntimeError(
                "Model produced non-finite logits."
            )

    finally:
        for handle in handles:
            handle.remove()

    return records


def module_is_supported(
    module: nn.Module,
) -> bool:
    if isinstance(
        module,
        nn.Linear,
    ):
        return True

    if isinstance(
        module,
        SUPPORTED_STATELESS_MODULES,
    ):
        return True

    return False


def module_has_parameters(
    module: nn.Module,
) -> bool:
    return any(
        parameter.numel() > 0
        for parameter in module.parameters(
            recurse=False
        )
    )


def retained_width(
    original_width: int,
    pruning_ratio: float,
) -> int:
    return max(
        1,
        round(
            (
                1.0
                - pruning_ratio
            )
            * original_width
        ),
    )


def compacted_linear_widths(
    original_widths: list[list[int]],
    pruning_ratio: float,
) -> list[list[int]]:
    hidden = [
        retained_width(
            layer[1],
            pruning_ratio,
        )
        for layer in original_widths[
            :-1
        ]
    ]

    widths: list[
        list[int]
    ] = []

    previous = (
        original_widths[0][0]
    )

    for width in hidden:
        widths.append(
            [
                previous,
                width,
            ]
        )

        previous = width

    widths.append(
        [
            previous,
            original_widths[-1][1],
        ]
    )

    return widths


def linear_parameter_count_from_widths(
    widths: list[list[int]],
) -> int:
    return int(
        sum(
            in_features
            * out_features
            + out_features
            for in_features, out_features
            in widths
        )
    )


def linear_macs_from_widths(
    widths: list[list[int]],
) -> int:
    return int(
        sum(
            in_features
            * out_features
            for in_features, out_features
            in widths
        )
    )


required_paths = (
    PHASE5_PROTOCOL,
    B0_PAIR_LOCK,
    B0_PAIR_LOCK_MANIFEST,
    B0_CHECKPOINT_REGISTRY,
    MODEL_SOURCE,
    X_TRAIN_PATH,
    SCALER_NPZ,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_REPORT,
    OUTPUT_MODULES_CSV,
    OUTPUT_PLAN_CSV,
    OUTPUT_LOCK,
):
    if output_path.exists():
        raise FileExistsError(
            "Physical-pruning preflight "
            "artifact already exists; "
            f"refusing to overwrite: {output_path}"
        )

protocol = read_json(
    PHASE5_PROTOCOL
)

b0_pair_lock = read_json(
    B0_PAIR_LOCK
)

b0_pair_lock_manifest = read_json(
    B0_PAIR_LOCK_MANIFEST
)

registry_rows = read_csv(
    B0_CHECKPOINT_REGISTRY
)

entry_checks = {
    "phase5_protocol_locked": (
        protocol.get("status")
        == "locked"
        and protocol.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
    ),
    "physical_pruning_required": (
        protocol["pruning"][
            "type"
        ]
        == (
            "physical_structured_"
            "hidden_unit_pruning"
        )
    ),
    "mask_only_pruning_disallowed": (
        protocol["pruning"][
            "mask_only_pruning_allowed"
        ]
        is False
    ),
    "pruning_ratios_match": (
        list(
            protocol["pruning"][
                "pruning_ratios"
            ]
        )
        == [
            0.25,
            0.50,
        ]
    ),
    "B0_pair_locked": (
        b0_pair_lock.get("status")
        == "locked"
        and b0_pair_lock.get(
            "all_checks_passed"
        )
        is True
        and b0_pair_lock.get(
            "ready_for_branch_source_generation"
        )
        is True
    ),
    "B0_registry_hash_matches": (
        b0_pair_lock.get(
            "checkpoint_registry_sha256"
        )
        == sha256_file(
            B0_CHECKPOINT_REGISTRY
        )
    ),
    "B0_lock_manifest_locked": (
        b0_pair_lock_manifest.get(
            "status"
        )
        == "locked"
        and b0_pair_lock_manifest.get(
            "all_checks_passed"
        )
        is True
    ),
    "registry_has_ten_rows": (
        len(registry_rows)
        == 10
    ),
    "registry_architectures_match": (
        {
            row["architecture"]
            for row in registry_rows
        }
        == set(ARCHITECTURES)
    ),
    "registry_seed_sets_match": all(
        sorted(
            int(row["seed"])
            for row in registry_rows
            if row["architecture"]
            == architecture
        )
        == list(SEEDS)
        for architecture in ARCHITECTURES
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
        "Physical-pruning preflight "
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

x_train = np.load(
    X_TRAIN_PATH,
    mmap_mode="r",
)

with np.load(
    SCALER_NPZ
) as values:
    scaler_mean64 = np.array(
        values["mean_float64"],
        copy=True,
    )

    scaler_scale64 = np.array(
        values["scale_float64"],
        copy=True,
    )

dummy_source = np.asarray(
    x_train[
        :DUMMY_BATCH_SIZE
    ],
    dtype=np.float64,
)

dummy_scaled = (
    (
        dummy_source
        - scaler_mean64
    )
    / scaler_scale64
).astype(np.float32)

if not np.isfinite(
    dummy_scaled
).all():
    raise RuntimeError(
        "Non-finite value in train-only "
        "dummy batch."
    )

dummy_batch = torch.from_numpy(
    dummy_scaled
)

model_module = load_model_module(
    MODEL_SOURCE
)

module_rows: list[
    dict[str, Any]
] = []

run_reports: list[
    dict[str, Any]
] = []

signature_by_architecture: dict[
    str,
    list[
        tuple[
            str,
            str,
            bool,
        ]
    ],
] = {}

for index, registry_row in enumerate(
    sorted(
        registry_rows,
        key=lambda row: (
            row["architecture"],
            int(row["seed"]),
        ),
    ),
    start=1,
):
    architecture = (
        registry_row[
            "architecture"
        ]
    )

    seed = int(
        registry_row["seed"]
    )

    model_symbol = (
        registry_row[
            "model_symbol"
        ]
    )

    checkpoint_path = Path(
        registry_row[
            "checkpoint_path"
        ]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    if (
        sha256_file(
            checkpoint_path
        )
        != registry_row[
            "checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "Checkpoint hash mismatch: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
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

    observed_widths = linear_widths(
        model
    )

    leaves = leaf_modules(
        model
    )

    unsupported = [
        {
            "name": name,
            "type": (
                type(child).__name__
            ),
        }
        for name, child in leaves
        if not module_is_supported(
            child
        )
    ]

    parameterized_non_linear = [
        {
            "name": name,
            "type": (
                type(child).__name__
            ),
        }
        for name, child in leaves
        if not isinstance(
            child,
            nn.Linear,
        )
        and module_has_parameters(
            child
        )
    ]

    forward_shapes = (
        inspect_forward_shapes(
            model,
            dummy_batch,
        )
    )

    signature = [
        (
            name,
            type(child).__name__,
            module_has_parameters(
                child
            ),
        )
        for name, child in leaves
    ]

    if architecture not in (
        signature_by_architecture
    ):
        signature_by_architecture[
            architecture
        ] = signature

    signature_matches = (
        signature
        == signature_by_architecture[
            architecture
        ]
    )

    checks = {
        "linear_widths_match_expected": (
            observed_widths
            == EXPECTED_LINEAR_WIDTHS[
                architecture
            ]
        ),
        "checkpoint_identity_matches": (
            checkpoint.get(
                "architecture"
            )
            == architecture
            and int(
                checkpoint.get(
                    "seed"
                )
            )
            == seed
            and checkpoint.get(
                "variant"
            )
            == "B0"
            and checkpoint.get(
                "model_symbol"
            )
            == model_symbol
        ),
        "all_leaf_modules_supported": (
            len(unsupported)
            == 0
        ),
        "no_parameterized_non_linear_leaf": (
            len(
                parameterized_non_linear
            )
            == 0
        ),
        "leaf_signature_matches_architecture": (
            signature_matches
        ),
        "forward_trace_not_empty": (
            len(
                forward_shapes
            )
            > 0
        ),
        "checkpoint_parameter_count_matches": (
            parameter_count(
                model
            )
            == int(
                registry_row[
                    "parameter_count"
                ]
            )
        ),
        "checkpoint_macs_match": (
            linear_macs_per_sample(
                model
            )
            == int(
                registry_row[
                    "linear_macs_per_sample"
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
            "pruning preflight failed: "
            + ", ".join(
                failed_checks
            )
            + " | unsupported="
            + json.dumps(
                unsupported,
                ensure_ascii=True,
            )
            + " | parameterized_non_linear="
            + json.dumps(
                parameterized_non_linear,
                ensure_ascii=True,
            )
        )

    shape_lookup = {
        (
            record[
                "module_name"
            ],
            record[
                "module_type"
            ],
        ): record
        for record in forward_shapes
    }

    for order, (
        name,
        child,
    ) in enumerate(
        leaves,
        start=1,
    ):
        shape_record = (
            shape_lookup.get(
                (
                    name,
                    type(child).__name__,
                ),
                {},
            )
        )

        module_rows.append(
            {
                "architecture": (
                    architecture
                ),
                "seed": seed,
                "module_order": order,
                "module_name": name,
                "module_type": (
                    type(child).__name__
                ),
                "has_direct_parameters": (
                    module_has_parameters(
                        child
                    )
                ),
                "parameter_count": int(
                    sum(
                        parameter.numel()
                        for parameter
                        in child.parameters(
                            recurse=False
                        )
                    )
                ),
                "input_shape": (
                    json.dumps(
                        shape_record.get(
                            "input_shape"
                        ),
                        separators=(
                            ",",
                            ":",
                        ),
                    )
                ),
                "output_shape": (
                    json.dumps(
                        shape_record.get(
                            "output_shape"
                        ),
                        separators=(
                            ",",
                            ":",
                        ),
                    )
                ),
                "supported_by_planned_engine": True,
            }
        )

    run_reports.append(
        {
            "architecture": (
                architecture
            ),
            "seed": seed,
            "model_symbol": (
                model_symbol
            ),
            "checkpoint": file_record(
                checkpoint_path
            ),
            "linear_widths": (
                observed_widths
            ),
            "parameter_count": (
                parameter_count(
                    model
                )
            ),
            "linear_macs_per_sample": (
                linear_macs_per_sample(
                    model
                )
            ),
            "leaf_module_count": (
                len(leaves)
            ),
            "leaf_signature": [
                {
                    "name": name,
                    "type": module_type,
                    "has_direct_parameters": (
                        has_parameters
                    ),
                }
                for name, module_type, has_parameters
                in signature
            ],
            "unsupported_modules": (
                unsupported
            ),
            "parameterized_non_linear_modules": (
                parameterized_non_linear
            ),
            "checks": checks,
            "all_checks_passed": True,
        }
    )

    print(
        f"[{index}/10] "
        f"{architecture} seed={seed} | "
        f"leaf_modules={len(leaves)} | "
        f"linear_widths={observed_widths} | "
        "all checks=True",
        flush=True,
    )

    del model
    del checkpoint

atomic_csv(
    OUTPUT_MODULES_CSV,
    module_rows,
    [
        "architecture",
        "seed",
        "module_order",
        "module_name",
        "module_type",
        "has_direct_parameters",
        "parameter_count",
        "input_shape",
        "output_shape",
        "supported_by_planned_engine",
    ],
)

plan_rows: list[
    dict[str, Any]
] = []

for architecture in ARCHITECTURES:
    original_widths = (
        EXPECTED_LINEAR_WIDTHS[
            architecture
        ]
    )

    original_parameters = (
        linear_parameter_count_from_widths(
            original_widths
        )
    )

    original_macs = (
        linear_macs_from_widths(
            original_widths
        )
    )

    for ratio in PRUNING_RATIOS:
        compact_widths = (
            compacted_linear_widths(
                original_widths,
                ratio,
            )
        )

        compact_parameters = (
            linear_parameter_count_from_widths(
                compact_widths
            )
        )

        compact_macs = (
            linear_macs_from_widths(
                compact_widths
            )
        )

        for seed in SEEDS:
            source_row = next(
                row
                for row in registry_rows
                if row[
                    "architecture"
                ]
                == architecture
                and int(
                    row["seed"]
                )
                == seed
            )

            ratio_name = (
                "P25"
                if math.isclose(
                    ratio,
                    0.25,
                )
                else "P50"
            )

            plan_rows.append(
                {
                    "architecture": (
                        architecture
                    ),
                    "seed": seed,
                    "source_variant": (
                        "B0"
                    ),
                    "source_checkpoint_path": (
                        source_row[
                            "checkpoint_path"
                        ]
                    ),
                    "source_checkpoint_sha256": (
                        source_row[
                            "checkpoint_sha256"
                        ]
                    ),
                    "pruning_variant": (
                        f"{ratio_name}-noFT"
                    ),
                    "pruning_ratio": (
                        ratio
                    ),
                    "original_linear_widths": (
                        json.dumps(
                            original_widths,
                            separators=(
                                ",",
                                ":",
                            ),
                        )
                    ),
                    "compacted_linear_widths": (
                        json.dumps(
                            compact_widths,
                            separators=(
                                ",",
                                ":",
                            ),
                        )
                    ),
                    "original_parameter_count": (
                        original_parameters
                    ),
                    "expected_compacted_parameter_count": (
                        compact_parameters
                    ),
                    "expected_parameter_reduction_fraction": (
                        1.0
                        - compact_parameters
                        / original_parameters
                    ),
                    "original_linear_macs_per_sample": (
                        original_macs
                    ),
                    "expected_compacted_linear_macs_per_sample": (
                        compact_macs
                    ),
                    "expected_mac_reduction_fraction": (
                        1.0
                        - compact_macs
                        / original_macs
                    ),
                    "selection_criterion": (
                        "deterministic_hidden_unit_"
                        "importance_to_be_locked_"
                        "in_engine_step"
                    ),
                    "source_is_independent_direct_B0": (
                        True
                    ),
                    "status": (
                        "planned_engine_not_yet_built"
                    ),
                }
            )

atomic_csv(
    OUTPUT_PLAN_CSV,
    plan_rows,
    [
        "architecture",
        "seed",
        "source_variant",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "pruning_variant",
        "pruning_ratio",
        "original_linear_widths",
        "compacted_linear_widths",
        "original_parameter_count",
        "expected_compacted_parameter_count",
        "expected_parameter_reduction_fraction",
        "original_linear_macs_per_sample",
        "expected_compacted_linear_macs_per_sample",
        "expected_mac_reduction_fraction",
        "selection_criterion",
        "source_is_independent_direct_B0",
        "status",
    ],
)

architecture_summaries = []

for architecture in ARCHITECTURES:
    architecture_runs = [
        row
        for row in run_reports
        if row["architecture"]
        == architecture
    ]

    architecture_summaries.append(
        {
            "architecture": (
                architecture
            ),
            "seed_count": (
                len(
                    architecture_runs
                )
            ),
            "model_symbol": (
                architecture_runs[0][
                    "model_symbol"
                ]
            ),
            "linear_widths": (
                architecture_runs[0][
                    "linear_widths"
                ]
            ),
            "leaf_signature": (
                architecture_runs[0][
                    "leaf_signature"
                ]
            ),
            "leaf_signature_identical_across_seeds": (
                all(
                    row[
                        "leaf_signature"
                    ]
                    == architecture_runs[0][
                        "leaf_signature"
                    ]
                    for row in architecture_runs
                )
            ),
            "parameter_count": (
                architecture_runs[0][
                    "parameter_count"
                ]
            ),
            "linear_macs_per_sample": (
                architecture_runs[0][
                    "linear_macs_per_sample"
                ]
            ),
            "all_modules_supported": True,
            "parameterized_non_linear_module_count": 0,
        }
    )

global_checks = {
    "ten_B0_checkpoints_audited": (
        len(
            run_reports
        )
        == 10
    ),
    "twenty_pruning_sources_planned": (
        len(
            plan_rows
        )
        == 20
    ),
    "all_run_checks_passed": (
        all(
            row[
                "all_checks_passed"
            ]
            for row in run_reports
        )
    ),
    "all_architecture_signatures_stable": (
        all(
            row[
                "leaf_signature_identical_across_seeds"
            ]
            for row in architecture_summaries
        )
    ),
    "all_non_linear_modules_stateless": (
        all(
            row[
                "parameterized_non_linear_module_count"
            ]
            == 0
            for row in architecture_summaries
        )
    ),
    "P25_and_P50_directly_from_B0": (
        all(
            row[
                "source_is_independent_direct_B0"
            ]
            is True
            for row in plan_rows
        )
    ),
    "compacted_counts_strictly_smaller": (
        all(
            int(
                row[
                    "expected_compacted_parameter_count"
                ]
            )
            < int(
                row[
                    "original_parameter_count"
                ]
            )
            and int(
                row[
                    "expected_compacted_linear_macs_per_sample"
                ]
            )
            < int(
                row[
                    "original_linear_macs_per_sample"
                ]
            )
            for row in plan_rows
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
        "Physical-pruning global "
        "preflight failed: "
        + ", ".join(
            failed_global_checks
        )
    )

report = {
    "status": "passed",
    "phase": 5,
    "artifact_name": (
        "physical_structured_pruning_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "generated_at_utc": utc_now(),
    "audit_scope": {
        "B0_checkpoint_count": 10,
        "architecture_count": 2,
        "seed_count_per_architecture": 5,
        "train_rows_used_for_forward_smoke": (
            DUMMY_BATCH_SIZE
        ),
        "validation_access_count": 0,
        "test_access_count": 0,
    },
    "architecture_summaries": (
        architecture_summaries
    ),
    "checkpoint_audits": (
        run_reports
    ),
    "pruning_source_plan": {
        "row_count": len(
            plan_rows
        ),
        "ratios": list(
            PRUNING_RATIOS
        ),
        "P25_and_P50_each_derived_directly_from_B0": True,
        "physical_compaction_required": True,
        "mask_only_pruning_allowed": False,
    },
    "engine_requirements": {
        "must_preserve_leaf_module_order": True,
        "must_preserve_stateless_activations_and_dropout_configuration": True,
        "must_physically_replace_linear_dimensions": True,
        "must_copy_selected_rows_and_following_columns": True,
        "must_preserve_output_class_count": (
            EXPECTED_OUTPUT_CLASSES
        ),
        "must_record_selected_unit_indices": True,
        "must_record_source_checkpoint_hash": True,
        "must_pass_output_shape_parameter_MAC_and_serialization_tests": True,
        "selection_criterion_status": (
            "not_yet_locked"
        ),
    },
    "entry_checks": (
        entry_checks
    ),
    "global_checks": (
        global_checks
    ),
    "source_artifacts": {
        "phase5_protocol": (
            file_record(
                PHASE5_PROTOCOL
            )
        ),
        "B0_pair_lock": (
            file_record(
                B0_PAIR_LOCK
            )
        ),
        "B0_pair_lock_manifest": (
            file_record(
                B0_PAIR_LOCK_MANIFEST
            )
        ),
        "B0_checkpoint_registry": (
            file_record(
                B0_CHECKPOINT_REGISTRY
            )
        ),
        "model_source": (
            file_record(
                MODEL_SOURCE
            )
        ),
        "scaler": (
            file_record(
                SCALER_NPZ
            )
        ),
    },
    "leaf_module_inventory_csv": (
        file_record(
            OUTPUT_MODULES_CSV
        )
    ),
    "pruning_source_plan_csv": (
        file_record(
            OUTPUT_PLAN_CSV
        )
    ),
    "next_action": (
        "Create and lock the repository "
        "physical structured-pruning engine, "
        "including its deterministic hidden-unit "
        "importance criterion, before generating "
        "P25-noFT and P50-noFT checkpoints."
    ),
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
        "physical_structured_pruning_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "B0_checkpoint_count": 10,
    "planned_pruning_source_count": 20,
    "architectures_supported": list(
        ARCHITECTURES
    ),
    "ratios_supported": list(
        PRUNING_RATIOS
    ),
    "all_leaf_modules_supported": True,
    "parameterized_non_linear_modules_present": False,
    "validation_access_count": 0,
    "test_access_count": 0,
    "report": str(
        OUTPUT_REPORT
    ),
    "report_sha256": (
        sha256_file(
            OUTPUT_REPORT
        )
    ),
    "leaf_module_inventory": str(
        OUTPUT_MODULES_CSV
    ),
    "leaf_module_inventory_sha256": (
        sha256_file(
            OUTPUT_MODULES_CSV
        )
    ),
    "pruning_source_plan": str(
        OUTPUT_PLAN_CSV
    ),
    "pruning_source_plan_sha256": (
        sha256_file(
            OUTPUT_PLAN_CSV
        )
    ),
    "ready_for_pruning_engine_build": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK,
    lock,
)

post_checks = {
    "report_exists": (
        OUTPUT_REPORT.exists()
    ),
    "module_inventory_exists": (
        OUTPUT_MODULES_CSV.exists()
    ),
    "source_plan_exists": (
        OUTPUT_PLAN_CSV.exists()
    ),
    "lock_exists": (
        OUTPUT_LOCK.exists()
    ),
    "lock_report_hash_matches": (
        read_json(
            OUTPUT_LOCK
        )["report_sha256"]
        == sha256_file(
            OUTPUT_REPORT
        )
    ),
    "ready_for_engine_build": (
        read_json(
            OUTPUT_LOCK
        ).get(
            "ready_for_pruning_engine_build"
        )
        is True
    ),
}

failed_post_checks = [
    name
    for name, passed
    in post_checks.items()
    if not passed
]

if failed_post_checks:
    raise RuntimeError(
        "Physical-pruning preflight "
        "post-lock checks failed: "
        + ", ".join(
            failed_post_checks
        )
    )

print()
print("=" * 92)
print("PHASE 5 PHYSICAL PRUNING PREFLIGHT SUMMARY")
print("=" * 92)
print(
    "B0 checkpoints audited          : "
    f"{len(run_reports)}"
)
print(
    "Architectures                   : "
    f"{list(ARCHITECTURES)}"
)
print(
    "Planned P25/P50 source artifacts: "
    f"{len(plan_rows)}"
)
print(
    "All leaf modules supported      : True"
)
print(
    "Parameterized non-linear modules: 0"
)
print(
    "Leaf signatures stable by model : True"
)
print(
    "P25 and P50 directly from B0    : True"
)
print(
    "Validation access count         : 0"
)
print(
    "Test access count               : 0"
)
print(
    "Preflight status                : LOCKED"
)
print(
    "Ready for pruning-engine build  : True"
)
print(
    "Preflight report                : "
    f"{OUTPUT_REPORT}"
)
print(
    "Module inventory                : "
    f"{OUTPUT_MODULES_CSV}"
)
print(
    "Pruning source plan             : "
    f"{OUTPUT_PLAN_CSV}"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 5 PHYSICAL PRUNING PREFLIGHT LOCKED"
)
