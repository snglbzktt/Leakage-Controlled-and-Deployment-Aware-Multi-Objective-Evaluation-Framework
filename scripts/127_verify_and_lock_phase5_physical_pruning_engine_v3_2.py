from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import io
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

PREFLIGHT_REPORT = (
    AUDIT
    / "phase5_physical_pruning_preflight_v3_2.json"
)

PREFLIGHT_LOCK = (
    AUDIT
    / "phase5_physical_pruning_preflight_locked_v3_2.json"
)

B0_PAIR_LOCK = (
    AUDIT / "phase5_B0_pair_locked_v3_2.json"
)

B0_CHECKPOINT_REGISTRY = (
    AUDIT / "phase5_B0_checkpoint_registry_v3_2.csv"
)

MODEL_SOURCE = (
    ROOT / "src" / "models" / "nbaiot_models.py"
)

ENGINE_PATH = (
    ROOT
    / "src"
    / "compression"
    / "phase5_physical_pruning_engine_v3_2.py"
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

OUTPUT_SELECTED_UNITS = (
    AUDIT
    / "phase5_physical_pruning_engine_selected_units_v3_2.csv"
)

OUTPUT_REPORT = (
    AUDIT
    / "phase5_physical_pruning_engine_verification_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase5_physical_pruning_engine_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase5_physical_pruning_engine_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
ENGINE_VERSION = "phase5_physical_pruning_engine_v3_2"

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

RATIOS = (
    0.25,
    0.50,
)

EXPECTED_INPUT_FEATURES = 115
EXPECTED_OUTPUT_CLASSES = 3
DUMMY_BATCH_SIZE = 64
CPU_THREADS = 4

EXPECTED_WIDTHS = {
    "tinyml_mlp": {
        0.25: [
            [115, 48],
            [48, 24],
            [24, 3],
        ],
        0.50: [
            [115, 32],
            [32, 16],
            [16, 3],
        ],
    },
    "compact_dnn": {
        0.25: [
            [115, 96],
            [96, 48],
            [48, 24],
            [24, 3],
        ],
        0.50: [
            [115, 64],
            [64, 32],
            [32, 16],
            [16, 3],
        ],
    },
}

HIDDEN_DIMS = {
    "tinyml_mlp": [64, 32],
    "compact_dnn": [128, 64, 32],
}


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
                "Unresolved constructor parameter: "
                f"{name}"
            )

    return kwargs


def instantiate_model(
    model_module: ModuleType,
    symbol_name: str,
    architecture: str,
) -> nn.Module:
    candidate = getattr(
        model_module,
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
            "Model constructor did not return nn.Module."
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


def tensor_state_digest(
    model: nn.Module,
) -> str:
    digest = hashlib.sha256()

    for name, tensor in sorted(
        model.state_dict().items()
    ):
        digest.update(
            name.encode("utf-8")
        )

        contiguous = (
            tensor.detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            str(
                contiguous.dtype
            ).encode("ascii")
        )

        digest.update(
            np.asarray(
                contiguous.shape,
                dtype=np.int64,
            ).tobytes()
        )

        digest.update(
            contiguous.numpy().tobytes()
        )

    return digest.hexdigest()


def named_linear_dict(
    model: nn.Module,
) -> dict[str, nn.Linear]:
    return {
        name: module
        for name, module
        in model.named_modules()
        if name
        and isinstance(
            module,
            nn.Linear,
        )
    }


def copied_weights_match(
    source_model: nn.Module,
    compact_model: nn.Module,
    metadata: dict[str, Any],
) -> bool:
    source_layers = named_linear_dict(
        source_model
    )

    compact_layers = named_linear_dict(
        compact_model
    )

    for record in metadata[
        "layer_copy_records"
    ]:
        name = record[
            "layer_name"
        ]

        source_layer = (
            source_layers[name]
        )

        compact_layer = (
            compact_layers[name]
        )

        input_indices = torch.tensor(
            record["input_indices"],
            dtype=torch.long,
        )

        output_indices = torch.tensor(
            record["output_indices"],
            dtype=torch.long,
        )

        expected_weight = (
            source_layer.weight
            .index_select(
                0,
                output_indices,
            )
            .index_select(
                1,
                input_indices,
            )
        )

        if not torch.equal(
            expected_weight,
            compact_layer.weight,
        ):
            return False

        if source_layer.bias is None:
            if compact_layer.bias is not None:
                return False
        else:
            expected_bias = (
                source_layer.bias
                .index_select(
                    0,
                    output_indices,
                )
            )

            if not torch.equal(
                expected_bias,
                compact_layer.bias,
            ):
                return False

    return True


def no_shared_parameter_storage(
    source_model: nn.Module,
    compact_model: nn.Module,
) -> bool:
    source_pointers = {
        parameter.data_ptr()
        for parameter
        in source_model.parameters()
    }

    return all(
        parameter.data_ptr()
        not in source_pointers
        for parameter
        in compact_model.parameters()
    )


required_paths = (
    PHASE5_PROTOCOL,
    PREFLIGHT_REPORT,
    PREFLIGHT_LOCK,
    B0_PAIR_LOCK,
    B0_CHECKPOINT_REGISTRY,
    MODEL_SOURCE,
    ENGINE_PATH,
    X_TRAIN_PATH,
    SCALER_NPZ,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_SELECTED_UNITS,
    OUTPUT_REPORT,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Pruning-engine artifact already exists; "
            f"refusing to overwrite: {output_path}"
        )

protocol = read_json(
    PHASE5_PROTOCOL
)

preflight_report = read_json(
    PREFLIGHT_REPORT
)

preflight_lock = read_json(
    PREFLIGHT_LOCK
)

b0_pair_lock = read_json(
    B0_PAIR_LOCK
)

registry_rows = read_csv(
    B0_CHECKPOINT_REGISTRY
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
    "physical_pruning_required": (
        protocol["pruning"][
            "type"
        ]
        == (
            "physical_structured_"
            "hidden_unit_pruning"
        )
    ),
    "mask_only_disallowed": (
        protocol["pruning"][
            "mask_only_pruning_allowed"
        ]
        is False
    ),
    "preflight_passed": (
        preflight_report.get(
            "status"
        )
        == "passed"
        and preflight_report.get(
            "all_checks_passed"
        )
        is True
    ),
    "preflight_locked": (
        preflight_lock.get(
            "status"
        )
        == "locked"
        and preflight_lock.get(
            "ready_for_pruning_engine_build"
        )
        is True
        and preflight_lock.get(
            "report_sha256"
        )
        == sha256_file(
            PREFLIGHT_REPORT
        )
    ),
    "B0_pair_locked": (
        b0_pair_lock.get(
            "status"
        )
        == "locked"
        and b0_pair_lock.get(
            "ready_for_branch_source_generation"
        )
        is True
    ),
    "registry_hash_matches": (
        b0_pair_lock.get(
            "checkpoint_registry_sha256"
        )
        == sha256_file(
            B0_CHECKPOINT_REGISTRY
        )
    ),
    "registry_has_ten_rows": (
        len(registry_rows)
        == 10
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
        "Pruning-engine entry gate failed: "
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

dummy_numpy = (
    (
        np.asarray(
            x_train[
                :DUMMY_BATCH_SIZE
            ],
            dtype=np.float64,
        )
        - mean64
    )
    / scale64
).astype(np.float32)

if not np.isfinite(
    dummy_numpy
).all():
    raise RuntimeError(
        "Non-finite train-only smoke batch."
    )

dummy_batch = torch.from_numpy(
    dummy_numpy
)

model_module = load_module(
    MODEL_SOURCE,
    "phase5_pruning_engine_model_source",
)

engine_module = load_module(
    ENGINE_PATH,
    "phase5_physical_pruning_engine_locked_source",
)

if (
    engine_module.ENGINE_VERSION
    != ENGINE_VERSION
):
    raise RuntimeError(
        "Pruning-engine version mismatch."
    )

selected_unit_rows: list[
    dict[str, Any]
] = []

branch_reports: list[
    dict[str, Any]
] = []

selection_lookup: dict[
    tuple[str, int, float, str],
    set[int],
] = {}

branch_index = 0

for registry_row in sorted(
    registry_rows,
    key=lambda row: (
        row["architecture"],
        int(row["seed"]),
    ),
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

    checkpoint_hash = (
        registry_row[
            "checkpoint_sha256"
        ]
    )

    if sha256_file(
        checkpoint_path
    ) != checkpoint_hash:
        raise RuntimeError(
            "B0 checkpoint hash mismatch: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
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

    source_digest_before = (
        tensor_state_digest(
            source_model
        )
    )

    with torch.inference_mode():
        source_logits_before = (
            extract_logits(
                source_model(
                    dummy_batch
                )
            ).clone()
        )

    for ratio in RATIOS:
        branch_index += 1

        (
            compact_model,
            metadata,
        ) = engine_module.physically_prune_mlp(
            source_model,
            ratio,
            source_checkpoint_sha256=(
                checkpoint_hash
            ),
        )

        (
            repeated_model,
            repeated_metadata,
        ) = engine_module.physically_prune_mlp(
            source_model,
            ratio,
            source_checkpoint_sha256=(
                checkpoint_hash
            ),
        )

        compact_model.eval()
        repeated_model.eval()

        with torch.inference_mode():
            compact_logits = (
                extract_logits(
                    compact_model(
                        dummy_batch
                    )
                )
            )

            repeated_logits = (
                extract_logits(
                    repeated_model(
                        dummy_batch
                    )
                )
            )

            source_logits_after = (
                extract_logits(
                    source_model(
                        dummy_batch
                    )
                )
            )

        source_digest_after = (
            tensor_state_digest(
                source_model
            )
        )

        expected_widths = (
            EXPECTED_WIDTHS[
                architecture
            ][ratio]
        )

        expected_variant = (
            "P25-noFT"
            if math.isclose(
                ratio,
                0.25,
            )
            else "P50-noFT"
        )

        buffer = io.BytesIO()

        torch.save(
            {
                "state_dict": (
                    compact_model.state_dict()
                ),
                "metadata": metadata,
            },
            buffer,
        )

        buffer.seek(0)

        serialized = torch.load(
            buffer,
            map_location="cpu",
            weights_only=False,
        )

        (
            roundtrip_model,
            roundtrip_metadata,
        ) = engine_module.physically_prune_mlp(
            source_model,
            ratio,
            source_checkpoint_sha256=(
                checkpoint_hash
            ),
        )

        roundtrip_model.load_state_dict(
            serialized[
                "state_dict"
            ],
            strict=True,
        )

        roundtrip_model.eval()

        with torch.inference_mode():
            roundtrip_logits = (
                extract_logits(
                    roundtrip_model(
                        dummy_batch
                    )
                )
            )

        checks = {
            "source_checkpoint_hash_matches": (
                metadata[
                    "source_checkpoint_sha256"
                ]
                == checkpoint_hash
            ),
            "generated_directly_from_B0": (
                metadata[
                    "generated_directly_from_B0"
                ]
                is True
            ),
            "physical_compaction_true": (
                metadata[
                    "physical_compaction"
                ]
                is True
                and metadata[
                    "mask_only_pruning"
                ]
                is False
            ),
            "compacted_widths_match_expected": (
                metadata[
                    "compacted_linear_widths"
                ]
                == expected_widths
                and engine_module.inspect_linear_widths(
                    compact_model
                )
                == expected_widths
            ),
            "parameter_count_reduced": (
                metadata[
                    "compacted_parameter_count"
                ]
                < metadata[
                    "source_parameter_count"
                ]
            ),
            "MAC_count_reduced": (
                metadata[
                    "compacted_linear_macs_per_sample"
                ]
                < metadata[
                    "source_linear_macs_per_sample"
                ]
            ),
            "copied_weights_match_exactly": (
                copied_weights_match(
                    source_model,
                    compact_model,
                    metadata,
                )
            ),
            "source_model_unchanged": (
                source_digest_before
                == source_digest_after
            ),
            "source_output_unchanged": (
                torch.equal(
                    source_logits_before,
                    source_logits_after,
                )
            ),
            "no_shared_parameter_storage": (
                no_shared_parameter_storage(
                    source_model,
                    compact_model,
                )
            ),
            "deterministic_selection": (
                metadata[
                    "hidden_unit_selections"
                ]
                == repeated_metadata[
                    "hidden_unit_selections"
                ]
            ),
            "deterministic_weights": (
                tensor_state_digest(
                    compact_model
                )
                == tensor_state_digest(
                    repeated_model
                )
                and torch.equal(
                    compact_logits,
                    repeated_logits,
                )
            ),
            "output_shape_preserved": (
                tuple(
                    compact_logits.shape
                )
                == (
                    DUMMY_BATCH_SIZE,
                    EXPECTED_OUTPUT_CLASSES,
                )
            ),
            "output_is_finite": (
                bool(
                    torch.isfinite(
                        compact_logits
                    ).all().item()
                )
            ),
            "serialization_roundtrip_metadata_matches": (
                serialized["metadata"]
                == metadata
                and roundtrip_metadata[
                    "compacted_linear_widths"
                ]
                == metadata[
                    "compacted_linear_widths"
                ]
            ),
            "serialization_roundtrip_output_matches": (
                torch.equal(
                    compact_logits,
                    roundtrip_logits,
                )
            ),
            "serialization_roundtrip_state_matches": (
                tensor_state_digest(
                    compact_model
                )
                == tensor_state_digest(
                    roundtrip_model
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
                f"{expected_variant} engine test failed: "
                + ", ".join(
                    failed_checks
                )
            )

        for selection in metadata[
            "hidden_unit_selections"
        ]:
            layer_name = (
                selection[
                    "layer_name"
                ]
            )

            selected_indices = (
                selection[
                    "selected_indices"
                ]
            )

            selection_lookup[
                (
                    architecture,
                    seed,
                    ratio,
                    layer_name,
                )
            ] = set(
                selected_indices
            )

            selected_unit_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "variant": expected_variant,
                    "pruning_ratio": ratio,
                    "source_checkpoint_path": (
                        str(
                            checkpoint_path
                        )
                    ),
                    "source_checkpoint_sha256": (
                        checkpoint_hash
                    ),
                    "hidden_layer_index": (
                        selection[
                            "hidden_layer_index"
                        ]
                    ),
                    "layer_name": layer_name,
                    "original_width": (
                        selection[
                            "original_width"
                        ]
                    ),
                    "retained_width": (
                        selection[
                            "retained_width"
                        ]
                    ),
                    "selected_indices": (
                        json.dumps(
                            selected_indices,
                            separators=(
                                ",",
                                ":",
                            ),
                        )
                    ),
                    "ranked_indices": (
                        json.dumps(
                            selection[
                                "ranked_indices"
                            ],
                            separators=(
                                ",",
                                ":",
                            ),
                        )
                    ),
                    "selected_scores": (
                        json.dumps(
                            selection[
                                "selected_scores"
                            ],
                            separators=(
                                ",",
                                ":",
                            ),
                        )
                    ),
                    "importance_criterion": (
                        metadata[
                            "importance_criterion"
                        ]
                    ),
                }
            )

        branch_reports.append(
            {
                "architecture": architecture,
                "seed": seed,
                "variant": expected_variant,
                "pruning_ratio": ratio,
                "source_checkpoint": (
                    file_record(
                        checkpoint_path
                    )
                ),
                "source_linear_widths": (
                    metadata[
                        "source_linear_widths"
                    ]
                ),
                "compacted_linear_widths": (
                    metadata[
                        "compacted_linear_widths"
                    ]
                ),
                "source_parameter_count": (
                    metadata[
                        "source_parameter_count"
                    ]
                ),
                "compacted_parameter_count": (
                    metadata[
                        "compacted_parameter_count"
                    ]
                ),
                "parameter_reduction_fraction": (
                    1.0
                    - metadata[
                        "compacted_parameter_count"
                    ]
                    / metadata[
                        "source_parameter_count"
                    ]
                ),
                "source_linear_macs_per_sample": (
                    metadata[
                        "source_linear_macs_per_sample"
                    ]
                ),
                "compacted_linear_macs_per_sample": (
                    metadata[
                        "compacted_linear_macs_per_sample"
                    ]
                ),
                "MAC_reduction_fraction": (
                    1.0
                    - metadata[
                        "compacted_linear_macs_per_sample"
                    ]
                    / metadata[
                        "source_linear_macs_per_sample"
                    ]
                ),
                "serialized_state_bytes": (
                    len(
                        buffer.getvalue()
                    )
                ),
                "checks": checks,
                "validation_access_count": 0,
                "test_access_count": 0,
                "all_checks_passed": True,
            }
        )

        print(
            f"[{branch_index}/20] "
            f"{architecture} seed={seed} "
            f"{expected_variant} | "
            f"widths={expected_widths} | "
            "deterministic=True | "
            "roundtrip=True | "
            "all checks=True",
            flush=True,
        )

        del compact_model
        del repeated_model
        del roundtrip_model

    del source_model
    del checkpoint

nested_checks: dict[
    str,
    bool,
] = {}

for architecture in ARCHITECTURES:
    for seed in SEEDS:
        p25_keys = [
            key
            for key in selection_lookup
            if key[0] == architecture
            and key[1] == seed
            and math.isclose(
                key[2],
                0.25,
            )
        ]

        for p25_key in p25_keys:
            layer_name = p25_key[3]

            p50_key = (
                architecture,
                seed,
                0.50,
                layer_name,
            )

            nested_checks[
                (
                    f"{architecture}_seed_{seed}_"
                    f"{layer_name}_P50_subset_P25"
                )
            ] = (
                selection_lookup[
                    p50_key
                ].issubset(
                    selection_lookup[
                        p25_key
                    ]
                )
            )

failed_nested_checks = [
    name
    for name, passed
    in nested_checks.items()
    if not passed
]

if failed_nested_checks:
    raise RuntimeError(
        "Nested-selection check failed: "
        + ", ".join(
            failed_nested_checks
        )
    )

atomic_csv(
    OUTPUT_SELECTED_UNITS,
    selected_unit_rows,
    [
        "architecture",
        "seed",
        "variant",
        "pruning_ratio",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "hidden_layer_index",
        "layer_name",
        "original_width",
        "retained_width",
        "selected_indices",
        "ranked_indices",
        "selected_scores",
        "importance_criterion",
    ],
)

global_checks = {
    "twenty_branch_tests_completed": (
        len(
            branch_reports
        )
        == 20
    ),
    "all_branch_checks_passed": (
        all(
            row[
                "all_checks_passed"
            ]
            for row in branch_reports
        )
    ),
    "all_P50_selections_subset_of_P25": (
        all(
            nested_checks.values()
        )
    ),
    "validation_access_count_zero": (
        all(
            row[
                "validation_access_count"
            ]
            == 0
            for row in branch_reports
        )
    ),
    "test_access_count_zero": (
        all(
            row[
                "test_access_count"
            ]
            == 0
            for row in branch_reports
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
        "Pruning-engine global checks failed: "
        + ", ".join(
            failed_global_checks
        )
    )

architecture_ratio_summary: list[
    dict[str, Any]
] = []

for architecture in ARCHITECTURES:
    for ratio in RATIOS:
        rows = [
            row
            for row in branch_reports
            if row[
                "architecture"
            ]
            == architecture
            and math.isclose(
                row[
                    "pruning_ratio"
                ],
                ratio,
            )
        ]

        architecture_ratio_summary.append(
            {
                "architecture": architecture,
                "pruning_ratio": ratio,
                "variant": (
                    "P25-noFT"
                    if math.isclose(
                        ratio,
                        0.25,
                    )
                    else "P50-noFT"
                ),
                "run_count": len(rows),
                "compacted_linear_widths": (
                    rows[0][
                        "compacted_linear_widths"
                    ]
                ),
                "source_parameter_count": (
                    rows[0][
                        "source_parameter_count"
                    ]
                ),
                "compacted_parameter_count": (
                    rows[0][
                        "compacted_parameter_count"
                    ]
                ),
                "parameter_reduction_fraction": (
                    rows[0][
                        "parameter_reduction_fraction"
                    ]
                ),
                "source_linear_macs_per_sample": (
                    rows[0][
                        "source_linear_macs_per_sample"
                    ]
                ),
                "compacted_linear_macs_per_sample": (
                    rows[0][
                        "compacted_linear_macs_per_sample"
                    ]
                ),
                "MAC_reduction_fraction": (
                    rows[0][
                        "MAC_reduction_fraction"
                    ]
                ),
            }
        )

report = {
    "status": "passed",
    "phase": 5,
    "artifact_name": (
        "physical_structured_pruning_engine"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "engine_version": (
        ENGINE_VERSION
    ),
    "verified_at_utc": utc_now(),
    "importance_criterion": (
        engine_module.IMPORTANCE_CRITERION
    ),
    "tie_break_policy": (
        "ascending original hidden-unit index"
    ),
    "source_policy": {
        "P25_generated_directly_from_B0": True,
        "P50_generated_directly_from_B0": True,
        "P50_generated_from_P25": False,
        "scores_computed_on_original_B0": True,
        "mask_only_pruning": False,
        "physical_linear_dimension_replacement": True,
    },
    "verification_scope": {
        "B0_checkpoint_count": 10,
        "pruning_ratio_count": 2,
        "branch_test_count": 20,
        "train_rows_used_for_smoke": (
            DUMMY_BATCH_SIZE
        ),
        "validation_access_count": 0,
        "test_access_count": 0,
    },
    "architecture_ratio_summary": (
        architecture_ratio_summary
    ),
    "branch_reports": (
        branch_reports
    ),
    "nested_selection_checks": (
        nested_checks
    ),
    "entry_checks": (
        entry_checks
    ),
    "global_checks": (
        global_checks
    ),
    "engine_source": (
        file_record(
            ENGINE_PATH
        )
    ),
    "selected_units_csv": (
        file_record(
            OUTPUT_SELECTED_UNITS
        )
    ),
    "ready_for_P25_noFT_and_P50_noFT_generation": True,
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
        "physical_structured_pruning_engine"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "engine_version": (
        ENGINE_VERSION
    ),
    "locked_at_utc": utc_now(),
    "engine_source": str(
        ENGINE_PATH
    ),
    "engine_source_sha256": (
        sha256_file(
            ENGINE_PATH
        )
    ),
    "verification_report": str(
        OUTPUT_REPORT
    ),
    "verification_report_sha256": (
        sha256_file(
            OUTPUT_REPORT
        )
    ),
    "selected_units_csv": str(
        OUTPUT_SELECTED_UNITS
    ),
    "selected_units_csv_sha256": (
        sha256_file(
            OUTPUT_SELECTED_UNITS
        )
    ),
    "importance_criterion": (
        engine_module.IMPORTANCE_CRITERION
    ),
    "B0_checkpoint_count_verified": 10,
    "branch_test_count": 20,
    "validation_access_count": 0,
    "test_access_count": 0,
    "ready_for_P25_noFT_and_P50_noFT_generation": True,
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
        "physical_structured_pruning_engine"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "engine_version": (
        ENGINE_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            PHASE5_PROTOCOL
        ),
        file_record(
            PREFLIGHT_REPORT
        ),
        file_record(
            PREFLIGHT_LOCK
        ),
        file_record(
            B0_PAIR_LOCK
        ),
        file_record(
            B0_CHECKPOINT_REGISTRY
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
            ENGINE_PATH
        ),
        file_record(
            OUTPUT_SELECTED_UNITS
        ),
        file_record(
            OUTPUT_REPORT
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "ready_for_P25_noFT_and_P50_noFT_generation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

post_checks = {
    "engine_exists": (
        ENGINE_PATH.exists()
    ),
    "report_exists": (
        OUTPUT_REPORT.exists()
    ),
    "lock_exists": (
        OUTPUT_LOCK.exists()
    ),
    "lock_manifest_exists": (
        OUTPUT_LOCK_MANIFEST.exists()
    ),
    "lock_engine_hash_matches": (
        read_json(
            OUTPUT_LOCK
        )[
            "engine_source_sha256"
        ]
        == sha256_file(
            ENGINE_PATH
        )
    ),
    "lock_report_hash_matches": (
        read_json(
            OUTPUT_LOCK
        )[
            "verification_report_sha256"
        ]
        == sha256_file(
            OUTPUT_REPORT
        )
    ),
    "ready_for_checkpoint_generation": (
        read_json(
            OUTPUT_LOCK
        ).get(
            "ready_for_P25_noFT_and_P50_noFT_generation"
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
        "Pruning-engine post-lock checks failed: "
        + ", ".join(
            failed_post_checks
        )
    )

print()
print("=" * 92)
print("PHASE 5 PHYSICAL PRUNING ENGINE LOCK SUMMARY")
print("=" * 92)
print(
    "Engine version                  : "
    f"{ENGINE_VERSION}"
)
print(
    "B0 checkpoints verified         : 10"
)
print(
    "P25/P50 branch tests            : 20"
)
print(
    "Physical dimension replacement  : PASSED"
)
print(
    "Exact selected-weight copy      : PASSED"
)
print(
    "Deterministic selection         : PASSED"
)
print(
    "Serialization roundtrip         : PASSED"
)
print(
    "P50 selected units subset P25   : PASSED"
)
print(
    "Source B0 models unchanged      : PASSED"
)
print(
    "Validation access count         : 0"
)
print(
    "Test access count               : 0"
)
print(
    "Pruning engine status           : LOCKED"
)
print(
    "Ready for P25/P50 noFT          : True"
)
print(
    "Engine source                   : "
    f"{ENGINE_PATH}"
)
print(
    "Verification report             : "
    f"{OUTPUT_REPORT}"
)
print(
    "Selected-unit registry          : "
    f"{OUTPUT_SELECTED_UNITS}"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 5 PHYSICAL PRUNING ENGINE LOCKED"
)
